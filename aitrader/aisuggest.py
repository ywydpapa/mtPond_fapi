import requests, time, datetime, math
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from xgboost import XGBClassifier
from collections import deque

# -----------------------------
# 1. Data Fetch
# -----------------------------
def fetch_upbit_candles(market: str, unit: str = '5m', count: int = 400) -> pd.DataFrame:
    unit_map = {
        '1m':1,'3m':3,'5m':5,'10m':10,'15m':15,'30m':30,'1h':60,'4h':240
    }
    if unit == '1d':
        url = f"https://api.upbit.com/v1/candles/days?market={market}&count={count}"
    else:
        m = unit_map[unit]
        url = f"https://api.upbit.com/v1/candles/minutes/{m}?market={market}&count={count}"
    r = requests.get(url, timeout=5)
    data = r.json()
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df['candle_date_time_kst'] = pd.to_datetime(df['candle_date_time_kst'])
    df = df.sort_values('candle_date_time_kst').set_index('candle_date_time_kst')
    return df[['trade_price','high_price','low_price','opening_price','candle_acc_trade_volume']]

# -----------------------------
# 2. Multi-timeframe Merge
# -----------------------------
def build_multiframe(market: str, base_unit='5m') -> pd.DataFrame:
    base = fetch_upbit_candles(market, base_unit, 800)
    if base.empty: return base
    base = base.rename(columns={'trade_price':'price','candle_acc_trade_volume':'volume'})
    base = base.asfreq('5T')
    # Higher frames
    frames = {}
    for u in ['15m','1h']:
        d = fetch_upbit_candles(market, u, 400)
        if d.empty: continue
        d = d.rename(columns={'trade_price':'price','candle_acc_trade_volume':'volume'})
        if u == '15m':
            d = d.asfreq('15T')
        elif u == '1h':
            d = d.asfreq('60T')
        frames[u] = d

    # Upsample higher frames forward fill to 5m alignment
    for u,dfu in frames.items():
        dfu = dfu[['price','volume']].copy()
        dfu.columns = [f'price_{u}', f'volume_{u}']
        # Reindex to base index
        dfu = dfu.reindex(base.index, method='ffill')
        base = base.join(dfu, how='left')

    return base

# -----------------------------
# 3. Feature Engineering
# -----------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 기본 결측 처리
    df[['price','volume']] = df[['price','volume']].fillna(method='ffill')
    df['log_ret'] = np.log(df['price']).diff()
    # 모멘텀
    for w in [3,6,12,24]:
        df[f'mom_sum_{w}'] = df['log_ret'].rolling(w).sum()
    # 변동성
    for w in [6,12,24,48]:
        df[f'vol_{w}'] = df['log_ret'].rolling(w).std()
    # 평균 절대 변화
    for w in [6,12]:
        df[f'absret_mean_{w}'] = df['log_ret'].abs().rolling(w).mean()
    # 거래량 파생
    df['vol_z'] = (df['volume'] - df['volume'].rolling(30).mean()) / (df['volume'].rolling(30).std() + 1e-9)
    # Bollinger 기반 숫자화 (20)
    mid = df['price'].rolling(20).mean()
    std = df['price'].rolling(20).std()
    upper = mid + 2*std
    lower = mid - 2*std
    band_width = upper - lower
    df['bb_pos'] = (df['price'] - mid) / (band_width.replace(0, np.nan))
    df['bb_width'] = band_width / (mid.replace(0,np.nan))
    df['bb_contr'] = df['bb_width'].pct_change()
    # 상위 타임프레임 상대강도
    for u in ['15m','1h']:
        if f'price_{u}' in df.columns:
            df[f'htf_ret_{u}'] = np.log(df[f'price_{u}']).diff()
    # RSI 간단(14)
    def rsi(series, period=14):
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.rolling(period).mean()
        ma_down = down.rolling(period).mean()
        rs = ma_up / (ma_down + 1e-9)
        return 100 - (100 / (1 + rs))
    df['rsi_14'] = rsi(df['price'],14)

    # 결측/무한 처리
    df = df.replace([np.inf,-np.inf], np.nan)
    df = df.dropna()
    return df

# -----------------------------
# 4. Target Construction
# -----------------------------
def make_target(df: pd.DataFrame, horizon=3, thresh_mode='dynamic', k=0.8) -> pd.DataFrame:
    df = df.copy()
    # 누적 미래 로그수익률
    df['fwd_ret'] = np.log(df['price'].shift(-horizon)) - np.log(df['price'])
    # 동적 임계값: 최근 60 구간 평균 절대 fwd_ret (proxy)
    if thresh_mode == 'dynamic':
        df['abs_ret_ref'] = df['fwd_ret'].rolling(120).apply(lambda x: np.nanmean(np.abs(x)), raw=True)
        df['theta'] = df['abs_ret_ref'] * k
    else:
        df['theta'] = 0.001  # 고정(예시)
    df['theta'] = df['theta'].fillna(df['theta'].median())

    def label_row(row):
        if row['fwd_ret'] > row['theta']:
            return 2  # Up
        elif row['fwd_ret'] < -row['theta']:
            return 0  # Down
        else:
            return 1  # Flat

    df['target'] = df.apply(label_row, axis=1)
    df = df.dropna(subset=['fwd_ret','target'])
    return df

# -----------------------------
# 5. Dataset Split (Walk-forward utility)
# -----------------------------
def time_split(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    n = len(df)
    train_end = int(n*train_ratio)
    val_end = int(n*(train_ratio+val_ratio))
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test

# -----------------------------
# 6. Model Wrapper / Ensemble
# -----------------------------
FEATURE_EXCLUDE = {'fwd_ret','target','abs_ret_ref','theta'}

def select_features(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in FEATURE_EXCLUDE]

@dataclass
class EnsembleModel:
    models: List[XGBClassifier]
    feature_cols: List[str]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        preds = [m.predict_proba(X[self.feature_cols]) for m in self.models]
        return np.mean(preds, axis=0)

def train_ensemble(train: pd.DataFrame, val: pd.DataFrame, feature_cols: List[str], n_models=3) -> EnsembleModel:
    models = []
    for seed in range(n_models):
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            learning_rate=0.05,
            objective='multi:softprob',
            num_class=3,
            random_state=seed,
            eval_metric='mlogloss'
        )
        model.fit(
            train[feature_cols], train['target'],
            eval_set=[(val[feature_cols], val['target'])],
            verbose=False
        )
        models.append(model)
    return EnsembleModel(models=models, feature_cols=feature_cols)

# -----------------------------
# 7. Signal Generation
# -----------------------------
def generate_signal(proba_row: np.ndarray, theta_prob=0.6) -> str:
    # proba order: class 0=Down,1=Flat,2=Up
    p_down, p_flat, p_up = proba_row
    if p_up > theta_prob and p_up > p_down:
        return "BUY"
    if p_down > theta_prob and p_down > p_up:
        return "SELL"
    return "NO_TRADE"

# -----------------------------
# 8. Evaluation (Simple)
# -----------------------------
def evaluate_direction(df: pd.DataFrame, proba: np.ndarray) -> Dict:
    df = df.copy()
    df['p_down'], df['p_flat'], df['p_up'] = proba[:,0], proba[:,1], proba[:,2]
    df['pred_class'] = np.argmax(proba, axis=1)
    acc = (df['pred_class'] == df['target']).mean()
    # Up/Down precision
    up_mask = df['pred_class'] == 2
    down_mask = df['pred_class'] == 0
    up_precision = (df.loc[up_mask, 'target'] == 2).mean() if up_mask.any() else np.nan
    down_precision = (df.loc[down_mask, 'target'] == 0).mean() if down_mask.any() else np.nan
    return {
        "overall_acc": acc,
        "up_precision": up_precision,
        "down_precision": down_precision,
        "n_samples": len(df)
    }

# -----------------------------
# 9. End-to-End Build + Example
# -----------------------------
def build_and_train(market='KRW-BTC', horizon=3):
    raw = build_multiframe(market)
    if raw.empty:
        raise RuntimeError("데이터 부족: 원시 데이터가 비어 있습니다. 데이터 소스/네트워크/심볼을 확인하세요.")
    # 원시 데이터 길이만으로 판단하지 않고, 피처 생성 이후 유효 샘플 기준으로 판단
    feat = add_features(raw)
    feat = make_target(feat, horizon=horizon)
    # 피처 생성 후 결측 처리
    feat = feat.dropna()

    # 데이터 충분성 동적 기준 (하드코딩 300 제거)
    # horizon이 커질수록 더 많은 안전 마진을 요구
    min_required = max(120, 40 * horizon)

    if len(feat) < min_required:
        raise RuntimeError(
            f"데이터 부족: 유효 샘플 {len(feat)}행, 필요 최소 {min_required}행 "
            f"(원시 데이터 {len(raw)}행). 데이터 수집 범위를 확대하거나 기간을 늘려주세요."
        )

    train, val, test = time_split(feat)
    feature_cols = select_features(feat)
    ensemble = train_ensemble(train, val, feature_cols)
    # Validation 평가
    val_proba = ensemble.predict_proba(val)
    val_eval = evaluate_direction(val, val_proba)
    print("Validation:", val_eval)
    test_proba = ensemble.predict_proba(test)
    test_eval = evaluate_direction(test, test_proba)
    print("Test:", test_eval)
    return ensemble, feature_cols, feat

# -----------------------------
# 10. Live Update Skeleton
# -----------------------------
class LivePredictor:
    def __init__(self, market: str, horizon=3, retrain_interval=60):
        self.market = market
        self.horizon = horizon
          # 재학습 주기(분 단위 캔들 수)
        self.retrain_interval = retrain_interval
        self.ensemble: Optional[EnsembleModel] = None
        self.feature_cols: List[str] = []
        self.buffer = deque(maxlen=2000)
        self.counter = 0

    def retrain(self):
        self.ensemble, self.feature_cols, feat = build_and_train(self.market, self.horizon)
        for idx, row in feat.iterrows():
            self.buffer.append(row.to_dict())
        self.counter = 0

    def step_predict(self):
        if self.ensemble is None or self.counter >= self.retrain_interval:
            print("[INFO] 재학습 수행")
            self.retrain()

        # 최신 데이터 다시 수집 → 피처 생성 → 마지막 1개 행
        raw = build_multiframe(self.market)
        feat = add_features(raw)
        feat = make_target(feat, horizon=self.horizon)
        latest = feat.iloc[[-1]]
        X = latest[self.feature_cols]
        proba = self.ensemble.predict_proba(X)[0]
        signal = generate_signal(proba)
        out = {
            "timestamp": latest.index[-1],
            "price": float(latest['price']),
            "p_down": float(proba[0]),
            "p_flat": float(proba[1]),
            "p_up": float(proba[2]),
            "signal": signal
        }
        self.counter += 1
        return out

if __name__ == "__main__":
    # 1회 학습 + 라이브 예시 3회
    try:
        ensemble, feature_cols, feat_all = build_and_train('KRW-BTC', horizon=3)
    except RuntimeError as e:
        # 학습 실패 원인을 명확히 출력하고, 라이브 예측 진입 방지
        print(f"[학습 중단] {e}")
        import sys
        sys.exit(1)

    live = LivePredictor('KRW-BTC', horizon=3, retrain_interval=120)
    for i in range(3):
        res = live.step_predict()
        print(res)
        time.sleep(2)