import time
import requests
import pandas as pd
from prophet import Prophet
import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
import numpy as np

# ---------- Feature / Model Helpers ----------
def create_features(df, n_lags=5):
    df_feat = df.copy()
    for i in range(1, n_lags+1):
        df_feat[f'lag_{i}'] = df_feat['trade_price'].shift(i)
    df_feat['volume'] = df_feat['candle_acc_trade_volume']
    df_feat['ma_3'] = df_feat['trade_price'].rolling(window=3).mean()
    df_feat['ma_7'] = df_feat['trade_price'].rolling(window=7).mean()
    df_feat['std_3'] = df_feat['trade_price'].rolling(window=3).std()
    df_feat['std_7'] = df_feat['trade_price'].rolling(window=7).std()
    df_feat = df_feat.dropna()
    return df_feat

def predict_future_price_xgb(df, periods=3, n_lags=5, method='xgb'):
    df_feat = create_features(df, n_lags)
    if df_feat.empty:
        return None
    feature_cols = [f'lag_{i}' for i in range(1, n_lags + 1)] + ['volume', 'ma_3', 'ma_7', 'std_3', 'std_7']
    X = df_feat[feature_cols]
    y = df_feat['trade_price']
    last_row = X.iloc[[-1]].copy()
    if method == 'xgb':
        model = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42, verbosity=0)
    else:
        model = lgb.LGBMRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X, y)
    preds = []
    recent_prices = list(df_feat['trade_price'].iloc[-n_lags:].values)
    current_feats = last_row.copy()
    for _ in range(periods):
        pred = model.predict(current_feats)[0]
        preds.append(pred)
        # 시프트
        new_lags = np.roll(current_feats.values[0][:n_lags], 1)
          # 가장 최근(lag_1)에 새 예측 삽입
        new_lags[0] = pred
        for i in range(n_lags):
            current_feats.iloc[0, i] = new_lags[i]
        # 최근 가격 리스트 갱신
        recent_prices = list(new_lags[:n_lags])
        # 통계 재계산 (열 순서 주의: lag들 다음 'volume','ma_3','ma_7','std_3','std_7')
        # volume(거래량)은 변화 없으니 유지 (current_feats.iloc[0, n_lags] = 기존 값)
        current_feats.iloc[0, n_lags + 1] = np.mean(recent_prices[-3:])  # ma_3
        current_feats.iloc[0, n_lags + 2] = np.mean(recent_prices[-7:])  # ma_7
        current_feats.iloc[0, n_lags + 3] = np.std(recent_prices[-3:])   # std_3
        current_feats.iloc[0, n_lags + 4] = np.std(recent_prices[-7:])   # std_7
    return float(np.mean(preds)) if preds else None

def predict_future_price_arima(df, periods=3):
    y = df['trade_price']
    if y.isna().all() or len(y) < 10:
        return None
    try:
        model = ARIMA(y, order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=periods)
        future_price = forecast.mean()
        return float(future_price)
    except Exception as e:
        print("ARIMA 예측 실패:", e)
        return None

def predict_future_price(df, periods=3, freq='3min'):
    prophet_df = df.reset_index()[['candle_date_time_kst', 'trade_price']].rename(
        columns={'candle_date_time_kst': 'ds', 'trade_price': 'y'}
    )
    # Prophet은 최소 몇 개 이상의 데이터 필요 (대략 20개 이상 권장)
    if len(prophet_df) < 20:
        return None
    model = Prophet(daily_seasonality=False, yearly_seasonality=False)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    future_price = forecast['yhat'].iloc[-periods:].mean()
    return float(future_price)

# 간단 버전(미사용 가능)
def predict_price_arima(df, periods=3):
    y = df['trade_price']
    model = ARIMA(y, order=(1,1,1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    return float(forecast.mean())

def predict_price_xgb(df, periods=3):
    df = df.copy()
    df['lag_1'] = df['trade_price'].shift(1)
    df['ma_3'] = df['trade_price'].rolling(3).mean()
    df = df.dropna()
    if df.empty:
        return None
    X = df[['lag_1', 'ma_3']]
    y = df['trade_price']
    model = xgb.XGBRegressor()
    model.fit(X, y)
    pred = model.predict(X.tail(periods))
    return float(pred.mean())

def add_predictPrice(datetag,coinn,aupr,adownr,cprice,pra,prb,prc,prd,rta,rtb,rtc,rtd,intv):
    # None 안전 처리 및 반올림(너무 긴 실수 URL 방지)
    def fmt(v):
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return 0
        return round(float(v), 6)
    url = f'http://ywydpapa.iptime.org:8000/rest_add_predict/{datetag}/{coinn}/{fmt(aupr)}/{fmt(adownr)}/{fmt(cprice)}/{fmt(pra)}/{fmt(prb)}/{fmt(prc)}/{fmt(prd)}/{fmt(rta)}/{fmt(rtb)}/{fmt(rtc)}/{fmt(rtd)}/{intv}'
    try:
        response = requests.get(url, timeout=5)
        return response
    except Exception as e:
        print("add_predictPrice 호출 실패:", e)
        return None

# ------------- Pattern Helpers -------------
def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def paa(series, segments=6):
    n = len(series)
    if n < segments or segments <= 0:
        return np.array(series)
    segment_size = n // segments
    if segment_size == 0:
        return np.array(series)
    paa_result = [series[i*segment_size:(i+1)*segment_size].mean() for i in range(segments)]
    return np.array(paa_result)

def sax_transform(paa_vector, alphabet='abcde'):
    if len(paa_vector) == 0:
        return ''
    breakpoints = np.percentile(paa_vector, np.linspace(0, 100, len(alphabet)+1)[1:-1])
    sax_string = ''
    for val in paa_vector:
        idx = sum(val > bp for bp in breakpoints)
        sax_string += alphabet[idx]
    return sax_string

# ------------- Bollinger Pattern Detection -------------
def detect_bb_patterns(
    df,
    lookback=6,
    tolerance_ratio=0.07,
    confirm_candles=2,
    slope_window=5,
    squeeze_lookback=80,
    squeeze_percentile=0.25
):
    meta = {}
    # 최소 행 수 조건
    if len(df) < max(lookback, slope_window, squeeze_lookback) + 25:
        return "BB_NO_SIGNAL", {"reason": "데이터 부족(raw length)"}

    needed = ['upper','lower','middle','mavg','bb_width','trade_price']
    for c in needed:
        if c not in df.columns:
            return "BB_NO_SIGNAL", {"reason": f"컬럼 누락:{c}"}

    recent = df.tail(lookback).copy()

    # 유효 데이터 필터
    recent = recent.dropna(subset=['upper','lower','trade_price','mavg'])
    if len(recent) < lookback * 0.8:  # 80% 미만 유효하면 중단
        return "BB_NO_SIGNAL", {"reason": "유효 캔들 부족(NaN 제거)"}

    # 밴드 폭
    band_range = (recent['upper'] - recent['lower'])
    # 0 또는 음수 제거
    mask_valid = band_range > 0
    recent = recent[mask_valid]
    if len(recent) < max(3, lookback - 2):
        return "BB_NO_SIGNAL", {"reason": "밴드 폭 0(=std=0) 다수"}

    # band_pos
    band_range = (recent['upper'] - recent['lower'])
    recent['band_pos'] = (recent['trade_price'] - recent['lower']) / band_range
    recent = recent.dropna(subset=['band_pos'])
    if recent.empty:
        return "BB_NO_SIGNAL", {"reason": "band_pos 전부 NaN"}

    # 터치 판정
    recent['upper_touch'] = recent['band_pos'] >= (1 - tolerance_ratio)
    recent['lower_touch'] = recent['band_pos'] <= tolerance_ratio

    upper_touch_count = int(recent['upper_touch'].sum())
    lower_touch_count = int(recent['lower_touch'].sum())

    # 기울기 함수
    def lin_slope(series):
        y = series.dropna()
        if len(y) < 2:
            return np.nan
        X = np.arange(len(y)).reshape(-1,1)
        model = LinearRegression()
        model.fit(X, y.values)
        return model.coef_[0]

    mavg_slope = lin_slope(df['mavg'].tail(slope_window))
    vwma_slope = lin_slope(df['vwma'].tail(slope_window)) if 'vwma' in df.columns else np.nan

    # squeeze
    if 'bb_width' in df.columns and len(df['bb_width'].dropna()) >= squeeze_lookback * 0.6:
        recent_width = df['bb_width'].iloc[-1]
        hist_widths = df['bb_width'].tail(squeeze_lookback).dropna()
        if not hist_widths.empty:
            pct_rank = (hist_widths < recent_width).mean()
            is_squeeze = pct_rank <= squeeze_percentile
            meta['bb_width_percentile'] = round(pct_rank,4)
        else:
            is_squeeze = False
    else:
        is_squeeze = False

    meta.update({
        "upper_touch_count": upper_touch_count,
        "lower_touch_count": lower_touch_count,
        "mavg_slope": mavg_slope,
        "vwma_slope": vwma_slope,
        "is_squeeze": is_squeeze
    })

    last = recent.iloc[-1]
    signal = None
    reason = None

    confirm_zone = recent.tail(confirm_candles)
    # 변경 가능: 최근 subset 으로 change 값이 없다면 0으로 간주
    if 'change' not in df.columns:
        df['change'] = df['trade_price'].diff()

    # 매도 전환
    if upper_touch_count > 0 and signal is None:
        ut_idx = recent[recent['upper_touch']].index[-1]
        after_ut = recent[recent.index > ut_idx]
        if len(after_ut) >= 1:
            if (last['trade_price'] < last.get('middle', last['trade_price'])) or (last['band_pos'] < 0.7):
                confirm_down_ratio = (confirm_zone['change'] < 0).mean()
                if confirm_down_ratio >= 0.5 and ( (not np.isnan(mavg_slope) and mavg_slope <= 0) or (not np.isnan(vwma_slope) and vwma_slope <= 0) ):
                    signal = "BB_REVERSAL_SELL"
                    reason = "상단터치→재진입+하향전환"

    # 매수 전환
    if lower_touch_count > 0 and signal is None:
        lt_idx = recent[recent['lower_touch']].index[-1]
        after_lt = recent[recent.index > lt_idx]
        if len(after_lt) >= 1:
            if (last['trade_price'] > last.get('middle', last['trade_price'])) or (last['band_pos'] > 0.3):
                confirm_up_ratio = (confirm_zone['change'] > 0).mean()
                if confirm_up_ratio >= 0.5 and ( (not np.isnan(mavg_slope) and mavg_slope >= 0) or (not np.isnan(vwma_slope) and vwma_slope >= 0) ):
                    signal = "BB_REVERSAL_BUY"
                    reason = "하단터치→재진입+상향전환"

    # 추세 지속
    if signal is None:
        mid_col = 'middle' if 'middle' in recent.columns else 'mavg'
        above_mid_ratio = (recent['trade_price'] > recent[mid_col]).mean()
        below_mid_ratio = (recent['trade_price'] < recent[mid_col]).mean()
        if upper_touch_count >= 2 and above_mid_ratio >= 0.7 and (not np.isnan(mavg_slope) and mavg_slope > 0):
            signal = "BB_TREND_CONT_UP"
            reason = "상단워킹"
        elif lower_touch_count >= 2 and below_mid_ratio >= 0.7 and (not np.isnan(mavg_slope) and mavg_slope < 0):
            signal = "BB_TREND_CONT_DOWN"
            reason = "하단워킹"

    # 스퀴즈 돌파
    if signal is None and is_squeeze:
        if len(df) > 5:
            prev_width_mean = df['bb_width'].tail(5).iloc[:-1].dropna().mean()
            cur_width = df['bb_width'].iloc[-1]
            if prev_width_mean and prev_width_mean > 0:
                expand_ratio = cur_width / prev_width_mean
                if expand_ratio >= 1.2:
                    mid_val = last.get('middle', np.nan)
                    if not np.isnan(mid_val):
                        if last['trade_price'] > mid_val and (not np.isnan(mavg_slope) and mavg_slope > 0):
                            signal = "BB_SQUEEZE_BREAKOUT_UP"
                            reason = "스퀴즈상향"
                        elif last['trade_price'] < mid_val and (not np.isnan(mavg_slope) and mavg_slope < 0):
                            signal = "BB_SQUEEZE_BREAKOUT_DOWN"
                            reason = "스퀴즈하향"

    if signal is None:
        signal = "BB_NO_SIGNAL"
        reason = "특이패턴없음"

    meta['reason'] = reason
    return signal, meta


# ------------- Main peak_trade -------------
def peak_trade(
        ticker='KRW-BTC',
        short_window=3,
        long_window=20,
        count=180,
        candle_unit='1h'
):
    candle_map = {
        '1d': ('days', ''),
        '4h': ('minutes', 240),
        '1h': ('minutes', 60),
        '30m': ('minutes', 30),
        '15m': ('minutes', 15),
        '10m': ('minutes', 10),
        '5m': ('minutes', 5),
        '3m': ('minutes', 3),
        '1m': ('minutes', 1),
    }
    if candle_unit not in candle_map:
        raise ValueError(f"지원하지 않는 단위입니다: {candle_unit}")
    api_type, minute = candle_map[candle_unit]
    if api_type == 'days':
        url = f'https://api.upbit.com/v1/candles/days?market={ticker}&count={count}'
    else:
        url = f'https://api.upbit.com/v1/candles/minutes/{minute}?market={ticker}&count={count}'

    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data)
    if df.empty:
        return {"error": "빈 데이터"}
    df['candle_date_time_kst'] = pd.to_datetime(df['candle_date_time_kst'], format='%Y-%m-%dT%H:%M:%S')
    df.set_index('candle_date_time_kst', inplace=True)
    df = df.sort_index(ascending=True)

    if isinstance(df.index, pd.DatetimeIndex):
        inferred_freq = pd.infer_freq(df.index)
        if inferred_freq is not None:
            df = df.asfreq(inferred_freq)
        else:
            freq_map = {
                '1m': '1min','3m':'3min','5m':'5min','10m':'10min',
                '15m':'15min','30m':'30min','1h':'60min','4h':'240min','1d':'D'
            }
            freq = freq_map.get(candle_unit, None)
            if freq:
                df = df.asfreq(freq)

    keep_cols = ['trade_price','high_price','low_price','candle_acc_trade_volume']
    existing = [c for c in keep_cols if c in df.columns]
    df = df[existing]

    df['prev_price'] = df['trade_price'].shift(1)
    df['change'] = df['trade_price'] - df['prev_price']
    df['rate'] = (df['trade_price'] - df['prev_price']) / df['prev_price']

    window = 20; k = 2
    window2 = 4; k2 = 4
    df['middle'] = df['trade_price'].rolling(window).mean()
    df['mavg'] = df['middle']
    df['std'] = df['trade_price'].rolling(window).std()
    df['upper'] = df['mavg'] + k * df['std']
    df['lower'] = df['mavg'] - k * df['std']

    df['middle2'] = df['trade_price'].rolling(window2).mean()
    df['mavg2'] = df['middle2']
    df['std2'] = df['trade_price'].rolling(window2).std()
    df['upper2'] = df['mavg2'] + k2 * df['std2']
    df['lower2'] = df['mavg2'] - k2 * df['std2']

    df['bb_width'] = df['upper'] - df['lower']
    df['bb_width2'] = df['upper2'] - df['lower2']

    avg_bb_width_10 = df['bb_width'].tail(10).mean()
    avg_bb_width_20 = df['bb_width'].tail(20).mean()

    df['vwma'] = (df['trade_price'] * df['candle_acc_trade_volume']).rolling(window).sum() / \
                 (df['candle_acc_trade_volume'].rolling(window).sum())

    up_candles = df[df['change'] > 0]
    down_candles = df[df['change'] < 0]
    avg_up_rate = up_candles['rate'].tail(10).mean() * 100
    avg_down_rate = down_candles['rate'].tail(10).mean() * 100

    trsignal = ''
    bb_signal = None
    bb_meta = {}

    try:
        freq = 'h' if 'h' in candle_unit else 'min'
        future_price = predict_future_price(df, periods=2, freq=freq)
        future_price_arima = predict_future_price_arima(df, periods=2)
        now_price = df['trade_price'].iloc[-1]
        pred_rate = ((future_price - now_price) / now_price * 100) if future_price else None
        pred_rate_arima = ((future_price_arima - now_price) / now_price * 100) if future_price_arima else None
        future_price_xgb = predict_future_price_xgb(df, periods=2, n_lags=5, method='xgb')
        pred_rate_xgb = ((future_price_xgb - now_price) / now_price * 100) if future_price_xgb else None

        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(f"현재가: {now_price:.2f}")
        if future_price_arima is not None:
            print(f"[ARIMA]   2캔들 뒤 예측가: {future_price_arima:.2f} / 예측 변화율: {pred_rate_arima:.3f}%")
        if future_price_xgb is not None:
            print(f"[XGBoost] 2캔들 뒤 예측가: {future_price_xgb:.2f} / 예측 변화율: {pred_rate_xgb:.3f}%")
        if future_price is not None:
            print(f"[Prophet] 2캔들 뒤 예측가: {future_price:.2f} / 예측 변화율: {pred_rate:.3f}%")

        print(f"상승봉 평균 변화율(최근10개): {avg_up_rate:.3f}%")
        print(f"하강봉 평균 변화율(최근10개): {avg_down_rate:.3f}%")

        if pred_rate_xgb is not None:
            if pred_rate_xgb > avg_up_rate:
                trsignal = 'BUY'
            elif pred_rate_xgb < avg_down_rate:
                trsignal = 'SELL'
            else:
                trsignal = 'HOLD'
        else:
            trsignal = 'HOLD'

        last = df.iloc[-1]
        last_bb_width = df['bb_width'].iloc[-1]
        last_bb_width2 = df['bb_width2'].iloc[-1]

        avg_bb_width_10_pct = avg_bb_width_10 / now_price * 100 if now_price else None
        avg_bb_width_20_pct = avg_bb_width_20 / now_price * 100 if now_price else None
        last_bb_width_pct = last_bb_width / now_price * 100 if now_price else None
        last_bb_width_pct2 = last_bb_width2 / now_price * 100 if now_price else None

        if last['trade_price'] >= (last['middle'] if not np.isnan(last['middle']) else last['trade_price']):
            bollinger_pos = (last['trade_price'] - last['middle']) / (last['upper'] - last['middle']) * 100 if (last['upper'] - last['middle']) != 0 else 0
        else:
            denom = (last['middle'] - last['lower'])
            bollinger_pos = (last['trade_price'] - last['middle']) / denom * 100 if denom != 0 else 0

        vwma_last = last['vwma']
        mavg_last = last['mavg']
        mavg2_last = last['mavg2']

        bb_signal, bb_meta = detect_bb_patterns(
            df,
            lookback=6,
            tolerance_ratio=0.07,
            confirm_candles=2,
            slope_window=5,
            squeeze_lookback=80,
            squeeze_percentile=0.25
        )
        print(f"[BB 패턴 신호] {bb_signal} / {bb_meta.get('reason')}")

        return {
            "ticker": ticker,
            "avg_up_rate": avg_up_rate,
            "avg_down_rate": avg_down_rate,
            "now_price": now_price,
            "future_price_prophet": future_price,
            "future_price_arima": future_price_arima,
            "future_price_xgb": future_price_xgb,
            "pred_rate_prophet": pred_rate,
            "pred_rate_arima": pred_rate_arima,
            "pred_rate_xgb": pred_rate_xgb,
            "model_signal": trsignal,
            "bb_signal": bb_signal,
            "bb_meta": bb_meta
        }

    except Exception as e:
        print("예측 실패:", e)
        return {"error": str(e)}

def get_upbit_krw_coins():
    url = "https://api.upbit.com/v1/market/all"
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers)
    markets = response.json()
    krw_coins = [market['market'] for market in markets if market['market'].startswith('KRW-')]
    return krw_coins

# ------------- 실행 루프 -------------
if __name__ == "__main__":
    coin_list = get_upbit_krw_coins()

    # 테스트를 위해 단일 코인 고정
    target_coins = ['KRW-IP']
    intervals = ['30m', '15m', '5m']

    while True:
        nowt = datetime.datetime.now()
        datetag = nowt.strftime("%Y%m%d%H%M%S")
        print('예측 시간 : ', nowt.strftime("%Y-%m-%d %H:%M:%S"))

        for coinn in target_coins:
            print("예측코인 :", coinn)
            for intv in intervals:
                print(f"{intv} 예측 시작")
                fpst = peak_trade(coinn, 1, 20, 200, intv)
                if 'error' in fpst:
                    print(f"peak_trade 오류: {fpst['error']}")
                    continue
                print("------------------------------------------------")
            print(f'{coinn} 예측 완료')
            time.sleep(0.5)

        # 다음 2분 단위까지 대기
        now = datetime.datetime.now()
        next_minute = (now.minute // 2 + 1) * 2
        if next_minute == 60:
            next_time = (now + datetime.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        else:
            next_time = now.replace(minute=next_minute, second=0, microsecond=0)
        sleep_seconds = (next_time - now).total_seconds()
        print(f"다음 실행까지 {sleep_seconds:.1f}초 대기")
        time.sleep(sleep_seconds)