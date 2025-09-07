import time
import requests
import pandas as pd
from prophet import Prophet
import datetime
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import lightgbm as lgb
import numpy as np

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
    feature_cols = [f'lag_{i}' for i in range(1, n_lags + 1)] + ['volume', 'ma_3', 'ma_7', 'std_3', 'std_7']
    X = df_feat[feature_cols]
    y = df_feat['trade_price']
    last_row = X.iloc[[-1]].copy()  # DataFrame 형태 유지
    if method == 'xgb':
        model = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42, verbosity=0)
    else:
        model = lgb.LGBMRegressor(n_estimators=100, max_depth=3, random_state=42, verbosity=0)
    model.fit(X, y)
    preds = []
    recent_prices = list(df_feat['trade_price'].iloc[-n_lags:].values)
    current_feats = last_row.copy()
    for _ in range(periods):
        pred = model.predict(current_feats)[0]
        preds.append(pred)
        new_lags = np.roll(current_feats.values[0][:n_lags], 1)
        new_lags[0] = pred
        for i in range(n_lags):
            current_feats.iloc[0, i] = new_lags[i]
        recent_prices = list(new_lags[:n_lags])
        current_feats.iloc[0, n_lags + 1] = np.mean(recent_prices[-3:])  # ma_3
        current_feats.iloc[0, n_lags + 2] = np.mean(recent_prices[-7:])  # ma_7
        current_feats.iloc[0, n_lags + 3] = np.std(recent_prices[-3:])  # std_3
        current_feats.iloc[0, n_lags + 4] = np.std(recent_prices[-7:])  # std_7
    return np.mean(preds)


def predict_future_price_arima(df, periods=3):
    # ARIMA는 시계열 index가 필요합니다.
    y = df['trade_price']
    # 간단한 (자동) 파라미터: (1,1,1) 또는 더 정교하게 pmdarima의 auto_arima로 최적화 가능
    try:
        model = ARIMA(y, order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=periods)
        future_price = forecast.mean()  # ARIMA는 미래값이 여러개면 평균을 사용
        return future_price
    except Exception as e:
        print("ARIMA 예측 실패:", e)
        return None

def predict_future_price(df, periods=3, freq='3min'):
    prophet_df = df.reset_index()[['candle_date_time_kst', 'trade_price']].rename(
        columns={'candle_date_time_kst': 'ds', 'trade_price': 'y'}
    )
    model = Prophet(daily_seasonality=False, yearly_seasonality=False)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    future_price = forecast['yhat'].iloc[-periods:].mean()
    return future_price

def predict_price_arima(df, periods=3):
    from statsmodels.tsa.arima.model import ARIMA
    y = df['trade_price']
    model = ARIMA(y, order=(1,1,1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    return forecast.mean()

def predict_price_xgb(df, periods=3):
    import xgboost as xgb
    df['lag_1'] = df['trade_price'].shift(1)
    df['ma_3'] = df['trade_price'].rolling(3).mean()
    df = df.dropna()
    X = df[['lag_1', 'ma_3']]
    y = df['trade_price']
    model = xgb.XGBRegressor()
    model.fit(X, y)
    pred = model.predict(X.tail(periods))
    return pred.mean()

def add_predictPrice(datetag,coinn,aupr,adownr,cprice,pra,prb,prc,prd,rta,rtb,rtc,rtd,intv):
    url = f'http://ywydpapa.iptime.org:8000/rest_add_predict/{datetag}/{coinn}/{aupr}/{adownr}/{cprice}/{pra}/{prb}/{prc}/{prd}/{rta}/{rtb}/{rtc}/{rtd}/{intv}'
    response = requests.get(url)
    return response

def levenshtein(s1, s2):
    """레벤슈타인 거리(문자열 유사도)"""
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
    """시계열을 segments 개의 구간 평균으로 단순화"""
    n = len(series)
    segment_size = n // segments
    paa_result = [series[i*segment_size:(i+1)*segment_size].mean() for i in range(segments)]
    return np.array(paa_result)

def sax_transform(paa_vector, alphabet='abcde'):
    """PAA 벡터를 심볼 문자열로 변환 (알파벳 개수만큼 등분위로 나눔)"""
    breakpoints = np.percentile(paa_vector, np.linspace(0, 100, len(alphabet)+1)[1:-1])
    sax_string = ''
    for val in paa_vector:
        idx = sum(val > bp for bp in breakpoints)
        sax_string += alphabet[idx]
    return sax_string

def peak_trade(
        ticker='KRW-BTC',
        short_window=3,
        long_window=20,
        count=180,
        candle_unit='1h'
):
    # 0. 캔들 단위 변환
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
    global trguide
    if candle_unit not in candle_map:
        raise ValueError(f"지원하지 않는 단위입니다: {candle_unit}")
    api_type, minute = candle_map[candle_unit]
    if api_type == 'days':
        url = f'https://api.upbit.com/v1/candles/days?market={ticker}&count={count}'
    else:
        url = f'https://api.upbit.com/v1/candles/minutes/{minute}?market={ticker}&count={count}'
    # 1. 데이터 가져오기
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data)
    df['candle_date_time_kst'] = pd.to_datetime(df['candle_date_time_kst'], format='%Y-%m-%dT%H:%M:%S')
    df.set_index('candle_date_time_kst', inplace=True)
    df = df.sort_index(ascending=True)
    # === 자동 freq 지정 ===
    if isinstance(df.index, pd.DatetimeIndex):
        inferred_freq = pd.infer_freq(df.index)
        if inferred_freq is not None:
            df = df.asfreq(inferred_freq)
        else:
            freq_map = {
                '1m': '1min', '3m': '3min', '5m': '5min', '10m': '10min',
                '15m': '15min', '30m': '30min', '1h': '60min', '4h': '240min', '1d': 'D'
            }
            freq = freq_map.get(candle_unit, None)
            if freq:
                df = df.asfreq(freq)
    # =====================
    df = df[['trade_price', 'candle_acc_trade_volume']]
    df['prev_price'] = df['trade_price'].shift(1)
    df['change'] = df['trade_price'] - df['prev_price']
    df['rate'] = (df['trade_price'] - df['prev_price']) / df['prev_price']
    # 패턴 분석
    segments = 6  # 구간 수(원하는 대로 조정)
    alphabet = 'abcde'  # 심볼 개수(원하는 대로 조정)

    # 최근 N개의 종가를 PAA + SAX로 변환
    recent_prices = df['trade_price'].tail(segments * 10)  # 예시: 60개 데이터
    paa_vec = paa(recent_prices, segments=segments)
    sax_str = sax_transform(paa_vec, alphabet=alphabet)
    print(f"SAX 변환 결과: {sax_str}")

    pattern_library = {
        'morning_star_like': 'cbaabc',
        'w_shape': 'abccba',
        'uptrend': 'abcde',
        'downtrend': 'edcba',
    }

    # 볼린저 밴드 계산 (20기간, 표준편차 2)
    window = 20
    k = 2
    window2 = 4
    k2 = 4
    df['middle'] = df['trade_price'].rolling(window).mean()
    df['mavg'] = df['trade_price'].rolling(window).mean()
    df['std'] = df['trade_price'].rolling(window).std()
    df['upper'] = df['mavg'] + k * df['std']
    df['lower'] = df['mavg'] - k * df['std']
    df['middle2'] = df['trade_price'].rolling(window2).mean()
    df['mavg2'] = df['trade_price'].rolling(window2).mean()
    df['std2'] = df['trade_price'].rolling(window2).std()
    df['upper2'] = df['mavg2'] + k2 * df['std2']
    df['lower2'] = df['mavg2'] - k2 * df['std2']
    # 볼린저밴드 폭 계산
    df['bb_width'] = df['upper'] - df['lower']
    df['bb_width2'] = df['upper2'] - df['lower2']
    # 최근 10개 캔들의 평균폭 계산
    avg_bb_width_10 = df['bb_width'].tail(10).mean()
    avg_bb_width_20 = df['bb_width'].tail(20).mean()
    # === VWMA(거래량 가중이동평균) 20 추가 ===
    df['vwma'] = (df['trade_price'] * df['candle_acc_trade_volume']).rolling(window).sum() / df['candle_acc_trade_volume'].rolling(window).sum()
    up_candles = df[df['change'] > 0]
    down_candles = df[df['change'] < 0]
    avg_up_rate = up_candles['rate'].tail(10).mean() * 100  # %
    avg_down_rate = down_candles['rate'].tail(10).mean() * 100  # %
    trsignal = ''
    try:
        freq = 'h' if 'h' in candle_unit else 'min'
        # Prophet 예측
        future_price = predict_future_price(df, periods=2, freq=freq)
        # ARIMA 예측
        future_price_arima = predict_future_price_arima(df, periods=2)
        now_price = df['trade_price'].iloc[-1]
        pred_rate = (future_price - now_price) / now_price * 100
        pred_rate_arima = (future_price_arima - now_price) / now_price * 100 if future_price_arima is not None else None
        # XGBoost 예측
        future_price_xgb = predict_future_price_xgb(df, periods=2, n_lags=5, method='xgb')
        pred_rate_xgb = (future_price_xgb - now_price) / now_price * 100 if future_price_xgb is not None else None
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(f"현재가: {now_price:.2f}")
        if future_price_arima is not None:
            print(f"[ARIMA]   2캔들 뒤 예측가: {future_price_arima:.2f} / 예측 변화율: {pred_rate_arima:.3f}%")
        print(f"[XGBoost] 2캔들 뒤 예측가: {future_price_xgb:.2f} / 예측 변화율: {pred_rate_xgb:.3f}%")
        print(f"상승봉 평균 변화율(최근10개): {avg_up_rate:.3f}%")
        print(f"하강봉 평균 변화율(최근10개): {avg_down_rate:.3f}%")
        # 3. 비교 및 신호 판단 (Prophet 기준, 필요시 ARIMA도 추가)
        if pred_rate_xgb > avg_up_rate:
            print("예측 변화율이 상승봉 평균 변화율보다 높음 → 강한 매수 신호! (XGB 기준)")
            trsignal = 'BUY'
        elif pred_rate_xgb < avg_down_rate:
            print("예측 변화율이 하강봉 평균 변화율보다 낮음 → 강한 매도 신호! (XGB 기준)")
            trsignal = 'SELL'
        else:
            print("예측 변화율이 평균 변화율 범위 내 → 특별 신호 없음 (XGB 기준)")
            trsignal = 'HOLD'
        last = df.iloc[-1]
        last_bb_width = df['bb_width'].iloc[-1]
        last_bb_width2 = df['bb_width2'].iloc[-1]
        avg_bb_width_10_pct = avg_bb_width_10 / now_price * 100
        avg_bb_width_20_pct = avg_bb_width_20 / now_price * 100
        last_bb_width_pct = last_bb_width / now_price * 100
        last_bb_width_pct2 = last_bb_width2 / now_price * 100
        print(f"최근 20개 캔들의 볼린저밴드 평균 폭: {avg_bb_width_20:.2f} ({avg_bb_width_20_pct:.2f}%)")
        print(f"최근 10개 캔들의 볼린저밴드 평균 폭: {avg_bb_width_10:.2f} ({avg_bb_width_10_pct:.2f}%)")
        print(f"마지막 캔들의 볼린저밴드 폭: {last_bb_width:.2f} ({last_bb_width_pct:.2f}%)")
        print(f"마지막 캔들의 볼린저밴드44 폭: {last_bb_width2:.2f} ({last_bb_width_pct2:.2f}%)")
        if last['trade_price'] >= last['middle']:
            bollinger_pos = (last['trade_price'] - last['middle']) / (last['upper'] - last['middle']) * 100
        else:
            bollinger_pos = (last['trade_price'] - last['middle']) / (last['middle'] - last['lower']) * 100

        print(f"볼린저밴드 내 위치: {bollinger_pos:.2f}% (-100%: 하단, 0%: 중심, +100%: 상단)")

        vwma_last = last['vwma']
        mavg_last = last['mavg']
        mavg2_last = last['mavg2']

        if pd.isna(vwma_last) or pd.isna(mavg_last):
            print("VWMA 또는 볼린저밴드 중간값 계산 불가: 데이터가 부족하거나 결측치가 있습니다.")
            vwma_vs_mavg_pos = None
        else:
            # VWMA가 볼린저밴드 중간값보다 얼마나 위/아래에 있는지 (%)
            vwma_vs_mavg_pos = (vwma_last - mavg_last) / mavg_last * 100
            vwma_vs_mavg2_pos = (vwma_last - mavg2_last) / mavg2_last * 100
            print(f"VWMA: {vwma_last:.2f} / 볼린저밴드 중간값(MA): {mavg_last:.2f}")
            print(f"VWMA가 볼린저밴드 중간값 대비 {vwma_vs_mavg_pos:.2f}% {'위' if vwma_vs_mavg_pos > 0 else '아래'}에 위치")
            print(f"VWMA: {vwma_last:.2f} / 볼린저밴드44 중간값(MA): {mavg2_last:.2f}")
            print(f"VWMA가 볼린저밴드44 중간값 대비 {vwma_vs_mavg2_pos:.2f}% {'위' if vwma_vs_mavg2_pos > 0 else '아래'}에 위치")
        # VWMA와 현재가의 위치도 같이 출력
        now_price = last['trade_price']
        if pd.isna(vwma_last):
            print("VWMA 계산 불가")
            price_vs_vwma_pos = None
        else:
            price_vs_vwma_pos = (now_price - vwma_last) / vwma_last * 100
            print(f"현재가가 VWMA 대비 {price_vs_vwma_pos:.2f}% {'위' if price_vs_vwma_pos > 0 else '아래'}에 위치")
        for name, pattern in pattern_library.items():
            dist = levenshtein(sax_str, pattern)
            print(f"패턴 [{name}]과의 거리: {dist}")
            if dist <= 2:  # 거리 임계값(조절 가능)
                print(f"→ {name} 패턴과 유사함!")
        return ticker, avg_up_rate,avg_down_rate,now_price, future_price, future_price_arima, future_price_xgb, pred_rate, pred_rate_arima, pred_rate_xgb

    except Exception as e:
        print("예측 실패:", e)


def get_upbit_krw_coins():
    url = "https://api.upbit.com/v1/market/all"
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers)
    markets = response.json()
    # KRW 마켓만 추출
    krw_coins = [market['market'] for market in markets if market['market'].startswith('KRW-')]
    return krw_coins

coin_list = get_upbit_krw_coins()

while True:
    nowt = datetime.datetime.now()
    datetag = nowt.strftime("%Y%m%d%H%M%S")
    print('예측 시간 : ', nowt.strftime("%Y-%m-%d %H:%M:%S"))
    for coinn in ['KRW-ADA']:
        try:
            print('예측 시간 : ', nowt.strftime("%Y-%m-%d %H:%M:%S"))
            print("예측코인 :", coinn)

            intervals = ['30m', '15m', '5m']
            for intv in intervals:
                print(f"{intv} 예측")
                fpst = peak_trade(coinn, 1, 20, 200, intv)
                add_predictPrice(datetag, fpst[0], fpst[1], fpst[2], fpst[3], fpst[4], fpst[5], fpst[6], 0, fpst[7], fpst[8], fpst[9], 0, intv)
                print("------------------------------------------------")
            print(f'{coinn} 예측 완료')
            time.sleep(0.3)
        except Exception as e:
            print(f'<UNK> <UNK>: {e}')
    # === 2분 정각까지 대기 ===
    now = datetime.datetime.now()
    # 다음 2분 단위 계산
    next_minute = (now.minute // 2 + 1) * 2
    if next_minute == 60:
        # 1시간 더하기 (timedelta 사용)
        next_time = (now + datetime.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    else:
        next_time = now.replace(minute=next_minute, second=0, microsecond=0)
    sleep_seconds = (next_time - now).total_seconds()
    print(f"다음 실행까지 {sleep_seconds:.1f}초 대기")
    time.sleep(sleep_seconds)
