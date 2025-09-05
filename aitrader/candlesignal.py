import time
import requests
import pandas as pd
from prophet import Prophet
import datetime
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import sklearn as sk
from lightgbm import LGBMRegressor

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

def predict_price_prophet(df, periods=3, freq='1min'):
    from prophet import Prophet
    df = df.rename(columns={'candle_date_time_kst': 'ds', 'trade_price': 'y'})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast['yhat'].iloc[-periods:].mean()

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
    up_candles = df[df['change'] > 0]
    down_candles = df[df['change'] < 0]
    avg_up_rate = up_candles['rate'].mean() * 100  # %
    avg_down_rate = down_candles['rate'].mean() * 100  # %
    print(f"상승봉 평균 상승률: {avg_up_rate:.3f}%")
    print(f"하강봉 평균 하강률: {avg_down_rate:.3f}%")
    # =====================================
    #AI 신호예측 부분
    trsignal = ''
    try:
        freq = 'h' if 'h' in candle_unit else 'min'
        # Prophet 예측
        future_price = predict_future_price(df, periods=3, freq=freq)
        # ARIMA 예측
        future_price_arima = predict_future_price_arima(df, periods=3)
        now_price = df['trade_price'].iloc[-1]
        pred_rate = (future_price - now_price) / now_price * 100
        pred_rate_arima = (future_price_arima - now_price) / now_price * 100 if future_price_arima is not None else None
        # XGBoost 예측
        future_price_xgb = predict_future_price_xgb(df, periods=3, n_lags=5, method='xgb')
        pred_rate_xgb = (future_price_xgb - now_price) / now_price * 100 if future_price_xgb is not None else None
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(f"현재가: {now_price:.2f}")
        print(f"[Prophet] 3캔들 뒤 예측가: {future_price:.2f} / 예측 변화율: {pred_rate:.3f}%")
        if future_price_arima is not None:
            print(f"[ARIMA]   3캔들 뒤 예측가: {future_price_arima:.2f} / 예측 변화율: {pred_rate_arima:.3f}%")
        print(f"[XGBoost] 3캔들 뒤 예측가: {future_price_xgb:.2f} / 예측 변화율: {pred_rate_xgb:.3f}%")
        print(f"상승봉 평균 변화율: {avg_up_rate:.3f}%")
        print(f"하강봉 평균 변화율: {avg_down_rate:.3f}%")
        # 3. 비교 및 신호 판단 (Prophet 기준, 필요시 ARIMA도 추가)
        if pred_rate > avg_up_rate:
            print("예측 변화율이 상승봉 평균 변화율보다 높음 → 강한 매수 신호! (Prophet 기준)")
            trsignal = 'BUY'
        elif pred_rate < avg_down_rate:
            print("예측 변화율이 하강봉 평균 변화율보다 낮음 → 강한 매도 신호! (Prophet 기준)")
            trsignal = 'SELL'
        else:
            print("예측 변화율이 평균 변화율 범위 내 → 특별 신호 없음 (Prophet 기준)")
            trsignal = 'HOLD'
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
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
    print("4h~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~4h")
    for coinn in coin_list:
        try:
            print("예측코인 :", coinn)
            fpst = peak_trade(coinn, 1, 20, 200, '4h')
            intv = '4h'
            print("4H 예측")
            add_predictPrice(datetag,fpst[0], fpst[1], fpst[2], fpst[3], fpst[4], fpst[5], fpst[6], 0,fpst[7], fpst[8], fpst[9], 0,intv)
            fpst = peak_trade(coinn, 1, 20, 200, '30m')
            intv = '30m'
            print("30M 예측")
            add_predictPrice(datetag, fpst[0], fpst[1], fpst[2], fpst[3], fpst[4], fpst[5], fpst[6], 0, fpst[7],
                             fpst[8], fpst[9], 0, intv)
            fpst = peak_trade(coinn, 1, 20, 200, '5m')
            intv = '5m'
            print("5M 예측")
            add_predictPrice(datetag, fpst[0], fpst[1], fpst[2], fpst[3], fpst[4], fpst[5], fpst[6], 0, fpst[7],
                             fpst[8], fpst[9], 0, intv)
            print(f'{coinn} 예측 완료')
            time.sleep(0.3)
        except Exception as e:
            print(f'<UNK> <UNK>: {e}')
        print('예측 시간 : ', nowt.strftime("%Y-%m-%d %H:%M:%S"))
    time.sleep(3600)