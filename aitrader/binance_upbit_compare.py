import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_time_lag():
    # 1. 업비트 1분봉 데이터 가져오기 (최근 200분)
    url_upbit = "https://api.upbit.com/v1/candles/minutes/1"
    params_upbit = {"market": "KRW-XRP", "count": 200}
    res_upbit = requests.get(url_upbit, params=params_upbit).json()
    df_upbit = pd.DataFrame(res_upbit)[['candle_date_time_kst', 'trade_price']]
    df_upbit.columns = ['time', 'upbit_price']
    df_upbit['time'] = pd.to_datetime(df_upbit['time'])

    # 2. 바이낸스 1분봉 데이터 가져오기 (최근 200분)
    url_binance = "https://fapi.binance.com/fapi/v1/klines" # 선물 API 사용(접근성 더 좋음)
    params_binance = {"symbol": "XRPUSDT", "interval": "1m", "limit": 200}
    res_binance = requests.get(url_binance, params=params_binance).json()
    df_binance = pd.DataFrame(res_binance)[[0, 4]]
    df_binance.columns = ['time', 'binance_price']
    df_binance['time'] = pd.to_datetime(df_binance['time'], unit='ms') + pd.Timedelta(hours=9)
    df_binance['binance_price'] = df_binance['binance_price'].astype(float)

    # 3. 데이터 병합
    df = pd.merge(df_upbit, df_binance, on='time', how='inner')
    df = df.sort_values('time').reset_index(drop=True)

    if df.empty:
        print("데이터 병합 실패")
        return

    # 4. 정규화 (Z-score normalization)
    df['upbit_norm'] = (df['upbit_price'] - df['upbit_price'].mean()) / df['upbit_price'].std()
    df['binance_norm'] = (df['binance_price'] - df['binance_price'].mean()) / df['binance_price'].std()

    # 5. 교차 상관관계(Cross-Correlation) 계산
    lags = range(-30, 31) # -30분 ~ +30분 시프트
    corrs = []

    for lag in lags:
        # lag > 0: 바이낸스 선행 (바이낸스 과거가 업비트 현재와 일치)
        # lag < 0: 업비트 선행
        corr = df['upbit_norm'].corr(df['binance_norm'].shift(lag))
        corrs.append(corr)

    best_lag = lags[np.argmax(corrs)]
    max_corr = max(corrs)

    # 6. 결과 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(lags, corrs, marker='o', color='#8E44AD')
    plt.axvline(best_lag, color='red', linestyle='--', label=f'Best Lag: {best_lag} min')
    plt.title(f'Cross-Correlation: Upbit vs Binance (1m candles)\nMax Correlation: {max_corr:.4f} at Lag {best_lag} min', fontsize=12)
    plt.xlabel('Lag (minutes)', fontsize=10)
    plt.ylabel('Correlation Coefficient', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"가장 유사한 패턴의 시간차: {best_lag}분 (상관계수: {max_corr:.4f})")
    if best_lag > 0:
        print(f"해석: 바이낸스 가격이 업비트보다 {best_lag}분 선행하여 움직입니다.")
    elif best_lag < 0:
        print(f"해석: 업비트 가격이 바이낸스보다 {abs(best_lag)}분 선행하여 움직입니다.")
    else:
        print("해석: 두 거래소의 가격이 지연 없이 동기화되어 움직입니다.")

if __name__ == "__main__":
    calculate_time_lag()
