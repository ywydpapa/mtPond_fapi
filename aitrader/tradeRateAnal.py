import requests
import pandas as pd
import time


def get_tick_size(price):
    """업비트 원화(KRW) 마켓 호가 단위 계산 함수"""
    if price >= 1000000:
        return 1000
    elif price >= 100000:
        return 100
    elif price >= 10000:
        return 10
    elif price >= 1000:
        return 1
    elif price >= 100:
        return 0.1
    elif price >= 10:
        return 0.01
    elif price >= 1:
        return 0.001
    elif price >= 0.1:
        return 0.0001
    else:
        return 0.00001


def get_filtered_coins(top_n=50, target_power=80.0, target_tick_percent=0.1):
    print("1. 업비트 KRW 마켓 목록을 불러옵니다...")
    url_markets = "https://api.upbit.com/v1/market/all"
    markets = requests.get(url_markets).json()
    krw_markets = [m['market'] for m in markets if m['market'].startswith('KRW-')]

    print("2. 현재 거래대금 상위 종목의 현재가 및 틱 변동률을 확인합니다...")
    url_ticker = "https://api.upbit.com/v1/ticker"
    tickers_res = requests.get(url_ticker, params={"markets": ",".join(krw_markets)}).json()

    df_tickers = pd.DataFrame(tickers_res)
    # 거래대금(acc_trade_price_24h) 기준으로 내림차순 정렬
    df_tickers = df_tickers.sort_values(by='acc_trade_price_24h', ascending=False)

    # 거래대금 상위 N개 추출 (필터링이 깐깐하므로 50개까지 넉넉히 조회)
    top_tickers = df_tickers.head(top_n).to_dict('records')

    print(f"\n3. [조건] 1틱 변동률 {target_tick_percent}% 이상 & 체결강도 {target_power}% 이상 필터링 시작...\n")
    result_coins = []

    for ticker in top_tickers:
        market = ticker['market']
        current_price = ticker['trade_price']

        # 1틱 사이즈 및 변동률(%) 계산
        tick_size = get_tick_size(current_price)
        tick_percent = (tick_size / current_price) * 100

        # 조건 1: 1틱 변동률이 0.1% 미만이면 체결강도 계산 없이 패스 (속도 최적화)
        if tick_percent < target_tick_percent:
            continue

        # API 호출 제한 방지
        time.sleep(0.15)

        # 최근 500개의 체결 내역(Tick) 가져오기
        url_ticks = "https://api.upbit.com/v1/trades/ticks"
        ticks = requests.get(url_ticks, params={"market": market, "count": 500}).json()

        buy_volume = 0.0
        sell_volume = 0.0

        for tick in ticks:
            if tick['ask_bid'] == 'ASK':
                buy_volume += tick['trade_volume']
            elif tick['ask_bid'] == 'BID':
                sell_volume += tick['trade_volume']

        # 체결강도 계산
        volume_power = 999.99 if sell_volume == 0 else (buy_volume / sell_volume) * 100

        # 조건 2: 체결강도가 80% 이상인 경우 결과에 추가
        if volume_power >= target_power:
            result_coins.append({
                "마켓": market,
                "현재가": current_price,
                "1틱단위": tick_size,
                "1틱변동률(%)": round(tick_percent, 3),
                "체결강도(%)": round(volume_power, 2)
            })
            print(f"[검색됨] {market} | 1틱: {tick_percent:.3f}% | 체결강도: {volume_power:.2f}%")

    # 결과 출력
    print("\n=== 최종 결과 (거래량 상위 중 1틱 0.1% 이상 & 체결강도 80% 이상) ===")
    if result_coins:
        df_result = pd.DataFrame(result_coins)
        # 1틱 변동률이 높은 순으로 정렬
        df_result = df_result.sort_values(by='1틱변동률(%)', ascending=False).reset_index(drop=True)
        print(df_result.to_string())
    else:
        print("조건을 모두 만족하는 코인이 현재 없습니다.")


if __name__ == "__main__":
    # 거래량 상위 50개 코인 중에서 검색
    get_filtered_coins(top_n=80, target_power=80.0, target_tick_percent=0.05)
