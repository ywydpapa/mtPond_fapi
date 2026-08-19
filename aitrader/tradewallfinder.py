import asyncio
import websockets
import json
import requests
import pandas as pd
from datetime import datetime
import time

# 모니터링할 코인 개수 (거래대금 상위 N개)
TOP_N_COINS = 20
# 스냅샷 주기 (초)
SNAPSHOT_INTERVAL = 60

# 메모리에 데이터를 누적할 딕셔너리
# 구조: { "KRW-BTC": [{"time": "07:30", "ask_size": 100, "bid_size": 50, "price": 80000000}, ...], ... }
memory_db = {}


def get_top_coins(n=20):
    """거래대금 상위 N개 코인 목록을 가져옵니다."""
    print("모니터링할 거래대금 상위 코인을 검색합니다...")
    url = "https://api.upbit.com/v1/market/all"
    markets = [m['market'] for m in requests.get(url).json() if m['market'].startswith('KRW-')]

    url_ticker = "https://api.upbit.com/v1/ticker"
    tickers = requests.get(url_ticker, params={"markets": ",".join(markets)}).json()

    df = pd.DataFrame(tickers).sort_values(by='acc_trade_price_24h', ascending=False)
    top_coins = df.head(n)['market'].tolist()
    print(f"대상 코인 ({n}개): {top_coins}\n")
    return top_coins


async def connect_websocket(target_coins):
    """업비트 웹소켓에 연결하여 실시간 호가창 데이터를 수신합니다."""
    uri = "wss://api.upbit.com/websocket/v1"

    # 웹소켓 구독 요청 데이터 포맷
    subscribe_data = [
        {"ticket": "wall_monitor_daemon"},
        {"type": "orderbook", "codes": target_coins, "isOnlySnapshot": False}
    ]

    async with websockets.connect(uri, ping_interval=60) as websocket:
        await websocket.send(json.dumps(subscribe_data))
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🟢 웹소켓 연결 성공 및 데이터 수집 시작 (9시 종료 예정)")

        # 마지막 스냅샷 시간을 기록
        last_snapshot_time = time.time()

        # 최신 호가창 상태를 임시 보관할 딕셔너리
        current_state = {coin: None for coin in target_coins}

        while True:
            now = datetime.now()

            # [종료 조건] 오전 9시 00분 00초가 되면 루프 탈출
            if now.hour == 9 and now.minute == 0:
                print(f"\n[{now.strftime('%H:%M:%S')}] 🛑 09:00 도달. 데이터 수집을 종료하고 분석을 시작합니다.")
                break

            try:
                # 데이터 수신 (0.1초 대기, 없으면 패스)
                data = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                orderbook = json.loads(data)

                code = orderbook['code']
                # 현재 상태 업데이트 (총 매도잔량, 총 매수잔량, 최우선 매도호가)
                current_state[code] = {
                    "ask_size": orderbook['total_ask_size'],  # 총 매도벽
                    "bid_size": orderbook['total_bid_size'],  # 총 매수벽
                    "price": orderbook['orderbook_units'][0]['ask_price']  # 현재가(최우선 매도호가)
                }

            except asyncio.TimeoutError:
                pass  # 수신된 데이터가 없으면 다음 루프로
            except Exception as e:
                print(f"웹소켓 에러 발생: {e}")
                break

            # [스냅샷 저장] 지정된 주기(예: 60초)마다 메모리에 기록
            current_time = time.time()
            if current_time - last_snapshot_time >= SNAPSHOT_INTERVAL:
                time_str = now.strftime('%H:%M:%S')

                for coin, state in current_state.items():
                    if state is not None:
                        if coin not in memory_db:
                            memory_db[coin] = []

                        memory_db[coin].append({
                            "time": time_str,
                            "price": state["price"],
                            "ask_wall": state["ask_size"],
                            "bid_wall": state["bid_size"]
                        })

                last_snapshot_time = current_time
                print(f"[{time_str}] 📸 호가창 스냅샷 메모리 저장 완료 (대상: {len(target_coins)}개 종목)")


def analyze_and_save_data():
    """수집된 데이터를 분석하고 CSV로 저장합니다."""
    print("\n=== 📊 수집 데이터 분석 ===")
    results = []

    for coin, snapshots in memory_db.items():
        if not snapshots: continue

        df = pd.DataFrame(snapshots)

        # 6시~9시 사이의 평균 매도벽, 매수벽 계산
        avg_ask_wall = df['ask_wall'].mean()
        avg_bid_wall = df['bid_wall'].mean()

        # 매도벽이 매수벽보다 얼마나 두꺼웠는지 비율 계산 (1.0 이상이면 매도벽이 더 두꺼움)
        wall_ratio = avg_ask_wall / avg_bid_wall if avg_bid_wall > 0 else 0

        # 가격 변동폭 계산
        max_price = df['price'].max()
        min_price = df['price'].min()
        volatility = ((max_price - min_price) / min_price) * 100

        results.append({
            "Coin": coin,
            "Avg_Ask_Wall(매도벽)": round(avg_ask_wall, 2),
            "Avg_Bid_Wall(매수벽)": round(avg_bid_wall, 2),
            "Wall_Ratio(매도/매수)": round(wall_ratio, 2),
            "Volatility(변동폭%)": round(volatility, 2)
        })

    df_result = pd.DataFrame(results)

    # 1. 매도벽이 매수벽보다 압도적으로 두껍고 (비율 2.0 이상)
    # 2. 가격 변동폭이 2% 이내로 억제된 코인 필터링 (누군가 누르고 있었던 코인)
    df_filtered = df_result[(df_result['Wall_Ratio(매도/매수)'] >= 2.0) & (df_result['Volatility(변동폭%)'] <= 2.0)]
    df_filtered = df_filtered.sort_values(by='Wall_Ratio(매도/매수)', ascending=False)

    print("\n[🔥 9시 펌핑 유력 후보 (매도벽으로 가격을 억누른 코인)]")
    if not df_filtered.empty:
        print(df_filtered.to_string(index=False))
    else:
        print("조건을 만족하는 뚜렷한 매집/억제 패턴이 발견되지 않았습니다.")

    # 전체 데이터 CSV 저장
    filename = f"orderbook_walls_{datetime.now().strftime('%Y%m%d')}.csv"
    df_result.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\n💾 전체 분석 결과가 '{filename}' 파일로 저장되었습니다.")


async def main():
    target_coins = get_top_coins(TOP_N_COINS)
    await connect_websocket(target_coins)
    analyze_and_save_data()


if __name__ == "__main__":
    # 비동기 이벤트 루프 실행
    asyncio.run(main())
