import asyncio
import websockets
import requests
import json
from collections import defaultdict, deque
from rich.console import Console
from rich.table import Table

def get_krw_markets():
    url = "https://api.upbit.com/v1/market/all"
    res = requests.get(url)
    data = res.json()
    return [m['market'] for m in data if m['market'].startswith('KRW-')]

async def upbit_trade_monitor(markets, window_size=10):
    uri = "wss://api.upbit.com/websocket/v1"
    subscribe_fmt = [
        {"ticket": "test"},
        {"type": "trade", "codes": markets}
    ]
    trade_queues = defaultdict(lambda: {'BID': deque(maxlen=window_size),
                                        'ASK': deque(maxlen=window_size)})
    console = Console()
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps(subscribe_fmt))
        print("Subscribed to trade streams...")

        while True:
            data = await websocket.recv()
            trade = json.loads(data)
            market = trade['code']
            side = trade['ask_bid']  # 'BID' or 'ASK'
            volume = float(trade['trade_volume'])
            price = float(trade['trade_price'])
            amount = volume * price

            trade_queues[market][side].append({'volume': volume, 'amount': amount})

            # 표를 만들기 위해 모든 마켓의 상태를 수집
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Market", justify="left")
            table.add_column("Side", justify="center")
            table.add_column("Sum Vol(10)", justify="right")
            table.add_column("Sum Amount(10)", justify="right")

            for m in markets:
                for s in ['BID', 'ASK']:
                    sum_vol = sum(x['volume'] for x in trade_queues[m][s])
                    sum_amt = sum(x['amount'] for x in trade_queues[m][s])
                    table.add_row(m, s, f"{sum_vol:.4f}", f"{sum_amt:,.0f}")

            console.clear()
            console.print(table)

async def main():
    markets = ["KRW-BTC", "KRW-ETH", "KRW-XRP"]  # 원하는 마켓만 선택
    await upbit_trade_monitor(markets)

if __name__ == "__main__":
    asyncio.run(main())
