# ==========================================
# 1. Python 표준 라이브러리
# ==========================================
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import json
import math
import os
import random
import time
from typing import Dict, List, Optional

# ==========================================
# 2. 서드파티 라이브러리
# ==========================================
import aiohttp
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import dotenv
import httpx
import jinja2
from pandas import DataFrame
import pyupbit
import redis.asyncio as aioredis  # Redis 비동기 모듈
import requests
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from starlette.middleware.sessions import SessionMiddleware
import websockets

# ==========================================
# 3. FastAPI 모듈
# ==========================================
from fastapi import (
    Body,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ==========================================
# 4. 환경 변수 및 DB/Redis 설정
# ==========================================
dotenv.load_dotenv()
DATABASE_URL = os.getenv("dburl")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


# ★ Redis 원격 연결 (특수문자 에러 방지용 개별 변수 우선 파싱)
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", "56379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

if REDIS_HOST:
    redis_client = aioredis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        db=REDIS_DB,
        decode_responses=True,
        socket_timeout=5.0,
        socket_connect_timeout=5.0,
        retry_on_timeout=True
    )
else:
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_client = aioredis.from_url(
        REDIS_URL,
        decode_responses=True,
        socket_timeout=5.0,
        socket_connect_timeout=5.0,
        retry_on_timeout=True
    )


# ==========================================
# 4b. Redis 동기화 헬퍼 함수
# ==========================================
async def sync_mtsetup_to_redis(user_no: int, db: AsyncSession):
    """DB에서 최신 mtSetup을 읽어와 Redis 캐시를 갱신하고 Pub/Sub 채널로 알림 전송"""
    try:
        sql = text("""
                   SELECT activeYN,
                          initAmt,
                          addAmt,
                          limitAmt,
                          minMargin,
                          maxMargin,
                          tickRate,
                          tickYN,
                          lcRate,
                          lcGap,
                          maxCoincnt,
                          martinYN,
                          stopYN,
                          stopAutoYN
                   FROM mtSetup
                   WHERE userNo = :userno
                     AND attrib NOT LIKE :attrib LIMIT 1
                   """)
        result = await db.execute(sql, {"userno": user_no, "attrib": "%XXX%"})
        row = result.fetchone()
        if row:
            setup_dict = dict(row._mapping)
            redis_key = f"mtpond:setup:{user_no}"
            channel_name = f"mtpond:config_channel:{user_no}"

            # 1. Redis 캐시 갱신 (TTL: 1시간)
            await redis_client.set(redis_key, json.dumps(setup_dict), ex=3600)
            # 2. 봇 클라이언트들에게 실시간 이벤트 브로드캐스트
            await redis_client.publish(channel_name, "reload")
            print(f"[REDIS] mtSetup 동기화 완료 (userNo: {user_no})")
    except Exception as e:
        print(f"[REDIS][WARN] mtSetup 동기화 실패 (userNo: {user_no}): {e}")


async def sync_excoins_to_redis(user_no: int, db: AsyncSession):
    """DB에서 최신 제외 코인 목록을 읽어와 Redis Set 캐시를 갱신"""
    try:
        sql = text("SELECT DISTINCT market FROM exCoinlist WHERE userNo IN (0, :userno) AND attrib NOT LIKE :attrib")
        result = await db.execute(sql, {"userno": user_no, "attrib": "%XXX%"})
        rows = result.fetchall()
        ex_markets = [r[0] for r in rows if r and r[0]]

        redis_key = f"mtpond:excoins:{user_no}"
        # 기존 Set 삭제 후 재등록
        await redis_client.delete(redis_key)
        if ex_markets:
            await redis_client.sadd(redis_key, *ex_markets)
            await redis_client.expire(redis_key, 3600)

        # 봇들에게 알림 전송
        channel_name = f"mtpond:config_channel:{user_no}"
        await redis_client.publish(channel_name, "reload_excoins")
        print(f"[REDIS] exCoins 동기화 완료: {ex_markets} (userNo: {user_no})")
    except Exception as e:
        print(f"[REDIS][WARN] exCoins 동기화 실패 (userNo: {user_no}): {e}")


# ==========================================
# 5. 스케줄러 & Lifespan 정의
# ==========================================
scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 6시간 간격 스케줄러 등록
    scheduler.add_job(
        job_collect_wallet_balances,
        trigger=IntervalTrigger(hours=6),
        id="wallet_balance_collector",
        name="6시간마다 지갑 잔고 기록",
        replace_existing=True,
    )
    scheduler.start()
    print("스케줄러 시작됨 (6시간 주기 실행)")

    # 서버 시작 직후 1회 백그라운드 수집 실행
    asyncio.create_task(job_collect_wallet_balances())

    yield

    scheduler.shutdown()
    await redis_client.aclose()
    print("스케줄러 및 Redis 연결 종료됨")


app = FastAPI(lifespan=lifespan)

app.add_middleware(SessionMiddleware, secret_key="supersecretkey")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


def format_currency(value):
    if isinstance(value, (int, float)):
        return "{:,.0f}".format(value)
    return value


templates.env.filters['currency'] = format_currency


# ==========================================
# 6. 비즈니스 로직 및 헬퍼 함수
# ==========================================
def require_login(request: Request):
    user_no = request.session.get("user_No")
    if not user_no:
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": "/"},
            detail="세션이 만료되어 재로그인이 필요합니다."
        )
    return user_no


async def get_current_prices():
    server_url = "https://api.upbit.com"
    params = {"quote_currencies": "KRW"}
    async with httpx.AsyncClient() as client:
        res = await client.get(f"{server_url}/v1/ticker/all", params=params)
        data = res.json()
        result = []
        for item in data:
            market = item.get("market")
            trade_price = item.get("trade_price")
            if market and trade_price:
                result.append({"market": market, "trade_price": trade_price})
        return result


async def get_current_price(coink):
    url = "https://api.upbit.com/v1/ticker"
    params = {"markets": coink}
    async with httpx.AsyncClient() as client:
        res = await client.get(url, params=params)
        data = res.json()
        if data and isinstance(data, list):
            return data[0].get("trade_price")
        return None


async def selectUsers(uid: str, upw: str, db: AsyncSession):
    try:
        sql = text(
            "SELECT userNo, userName, serverNo, userRole, tradeCnt FROM traceUser WHERE userPasswd=password(:passwd) AND userId=:userid AND attrib NOT LIKE :xattr")
        result = await db.execute(sql, {"passwd": upw, "userid": uid, "xattr": "%XXXUP%"})
        row = result.fetchone()
        setkey = random.randint(100000, 999999) if row is not None else None
        return row, setkey
    except Exception as e:
        print('사용자 인증 오류:', e)
        return None, None


async def listUsers(db: AsyncSession):
    try:
        sql = text("SELECT * FROM traceUser WHERE attrib NOT LIKE :xattr")
        result = await db.execute(sql, {"xattr": "%XXXUP%"})
        return result.mappings().all()
    except Exception as e:
        print('사용자 목록 조회 오류:', e)
        return []


async def get_hotcoins(request, db: AsyncSession):
    try:
        query = text("SELECT * FROM orderbookAmt where dateTag = (select max(dateTag) from orderbookAmt)")
        result = await db.execute(query)
        return result.fetchall()
    except Exception as e:
        print("Hotcoins 조회 오류:", e)
        return []


async def get_hotamt(request, db: AsyncSession):
    try:
        query = text("select * from tradeAmt order by regDate desc limit 1")
        result = await db.execute(query)
        return result.fetchone()
    except Exception as e:
        print("Hotamt 조회 오류:", e)
        return None


async def detailuser(uno: int, db: AsyncSession):
    try:
        sql = text("SELECT * FROM traceUser WHERE userNo = :userno and attrib NOT LIKE :xattr")
        result = await db.execute(sql, {"userno": uno, "xattr": "%XXXUP%"})
        return result.fetchone()
    except Exception as e:
        print('사용자 세부정보 조회 오류:', e)
        return None


async def get_onoff(uno: int, db: AsyncSession):
    try:
        sql = text("SELECT activeYN FROM mtSetup WHERE userNo = :userno and attrib NOT LIKE :xattr")
        result = await db.execute(sql, {"userno": uno, "xattr": "%XXX%"})
        return result.fetchone()
    except Exception as e:
        print('ON/OFF 상태 조회 오류:', e)
        return None


async def setKeys(uno: int, setkey: str, db: AsyncSession):
    try:
        sql = text("UPDATE traceUser SET setupKey = :setk, lastLogin = now() where userNo=:userno")
        await db.execute(sql, {"userno": uno, "setk": setkey})
        await db.commit()
        return True
    except Exception as e:
        print('setupKey 저장 오류:', e)
        return False


async def check_setkey(uno: int, setkey: str, db: AsyncSession):
    try:
        sql = text("SELECT userNo FROM traceUser WHERE userNo=:userno and setupKey=:setkey and attrib not like :xattr")
        result = await db.execute(sql, {"userno": uno, "setkey": setkey, "xattr": "%XXXUP%"})
        row = result.fetchone()
        return bool(row and row[0] == uno)
    except Exception as e:
        print("코드 체크 에러:", e)
        return False


async def get_userdetail(uno: int, setkey: str, db: AsyncSession):
    try:
        sql = text("SELECT * FROM traceUser WHERE userNo=:userno and setupKey=:setkey and attrib not like :xattr")
        result = await db.execute(sql, {"userno": uno, "setkey": setkey, "xattr": "%XXXUP%"})
        row = result.fetchone()
        return row if (row and row[0] == uno) else False
    except Exception as e:
        print("유저정보 취득 에러:", e)
        return False


async def getKeys(uno: int, setkey: str, db: AsyncSession):
    try:
        sql = text(
            "SELECT apiKey1, apiKey2 FROM traceUser WHERE setupKey=:setk AND userNo=:userno and attrib not like :xattr")
        result = await db.execute(sql, {"setk": setkey, "userno": uno, "xattr": "%XXXUP%"})
        keys = result.fetchone()
        if keys:
            return keys[0], keys[1]
        return None, None
    except Exception as e:
        print('키 로드 오류:', e)
        return None, None


async def api_getKeys(uno: int, db: AsyncSession):
    try:
        sql = text("SELECT apiKey1, apiKey2 FROM traceUser WHERE userNo=:userno and attrib not like :xattr")
        result = await db.execute(sql, {"userno": uno, "xattr": "%XXXUP%"})
        keys = result.fetchone()
        if keys:
            return keys[0], keys[1]
        return None, None
    except Exception as e:
        print('키 로드 오류:', e)
        return None, None


async def checkwallet(uno: int, setkey: str, db: AsyncSession):
    try:
        key1, key2 = await getKeys(uno, setkey, db)
        if not key1 or not key2:
            return []
        upbit = pyupbit.Upbit(key1, key2)
        return await asyncio.to_thread(upbit.get_balances)
    except Exception as e:
        print("지갑 불러오기 에러:", e)
        return []


async def api_checkwallet(uno: int, db: AsyncSession):
    try:
        key1, key2 = await api_getKeys(uno, db)
        if not key1 or not key2:
            return []
        upbit = pyupbit.Upbit(key1, key2)
        return await asyncio.to_thread(upbit.get_balances)
    except Exception as e:
        print(f"지갑 불러오기 에러 (uno: {uno}):", e)
        return []


async def clearcache(db: AsyncSession):
    try:
        await db.execute(text("RESET QUERY CACHE"))
        await db.commit()
        return True
    except Exception as e:
        print("Clear Cache Error:", e)
        return False


async def buycoinmarket(uno: int, coink: str, setkey: str, amt: float, db: AsyncSession):
    try:
        key1, key2 = await getKeys(uno, setkey, db)
        upbit = pyupbit.Upbit(key1, key2)
        return await asyncio.to_thread(upbit.buy_market_order, coink, amt)
    except Exception as e:
        print("시장가 구매 에러:", e)
        return False


async def api_buycoinmarket(uno: int, coink: str, amt: float, db: AsyncSession):
    try:
        key1, key2 = await api_getKeys(uno, db)
        upbit = pyupbit.Upbit(key1, key2)
        return await asyncio.to_thread(upbit.buy_market_order, coink, amt)
    except Exception as e:
        print("시장가 구매 에러:", e)
        return False


async def sellcoinpercent(uno: int, coink: str, setkey: str, volm: float, db: AsyncSession):
    try:
        key1, key2 = await getKeys(uno, setkey, db)
        upbit = pyupbit.Upbit(key1, key2)
        walt = await asyncio.to_thread(upbit.get_balances)
        crp = await asyncio.to_thread(pyupbit.get_current_price, coink)
        currency_target = coink.split('-')[1]
        for coin in walt:
            if coin.get('currency') == currency_target:
                if float(coin.get('balance', 0)) * float(crp) < 5000:
                    return await asyncio.to_thread(upbit.buy_market_order, coink, 5000)
                else:
                    return await asyncio.to_thread(upbit.sell_market_order, coink, volm)
        return False
    except Exception as e:
        print("시장가 매도 에러:", e)
        return False


async def api_sellcoinpercent(uno: int, coink: str, volm: float, db: AsyncSession):
    try:
        key1, key2 = await api_getKeys(uno, db)
        upbit = pyupbit.Upbit(key1, key2)
        walt = await asyncio.to_thread(upbit.get_balances)
        crp = await asyncio.to_thread(pyupbit.get_current_price, coink)
        currency_target = coink.split('-')[1]
        for coin in walt:
            if coin.get('currency') == currency_target:
                if float(coin.get('balance', 0)) * float(crp) < 5000:
                    return await asyncio.to_thread(upbit.buy_market_order, coink, 5000)
                else:
                    return await asyncio.to_thread(upbit.sell_market_order, coink, volm)
        return False
    except Exception as e:
        print("시장가 매도 에러:", e)
        return False


async def cancelorder(uno: int, setkey: str, uuid: str, db: AsyncSession):
    try:
        key1, key2 = await getKeys(uno, setkey, db)
        upbit = pyupbit.Upbit(key1, key2)
        return await asyncio.to_thread(upbit.cancel_order, uuid)
    except Exception as e:
        print("거래 취소 에러:", e)
        return False


async def api_cancelorder(uno: int, uuid: str, db: AsyncSession):
    try:
        key1, key2 = await api_getKeys(uno, db)
        upbit = pyupbit.Upbit(key1, key2)
        return await asyncio.to_thread(upbit.cancel_order, uuid)
    except Exception as e:
        print("거래 취소 에러:", e)
        return False


async def tradedcoins(uno: int, db: AsyncSession):
    try:
        sql = text(
            "select distinct bidCoin from traceSetup where userNo=:userno and attrib not like :xattr order by bidCoin asc")
        rows = await db.execute(sql, {"userno": uno, "xattr": "%XXXUP%"})
        return [list(r) for r in rows.fetchall()]
    except Exception as e:
        print("거래 코인 목록 조회 에러:", e)
        return []


async def get_tradelogupbit(coink: str, userno: int, setkey: str, db: AsyncSession):
    try:
        key1, key2 = await getKeys(userno, setkey, db)
        upbit = pyupbit.Upbit(key1, key2)
        return await asyncio.to_thread(upbit.get_order, coink, state="done")
    except Exception as e:
        print("거래이력 불러오기 에러:", e)
        return []


async def get_orderlist(userno: int, setkey: str, slot: int, db: AsyncSession):
    try:
        key1, key2 = await getKeys(userno, setkey, db)
        upbit = pyupbit.Upbit(key1, key2)
        setups = await getsetups(userno, slot, db)
        orders = []
        for setup in setups:
            coink = setup[6]
            order = await asyncio.to_thread(upbit.get_order, coink)
            if order and isinstance(order, list):
                orders.extend(order)
        return orders
    except Exception as e:
        print("주문내용 불러오기 에러:", e)
        return []


async def get_mtorderlist(userno: int, setkey: str, db: AsyncSession):
    try:
        key1, key2 = await getKeys(userno, setkey, db)
        upbit = pyupbit.Upbit(key1, key2)
        setups = await checkwallet(userno, setkey, db)
        orders = []
        for setup in setups:
            if setup.get("currency") != "KRW":
                coink = "KRW-" + setup["currency"]
                order = await asyncio.to_thread(upbit.get_order, coink)
                if order and isinstance(order, list):
                    orders.extend(order)
        return orders
    except Exception as e:
        print("mt주문내용 불러오기 에러:", e)
        return []


async def api_mtorderlist(userno: int, db: AsyncSession):
    try:
        key1, key2 = await api_getKeys(userno, db)
        upbit = pyupbit.Upbit(key1, key2)
        setups = await api_checkwallet(userno, db)
        orders = []
        for setup in setups:
            if setup.get("currency") != "KRW":
                coink = "KRW-" + setup["currency"]
                order = await asyncio.to_thread(upbit.get_order, coink)
                if order and isinstance(order, list):
                    orders.extend(order)
        return orders
    except Exception as e:
        print("mt주문내용 불러오기 에러:", e)
        return []


async def getsetups(uno: int, slotno: int, db: AsyncSession):
    try:
        if slotno == 0:
            sql = text("select * from traceSetup where userNo=:userno and attrib not like :xatts")
            rows = await db.execute(sql, {"userno": uno, "xatts": '%XXXUP%'})
        else:
            sql = text("select * from traceSetup where userNo=:userno and slot = :slot and attrib not like :xattr")
            rows = await db.execute(sql, {"userno": uno, "slot": slotno, "xattr": '%XXXUP%'})
        return list(rows.fetchall())
    except Exception as e:
        print('설정 불러오기 오류:', e)
        return []


async def get_mtsetups(uno: int, db: AsyncSession):
    try:
        sql = text("select * from mtSetup where userNo=:userno and attrib not like :xatts")
        rows = await db.execute(sql, {"userno": uno, "xatts": '%XXXUP%'})
        return list(rows.fetchall())
    except Exception as e:
        print('설정 불러오기 오류:', e)
        return []


async def setonoffs(setno: int, yesno: str, db: AsyncSession):
    try:
        sql = text("UPDATE mtPondSetup SET activeYN = :yesno where setupNo=:setno AND attrib not like :xattr")
        await db.execute(sql, {"setno": setno, "yesno": yesno, "xattr": '%XXXUP%'})
        await db.commit()
    except Exception as e:
        print('거래 ON/OFF 오류:', e)


async def setonoff(uno: int, yesno: str, db: AsyncSession):
    try:
        sql = text("UPDATE mtSetup SET activeYN = :yesno where userNo=:userno AND attrib not like :xattr")
        await db.execute(sql, {"userno": uno, "yesno": yesno, "xattr": '%XXXUP%'})
        await db.commit()
        # ★ DB 갱신 후 Redis 캐시 동기화 및 PubSub 알림
        await sync_mtsetup_to_redis(uno, db)
    except Exception as e:
        print('거래 ON/OFF 오류:', e)


async def get_trsetups(uno: int, db: AsyncSession):
    try:
        query = text("SELECT * FROM polarisSets where userNo = :uno and attrib not like :attxx")
        result = await db.execute(query, {"uno": uno, "attxx": "%XXX%"})
        mysetups = result.fetchall()
        return [
            {
                "setupNo": setup[0],
                "coinName": setup[2],
                "stepAmt": setup[3],
                "tradeType": setup[4],
                "maxAmt": setup[5],
                "useYN": setup[6],
            }
            for setup in mysetups
        ]
    except Exception as e:
        print("Get Setup Error:", e)
        return []


async def setautostop(sno: int, yesno: str, db: AsyncSession):
    try:
        sql = text("UPDATE traceSetup SET doubleYN = :yesno where setupNo=:sno")
        await db.execute(sql, {"sno": sno, "yesno": yesno})
        await db.commit()
    except Exception as e:
        print('자동 멈춤 설정 오류:', e)


async def setlconoff(setno: int, lcrate: float, yesno: str, db: AsyncSession):
    try:
        sql = text("UPDATE mtPondSetup SET losscut = :lcrate, lcYN = :yesno where setupNo=:setno")
        await db.execute(sql, {"lcrate": lcrate, "yesno": yesno, "setno": setno})
        await db.commit()
    except Exception as e:
        print('손절 설정 오류:', e)


async def selectsetlist(db: AsyncSession):
    try:
        sql = text("SELECT * FROM traceSets WHERE useYN = :useyn and attrib NOT LIKE :xattr")
        rows = await db.execute(sql, {"useyn": "Y", "xattr": "%XXXUP%"})
        return rows.fetchall()
    except Exception as e:
        print('트레이딩 설정 목록 오류:', e)
        return []


async def erasebid(uno: int, setkey: str, tabindex: int, db: AsyncSession):
    try:
        sql = text("update traceSetup set attrib=:xattr where userNo=:userno and slot = :slot")
        await db.execute(sql, {"xattr": "XXXUPXXXUPXXXUP", "userno": uno, "slot": tabindex})
        await db.commit()
        return True
    except Exception as e:
        return False


async def erasemtpondsetup(uno: int, setkey: str, db: AsyncSession):
    try:
        sql = text("update mtSetup set attrib=:xattr where userNo=:userno")
        await db.execute(sql, {"xattr": "XXXUPXXXUP", "userno": uno})
        await db.commit()
        return True
    except Exception as e:
        return False


async def update_userdtl(uno: int, key1: str, key2: str, svrno: int, db: AsyncSession):
    try:
        sql = text("update traceUser set apiKey1 = :key1, apiKey2 = :key2, serverNo = :svrno where userNo=:userno")
        await db.execute(sql, {"key1": key1, "key2": key2, "svrno": svrno, "userno": uno})
        await db.commit()
        return True
    except Exception as e:
        return False


async def setupbid(uno, setkey, initbid, bidstep, bidrate, askrate, coinn, svrno, tradeset, holdNo, doubleYN, limitamt,
                   limityn, slot, db: AsyncSession):
    if await check_setkey(uno, setkey, db):
        try:
            sql = text("""
                       insert into traceSetup
                       (userNo, initAsset, bidInterval, bidRate, askrate, bidCoin, custKey, serverNo, holdNo, doubleYN,
                        limitAmt, limitYN, slot, regDate)
                       VALUES (:uno, :initbid, :bidstep, :bidrate, :askrate, :coinn, :tradeset, :svrno, :holdNo,
                               :doubleYN, :limitamt, :limityn, :slot, now())
                       """)
            await db.execute(sql, {
                "uno": uno, "initbid": initbid, "bidstep": bidstep,
                "bidrate": bidrate, "askrate": askrate, "coinn": coinn,
                "tradeset": tradeset, "svrno": svrno, "holdNo": holdNo,
                "doubleYN": doubleYN, "limitamt": limitamt, "limityn": limityn, "slot": slot,
            })
            await db.commit()
            return True
        except Exception as e:
            print('트레이딩 설정 저장 오류:', e)
    return False


async def setupmymtpondset(uno, setkey, initbid, addbid, limitbid, minmargin, cutrate, db: AsyncSession):
    if await check_setkey(uno, setkey, db):
        try:
            sql = text("""
                       insert into mtSetup (userNo, initAmt, addAmt, limitAmt, minMargin, lcRate)
                       VALUES (:userNo, :iniBid, :addBid, :limitBid, :minMargin, :losscut)
                       """)
            await db.execute(sql, {
                "userNo": uno, "iniBid": initbid, "addBid": addbid, "limitBid": limitbid, "minMargin": minmargin,
                "losscut": cutrate
            })
            await db.commit()
            # ★ DB 저장 성공 시 Redis 동기화 및 봇 알림
            await sync_mtsetup_to_redis(uno, db)
            return True
        except Exception as e:
            print('mtPond 트레이딩 설정 저장 오류:', e)
    return False


async def editbidsetup(sno, uno, setkey, initbid, bidstep, bidrate, askrate, coinn, svrno, tradeset, holdNo, doubleYN,
                       limitamt, limityn, slot, db: AsyncSession):
    if await check_setkey(uno, setkey, db):
        try:
            await db.execute(text("update traceSetup set attrib=:xattr where setupNo=:sno"),
                             {"sno": sno, "xattr": "XXXUPXXXUPXXXUP"})
            sql = text("""
                       insert into traceSetup
                       (userNo, initAsset, bidInterval, bidRate, askrate, bidCoin, custKey, serverNo, holdNo, doubleYN,
                        limitAmt, limitYN, slot, regDate)
                       VALUES (:uno, :initbid, :bidstep, :bidrate, :askrate, :coinn, :tradeset, :svrno, :holdNo,
                               :doubleYN, :limitamt, :limityn, :slot, now())
                       """)
            await db.execute(sql, {
                "uno": uno, "initbid": initbid, "bidstep": bidstep,
                "bidrate": bidrate, "askrate": askrate, "coinn": coinn,
                "tradeset": tradeset, "svrno": svrno, "holdNo": holdNo,
                "doubleYN": doubleYN, "limitamt": limitamt, "limityn": limityn, "slot": slot,
            })
            await db.commit()
            return True
        except Exception as e:
            print('설정 수정 오류:', e)
    return False


# ==========================================
# 7. 주기적 백그라운드 태스크
# ==========================================
async def job_collect_wallet_balances():
    now = datetime.now()
    print(f"[{now}] === 지갑 잔고 스케줄러 수집 시작 ===")

    async with AsyncSessionLocal() as db:
        users = await listUsers(db)
        if not users:
            print("수집 대상 사용자가 없습니다.")
            return

        insert_sql = text("""
                          INSERT INTO walletBalance (userNo, timeStamp, balanceKRW, totalKRW, attrib)
                          VALUES (:userNo, :timeStamp, :balanceKRW, :totalKRW, :attrib)
                          """)

        for user in users:
            user_no = user.get("userNo")
            if not user_no:
                continue

            balances = await api_checkwallet(user_no, db)
            if not balances or not isinstance(balances, list):
                continue

            balance_krw = 0.0
            coin_buy_total = 0.0

            for item in balances:
                currency = item.get("currency", "")
                balance = float(item.get("balance", 0.0))
                locked = float(item.get("locked", 0.0))
                avg_buy_price = float(item.get("avg_buy_price", 0.0))

                if currency == "KRW":
                    balance_krw += (balance + locked)
                else:
                    coin_buy_total += (balance + locked) * avg_buy_price

            total_krw = balance_krw + coin_buy_total

            try:
                await db.execute(
                    insert_sql,
                    {
                        "userNo": user_no,
                        "timeStamp": now,
                        "balanceKRW": round(balance_krw, 2),
                        "totalKRW": round(total_krw, 2),
                        "attrib": "1000010000",
                    },
                )
                await db.commit()
                print(f"[userNo: {user_no}] 저장 완료 - 원화: {balance_krw:,.0f}원 | 총 매수금액: {total_krw:,.0f}원")
            except Exception as e:
                await db.rollback()
                print(f"[userNo: {user_no}] DB 저장 실패:", e)

            await asyncio.sleep(0.2)

    print(f"[{datetime.now()}] === 지갑 잔고 스케줄러 수집 완료 ===")


# ==========================================
# 8. WebSockets
# ==========================================
async def upbit_ws_price_stream(markets: list):
    uri = "wss://api.upbit.com/websocket/v1"
    subscribe_data = [{"ticket": "test"}, {"type": "ticker", "codes": markets, "isOnlyRealtime": True}]
    async with websockets.connect(uri, ping_interval=60) as websocket:
        await websocket.send(json.dumps(subscribe_data))
        while True:
            data = await websocket.recv()
            parsed = json.loads(data)
            yield parsed.get('code'), parsed.get('trade_price'), parsed.get('change')


@app.websocket("/ws/coinprice")
async def coin_price_ws(websocket: WebSocket):
    await websocket.accept()
    coins = websocket.query_params.get("coins", "")
    coin_list = coins.split(",") if coins else []
    try:
        async for market, current_price, change in upbit_ws_price_stream(coin_list):
            await websocket.send_json({"market": market, "current_price": current_price, "change": change})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print("WebSocket Price Error:", e)


async def upbit_ws_orderbook_stream(markets: list):
    uri = "wss://api.upbit.com/websocket/v1"
    subscribe_data = [{"ticket": "test"}, {"type": "orderbook", "codes": markets, "isOnlyRealtime": True}]
    async with websockets.connect(uri, ping_interval=60) as websocket:
        await websocket.send(json.dumps(subscribe_data))
        while True:
            data = await websocket.recv()
            parsed = json.loads(data)
            if parsed.get("type") == "orderbook":
                yield {
                    "market": parsed.get("code"),
                    "orderbook_units": parsed.get("orderbook_units")
                }


@app.websocket("/ws/orderbook")
async def coin_orderbook_ws(websocket: WebSocket):
    await websocket.accept()
    coins = websocket.query_params.get("coins", "")
    coin_list = coins.split(",") if coins else []
    try:
        async for ob_data in upbit_ws_orderbook_stream(coin_list):
            await websocket.send_json(ob_data)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print("WebSocket Orderbook Error:", e)


async def upbit_ws_trade_stream(markets: list):
    uri = "wss://api.upbit.com/websocket/v1"
    subscribe_data = [{"ticket": "test"}, {"type": "trade", "codes": markets, "isOnlyRealtime": True}]
    async with websockets.connect(uri, ping_interval=60) as websocket:
        await websocket.send(json.dumps(subscribe_data))
        while True:
            data = await websocket.recv()
            parsed = json.loads(data)
            if parsed.get("type") == "trade":
                yield {
                    "market": parsed.get("code"),
                    "trade_price": parsed.get("trade_price"),
                    "trade_volume": parsed.get("trade_volume"),
                    "ask_bid": parsed.get("ask_bid"),
                    "trade_time": parsed.get("trade_time"),
                    "trade_timestamp": parsed.get("trade_timestamp")
                }


@app.websocket("/ws/trade")
async def coin_trade_ws(websocket: WebSocket):
    await websocket.accept()
    coins = websocket.query_params.get("coins", "")
    coin_list = coins.split(",") if coins else []
    try:
        async for trade_data in upbit_ws_trade_stream(coin_list):
            await websocket.send_json(trade_data)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print("WebSocket Trade Error:", e)


# ==========================================
# 9. 라우트 핸들러 (Routes)
# ==========================================
@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("/login/login.html", {"request": request})


@app.post("/loginchk")
async def login(
        request: Request,
        uid: str = Form(...),
        upw: str = Form(...),
        db: AsyncSession = Depends(get_db)
):
    user_row, setkey = await selectUsers(uid, upw, db)
    if not user_row or not setkey:
        return templates.TemplateResponse("login/login.html", {"request": request, "error": "Invalid credentials"})

    await setKeys(user_row[0], setkey, db)
    request.session["user_No"] = user_row[0]
    request.session["user_Name"] = user_row[1]
    request.session["server_No"] = user_row[2]
    request.session["user_Role"] = user_row[3]
    request.session["License"] = user_row[4]
    request.session["setKey"] = setkey
    return RedirectResponse(url=f"/upbittop30/{user_row[0]}/{setkey}", status_code=303)


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/")


@app.get("/hotcoin_list/{uno}")
async def hotcoinlist(request: Request, uno: int, user_session: int = Depends(require_login),
                      db: AsyncSession = Depends(get_db)):
    if uno != user_session:
        return RedirectResponse(url="/", status_code=303)
    usern = request.session.get("user_Name")
    setkey = request.session.get("setKey")
    try:
        orderbooks = await get_hotcoins(request, db)
        hotamt = await get_hotamt(request, db)
        gettime = orderbooks[0][8] if orderbooks else datetime.now()
        diff = datetime.now() - gettime
        days = diff.days
        hours = diff.seconds // 3600
        minutes = (diff.seconds % 3600) // 60
        seconds = diff.seconds % 60
        time_diff = f"{days}일 {hours}시간 {minutes}분 {seconds}초"
        is_reloadable = "Y" if diff.total_seconds() > 10800 else "N"
        trsetups = await get_trsetups(uno, db)
        return templates.TemplateResponse(
            "/trade/hotcoinlist.html",
            {
                "request": request,
                "userNo": uno,
                "user_No": uno,
                "userName": usern,
                "setkey": setkey,
                "orderbooks": orderbooks,
                "time_diff": time_diff,
                "trsetups": trsetups,
                "reloadable": is_reloadable,
                "hotamt": hotamt,
            }
        )
    except Exception as e:
        print("Hotcoin view error:", e)
        return RedirectResponse(url="/", status_code=303)


@app.get("/balance/{uno}")
async def my_balance(request: Request, uno: int, user_session: int = Depends(require_login),
                     db: AsyncSession = Depends(get_db)):
    if uno != user_session:
        return RedirectResponse(url="/", status_code=303)
    setKey = request.session.get("setKey")
    userName = request.session.get("user_Name")
    userRole = request.session.get("user_Role")
    userLicense = request.session.get("License")
    mycoins = await checkwallet(uno, setKey, db)
    cprices = await get_current_prices()
    return templates.TemplateResponse(
        "wallet/mywallet.html",
        {
            "request": request,
            "user_No": uno,
            "user_Name": userName,
            "user_Role": userRole,
            "setkey": setKey,
            "license": userLicense,
            "mycoins": mycoins,
            "myavgp": None,
            "cuprices": cprices
        }
    )


@app.post("/tradebuymarket/{uno}/{setkey}/{coinn}/{costk}")
async def tradebuymarket(request: Request, uno: int, setkey: str, coinn: str, costk: float,
                         user_session: int = Depends(require_login), db: AsyncSession = Depends(get_db)):
    if uno != user_session or str(request.session.get("setKey")) != str(setkey):
        return JSONResponse({"success": False, "message": "권한이 없습니다.", "redirect": "/"})
    coink = "KRW-" + coinn
    butm = await buycoinmarket(uno, coink, setkey, costk, db)
    return JSONResponse({"success": bool(butm), "redirect": f"/balance/{uno}"})


@app.post("/tradesellmarket/{uno}/{setkey}/{coinn}/{volm}")
async def tradesellmarket(request: Request, uno: int, setkey: str, coinn: str, volm: float,
                          user_session: int = Depends(require_login), db: AsyncSession = Depends(get_db)):
    if uno != user_session or str(request.session.get("setKey")) != str(setkey):
        return JSONResponse({"success": False, "message": "권한이 없습니다.", "redirect": "/"})
    coink = "KRW-" + coinn
    sellm = await sellcoinpercent(uno, coink, setkey, volm, db)
    return JSONResponse({"success": bool(sellm), "redirect": f"/balance/{uno}"})


@app.get('/tradedetail/{userno}/{setkey}')
async def tradedetail(request: Request, userno: int, setkey: str, db: AsyncSession = Depends(get_db)):
    coinlist = pyupbit.get_tickers(fiat="KRW")
    trcoins = await tradedcoins(userno, db)
    userName = request.session.get("user_Name")
    userRole = request.session.get("user_Role")
    return templates.TemplateResponse('./trade/mytradingresult.html', {
        "request": request, "coinlist": coinlist, "trcoins": trcoins,
        "user_No": userno, "user_Name": userName, "user_Role": userRole,
        "setkey": setkey, "reqitems": [], "dates": []
    })


@app.get('/tradedetails/{userno}/{setkey}/{coink}')
async def tradedetails(request: Request, userno: int, setkey: str, coink: str, db: AsyncSession = Depends(get_db)):
    coinlist = pyupbit.get_tickers(fiat="KRW")
    trcoins = await tradedcoins(userno, db)
    trlogs = await get_tradelogupbit(coink, userno, setkey, db)
    dates = sorted(list({item['created_at'][:10] for item in trlogs if 'created_at' in item}), reverse=True)
    userName = request.session.get("user_Name")
    userRole = request.session.get("user_Role")
    return templates.TemplateResponse('./trade/mytradingresult.html', {
        "request": request, "coinlist": coinlist, "trcoins": trcoins,
        "user_No": userno, "user_Name": userName, "user_Role": userRole,
        "setkey": setkey, "reqitems": trlogs, "dates": dates, "coink": coink
    })


@app.get('/tradetrend/{userno}/{setkey}')
async def tradetrend(request: Request, userno: int, setkey: str, db: AsyncSession = Depends(get_db)):
    trcoins = await tradedcoins(userno, db)
    mycoins = await checkwallet(userno, setkey, db)
    userName = request.session.get("user_Name")
    userRole = request.session.get("user_Role")
    return templates.TemplateResponse('./trade/mytradingtrend.html', {
        "request": request, "trcoins": trcoins, "user_No": userno,
        "user_Name": userName, "user_Role": userRole, "setkey": setkey, "mycoins": mycoins
    })


@app.get('/upbittradetrend/{userno}/{setkey}')
async def upbittradetrend(request: Request, userno: int, setkey: str, db: AsyncSession = Depends(get_db)):
    hotcoins = await get_hotcoins(request, db)
    trcoins = [row[3] for row in hotcoins]
    userName = request.session.get("user_Name")
    userRole = request.session.get("user_Role")
    return templates.TemplateResponse('./trade/upbittradingtrend.html', {
        "request": request, "trcoins": trcoins, "user_No": userno,
        "user_Name": userName, "user_Role": userRole, "setkey": setkey
    })


@app.get('/settletrend/{userno}/{setkey}')
async def settletrend(request: Request, userno: int, setkey: str, db: AsyncSession = Depends(get_db)):
    trcoins = await tradedcoins(userno, db)
    mycoins = await checkwallet(userno, setkey, db)
    userName = request.session.get("user_Name")
    userRole = request.session.get("user_Role")
    return templates.TemplateResponse('./trade/mysettletrend.html', {
        "request": request, "trcoins": trcoins, "user_No": userno,
        "user_Name": userName, "user_Role": userRole, "setkey": setkey, "mycoins": mycoins
    })


@app.get('/upbitsettletrend/{userno}/{setkey}')
async def upbitsettletrend(request: Request, userno: int, setkey: str, db: AsyncSession = Depends(get_db)):
    hotcoins = await get_hotcoins(request, db)
    trcoins = [row[3] for row in hotcoins]
    userName = request.session.get("user_Name")
    userRole = request.session.get("user_Role")
    return templates.TemplateResponse('./trade/upbitsettletrend.html', {
        "request": request, "trcoins": trcoins, "user_No": userno,
        "user_Name": userName, "user_Role": userRole, "setkey": setkey
    })


@app.get('/userEdit/{userno}/{setkey}')
async def useredit(request: Request, userno: int, setkey: str, db: AsyncSession = Depends(get_db)):
    userName = request.session.get("user_Name")
    userRole = request.session.get("user_Role")
    userdtl = await get_userdetail(userno, setkey, db)
    return templates.TemplateResponse('./login/userDtl.html', {
        "request": request, "user_No": userno, "user_Name": userName,
        "user_Role": userRole, "setkey": setkey, "userdtl": userdtl
    })


@app.get('/rest_getwallet/{userno}/{setkey}')
async def restgetwallet(userno: int, setkey: str, db: AsyncSession = Depends(get_db)):
    mycoins = await checkwallet(userno, setkey, db)
    return JSONResponse({"success": True, "data": mycoins})


@app.get('/mytradestat/{userno}/{setkey}/{slot}')
async def mytradestat(request: Request, userno: int, setkey: str, slot: int, user_session: int = Depends(require_login),
                      db: AsyncSession = Depends(get_db)):
    setups = await getsetups(userno, slot, db)
    userName = request.session.get("user_Name")
    userRole = request.session.get("user_Role")
    userLicense = request.session.get("License")
    mycoins = await checkwallet(userno, setkey, db)
    orderlist = await get_orderlist(userno, setkey, slot, db)
    return templates.TemplateResponse('/trade/mytrademain.html', {
        "request": request, "setups": setups, "user_No": userno, "user_Name": userName,
        "user_Role": userRole, "setkey": setkey, "license": userLicense, "mycoins": mycoins,
        "slot": slot, "orderlist": orderlist
    })


@app.get('/mymtpondstat/{userno}/{setkey}')
async def mymtpondstat(request: Request, userno: int, setkey: str, user_session: int = Depends(require_login),
                       db: AsyncSession = Depends(get_db)):
    userName = request.session.get("user_Name")
    userRole = request.session.get("user_Role")
    userLicense = request.session.get("License")
    onoffstat = await get_onoff(userno, db)
    mysettings = await get_mtsetups(userno, db)
    mycoins = await checkwallet(userno, setkey, db)
    myorders = await get_mtorderlist(userno, setkey, db)
    return templates.TemplateResponse('/trade/mypondmain.html', {
        "request": request, "user_No": userno, "user_Name": userName,
        "user_Role": userRole, "setkey": setkey, "license": userLicense,
        "onoffstat": onoffstat[0] if onoffstat else 'N',
        "mysettings": mysettings, "myorders": myorders, "mycoins": mycoins
    })


@app.get('/mytradeSet/{userno}')
async def mytradeSet(request: Request, userno: int, db: AsyncSession = Depends(get_db)):
    coinlist = pyupbit.get_tickers(fiat="KRW")
    userName = request.session.get("user_Name")
    userRole = request.session.get("user_Role")
    setkey = request.session.get("setKey")
    trcnt = request.session.get("License")
    serverno = request.session.get("server_No")
    setlist = await selectsetlist(db)
    return templates.TemplateResponse('/trade/setmytrades.html', {
        "request": request, "coinlist": coinlist, "setlist": setlist, "trcnt": trcnt,
        "user_Name": userName, "setkey": setkey, "user_No": userno, "user_Role": userRole, "server_No": serverno
    })


@app.get('/mymtpondSet/{userno}')
async def mypondSet(request: Request, userno: int, db: AsyncSession = Depends(get_db)):
    coinlist = pyupbit.get_tickers(fiat="KRW")
    userName = request.session.get("user_Name")
    userRole = request.session.get("user_Role")
    setkey = request.session.get("setKey")
    trcnt = request.session.get("License")
    serverno = request.session.get("server_No")
    return templates.TemplateResponse('/trade/setmtpond.html', {
        "request": request, "coinlist": coinlist, "trcnt": trcnt,
        "user_Name": userName, "setkey": setkey, "user_No": userno, "user_Role": userRole, "server_No": serverno
    })


@app.get('/editSetup')
async def editSetup(
        request: Request,
        setno: str = Query(...),
        coinA: str = Query(...),
        coinB: str = Query(...),
        tabindex: str = Query(...),
        db: AsyncSession = Depends(get_db)
):
    coinlist = pyupbit.get_tickers(fiat="KRW")
    setlist = await selectsetlist(db)
    return templates.TemplateResponse('./trade/editmytrade.html', {
        "request": request, "coinlist": coinlist, "setno": setno,
        "coinA": coinA, "coinB": coinB, "setlist": setlist, "tabindex": tabindex,
        "setkey": request.session.get("setKey"),
        "user_No": request.session.get("user_No"),
        "user_Name": request.session.get("user_Name"),
        "user_Role": request.session.get("user_Role"),
        "server_No": request.session.get("server_No"),
    })


@app.post("/setupbids")
async def setupmybids(
        userno: str = Form(...),
        tabindex: str = Form(...),
        initprice: str = Form(...),
        lcrate: Optional[str] = Form(None),
        lcchk: Optional[str] = Form(None),
        tradeset: str = Form(...),
        coinn1: Optional[str] = Form(None),
        coinn2: Optional[str] = Form(None),
        coinn3: Optional[str] = Form(None),
        setkey: str = Form(...),
        svrno: str = Form(...),
        limityn: Optional[str] = Form(None),
        limitamt: Optional[str] = Form(None),
        db: AsyncSession = Depends(get_db),
):
    uno = int(userno)
    slot = int(tabindex)
    price = initprice.replace(',', '') if initprice else '0'
    tradeset_split = tradeset.split(',')
    tradeset_val = tradeset_split[0]
    bidsetps = tradeset_split[1] if len(tradeset_split) > 1 else "0"
    hno = tradeset_split[1] if len(tradeset_split) > 1 else "0"
    dyn = 'Y' if limityn == 'on' else 'N'
    lmtamt = (limitamt or '').replace(',', '') if limitamt else '0'
    bidrate = 1.0 if lcchk == 'on' else 0.0

    await erasebid(uno, setkey, slot, db)
    for coin in [coinn1, coinn2, coinn3]:
        if coin:
            await setupbid(
                uno, setkey, price, bidsetps, bidrate, lcrate, coin, svrno,
                tradeset_val, hno, dyn, lmtamt, dyn, slot, db
            )
    return RedirectResponse(url=f"/mytradestat/{uno}/{setkey}/{slot}", status_code=303)


@app.post("/setupmtponds")
async def setupmtponds(
        userno: str = Form(...),
        initprice: str = Form(...),
        addprice: str = Form(...),
        limitamt: str = Form(...),
        minmargin: str = Form(...),
        lcrate: Optional[str] = Form(None),
        setkey: str = Form(...),
        db: AsyncSession = Depends(get_db),
):
    uno = int(userno)
    initp = initprice.replace(',', '') if initprice else '0'
    addp = addprice.replace(',', '') if addprice else '0'
    limitp = limitamt.replace(',', '') if limitamt else '0'
    lcr = lcrate or '0'
    minm = minmargin.replace(',', '') if minmargin else '0'

    await erasemtpondsetup(uno, setkey, db)
    # setupmymtpondset 내부에서 DB 저장 후 Redis 동기화가 호출됩니다.
    await setupmymtpondset(uno, setkey, initp, addp, limitp, minm, lcr, db)
    return RedirectResponse(url=f"/mymtpondstat/{uno}/{setkey}", status_code=303)


@app.post("/setupbid")
async def setupmybid(
        setno: str = Form(...),
        userno: str = Form(...),
        slot: str = Form(...),
        coinn: str = Form(...),
        initprice: str = Form(...),
        lcrate: Optional[str] = Form(None),
        lcchk: Optional[str] = Form(None),
        tradeset: str = Form(...),
        setkey: str = Form(...),
        svrno: str = Form(...),
        limityn: Optional[str] = Form(None),
        limitamt: Optional[str] = Form(None),
        db: AsyncSession = Depends(get_db),
):
    sno = int(setno)
    uno = int(userno)
    slot_num = int(slot)
    price = initprice.replace(',', '') if initprice else '0'
    tradeset_split = tradeset.split(',')
    tradeset_val = tradeset_split[0]
    bidsetps = tradeset_split[1] if len(tradeset_split) > 1 else "0"
    hno = tradeset_split[1] if len(tradeset_split) > 1 else "0"
    dyn = 'Y' if limityn == 'on' else 'N'
    lmtamt = (limitamt or '').replace(',', '') if limitamt else '0'
    bidrate = 1.0 if lcchk == 'on' else 0.0

    await editbidsetup(
        sno, uno, setkey, price, bidsetps, bidrate, lcrate, coinn, int(svrno),
        tradeset_val, hno, dyn, lmtamt, dyn, slot_num, db
    )
    return RedirectResponse(url=f"/mytradestat/{uno}/{setkey}/{slot_num}", status_code=303)


@app.post("/changemypass")
async def change_password(data: dict = Body(...), db: AsyncSession = Depends(get_db)):
    sql = text("UPDATE traceUser SET userPasswd = PASSWORD(:passwd) WHERE userNo = :userno")
    await db.execute(sql, {"passwd": data["passwd"], "userno": data["uno"]})
    await db.commit()
    return {"result": "success"}


@app.post("/updateuserdtl")
async def update_userdetail(
        request: Request,
        uno: str = Form(...),
        apikey1: str = Form(...),
        apikey2: str = Form(...),
        svrno: str = Form(...),
        db: AsyncSession = Depends(get_db),
):
    setkey = request.session.get("setKey")
    await update_userdtl(int(uno), apikey1, apikey2, int(svrno), db)
    return RedirectResponse(url=f"/userEdit/{uno}/{setkey}", status_code=303)


@app.get('/rest_getorder/{userno}/{setkey}/{slot}')
async def restgetorder(userno: int, setkey: str, slot: int, db: AsyncSession = Depends(get_db)):
    orderlist = await get_orderlist(userno, setkey, slot, db)
    return JSONResponse({"success": True, "data": orderlist})


@app.get('/rest_getmtorder/{userno}/{setkey}')
async def restgetmtorder(userno: int, setkey: str, db: AsyncSession = Depends(get_db)):
    orderlist = await get_mtorderlist(userno, setkey, db)
    return JSONResponse({"success": True, "data": orderlist})


@app.post('/cancelOrder')
async def cancelorder_api(uno: int = Form(...), setkey: str = Form(...), uuid: str = Form(...),
                          db: AsyncSession = Depends(get_db)):
    order = await cancelorder(uno, setkey, uuid, db)
    return JSONResponse({"success": bool(order), "data": order})


@app.post('/setyns')
async def setyns(setno: int = Form(...), yn: str = Form(...), db: AsyncSession = Depends(get_db)):
    await setonoffs(setno, yn, db)
    return JSONResponse({"success": True, "data": yn})


@app.post('/setautostop')
async def setatstop(sno: int = Form(...), yesno: str = Form(...), db: AsyncSession = Depends(get_db)):
    await setautostop(sno, yesno, db)
    return JSONResponse({"success": True, "data": yesno})


@app.post('/setlosscut')
async def setlosscut(sno: int = Form(...), rate: float = Form(...), onoff: str = Form(...),
                     db: AsyncSession = Depends(get_db)):
    await setlconoff(sno, rate, onoff, db)
    return JSONResponse({"success": True, "data": rate})


@app.get('/upbittop30/{uno}/{setkey}')
async def upbittop30(request: Request, uno: int, setkey: str, db: AsyncSession = Depends(get_db)):
    coins = pyupbit.get_tickers(fiat="KRW")
    return templates.TemplateResponse('/trade/upbittop30.html', {
        "request": request, "trcnt": request.session.get("License"),
        "user_Name": request.session.get("user_Name"), "setkey": setkey,
        "user_No": uno, "user_Role": request.session.get("user_Role"),
        "coins": coins, "server_No": request.session.get("server_No")
    })


# ★ [수정] Redis 캐시 우선 조회 엔드포인트
@app.get('/api/mtpondsetup/{userno}')
async def mtpondsetup_all(userno: int, db: AsyncSession = Depends(get_db)):
    redis_key = f"mtpond:setup:{userno}"
    try:
        cached = await redis_client.get(redis_key)
        if cached:
            return jsonable_encoder([json.loads(cached)])
    except Exception as e:
        print(f"[REDIS][WARN] 캐시 조회 실패: {e}")

    # 캐시 미스 시 DB 조회
    sql = text(
        "SELECT activeYN,initAmt,addAmt,limitAmt,minMargin,maxMargin,tickRate,tickYN,lcRate,lcGap,maxCoincnt, martinYN, stopYN, stopAutoYN FROM mtSetup WHERE userNo = :userno AND attrib NOT LIKE :attrib")
    result = await db.execute(sql, {"userno": userno, "attrib": "%XXX%"})
    rows = result.fetchall()
    data = [dict(r._mapping) for r in rows]

    # 캐시에 채워넣기
    if data:
        try:
            await redis_client.set(redis_key, json.dumps(data[0]), ex=3600)
        except Exception:
            pass
    return jsonable_encoder(data)


# ★ [수정] ON/OFF 토글 시 DB 업데이트 및 Redis 즉시 전파
@app.post("/api/mtpondsetonoff/{userno}/{active}")
async def toggle_active_simple(userno: int, active: str, db: AsyncSession = Depends(get_db)):
    active_norm = active.strip().upper()
    if active_norm not in ("Y", "N"):
        raise HTTPException(status_code=400, detail="active 값은 Y 또는 N 이어야 합니다.")

    # setonoff 함수 내부에서 sync_mtsetup_to_redis 호출됨
    await setonoff(userno, active_norm, db)
    return {"userNo": userno, "activeYN": active_norm, "updated": True}


@app.post('/api/myorders/{userno}/{setkey}')
async def myorders(userno: int, setkey: str, db: AsyncSession = Depends(get_db)):
    try:
        myorders = await api_mtorderlist(userno, db)
        cprices = await get_current_prices()
        return JSONResponse({"success": True, "data": myorders, "cprices": cprices})
    except Exception as e:
        return JSONResponse({"success": False, "data": [], "cprices": []})


@app.get("/phapp/mlogin/{userid}/{passwd}")
async def mlogin(userid: str, passwd: str, db: AsyncSession = Depends(get_db)):
    try:
        query = text(
            "SELECT userNo, userName, setupKey from traceUser where userId = :userid and userPasswd = PASSWORD(:passwd)")
        r = await db.execute(query, {"userid": userid, "passwd": passwd})
        rows = r.fetchone()
        if rows is None:
            return {"error": "No data found for the given data."}
        return {"userno": rows[0], "username": rows[1], "setupkey": rows[2]}
    except Exception as e:
        print("mLogin error:", e)
        return {"error": "Authentication failed"}


@app.get("/api/balance/{userno}/{setkey}")
async def api_my_balance(request: Request, userno: int, setkey: str, db: AsyncSession = Depends(get_db)):
    try:
        mycoins = await checkwallet(userno, setkey, db)
        cprices = await get_current_prices()
        return {"success": True, "userNo": userno, "mycoins": mycoins, "cuprices": cprices}
    except Exception as e:
        print("Get API Balances Error:", e)
        raise HTTPException(status_code=500, detail="지갑 정보를 불러오는데 실패했습니다.")


@app.get("/excoinlist/{userNo}/{setkey}")
async def excoin(request: Request, userNo: int, setkey: str, db: AsyncSession = Depends(get_db)):
    coinlist = pyupbit.get_tickers(fiat="KRW")
    try:
        query = text("SELECT DISTINCT market FROM exCoinlist WHERE userNo in (0, :userno) and attrib NOT LIKE :attrib")
        r = await db.execute(query, {"userno": userNo, "attrib": "%XXX%"})
        rows = r.fetchall()
        excoinlist = [row[0] for row in rows] if rows else []
    except Exception as e:
        print("excoinlist error:", e)
        excoinlist = []
    return templates.TemplateResponse('/trade/excoin.html', {
        "request": request, "user_No": userNo, "setkey": setkey,
        "coinlist": coinlist, "excoinlist": excoinlist
    })


# ★ [수정] 제외 코인 변경 시 Redis Set 갱신 및 PubSub 전파
@app.post("/setexCoin/{userNo}")
async def setexcoin(
        request: Request,
        userNo: int,
        selcoin: Optional[List[str]] = Form(default=None, alias="selcoin[]"),
        db: AsyncSession = Depends(get_db)
):
    selcoin = selcoin or []
    try:
        query = text("UPDATE exCoinlist set attrib = :attx WHERE userNo = :userno")
        await db.execute(query, {"attx": "XXXUPXXXUP", "userno": userNo})
        for coin in selcoin:
            query = text("INSERT INTO exCoinlist (userNo, market) values (:userNo, :market)")
            await db.execute(query, {"userNo": userNo, "market": coin})
        await db.commit()

        # ★ DB 갱신 후 Redis Set 업데이트 및 알림
        await sync_excoins_to_redis(userNo, db)
    except Exception as e:
        await db.rollback()
        print("setexcoin error:", e)
    return RedirectResponse(url=f"/excoinlist/{userNo}/{request.session.get('setKey')}", status_code=303)


@app.get("/balancegraph/{uno}")
async def balance_graph(
        request: Request,
        uno: int,
        user_session: int = Depends(require_login),
        db: AsyncSession = Depends(get_db)
):
    if uno != user_session:
        return RedirectResponse(url="/", status_code=303)

    userName = request.session.get("user_Name")
    userRole = request.session.get("user_Role")
    setKey = request.session.get("setKey")

    sql = text("""
               SELECT logNo, timeStamp, totalKRW, balanceKRW
               FROM walletBalance
               WHERE userNo = :userno AND attrib NOT LIKE :xattr
               ORDER BY timeStamp ASC
                   LIMIT 120
               """)
    result = await db.execute(sql, {"userno": uno, "xattr": "%XXX%"})
    rows = result.fetchall()

    items = []
    tval = []
    ival = []

    for r in rows:
        ts_str = r[1].strftime("%Y-%m-%d %H:%M") if isinstance(r[1], datetime) else str(r[1])
        total_krw = int(r[2]) if r[2] is not None else 0
        items.append([r[0], ts_str, total_krw])
        tval.append(ts_str)
        ival.append(total_krw)

    return templates.TemplateResponse(
        "/trade/incsum.html",
        {
            "request": request,
            "user_No": uno,
            "user_Name": userName,
            "user_Role": userRole,
            "setkey": setKey,
            "items": items,
            "tval": json.dumps(tval),
            "ival": json.dumps(ival),
        }
    )