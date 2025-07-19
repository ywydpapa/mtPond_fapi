from fastapi import FastAPI
import random
import dotenv
import os
import asyncio
import time
from typing import Dict
import math
import aiohttp
from fastapi import FastAPI, Depends, Request, Form, Response, HTTPException, status, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from starlette.middleware.sessions import SessionMiddleware
from fastapi.staticfiles import StaticFiles
import jinja2
import dotenv
import pyupbit
import requests
from fastapi import WebSocket, WebSocketDisconnect
import httpx
import websockets
import json
from sqlalchemy import text
from pandas import DataFrame


dotenv.load_dotenv()
DATABASE_URL = os.getenv("dburl")
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

app = FastAPI()

app.add_middleware(SessionMiddleware, secret_key="supersecretkey")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
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

async def get_db():
    async with async_session() as session:
        yield session

async def get_current_prices():
    server_url = "https://api.upbit.com"
    params = {"quote_currencies": "KRW"}
    res = requests.get(server_url + "/v1/ticker/all", params=params)
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
            item = data[0]
            return item.get("trade_price")
        return None

async def selectUsers(uid:str, upw:str,  db: AsyncSession = Depends(get_db)):
    row = None
    setkey = None
    try:
        sql = text("SELECT userNo, userName, serverNo, userRole FROM traceUser WHERE userPasswd=password(:passwd) AND userId=:userid AND attrib NOT LIKE :xattr")
        result = await db.execute(sql,{"passwd":upw,"userid":uid,"xattr":"%XXXUP%"})
        row = result.fetchone()
        if row is not None:
            setkey = random.randint(100000, 999999)
    except Exception as e:
        print('접속오류', e)
    finally:
        return row, setkey

async def listUsers(db: AsyncSession = Depends(get_db)):
    rows = None
    try:
        sql = text("SELECT * FROM traceUser WHERE attrib NOT LIKE :xattr")
        result = db.execute(sql, {"xattr": "%XXXUP%"})
        rows = result.fetchall()
    except Exception as e:
        print('접속오류', e)
    finally:
        return rows

async def detailuser(uno:int, db: AsyncSession = Depends(get_db)):
    row = None
    try:
        sql = text("SELECT * FROM traceUser WHERE userNo = :userno and attrib NOT LIKE :xattr")
        result = db.execute(sql, {"userno":uno,"xattr": "%XXXUP%"})
        row = result.fetchone()
    except Exception as e:
        print('접속오류', e)
    finally:
        return row

async def setKeys(uno:int, setkey:str, db: AsyncSession = Depends(get_db)):
    try:
        sql = text("UPDATE traceUser SET setupKey = :setk, lastLogin = now() where userNo=:userno")
        await db.execute(sql, {"userno":uno,"setk":setkey})
        await db.commit()
    except Exception as e:
        print('접속오류', e)
    finally:
        return True

async def check_setkey(uno:int,setkey:str, db: AsyncSession = Depends(get_db)):
    try:
        sql = text("SELECT userNo FROM traceUser WHERE userNo=:userno and setupKey=:setkey and attrib not like :xattr")
        result = await db.execute(sql, {"userno":uno,"setkey":setkey,"xattr": "%XXXUP%"})
        row = result.fetchone()
        if row[0] == uno:
            return True
        else:
            return False
    except Exception as e:
        print("코드 체크 에러",e)

def require_login(request: Request):
    user_no = request.session.get("user_No")
    if not user_no:
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": "/"},
            detail="세션이 만료되어 재로그인이 필요합니다."
        )
    return user_no

async def getKeys(uno:int,setkey:str, db: AsyncSession = Depends(get_db)):
    global key1, key2
    key1 = None
    key2 = None
    try:
        sql = text("SELECT apiKey1, apiKey2 FROM traceUser WHERE setupKey=:setk AND userNo=:userno and attrib not like :xattr")
        result = await db.execute(sql, {"setk":setkey,"userno":uno,"xattr": "%XXXUP%"})
        keys = result.fetchone()
        if len(keys) == 0:
            print("No available Keys !!")
        else:
            key1 = keys[0]
            key2 = keys[1]
    except Exception as e:
        print('키로드 오류')
        return False
    finally:
        return key1, key2

async def checkwallet(uno:int, setkey:str, db: AsyncSession = Depends(get_db)):
    global key1, key2, walletitems
    walletitems = []
    try:
        keys = await getKeys(uno,setkey,db)
        key1 = keys[0]
        key2 = keys[1]
        upbit = pyupbit.Upbit(key1, key2)
        walletitems = upbit.get_balances()
    except Exception as e:
        print("지갑 불러오기 에러", e)
        return False
    finally:
        return walletitems

async def clearcache(db: AsyncSession = Depends(get_db)):
    try:
        sql = text("RESET QUERY CACHE")
        await db.execute(sql)
        await db.commit()
    except Exception as e:
        print("Clear Cache Error !!",e)
        return False
    finally:
        return True

async def buycoinmarket(uno, coink, setkey, amt, db: AsyncSession = Depends(get_db)):
    try:
        keys = await getKeys(uno,setkey,db)
        key1 = keys[0]
        key2 = keys[1]
        upbit = pyupbit.Upbit(key1, key2)
        buy = upbit.buy_market_order(coink, amt)
        return buy
    except Exception as e:
        print("시장가 구매 에러",e)
        return False

async def sellcoinpercent(uno, coink, setkey, volm, db: AsyncSession = Depends(get_db)):
    try:
        keys = await getKeys(uno, setkey, db)
        key1 = keys[0]
        key2 = keys[1]
        upbit = pyupbit.Upbit(key1, key2)
        walt = upbit.get_balances() #지갑내 잔고 확보
        crp = pyupbit.get_current_price(coink) #현재가 재확인
        for coin in walt: #잔고중 일치하는 코인만 잔고
            if coin['currency'] == coink.split('-')[1]:
                if float(coin['balance'])*float(crp) < 5000:
                    buy5000 = upbit.buy_market_order(coink, 5000) #5000원 추가구매
                    break
                else:
                    result = upbit.sell_market_order(coink, volm)
                    return result
            else:
                continue
    except Exception as e:
        print("시장가 비율 매도 에러",e)
        return False

async def cancelorder(uno:int,setkey:str ,uuid:str, db: AsyncSession = Depends(get_db)):
    global rows
    try:
        keys = await getKeys(uno,setkey,db)
        upbit = await pyupbit.Upbit(keys[0], keys[1])
        await upbit.cancel_order(uuid)
        return True
    except Exception as e:
        print("거래 취소 에러 ", e)
        return False

async def setmypasswd(uno:int,setkey:str, passwd:str, db: AsyncSession = Depends(get_db)):
    try:
        sql = text("UPDATE traceUser SET userPasswd = password(:pass) where userNo=:userno")
        await db.execute(sql, {"userno":uno,"pass":passwd})
        await db.commit()
        return True
    except Exception as e:
        print('비밀번호 업데이트 오류', e)

async def dashcandle548(coink):
    candles: DataFrame | None = await pyupbit.get_ohlcv(coink, interval="minute5", count=48)
    return candles


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("/login/login.html", {"request": request})

@app.post("/loginchk")
async def login(request: Request, response: Response, uid: str = Form(...), upw: str = Form(...),
                db: AsyncSession = Depends(get_db)):
    user =await selectUsers(uid, upw, db)
    print(user)
    if user is None:
        return templates.TemplateResponse("login/login.html", {"request": request, "error": "Invalid credentials"})
    else:
        await setKeys(user[0][0], user[1], db)
    # 서버 세션에 사용자 ID 저장
    request.session["user_No"] = user[0][0]
    request.session["user_Name"] = user[0][1]
    request.session["server_no"] = user[0][2]
    request.session["user_Role"] = user[0][3]
    request.session["setKey"] = user[1]
    return RedirectResponse(url=f"/balance/{user[0][0]}", status_code=303)

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()  # 세션 삭제
    return RedirectResponse(url="/")

@app.get("/balance/{uno}")
async def my_balance(request: Request, uno: int, user_session: int = Depends(require_login),
                     db: AsyncSession = Depends(get_db)):
    global myavgp
    mycoins = None
    myavgp = None
    if uno != user_session:
        return RedirectResponse(url="/", status_code=303)
    try:
        setKey = request.session.get("setKey")
        userName = request.session.get("user_Name")
        userRole = request.session.get("user_Role")
        mycoins = await checkwallet(uno,setKey, db)
        cprices = await get_current_prices()
    except Exception as e:
        print("Get Balances Error !!", e)
        mycoins = None
    return templates.TemplateResponse("wallet/mywallet.html",
                                      {"request": request, "user_No": uno,"user_Name":userName, "user_Role":userRole, "setkey":setKey,"mycoins": mycoins, "myavgp":myavgp, "cuprices":cprices})

@app.post("/tradebuymarket/{uno}/{setkey}/{coinn}/{costk}")
async def tradebuymarket(request: Request,uno: int,setkey:str, coinn: str, costk: float, user_session: int = Depends(require_login), db: AsyncSession = Depends(get_db)):
    if uno != user_session:
        return JSONResponse({"success": False, "message": "권한이 없습니다.", "redirect": "/"})
    try:
        coink = "KRW-" + coinn
        butm = await buycoinmarket(uno, coink, setkey, costk,db)
        if butm:
            # 거래 성공
            return JSONResponse({"success": True, "redirect": f"/balance/{uno}"})
        else:
            # 거래 실패
            return JSONResponse({"success": False, "message": "거래 실패", "redirect": f"/balance/{uno}"})
    except Exception as e:
        print("Error!!", e)
        return JSONResponse({"success": False, "message": "서버 오류", "redirect": f"/balance/{uno}"})

@app.post("/tradesellmarket/{uno}/{setkey}/{coinn}/{volm}")
async def tradesellmarket(request: Request,uno: int,setkey:str, coinn: str, volm: float, user_session: int = Depends(require_login), db: AsyncSession = Depends(get_db)):
    if uno != user_session:
        return JSONResponse({"success": False, "message": "권한이 없습니다.", "redirect": "/"})
    try:
        coink = "KRW-" + coinn
        sellm = await sellcoinpercent(uno, coink, setkey, volm ,db)
        if sellm:
            # 거래 성공
            return JSONResponse({"success": True, "redirect": f"/balance/{uno}"})
        else:
            # 거래 실패
            return JSONResponse({"success": False, "message": "거래 실패", "redirect": f"/balance/{uno}"})
    except Exception as e:
        print("Error!!", e)
        return JSONResponse({"success": False, "message": "서버 오류", "redirect": f"/balance/{uno}"})

@app.websocket("/ws/coinprice")
async def coin_price_ws(websocket: WebSocket):
    await websocket.accept()
    coins = websocket.query_params.get("coins", "")
    coin_list = coins.split(",") if coins else []
    try:
        async for market, current_price, change in upbit_ws_price_stream(coin_list):
            await websocket.send_json({"market": market, "current_price": current_price, "change": change})
    except WebSocketDisconnect:
        print(f"WebSocket disconnected: coins {coin_list}")
    except Exception as e:
        print("WebSocket Error:", e)

async def upbit_ws_price_stream(markets: list):
    uri = "wss://api.upbit.com/websocket/v1"
    subscribe_data = [{
        "ticket": "test",
    }, {
        "type": "ticker",
        "codes": markets,
        "isOnlyRealtime": True
    }]
    async with websockets.connect(uri, ping_interval=60) as websocket:
        await websocket.send(json.dumps(subscribe_data))
        while True:
            data = await websocket.recv()
            parsed = json.loads(data)
            yield parsed['code'], parsed['trade_price'], parsed['change']

