from fastapi import FastAPI
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

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("/login/login.html", {"request": request})

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
