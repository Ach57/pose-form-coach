"""
EDITH Backend
=============
FastAPI application entry point.

Run with:
    uvicorn main:app --reload --port 8000
"""

from config.config import FAST_API_TITLE, ALLOWED_ORIGINS

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.ws import router as ws_router
from routers.health import router as health_router

app = FastAPI(title=FAST_API_TITLE)

# Allow the React dev server (port 5173) and any local origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ws_router)
app.include_router(health_router)