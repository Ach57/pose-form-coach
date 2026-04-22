"""
EDITH Backend
=============
FastAPI application entry point.

Run with:
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.ws import router as ws_router

app = FastAPI(title="EDITH Pose Detection API")

# Allow the React dev server (port 5173) and any local origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ws_router)


@app.get("/health")
async def health():
    """Simple health check — useful to confirm the server is up."""
    return {"status": "online", "system": "EDITH"}