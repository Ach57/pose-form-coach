from fastapi import APIRouter

router = APIRouter()

@router.get('/api/health')
async def health():
    """Simple health check — useful to confirm the server is up."""
    return {"status": "online", "system": "EDITH"}
