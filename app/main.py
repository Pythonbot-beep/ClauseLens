from fastapi import Depends, FastAPI
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.session import get_db

app = FastAPI(title="ClauseLens API", version="0.1.0")


@app.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    db_ok = (await db.execute(text("SELECT 1"))).scalar() == 1
    pgvector_ver = (
        await db.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'"))
    ).scalar()

    redis_client = Redis.from_url(settings.redis_url)
    try:
        redis_ok = await redis_client.ping()
    finally:
        await redis_client.aclose()

    return {
        "status": "ok" if (db_ok and redis_ok) else "degraded",
        "postgres": db_ok,
        "pgvector_version": pgvector_ver,
        "redis": redis_ok,
    }