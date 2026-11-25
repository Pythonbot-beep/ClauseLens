import hashlib
from pathlib import Path
from arq import create_pool
from arq.connections import RedisSettings
from fastapi import Depends, FastAPI, File, Form, UploadFile
from pydantic import BaseModel
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.session import get_db
from app.rag.embeddings import embed_texts
from app.rag.retriever import hybrid_search

app = FastAPI(title="ClauseLens API", version="0.2.0")
UPLOAD_DIR = Path("data/raw")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

class SearchRequest(BaseModel):
    query: str
    vendor_name: str | None = None
    doc_type: str | None = None
    top_k: int = 5

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

@app.post("/documents/upload")
async def upload_document(
    vendor_name: str = Form(...),
    doc_type: str = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    file_bytes = await file.read()
    content_hash = hashlib.sha256(file_bytes).hexdigest()

    # Skip re-indexing if this exact file was already uploaded
    existing = (
        await db.execute(
            text("SELECT id, status FROM documents WHERE content_hash = :hash"),
            {"hash": content_hash},
        )
    ).mappings().first()
    if existing:
        return {"document_id": str(existing["id"]), "status": existing["status"], "deduplicated": True}

    file_path = UPLOAD_DIR / f"{content_hash}_{file.filename}"
    file_path.write_bytes(file_bytes)

    doc_id = (
        await db.execute(
            text("""
                INSERT INTO documents (vendor_name, doc_type, content_hash, status)
                VALUES (:vendor_name, :doc_type, :content_hash, 'indexing')
                RETURNING id
            """),
            {"vendor_name": vendor_name, "doc_type": doc_type, "content_hash": content_hash},
        )
    ).scalar_one()
    await db.commit()

    arq_pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    await arq_pool.enqueue_job(
        "ingest_document", str(doc_id), str(file_path), vendor_name, doc_type
    )
    await arq_pool.aclose()

    return {"document_id": str(doc_id), "status": "indexing", "deduplicated": False}


@app.post("/search")
async def search_clauses(req: SearchRequest, db: AsyncSession = Depends(get_db)):
    query_vec = (await embed_texts([req.query]))[0]
    results = await hybrid_search(
        db=db,
        query_text=req.query,
        query_embedding=query_vec,
        vendor_name=req.vendor_name.strip().lower(),
        doc_type=req.doc_type.strip().lower(),
        top_k=req.top_k,
    )
    return {"count": len(results), "results": results}