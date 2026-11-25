import asyncio
from uuid import UUID
from arq.connections import RedisSettings
from sqlalchemy import text

from app.config import settings
from app.db.session import AsyncSessionLocal
from app.rag.embeddings import embed_texts
from app.rag.parser import parse_and_chunk


async def ingest_document(
        ctx:dict,
        document_id :str,
        file_path:str,
        vendor_name: str,
        doc_type: str,
        tenant_id: str | None = None,
) -> dict:
    chunks = asyncio.to_thread(parse_and_chunk, file_path, vendor_name, doc_type)
    if not chunks:
        return {"document_id": document_id, "chunks_indexed": 0}    

    #Embed contextualized chunk text in batches
    embeddings= await embed_texts([c.contextual_content for c in chunks])

    # 3. Atomic write to PostgreSQL + flip document status to 'active'
    async with AsyncSessionLocal() as db:
        for chunk, vector in zip(chunks, embeddings):
            await db.execute(
                text("""
                    INSERT INTO document_chunks (
                        document_id, tenant_id, vendor_name, doc_type,
                        section_path, page_number, raw_content,
                        contextual_content, chunk_hash, embedding
                    ) VALUES (
                        :document_id, :tenant_id, :vendor_name, :doc_type,
                        :section_path, :page_number, :raw_content,
                        :contextual_content, :chunk_hash, CAST(:embedding AS vector)
                    )
                """),
                {
                    "document_id": document_id,
                    "tenant_id": tenant_id,
                    "vendor_name": vendor_name,
                    "doc_type": doc_type,
                    "section_path": chunk.section_path,
                    "page_number": chunk.page_number,
                    "raw_content": chunk.raw_content,
                    "contextual_content": chunk.contextual_content,
                    "chunk_hash": chunk.chunk_hash,
                    "embedding": str(vector),
                },
            )

        await db.execute(
            text("UPDATE documents SET status = 'active' WHERE id = :doc_id"),
            {"doc_id": document_id},
        )
        await db.commit()

    return {"document_id": document_id, "chunks_indexed": len(chunks)}


class WorkerSettings:
    functions = [ingest_document]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
