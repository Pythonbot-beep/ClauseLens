from typing import Optional
from uuid import UUID
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


HYBRID_RRF_SQL = text("""
WITH semantic_search AS (
    SELECT
        id,
        ROW_NUMBER() OVER (ORDER BY embedding <=> CAST(:query_embedding AS vector)) AS rank
    FROM document_chunks
    WHERE (tenant_id IS NULL OR tenant_id = CAST(:tenant_id AS uuid))
      AND (CAST(:vendor_name AS text) IS NULL OR LOWER(vendor_name) = LOWER(CAST(:vendor_name AS text)))
      AND (CAST(:doc_type AS text) IS NULL OR LOWER(doc_type) = LOWER(CAST(:doc_type AS text)))
    ORDER BY embedding <=> CAST(:query_embedding AS vector)
    LIMIT :candidate_pool
),
keyword_search AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            ORDER BY ts_rank_cd(fts_vector, websearch_to_tsquery('english', :query_text)) DESC
        ) AS rank
    FROM document_chunks
    WHERE (tenant_id IS NULL OR tenant_id = CAST(:tenant_id AS uuid))
      AND (CAST(:vendor_name AS text) IS NULL OR LOWER(vendor_name) = LOWER(CAST(:vendor_name AS text)))
      AND (CAST(:doc_type AS text) IS NULL OR LOWER(doc_type) = LOWER(CAST(:doc_type AS text)))
      AND fts_vector @@ websearch_to_tsquery('english', :query_text)
    ORDER BY ts_rank_cd(fts_vector, websearch_to_tsquery('english', :query_text)) DESC
    LIMIT :candidate_pool
),
rrf_fused AS (
    SELECT
        COALESCE(s.id, k.id) AS id,
        COALESCE(1.0 / (:rrf_k + s.rank), 0.0) +
        COALESCE(1.0 / (:rrf_k + k.rank), 0.0) AS rrf_score,
        s.rank AS dense_rank,
        k.rank AS lexical_rank
    FROM semantic_search s
    FULL OUTER JOIN keyword_search k ON s.id = k.id
)
SELECT
    c.id,
    c.document_id,
    c.vendor_name,
    c.doc_type,
    c.section_path,
    c.page_number,
    c.raw_content,
    r.rrf_score,
    r.dense_rank,
    r.lexical_rank
FROM rrf_fused r
JOIN document_chunks c ON c.id = r.id
ORDER BY r.rrf_score DESC
LIMIT :top_k;
""")


async def hybrid_search(
    db: AsyncSession,
    query_text: str,
    query_embedding: list[float],
    tenant_id: Optional[UUID] = None,
    vendor_name: Optional[str] = None,
    doc_type: Optional[str] = None,
    candidate_pool: int = 50,
    top_k: int = 20,
    rrf_k: int = 60,
) -> list[dict]:
    # Enable pgvector 0.8+ iterative scan so filtered HNSW queries don't under-return rows
    await db.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order;"))

    result = await db.execute(
        HYBRID_RRF_SQL,
        {
            "query_text": query_text,
            "query_embedding": str(query_embedding),
            "tenant_id": str(tenant_id) if tenant_id else None,
            "vendor_name": vendor_name,
            "doc_type": doc_type,
            "candidate_pool": candidate_pool,
            "top_k": top_k,
            "rrf_k": rrf_k,
        },
    )
    return [dict(row._mapping) for row in result.fetchall()]