CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- 1. Document Registry (supports versioning & clean soft-deletes)
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID,                     -- NULL = public pre-indexed vendor docs
    vendor_name TEXT NOT NULL,          -- e.g., 'AWS', 'Stripe', 'OpenAI'
    doc_type TEXT NOT NULL,             -- 'SLA', 'DPA', 'TOS', 'SECURITY'
    source_url TEXT,
    effective_date DATE,
    content_hash TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL DEFAULT 'indexing', -- 'indexing' | 'active' | 'superseded'
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Hybrid Search Chunks
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tenant_id UUID,
    vendor_name TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    section_path TEXT NOT NULL,         -- e.g., 'Section 4 > 4.2 Service Credits'
    page_number INT,
    raw_content TEXT NOT NULL,
    contextual_content TEXT NOT NULL,   -- Document/section prefix + raw_content
    chunk_hash TEXT NOT NULL,
    embedding VECTOR(1536) NOT NULL,
    fts_vector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(section_path, '') || ' ' || coalesce(raw_content, ''))
    ) STORED,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Vector HNSW index (cosine distance)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON document_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Lexical GIN index
CREATE INDEX IF NOT EXISTS idx_chunks_fts_gin
    ON document_chunks USING gin (fts_vector);

-- Filter index for multi-tenant + vendor isolation
CREATE INDEX IF NOT EXISTS idx_chunks_tenant_vendor
    ON document_chunks (tenant_id, vendor_name, doc_type);

-- 3. Structured SLA Metrics (for deterministic SQL & Python tools)
CREATE TABLE IF NOT EXISTS sla_tiers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    vendor_name TEXT NOT NULL,
    service_name TEXT NOT NULL,
    min_uptime_pct NUMERIC(6,3) NOT NULL,  -- e.g., 99.000
    max_uptime_pct NUMERIC(6,3) NOT NULL,  -- e.g., 99.950
    credit_pct NUMERIC(5,2) NOT NULL,      -- e.g., 10.00
    claim_window_days INT
);