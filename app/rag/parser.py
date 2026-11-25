import hashlib
from dataclasses import dataclass
from pathlib import Path

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter


_converter = DocumentConverter()
_chunker = HybridChunker(merge_peers=True)

@dataclass
class ParsedChunk:
    section_path: str
    page_number: int | None
    raw_content: str
    contextual_content: str
    chunk_hash: str

def parse_and_chunk(file_path: str | Path, vendor_name: str, doc_type: str) -> list[ParsedChunk]:
    doc = _converter.convert(str(file_path)).document
    parsed_chunks: list[ParsedChunk] = []

    for chunk in _chunker.chunk(dl_doc=doc):
        raw_text = chunk.text.strip()
        if not raw_text:
            continue

        headings = getattr(chunk.meta, "headings", None) or []
        section_path = " > ".join(headings) if headings else "General"

        pages = sorted(
            {
                prov.page_no
                for item in getattr(chunk.meta, "doc_items", [])
                for prov in getattr(item, "prov", [])
                if getattr(prov, "page_no", None) is not None
            }
        )
        page_number = pages[0] if pages else None

        # Prepend vendor/doc metadata + Docling's heading/table-enriched text for embedding
        docling_context = _chunker.contextualize(chunk=chunk)
        contextual_text = f"Vendor: {vendor_name} | Document: {doc_type}\n{docling_context}"
        chunk_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

        parsed_chunks.append(
            ParsedChunk(
                section_path=section_path,
                page_number=page_number,
                raw_content=raw_text,
                contextual_content=contextual_text,
                chunk_hash=chunk_hash,
            )
        )

    return parsed_chunks