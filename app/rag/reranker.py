import asyncio
from sentence_transformers import CrossEncoder

_reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", automodel_args={"torch_dtype": "auto"})

def _score_sync(query: str, chunks: list[dict], top_n: int,min_score: float) -> list[dict]:
    if not chunks:
        return []
    pairs = [(query, f"{c['section_path']}\n{c['raw_content']}") for c in chunks]
    scores = _reranker.predict(pairs)

    reranked=[]

    for chunk, score in zip(chunks,scores):
        score_val = float(score)
        if score_val >= min_score:
            item = dict(chunk)
            item["rerank_score"] = round(score_val, 4)
            reranked.append(item)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]


async def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_n: int = 5,
    min_score: float = 0.30,
) -> list[dict]:
    return await asyncio.to_thread(_score_sync, query, chunks, top_n, min_score)