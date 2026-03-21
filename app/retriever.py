import logging
from typing import List
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from app.ingestor import get_chroma_collection

log = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text:     str
    page:     int
    source:   str
    score:    float
    method:   str = "hybrid"


def retrieve_vector(query: str, model: SentenceTransformer,
                    top_k: int = 10,
                    collection_name: str = "research_docs",
                    source_filter: str = None) -> List[RetrievedChunk]:
    """Pure vector similarity search."""
    query_embedding = model.encode([query]).tolist()
    collection      = get_chroma_collection(collection_name)
    where           = {"source": source_filter} if source_filter else None

    results = collection.query(
        query_embeddings = query_embedding,
        n_results        = top_k,
        where            = where,
        include          = ["documents", "metadatas", "distances"]
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        distance = results["distances"][0][i]
        chunks.append(RetrievedChunk(
            text   = results["documents"][0][i],
            page   = results["metadatas"][0][i].get("page", 0),
            source = results["metadatas"][0][i].get("source", "unknown"),
            score  = round(1 - distance, 4),
            method = "vector"
        ))
    return chunks


def retrieve_bm25(query: str,
                  collection_name: str = "research_docs",
                  source_filter: str = None,
                  top_k: int = 10) -> List[RetrievedChunk]:
    """BM25 keyword search over all stored chunks."""
    collection = get_chroma_collection(collection_name)
    where      = {"source": source_filter} if source_filter else None
    all_data   = collection.get(where=where,
                                include=["documents", "metadatas"])

    if not all_data["documents"]:
        return []

    docs      = all_data["documents"]
    metadatas = all_data["metadatas"]

    # tokenise for BM25
    tokenised = [doc.lower().split() for doc in docs]
    bm25      = BM25Okapi(tokenised)
    scores    = bm25.get_scores(query.lower().split())

    # get top_k indices
    top_indices = sorted(range(len(scores)),
                         key=lambda i: scores[i],
                         reverse=True)[:top_k]

    chunks = []
    for idx in top_indices:
        if scores[idx] > 0:
            chunks.append(RetrievedChunk(
                text   = docs[idx],
                page   = metadatas[idx].get("page", 0),
                source = metadatas[idx].get("source", "unknown"),
                score  = round(float(scores[idx]), 4),
                method = "bm25"
            ))
    return chunks


def reciprocal_rank_fusion(vector_chunks: List[RetrievedChunk],
                           bm25_chunks: List[RetrievedChunk],
                           k: int = 60) -> List[RetrievedChunk]:
    """
    Combine vector and BM25 results using RRF scoring.
    Documents appearing in both lists get boosted.
    k=60 is the standard constant that prevents over-emphasis
    on top positions while maintaining rank order.
    """
    rrf_scores  = {}
    all_chunks  = {}

    for rank, chunk in enumerate(vector_chunks):
        key = chunk.text[:100]
        rrf_scores[key]  = rrf_scores.get(key, 0) + 1 / (k + rank + 1)
        all_chunks[key]  = chunk

    for rank, chunk in enumerate(bm25_chunks):
        key = chunk.text[:100]
        rrf_scores[key]  = rrf_scores.get(key, 0) + 1 / (k + rank + 1)
        all_chunks[key]  = chunk

    sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

    fused = []
    for key in sorted_keys:
        chunk       = all_chunks[key]
        fused_chunk = RetrievedChunk(
            text   = chunk.text,
            page   = chunk.page,
            source = chunk.source,
            score  = round(rrf_scores[key], 6),
            method = "hybrid_rrf"
        )
        fused.append(fused_chunk)

    return fused


def retrieve(query: str,
             model: SentenceTransformer,
             top_k: int = 4,
             collection_name: str = "research_docs",
             source_filter: str = None) -> List[RetrievedChunk]:
    """
    Full hybrid retrieval pipeline:
    1. Vector search (semantic)
    2. BM25 search (keyword/exact)
    3. RRF fusion — documents in both lists get boosted
    4. Return top_k fused results
    """
    log.info(f"Hybrid retrieval for: '{query[:60]}'")

    vector_results = retrieve_vector(query, model,
                                     top_k=top_k * 2,
                                     collection_name=collection_name,
                                     source_filter=source_filter)

    bm25_results   = retrieve_bm25(query,
                                   collection_name=collection_name,
                                   source_filter=source_filter,
                                   top_k=top_k * 2)

    fused          = reciprocal_rank_fusion(vector_results, bm25_results)
    final          = fused[:top_k]

    log.info(f"Vector: {len(vector_results)} | "
             f"BM25: {len(bm25_results)} | "
             f"Fused: {len(final)} returned")

    if final:
        log.info(f"Top result — Page {final[0].page} | "
                 f"Score {final[0].score} | Method {final[0].method}")

    return final


def format_context(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks into context string for the LLM."""
    parts = []
    for chunk in chunks:
        parts.append(
            f"[Source: {chunk.source} | Page {chunk.page} | "
            f"Score: {chunk.score} | Method: {chunk.method}]\n"
            f"{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    from app.ingestor import load_embedding_model
    model = load_embedding_model()

    test_queries = [
        "What is the main topic of this paper?",
        "What was the accuracy improvement percentage?",
        "MAML gradient meta-learning"
    ]

    for q in test_queries:
        print(f"\n{'='*55}")
        print(f"Query: {q}")
        results = retrieve(q, model, top_k=3)
        for r in results:
            print(f"  [{r.method}] Page {r.page} | "
                  f"Score {r.score} | {r.text[:80]}...")