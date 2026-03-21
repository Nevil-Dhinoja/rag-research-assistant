import logging
from typing import List
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from app.ingestor import get_chroma_collection

log = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text:     str
    page:     int
    source:   str
    score:    float


def retrieve(query: str, model: SentenceTransformer,
             top_k: int = 4,
             collection_name: str = "research_docs",
             source_filter: str = None) -> List[RetrievedChunk]:
    """
    Embed the query, search ChromaDB for most similar chunks,
    return top_k results with page numbers and similarity scores.
    """
    log.info(f"Retrieving top-{top_k} chunks for: '{query[:60]}'")

    query_embedding = model.encode([query]).tolist()
    collection      = get_chroma_collection(collection_name)

    where = {"source": source_filter} if source_filter else None

    results = collection.query(
        query_embeddings = query_embedding,
        n_results        = top_k,
        where            = where,
        include          = ["documents", "metadatas", "distances"]
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        doc      = results["documents"][0][i]
        meta     = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        score    = round(1 - distance, 4)   # cosine distance → similarity

        chunks.append(RetrievedChunk(
            text   = doc,
            page   = meta.get("page", 0),
            source = meta.get("source", "unknown"),
            score  = score
        ))

    log.info(f"Retrieved {len(chunks)} chunks — "
             f"top score: {chunks[0].score if chunks else 0}")
    return chunks


def format_context(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks into a single context string for the LLM."""
    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(
            f"[Source: {chunk.source} | Page {chunk.page} | "
            f"Relevance: {chunk.score}]\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    from app.ingestor import load_embedding_model
    model = load_embedding_model()

    query   = "What is the main topic of this paper?"
    results = retrieve(query, model, top_k=3)

    print(f"\nQuery: {query}")
    print(f"Found {len(results)} chunks\n")
    for r in results:
        print(f"Page {r.page} | Score {r.score}")
        print(f"{r.text[:200]}...")
        print()