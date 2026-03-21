import fitz  # pymupdf
import os
import logging
from dataclasses import dataclass, field
from typing import List
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

EMBED_MODEL  = "all-MiniLM-L6-v2"
CHROMA_PATH  = "vectorstore"
CHUNK_SIZE   = 500
CHUNK_OVERLAP = 50


@dataclass
class Chunk:
    text:     str
    page:     int
    chunk_id: str
    source:   str


def load_embedding_model():
    log.info(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    log.info("Embedding model loaded")
    return model


def extract_pages(pdf_path: str) -> List[dict]:
    """Extract text from every page of a PDF."""
    log.info(f"Extracting text from {pdf_path}")
    doc    = fitz.open(pdf_path)
    pages  = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    log.info(f"Extracted {len(pages)} pages")
    return pages


def chunk_pages(pages: List[dict], source: str) -> List[Chunk]:
    """Split pages into overlapping chunks."""
    chunks = []
    for page_data in pages:
        text     = page_data["text"]
        page_num = page_data["page"]
        words    = text.split()

        start = 0
        idx   = 0
        while start < len(words):
            end        = min(start + CHUNK_SIZE, len(words))
            chunk_text = " ".join(words[start:end])
            chunk_id   = f"{source}_p{page_num}_c{idx}"

            chunks.append(Chunk(
                text=chunk_text,
                page=page_num,
                chunk_id=chunk_id,
                source=source
            ))
            start += CHUNK_SIZE - CHUNK_OVERLAP
            idx   += 1

    log.info(f"Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks


def get_chroma_collection(collection_name: str = "research_docs"):
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )


def ingest_pdf(pdf_path: str, model: SentenceTransformer,
               collection_name: str = "research_docs") -> dict:
    """Full ingestion pipeline — PDF to ChromaDB."""
    source   = os.path.basename(pdf_path).replace(".pdf", "")
    pages    = extract_pages(pdf_path)
    chunks   = chunk_pages(pages, source)

    log.info(f"Embedding {len(chunks)} chunks...")
    texts      = [c.text     for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection = get_chroma_collection(collection_name)

    # remove old chunks from same source before re-ingesting
    try:
        existing = collection.get(where={"source": source})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            log.info(f"Removed {len(existing['ids'])} old chunks for {source}")
    except Exception:
        pass

    collection.add(
        ids        = [c.chunk_id for c in chunks],
        documents  = texts,
        embeddings = embeddings,
        metadatas  = [{"page": c.page, "source": c.source} for c in chunks]
    )

    log.info(f"Ingested {len(chunks)} chunks into ChromaDB")
    return {
        "source":  source,
        "pages":   len(pages),
        "chunks":  len(chunks),
        "path":    pdf_path
    }


def list_ingested_docs(collection_name: str = "research_docs") -> List[str]:
    """Return list of unique source docs in ChromaDB."""
    try:
        collection = get_chroma_collection(collection_name)
        results    = collection.get()
        sources    = list(set(
            m["source"] for m in results["metadatas"]
        )) if results["metadatas"] else []
        return sources
    except Exception:
        return []


def delete_doc(source: str,
               collection_name: str = "research_docs") -> int:
    """Remove all chunks for a given source document."""
    collection = get_chroma_collection(collection_name)
    existing   = collection.get(where={"source": source})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        log.info(f"Deleted {len(existing['ids'])} chunks for {source}")
        return len(existing["ids"])
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    model = load_embedding_model()
    print("Ingestor ready. Call ingest_pdf(path, model) to index a PDF.")
    print(f"Embed model : {EMBED_MODEL}")
    print(f"Chunk size  : {CHUNK_SIZE} words")
    print(f"Overlap     : {CHUNK_OVERLAP} words")
    print(f"Vector store: {CHROMA_PATH}/")