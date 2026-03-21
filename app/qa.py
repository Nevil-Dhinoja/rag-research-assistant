import logging
import os
from typing import List
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from app.retriever import RetrievedChunk, retrieve, format_context
from sentence_transformers import SentenceTransformer

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
load_dotenv()
log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research assistant that answers questions
strictly based on the provided document context.

Rules:
- Answer ONLY from the provided context — never from general knowledge
- Always cite the page number like [Page X] after each key point
- If the context does not contain enough information, say exactly:
  "The document does not contain enough information to answer this."
- Keep answers clear, structured, and concise
- Do not make up facts or infer beyond what the context says"""


def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
        max_tokens=1024
    )


def answer(question: str,
           model: SentenceTransformer,
           top_k: int = 4,
           source_filter: str = None) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks
    2. Format as context
    3. Ask Groq to answer from context only
    4. Return answer + sources
    """
    chunks  = retrieve(question, model,
                       top_k=top_k,
                       source_filter=source_filter)
    context = format_context(chunks)

    prompt = f"""Context from the document:

{context}

---

Question: {question}

Answer based strictly on the context above.
Cite page numbers like [Page X] for every key claim."""

    try:
        llm      = get_llm()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ]
        response = llm.invoke(messages)
        answer_text = response.content.strip()
    except Exception as e:
        log.error(f"LLM call failed: {e}")
        answer_text = f"Error generating answer: {e}"

    return {
        "question": question,
        "answer":   answer_text,
        "chunks":   chunks,
        "context":  context
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    from app.ingestor import load_embedding_model
    model = load_embedding_model()

    questions = [
        "What is the main topic of this paper?",
        "What problem does MAML solve?",
        "What are the key findings or results?"
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = answer(q, model)
        print(f"A: {result['answer']}")
        print(f"\nSources:")
        for chunk in result['chunks']:
            print(f"  Page {chunk.page} | Score {chunk.score} | "
                  f"{chunk.text[:80]}...")