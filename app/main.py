import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

import streamlit as st
import logging
from app.ingestor import (load_embedding_model, ingest_pdf,
                          list_ingested_docs, delete_doc)
from app.qa import answer
from app.voice import speak

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")

st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("RAG Research Assistant")
st.caption("Upload a PDF — ask questions — get cited answers from the document")

@st.cache_resource
def get_model():
    return load_embedding_model()

model = get_model()

with st.sidebar:
    st.markdown("### Upload PDF")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded:
        save_path = f"data/{uploaded.name}"
        os.makedirs("data", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())

        with st.spinner("Indexing PDF..."):
            result = ingest_pdf(save_path, model)

        st.success(
            f"Indexed: {result['source']}\n"
            f"{result['pages']} pages · {result['chunks']} chunks"
        )
        st.cache_data.clear()

    st.divider()
    st.markdown("### Indexed documents")
    docs = list_ingested_docs()
    if docs:
        for doc in docs:
            col1, col2 = st.columns([3, 1])
            col1.write(doc)
            if col2.button("Del", key=f"del_{doc}"):
                deleted = delete_doc(doc)
                st.success(f"Removed {deleted} chunks")
                st.rerun()
    else:
        st.info("No documents indexed yet.")

    st.divider()
    selected_doc = st.selectbox(
        "Filter by document",
        options=["All documents"] + docs
    )
    top_k      = st.slider("Chunks to retrieve", 2, 8, 4)
    read_aloud = st.toggle("Read answer aloud", value=True)

    st.divider()
    st.markdown("### Try these questions")
    sample_questions = [
        "What is the main contribution of this paper?",
        "What problem does this research solve?",
        "What are the key findings and results?",
        "What methods or algorithms are used?",
        "What are the limitations of this work?",
        "What future work is suggested?",
        "Who are the authors of this paper?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True, key=f"sample_{q[:20]}"):
            st.session_state.pending_q = q

if "history"   not in st.session_state:
    st.session_state.history   = []
if "pending_q" not in st.session_state:
    st.session_state.pending_q = ""

question = st.text_input(
    "Ask a question about your documents:",
    placeholder="What is the main contribution of this paper?"
)

col1, col2 = st.columns([1, 5])
with col1:
    ask = st.button("Ask", type="primary")
with col2:
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

if st.session_state.pending_q:
    question = st.session_state.pending_q
    st.session_state.pending_q = ""
    ask = True

source_filter = None if selected_doc == "All documents" else selected_doc

if ask and question.strip():
    with st.spinner("Retrieving and generating answer..."):
        result = answer(
            question.strip(), model,
            top_k=top_k,
            source_filter=source_filter
        )
        audio_path = speak(result["answer"]) if read_aloud else None

    st.session_state.history.append({
        "question":   result["question"],
        "answer":     result["answer"],
        "chunks":     result["chunks"],
        "audio_path": audio_path
    })

for item in reversed(st.session_state.history):
    with st.expander(f"Q: {item['question']}", expanded=True):
        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown("#### Answer")
            st.markdown(item["answer"])
            if item.get("audio_path"):
                st.audio(item["audio_path"])

        with col2:
            st.markdown("#### Source chunks")
            for chunk in item["chunks"]:
                st.markdown(
                    f"**Page {chunk.page}** — "
                    f"score `{chunk.score}` — "
                    f"*{chunk.source}*"
                )
                st.caption(chunk.text[:300] + "...")
                st.divider()