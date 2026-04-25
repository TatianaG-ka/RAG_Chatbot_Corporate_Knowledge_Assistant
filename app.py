import os
import tempfile
from pathlib import Path
from typing import Any, List

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

from rag_index import (
    build_embeddings,
    build_faiss_from_docs,
    load_paths,
)

# Supported suffixes for the demo index — kept in sync with build_demo_index.py.
_DEMO_ASSET_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "")
GROQ_ENV = os.getenv("GROQ_API_KEY", "")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

st.set_page_config(page_title="RAG + ChatGroq Demo", layout="wide")
st.title("Company Knowledge Assistant - RAG (PDF/TXT/MD) Demo - ChatGroq + FAISS + Citations")

# --- sidebar -------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    workspace = st.text_input("Workspace (company/project name)", value="default_company")
    persist_dir = Path("vectorstore") / workspace

    groq_api_key = st.text_input("GROQ_API_KEY", value=GROQ_ENV, type="password")
    model_name = st.selectbox(
        "Groq model",
        options=[
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
        ],
        index=0,
    )
    score_threshold = st.slider(
        "Retrieval score threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
        help="Lower = more permissive retrieval; higher = stricter 'I don't know' behaviour.",
    )

    st.caption("Embeddings: sentence-transformers/all-MiniLM-L6-v2")
    st.markdown("---")
    st.caption("Quick demo uses a pre-built index (from the repo). Upload builds the index in session memory.")


# --- cached factories (survive re-renders) ------------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return build_embeddings()


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str, model: str) -> ChatGroq:
    return ChatGroq(groq_api_key=api_key, model_name=model)


@st.cache_resource(show_spinner="Building demo index from ./assets…")
def get_demo_index(assets_dir: str, _emb):
    """Rebuild the demo FAISS index from ./assets on every cold start.

    We intentionally do not ship a pickled index in the repo. Pickle state
    drifts across major dependency versions (e.g. pydantic v1 -> v2), and a
    hosting-platform runtime bump silently breaks a previously-working index
    with cryptic errors like ``KeyError: '__fields_set__'``. Rebuilding from
    raw PDF/TXT/MD on cold start trades ~15-30s of first-load latency for
    permanent compatibility across container upgrades.
    """
    asset_paths = [
        p for p in Path(assets_dir).glob("*")
        if p.suffix.lower() in _DEMO_ASSET_SUFFIXES
    ]
    if not asset_paths:
        return None, 0, [f"No supported files in {assets_dir}/ (expected .pdf/.txt/.md)."]
    docs, errors = load_paths(asset_paths)
    if not docs:
        return None, 0, errors
    vs, n_chunks = build_faiss_from_docs(docs, _emb)
    return vs, n_chunks, errors


# --- helpers -------------------------------------------------------------
def _safe_filename(name: str) -> str:
    """Strip path components to prevent traversal (e.g. '../../app.py')."""
    return Path(name).name


def _format_citations(context_docs: List[Document]) -> List[str]:
    seen = set()
    out = []
    for d in context_docs:
        src = d.metadata.get("source") or d.metadata.get("file_path") or "unknown"
        page = d.metadata.get("page")
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        name = Path(src).name
        out.append(f"- {name}, page {page + 1}" if page is not None else f"- {name}")
    return out


def _build_index_from_uploads(files: List[Any], emb):
    """Build FAISS from session uploads using a per-call temp dir.

    Files live only long enough for the document loaders to read them
    into memory; the ``TemporaryDirectory`` cleans up automatically
    once the with-block exits. Avoids a hardcoded ``tmp/`` collision
    across concurrent Space sessions and leaves no upload bytes on
    disk between rebuilds.

    Path-traversal protection (``_safe_filename``) and ``OSError``
    handling are preserved from the previous on-disk helper.
    """
    if not files:
        st.warning("No files to load.")
        return None, 0
    with tempfile.TemporaryDirectory(prefix="rag_uploads_") as tmp:
        tmp_dir = Path(tmp)
        paths: List[Path] = []
        for up in files:
            safe_name = _safe_filename(up.name)
            if not safe_name:
                st.error(f"Rejected upload with invalid name: {up.name!r}")
                continue
            target = tmp_dir / safe_name
            try:
                target.write_bytes(up.getbuffer())
                paths.append(target)
            except OSError as e:
                st.error(f"Write error {safe_name}: {e}")
        if not paths:
            st.warning("No files to load.")
            return None, 0
        # Loaders must finish reading before we drop out of the
        # with-block — they hold no file handles across the call,
        # but the file itself disappears on cleanup.
        docs, errors = load_paths(paths)
    for err in errors:
        st.warning(err)
    if not docs:
        st.warning("Failed to load documents (check formats).")
        return None, 0
    vs, n_chunks = build_faiss_from_docs(docs, emb)
    return vs, n_chunks


# --- mode & index configuration ----------------------------------------
MODE_QUICK = "Quick demo (prebuilt)"
MODE_UPLOAD = "Upload files (sessionally)"

mode = st.radio("Mode:", [MODE_QUICK, MODE_UPLOAD], horizontal=True)
embeddings = get_embeddings()

VS_KEY = "vs"
VS_USER_KEY = "vs_user"

if MODE_QUICK == mode:
    vs, n_chunks, load_errors = get_demo_index("assets", embeddings)
    for msg in load_errors:
        st.warning(msg)
    if vs is None:
        st.error("No demo documents found in ./assets/ (expected .pdf/.txt/.md).")
        st.stop()
    st.session_state[VS_KEY] = vs
    st.success(f"Demo index built from ./assets/ ({n_chunks} chunks).")

    cols = st.columns(3)
    examples = [
        "How long does a refund take?",
        "How to reset my password?",
        "How to apply a software update?",
    ]
    for i, ex in enumerate(examples):
        if cols[i % 3].button(ex):
            st.session_state["query"] = ex
else:
    uploads = st.file_uploader(
        "Upload documents (PDF/TXT/MD)",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
        help="Supported: PDF/TXT/MD (MD requires the unstructured package).",
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Build an index from my files"):
            with st.spinner("Building an index..."):
                vs_user, n_chunks = _build_index_from_uploads(uploads or [], embeddings)
                if vs_user:
                    st.session_state[VS_USER_KEY] = vs_user
                    st.session_state[VS_KEY] = vs_user
                    st.success(f"Index ready ({n_chunks} chunks).")
    with c2:
        if st.button("Index reset (session)"):
            st.session_state.pop(VS_USER_KEY, None)
            st.session_state.pop(VS_KEY, None)
            st.info("Session index cleared.")


# --- chat (LLM + retrieval) --------------------------------------------
st.subheader("Chat")
session_id = st.text_input("Session ID", value="default_session")
query = st.text_input("Your question:", value=st.session_state.get("query", ""))

if "stores" not in st.session_state:
    st.session_state.stores = {}


def _get_session_history(sid: str) -> BaseChatMessageHistory:
    if sid not in st.session_state.stores:
        st.session_state.stores[sid] = ChatMessageHistory()
    return st.session_state.stores[sid]


if not groq_api_key:
    st.info("Enter GROQ_API_KEY in the sidebar to chat.")
    st.stop()

llm = get_llm(groq_api_key, model_name)

contextualize_q_system_prompt = (
    "Given the chat history and the latest user input, rewrite it as a standalone question "
    "that can be understood without the chat history. Do NOT answer the question."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_system_prompt = (
    "You are a QA assistant answering questions based on the company's knowledge base.\n"
    "Use ONLY the provided context. If the answer is not in the context, say 'I don't know'.\n"
    "At the end, append a 'Citations' section listing unique file names (and pages if available).\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def _get_retriever(threshold: float):
    store = st.session_state.get(VS_KEY)
    if store is None:
        st.warning("Index not loaded. Use 'Quick demo (prebuilt)' or build index from uploads.")
        st.stop()
    # similarity_score_threshold keeps the retriever honest: when nothing in the
    # index is close enough to the query, it returns [] and the LLM answers
    # "I don't know" instead of hallucinating over unrelated chunks.
    return store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 4, "score_threshold": threshold},
    )


if st.button("Send") and query.strip():
    retriever = _get_retriever(score_threshold)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    doc_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, doc_chain)

    conv = RunnableWithMessageHistory(
        rag_chain,
        _get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    cfg = {"configurable": {"session_id": session_id}}

    with st.spinner("Thinking..."):
        result = conv.invoke({"input": query}, config=cfg)

    answer = result.get("answer") or result.get("result") or ""
    ctx_docs: List[Document] = result.get("context", [])
    citations = _format_citations(ctx_docs)

    st.markdown("### Answer")
    st.write(answer)

    st.markdown("### Citations")
    if citations:
        st.write("\n".join(citations))
    else:
        st.write("No citations (retriever returned nothing above the score threshold).")

    with st.expander("Debug: context (top-k)"):
        for i, d in enumerate(ctx_docs, 1):
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page")
            head = f"[{i}] {Path(src).name}"
            if page is not None:
                head += f" (page {page + 1})"
            st.markdown(f"**{head}**")
            snippet = d.page_content.strip()
            st.write(snippet[:800] + ("..." if len(snippet) > 800 else ""))

with st.expander("Info / Limits"):
    st.markdown(
        "- Quick demo: uses prebuilt FAISS (from repo)\n"
        "- Upload: index created in session memory (not saved to disk)\n"
        "- FAISS load uses `allow_dangerous_deserialization=True` (required by LangChain); "
        "this is safe only because the index is generated by this app.\n"
        "- LLM: ChatGroq (selectable in sidebar)\n"
        "- Embeddings: sentence-transformers/all-MiniLM-L6-v2\n"
        "- Retrieval: similarity_score_threshold (k=4). Threshold adjustable in sidebar."
    )
