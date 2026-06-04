import os
import tempfile
import time
from pathlib import Path
from typing import Any, List, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from rag_index import (
    EMB_MODEL,
    ScoredChunk,
    build_embeddings,
    build_faiss_from_docs,
    load_paths,
    retrieve_scored,
    select_context,
)

# Retrieval breadth. RETRIEVAL_K chunks are scored for the debug panel (so we can
# show what fell below the threshold); at most CONTEXT_K passing chunks are fed to
# the LLM. MAX_PER_DOC caps how many chunks a single source contributes.
#
# Tuned with the 600/120 chunking (see rag_index.CHUNK_SIZE). Smaller chunks mean a
# single fact spans fewer chars, so the answer-bearing chunk often sits deeper in the
# ranking and a fact-rich source contributes several near-tied chunks. We measured
# all 5 demo questions: the "minimalna kwota" answer is the 4th-best chunk *within its
# own PDF* (needs MAX_PER_DOC>=4), and the de minimis "120 mies." chunk ranks ~8th
# overall behind FENG chunks (needs CONTEXT_K>=8 and RETRIEVAL_K>=12 to reach it).
# These wider budgets surface every demo fact while the 0.35 threshold still rejects
# out-of-corpus queries. The old 8/4/2 was tuned for 1200-char chunks and starved
# single-document questions once chunks shrank.
RETRIEVAL_K = 12
CONTEXT_K = 8
MAX_PER_DOC = 4

# Supported suffixes for the demo index — kept in sync with build_demo_index.py.
_DEMO_ASSET_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "")
GROQ_ENV = os.getenv("GROQ_API_KEY", "")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

st.set_page_config(page_title="Asystent Wiedzy BGK — RAG demo", layout="wide")
st.title("Asystent Wiedzy BGK — RAG z cytowaniem źródeł (demo)")

# --- sidebar -------------------------------------------------------------
with st.sidebar:
    st.header("Ustawienia")

    workspace = st.text_input("Przestrzeń robocza (nazwa firmy/projektu)", value="default_company")
    persist_dir = Path("vectorstore") / workspace

    # When the key is provided via env/secret (e.g. an HF Space secret), don't show
    # an input — one less thing to fiddle with on stage. Only prompt when it's missing.
    if GROQ_ENV:
        groq_api_key = GROQ_ENV
        st.caption("🔑 GROQ_API_KEY: z konfiguracji serwera")
    else:
        groq_api_key = st.text_input("GROQ_API_KEY", value="", type="password")
    model_name = st.selectbox(
        "Model Groq",
        options=[
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
        ],
        index=0,
    )
    score_threshold = st.slider(
        "Próg trafności",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
        help="Niżej = bardziej liberalny retrieval; wyżej = surowsze „Nie wiem” (więcej odmów).",
    )

    st.caption(f"Embeddingi: {EMB_MODEL}")
    st.markdown("---")
    st.caption("Tryb „Quick demo” buduje indeks z dokumentów w repo. Tryb „Wgraj pliki” buduje indeks w pamięci sesji.")


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
        return None, 0, [f"Brak obsługiwanych plików w {assets_dir}/ (oczekiwane .pdf/.txt/.md)."]
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
        out.append(f"- {name}, strona {page + 1}" if page is not None else f"- {name}")
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
        st.warning("Brak plików do wczytania.")
        return None, 0
    with tempfile.TemporaryDirectory(prefix="rag_uploads_") as tmp:
        tmp_dir = Path(tmp)
        paths: List[Path] = []
        for up in files:
            safe_name = _safe_filename(up.name)
            if not safe_name:
                st.error(f"Odrzucono plik o nieprawidłowej nazwie: {up.name!r}")
                continue
            target = tmp_dir / safe_name
            try:
                target.write_bytes(up.getbuffer())
                paths.append(target)
            except OSError as e:
                st.error(f"Błąd zapisu {safe_name}: {e}")
        if not paths:
            st.warning("Brak plików do wczytania.")
            return None, 0
        # Loaders must finish reading before we drop out of the
        # with-block — they hold no file handles across the call,
        # but the file itself disappears on cleanup.
        docs, errors = load_paths(paths)
    for err in errors:
        st.warning(err)
    if not docs:
        st.warning("Nie udało się wczytać dokumentów (sprawdź format).")
        return None, 0
    vs, n_chunks = build_faiss_from_docs(docs, emb)
    return vs, n_chunks


# --- mode & index configuration ----------------------------------------
MODE_QUICK = "Quick demo (gotowy indeks)"
MODE_UPLOAD = "Wgraj pliki (sesyjnie)"

# Demo questions rehearsed for the BGK presentation (docs/DEMO_RAG_dokumenty_BGK.md).
# Q5 is the deliberate refuse-on-no-context trap — kept neutral so it reads like a
# normal question; the system must answer „Nie wiem” because the corpus is for MŚP,
# not consumer mortgages.
DEMO_QUESTIONS = [
    "Jaka jest minimalna kwota Pożyczki na cyfryzację i kto może wnioskować?",
    "Do jakiej części kredytu sięga gwarancja de minimis?",
    "Co finansuje gwarancja Biznesmax, a co gwarancja Ekomax?",
    "Jaki jest okres gwarancji dla kredytu inwestycyjnego de minimis?",
    "Czy gwarancja de minimis obejmuje kredyt hipoteczny dla osoby fizycznej?",
]

mode = st.radio("Tryb:", [MODE_QUICK, MODE_UPLOAD], horizontal=True)
embeddings = get_embeddings()

VS_KEY = "vs"
VS_USER_KEY = "vs_user"

if MODE_QUICK == mode:
    vs, n_chunks, load_errors = get_demo_index("assets", embeddings)
    for msg in load_errors:
        st.warning(msg)
    if vs is None:
        st.error("Brak dokumentów demo w ./assets/ (oczekiwane .pdf/.txt/.md).")
        st.stop()
    st.session_state[VS_KEY] = vs
    st.success(f"Indeks demo zbudowany z ./assets/ ({n_chunks} fragmentów).")

    st.caption("Przykładowe pytania demo:")
    for i, q in enumerate(DEMO_QUESTIONS):
        if st.button(q, key=f"demo_q_{i}", use_container_width=True):
            st.session_state["query"] = q
else:
    uploads = st.file_uploader(
        "Wgraj dokumenty (PDF/TXT/MD)",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
        help="Obsługiwane: PDF/TXT/MD (MD wymaga pakietu „unstructured”).",
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Zbuduj indeks z moich plików"):
            with st.spinner("Buduję indeks…"):
                vs_user, n_chunks = _build_index_from_uploads(uploads or [], embeddings)
                if vs_user:
                    st.session_state[VS_USER_KEY] = vs_user
                    st.session_state[VS_KEY] = vs_user
                    st.success(f"Indeks gotowy ({n_chunks} fragmentów).")
    with c2:
        if st.button("Wyczyść indeks (sesja)"):
            st.session_state.pop(VS_USER_KEY, None)
            st.session_state.pop(VS_KEY, None)
            st.info("Indeks sesyjny wyczyszczony.")


# --- chat (LLM + retrieval) --------------------------------------------
st.subheader("Czat")
# Fixed session id — chat memory (multi-turn) still works, but a non-technical
# reviewer shouldn't have to see or fill a "session id" field during the demo.
session_id = "default_session"
query = st.text_input("Twoje pytanie:", value=st.session_state.get("query", ""))

if "stores" not in st.session_state:
    st.session_state.stores = {}


def _get_session_history(sid: str) -> BaseChatMessageHistory:
    if sid not in st.session_state.stores:
        st.session_state.stores[sid] = ChatMessageHistory()
    return st.session_state.stores[sid]


if not groq_api_key:
    st.info("Wpisz GROQ_API_KEY w panelu bocznym, aby rozmawiać.")
    st.stop()

llm = get_llm(groq_api_key, model_name)

contextualize_q_system_prompt = (
    "Na podstawie historii rozmowy i ostatniego pytania użytkownika przeformułuj je "
    "w samodzielne pytanie, zrozumiałe bez znajomości historii rozmowy. "
    "NIE odpowiadaj na pytanie — tylko je przeformułuj."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_system_prompt = (
    "Jesteś asystentem, który odpowiada na pytania na podstawie bazy wiedzy BGK "
    "(publiczne dokumenty Banku Gospodarstwa Krajowego).\n"
    "Korzystaj WYŁĄCZNIE z podanego poniżej kontekstu. Jeśli odpowiedzi nie ma w kontekście, "
    "napisz dokładnie: „Nie wiem — brak podstawy w dokumentach.” i nie dodawaj nic więcej.\n"
    "Nie zgaduj i nie korzystaj z wiedzy spoza kontekstu. Odpowiadaj po polsku.\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def _render_debug(
    original: str,
    rewritten: str,
    scored: List[ScoredChunk],
    used: List[Document],
    threshold: float,
    model: str,
    rewrite_ms: int,
    retrieval_ms: int,
    llm_ms: Optional[int],
) -> None:
    """Make the retrieval pipeline observable: rewritten query, per-chunk scores
    (relevance vs raw L2), which chunks actually reached the LLM, and stage
    latency. Marks distinguish chunks fed to the LLM (✅) from ones that cleared
    the threshold but were dropped by the per-document cap (➖) and ones below the
    threshold (✗) — so the panel matches exactly what the LLM received."""
    used_ids = {id(d) for d in used}
    with st.expander("🔎 Debug: retrieval i pipeline"):
        st.markdown(f"**Zapytanie użytkownika:** {original}")
        if rewritten != original:
            st.markdown(f"**Przeformułowane (history-aware):** {rewritten}")
        else:
            st.caption("Brak historii rozmowy — zapytanie nieprzeformułowane.")

        timings = []
        if rewrite_ms:
            timings.append(f"rewrite {rewrite_ms} ms")
        timings.append(f"retrieval {retrieval_ms} ms")
        if llm_ms is not None:
            timings.append(f"LLM {llm_ms} ms")
        st.markdown(
            f"**Próg trafności:** {threshold:.2f}  ·  **Model:** {model}  ·  "
            f"**Czas:** {' · '.join(timings)}"
        )

        st.markdown(
            f"**Top-{len(scored)} fragmenty** — ✅ trafia do LLM · "
            f"➖ powyżej progu, odcięte limitem {MAX_PER_DOC}/dokument · ✗ poniżej progu:"
        )
        for i, sc in enumerate(scored, 1):
            src = Path(sc.doc.metadata.get("source", "unknown")).name
            page = sc.doc.metadata.get("page")
            loc = f", str. {page + 1}" if page is not None else ""
            if id(sc.doc) in used_ids:
                mark = "✅"
            elif sc.relevance >= threshold:
                mark = "➖"
            else:
                mark = "✗"
            st.markdown(
                f"{mark} **[{i}]** relevance=`{sc.relevance:.3f}`  ·  "
                f"L2=`{sc.distance:.2f}`  ·  {src}{loc}"
            )
            snippet = sc.doc.page_content.strip()
            st.caption(snippet[:300] + ("…" if len(snippet) > 300 else ""))

        capped = [
            sc for sc in scored
            if sc.relevance >= threshold and id(sc.doc) not in used_ids
        ]
        if capped:
            st.caption(
                f"➖ {len(capped)} fragment(ów) przekroczyło próg, ale zostało "
                f"odciętych limitem {MAX_PER_DOC}/dokument (dywersyfikacja źródeł — "
                "by jeden duży dokument nie zdominował kontekstu)."
            )


if st.button("Wyślij") and query.strip():
    store = st.session_state.get(VS_KEY)
    if store is None:
        st.warning("Indeks nie został wczytany. Użyj trybu „Quick demo” lub zbuduj indeks z plików.")
    else:
        history = _get_session_history(session_id)
        # Snapshot the prior turns: history.messages is a live reference, and we
        # append this turn's user+AI messages below — copy so what we send to the
        # LLM can never include the turn it is currently answering.
        hist_msgs = list(history.messages)

        # 1. History-aware rewrite — only when there is prior context to fold in.
        if hist_msgs:
            rw_t0 = time.perf_counter()
            rewritten = (
                (contextualize_q_prompt | llm | StrOutputParser())
                .invoke({"input": query, "chat_history": hist_msgs})
                .strip()
            )
            rewrite_ms = round((time.perf_counter() - rw_t0) * 1000)
        else:
            rewritten = query
            rewrite_ms = 0

        # 2. Manual retrieval with scores, then honest threshold filter.
        ret_t0 = time.perf_counter()
        scored = retrieve_scored(store, rewritten, k=RETRIEVAL_K)
        retrieval_ms = round((time.perf_counter() - ret_t0) * 1000)
        used = select_context(scored, score_threshold, CONTEXT_K, MAX_PER_DOC)

        if not used:
            # 3. Nothing cleared the threshold → refuse honestly, skip the LLM.
            answer = "Nie wiem — brak podstawy w dokumentach."
            history.add_user_message(query)
            history.add_ai_message(answer)
            st.markdown("### Odpowiedź")
            st.warning(answer)
            st.caption(
                "Żaden fragment nie przekroczył progu trafności — "
                "odpowiedź wstrzymana bez wywołania LLM (honest refusal)."
            )
            st.markdown("### Źródła")
            st.write("Brak źródeł powyżej progu trafności.")
            _render_debug(
                query, rewritten, scored, used, score_threshold, model_name,
                rewrite_ms, retrieval_ms, None,
            )
        else:
            # 4. Stuff the passing chunks and call the LLM (history-aware).
            context = "\n\n".join(d.page_content for d in used)
            llm_t0 = time.perf_counter()
            with st.spinner("Myślę…"):
                answer = (qa_prompt | llm).invoke(
                    {"context": context, "chat_history": hist_msgs, "input": query}
                ).content
            llm_ms = round((time.perf_counter() - llm_t0) * 1000)

            history.add_user_message(query)
            history.add_ai_message(answer)

            # 5. Answer + deterministic citations (from used chunks) + debug panel.
            st.markdown("### Odpowiedź")
            st.write(answer)

            citations = _format_citations(used)
            st.markdown("### Źródła")
            st.write("\n".join(citations) if citations else "Brak źródeł.")

            _render_debug(
                query, rewritten, scored, used, score_threshold, model_name,
                rewrite_ms, retrieval_ms, llm_ms,
            )

with st.expander("Informacje / Ograniczenia"):
    st.markdown(
        "- Quick demo: indeks budowany w pamięci przy starcie z `./assets/` (cache na czas życia kontenera)\n"
        "- Wgrywanie: indeks tworzony w pamięci sesji (nie zapisywany na dysk)\n"
        "- Brak deserializacji pickle w runtime — uzasadnienie w ADR-4 w README\n"
        "- LLM: ChatGroq (wybierany w panelu bocznym)\n"
        f"- Embeddingi: {EMB_MODEL}\n"
        f"- Retrieval: ręczny pipeline — top-{RETRIEVAL_K} fragmentów ze score'ami, "
        f"filtr progu, max {CONTEXT_K} do LLM (max {MAX_PER_DOC}/dokument). "
        "Próg regulowany w panelu bocznym; szczegóły w „Debug: retrieval i pipeline”."
    )
