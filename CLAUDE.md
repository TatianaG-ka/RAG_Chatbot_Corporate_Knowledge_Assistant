# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this is

A **Retrieval-Augmented Generation (RAG) chatbot** that answers questions about a
document corpus **with source citations** and **refuses to answer when the corpus
doesn't support the question** (no hallucination). Streamlit UI, deployed to
Hugging Face Spaces (free tier).

Stack: **Streamlit + LangChain + FAISS (in-memory) + HuggingFace embeddings + ChatGroq (LLM)**.

Live Space: `TatianaGol/RAG_Chatbot-Corporate_Knowledge_Assistant`
(synced automatically from `main` via `.github/workflows/main.yml`).

## Current goal — BGK pivot (recruitment demo)

This repo is the **flagship portfolio project** for an application to
**"Menedżer ds. AI" at Bank Gospodarstwa Krajowego (BGK)**. The job posting is in
`docs/ogloszemia_stanowiska/`.

We are pivoting the demo from a generic English corpus to an **"Asystent Wiedzy BGK"**
built on **public BGK documents** (bgk.pl). The spec is in
`docs/DEMO_RAG_dokumenty_BGK.md` — read it before touching the corpus or demo questions.

Why this pivot maps to the job:
- **AI Act / RODO awareness** → the refuse-on-no-context rule = compliance-grade, trustworthy AI for a regulated bank.
- **Answering supervisory bodies (KNF, UODO)** → citations + debug panel = auditability.
- **Translating business needs to tech** → the whole demo, presented simply for non-technical reviewers.

The earlier (now superseded) plan — a generic synthetic corpus — is archived in
`docs/wczesniej/`. The BGK corpus replaces it.

## Commands

```powershell
# Install
pip install -r requirements.txt

# Run the app locally (Quick demo mode rebuilds the index from ./assets at startup)
streamlit run app.py

# Build a persistent FAISS index to ./vectorstore/default_company (optional; app rebuilds in-memory anyway)
python build_demo_index.py

# CI smoke test (what GitHub Actions runs before syncing to HF Space)
python -m compileall -q app.py rag_index.py build_demo_index.py
python -c "import rag_index; import build_demo_index; print('imports ok')"
```

Requires `GROQ_API_KEY` (entered in the sidebar or via `.env`). Embeddings run locally (CPU), no key needed.

## Architecture

- **`rag_index.py`** — pure RAG plumbing (no Streamlit). Document loading (PDF/TXT/MD),
  chunking (`RecursiveCharacterTextSplitter`, size 1200 / overlap 200), FAISS build/save/load,
  embeddings factory. Reusable, unit-testable.
- **`app.py`** — Streamlit UI + the LangChain retrieval chain
  (`create_history_aware_retriever` → `create_stuff_documents_chain` →
  `create_retrieval_chain`, wrapped in `RunnableWithMessageHistory` for chat memory).
- **`build_demo_index.py`** — offline index builder (CLI).
- **`assets/`** — the demo corpus. Currently generic English (`policy.md`, `faq.txt`,
  `manual.pdf`) — to be replaced with BGK PDFs.
- **`vectorstore/default_company/`** — gitignored; rebuilt at cold start.

### Key design decisions (see README ADRs)
- **No pickled index shipped.** The index is rebuilt in-memory from `./assets/` on every
  cold start (`get_demo_index`, cached for container lifetime). Pickle drifts across major
  dep versions and silently breaks on platform runtime bumps. ~15–30s first-load latency is
  the deliberate trade for permanent compatibility.
- **Honest retrieval.** Retriever uses `similarity_score_threshold` (k=4, threshold from
  sidebar slider, default 0.35). When nothing clears the threshold it returns `[]` and the
  LLM answers "I don't know" instead of hallucinating.
- **Uploads never trust disk pickles** — rebuilt in-memory from a `TemporaryDirectory`.

## Gotchas / things to verify

- **Embedding model (resolved 2026-06-04).** Was English-only `all-MiniLM-L6-v2`; switched to
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` + `normalize_embeddings=True`
  (`rag_index.py:EMB_MODEL`) because the corpus is Polish. Decision is data-driven (`_diag_step0.py`):
  e5-base ranks slightly better but compresses relevance scores to ~0.65–0.83 so the threshold can't
  reject out-of-corpus queries ("stolica Mongolii" → 0.72); paraphrase-multilingual gives clean
  separation (out-of-corpus relevance ~0), which the honest-refusal demo relies on.
- **PDFs must be text-based, not scanned.** `PyPDFLoader` extracts no text from scanned/image
  PDFs (no OCR in the stack). Verify each downloaded BGK PDF actually yields text.
- **`qa_system_prompt` is English** (`app.py`). For a Polish demo, switch the prompt (and
  the "I don't know" wording) to Polish so answers and the refusal read naturally.
- **de minimis: 60% vs 80%.** The demo doc warns the BGK mockup slide says "80%" but de minimis
  is **60%** (80% is Biznesmax/Ekomax). Make sure the corpus + live answers say 60%.
- **HF Space deploy.** Pushing to `main` force-pushes to the Space. Expect 5–15 min rebuild;
  watch for pickle/runtime incompatibilities. Don't deploy untested.

## Conventions

- Code and README are in **English**; the BGK demo corpus and demo questions are in **Polish**.
- Keep `rag_index.py` free of Streamlit imports (testability boundary).
- `_DEMO_ASSET_SUFFIXES` in `app.py` and `SUPPORTED_SUFFIXES` in `build_demo_index.py`
  must stay in sync.
- Don't commit `vectorstore/` or upload bytes; don't ship a pickled index.
