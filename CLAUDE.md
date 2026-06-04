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
  embeddings factory, and the scored-retrieval helpers `retrieve_scored()` (top-k with 0-1
  relevance + raw L2, single embedding pass) and `select_context()` (threshold filter +
  per-document cap). Reusable, unit-testable.
- **`app.py`** — Streamlit UI + a **manual RAG pipeline** (since Faza 4): history-aware query
  rewrite (only when prior turns exist) → `retrieve_scored` → `select_context` (threshold +
  `MAX_PER_DOC`) → honest-refusal short-circuit (skips the LLM) → manual stuff + `qa_prompt | llm`
  → answer + deterministic citations + a Debug panel. Chat memory is managed manually via
  `ChatMessageHistory` (read a snapshot before the turn, append user+AI after). The old opaque
  `create_history_aware_retriever`/`create_stuff_documents_chain`/`create_retrieval_chain`/
  `RunnableWithMessageHistory` chain was removed so every stage is observable.
- **`build_demo_index.py`** — offline index builder (CLI).
- **`assets/`** — the demo corpus: **6 public BGK PDFs** (de minimis, FENG Biznesmax Plus,
  FENG criteria guide, Pożyczka na cyfryzację rules + RODO, Strategia BGK 2025-2030). The old
  generic English files are archived under `docs/wczesniej/old_assets_generic/`.
- **`_diag_step0.py`** — committed diagnostic that justified the embedding-model choice (compares
  3 models on the BGK corpus). Evidence/audit trail, not part of the runtime path.
- **`vectorstore/default_company/`** — gitignored; rebuilt at cold start.

### Key design decisions (see README ADRs)
- **No pickled index shipped.** The index is rebuilt in-memory from `./assets/` on every
  cold start (`get_demo_index`, cached for container lifetime). Pickle drifts across major
  dep versions and silently breaks on platform runtime bumps. ~15–30s first-load latency is
  the deliberate trade for permanent compatibility.
- **Honest retrieval.** `RETRIEVAL_K=8` chunks are scored; relevance comes from the vectorstore's
  own `_select_relevance_score_fn` (the same value a `similarity_score_threshold` retriever uses,
  so the debug panel matches what the LLM gets). Up to `CONTEXT_K=4` chunks above the sidebar
  threshold (default 0.35) reach the LLM; when nothing clears it the app answers
  „Nie wiem — brak podstawy w dokumentach." **without calling the LLM**.
- **Per-document cap (`MAX_PER_DOC=2`).** One large PDF (the 36-page FENG/Biznesmax doc is ~30%
  of the corpus) otherwise monopolizes the top-k with near-duplicate chunks. The cap forces a
  smaller authoritative source into context — concretely it fixes the **de minimis trap**: the
  "gwarancja de minimis" question used to fill its top-4 entirely from the FENG doc (80%) and
  never reach the dedicated de minimis doc (60%).
- **Citations are deterministic.** Built by `_format_citations()` from the chunks actually fed to
  the LLM (not from the model's output). The LLM is NOT asked to produce citations (it mangles
  filenames). Core to the auditability story.
- **Uploads never trust disk pickles** — rebuilt in-memory from a `TemporaryDirectory`.

## Gotchas / things to verify

- **Embedding model (resolved 2026-06-04).** Was English-only `all-MiniLM-L6-v2`; switched to
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` + `normalize_embeddings=True`
  (`rag_index.py:EMB_MODEL`) because the corpus is Polish. Decision is data-driven (`_diag_step0.py`):
  e5-base ranks slightly better but compresses relevance scores to ~0.65–0.83 so the threshold can't
  reject out-of-corpus queries ("stolica Mongolii" → 0.72); paraphrase-multilingual gives clean
  separation (out-of-corpus relevance ~0), which the honest-refusal demo relies on.
- **PDFs must be text-based, not scanned.** `PyPDFLoader` extracts no text from scanned/image
  PDFs (no OCR in the stack). All 6 current BGK PDFs were verified text-based (Faza 1).
- **Prompts + UI are Polish (resolved Faza 3/5).** `qa_system_prompt`, `contextualize_q_system_prompt`,
  the refusal („Nie wiem — brak podstawy w dokumentach."), and all user-facing chrome are Polish.
  Code/docstrings/comments stay English (convention).
- **Polish strings: never use an ASCII `"` to close a `„` quote** — it terminates the Python string
  literal mid-sentence. Use the typographic `”` (U+201D). This bug bit us 3× during the pivot; a
  passing `compileall` catches the hard crash but always eyeball quotes in edited Polish strings.
- **de minimis: 60% vs 80%.** The demo doc warns the BGK mockup slide says "80%" but de minimis
  is **60%** (80% is Biznesmax/Ekomax). The corpus says 60%; the per-doc cap ensures that source
  reaches context. **Still verify the live LLM answer says 60% (needs GROQ_API_KEY).**
- **Live-validation gap.** Retrieval is verified offline, but the LLM path needs `GROQ_API_KEY`.
  Before the demo, `streamlit run app.py` and confirm Q2 answers 60% and Q5 (kredyt hipoteczny)
  refuses — Q5 scores ABOVE threshold so the refusal must come from the prompt, not the threshold.
- **HF Space deploy.** Pushing to `main` force-pushes to the Space. Expect 5–15 min rebuild;
  watch for pickle/runtime incompatibilities. Don't deploy untested.

## Conventions

- Code and README are in **English**; the BGK demo corpus and demo questions are in **Polish**.
- Keep `rag_index.py` free of Streamlit imports (testability boundary).
- `_DEMO_ASSET_SUFFIXES` in `app.py` and `SUPPORTED_SUFFIXES` in `build_demo_index.py`
  must stay in sync.
- Don't commit `vectorstore/` or upload bytes; don't ship a pickled index.
