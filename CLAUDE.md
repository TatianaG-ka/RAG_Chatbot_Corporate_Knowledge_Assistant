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

## Current goal — recruitment demo ("Asystent Wiedzy BRH")

This repo is the **flagship portfolio project** for an application to
**"Menedżer ds. AI" at Bank Gospodarstwa Krajowego (BGK)** — that is the *employer* we
are applying to. The job posting is in `docs/ogloszemia_stanowiska/`.

The demo corpus is a **fully fictional development bank — Bank Rozwoju Horyzont S.A.
(BRH)** — built from scratch (7 markdown documents in `assets/`). The spec + rehearsed
demo questions are in `docs/pytanie_prawne/00_Przewodnik_DEMO_RAG_BRH.md` — read it before
touching the corpus or demo questions.

**Why fictional, not real BGK documents (decided 2026-06-04):** "public" ≠ "free to reuse".
Real BGK docs are copyright-protected works, their name/logo are trademarks, and building a
public product on them — then showing it to BGK at the interview — is the *wrong* signal for a
compliance/AI-governance role (it reads as careless with others' IP/data). A self-authored
fictional corpus demonstrates the **identical** skills (chunking, retrieval, citations,
refuse-on-no-context) with zero legal/brand risk, and the deliberate choice itself is an
asset on the interview. The full legal reasoning + the BRH source docs are in
`docs/pytanie_prawne/`.

Why the demo maps to the job:
- **AI Act / RODO awareness** → the refuse-on-no-context rule = compliance-grade, trustworthy AI for a regulated bank.
- **Answering supervisory bodies (KNF, UODO)** → citations + debug panel = auditability.
- **Respecting IP/data** → using a fictional corpus instead of a real institution's documents.
- **Translating business needs to tech** → the whole demo, presented simply for non-technical reviewers.

Superseded earlier plans are archived in `docs/wczesniej/` (generic English corpus) and
`docs/DEMO_RAG_dokumenty_BGK.md` (the BGK-document approach we deliberately dropped for the reasons above).

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

Requires `GROQ_API_KEY` (via `.env`/env/HF-Space secret, or the sidebar field — which is shown
only when the key is *not* already in the environment). Embeddings run locally (CPU), no key needed.

## Architecture

- **`rag_index.py`** — pure RAG plumbing (no Streamlit). Document loading (PDF/TXT/MD),
  chunking (`RecursiveCharacterTextSplitter`, size 600 / overlap 120), FAISS build/save/load,
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
- **`assets/`** — the demo corpus: **7 fictional BRH markdown docs** (Pożyczka na Cyfryzację i AI
  regulamin, gwarancja de minimis „Rozwój", gwarancja „EkoHoryzont", procedura obsługi wniosku,
  klauzula RODO, polityka odpowiedzialnego AI, skrót strategii 2030). All self-authored (`Bank
  Rozwoju Horyzont S.A.`) — see `docs/pytanie_prawne/`. Earlier corpora archived under
  `docs/wczesniej/old_assets_generic/` (generic EN) and `docs/wczesniej/` (BGK PDFs, dropped).
- **`_diag_step0.py`** — committed diagnostic that justified the embedding-model choice (compares
  3 models on a Polish corpus). Evidence/audit trail, not part of the runtime path.
- **`vectorstore/default_company/`** — gitignored; rebuilt at cold start.

### Key design decisions (see README ADRs)
- **No pickled index shipped.** The index is rebuilt in-memory from `./assets/` on every
  cold start (`get_demo_index`, cached for container lifetime). Pickle drifts across major
  dep versions and silently breaks on platform runtime bumps. ~15–30s first-load latency is
  the deliberate trade for permanent compatibility.
- **Honest retrieval.** `RETRIEVAL_K=12` chunks are scored; relevance comes from the vectorstore's
  own `_select_relevance_score_fn` (the same value a `similarity_score_threshold` retriever uses,
  so the debug panel matches what the LLM gets). Up to `CONTEXT_K=8` chunks above the sidebar
  threshold (default 0.35) reach the LLM; when nothing clears it the app answers
  „Nie wiem — brak podstawy w dokumentach." **without calling the LLM**.
- **Per-document cap (`MAX_PER_DOC=4`).** Caps how many chunks one document contributes so a large
  source can't fill the whole context, while still allowing a single authoritative doc to answer a
  single-source question.
- **Retrieval budgets `RETRIEVAL_K=12 / CONTEXT_K=8 / MAX_PER_DOC=4` with 600/120 chunking.**
  Re-validated offline on the BRH corpus (`docs/testowanie_rag/_WALIDACJA_BRH.txt`): all 5 demo
  facts reach context (Q1 „500 000/10 000 000", Q2 „60%/3,5 mln", Q3 „80% + 60%", Q4 „człowiek
  decyduje"), the Q5 trap („oprocentowanie lokaty") and other out-of-corpus queries score below the
  0.35 threshold → clean refusal. **Threshold stays 0.35** (the budgets/chunking carry recall, not a
  lower threshold). Phrasing matters: a question that repeats the *program name* („…Pożyczki na
  Cyfryzację i AI") pulls the match toward title/intro chunks, so Q1 is phrased plainly („…kwota
  Pożyczki?") to surface the §4 amounts chunk; Q3 uses „Co finansuje X, a co Y" to pull both
  guarantee docs. Re-run the validation after any corpus edit.
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
- **Corpus is markdown (`.md`) loaded via `UnstructuredMarkdownLoader`** (`unstructured` is in
  `requirements.txt`). If a future corpus uses PDFs, they must be text-based, not scanned —
  `PyPDFLoader` has no OCR.
- **Prompts + UI are Polish (resolved Faza 3/5).** `qa_system_prompt`, `contextualize_q_system_prompt`,
  the refusal („Nie wiem — brak podstawy w dokumentach."), and all user-facing chrome are Polish.
  Code/docstrings/comments stay English (convention).
- **Polish strings: never use an ASCII `"` to close a `„` quote** — it terminates the Python string
  literal mid-sentence. Use the typographic `”` (U+201D). This bug bit us 3× during the pivot; a
  passing `compileall` catches the hard crash but always eyeball quotes in edited Polish strings.
- **BRH facts are internally consistent (by design).** de minimis „Rozwój" = **60%** / max 3,5 mln zł;
  „EkoHoryzont" = **80%** / efektywność energetyczna; Pożyczka = 500 000–10 000 000 zł; Polityka AI =
  człowiek decyduje (no fully-automated credit decisions). Numbers are consistent across files so no
  contradiction surfaces live. **Still verify the live LLM answers (needs GROQ_API_KEY).**
- **Demo questions are phrased for retrieval (see budgets above).** Q1 „…kwota Pożyczki?" (not „…na
  Cyfryzację i AI"), Q3 „Co finansuje EkoHoryzont, a co de minimis Rozwój?". `app.py:DEMO_QUESTIONS`
  and `docs/pytanie_prawne/00_Przewodnik_DEMO_RAG_BRH.md` should stay in sync.
- **Live-validation gap.** Retrieval is verified offline, but the LLM path needs `GROQ_API_KEY`.
  Before the demo, `streamlit run app.py` and confirm Q2 answers 60%, Q4 says „człowiek decyduje",
  and Q5 (oprocentowanie lokaty) refuses — Q5 is a clean threshold refusal (out-of-corpus, no
  deposit products in the corpus).
- **HF Space deploy.** Pushing to `main` triggers `.github/workflows/main.yml`: a smoke job
  (deps + `compileall` + import) gates a force-push to the Space. Expect 5–15 min rebuild.
- **Binaries in `assets/` MUST be LFS-tracked before committing (resolved Faza 6 deploy).** The HF
  Space `pre-receive` hook **rejects plain binary blobs** („Your push was rejected because it
  contains binary files… use xet"). `.gitattributes` LFS-tracks `*.png` and `*.pdf`; any new binary
  type (e.g. `*.docx`) must be added there *before* the commit. If a binary already slipped in as a
  raw blob, fix with `git lfs migrate import --include="*.ext"` (rewrites history → force-push). This
  bit the first deploy: PDFs were committed as raw blobs and HF rejected the sync. (The current BRH
  corpus is markdown, so this is less of a risk now — but the rule stands for any future binary.)

## Conventions

- Code and README are in **English**; the BRH demo corpus and demo questions are in **Polish**.
- Keep `rag_index.py` free of Streamlit imports (testability boundary).
- `_DEMO_ASSET_SUFFIXES` in `app.py` and `SUPPORTED_SUFFIXES` in `build_demo_index.py`
  must stay in sync.
- Don't commit `vectorstore/` or upload bytes; don't ship a pickled index.
