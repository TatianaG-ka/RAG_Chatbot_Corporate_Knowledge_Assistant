---
title: RAG Chatbot — Corporate Knowledge Assistant
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
---


# Project 1 – Company Knowledge Assistant — RAG (PDF/TXT/MD) Demo  
**ChatGroq + FAISS + Citations**

---
<a id="toc"></a>
## Table of Contents
- [Project Overview](#project-overview)
- [Why this project](#why-this-project)
- [Planned Solution & Architecture](#solution-architecture)
- [What builds the FAISS index?](#what-builds-faiss-index)
- [Key architectural decisions](#key-decisions)
- [Technologies Used](#technologies-used)
- [Demo](#demo)
- [How to Run the Project](#how-to-run)
- [Screenshots](#screenshots)

---

<a id="project-overview"></a>
## Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot that enables users to query a document corpus (PDF/TXT/MD) in natural language.
The assistant returns answers **with source citations**, maintains **conversational context**, and — critically — **refuses to answer when the corpus doesn't support the question** (no hallucination).

The live demo (*„Asystent Wiedzy BGK"*) runs on a small corpus of **public Bank Gospodarstwa Krajowego documents** (bgk.pl): de minimis / FENG Biznesmax Plus / Ekomax guarantees, the "Pożyczka na cyfryzację" rules, and the BGK 2025–2030 strategy. Nothing confidential — the same pipeline runs identically on private documents in a real tenant.

---

<a id="why-this-project"></a>
## Why this project

Internal documentation in most companies is a silent productivity sink: onboarding PDFs, policy docs, and process wikis live in five places at once, and a new hire spends their first month learning which one is current. Off-the-shelf chat assistants make it worse — they sound confident, don't cite their sources, and cheerfully hallucinate when asked about a policy they've never seen.

I built this project to exercise the full RAG stack end-to-end under one hard constraint: **the assistant must refuse to answer when the knowledge base doesn't support the question.** That rule drives every architectural choice below — the score-threshold retriever, the cite-or-admit prompt, and the session-memory rebuild for user uploads all exist because hallucination in a corporate knowledge tool is worse than saying "I don't know".

A secondary goal was to ship this to Hugging Face Spaces on the free tier so reviewers and non-engineers can click the live demo without an API-key lottery. Most of the stack choices in the [Key architectural decisions](#key-decisions) section are downstream of that deployment constraint.

---

<a id="solution-architecture"></a>
## Planned Solution & Architecture

- **Ingestion (offline)**: `build_demo_index.py` parses files from `./assets` (PDF/TXT/MD), splits them into chunks, generates embeddings (HF: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`), and saves a FAISS index to `./vectorstore/default_company/`. *Note:* the deployed app does not use this saved index — it rebuilds in memory on cold start (see [ADR-4](#key-decisions)).

- **Retrieval**: the user’s query is (optionally) **rewritten** into a standalone question (history-aware, only when there is prior conversation), then `retrieve_scored()` fetches the top-k chunks with relevance scores and `select_context()` filters them by a tunable threshold and a per-document cap (see [ADR-2](#key-decisions) / [ADR-6](#key-decisions)). When nothing clears the threshold, the app refuses **without calling the LLM**.

- **Generation (LLM)**: the **Groq** model (ChatGroq) receives the selected context + a cite-or-admit prompt and generates the answer; citations are built deterministically from the chunks actually fed to the LLM.

- **Memory**: a manual `ChatMessageHistory` keyed by session id — a snapshot of prior turns is read before the turn and passed explicitly as `chat_history` to each prompt (no `RunnableWithMessageHistory` wrapper, so every stage stays observable).

- **UI**: Streamlit (locally or on **Hugging Face Spaces**), with **Quick demo (in-memory index from `./assets/`)** and **Upload (session-memory index)** modes, plus a debug panel exposing per-chunk scores and stage latencies.

---

## Architecture Diagram
```
[User (Streamlit UI on Hugging Face Spaces)]
        |
        v
[History-aware query rewrite (manual)]
        |
        v
[FAISS — retrieve_scored() top-k + scores]
        |
        v
[select_context(): threshold + per-doc cap]
        |
   nothing passes? --> [Refuse "Nie wiem" — LLM skipped]
        |
[Selected Context]
        v
[LLM (Groq) — cite-or-admit prompt]
        |
[Response + deterministic Citations + Debug panel]
        v
[UI Display]

```
---

<a id="what-builds-faiss-index"></a>
#### What builds the FAISS index?

The deployed app **does not ship or load a pickled index**. In Quick demo mode it rebuilds the FAISS index **in memory from `./assets/` on every cold start** (`get_demo_index` in `app.py`, cached for the container's lifetime). Uploads build a session-only index in memory and are never persisted. See [ADR-4](#key-decisions) for why a pickled index was abandoned.

`build_demo_index.py` can still write a FAISS index to `./vectorstore/default_company/` for **local/offline** use, but that path is gitignored and not used by the deployed app.

#### File roles

- `rag_index.py` – Streamlit-free RAG plumbing: loading assets, chunking, embeddings, FAISS build/save/load, and the scored-retrieval helpers `retrieve_scored()` / `select_context()`.

- `build_demo_index.py` – optional offline CLI to build a demo index from `./assets/*` to `./vectorstore/default_company` (local use only).

- `app.py` – Streamlit app + the manual RAG pipeline. Builds the demo index in memory via `get_demo_index()` (Quick demo) or `build_faiss_from_docs()` (uploads); `load_faiss()` is retained in `rag_index.py` for local offline use only.

---

<a id="key-decisions"></a>
## Key architectural decisions

### ADR-1 — FAISS over Chroma / Qdrant for vector storage

**Context:** single-process Streamlit app deployed to Hugging Face Spaces free tier — no sidecar services, ephemeral storage, cold starts measured in minutes when a model has to be re-downloaded.

**Decision:** use FAISS in-process. The demo index is **rebuilt in memory from `./assets/` on every cold start** (see `get_demo_index` in `app.py`) — *not* committed to the repo and *not* loaded from a pickle (see [ADR-4](#key-decisions) for why shipping a pickled index was abandoned).

**Why not Chroma / Qdrant:** both shine when you need metadata filters, hybrid search, or multi-tenant isolation. For a single demo corpus loaded once, they add a service dependency (Chroma server process, Qdrant container) that breaks the free-tier deployment model. FAISS runs in-process with no sidecar, so the whole app stays a single Streamlit process on the free tier.

**Trade-off:** no server-side metadata filtering and no concurrent writes. A production tenant-per-workspace deployment would outgrow this quickly — a companion portfolio project (`invoice-processor`) uses Qdrant precisely because that use case needs it.

---

### ADR-2 — A manual, scored retrieval pipeline with a relevance threshold

**Context:** the assistant sits in front of a small, curated corpus. If the user asks a question the corpus doesn't cover (which they will), a plain top-k retriever still returns weakly-related chunks and the LLM cheerfully writes an answer based on them. The opaque LangChain `create_retrieval_chain` made it impossible to *show* why a given answer was produced — a problem for an auditability story.

**Decision:** replace the opaque chain with a **manual pipeline** (see `app.py`) built on two helpers in `rag_index.py`:
- `retrieve_scored()` fetches the top `RETRIEVAL_K=12` chunks with a 0–1 relevance score (from the vectorstore's own `_select_relevance_score_fn`, the same value a `similarity_score_threshold` retriever would use).
- `select_context()` keeps chunks above a **sidebar-tunable threshold** (default `0.35`), up to `CONTEXT_K=8`, with a per-document cap (`MAX_PER_DOC=4`) so one large PDF can't monopolize context.

When **no** chunk clears the threshold, the app answers *„Nie wiem — brak podstawy w dokumentach."* **without ever calling the LLM** (a deterministic honest refusal). Every stage — rewritten query, per-chunk relevance/L2 scores, which chunks reached the LLM, stage latencies — is exposed in a Debug panel.

**Why:** hallucination in a corporate knowledge tool is a trust-killer heavier than missed recall. Users learn to verify an assistant that sometimes says "I don't know"; they abandon one that confidently cites the wrong policy. Making the pipeline observable turns "trust me" into "here's exactly what I retrieved and why".

**Trade-off:** valid questions phrased very differently from the source document can fall below the threshold. The sidebar slider lets power users loosen it for exploratory queries, citations are always rendered so the user can verify the match, and the threshold is deliberately *not* set lower than 0.35 because that starts admitting out-of-corpus noise (see [ADR-6](#key-decisions)).

---

### ADR-3 — Groq for LLM inference (not OpenAI, not local)

**Context:** two constraints — (a) keep the Hugging Face Space runnable without users funding my OpenAI bill, (b) target conversational latency of roughly 2 seconds so the live demo stays interactive.

**Decision:** use `ChatGroq` with three selectable models — `llama-3.1-8b-instant` (default, fastest), `llama-3.1-70b-versatile` (higher quality), and `mixtral-8x7b-32768` (long context). The user provides their own Groq API key via the sidebar.

**Why not OpenAI:** free-tier-friendly demos break the moment the author stops paying. Groq's free tier is generous enough that a reviewer can register for a key and run the demo end-to-end in under a minute.

**Why not local inference:** the HF Space free tier is CPU-only. A 4-bit quantized Llama 3.1 8B on CPU is well into double-digit seconds per response — slow enough that reviewers would close the tab before the first answer finished streaming.

---

### ADR-4 — Never deserialize pickles, rebuild on every cold start

**Context:** `FAISS.load_local` requires `allow_dangerous_deserialization=True` because LangChain persists metadata via `pickle`. Pickle loading is arbitrary code execution — fine for a file *I* produced, catastrophic for files uploaded by strangers on the internet. But *"fine for my own pickle"* turned out to be too generous: pickle state format drifts across dependency major versions, and a single hosting-platform runtime bump silently breaks a previously-working index.

**Decision:** never deserialize a pickle at runtime, even our own. Both index paths rebuild the FAISS store in memory from raw bytes:
- **Quick demo mode** rebuilds the index from `./assets/` (PDF/TXT/MD) on cold start, cached via `@st.cache_resource` for the lifetime of the container (see `get_demo_index` in `app.py`). One-time ~15-30s build cost.
- **User uploads** rebuild in memory from uploaded PDF/TXT/MD bytes via `build_faiss_from_docs()`; session-scoped, never persisted.

Uploaded filenames are stripped via `_safe_filename` to prevent path traversal. `load_faiss()` with the unsafe flag is retained in `rag_index.py` for local offline use, but is no longer called by the deployed app.

**Why:** two reasons for the same "never load a pickle at runtime" rule.

1. **Security.** "Upload anything → gets pickle-unmarshalled" is a footgun I wouldn't want a reviewer to find.
2. **Stability (validated the hard way).** An HF Space runtime bump on 2026-04-24 broke the previously-shipped prebuilt pickle with `KeyError: '__fields_set__'` at `pydantic/v1/main.py:423`. The pydantic v1 state schema embedded in the pickle did not match what pydantic v2's v1-compat layer expected on the upgraded Python 3.13 container. Rebuilding from raw documents at cold start sidesteps dependency-version drift entirely — the index is whatever the currently-installed libraries produce, on every cold start.

**Trade-off:** the first request after a container cold start waits ~15-30s for the index to build. `@st.cache_resource` keeps it hot for the rest of the container's lifetime, and user-upload indexes intentionally do not survive across sessions.

---

### ADR-5 — Multilingual embeddings chosen for clean out-of-corpus separation

**Context:** the corpus pivoted to **public Polish-language BGK documents**. The original `sentence-transformers/all-MiniLM-L6-v2` is English-only and ranked Polish chunks poorly.

**Decision:** use `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` with `normalize_embeddings=True` (see `rag_index.py:EMB_MODEL`). The choice is data-driven — `_diag_step0.py` compares three models on the BGK corpus and is committed as an audit trail.

**Why not e5-base:** it ranks chunks slightly better, but compresses all relevance scores into ~0.65–0.83 — so an out-of-corpus query ("stolica Mongolii") still scores ~0.72 and a fixed threshold can't reject it. `paraphrase-multilingual` gives clean separation (out-of-corpus relevance ≈ 0), which the entire honest-refusal demo ([ADR-2](#key-decisions)) depends on.

**Trade-off:** absolute in-corpus scores are modest (top matches often 0.4–0.8), so the threshold and retrieval budgets ([ADR-6](#key-decisions)) had to be tuned to these score ranges rather than borrowed from an English-corpus default.

---

### ADR-6 — Chunk size 600 and retrieval budgets tuned to the corpus

**Context:** with 1200-char chunks, a one-line fact (e.g. *"Minimalna wartość udzielonej Pożyczki wynosi 5 mln zł"*) was ~8% of a chunk packed with ~12 unrelated legal clauses. Its averaged embedding was dominated by the surrounding text, so the answer-bearing chunk never reached the top of the ranking and the assistant wrongly answered *"brak informacji"* — a recall failure, not hallucination.

**Decision:** halve the chunking to `CHUNK_SIZE=600 / CHUNK_OVERLAP=120` (`rag_index.py`) and widen the retrieval budgets to `RETRIEVAL_K=12 / CONTEXT_K=8 / MAX_PER_DOC=4` (`app.py`). Smaller chunks are topically focused so a single fact surfaces; wider budgets let the answer chunk reach the LLM even when it ranks behind near-tied chunks or chunks from a different document.

**Why these numbers:** measured offline against all five rehearsed demo questions. The "minimalna kwota" answer is the **4th-best chunk within its own PDF** (so `MAX_PER_DOC` had to allow 4); the de minimis "120 miesięcy" chunk ranks **~8th overall**, behind FENG chunks that also discuss guarantee periods (so `CONTEXT_K`/`RETRIEVAL_K` had to widen). All five facts land in context with these budgets, while the `0.35` threshold still cleanly rejects out-of-corpus queries.

**Trade-off:** the threshold was deliberately **not** lowered to improve recall further — at `0.30`, an out-of-corpus query like "przepis na sernik" leaks in at relevance `0.312` (the Polish word *przepis* collides with *przepisy* = regulations, which pervade the corpus). Recall is bought with chunking and budgets, not by weakening the refusal guarantee.

---

<a id="technologies-used"></a>
## Technologies Used

| Component                                     | Role                                                |
| --------------------------------------------- | --------------------------------------------------- |
| **LangChain** (core/community/text-splitters) | Retrieval pipeline + prompts + conversation history |
| **FAISS**                                     | Vector search over context                          |
| **HuggingFace Embeddings**                    | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| **ChatGroq**                                  | LLM (Groq API) for answer generation                |
| **Streamlit**                                 | UI locally / on Hugging Face Spaces                 |

---

<a id="demo"></a>
## Demo

- Live Demo (Hugging Face Spaces): https://huggingface.co/spaces/TatianaGol/RAG_Chatbot-Corporate_Knowledge_Assistant

- GitHub Repository: https://github.com/TatianaG-ka/RAG_Chatbot_Corporate_Knowledge_Assistant

---

<a id="how-to-run"></a>
## How to Run the Project

You can run the project in two ways — either directly in your browser (Hugging Face Space) or locally from your machine.

---

### **Option 1: Run on Hugging Face Spaces (Recommended)**

No installation required — just open the demo link below:

Steps:
1. Open the Hugging Face Space.  
2. (Quick demo) rebuilds an index **in memory** from the public BGK documents in `/assets`.
   (Upload) lets you upload your own files (PDF/TXT/MD) and build an index **in session memory**. 
3. The Quick demo corpus (`/assets`) is six public BGK PDFs:  
   - `Gwarancja_de_minimis_warunki_od_2026-04-16.pdf`  
   - `Gwarancja_FENG_Biznesmax_Plus_warunki.pdf`  
   - `Gwarancja_FENG_przewodnik_po_kryteriach.pdf`  
   - `Pozyczka_na_cyfryzacje_zasady_naboru.pdf` / `Pozyczka_na_cyfryzacje_klauzula_RODO.pdf`  
   - `Strategia_BGK_2025-2030.pdf`  
4. Ask a question in natural language (the demo ships five rehearsed Polish questions as buttons):  
   - „Jaka jest minimalna kwota Pożyczki na cyfryzację i kto może wnioskować?"  
   - „Do jakiej części kredytu sięga gwarancja de minimis?"  
   - „Co finansuje gwarancja Biznesmax, a co gwarancja Ekomax?"  
   - **(refuse-on-no-context trap)** „Czy gwarancja de minimis obejmuje kredyt hipoteczny dla osoby fizycznej?" → *„Nie wiem — brak podstawy w dokumentach."*  
5. View AI-generated responses with **citations**, or the deterministic refusal when the corpus doesn't cover the question.

> A `GROQ_API_KEY` is required for the LLM step (sidebar, or an env/Space secret). Embeddings run locally on CPU — no key needed.

---

### **Option 2: Run Locally from GitHub**

If you prefer to run the project locally:

1. Clone the repository
`git clone https://github.com/TatianaG-ka/RAG_Chatbot_Corporate_Knowledge_Assistant`

2. Navigate to the project directory
`cd RAG_Chatbot_Corporate_Knowledge_Assistant`

3. Install dependencies
`pip install -r requirements.txt`

4. (Optional) Build a local FAISS index to disk
`python build_demo_index.py`

*Note:* this is **optional** — the app rebuilds the index in memory from `./assets/` at startup regardless (see [ADR-4](#key-decisions)). The script is only for local/offline experiments.

5. Run the Streamlit app
`streamlit run app.py`

6. Then open your browser and go to
`http://localhost:8501`

7. Example queries (Quick demo corpus):
- „Jaki jest okres gwarancji dla kredytu inwestycyjnego de minimis?"
- „Do jakiej części kredytu sięga gwarancja de minimis?"
- „Co finansuje gwarancja Biznesmax, a co gwarancja Ekomax?"

---
<a id="screenshots"></a>
### Screenshots

*„Asystent Wiedzy BGK" — public BGK documents, Polish demo questions, source citations and honest refusal.*

**Landing — Quick demo index + the five rehearsed questions (GROQ key supplied via server config):**
![](./screenshots/demo_new_1.png)  

**Refuse-on-no-context trap — „Czy gwarancja de minimis obejmuje kredyt hipoteczny dla osoby fizycznej?" → „Nie wiem — brak podstawy w dokumentach.":**
![](./screenshots/demo_new_2.png)  

**Grounded answer with citations — „minimalna kwota Pożyczki" → 5 mln zł, and an honest „no info" on the part the corpus doesn't cover:**
![](./screenshots/demo_new_3.png)  

**Debug panel — per-chunk relevance/L2 scores, which chunks reached the LLM, and stage latencies:**
![](./screenshots/demo_new_4.png)  

**Comparison answer — „Co finansuje gwarancja Biznesmax, a co Ekomax?" with sources:**
![](./screenshots/demo_new_5.png)  


### License

MIT License © 2025 [Tatiana Golinska]




