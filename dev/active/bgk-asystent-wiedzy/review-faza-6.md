# Code Review — Faza 6 (retrieval re-tuning) — commit `3043fa3`

**Data:** 2026-06-04
**Subagent:** code-architecture-reviewer (`/dev-docs-review`)
**Branch:** `feature/bgk-asystent-wiedzy`

## Statystyki
- Plików w commicie: 27 (kod: `rag_index.py`, `app.py`, `.gitignore`; docs: `CLAUDE.md`, `README.md`, `docs/DEMO_RAG_dokumenty_BGK.md`; reszta: archiwum + screenshoty)
- 🔴 blocking: **0**
- 🟠 important: **2**
- 🟡 nit: **3**
- 🔵 suggestion: **3**
- `compileall` (app.py / rag_index.py / build_demo_index.py): ✅ czysto

## Werdykt
**Brak blockerów przed merge do main.** Oba problemy 🟠 to nieaktualne zdania w *ciele* README
(poza ADR-ami) — nie wpływają na działanie ani bezpieczeństwo. Warunek przed merge bez zmian:
**live E2E z `GROQ_API_KEY`** (Q2=60%, Q5=odmowa) — poza zakresem tego review (robi użytkownik).

## Problemy

### 🟠 important
1. **`README.md:58`** — opis Memory mówi „`RunnableWithMessageHistory`", a to zostało usunięte w
   Fazie 4. Aktualnie ręczny `_get_session_history` + `ChatMessageHistory` (snapshot przekazywany
   jako `chat_history`). Publiczny dokument → widoczna sprzeczność z kodem.
2. **`README.md:102`** — „for the prebuilt demo it loads via `load_faiss()`" — faktycznie Quick
   demo robi rebuild in-memory przez `get_demo_index()`. Sprzeczne z ADR-4 w tym samym pliku.

### 🟡 nit
3. **`CLAUDE.md:52`** — „entered in the sidebar" nie odzwierciedla, że pole znika, gdy klucz jest w env.
4. **`app.py:60`** — pole „Przestrzeń robocza" widoczne w Quick demo, choć `persist_dir` nieużywane (kosmetyka).
5. **`docs/DEMO_RAG_dokumenty_BGK.md:20`** — Pożyczka „zawieszona od 1.10.2025"; zweryfikować na żywo,
   czy LLM przy Q1 nie sugeruje, że nabór jest aktywny.

### 🔵 suggestion
6. Q5 refusal zależy od promptu LLM (top1=0.666 > próg), nie od mechanizmu progu — warto powiedzieć
   to recenzentowi wprost (to też wartościowy punkt: „LLM odmawia na bazie kontekstu MŚP").
7. **`README.md:60`** — „retriever (top-k)" to stara architektura; teraz bezpośrednie `retrieve_scored()`.
8. **`app.py:37`** — komentarz „ranks ~8th" jest orientacyjny (tylda OK); dodać odnośnik do `_WALIDACJA_KONCOWA.txt`.

## Zweryfikowane gotchas (wszystkie ✅)
- Polskie cudzysłowy: wszystkie `„…”` zamknięte typograficznym U+201D (brak bug ASCII).
- `_DEMO_ASSET_SUFFIXES` (app.py) ↔ `SUPPORTED_SUFFIXES` (build_demo_index.py): zsynchronizowane.
- `rag_index.py` bez importów Streamlit: granica testowalności nienaruszona.
- Pamięć czatu po zahardkodowaniu `session_id="default_session"`: działa (multi-turn OK).
- `.gitignore` inline-comment fix: poprawny.
- Liczby kod↔docs (600/120, 12/8/4, próg 0.35): zsynchronizowane.
- Q3 app.py ↔ DEMO_RAG: identyczne brzmienie.
- de minimis trap przy MAX_PER_DOC=4: zmierzone — dok de minimis (60%) jest w kontekście Q2.

## Dobre rozwiązania (docenione)
- ADR-6 wzorcowe (problem + zmierzone dane + uzasadnienie liczb + świadoma decyzja o progu).
- `_WALIDACJA_KONCOWA.txt` jako artefakt audytu (5 pytań + 3 OOD).
- Prosty, czytelny UX klucza GROQ (`if GROQ_ENV` zamiast `st.secrets`).
- Komentarz przy stałych retrievalu (app.py:25-38) — konkretny, mierzalny, z trade-offem.
- `session_id` ukryty, nie usunięty → brak regresji pamięci.
