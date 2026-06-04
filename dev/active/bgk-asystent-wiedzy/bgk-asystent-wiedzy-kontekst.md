# Kontekst — Asystent Wiedzy BGK

**Branch:** `feature/bgk-asystent-wiedzy`
**Ostatnia aktualizacja:** 2026-06-04

---

## Powiązane pliki

### Kod (do zmiany)
- `rag_index.py:17` — `EMB_MODEL` (EN → multilingual). Plik **bez importów Streamlit** (granica testowalności — utrzymać).
- `rag_index.py:42-68` — `load_paths` (loadery PDF/TXT/MD; PyPDFLoader bez OCR).
- `rag_index.py:71-75` — `build_faiss_from_docs` (FAISS domyślnie L2/EUCLIDEAN, nie cosine).
- `app.py:33-34` — tytuł/nagłówek (→ PL/BGK).
- `app.py:62, 330` — podpis modelu embeddingów (zaktualizować po zmianie modelu).
- `app.py:189-195` — 3 quick-buttons EN (→ 5 pytań demo PL).
- `app.py:253-258` — `qa_system_prompt` + refusal „I don't know" (→ PL).
- `app.py:241-244` — `contextualize_q_system_prompt` (rozważyć PL).
- `app.py:268-279` — `_get_retriever` (similarity_score_threshold, k=4).
- `app.py:282-322` — blok „Send": opaque `create_retrieval_chain` → **rozbić na ręczny pipeline** + debug panel.
- `app.py:24` / `build_demo_index.py` — `_DEMO_ASSET_SUFFIXES` / `SUPPORTED_SUFFIXES` muszą być zsynchronizowane.

### Korpus (`assets/`) — 6 PDF BGK (niezacommitowane)
- `Pozyczka_na_cyfryzacje_zasady_naboru.pdf` — min. 5 mln zł, do 100% netto, firmy de minimis 0,5%.
- `Pozyczka_na_cyfryzacje_klauzula_RODO.pdf` — RODO/klauzula informacyjna.
- `Gwarancja_de_minimis_warunki_od_2026-04-16.pdf` — **de minimis 60%**, 60/120 mies., prowizja 0,5%.
- `Gwarancja_FENG_Biznesmax_Plus_warunki.pdf` — FENG do 80%, maks. 2,5 mln EUR.
- `Gwarancja_FENG_przewodnik_po_kryteriach.pdf` — Biznesmax (innowacje) vs Ekomax (energia).
- `Strategia_BGK_2025-2030.pdf` — „co BGK robi / dokąd zmierza".

### Dokumentacja źródłowa
- `docs/DEMO_RAG_dokumenty_BGK.md` — spec korpusu + **5 pytań demo** (przećwiczyć).
- `docs/wczesniej/RAG_UPGRADE_DECISION_2026-05-08.md` — pełny wzór refactoru debug panelu (§2.3 fix, §7 krok 5).
- `CLAUDE.md` — architektura, gotchas, konwencje.
- Pamięć: `bgk-rag-demo-pivot`, `rag-upgrade-variant-b`.

## Decyzje techniczne

1. **Multilingual embeddingi** — `all-MiniLM-L6-v2` jest EN-only; korpus PL → **`paraphrase-multilingual-MiniLM-L12-v2`**
   + `normalize_embeddings=True` (czyste relevance ∈ [0,1]). Wybór z danych Step 0: e5-base ma lepszy ranking, ale
   kompresuje score'y (OOD „Mongolia" 0.72 → próg bezużyteczny); paraphrase daje OOD ~0 = czysta separacja dla
   honest-refusal. e5 odrzucony też z powodu komplikacji prefiksów `query:`/`passage:`.
2. **Konwersja score'ów = publiczne `vs.similarity_search_with_relevance_scores(k, score_threshold)`**
   (poprawka plan-reviewer A1) — to dokładnie ta sama ścieżka, której używa retriever; spójność panel↔retriever
   gwarantowana, bez zależności od prywatnej `_select_relevance_score_fn`. **NIE** używać `1 - threshold` (bug kolegi —
   junior signal). Dla wyświetlenia można dodać surowy dystans L2 z `similarity_search_with_score`.
2b. **Cytaty deterministyczne, jedno źródło** (poprawka plan-reviewer C3) — usunąć z `qa_system_prompt` instrukcję
   „append a Citations section" (LLM zmyśla/myli nazwy plików). Zostają tylko cytaty z `_format_citations(used)`.
   Krytyczne dla narracji audytowalności (KNF/UODO).
3. **Brak shipowanego pickle** — indeks rebuildowany in-memory z `assets/` przy cold starcie (ADR-4).
   Trade: ~15-30s first-load za trwałą kompatybilność. Nie commitować `vectorstore/`.
4. **de minimis = 60%** (nie 80%). 80% dotyczy Biznesmax/Ekomax. Makieta DEMO myli — wyrównać do dokumentu.
5. **Podwójna kontrola** — każda faza kodu (2-6) → `/dev-docs-review` przez subagenta PRZED zatwierdzeniem.
6. **Deploy świadomy** — push do `main` force-pushuje na HF Space (5-15 min rebuild). Dopiero po akceptacji.

## Review Fazy 6 (2026-06-04, commit `3043fa3`)
- **Podwójna kontrola (6.R)** przez `code-architecture-reviewer`: **0 blockerów**, 2🟠 / 3🟡 / 3🔵.
  Raport: `review-faza-6.md`. compileall ✅. Poprawki w „Do poprawy po review fazy 6" (zadania.md).
- **Kluczowy wniosek:** kod retrievalu poprawny i spójny z docs (liczby 600/120, 12/8/4, próg 0.35
  zsynchronizowane); wszystkie gotchas (cudzysłowy, sufiksy, granica testowalności, pamięć czatu) OK.
  Oba 🟠 to stale zdania w *ciele* README (poza ADR-ami): `:58` RunnableWithMessageHistory (usunięte
  w Fazie 4), `:102` load_faiss (Quick demo robi rebuild in-memory). Nie-blokujące, do poprawy
  przed/przy merge.
- **Decyzja progu utrwalona:** próg 0.35 świadomie NIE obniżony — 0.30 przepuszcza OOD „przepis na
  sernik" (0.312, kolizja „przepis"=regulacja). Recall kupiony chunkowaniem + budżetami, nie progiem.
- **Warunek przed main (= deploy HF):** live E2E z GROQ (Q2=60%, Q5=odmowa) — poza zakresem review.

## Zależności
- `GROQ_API_KEY` (sidebar lub `.env`).
- `.venv` Python 3.11.9 — sprawne, zależności z `requirements.txt` zainstalowane.
- Stack: Streamlit + LangChain + FAISS (in-memory) + HuggingFace embeddings + ChatGroq.
- Embeddingi: CPU, lokalnie, bez klucza.

## Konwencje (z CLAUDE.md)
- Kod i README po **angielsku**; korpus demo i pytania po **polsku**.
- `rag_index.py` bez importów Streamlit (granica testowalności).
- `_DEMO_ASSET_SUFFIXES` ↔ `SUPPORTED_SUFFIXES` w sync.
- Nie commitować `vectorstore/` ani bajtów uploadu; nie shipować pickle.
