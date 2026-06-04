# Plan — Asystent Wiedzy BGK (demo rekrutacyjne)

**Branch:** `feature/bgk-asystent-wiedzy`
**Ostatnia aktualizacja:** 2026-06-04

---

## Podsumowanie wykonawcze

Pivot flagowego projektu portfolio (`rag_chatbot`) z generycznego korpusu EN na
**„Asystenta Wiedzy BGK"** zbudowanego na publicznych dokumentach bgk.pl — pod aplikację
na stanowisko **„Menedżer ds. AI" w Banku Gospodarstwa Krajowego**. Cel: działające demo
na żywo (5 przećwiczonych pytań, w tym jedno „refuse-on-no-context").

Zakres łączy dwa wątki:
- **(A) Pivot korpusu + lokalizacja PL** — żeby retrieval na polskich PDF-ach w ogóle działał.
- **(B) Refactor obserwowalności** (z planu `rag-upgrade-variant-b`) — debug panel rozbijający
  opaque `create_retrieval_chain` na ręczny pipeline, alert „Nie wiem", quick-buttons.
  To największy *senior signal* dla rekrutera.

Każda faza zmieniająca kod kończy się **podwójną kontrolą** (`/dev-docs-review` — przegląd
przez subagenta) **przed zatwierdzeniem**.

## Analiza obecnego stanu

- **Korpus:** 6 publicznych PDF-ów BGK już skopiowanych do `assets/` (niezacommitowane),
  stare pliki EN usunięte i zarchiwizowane w `docs/wczesniej/old_assets_generic/`.
- **Embeddingi:** `sentence-transformers/all-MiniLM-L6-v2` (`rag_index.py:17`) — **angielski**,
  słaby dla PL. #1 ryzyko techniczne.
- **Prompty:** `qa_system_prompt` i komunikat „I don't know" (`app.py:253-258`) — **po angielsku**.
- **Pipeline RAG:** opaque `create_history_aware_retriever` → `create_stuff_documents_chain` →
  `create_retrieval_chain` (`app.py:282-298`). Nie zwraca przepisanego zapytania ani raw score'ów.
- **Quick-buttons:** 3 przyciski z pytaniami EN (`app.py:189-195`).
- **Debug:** istnieje minimalny expander „context (top-k)" (`app.py:313-322`) — bez score'ów,
  bez przepisanego zapytania, bez latencji.
- **Środowisko:** `.venv` (Python 3.11.9) sprawne, zależności doinstalowane, importy OK.

## Stan docelowy

- Retrieval na PL działa (model multilingual) → 5 pytań demo zwraca trafne cytaty.
- Prompty + refusal po polsku — odpowiedzi i „Nie wiem" czytają się naturalnie.
- Debug panel pokazuje: przepisane zapytanie, raw L2 distance + similarity 0-1 (konwersja przez
  `vs._select_relevance_score_fn()`, **nie** bugowy `1 - threshold`), próg, latencję, model.
- Quick-buttons = 5 pytań demo z `docs/DEMO_RAG_dokumenty_BGK.md`.
- Alert „Nie wiem" (żółty) gdy zero chunków powyżej progu — pomija LLM (oszczędność + jasny sygnał).
- Korpus i odpowiedzi mówią **de minimis = 60%** (nie 80%).
- Wszystko przetestowane lokalnie; deploy na HF Space dopiero po code-review i akceptacji.

---

## Fazy wdrożenia

### Faza 1 — Walidacja korpusu (blokująca) — **S**
Bez tego reszta nie ma sensu (PyPDFLoader nie ma OCR).
1.1 Sprawdź, że każdy z 6 PDF-ów zwraca **tekst** (nie skan) — `load_paths` + licznik znaków/strona.
1.2 Zsynchronizuj `_DEMO_ASSET_SUFFIXES` (`app.py`) i `SUPPORTED_SUFFIXES` (`build_demo_index.py`).
1.3 Potwierdź, że dokumenty zawierają fakty z 5 pytań demo (kwoty, %, okresy).
- **Akceptacja:** każdy PDF > ~200 znaków/stronę; brak pliku zerowego; fakty obecne. Bez zmian w kodzie → bez code-review.

### Faza 2 — Model embeddingów multilingual — **M** ✅ (kod gotowy, czeka na 2.R)
2.1 ✅ `EMB_MODEL` → **`paraphrase-multilingual-MiniLM-L12-v2`** + `normalize_embeddings=True`.
   **Decyzja oparta na danych (Step 0, `_diag_step0.py`):** porównano 3 modele na korpusie BGK z polskimi
   diakrytykami. e5-base ma najlepszy ranking, ale **kompresuje score'y do 0.65–0.83** → „stolica Mongolii"
   dostaje 0.72, więc próg nie odrzuci OOD (zabija wizualną narrację honest-refusal). paraphrase-multilingual:
   Mongolia **−0.01** (czysta separacja), cyfryzacja TOP1, drop-in bez prefiksów, mniejszy → szybszy cold-start.
2.2 ✅ Podpis embeddingów (`app.py:62`, `:330`) renderowany z importu `EMB_MODEL` (single source of truth).
2.3 ✅ Prefiksy E5 — porzucone wraz z e5 (paraphrase nie wymaga). Wrapper E5 został w `_diag_step0.py`.
2.4 ✅ Trafność 5 pytań zweryfikowana w Step 0.
- **Akceptacja:** ~~pytanie #5 zwraca `[]`~~ **KOREKTA:** #5 „hipoteczny" scoruje wysoko (blisko de minimis),
  refuse #5 idzie z promptu (Faza 3), nie z progu. Akcept.: 5 pytań trafia w dobre chunki; OOD „Mongolia" → ~0. **→ 2.R podwójna kontrola.**

### Faza 3 — Lokalizacja promptów na PL — **S**
3.1 Przetłumacz `qa_system_prompt` (`app.py:253-258`) na PL, zachowując regułę „używaj TYLKO kontekstu" + sekcję „Źródła".
3.2 Zmień refusal na „Nie wiem — brak podstawy w dokumentach".
3.3 Rozważ PL także dla `contextualize_q_system_prompt` (rewrite działa lepiej w języku korpusu).
3.4 Zmień tytuł strony i nagłówek (`app.py:33-34`) na PL/branding BGK.
- **Akceptacja:** odpowiedzi i refusal czytają się naturalnie po polsku. **→ podwójna kontrola.**

### Faza 4 — Refactor pipeline + debug panel (największa) — **L**
Rozbij `create_retrieval_chain` na ręczny pipeline (wzór: `docs/wczesniej/RAG_UPGRADE_DECISION_2026-05-08.md` §7 krok 5).
4.1 Ręczny krok rewrite (capture `rewritten_question`).
4.2 Ręczny retrieval ze score'ami: **publiczne** `vs.similarity_search_with_relevance_scores(k, score_threshold)`
   (poprawka plan-reviewer A1 — to dokładnie to, czego używa retriever wewnętrznie; usuwa zależność od prywatnej
   `_select_relevance_score_fn` i ryzyko niezgodności). Opcjonalnie drugie wywołanie `similarity_search_with_score`
   dla wyświetlenia surowego dystansu L2 obok similarity 0-1.
4.3 Jeśli nic nie przejdzie progu → alert „Nie wiem", pomiń LLM.
4.4 Ręczny stuff + wywołanie LLM; zachowaj pamięć rozmowy (history-aware).
4.5 Debug expander: przepisane zapytanie, top-K ze score'ami (✅/✗ próg), próg, model, latency_ms.
4.6 Zachowaj cytaty (`_format_citations`) i ścieżkę uploadów.
- **Akceptacja:** panel pokazuje spójne score'y zgodne z faktycznym filtrowaniem retrievera; turn 1 i turn 2 (history) działają; cytaty bez zmian. **→ podwójna kontrola (najważniejsza).**

### Faza 5 — Quick-buttons + sanity de minimis — **S**
5.1 Zamień 3 przyciski EN (`app.py:189-195`) na 5 pytań demo PL.
5.2 Potwierdź, że odpowiedź na żywo mówi **de minimis 60%** (nie 80% z makiety).
- **Akceptacja:** przyciski wstawiają pytania demo; pytanie #2 zwraca 60% z cytatem. **→ podwójna kontrola.**

### Faza 6 — Test E2E + commit + (opcjonalny) deploy — **M**
6.1 `streamlit run app.py` — przeklik 5 pytań, weryfikacja debug panelu i refusal #5.
6.2 CI smoke: `python -m compileall` + `import` test.
6.3 Commit faz na branchu; merge do `main` dopiero po akceptacji.
6.4 Deploy HF Space — **świadomie**, oczekuj 5-15 min rebuild, pilnuj pickle/runtime.
- **Akceptacja:** demo działa lokalnie end-to-end; CI zielone. Deploy = osobna, świadoma decyzja.

---

## Ocena ryzyka

| Ryzyko | Praw. | Wpływ | Mitygacja |
|---|---|---|---|
| Model EN słaby dla PL | Pewne | Wysoki | Faza 2 multilingual — robiona wcześnie |
| Któryś PDF to skan (brak OCR) | Średnie | Wysoki | Faza 1 blokująca przed resztą |
| E5 wymaga prefiksów query/passage | Średnie | Średni | Zadanie 2.3 — zweryfikować/owrappować |
| `_select_relevance_score_fn()` to API prywatne | Niskie | Średni | Przypiąć wersję langchain; fallback na dystans surowy |
| Pickle/runtime niezgodność na HF | Średnie | Średni | Brak shipowanego pickle (rebuild in-memory); deploy świadomy |
| Refactor pipeline psuje history-aware | Średnie | Wysoki | Test inkrementalny turn1→turn2; podwójna kontrola |
| Halucynacja na żywo na demo | Niskie | Wysoki | Pytanie #5 + walidacja odpowiedzi przed rozmową |

## Mierniki sukcesu
- 5/5 pytań demo daje poprawną odpowiedź z trafnym cytatem (pytanie #5 = poprawny refusal).
- Debug panel pokazuje score'y spójne z filtrowaniem retrievera (zero rozjazdu junior-signal).
- Odpowiedzi i refusal naturalne po polsku; de minimis = 60%.
- Każda faza kodu przeszła `/dev-docs-review` przed zatwierdzeniem.

## Zależności
- `GROQ_API_KEY` (sidebar/.env). Embeddingi lokalnie (CPU), bez klucza.
- `.venv` (Python 3.11.9) — sprawne.
- Pliki źródłowe: `CLAUDE.md`, `docs/DEMO_RAG_dokumenty_BGK.md`, pamięć `bgk-rag-demo-pivot`,
  `rag-upgrade-variant-b`, `docs/wczesniej/RAG_UPGRADE_DECISION_2026-05-08.md`.

## Szacunki czasowe
Faza 1: 0.5h · Faza 2: 1-1.5h · Faza 3: 0.5h · Faza 4: 2-3h · Faza 5: 0.5h · Faza 6: 1-2h
**Razem: ~6-8h** (+ czas podwójnych kontroli).
