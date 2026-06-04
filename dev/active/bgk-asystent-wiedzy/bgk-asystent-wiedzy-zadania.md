# Zadania — Asystent Wiedzy BGK

**Branch:** `feature/bgk-asystent-wiedzy`
**Ostatnia aktualizacja:** 2026-06-04

Legenda: `[ ]` do zrobienia · `[~]` w toku · `[x]` zrobione · `[R]` po podwójnej kontroli (code-review OK)

---

## Faza 1 — Walidacja korpusu (blokująca) — S
- [ ] 1.1 Sprawdź, że każdy z 6 PDF-ów BGK zwraca tekst (nie skan) — `load_paths` + znaki/stronę
- [ ] 1.2 Zsynchronizuj `_DEMO_ASSET_SUFFIXES` (app.py) ↔ `SUPPORTED_SUFFIXES` (build_demo_index.py)
- [ ] 1.3 Potwierdź obecność faktów z 5 pytań demo (kwoty, %, okresy) w tekście PDF-ów
- _Akceptacja:_ każdy PDF > ~200 znaków/stronę, brak pliku zerowego, fakty obecne. (bez code-review — brak zmian w kodzie)

## Faza 2 — Model embeddingów multilingual — M
- [ ] 2.1 `EMB_MODEL` → `intfloat/multilingual-e5-base` (rag_index.py:17)
- [ ] 2.2 Zaktualizuj podpis embeddingów (app.py:62, 330)
- [ ] 2.3 Zweryfikuj prefiksy E5 `query:`/`passage:` — wrapper czy bez?
- [ ] 2.4 Przebuduj indeks + sprawdź trafność 5 pytań
- [ ] **2.R Podwójna kontrola — `/dev-docs-review`**
- _Akceptacja:_ 5 pytań trafia; pytanie-pułapka #5 zwraca `[]`

## Faza 3 — Lokalizacja promptów na PL — S
- [ ] 3.1 `qa_system_prompt` → PL (zachowaj „tylko kontekst" + sekcję Źródła)
- [ ] 3.2 Refusal → „Nie wiem — brak podstawy w dokumentach"
- [ ] 3.3 Rozważ PL dla `contextualize_q_system_prompt`
- [ ] 3.4 Tytuł/nagłówek → PL/BGK (app.py:33-34)
- [ ] **3.R Podwójna kontrola — `/dev-docs-review`**
- _Akceptacja:_ odpowiedzi i refusal naturalne po polsku

## Faza 4 — Refactor pipeline + debug panel — L
- [ ] 4.1 Ręczny rewrite (capture rewritten_question)
- [ ] 4.2 Ręczny retrieval ze score'ami + konwersja `vs._select_relevance_score_fn()` (raw_distance + score)
- [ ] 4.3 Zero chunków powyżej progu → alert „Nie wiem", pomiń LLM
- [ ] 4.4 Ręczny stuff + LLM, zachowaj history-aware (pamięć rozmowy)
- [ ] 4.5 Debug expander: rewritten query, top-K score'y (✅/✗), próg, model, latency_ms
- [ ] 4.6 Zachowaj cytaty + ścieżkę uploadów
- [ ] **4.R Podwójna kontrola — `/dev-docs-review` (najważniejsza)**
- _Akceptacja:_ score'y spójne z retrieverem; turn1+turn2 działają; cytaty bez zmian

## Faza 5 — Quick-buttons + sanity de minimis — S
- [ ] 5.1 3 przyciski EN → 5 pytań demo PL (app.py:189-195)
- [ ] 5.2 Potwierdź odpowiedź de minimis = 60% (nie 80%)
- [ ] **5.R Podwójna kontrola — `/dev-docs-review`**
- _Akceptacja:_ przyciski wstawiają pytania; #2 zwraca 60% z cytatem

## Faza 6 — Test E2E + commit + deploy — M
- [ ] 6.1 `streamlit run app.py` — przeklik 5 pytań + debug + refusal #5
- [ ] 6.2 CI smoke: `compileall` + import test
- [ ] 6.3 Commit faz na branchu; merge do main po akceptacji
- [ ] 6.4 Deploy HF Space — świadomie (5-15 min rebuild, pilnuj pickle/runtime)
- _Akceptacja:_ demo E2E lokalnie OK; CI zielone; deploy = osobna decyzja

---

## Notatki postępu
- 2026-06-04: plan utworzony. Środowisko (.venv) naprawione po przerwanej instalacji. Korpus 6 PDF już w `assets/` (niezacommitowany). Kod jeszcze nietknięty.
