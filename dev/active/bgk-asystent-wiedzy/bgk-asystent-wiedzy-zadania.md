# Zadania — Asystent Wiedzy BGK

**Branch:** `feature/bgk-asystent-wiedzy`
**Ostatnia aktualizacja:** 2026-06-04

Legenda: `[ ]` do zrobienia · `[~]` w toku · `[x]` zrobione · `[R]` po podwójnej kontroli (code-review OK)

---

## Faza 1 — Walidacja korpusu (blokująca) — S ✅
- [x] 1.1 Sprawdź, że każdy z 6 PDF-ów BGK zwraca tekst (nie skan) — `load_paths` + znaki/stronę
      → wszystkie tekstowe (de minimis 13s, Biznesmax 36s, przewodnik 32s, RODO 3s, zasady 15s, Strategia 21s ~rzadki tekst, prezentacja)
- [x] 1.2 Zsynchronizuj `_DEMO_ASSET_SUFFIXES` (app.py) ↔ `SUPPORTED_SUFFIXES` (build_demo_index.py) — już zgodne ({.pdf,.txt,.md,.markdown})
- [x] 1.3 Potwierdź obecność faktów z 5 pytań demo → de minimis **60%** (0× „80%"), 60/120 mies., prowizja 0,5%, Pożyczka min 5 mln zł, do 100% netto. ⚠️ „JST/uczelnie" brak w PDF (→ Faza 5/luka).
- _Akceptacja:_ ✅ spełniona. (bez code-review — brak zmian w kodzie)

## Faza 2 — Model embeddingów multilingual — M [R] (2.R OK po fixie)
- [x] 2.1 `EMB_MODEL` → **`paraphrase-multilingual-MiniLM-L12-v2`** (rag_index.py) + `normalize_embeddings=True`
      → NIE e5-base. Dowód Step 0 (`_diag_step0.py`): e5 świetnie rankinguje, ale kompresuje score'y do 0.65–0.83 → „stolica Mongolii" = 0.72, próg nie odrzuci OOD. paraphrase daje Mongolię **−0.01** (czysta separacja) + cyfryzacja TOP1.
      ⚠️ **2.R FIX:** `normalize_embeddings=True` było w `_diag_step0.py`, ale BRAKOWAŁO w produkcyjnym `build_embeddings()` → bez niego WSZYSTKIE pytania IN scorowały <0 (−4.9…−22) → retriever zwracał `[]` na wszystko. Dodane (`rag_index.py:34`). Po fixie IN=0.39–0.80, OOD=−0.01. ✅
- [x] 2.2 Podpis embeddingów (app.py:62, 330) → teraz z importu `EMB_MODEL` (single source of truth, nie może się rozjechać)
- [x] 2.3 Prefiksy E5 — N/D (porzucono e5; paraphrase jest drop-in bez prefiksów). Wrapper E5 zachowany tylko w `_diag_step0.py`.
- [x] 2.4 Indeks + trafność 5 pytań → po fixie IN przechodzi próg. 🟠 OTWARTE: top-1 dla „de minimis" to czasem Biznesmax_Plus, nie de_minimis — ryzyko cytowania 80% zamiast 60%. Dotunować w Fazie 4 (k / chunk / prompt).
- [x] **2.R Podwójna kontrola — `/dev-docs-review`** → subagent (code-architecture-reviewer) empirycznie potwierdził blocker normalize; fix zaaplikowany + zweryfikowany; CI zielone. 🟠 do zrobienia: zdecydować los `_diag_step0.py` (commit jako dowód decyzji vs .gitignore); zwalidować demo przy progu 0.40 (Pożyczka=0.39 jest na styk).
- _Akceptacja:_ ~~pytanie #5 zwraca `[]`~~ **KOREKTA (dowód Step 0):** #5 „hipoteczny" scoruje WYSOKO (0.52–0.81 we wszystkich modelach — semantycznie blisko de minimis), więc **NIE** zwraca `[]`. Refuse #5 musi przyjść z promptu LLM (cite-or-admit, Faza 3), nie z progu. Próg łapie tylko czyste OOD typu „Mongolia". → Faza 2 akcept.: 5 pytań trafia w dobre chunki; OOD-Mongolia → relevance ~0.

## Faza 3 — Lokalizacja promptów na PL — S [R] (commit 6909bec)
- [x] 3.1 `qa_system_prompt` → PL + decyzja 2b (usunięta instrukcja „append Citations" — cytaty tylko z `_format_citations`)
- [x] 3.2 Refusal → „Nie wiem — brak podstawy w dokumentach." (cudzysłów typograficzny „…”, nie ASCII — był 1 bug składni, naprawiony)
- [x] 3.3 `contextualize_q_system_prompt` → PL
- [x] 3.4 Tytuł/nagłówek + spinner + nagłówki Odpowiedź/Źródła + „page→strona" → PL
- [x] **3.R Podwójna kontrola — `/dev-docs-review`** → subagent: 0 blockerów, akceptacja. Nity Fazy 5: tooltip slidera wciąż EN.
- _Akceptacja:_ ✅ prompty i refusal naturalne po polsku, składnia/encoding OK, CI zielone

## Faza 4 — Refactor pipeline + debug panel — L [R] (commit d84e475)
- [x] 4.1 Ręczny rewrite (history-aware tylko gdy jest historia)
- [x] 4.2 `retrieve_scored()` w rag_index.py — jeden embed, relevance via `_select_relevance_score_fn` (= retriever) + raw L2. 🔵 NOWE: `select_context()` z limitem 2/dokument → naprawia pułapkę de minimis (top-4 było całe z Biznesmax/80%, teraz dochodzi dok de minimis/60%)
- [x] 4.3 Zero powyżej progu → „Nie wiem — brak podstawy w dokumentach", LLM pominięty
- [x] 4.4 Ręczny stuff + LLM history-aware; tura zapisywana w obu gałęziach (odpowiedź i refuse)
- [x] 4.5 Debug panel: rewritten query, relevance+L2, ✅/➖/✗ (do LLM / odcięte limitem / poniżej progu), próg, model, latencje (rewrite/retrieval/LLM)
- [x] 4.6 Cytaty z `used` (nie `scored`) + ścieżka uploadów nietknięta
- [x] **4.R Podwójna kontrola — `/dev-docs-review`** → subagent: 0 blockerów. Naprawione: A (podwójny embed+kruchy zip → jeden embed), B (żywa ref historii → snapshot `list()`), C (opis Info/Limits), E (przypis o limicie). D (quick-buttons EN) → Faza 5.
- _Akceptacja:_ ✅ score'y spójne z retrieverem (identyczne przed/po fixie A); 5/5 pytań ma właściwy dok w kontekście; OOD→refuse. ⚠️ ścieżka LLM niezweryfikowana bez GROQ_API_KEY → walidacja na żywo w Fazie 6

## Faza 5 — Quick-buttons + sanity de minimis — S [R] (commit b2ded3c)
- [x] 5.1 3 przyciski EN → **5 pytań demo PL** (`DEMO_QUESTIONS`, full-width, key=demo_q_i). Bonus: pełna lokalizacja UI (sidebar, radio, upload, czat, Wyślij, Info, komunikaty błędów)
- [x] 5.2 de minimis 60%: retrieval potwierdzony (dok de minimis z „nie większy niż 60%" w kontekście Q2/Q4). ⚠️ tekst odpowiedzi „60%" do potwierdzenia na żywo (Faza 6, wymaga LLM)
- [x] **5.R Podwójna kontrola — `/dev-docs-review`** → subagent: 0 blockerów. Naprawione: 🟠 angielski warning w get_demo_index (linia 105), nit slidera „(relevance)". „(history-aware)" zostawione (terminologia debug panelu). Bug cudzysłowów NIE wystąpił (sweep czysty).
- _Akceptacja:_ ✅ przyciski wstawiają 5 pytań demo; pytania zgodne ze spec; UI po polsku; #2 ma źródło 60% w kontekście (tekst → Faza 6)

## Faza 6 — Test E2E + commit + deploy — M
- [ ] 6.1 `streamlit run app.py` — przeklik 5 pytań + debug + refusal #5
- [ ] 6.2 CI smoke: `compileall` + import test
- [ ] 6.3 Commit faz na branchu; merge do main po akceptacji
- [ ] 6.4 Deploy HF Space — świadomie (5-15 min rebuild, pilnuj pickle/runtime)
- _Akceptacja:_ demo E2E lokalnie OK; CI zielone; deploy = osobna decyzja

---

## Notatki postępu
- 2026-06-04: plan utworzony. Środowisko (.venv) naprawione po przerwanej instalacji. Korpus 6 PDF już w `assets/` (niezacommitowany). Kod jeszcze nietknięty.
- 2026-06-04 (cd.): **Faza 1 ✅** (walidacja korpusu). **Bramka 1 planu Fazy 4** wykonana z wyprzedzeniem: `refactor-planner` → plan refactoru debug panelu, `plan-reviewer` → adversarial review (kluczowe poprawki: użyć **publicznego** `similarity_search_with_relevance_scores` zamiast prywatnego `_select_relevance_score_fn`; usunąć instrukcję cytatów z `qa_system_prompt` by uniknąć podwójnych cytatów; wydzielić czystą `score_documents` do `rag_index.py` jako pierwszy unit test; try/except na rewrite). **Step 0 ✅** (`_diag_step0.py`) — porównanie 3 modeli na korpusie BGK z polskimi diakrytykami → wybór paraphrase-multilingual (dowody w Fazie 2). **Faza 2 kod ✅** (EMB_MODEL + normalizacja + podpisy z importu), compileall OK, czeka na 2.R.
- ⚠️ Korekta planu: refuse #5 idzie z promptu (Faza 3), nie z progu (Faza 2) — dowód w Step 0.
- 🔧 Środowisko: lokalnie zainstalowano Python 3.11.9 (winget) — stare piny (numpy 1.26) nie mają wheeli na 3.13; 3.11 = zgodność z HF Space/CI.
