# RAG Chatbot Upgrade — decyzja do podjęcia

**Data analizy:** 2026-05-08
**Status:** ⏸️ DECYZJA WSTRZYMANA — wracam jutro
**Kontekst:** kolega zaproponował upgrade RAG chatbota (HF Space `TatianaGol/RAG_Chatbot-Corporate_Knowledge_Assistant`)

---

## 0. TL;DR — co decydujemy

Czy zrobić **pełny upgrade RAG** (4 zmiany) na osobnym branchu **przed apply** (lub między Multimodal Loom a apply), czy **odłożyć na post-apply** jako LinkedIn content?

**Moja rekomendacja:** Wariant B (sekcja 6) — pełny upgrade lokalnie na branchu **bez deploy** przed apply, deploy + Loom video po apply.

---

## 1. Co zaproponował kolega

Pliki w `invoice_processor/docs/new_rag/` (4 markdown, drop-in):

### 1.1 Dwa synthetic corpora (cross-referenced)

- `employee_handbook.md` — 9 sekcji (working hours, PTO, onboarding, conduct, benefits, termination)
- `it_security_policy.md` — 9 sekcji (accounts, MFA, BYOD, VPN, data classification, incident response)

**Co czyni je smart:** wewnętrzne cross-references między dokumentami:
- PTO §2.3 (working from non-EU country) → IT Security §4.1 (sanctioned countries)
- Onboarding §4.1 (Week 1 IT setup) → IT Security §6.1 (training within 5 business days)
- Code of Conduct §5 (gift threshold EUR 75) — single-source for citation precision test

### 1.2 8 curated demo questions (`demo_questions_guide.md`)

4 kategorie × 2 pytania, każda mapuje 1:1 do features RAG:

| Kategoria | Pytanie | Co testuje |
| --- | --- | --- |
| **A. Multi-doc synthesis** | "Working from non-EU country during PTO?" | Retrieval z 2 plików, citations span both |
| | "First-week onboarding setup?" | Pull z 2-3 sekcji 2 plików |
| **B. History-aware** | T1: "Password requirements for admin?" → T2: "Does that apply to contractors?" | `history_aware_retriever` rewrite query |
| | T1: "How does PTO accrual work?" → T2: "What if I don't use it?" | Antecedent resolution |
| **C. "I don't know"** | "Stance on remote work in 2027?" | Out-of-corpus refusal |
| | "CEO's email?" | No fabrication of generic emails |
| **D. Specific lookup** | "Max gift value without declaring?" | Precise number + cite (EUR 75) |
| | "MFA methods for privileged accounts?" | Hierarchy preserved (YubiKey > Authenticator > push, SMS prohibited) |

### 1.3 Debug expander Streamlit (`streamlit_ui_improvements.md`)

Pod każdą odpowiedzią collapsible panel pokazujący:
- **Original question** vs **rewritten question** (po `history_aware_retriever`)
- Top-K chunks ze **scorami** (✅ passed threshold / ✗ filtered)
- Threshold użyty + chunk count
- Model + latency_ms + context_chars

### 1.4 Pozostałe 3 zmiany Streamlit

- Sidebar corpus info (files + chunk count)
- "I don't know" yellow alert (skip LLM jeśli zero chunków powyżej threshold = oszczędność kosztu)
- 8 quick-buttons grupowane po kategorii w main panelu

---

## 2. Weryfikacja techniczna — FAISS distance_strategy

### 2.1 Stan obecny w repo

`rag_index.py:74`:
```python
vs = FAISS.from_documents(chunks, emb)  # bez distance_strategy=
```

→ LangChain default = **`EUCLIDEAN_DISTANCE` (L2)**, NIE cosine.

`app.py:240-243` retriever:
```python
return store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 4, "score_threshold": threshold},
)
```

LangChain wewnętrznie konwertuje L2 → similarity (0-1) i porównuje z threshold. **User widzi threshold=0.35 jako similarity score** w sliderze.

### 2.2 Bug w propozycji kolegi (linia 113-114 `streamlit_ui_improvements.md`)

```python
"passes_threshold": float(score) <= (1 - threshold)
                    if vectorstore.distance_strategy.name == "EUCLIDEAN_DISTANCE"
                    else float(score) >= threshold,
```

To **uproszczenie**. `similarity_search_with_score()` zwraca **raw L2 distance** (nie similarity!). Porównanie z `(1 - 0.35) = 0.65` to różna semantyka niż obecny retriever.

**Konsekwencja jeśli zostawimy as-is:** debug panel pokazuje "passed/filtered" które NIE matchuje rzeczywistości retrievera. Czyli recruiter widzi inconsistency między tym co jest w panelu a tym co LLM dostał. Senior signal **odwraca się w junior signal**.

### 2.3 Fix (5 min)

Użyć tej samej funkcji konwersji co LangChain wewnątrz:

```python
# Replace lines 109-118 in proposed code:
relevance_fn = vectorstore._select_relevance_score_fn()
scored = []
for doc, distance in raw_results:
    relevance = relevance_fn(distance)  # convert L2 → 0-1 similarity
    scored.append({
        "doc": doc,
        "raw_distance": float(distance),
        "score": float(relevance),         # 0-1 similarity (matches slider)
        "passes_threshold": relevance >= threshold,
    })
```

Bonus: pokazujemy w panelu BOTH `raw_distance` (L2) AND `score` (relevance). To dodatkowy senior signal — "rozumiem różnicę między distance a similarity i dlaczego LangChain konwertuje".

---

## 3. Czy potrzebny debug expander? **TAK.**

### 3.1 Test recruiterski

**Bez debug:** input → answer → cytacja. Wniosek: *"OK, kolejny RAG"*.

**Z debug:** input → rewritten query (pokazuje history-aware działa) → top-K chunks ze scorami → threshold filter → final answer + cytacje. Wniosek: *"ta osoba rozumie retrieval mechanics, nie tylko `RetrievalQA.from_chain_type()`"*.

### 3.2 Specyficznie dla "I don't know" cases

- **Bez debug:** "I don't know" wygląda jak bug. Recruiter myśli *"może wadliwie działa?"*
- **Z debug:** widać że wszystkie scores są poniżej threshold → system **honestly refused**. To dokładnie ten **trustworthy AI signal** którego szuka IN4GE (AI Integration role).

### 3.3 Wniosek

Korpus + quick-buttons bez debug = visual upgrade.
Z debug = architectural upgrade.

**Debug expander = ~60% wartości całej propozycji.**

---

## 4. Estymacja kosztu pełnego upgrade'u

| Krok | Czas | Ryzyko |
| --- | --- | --- |
| Korpus swap (`policy.md`/`faq.txt`/`manual.pdf` → 2 nowe pliki) + rebuild FAISS index | 45 min | Niskie |
| 8 quick-buttons + sidebar corpus info | 30 min | Niskie |
| "I don't know" yellow alert (skip LLM gdy zero chunków) | 30 min | Niskie |
| **Debug expander** (capture rewritten query + raw scores + latency) | **2-3h** | **Wysokie** — obecny `create_retrieval_chain` jest opaque, prawdopodobnie trzeba zrefaktorować na ręczny pipeline żeby wyciągnąć rewritten query i raw scores |
| Score conversion fix (sekcja 2.3 powyżej) | 15 min | Niskie |
| Testowanie 8 questions lokalnie | 30 min | Średnie |
| HF Space deploy + ewentualny pickle compat fix | 30-60 min | **Wysokie** (memory `reference_hf_spaces_gotchas`) |
| Retake 3 screenshotów (`screenshots/demo_1-3.png`) | 30 min | Niskie |
| Update Notion sub-page B/RAG (jeśli mówi o starym corpusie) | 15 min | Niskie |

**Total realistic: 5-8h** (jeden pełny dzień jeśli coś się zatnie).

---

## 5. Stan apply readiness — kontekst decyzji

**Apply:** 2026-05-08 (DZIŚ). Sprawdzić czy już aplikowała czy nie.

**Critical path z CLAUDE.md (Day 11+):**
1. ⏳ Multimodal Loom recording (W TRAKCIE)
2. ⏳ git push 4 commitów (`353b4e1`/`acdb115`/`ebb3843`/`7b08438`)
3. ⏳ IN4GE Mapping doc
4. ⏳ Pass 2 audit

Łącznie minimum: ~1 dzień solid pracy + buffer.

---

## 6. Dwa warianty decyzji

### Wariant A — Safe (minimum viable)

Branch `feature/recruiter-grade-upgrade` w `_PLANNING_IN4GE/inne_projekty_reference/project_1/RAG_Chatbot-Corporate_Knowledge_Assistant/`:
- ✅ Korpus swap (2 nowe cross-referenced pliki)
- ✅ 8 quick-buttons + sidebar corpus info
- ✅ "I don't know" yellow alert
- ❌ Debug expander **POMIJAMY**

**Czas:** ~2-3h
**Ryzyko:** zero
**Wartość portfolio:** visual win, **brak architectural senior signal**
**Deploy:** kiedy chcesz, niskie ryzyko

### Wariant B — Ambitious (rekomendowany)

Branch z **PEŁNYM upgrade'em w lokalnym repo, BEZ deploy do HF Space przed apply:**
- ✅ Wszystko z Wariantu A
- ✅ Debug expander (Z fixem score conversion z sekcji 2.3)
- ✅ Testowanie lokalne
- ❌ NO HF Space deploy przed apply
- ❌ NO Notion update przed apply

**Po apply (8.05+):**
- merge → push do HF Space → retake screenshots → nagraj **3-cie Loom video** *"RAG deep dive — what production RAG looks like"*
- LinkedIn post z trzema wideami pre-apply + RAG deep dive jako "just shipped"
- Interview talking point: *"właśnie wczoraj upgrade'owałam RAG observability — let me show you the debug panel"*

**Czas:** 5-8h (rozłożone na 2-3 dni między Loom recordings)
**Ryzyko dla apply:** **zero** (nic nie deployujemy)
**Wartość portfolio:** maksymalna — architectural senior signal + świeży LinkedIn content

### Czego NIE robić

❌ Pełny upgrade + deploy do HF Space + retake screenshotów + update Notion **przed apply**.

Powody:
- HF Space rebuild = 5-15 min waiting + ryzyko pickle incompat
- Subtelny bug w debug expanderze → recruiter klika → widzi traceback → minus dla aplikacji
- Czas wyrwany z critical path (Multimodal Loom, IN4GE Mapping, Pass 2)

---

## 7. Action plan jutro (jeśli decyzja = Wariant B)

### Krok 1 — założenie branchu (5 min)

```powershell
cd C:\Users\tatia\PROJECTS\_PLANNING_IN4GE\inne_projekty_reference\project_1\RAG_Chatbot-Corporate_Knowledge_Assistant
git status                                          # sprawdź clean state
git checkout -b feature/recruiter-grade-upgrade
```

### Krok 2 — korpus swap (45 min)

```powershell
# Skopiuj nowe corpora
Copy-Item C:\Users\tatia\PROJECTS\invoice_processor\docs\new_rag\employee_handbook.md .\assets\
Copy-Item C:\Users\tatia\PROJECTS\invoice_processor\docs\new_rag\it_security_policy.md .\assets\

# Usuń stare
Remove-Item .\assets\policy.md
Remove-Item .\assets\faq.txt
Remove-Item .\assets\manual.pdf

# Rebuild prebuilt index — najprościej uruchomić app.py w "Upload mode" + zapisać
# albo dodać helper script `scripts/rebuild_demo_index.py` który robi:
#   docs, _ = load_paths([Path("assets/employee_handbook.md"), Path("assets/it_security_policy.md")])
#   vs, n = build_faiss_from_docs(docs, build_embeddings())
#   save_faiss(vs, Path("vectorstore/default_company"))
```

### Krok 3 — quick-buttons + sidebar (30 min)

Z `streamlit_ui_improvements.md` sekcja 1 (sidebar corpus info) + sekcja 4 (quick-buttons).
Wklej w `app.py` zastępując obecne `examples = [...]` (linia 153-160).

### Krok 4 — "I don't know" alert (30 min)

Z `streamlit_ui_improvements.md` sekcja 3 (`render_idk_alert`).
Trigger w `app.py` w bloku po `conv.invoke()` jeśli `result.get("context", [])` jest puste.

### Krok 5 — debug expander (2-3h, najtrudniejsze)

**UWAGA — wymaga refaktoringu obecnego pipeline'u:**

Obecny `create_retrieval_chain(history_aware_retriever, doc_chain)` jest opaque — nie zwraca rewritten query ani raw scores. Trzeba rozbić na:

```python
# 1. Manual rewrite step (capture rewritten_question)
if chat_history:
    rewritten = (contextualize_q_prompt | llm | StrOutputParser()).invoke({
        "input": query, "chat_history": chat_history
    })
else:
    rewritten = query

# 2. Manual retrieval with scores (use fix from sekcja 2.3 above)
raw_results = vs.similarity_search_with_score(rewritten, k=8)
relevance_fn = vs._select_relevance_score_fn()
scored = [...]  # apply fix from sekcja 2.3
used = [r for r in scored if r["passes_threshold"]]

# 3. If nothing passes — render IDK alert, skip LLM
if not used:
    render_idk_alert(...)
    return

# 4. Manual stuff documents + LLM call
context = "\n\n".join(r["doc"].page_content for r in used)
answer = (qa_prompt | llm | StrOutputParser()).invoke({
    "context": context, "input": query, "chat_history": chat_history
})

# 5. Render answer + citations + debug expander
```

To największy chunk pracy. Test inkrementalnie — najpierw bez history (turn 1), potem z history (turn 2).

### Krok 6 — testowanie 8 questions lokalnie (30 min)

`streamlit run app.py` lokalnie, przeklik wszystkich 8 quick-buttons + check debug panel pokazuje sensible scores.

### Krok 7 — STOP. Nie deployuj. Nie commituj do main.

Branch zostaje na disku. Po apply:
- merge do main
- push do HF Space (poczekaj 5-15 min na rebuild)
- retake screenshots
- nagraj Loom RAG deep dive
- LinkedIn post

---

## 8. Pliki źródłowe (do których wracam)

- `C:\Users\tatia\PROJECTS\invoice_processor\docs\new_rag\demo_questions_guide.md` — 8 questions
- `C:\Users\tatia\PROJECTS\invoice_processor\docs\new_rag\employee_handbook.md` — corpus 1
- `C:\Users\tatia\PROJECTS\invoice_processor\docs\new_rag\it_security_policy.md` — corpus 2
- `C:\Users\tatia\PROJECTS\invoice_processor\docs\new_rag\streamlit_ui_improvements.md` — drop-in code (z bugiem do fixu sekcja 2.3)
- `C:\Users\tatia\PROJECTS\_PLANNING_IN4GE\inne_projekty_reference\project_1\RAG_Chatbot-Corporate_Knowledge_Assistant\app.py` — current Streamlit app
- `C:\Users\tatia\PROJECTS\_PLANNING_IN4GE\inne_projekty_reference\project_1\RAG_Chatbot-Corporate_Knowledge_Assistant\rag_index.py` — FAISS wrapper

---

## 9. Otwarte pytania na jutro

1. **Czy aplikowałaś już do IN4GE?** Jeśli tak → Wariant B znika z presji, można robić w spokoju. Jeśli nie → potwierdza Wariant B.
2. **Multimodal Loom DONE czy nadal w trakcie?** Jeśli DONE → więcej buforu na RAG upgrade.
3. **Czy chcesz 3-cie Loom video po apply?** Jeśli nie → Wariant A wystarczy (mniej pracy, brak Loom = brak retake screenshots pressure).

---

**Last updated:** 2026-05-08, sesja Claude Code (Opus 4.7 1M)
**Next session:** wracamy do tego pliku, zaczynamy od sekcji 9 (3 pytania) → wybór wariantu → krok 1 z action planu
