# Sections to Add — README & Notion Updates

These are drop-in sections to upgrade the existing GitHub README and Notion case study so they match the new senior-level demo.

---

## For GitHub README — new section "Demo corpus & query design"

Add this **after the "Why this project" section** and **before "Planned Solution & Architecture"**.

````markdown
## Demo corpus & query design

The demo runs against a synthetic but realistic corporate knowledge base — two policy documents totaling ~90 chunks across **Employee Handbook** (working hours, PTO, onboarding, conduct, benefits, termination) and **IT Security Policy** (accounts, MFA, BYOD, VPN, data classification, incident response). The corpus is small enough to deploy on the HF Spaces free tier but large enough to demonstrate four distinct retrieval behaviors:

**Multi-document synthesis** — questions whose correct answer spans both documents. Example: *"What's our policy on working from a non-EU country during PTO?"* requires combining Employee Handbook §2.3 (30-day allowance, no system access during PTO) with IT Security Policy §4.1 (sanctioned country VPN restrictions).

**History-aware follow-up** — questions that have no semantic anchor without prior chat context. Example: after asking about admin password requirements, the follow-up *"Does that apply to contractors too?"* is rewritten by `history_aware_retriever` into a self-contained query before retrieval.

**"I don't know" edge cases** — questions deliberately outside the corpus scope (e.g., *"What's the company's stance on remote work in 2027?"*). The threshold-based retriever returns `[]`, the LLM is never called, and a clear UI alert explains why.

**Specific lookup with citation** — questions with a single correct numeric or factual answer (e.g., *"Maximum gift value I can accept without declaring?"* → EUR 75 from Handbook §5). Tests grounded retrieval and precise citation rendering.

The eight curated demo questions are documented in [`docs/demo_questions_guide.md`](docs/demo_questions_guide.md) with full rationale for each. They are also wired into the Streamlit UI as quick-buttons, grouped by category, so reviewers can run a 5-minute architectural walkthrough by clicking through them in order.
````

---

## For GitHub README — update the existing "How to Run" demo questions

In the **"Option 1: Run on Hugging Face Spaces"** section, replace:

```markdown
4. Ask a question in natural language. Example demo questions:  
   - "How long does a refund take?"  
   - "How to reset my password?"  
   - "How to apply a software update?"  
```

With:

```markdown
4. Try the curated demo questions in the sidebar — they're grouped into four categories that each demonstrate a different RAG capability:

   **📋 Specific lookup with citation**
   - *"What MFA methods are approved for admin accounts?"*
   - *"Maximum gift value I can accept without declaring?"*

   **🔗 Multi-document synthesis** (answer combines both source documents)
   - *"What's our policy on working from a non-EU country during PTO?"*
   - *"As a new hire, what do I need to set up in my first week?"*

   **💬 History-aware follow-up** (run as a pair, in order)
   - *"What are the password requirements for admin accounts?"*
   - *"Does that apply to contractors too?"*

   **❓ I don't know — out of scope** (system refuses to hallucinate)
   - *"What's the company's stance on remote work in 2027?"*
   - *"What's the CEO's email address?"*

   Each answer includes a **🔍 Retrieval debug** expander showing the rewritten query (if history-aware), top chunks with similarity scores, threshold, latency, and model used.
```

---

## For GitHub README — new section "What the debug panel shows"

Add this section **after the "Key architectural decisions" section** and **before "Technologies Used"**. This is the section that converts an abstract ADR document into a "I can verify this works" demonstration.

````markdown
## What the debug panel shows (verifying the architecture)

Every answer in the UI includes a **🔍 Retrieval debug** expander that exposes the inner workings of the RAG pipeline. This is the single piece of UI that distinguishes a tutorial-grade demo from a production RAG system — it lets a reviewer verify each architectural decision claim against actual runtime behavior.

The debug panel shows:

| Field | What it demonstrates |
| --- | --- |
| **Original / Rewritten question** | History-aware retriever (ADR-2 derivative). When chat history exists, the rewritten query is shown; otherwise only the original. Reviewer can verify that *"Does that apply to contractors too?"* gets rewritten to *"Does the admin account password policy apply to contractors as well as employees?"* |
| **Threshold + chunks used** | Score threshold ADR-2 in action. Shows how many chunks passed the similarity threshold. For "I don't know" responses, this is `0`. |
| **Top retrieved chunks (passed + filtered)** | All top-K chunks with their similarity scores, marked ✅ if passed threshold or ✗ if filtered out. Shows the chunks that *almost* matched but were correctly rejected — a key transparency feature. |
| **Model + latency** | Which Groq model was used, and end-to-end latency. Validates ADR-3 (Groq for sub-2s response times). |
| **Context size** | Number of characters sent to the LLM as context. Shows the actual context-stuffing behavior. |

For "I don't know" responses, the panel additionally shows the highest similarity score that fell below threshold, so the reviewer can see exactly why no answer was generated.
````

---

## For Notion case study — replace "Co robi (workflow)" section

The current "Co robi" section is a 7-step pipeline description. Replace it with this — same content but reframed around what the **reviewer experiences** rather than what the system does mechanically:

````markdown
### Co robi system (z perspektywy użytkownika)

**Krok 1 — pytanie wpisane lub wybrane z quick-buttons.** UI ma 8 curated questions w 4 kategoriach (multi-document, history-aware, "I don't know", specific lookup), każda zaprojektowana, by przetestować inną cechę systemu.

**Krok 2 — przepisanie pytania, jeśli trzeba.** Jeśli w czacie była już rozmowa, `history_aware_retriever` przepisuje pytanie tak, żeby było samodzielne. Przykład: po rozmowie o admin passwords, pytanie "Does that apply to contractors too?" zostaje przepisane na "Does the admin account password policy apply to contractors as well as employees?". Bez tego kroku retriever zwracałby losowe chunki.

**Krok 3 — semantic search z thresholdem.** Pytanie embeddowane w 384-wymiarową przestrzeń (`sentence-transformers MiniLM`), top-K chunków pobranych z FAISS index. **Jeśli żaden chunk nie przekroczy similarity threshold 0.35** → retriever zwraca pustą listę, LLM nie jest wywoływany, użytkownik widzi "I don't know" alert. To jest **najważniejsza decyzja architektoniczna** w tym projekcie — chatbot musi odmówić odpowiedzi, gdy korpus nie wspiera pytania.

**Krok 4 — context-stuffing + grounded generation.** Chunki, które przeszły threshold, są wstawione jako context do system promptu. LLM (ChatGroq, domyślnie llama-3.1-8b-instant) generuje odpowiedź **wyłącznie na bazie tego kontekstu** — system prompt explicit instructs *"if the answer is not in the context, say 'I don't know'"*.

**Krok 5 — citations + debug panel.** Odpowiedź wyświetlona z cytacjami (plik źródłowy, numer strony, similarity score). Pod odpowiedzią rozwijany **debug panel** pokazuje: oryginalne i przepisane pytanie, threshold, top retrieved chunks (passed + filtered z ich scores), model, latency. To jest UI feature, który odróżnia tutorial-grade demo od production RAG — reviewer może zweryfikować każdą decyzję architektoniczną na żywym przykładzie.
````

---

## For Notion case study — new section "Demo corpus design"

Add this **after the "Co robi system" section**. To jest sekcja, która tłumaczy rekruterowi, **dlaczego korpus wygląda tak jak wygląda** — bo to jest świadoma decyzja, nie przypadek:

````markdown
### Korpus demo — świadomy design pod feature demonstration

Korpus ma celowo określoną strukturę: **2 dokumenty firmowe (~90 chunków łącznie)**, pokrywające typowe corporate documentation: Employee Handbook + IT Security Policy. Skala jest dobrana świadomie pod cztery cele:

- **Wystarczająco mały, żeby działać na HF Spaces free tier** — pełen index w pamięci, cold start <30s.
- **Wystarczająco duży, żeby pokazać multi-document retrieval** — pytania typu *"working from non-EU during PTO"* wymagają chunków z OBU plików.
- **Z wbudowanymi lukami pod "I don't know" demo** — np. brak info o roku 2027, brak email CEO. Pytania spoza scope są zaprojektowane.
- **Z conflicting/overlapping info** w 1-2 miejscach — żeby pokazać, jak bot radzi sobie z niejednoznacznością (oba dokumenty wspominają o contractorach z różnych perspektyw).

Każdy z 8 curated questions jest dopasowany do konkretnego punktu w korpusie i konkretnej cechy systemu. Lista pytań z uzasadnieniem jest w `docs/demo_questions_guide.md` w repozytorium GitHub.

**Świadoma decyzja: 2 mocne dokumenty zamiast 7 słabych.** Pierwsza wersja korpusu miała 7 plików po 1-2 strony — wyglądało imponująco z liczby plików, ale każdy plik miał za mało treści, żeby pokazać multi-document synthesis. Wersja z 2 grubymi dokumentami (~3000 słów każdy) lepiej demonstrate prawdziwy use case korporacyjny: pracownik pyta o coś, co wymaga połączenia info z handbook'u i security policy.
````

---

## For Notion — replace "Honest disclaimer (recruiter-friendly)" with stronger framing

Current sekcja jest dobra, ale można ją wzmocnić. Replace z:

````markdown
### Świadome decyzje skali (production-grade vs demo-grade)

Ten projekt to **clean RAG fundamentals na poziomie production-ready dla simple corpus** — nie production-scale dla enterprise corpus. Świadomie nie dodałam techniques, które miałyby sens przy znacznie większej skali, bo dla 90-chunk corpus są over-engineering:

- **Re-ranking drugim modelem (cross-encoder)** — przy 90 chunkach top-K wystarczająco dokładny. Sensowne przy 10k+ chunków.
- **Hybrid search BM25 + semantic** — przydatne dla kodów/numerów/exact-match. Dla policy text gdzie phrasing varies, sam semantic wystarcza.
- **Qdrant zamiast FAISS** — sensowne przy multi-tenant deployment z metadata filters. Single-tenant demo nie potrzebuje.
- **Eval framework z F1/recall@k** — wymagałby hand-labeled QA pairs, które nie skalują się dla side-projektu. **Manualnie testowałam wszystkie 8 curated questions** przed każdym deploymentem — to jest mój "lightweight eval".

W moim drugim projekcie (`invoice-processor`, scale: faktury miesięcznie z różnymi formatami i wysoką stawką błędu) zastosowałam część z tych technik świadomie — bo tam stakes są wyższe (błędna kwota = błąd księgowy z konsekwencjami legal). To jest celowa separacja: **tutorial-grade fundamentals tu, production-grade extensions w invoice-processor, i opisuję świadomie różnice**.

Reviewer, który zauważy tę separację, dostaje sygnał: ta osoba rozumie, kiedy dodać complexity, a kiedy świadomie jej nie dodawać. Senior decision-making.
````

---

## Summary of changes to make

In your GitHub README:

1. Add **"Demo corpus & query design"** section after "Why this project"
2. Replace generic 3 questions in "How to Run" with 8 categorized
3. Add **"What the debug panel shows"** section after Key Decisions
4. (Optional) Add screenshot of debug panel to `screenshots/` folder

In your Notion case study:

1. Replace "Co robi (workflow)" with new user-perspective version
2. Add **"Korpus demo — świadomy design"** section
3. Replace "Honest disclaimer" with stronger "Świadome decyzje skali" framing
4. Update screenshots showing new corpus + debug panel + curated quick-buttons

In your `app.py`:

1. Apply Streamlit improvements from `streamlit_ui_improvements.md`
2. Wire up the 8 curated questions as quick-buttons grouped by category
3. Test all 8 demo questions end-to-end before screenshot capture

In your `./assets/` folder:

1. Replace existing 3 files with `employee_handbook.md` and `it_security_policy.md`
2. Run `python build_demo_index.py` to rebuild index
3. Verify chunk count is ~90 (sidebar will show this)
````
