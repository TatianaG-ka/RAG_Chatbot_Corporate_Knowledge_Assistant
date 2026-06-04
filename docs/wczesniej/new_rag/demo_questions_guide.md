# Demo Questions Guide

Curated set of 8 demo questions, designed to **systematically demonstrate** every architectural decision in this RAG system. Each question targets a specific technical capability — together they form a 5-minute walkthrough that takes a reviewer from "this is a tutorial chatbot" to "this person understands production RAG."

The corpus this is designed against:

- `employee_handbook.md` (~50 chunks worth of content covering working hours, PTO, onboarding, conduct, benefits, termination)
- `it_security_policy.md` (~40 chunks covering accounts, MFA, BYOD, VPN, data classification, incident response)

Each question below is also configured as a **quick-button** in the Streamlit UI sidebar, so reviewers can click rather than type.

---

## Category A — Multi-document synthesis

These questions force the retriever to pull chunks from **multiple source documents** and the LLM to synthesize a coherent answer. A naive top-K retriever over a single corpus would return noise; with multi-document grounding, the answer is precise and citations span 2+ files.

### A1. "What's our policy on working from a non-EU country during PTO?"

**Why this question:** the answer requires combining:
- *Employee Handbook §2.3* (working from a different country — 30-day allowance, no system access during PTO outside allowance)
- *IT Security Policy §4.1* (sanctioned countries blocked, ISO notification for >7 day stays)

Without multi-document retrieval, the bot would only quote one policy and miss the other. With multi-document, the answer references both — a clear demonstration that the system grounds across the corpus, not just within a single file.

**Expected answer characteristics:**
- Mentions the 30-day annual cumulative limit
- Mentions that accessing systems during PTO outside the allowance is not permitted (even briefly)
- Mentions VPN restrictions for sanctioned countries
- Citations from BOTH Employee Handbook AND IT Security Policy

---

### A2. "As a new hire, what do I need to set up in my first week?"

**Why this question:** broad onboarding question — the answer should pull from:
- *Employee Handbook §4.1* (Week 1 checklist — IT setup, HR docs, training, benefits, manager 1:1, buddy)
- *IT Security Policy §2* (account provisioning, MFA enrollment, password requirements)
- *IT Security Policy §6.1* (mandatory onboarding security training within 5 business days)

This question demonstrates that the bot can act as an **actual onboarding assistant** for a new employee — not just a Q&A toy.

**Expected answer characteristics:**
- Lists at least 4 items from Week 1 checklist
- References specific deadlines (5 business days, 14 days, etc.)
- Citations span Employee Handbook AND IT Security Policy

---

## Category B — History-aware follow-up

These questions test the `history_aware_retriever` chain step. Without query rewriting from chat history, the second question would retrieve random chunks. With history-aware rewriting, it retrieves the right section.

### B1. Two-turn conversation — IT Security follow-up

**Turn 1:** *"What are the password requirements for admin accounts?"*

Expected: lists 20-character minimum, 90-day rotation, vault storage, no reuse — all from IT Security Policy §2.2.

**Turn 2:** *"Does that apply to contractors too?"*

**Why this is important:** the bare question "Does that apply to contractors too?" has no semantic anchor — a naive retriever would return irrelevant chunks (everything that mentions "contractors" weakly).

With `history_aware_retriever`:
- Chat history says we're talking about admin password requirements
- Retriever rewrites the question into something like *"Does the admin account password policy apply to contractors as well as employees?"*
- The rewritten query retrieves *IT Security Policy §1* (scope and applicability — explicitly mentions contractors are subject in full from day one)

**Recruiter takeaway:** the bot handles natural conversational follow-ups, not just one-shot questions.

---

### B2. Two-turn conversation — PTO follow-up

**Turn 1:** *"How does PTO accrual work?"*

Expected: 2.0 days/month for full-time, proportional for part-time, accrual from day 1, up to 5 days advance.

**Turn 2:** *"And what happens to it if I don't use it by year-end?"*

**Why this is important:** "What happens to it" has no antecedent without the chat history. The history-aware retriever rewrites it to something like *"What happens to unused PTO days at the end of the calendar year?"* which retrieves *Employee Handbook §3.2* (carry-over rules, March 31 deadline, no cash-out).

---

## Category C — "I don't know" edge cases

These questions are **not answerable from the corpus** — by design. They test whether the system honestly admits ignorance instead of hallucinating.

### C1. "What's the company's stance on remote work in 2027?"

**Why this question:** the corpus describes current 2026 policies but says nothing about 2027 future direction. A naive system would extrapolate or make up a "based on current trends..." answer. The threshold-based retriever should return `[]` (no chunks above 0.35) or near-empty, and the system prompt should produce *"I don't know"*.

**Expected answer characteristics:**
- Response: *"I don't know"* or equivalent ("the knowledge base doesn't contain information about...")
- Debug panel shows top retrieval scores all below threshold
- No fabricated speculation about 2027

**Recruiter takeaway:** this is the single most important behavioral test for a corporate knowledge assistant. Confidence in admitting "I don't know" is what makes the tool trustworthy in real deployment.

---

### C2. "What's the CEO's email address?"

**Why this question:** the corpus deliberately does not contain executive contact info. Generic helpdesk emails (`it-helpdesk@`, `people@`, `security-incident@`) appear, but no CEO contact exists in the documents. A weak system would return one of those generic emails as a "best guess" — wrong.

**Expected answer characteristics:**
- Response: *"I don't know"* — no CEO email is documented in the knowledge base
- Optionally: suggestion to use the People Operations channel for HR-related escalation

---

## Category D — Specific lookup with citations

These questions test the **grounded answer with citation** capability. Each has a specific, verifiable correct answer in the corpus, and the citation should point to the exact section.

### D1. "What's the maximum value of a gift from a supplier that I can accept without declaring it?"

**Why this question:** specific number, single source.

**Expected answer:**
- *"EUR 75 — gifts above this value must be declared"*
- Citation: *Employee Handbook §5* (Code of Conduct summary)

This demonstrates that the bot retrieves precise factual information and cites the exact section. Critical for compliance and policy lookups where "approximately" is not acceptable.

---

### D2. "What MFA methods are approved for privileged accounts?"

**Why this question:** specific technical detail, with implicit ranking.

**Expected answer:**
- Hardware security keys (YubiKey, Google Titan) — required for privileged accounts
- Authenticator app — accepted but lower preference
- SMS-based MFA explicitly prohibited
- Citation: *IT Security Policy §2.3*

Demonstrates retrieval of structured technical content with correct hierarchy preserved.

---

## Suggested demo flow (5 minutes)

For a live walkthrough during an interview, run questions in this order:

1. **D2** (specific lookup) — fastest visible win, shows citations work, ~10 seconds
2. **B1 turn 1, then B1 turn 2** — demonstrates history-aware retriever, the most "magical" feature, ~30 seconds
3. **A1** (multi-document) — shows synthesis capability, ~30 seconds
4. **C1** (I don't know) — shows the system refuses to hallucinate; show the debug panel here, ~30 seconds
5. **A2** (broad onboarding) — closing question that shows real-world utility, ~45 seconds

Each step takes 30–60 seconds including the time for reviewers to read the answer and citations. Total: under 5 minutes for the full architectural tour.

---

## Adding these as quick-buttons in Streamlit UI

In `app.py`, replace the existing quick-button list with:

```python
DEMO_QUESTIONS = [
    # Specific lookup
    ("📋 What MFA methods are approved for admin accounts?", "specific"),
    ("📋 Max gift value I can accept without declaring?", "specific"),
    # Multi-document
    ("🔗 Working from non-EU country during PTO?", "multi-doc"),
    ("🔗 What do I need to set up in my first week?", "multi-doc"),
    # History-aware (these need to be triggered as conversational pairs)
    ("💬 Password requirements for admin accounts?", "history-1"),
    ("💬 Does that apply to contractors too?", "history-2"),
    # I don't know
    ("❓ Company stance on remote work in 2027?", "idk"),
    ("❓ What's the CEO's email?", "idk"),
]
```

The category tag is for internal use — UI shows only the question text. The categorization helps you maintain the curated set as the corpus evolves.
