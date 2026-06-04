# UI Improvements for Streamlit App

This file contains drop-in code snippets to upgrade the existing `app.py` with:

1. **Sidebar corpus info** — shows what's loaded so reviewers see scale immediately
2. **Debug expander** under each answer — retrieval scores, rewritten query, latency
3. **"I don't know" visual treatment** — yellow alert with retrieval scores
4. **Curated quick-buttons** — replacing the old generic 3 questions with the 8 from `demo_questions_guide.md`

Each snippet is independent — apply only the ones you want.

---

## 1. Sidebar corpus info

Add this to the sidebar, after the existing Embeddings line. Shows reviewers the scale of the knowledge base before they ask any question.

```python
# In app.py, inside the sidebar block:

import os
from pathlib import Path

def get_corpus_summary(assets_dir: str = "./assets") -> dict:
    """Inspect the assets folder and return a summary for sidebar display."""
    p = Path(assets_dir)
    if not p.exists():
        return {"files": 0, "details": []}
    files = []
    for f in sorted(p.iterdir()):
        if f.suffix.lower() in {".pdf", ".md", ".txt"}:
            size_kb = f.stat().st_size / 1024
            files.append({
                "name": f.name,
                "size_kb": round(size_kb, 1),
                "ext": f.suffix.lower(),
            })
    return {"files": len(files), "details": files}

# Usage in sidebar:
with st.sidebar:
    # ... existing widgets ...

    st.markdown("---")
    st.markdown("**📚 Demo corpus**")

    summary = get_corpus_summary()

    # Number of chunks comes from the actual index, not estimated
    if "demo_index" in st.session_state and st.session_state.demo_index is not None:
        # FAISS index stores docstore — count entries
        try:
            chunk_count = st.session_state.demo_index.index.ntotal
            st.caption(f"{summary['files']} documents · {chunk_count} chunks indexed")
        except Exception:
            st.caption(f"{summary['files']} documents loaded")
    else:
        st.caption(f"{summary['files']} documents loaded")

    for f in summary["details"]:
        st.caption(f"  • {f['name']} ({f['size_kb']} KB)")
```

---

## 2. Debug expander under every answer

This is the **most important upgrade** — it shows the retrieval mechanics that make this system different from a tutorial chatbot.

The debug expander needs three pieces of information that you must capture during retrieval:

- The **rewritten question** (output of `history_aware_retriever`)
- The **top-K chunks with their similarity scores** (use `similarity_search_with_score` instead of `similarity_search`)
- **Latency** (`time.time()` before and after the LLM call)

```python
# In app.py — modify the retrieval/answer code path:

import time

def answer_with_debug(question: str, chat_history, vectorstore, llm, threshold: float):
    """Run the retrieval+QA pipeline and return both the answer and debug info."""

    debug = {}
    t_start = time.time()

    # 1. History-aware question rewriting (if there's history)
    if chat_history:
        # Use your existing history_aware_retriever or call the rewriting chain explicitly
        rewritten = history_aware_chain.invoke({
            "input": question,
            "chat_history": chat_history,
        })
        debug["original_question"] = question
        debug["rewritten_question"] = rewritten
        retrieval_query = rewritten
    else:
        debug["original_question"] = question
        debug["rewritten_question"] = None  # no rewriting needed
        retrieval_query = question

    # 2. Retrieve with scores (NOT the .as_retriever wrapper)
    raw_results = vectorstore.similarity_search_with_score(
        retrieval_query,
        k=8,  # over-fetch so we can show what was filtered out
    )
    # FAISS returns (Document, distance) — convert distance to similarity
    # For default L2 distance: smaller is better; for cosine: directly usable
    scored = [
        {
            "doc": doc,
            "score": float(score),
            "passes_threshold": float(score) <= (1 - threshold)
                                if vectorstore.distance_strategy.name == "EUCLIDEAN_DISTANCE"
                                else float(score) >= threshold,
        }
        for doc, score in raw_results
    ]
    debug["retrieved_chunks"] = scored

    # 3. Filter to chunks above threshold
    used_chunks = [r for r in scored if r["passes_threshold"]]
    debug["used_chunk_count"] = len(used_chunks)
    debug["threshold"] = threshold

    # 4. If no chunks pass — return "I don't know" without calling LLM
    if not used_chunks:
        debug["latency_ms"] = round((time.time() - t_start) * 1000)
        debug["model"] = None
        return {
            "answer": None,  # signal to UI to show the "I don't know" alert
            "citations": [],
            "debug": debug,
        }

    # 5. Build context and call LLM
    context = "\n\n".join(r["doc"].page_content for r in used_chunks)
    answer = llm_chain.invoke({
        "context": context,
        "input": question,
        "chat_history": chat_history,
    })

    debug["latency_ms"] = round((time.time() - t_start) * 1000)
    debug["model"] = llm.model_name
    debug["context_chars"] = len(context)

    citations = [
        {
            "source": r["doc"].metadata.get("source", "unknown"),
            "page": r["doc"].metadata.get("page"),
            "score": round(r["score"], 3),
        }
        for r in used_chunks
    ]

    return {"answer": answer, "citations": citations, "debug": debug}


# UI rendering — replace existing answer rendering with:
def render_answer(result: dict):
    if result["answer"] is None:
        # I don't know — show alert
        render_idk_alert(result["debug"])
        return

    st.subheader("Answer")
    st.write(result["answer"])

    if result["citations"]:
        st.subheader("Citations")
        for c in result["citations"]:
            page = f", page {c['page']}" if c["page"] else ""
            st.markdown(f"- **{c['source']}**{page} — similarity {c['score']}")

    # Debug expander
    with st.expander("🔍 Retrieval debug"):
        d = result["debug"]
        if d.get("rewritten_question"):
            st.markdown(f"**Original question:** _{d['original_question']}_")
            st.markdown(f"**Rewritten question:** _{d['rewritten_question']}_")
            st.caption("(history-aware retriever invoked because chat has prior turns)")
        else:
            st.markdown(f"**Question (no history):** _{d['original_question']}_")

        st.markdown(f"**Threshold:** {d['threshold']} | **Chunks used:** {d['used_chunk_count']}")
        st.markdown(f"**Model:** `{d.get('model', 'n/a')}` | **Latency:** {d['latency_ms']} ms")

        if d.get("context_chars"):
            st.markdown(f"**Context size:** {d['context_chars']} chars")

        st.markdown("**Top chunks (passed/filtered):**")
        for r in d["retrieved_chunks"][:8]:
            mark = "✅" if r["passes_threshold"] else "✗"
            src = r["doc"].metadata.get("source", "?")
            page = r["doc"].metadata.get("page")
            page_str = f" p.{page}" if page else ""
            preview = r["doc"].page_content[:100].replace("\n", " ")
            st.text(f"{mark} {r['score']:.3f}  {src}{page_str}: {preview}...")
```

---

## 3. "I don't know" visual treatment

When the retriever returns no chunks above threshold, the system should **not** call the LLM (saves cost) and should display a clear, distinguishable alert.

```python
def render_idk_alert(debug: dict):
    """Render a clear yellow alert when no chunks pass the similarity threshold."""

    st.warning(
        "⚠️ **No relevant context found.**\n\n"
        f"The knowledge base does not appear to cover this topic. "
        f"The closest chunk in the corpus had a similarity below the threshold "
        f"(threshold: {debug['threshold']}). The system is configured to refuse "
        f"answering rather than guess from weak evidence."
    )

    # Still show what was almost-matched, so the reviewer can see why
    with st.expander("🔍 Why no answer? (debug)"):
        st.markdown(f"**Question:** _{debug['original_question']}_")
        if debug.get("rewritten_question"):
            st.markdown(f"**Rewritten:** _{debug['rewritten_question']}_")
        st.markdown(f"**Threshold:** {debug['threshold']}")
        st.markdown("**Top retrieved chunks (none passed):**")
        for r in debug["retrieved_chunks"][:5]:
            src = r["doc"].metadata.get("source", "?")
            preview = r["doc"].page_content[:100].replace("\n", " ")
            st.text(f"  {r['score']:.3f}  {src}: {preview}...")

        st.caption(
            "💡 Try rephrasing the question, or lower the threshold in the sidebar "
            "(at your own risk — lower threshold = higher hallucination risk)."
        )
```

---

## 4. Curated quick-buttons

Replace the old buttons with the 8 curated questions. Group them visually by category so the reviewer sees a structured demo path.

```python
DEMO_QUESTIONS = {
    "📋 Specific lookup with citation": [
        "What MFA methods are approved for admin accounts?",
        "Maximum gift value I can accept without declaring?",
    ],
    "🔗 Multi-document synthesis": [
        "What's our policy on working from a non-EU country during PTO?",
        "As a new hire, what do I need to set up in my first week?",
    ],
    "💬 History-aware follow-up (run as a pair)": [
        "What are the password requirements for admin accounts?",
        "Does that apply to contractors too?",  # follow-up
    ],
    "❓ I don't know (out of scope)": [
        "What's the company's stance on remote work in 2027?",
        "What's the CEO's email address?",
    ],
}

# Render in the main panel above the input box:
st.markdown("**Try a demo question:**")
for category, questions in DEMO_QUESTIONS.items():
    st.caption(category)
    cols = st.columns(len(questions))
    for col, q in zip(cols, questions):
        if col.button(q, key=f"qbtn_{q[:20]}"):
            st.session_state.pending_question = q
            st.rerun()
```

In the input handling, after `st.text_input(...)`, check for `st.session_state.pending_question` and use that as the question if set, then clear it.

---

## 5. Tiny UX touches

**Show retrieval scores inline with citations** (so the reviewer sees grounding strength):

```python
# Replace plain citation text with score-annotated:
for c in result["citations"]:
    page = f", page {c['page']}" if c["page"] else ""
    score_color = "🟢" if c["score"] >= 0.6 else ("🟡" if c["score"] >= 0.45 else "🟠")
    st.markdown(f"- {score_color} **{c['source']}**{page} — score {c['score']}")
```

**Show "system status" badge in the sidebar:**

```python
# In sidebar, after corpus info:
if st.session_state.get("demo_index") is not None:
    st.success("✓ Index loaded")
else:
    st.info("Index not yet built")
```

---

## Summary of what reviewer will now see

Before vs after, side by side:

| | Before | After |
| --- | --- | --- |
| Quick demo questions | 3 generic ("How long does refund take?") | 8 categorized, each demonstrating a feature |
| Answer display | Answer + citation file name | Answer + citation w/ similarity score + collapsible debug panel showing rewritten query, top chunks, threshold, latency, model |
| "I don't know" handling | Generic LLM response | Clear yellow alert + debug expander showing why no chunks passed |
| Sidebar | Just settings | Settings + corpus summary (files & chunk count) + system status badge |
| Conversation memory | Works invisibly | Visible in debug — reviewer sees query rewriting in action |

This UI rewrite alone takes the demo from "tutorial-grade" to "I can see this person built production RAG before."
