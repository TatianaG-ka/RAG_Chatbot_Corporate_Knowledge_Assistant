---
title: RAG Chatbot — Corporate Knowledge Assistant
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py        
pinned: false
---


# Project 1 – Company Knowledge Assistant — RAG (PDF/TXT/MD) Demo  
**ChatGroq + FAISS + Citations**

Each project follows the same consistent format: **overview → tech stack→ architecture → demo → how to run the Project → Screenshots**.
---
<a id="toc"></a>
## Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Planned Solution & Architecture](#solution-architecture)
- [What builds the FAISS index?](#what-builds-faiss-index)
- [Technologies Used](#technologies-used)
- [Demo](#demo)
- [How to Run the Project](#how-to-run)
- [Screenshots](#screenshots)

---

<a id="project-overview"></a>
## Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot that enables users to query internal company documentation (PDF/TXT/MD, etc.) in natural language.
The assistant returns answers **with source citations** and maintains **context** of the conversation.

---

<a id="problem-statement"></a>
## Problem Statement

In many organizations, knowledge is scattered across multiple formats — PDF/TXT/MD, internal wikis, and emails.  
Standard chatbots lack access to these sources, often providing shallow or hallucinated answers.  

A more intelligent **Company Knowledge Assistant** is needed — one that can answer user questions in natural language, **cite internal document sources**, and **maintain conversational context**.

---

<a id="solution-architecture"></a>
## Planned Solution & Architecture

- **Ingestion (offline)**: `build_demo_index.py` parses files from `./assets` (PDF/TXT/MD), splits them into chunks, generates embeddings (HF: `sentence-transformers/all-MiniLM-L6-v2`), and saves a FAISS index to `./vectorstore/default_company/`.

- **Retrieval**: the user’s query is (optionally) **rewritten** into a standalone question (history-aware), and then a `retriever (top-k)` is used to fetch the relevant context.

- **Generation (LLM)**: the **Groq** model (ChatGroq) receives the context + prompt and generates the answer.

- **Memory**: I use **ChatMessageHistory** (LangChain `RunnableWithMessageHistory`) to keep conversational history.

- **UI**: Streamlit (locally or on **Hugging Face Spaces**), with **Quick demo (prebuilt index)** and **Upload (session-memory index)** modes.

---

## Architecture Diagram
```
[User (Streamlit UI on Hugging Face Spaces)]
        |
        v
[LangChain Retriever + Memory]
        |
        v
[Vector DB: FAISS]
        |
[Retrieved Context]
        v
[LLM (Groq)]
        |
[Response + Citations]
        v
[UI Display]

```
---

<a id="what-builds-faiss-index"></a>
#### What builds the FAISS index?

*Note*: The FAISS index is built using the `build_demo_index.py` script (which imports utilities from `rag_index.py`).
The resulting index is saved under:

- `vectorstore/default_company/index.faiss` and `index.pkl` (current)
- or `vectorstore/default_company/faiss.index` and `docs.pkl` (legacy)

| The app accepts both variants.

#### File roles

- `rag_index.py` – shared utilities: loading assets, building/saving/loading FAISS, embeddings, split params, loaders (PDF/TXT/MD).

- `build_demo_index.py` – one-off script to build a demo index from `./assets/*` and save it to `./vectorstore/default_company`.

- `app.py` – Streamlit app. Uses `build_embeddings()`, and for uploads builds a session-only index via `build_faiss_from_docs()`; for the prebuilt demo it loads via `load_faiss()`.

---

<a id="technologies-used"></a>
## Technologies Used

| Component                                     | Role                                                |
| --------------------------------------------- | --------------------------------------------------- |
| **LangChain** (core/community/text-splitters) | Retrieval pipeline + prompts + conversation history |
| **FAISS**                                     | Vector search over context                          |
| **HuggingFace Embeddings**                    | `sentence-transformers/all-MiniLM-L6-v2`            |
| **ChatGroq**                                  | LLM (Groq API) for answer generation                |
| **Streamlit**                                 | UI locally / on Hugging Face Spaces                 |

---

<a id="demo"></a>
## Demo

- Live Demo (Hugging Face Spaces): https://huggingface.co/spaces/TetianaGol/RAG_Chatbot-Corporate_Knowledge_Assistant

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
2. (Quick demo) uses the **pre-built** index from the repo. 
   (Upload) allows you to upload your own files (PDF/TXT/MD) and build an index **in session memory**. 
3. Upload or use the example documents from the `/assets` folder.  
   - `manual.pdf`  
   - `faq.txt`  
   - `policy.md`  
4. Ask a question in natural language. Example demo questions:  
   - "How long does a refund take?"  
   - "How to reset my password?"  
   - "How to apply a software update?"  
5. View AI-generated responses with **citations** from your documents.

---

### **Option 2: Run Locally from GitHub**

If you prefer to run the project locally:

1. Clone the repository
`git clone https://github.com/TatianaG-ka/RAG_Chatbot_Corporate_Knowledge_Assistant`

2. Navigate to the project directory
`cd RAG_Chatbot-Corporate_Knowledge_Assistant`

3. Install dependencies
`pip install -r requirements.txt`

4. (Optional) Build the document index if you change files in ./assets
`python build_demo_index.py`

*Note:*
- The FAISS index is built using the `build_demo_index.py` script (which imports utilities from `rag_index.py`).
- This step only needs to be run once, or whenever new documents are added to the knowledge base.

5. Run the Streamlit app
`streamlit run app.py`

6. Then open your browser and go to
`http://localhost:8501`

7. Example Queries:
- "How long does a refund take?"
- "How to reset my password?"
- "How to apply a software update?"

---
<a id="screenshots"></a>
### Screenshots

![](./screenshots/demo_1.png)  

![](./screenshots/demo_2.png)  

![](./screenshots/demo_3.png)  


### License

MIT License © 2025 [Tatiana Golinska]




