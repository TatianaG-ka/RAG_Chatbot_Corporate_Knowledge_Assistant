from pathlib import Path
from typing import List
from langchain_core.documents import Document

from rag_index import (
    EMB_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    HAS_MD,
    build_embeddings,
    load_paths,
    build_faiss_from_docs,
    save_faiss,
)

# Paths
ASSETS = Path("./assets")
OUTDIR = Path("./vectorstore") / "default_company"


def load_assets() -> List[Document]:
    """
    Loads all supported files from the ./assets directory
    (PDF, TXT, MD/MARKDOWN).
    """
    paths = [p for p in ASSETS.glob("*")]
    if not paths:
        print("[WARN] No files found in ./assets. Please add files such as faq.txt, policy.md, or manual.pdf.")
        return []
    return load_paths(paths)


def main():
    print("[INFO] Configuration:")
    print(f"  - EMB_MODEL = {EMB_MODEL}")
    print(f"  - CHUNK_SIZE = {CHUNK_SIZE}, CHUNK_OVERLAP = {CHUNK_OVERLAP}")
    print(f"  - Markdown support: {'YES' if HAS_MD else 'NO (install unstructured)'}")

    emb = build_embeddings()
    raw_docs = load_assets()
    if not raw_docs:
        raise SystemExit("No documents found in ./assets. Aborted.")

    vs, n_chunks = build_faiss_from_docs(raw_docs, emb)
    print(f"[INFO] Created {n_chunks} text fragments (chunks).")

    save_faiss(vs, OUTDIR)
    print(f"[OK] Index saved to: {OUTDIR}/faiss.index and {OUTDIR}/docs.pkl")


if __name__ == "__main__":
    main()
