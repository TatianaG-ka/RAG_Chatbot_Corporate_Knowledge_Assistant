from pathlib import Path
from typing import List

from langchain_core.documents import Document

from rag_index import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMB_MODEL,
    HAS_MD,
    build_embeddings,
    build_faiss_from_docs,
    load_paths,
    save_faiss,
)

ASSETS = Path("./assets")
OUTDIR = Path("./vectorstore") / "default_company"
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}


def load_assets() -> List[Document]:
    """Load all supported files from ./assets (PDF / TXT / MD)."""
    paths = [p for p in ASSETS.glob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES]
    if not paths:
        print(
            "[WARN] No supported files found in ./assets "
            "(expected .pdf/.txt/.md). Please add faq.txt, policy.md, or manual.pdf."
        )
        return []

    docs, errors = load_paths(paths)
    for err in errors:
        print(f"[WARN] {err}")
    return docs


def main() -> None:
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
    print(f"[OK] Index saved to: {OUTDIR}/index.faiss and {OUTDIR}/index.pkl")


if __name__ == "__main__":
    main()
