from pathlib import Path
from typing import List, Iterable, Tuple, Optional
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

try:
    from langchain_community.document_loaders import UnstructuredMarkdownLoader
    HAS_MD = True
except Exception:
    HAS_MD = False

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

@dataclass(frozen=True)
class IndexInfo:
    n_chunks: int
    outdir: Path

def build_embeddings():
    """Returns the prepared HF embeddings object."""
    return HuggingFaceEmbeddings(model_name=EMB_MODEL)

def split_docs(docs: List[Document]) -> List[Document]:
    """Splits documents into coherent fragments for RAG."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)

def load_paths(paths: Iterable[Path]) -> List[Document]:
    """Loads PDF/TXT/MD from given paths."""
    docs: List[Document] = []
    for p in paths:
        sfx = p.suffix.lower()
        try:
            if sfx == ".pdf":
                docs.extend(PyPDFLoader(str(p)).load())
            elif sfx == ".txt":
                docs.extend(TextLoader(str(p), encoding="utf-8").load())
            elif sfx in (".md", ".markdown"):
                if HAS_MD:
                    docs.extend(UnstructuredMarkdownLoader(str(p)).load())
                else:
                    print(f"[WARN] Omitted {p.name} — 'unstructured' package missing for MD.")
            else:
                print(f"[SKIP] Unsupported format: {p.name}")
        except Exception as e:
            print(f"[ERROR] Error while loading {p.name}: {e}")
    return docs


def build_faiss_from_docs(docs: List[Document], emb) -> Tuple[FAISS, int]:
    """Buduje FAISS z listy dokumentów. Zwraca (vectorstore, liczba_chunks)."""
    chunks = split_docs(docs)
    vs = FAISS.from_documents(chunks, emb)
    return vs, len(chunks)


def save_faiss(vs: FAISS, outdir: Path) -> IndexInfo:
    """Writes FAISS to disk."""
    outdir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(outdir))
    return IndexInfo(n_chunks=-1, outdir=outdir)


def load_faiss(outdir: Path, emb) -> FAISS:
    """
    Loads FAISS from disk. NOTE:
    LangChain saves metadata in docs.pkl (pickle),
    so we require allow_dangerous_deserialization=True.
    """
    return FAISS.load_local(
        str(outdir),
        embeddings=emb,
        allow_dangerous_deserialization=True
    )


def ensure_demo_index_exists(outdir: Path) -> str | None:
    """
    Accept both legacy and current LangChain filenames:
    - index.faiss / index.pkl (current)
    - faiss.index / docs.pkl (legacy)
    """
    possible_faiss = [outdir / "index.faiss", outdir / "faiss.index"]
    possible_meta  = [outdir / "index.pkl",   outdir / "docs.pkl"]

    has_faiss = any(p.exists() for p in possible_faiss)
    has_meta  = any(p.exists() for p in possible_meta)

    if not (has_faiss and has_meta):
        return (f"No demo index in {outdir} "
                f"(expected one of: index.faiss|faiss.index and index.pkl|docs.pkl).")
    return None

