from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_community.document_loaders import UnstructuredMarkdownLoader
    HAS_MD = True
except ImportError:
    HAS_MD = False

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


@dataclass(frozen=True)
class IndexInfo:
    n_chunks: int
    outdir: Path


def build_embeddings() -> HuggingFaceEmbeddings:
    """Return the HF embeddings object. Call sites are expected to cache this."""
    return HuggingFaceEmbeddings(model_name=EMB_MODEL)


def split_docs(docs: List[Document]) -> List[Document]:
    """Split documents into coherent fragments for RAG."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def load_paths(paths: Iterable[Path]) -> Tuple[List[Document], List[str]]:
    """Load PDF/TXT/MD from given paths.

    Returns (loaded_documents, load_errors). Callers can surface errors in
    the UI instead of relying on stdout, which was the earlier behavior.
    """
    docs: List[Document] = []
    errors: List[str] = []
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
                    errors.append(
                        f"{p.name}: markdown loader requires the 'unstructured' package"
                    )
            else:
                errors.append(f"{p.name}: unsupported format '{sfx}'")
        except (FileNotFoundError, PermissionError, UnicodeDecodeError, ValueError) as e:
            errors.append(f"{p.name}: {type(e).__name__}: {e}")
    return docs, errors


def build_faiss_from_docs(docs: List[Document], emb) -> Tuple[FAISS, int]:
    """Build a FAISS store from documents. Returns (vectorstore, chunk_count)."""
    chunks = split_docs(docs)
    vs = FAISS.from_documents(chunks, emb)
    return vs, len(chunks)


def save_faiss(vs: FAISS, outdir: Path) -> IndexInfo:
    """Persist FAISS index to disk."""
    outdir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(outdir))
    return IndexInfo(n_chunks=-1, outdir=outdir)


def load_faiss(outdir: Path, emb) -> FAISS:
    # LangChain persists metadata via pickle. allow_dangerous_deserialization
    # is required for load and is safe here because the index is produced by
    # the same application that loads it. For user-uploaded indexes we rebuild
    # in-memory via build_faiss_from_docs instead of trusting disk pickles.
    return FAISS.load_local(
        str(outdir),
        embeddings=emb,
        allow_dangerous_deserialization=True,
    )


def ensure_demo_index_exists(outdir: Path) -> Optional[str]:
    """Accept both legacy and current LangChain filenames.

    - index.faiss / index.pkl  (current)
    - faiss.index / docs.pkl   (legacy)
    """
    possible_faiss = [outdir / "index.faiss", outdir / "faiss.index"]
    possible_meta = [outdir / "index.pkl", outdir / "docs.pkl"]

    has_faiss = any(p.exists() for p in possible_faiss)
    has_meta = any(p.exists() for p in possible_meta)

    if not (has_faiss and has_meta):
        return (
            f"No demo index in {outdir} "
            f"(expected one of: index.faiss|faiss.index and index.pkl|docs.pkl)."
        )
    return None
