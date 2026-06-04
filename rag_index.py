from collections import defaultdict
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

# Multilingual model: the demo corpus is Polish (public BGK documents), and the
# English-only all-MiniLM-L6-v2 gave poor retrieval + compressed scores on Polish
# queries. paraphrase-multilingual-MiniLM-L12-v2 ranks Polish chunks correctly AND
# keeps clean in-corpus vs out-of-corpus separation (out-of-corpus relevance goes
# ~0), which the similarity threshold and debug panel rely on. (e5-base ranks well
# but compresses all scores into ~0.65-0.83, so a threshold can't reject nonsense.)
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


@dataclass(frozen=True)
class IndexInfo:
    n_chunks: int
    outdir: Path


@dataclass(frozen=True)
class ScoredChunk:
    """A retrieved chunk with both score representations.

    relevance: 0-1 similarity, the SAME number the sidebar threshold compares
               against and the same value the similarity_score_threshold
               retriever uses internally — so a debug panel built from this is
               guaranteed to agree with what the chain actually feeds the LLM.
    distance:  raw L2 distance straight from FAISS (lower = closer). Shown
               alongside relevance to make the distance->similarity conversion
               explicit rather than hand-wavy.
    """

    doc: Document
    relevance: float
    distance: float


def retrieve_scored(vs: FAISS, query: str, k: int = 8) -> List[ScoredChunk]:
    """Retrieve top-k chunks with 0-1 relevance AND raw L2 distance, in a single
    embedding pass.

    ``similarity_search_with_score`` returns the raw L2 distance (lower = closer).
    Each distance is converted to a 0-1 relevance with the vectorstore's OWN
    relevance score function — the exact callable that
    ``similarity_search_with_relevance_scores`` (and hence the
    ``similarity_score_threshold`` retriever) uses internally — so the relevance
    reported here is guaranteed to equal what a threshold retriever would compute,
    without a second search or a fragile positional ``zip`` of two result lists.
    No threshold is applied; callers filter on ``relevance`` so the debug panel
    can also show what got rejected. (Out-of-corpus queries legitimately yield
    relevance < 0 — that negative score is the honest-refusal signal.)
    """
    relevance_fn = vs._select_relevance_score_fn()
    return [
        ScoredChunk(doc=doc, relevance=float(relevance_fn(distance)), distance=float(distance))
        for doc, distance in vs.similarity_search_with_score(query, k=k)
    ]


def select_context(
    scored: List[ScoredChunk],
    threshold: float,
    k: int,
    max_per_doc: int = 2,
) -> List[Document]:
    """Pick up to ``k`` above-threshold chunks, capping how many come from any
    one source document.

    ``scored`` must be ordered by descending relevance (as ``retrieve_scored``
    returns it). Without the per-document cap a single large file monopolises the
    top-k with near-duplicate chunks and crowds out a smaller, more authoritative
    source. Concretely for the BGK corpus: a "gwarancja de minimis" question fills
    its top-4 entirely from the 36-page FENG/Biznesmax PDF (which quotes 80%) and
    never reaches the dedicated de minimis document (which states 60%). Capping
    per document forces that authoritative source into the context.
    """
    per_doc: dict = defaultdict(int)
    picked: List[Document] = []
    for sc in scored:
        if sc.relevance < threshold:
            continue
        src = sc.doc.metadata.get("source", "?")
        if per_doc[src] >= max_per_doc:
            continue
        per_doc[src] += 1
        picked.append(sc.doc)
        if len(picked) >= k:
            break
    return picked


def build_embeddings() -> HuggingFaceEmbeddings:
    """Return the HF embeddings object. Call sites are expected to cache this.

    normalize_embeddings=True is REQUIRED, not optional: FAISS defaults to
    EUCLIDEAN_DISTANCE and its relevance score formula (1.0 - L2/sqrt(2)) is
    only meaningful for unit vectors. Without normalization, paraphrase-multilingual
    emits arbitrary-magnitude vectors whose L2 distances far exceed sqrt(2), so
    every in-corpus query scores negative and falls below the similarity threshold —
    the retriever returns [] for everything and the demo answers "I don't know" to
    every question. This config must match _diag_step0.py, which proved the model
    choice with normalization on.
    """
    return HuggingFaceEmbeddings(
        model_name=EMB_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


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
