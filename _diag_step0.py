"""Step 0 diagnostic v2: embeddings on the real BGK corpus, PROPER Polish diacritics.

Compares 3 models + reports relevance, whether the expected file is top-1/top-3,
and IN-vs-OUT separation. e5 uses query:/passage: prefixes via a wrapper.

Run: .venv/Scripts/python.exe _diag_step0.py
"""
import warnings
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from rag_index import load_paths, split_docs

warnings.filterwarnings("ignore")

ASSETS = Path("assets")
SUFF = {".pdf"}

# Proper Polish diacritics this time.
QUERIES = [
    ("IN", "Do jakiej części kredytu sięga gwarancja de minimis?", "de_minimis"),
    ("IN", "Jaki jest okres gwarancji dla kredytu inwestycyjnego de minimis?", "de_minimis"),
    ("IN", "Jaka jest minimalna kwota Pożyczki na cyfryzację?", "cyfryzacje"),
    ("IN", "Czym różni się gwarancja Biznesmax od Ekomax?", "FENG"),
    ("OUT", "Czy gwarancja de minimis obejmuje kredyt hipoteczny dla osoby fizycznej?", None),
    ("OUT", "Jaka jest stolica Mongolii?", None),
]


class E5Embeddings(HuggingFaceEmbeddings):
    """intfloat e5 needs 'query: ' / 'passage: ' prefixes (asymmetric)."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return super().embed_documents([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(f"query: {text}")


def mk_minilm_en():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )


def mk_paraphrase():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        encode_kwargs={"normalize_embeddings": True},
    )


def mk_e5():
    return E5Embeddings(
        model_name="intfloat/multilingual-e5-base",
        encode_kwargs={"normalize_embeddings": True},
    )


MODELS = [
    ("all-MiniLM-L6-v2 (EN, current)", mk_minilm_en),
    ("paraphrase-multilingual-MiniLM-L12-v2", mk_paraphrase),
    ("intfloat/multilingual-e5-base (query:/passage:)", mk_e5),
]


def fname(doc):
    return Path(doc.metadata.get("source", "?")).name


print("[load] PDFs from assets/ ...")
paths = [p for p in ASSETS.glob("*") if p.suffix.lower() in SUFF]
docs, _ = load_paths(paths)
chunks = split_docs(docs)
print(f"[load] {len(docs)} docs -> {len(chunks)} chunks\n")

for label, factory in MODELS:
    print("=" * 80)
    print(f"MODEL: {label}")
    print("=" * 80)
    vs = FAISS.from_documents(chunks, factory())
    in_top, out_top = [], []
    for kind, q, expect in QUERIES:
        res = vs.similarity_search_with_relevance_scores(q, k=5)
        files = [fname(d) for d, _ in res]
        top_rel = res[0][1] if res else float("nan")
        if expect is None:
            hit = "?"
            out_top.append(top_rel)
        else:
            t1 = expect.lower() in files[0].lower()
            t3 = any(expect.lower() in f.lower() for f in files[:3])
            hit = "TOP1" if t1 else ("top3" if t3 else "MISS")
            in_top.append(top_rel)
        print(f"\n[{kind}] {hit:4} top_rel={top_rel:6.3f} | {q}")
        for d, s in res:
            print(f"     {s:6.3f}  {fname(d)}  (p{d.metadata.get('page','?')})")
    sep = (min(in_top) if in_top else 0) - (max(out_top) if out_top else 0)
    print(f"\n  >>> IN min={min(in_top):.3f}  OUT max={max(out_top):.3f}  separation={sep:+.3f}")
    print()
print("[done]")
