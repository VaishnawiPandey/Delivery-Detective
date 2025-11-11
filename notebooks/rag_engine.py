"""
rag_engine.py

Simple, practical RAG engine:
- Ingests CSV/text files and builds a FAISS index with sentence-transformers embeddings.
- Offers a `query()` function that retrieves top-k docs and generates an answer using
  either OpenAI (if OPENAI_API_KEY set) or a Hugging Face text2text-generation model.

Usage:
    # Build index (run once, or when your files change)
    python rag_engine.py --build-index \
        --docs "final.csv,dri_by_location.csv,dri_generated_insights.txt,dri_feature_importance.csv"

    # Query (interactive)
    python rag_engine.py --query "Which location has lowest DRI and why?"

Dependencies: see requirements.txt
"""

import os
import argparse
import pickle
from typing import List, Dict, Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Generator libs
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
# Optional OpenAI
import openai

# ---------- CONFIG ----------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = "rag_index"
INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")
DOCS_FILE = os.path.join(INDEX_DIR, "docs.pkl")
EMB_FILE = os.path.join(INDEX_DIR, "embeddings.npy")
META_FILE = os.path.join(INDEX_DIR, "meta.pkl")

# HF generator model (change if you have GPU and want a larger model)
HF_GENERATOR_MODEL = "google/flan-t5-base"  # modest; if you have GPU you can use flan-t5-large

# retrieval + generation settings
TOP_K = 4
MAX_CHARS_PER_DOC = 1000  # truncate docs for context


# ---------- Helpers: ingest & convert files to text ----------
def read_csv_as_text_rows(path: str, text_columns: List[str] = None) -> List[Dict]:
    """
    Convert CSV rows to textual documents (one doc per row).
    If text_columns provided, concatenate only those columns; else concat all columns.
    Returns list of dicts: {"id": "<filename_rowidx>", "text": "...", "meta": {...}}
    """
    df = pd.read_csv(path)
    docs = []
    text_cols = text_columns if text_columns else df.columns.tolist()
    for i, row in df.iterrows():
        parts = []
        for c in text_cols:
            try:
                v = row[c]
            except Exception:
                v = ""
            if pd.isna(v):
                continue
            parts.append(f"{c}: {v}")
        text = " | ".join(parts)
        text = text.strip()
        docs.append({
            "id": f"{os.path.basename(path)}_row{i}",
            "text": text[:MAX_CHARS_PER_DOC],
            "meta": {"source": os.path.basename(path), "row": int(i)}
        })
    return docs


def read_txt_file(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    # split into paragraphs if long
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    docs = []
    for i, p in enumerate(paragraphs):
        docs.append({
            "id": f"{os.path.basename(path)}_p{i}",
            "text": p[:MAX_CHARS_PER_DOC],
            "meta": {"source": os.path.basename(path), "part": i}
        })
    if not docs:
        docs = [{"id": os.path.basename(path), "text": content[:MAX_CHARS_PER_DOC], "meta": {"source": os.path.basename(path)}}]
    return docs


def ingest_paths(paths: List[str]) -> List[Dict]:
    """
    Given a list of file paths, return list of document dicts.
    Accepts CSV, TXT. You can extend to PDF/MD later.
    """
    all_docs = []
    for p in paths:
        p = p.strip()
        if not os.path.exists(p):
            print(f"Warning: {p} not found — skipping")
            continue
        ext = os.path.splitext(p)[1].lower()
        if ext in [".csv", ".tsv"]:
            # prefer some text columns for CSVs (heuristic)
            if os.path.basename(p).lower().startswith("final") or "dri" in os.path.basename(p).lower():
                # include key columns
                text_cols = None  # use all, or you can supply explicit list
            else:
                text_cols = None
            docs = read_csv_as_text_rows(p, text_columns=text_cols)
        elif ext in [".txt", ".md"]:
            docs = read_txt_file(p)
        else:
            print(f"Unsupported extension {ext} for {p}, skipping.")
            docs = []
        all_docs.extend(docs)
    return all_docs


# ---------- Build FAISS index ----------
def build_faiss_index(docs: List[Dict], embed_model_name: str = EMBED_MODEL_NAME):
    os.makedirs(INDEX_DIR, exist_ok=True)
    print(f"Loading embedding model: {embed_model_name}")
    embedder = SentenceTransformer(embed_model_name)

    texts = [d["text"] for d in docs]
    print(f"Encoding {len(texts)} documents...")
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    print(f"Embeddings shape: {embeddings.shape}")

    # build index (cosine via inner product on normalized vectors -> use IndexFlatIP)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built, {index.ntotal} vectors")

    # persist
    faiss.write_index(index, INDEX_FILE)
    np.save(EMB_FILE, embeddings)
    with open(DOCS_FILE, "wb") as f:
        pickle.dump(docs, f)
    # also save metadata list for convenience
    with open(META_FILE, "wb") as f:
        pickle.dump({"embed_model": embed_model_name, "dim": dim}, f)

    print(f"Saved index -> {INDEX_FILE}, docs -> {DOCS_FILE}")
    return index, embeddings


# ---------- Load index ----------
def load_faiss_index():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(DOCS_FILE):
        raise FileNotFoundError("Index or docs not found. Run build-index first.")
    index = faiss.read_index(INDEX_FILE)
    with open(DOCS_FILE, "rb") as f:
        docs = pickle.load(f)
    embeddings = None
    if os.path.exists(EMB_FILE):
        embeddings = np.load(EMB_FILE)
    return index, docs, embeddings


# ---------- Retrieval ----------
def retrieve(query: str, top_k: int = TOP_K, embed_model_name: str = EMBED_MODEL_NAME) -> List[Tuple[Dict, float]]:
    index, docs, _ = load_faiss_index()
    embedder = SentenceTransformer(embed_model_name)
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(docs):
            continue
        results.append((docs[idx], float(score)))
    return results


# ---------- Generation ----------
def generate_answer_with_openai(prompt: str, max_tokens: int = 256) -> str:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    openai.api_key = api_key
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini" if True else "gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.1
    )
    return resp["choices"][0]["message"]["content"]


# HF fallback generator (text2text)
# _hf_generator = None
# def get_hf_generator():
#     global _hf_generator
#     if _hf_generator is None:
#         print(f"Loading HF generator model: {HF_GENERATOR_MODEL} (this may take time)...")
#         _hf_generator = pipeline("text2text-generation", model=HF_GENERATOR_MODEL, tokenizer=HF_GENERATOR_MODEL, device=0 if (os.environ.get("CUDA_VISIBLE_DEVICES")) else -1)
#     return _hf_generator
from transformers import pipeline

def get_hf_generator():
    global _hf_generator
    if '_hf_generator' not in globals():
        print("Loading HF generator model: google/flan-t5-base (PyTorch backend)...")
        _hf_generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            tokenizer="google/flan-t5-base",
            framework="pt",     # ✅ forces PyTorch
            device=-1           # use CPU; set 0 if you have GPU
        )
    return _hf_generator


def generate_answer_with_hf(prompt: str, max_length:int = 256) -> str:
    gen = get_hf_generator()
    out = gen(prompt, max_length=max_length, do_sample=False)
    if isinstance(out, list):
        return out[0]["generated_text"]
    return str(out)


# ---------- High-level query API ----------
def build_prompt_with_context(question: str, retrieved: List[Tuple[Dict, float]]) -> str:
    # Build a concise prompt for generator. include source metadata and excerpts.
    ctx_parts = []
    for doc, score in retrieved:
        src = doc.get("meta", {}).get("source", "unknown")
        excerpt = doc["text"][:800]
        ctx_parts.append(f"[Source: {src} | score: {score:.3f}] {excerpt}")
    context = "\n\n".join(ctx_parts)
    prompt = (
        "You are an assistant with access to the following documents (excerpts included). "
        "Use the context to answer the user's question. If the answer is not present, say 'I couldn't find a definitive answer in the provided documents.'\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nAnswer concisely, cite source names in square brackets when possible."
    )
    return prompt


def answer_query(question: str, use_openai_if_available: bool = True) -> Dict:
    retrieved = retrieve(question, top_k=TOP_K)
    if not retrieved:
        return {"answer": "No documents available in index.", "sources": [], "retrieved": []}

    prompt = build_prompt_with_context(question, retrieved)
    # pick generator
    api_key = os.environ.get("OPENAI_API_KEY")
    if use_openai_if_available and api_key:
        try:
            ans = generate_answer_with_openai(prompt)
            source_names = list({d[0]["meta"].get("source", "") for d in retrieved})
            return {"answer": ans, "sources": source_names, "retrieved": retrieved}
        except Exception as e:
            print("OpenAI generation failed, falling back to HF:", e)

    # HF fallback
    ans = generate_answer_with_hf(prompt)
    source_names = list({d[0]["meta"].get("source", "") for d in retrieved})
    return {"answer": ans, "sources": source_names, "retrieved": retrieved}


# ---------- CLI interface ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-index", action="store_true", help="Build FAISS index from provided docs")
    parser.add_argument("--docs", type=str, help="Comma-separated list of documents to ingest (csv/txt)")
    parser.add_argument("--query", type=str, help="Query the index")
    args = parser.parse_args()

    if args.build_index:
        if not args.docs:
            print("Provide --docs 'file1,file2,...'")
            return
        paths = [p.strip() for p in args.docs.split(",")]
        docs = ingest_paths(paths)
        print(f"Ingested {len(docs)} docs.")
        build_faiss_index(docs)
        print("Index built.")
        return

    if args.query:
        res = answer_query(args.query)
        print("\n=== ANSWER ===\n")
        print(res["answer"])
        print("\n=== SOURCES ===")
        print(", ".join(res["sources"]))
        print("\n=== RETRIEVED DOCS ===")
        for d,score in res["retrieved"]:
            print(f"- {d['meta'].get('source')} (score {score:.3f}) : {d['text'][:200]}...")
        return

    parser.print_help()

# --- Add this helper for Streamlit integration ---
def query_index(user_query: str):
    """
    Simple wrapper for Streamlit app.
    Takes a user query and returns generated answer as a string.
    """
    try:
        result = answer_query(user_query)
        return result
    except Exception as e:
        return f"❌ Error: {str(e)}"



if __name__ == "__main__":
    main()
