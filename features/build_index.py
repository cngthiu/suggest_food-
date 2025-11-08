#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds BM25 (token-based) and FAISS (embedding) indices for recipe retrieval.
- BM25 over [title, summary, tags, ingredients.name]
- Embedding via sentence-transformers; FAISS IndexFlatIP on L2-normalized vectors
Outputs saved to serverAI/.cache (configurable in YAML).
"""

import os, json, glob, unicodedata, pickle, argparse, sys, random
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

try:
    import yaml
except Exception as e:
    print("Please `pip install pyyaml`", file=sys.stderr); raise

# ---- Tokenizer (Underthesea optional) ----

def _try_import_underthesea():
    try:
        from underthesea import word_tokenize
        return word_tokenize
    except Exception:
        return None

UTS_WORD_TOKENIZE = _try_import_underthesea()

def strip_diacritics(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")

def norm_text(s: str, lowercase=True, strip_dia=True) -> str:
    if s is None: return ""
    s = s.strip()
    if lowercase: s = s.lower()
    if strip_dia: s = strip_diacritics(s)
    return s

def tokenize_vi(s: str, mode: str = "auto") -> List[str]:
    # mode: "auto" | "underthesea" | "none"
    if not s: return []
    if mode == "none":
        import re
        return re.findall(r"[a-z0-9]+", s.lower())
    if mode == "underthesea" or (mode == "auto" and UTS_WORD_TOKENIZE):
        text = s.replace("/", " / ").replace("-", " - ")
        toks = UTS_WORD_TOKENIZE(text, format="text").split()
        return [t for t in toks if t.strip()]
    # fallback regex
    import re
    return re.findall(r"[a-z0-9]+", s.lower())

# ---- BM25 ----

def build_bm25(tokenized_corpus: List[List[str]]):
    try:
        from rank_bm25 import BM25Okapi
    except Exception:
        print("Please `pip install rank_bm25`", file=sys.stderr); raise
    return BM25Okapi(tokenized_corpus)

# ---- Embeddings ----

def build_embeddings(texts: List[str], model_name: str, batch_size: int = 64, normalize: bool = True):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        print("Please `pip install sentence-transformers`", file=sys.stderr); raise
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=normalize)
    return emb.astype(np.float32)

# ---- FAISS ----

def build_faiss_index(emb: np.ndarray, metric: str = "ip"):
    try:
        import faiss
    except Exception:
        print("Please `pip install faiss-cpu` (or faiss-gpu)", file=sys.stderr); raise
    d = emb.shape[1]
    if metric == "ip":
        index = faiss.IndexFlatIP(d)
    elif metric == "l2":
        index = faiss.IndexFlatL2(d)
    else:
        raise ValueError("faiss_metric must be 'ip' or 'l2'")
    index.add(emb)
    return index

# ---- Load recipes ----

def load_recipes(recipes_dir: str) -> List[Dict[str, Any]]:
    files = sorted(glob.glob(os.path.join(recipes_dir, "*.json")))
    items = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            try:
                obj = json.load(fh)
                items.append(obj)
            except Exception as e:
                print(f"[WARN] Skip {f}: {e}", file=sys.stderr)
    if not items:
        raise RuntimeError(f"No recipe JSON found in {recipes_dir}")
    return items

def recipe_to_index_text(r: Dict[str, Any], fields: List[str], lowercase: bool, strip_dia: bool) -> str:
    parts = []
    for field in fields:
        if field == "ingredients":
            names = [ing.get("name","") for ing in r.get("ingredients",[])]
            parts.append(" ".join(names))
        else:
            v = r.get(field, "")
            if isinstance(v, list): v = " ".join(map(str, v))
            parts.append(str(v))
    text = " . ".join(parts)
    return norm_text(text, lowercase=lowercase, strip_dia=strip_dia)

def main():
    import numpy as np
    import random
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="serverAI/config/app.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    random.seed(cfg.get("seed", 2025))
    np.random.seed(cfg.get("seed", 2025))

    paths = cfg["paths"]
    idx_cfg = cfg["indexing"]
    ret_cfg = cfg["retrieval"]

    recipes = load_recipes(paths["recipes_dir"])
    recipe_ids = [r.get("id") for r in recipes]

    # 1) Build corpus texts
    texts = [recipe_to_index_text(r, idx_cfg["fields"], idx_cfg["lowercase"], idx_cfg["strip_diacritics"]) for r in recipes]

    # 2) Tokenize corpus for BM25
    tok_mode = idx_cfg.get("use_vietnamese_tokenizer", "auto")
    corpus_tokens = [tokenize_vi(t, tok_mode) for t in texts]

    # 3) BM25
    print("Building BM25 ...")
    bm25 = build_bm25(corpus_tokens)

    cache_dir = Path(paths["cache_dir"]); cache_dir.mkdir(parents=True, exist_ok=True)
    with open(paths["bm25_path"], "wb") as fh:
        pickle.dump(bm25, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(paths["corpus_path"], "wb") as fh:
        pickle.dump(corpus_tokens, fh, protocol=pickle.HIGHEST_PROTOCOL)

    with open(paths["ids_path"], "w", encoding="utf-8") as fh:
        json.dump({"recipe_ids": recipe_ids}, fh, ensure_ascii=False, indent=2)

    # 4) Embeddings
    print(f"Embedding with {ret_cfg['embedder_model']} ...")
    emb = build_embeddings(texts,
                           model_name=ret_cfg["embedder_model"],
                           batch_size=ret_cfg.get("embed_batch_size", 64),
                           normalize=ret_cfg.get("normalize_embeddings", True))
    np.save(paths["embed_matrix"], emb)

    # 5) FAISS
    print("Building FAISS index ...")
    index = build_faiss_index(emb, metric=ret_cfg.get("faiss_metric", "ip"))

    # Save FAISS
    try:
        import faiss
        faiss.write_index(index, paths["faiss_index"])
    except Exception as e:
        print(f"[ERROR] Saving FAISS index failed: {e}", file=sys.stderr); raise

    print("Done. Artifacts saved to:", cache_dir)

if __name__ == "__main__":
    main()
