#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, glob, pickle, argparse, sys, random
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import yaml
from serverAI.inference.utils import norm_text, tokenize_vi

# ---- BM25 ----
def build_bm25(tokenized_corpus: List[List[str]]):
    try:
        from rank_bm25 import BM25Okapi
        return BM25Okapi(tokenized_corpus)
    except ImportError:
        print("Please `pip install rank_bm25`", file=sys.stderr); raise

# ---- Embeddings ----
def build_embeddings(texts: List[str], model_name: str, batch_size: int = 64, normalize: bool = True):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=normalize)
        return emb.astype(np.float32)
    except ImportError:
        print("Please `pip install sentence-transformers`", file=sys.stderr); raise

# ---- FAISS ----
def build_faiss_index(emb: np.ndarray, metric: str = "ip"):
    try:
        import faiss
        d = emb.shape[1]
        if metric == "ip":
            index = faiss.IndexFlatIP(d)
        else:
            index = faiss.IndexFlatL2(d)
        index.add(emb)
        return index
    except ImportError:
        print("Please `pip install faiss-cpu`", file=sys.stderr); raise

def load_recipes(recipes_path: str) -> List[Dict[str, Any]]:
    p = Path(recipes_path)

    # Nếu đưa vào là folder -> tự tìm file json bên trong
    if p.is_dir():
        candidates = [
            p / "recipies.json",
            p / "recipes.json",
            p / "recipe.json",
        ]
        found = next((c for c in candidates if c.exists() and c.is_file()), None)
        if found is None:
            # fallback: tìm file .json đầu tiên trong folder
            js = sorted(p.glob("*.json"))
            if not js:
                raise FileNotFoundError(f"No .json file found in directory: {p}")
            found = js[0]
        p = found

    if not p.exists():
        raise FileNotFoundError(f"Recipes path not found: {p}")

    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    # Hỗ trợ nhiều format
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if isinstance(data.get("recipes"), list):
            return data["recipes"]
        if isinstance(data.get("items"), list):
            return data["items"]
        if all(isinstance(v, dict) for v in data.values()):
            return list(data.values())

    raise ValueError(f"Unsupported recipies.json format: root type={type(data)} at {p}")


def recipe_to_index_text(r: Dict[str, Any], fields: List[str]) -> str:
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
    # Không cần strip_dia ở đây vì tokenizer sẽ xử lý, hoặc tùy config
    return text

def main():
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

    print("[INFO] Loading recipes...")
    recipes = load_recipes(paths["recipes_dir"])
    recipe_ids = [r.get("id") for r in recipes]

    # 1) Tokenize corpus for BM25
    # Logic mới: Luôn dùng tokenize_vi từ utils, bỏ qua config thừa
    print("[INFO] Tokenizing corpus (Underthesea)...")
    texts_for_bm25 = [recipe_to_index_text(r, idx_cfg["fields"]) for r in recipes]
    
    # Tokenizer chuẩn hóa bên trong hàm tokenize_vi
    corpus_tokens = [tokenize_vi(t) for t in texts_for_bm25]

    # 2) Build BM25
    print("[INFO] Building BM25 index...")
    bm25 = build_bm25(corpus_tokens)

    # 3) Embeddings (Giữ nguyên logic text processing cho embedding)
    # Embedding models thường thích raw text hơn là tokenized text gạch dưới
    print(f"[INFO] Embedding with {ret_cfg['embedder_model']} ...")
    texts_for_emb = [norm_text(t, lowercase=True, strip_dia=True) for t in texts_for_bm25]
    
    emb = build_embeddings(texts_for_emb,
                           model_name=ret_cfg["embedder_model"],
                           batch_size=ret_cfg.get("embed_batch_size", 64),
                           normalize=ret_cfg.get("normalize_embeddings", True))

    # 4) Save artifacts
    cache_dir = Path(paths["cache_dir"]); cache_dir.mkdir(parents=True, exist_ok=True)
    
    with open(paths["bm25_path"], "wb") as fh:
        pickle.dump(bm25, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(paths["corpus_path"], "wb") as fh:
        pickle.dump(corpus_tokens, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(paths["ids_path"], "w", encoding="utf-8") as fh:
        json.dump({"recipe_ids": recipe_ids}, fh, ensure_ascii=False, indent=2)
        
    np.save(paths["embed_matrix"], emb)
    
    # 5) FAISS
    print("[INFO] Building FAISS index...")
    index = build_faiss_index(emb, metric=ret_cfg.get("faiss_metric", "ip"))
    import faiss
    faiss.write_index(index, paths["faiss_index"])

    print(f"[SUCCESS] Done. Artifacts saved to {cache_dir}")

if __name__ == "__main__":
    main()