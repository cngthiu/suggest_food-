# File: inference/retrieval.py
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .utils import (
    load_json,
    load_recipes_any,
    norm_text,
    normalize_ingredient_mappings,
    recipes_to_dict,
)

try:
    from underthesea import word_tokenize
except Exception:  # pragma: no cover
    word_tokenize = None


def vi_tokenize(s: str) -> List[str]:
    if not s:
        return []
    if word_tokenize is None:
        return norm_text(s, lowercase=True, strip_dia=True).split()
    return word_tokenize(norm_text(s), format="text").split()


def resolve_encoder_device() -> str:
    pref = os.environ.get("SENTENCE_EMB_DEVICE", "auto").lower()
    if pref != "auto":
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class Retriever:
    def __init__(
        self,
        cfg: Dict[str, Any],
        ingredient_map: Optional[Dict[str, Any]] = None,
        product_catalog: Optional[Dict[str, Any]] = None,
        recipes: Optional[Dict[str, Any]] = None,
        recipe_fetcher=None,
    ):
        self.cfg = cfg
        paths = cfg["paths"]

        for key in ["bm25_path", "corpus_path", "ids_path"]:
            if not Path(paths[key]).exists():
                raise FileNotFoundError(
                    f"Missing artifact `{key}` at `{paths[key]}`. "
                    "Bạn cần chạy `python -m serverAI.features.build_index --config serverAI/config/app.yaml` để build cache."
                )

        with open(paths["bm25_path"], "rb") as f:
            self.bm25 = pickle.load(f)
        with open(paths["corpus_path"], "rb") as f:
            self.corpus_tokens = pickle.load(f)
        with open(paths["ids_path"], "r", encoding="utf-8") as f:
            self.recipe_ids = json.load(f)["recipe_ids"]

        self.semantic_enabled = False
        self.faiss = None
        self.emb = None
        self.embedder = None

        if recipes is not None:
            self.recipes = recipes
        elif recipe_fetcher is not None:
            self.recipes = recipe_fetcher([str(rid) for rid in self.recipe_ids])
        else:
            self.recipes = self._load_recipes(paths["recipes_dir"])

        raw_mapping = ingredient_map
        if not raw_mapping:
            raw_mapping = load_json(os.path.join(paths["mapping_dir"], "ingredient_mappings.json"))
        self.mapping = normalize_ingredient_mappings(raw_mapping)

        self.catalog = product_catalog or {}
        self.idx_cfg = cfg["indexing"]
        self.ret_cfg = cfg["retrieval"]
        self.hy_cfg = cfg["hybrid"]

        self.encoder_device = resolve_encoder_device()

        embedder_model = (self.ret_cfg.get("embedder_model") or "").strip()
        if embedder_model:
            for key in ["embed_matrix", "faiss_index"]:
                if not Path(paths[key]).exists():
                    raise FileNotFoundError(
                        f"Missing artifact `{key}` at `{paths[key]}`. "
                        "Bạn cần chạy `python -m serverAI.features.build_index --config serverAI/config/app.yaml` để build cache."
                    )
            try:
                import faiss
                from sentence_transformers import SentenceTransformer

                self.faiss = faiss.read_index(paths["faiss_index"])
                self.emb = np.load(paths["embed_matrix"], mmap_mode="r")
                self.embedder = SentenceTransformer(embedder_model, device=self.encoder_device)
                self.semantic_enabled = True
            except Exception as e:
                print(f"[WARN] Semantic retrieval disabled: {e}")

    def _tokenize(self, s: str) -> List[str]:
        return vi_tokenize(s)

    def _load_recipes(self, recipes_dir: str) -> Dict[str, Any]:
        recipes = load_recipes_any(recipes_dir)
        return recipes_to_dict(recipes)

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        s = norm_text(query, lowercase=True, strip_dia=True)
        toks = self._tokenize(s)
        bm25_scores = self.bm25.get_scores(toks)

        k_bm25 = self.hy_cfg["k_bm25"]
        top_bm25_idx = np.argsort(bm25_scores)[::-1][:k_bm25]

        q_emb = None
        I = np.array([], dtype=np.int64)
        if self.semantic_enabled and self.embedder is not None and self.faiss is not None:
            q_emb = self.embedder.encode([s], normalize_embeddings=self.ret_cfg.get("normalize_embeddings", True))
            _, I = self.faiss.search(np.array(q_emb, dtype=np.float32), self.hy_cfg["k_emb"])
            I = I[0]

        candidate_indices: List[int] = []
        seen = set()
        cap = int(self.hy_cfg.get("k_total", 0))

        def add_indices(indices):
            for idx in indices:
                if idx < 0 or idx in seen:
                    continue
                candidate_indices.append(int(idx))
                seen.add(int(idx))
                if cap and len(candidate_indices) >= cap:
                    return True
            return False

        if not add_indices(top_bm25_idx.tolist()):
            add_indices(I.tolist())

        cand = []
        for idx in candidate_indices:
            rid = self.recipe_ids[idx]
            recipe = self.recipes.get(str(rid)) if isinstance(self.recipes, dict) else None
            if not recipe:
                continue
            semantic = 0.0
            if self.semantic_enabled and q_emb is not None and self.emb is not None:
                semantic = float(np.dot(q_emb[0], self.emb[idx]))
            cand.append(
                {
                    "id": rid,
                    "bm25": float(bm25_scores[idx]),
                    "semantic": semantic,
                    "recipe": recipe,
                }
            )
        if not cand:
            return []

        bm_vals = np.array([c["bm25"] for c in cand])
        sem_vals = np.array([c["semantic"] for c in cand])

        def mm(x):
            if x.size == 0:
                return x
            return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-9)

        bm_n = mm(bm_vals)
        sem_n = mm(sem_vals)

        out = []
        for i, c in enumerate(cand):
            out.append(
                {
                    "id": c["id"],
                    "bm25_n": float(bm_n[i]),
                    "semantic_n": float(sem_n[i]),
                    "time_fit": 1.0,
                    "diet_fit": 1.0,
                    "availability_ratio": self._availability_ratio(c["recipe"]),
                    "promo_coverage": 0.0,
                    "history_affinity": 0.0,
                    "recipe": c["recipe"],
                }
            )
        return out

    def _availability_ratio(self, recipe: Dict[str, Any]) -> float:
        need = recipe.get("ingredients", [])
        if not need:
            return 0.0
        total_weight, score = 0.0, 0.0
        for ing in need:
            weight = 3.0 if ing.get("type") == "main" else 1.0
            total_weight += weight
            mappings = self.mapping.get(ing.get("name", "").strip(), [])
            is_in_stock = False
            if self.catalog:
                for m in mappings:
                    if self.catalog.get(m.get("sku"), {}).get("stock", 0) > 0:
                        is_in_stock = True
                        break
            elif mappings:
                is_in_stock = True
            if is_in_stock:
                score += weight
        return float(score / total_weight) if total_weight > 0 else 0.0
