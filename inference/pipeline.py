# -*- coding: utf-8 -*-
"""End-to-end inference pipeline: NLU → Retrieval → Ranking (rule) → SKU mapping → Cart suggestion."""

import os, json, pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from .utils import norm_text, load_json

try:
    import yaml
except Exception as e:
    raise

# Optional Underthesea
try:
    from underthesea import word_tokenize
    def vi_tokenize(s: str):
        return word_tokenize(s, format='text').split()
except Exception:
    import re
    def vi_tokenize(s: str):
        return re.findall(r"[a-z0-9]+", s.lower())


def resolve_encoder_device() -> str:
    """Decide whether SentenceTransformer should run on CPU or GPU."""
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


def recipe_matches_protein(recipe: Dict[str, Any], protein: str) -> bool:
    """Heuristic check whether recipe focuses on requested protein."""
    if not protein:
        return True
    target = norm_text(protein, lowercase=True, strip_dia=True)
    if not target:
        return True
    # check title & tags
    title = norm_text(recipe.get("title", ""), lowercase=True, strip_dia=True)
    if target in title:
        return True
    for tag in recipe.get("tags", []):
        if target in norm_text(str(tag), lowercase=True, strip_dia=True):
            return True
    # check ingredients + aliases
    for ing in recipe.get("ingredients", []):
        names = [ing.get("name", "")]
        names.extend(ing.get("aliases", []))
        for name in names:
            if target in norm_text(name, lowercase=True, strip_dia=True):
                return True
    return False


class NLU:
    def __init__(self, model_dir: str, gazetteer_dir: str):
        import joblib
        self.model_dir = model_dir
        self.gaz = self._load_gazetteer(gazetteer_dir)
        self.vectorizer = joblib.load(Path(model_dir)/"vectorizer.pkl")
        self.clf = joblib.load(Path(model_dir)/"classifier.pkl")
        with open(Path(model_dir)/"label_map.json", "r", encoding="utf-8") as f:
            self.labels = json.load(f)["labels"]

    def _load_gazetteer(self, root: str) -> Dict[str, List[str]]:
        gaz = {}
        if not root or not os.path.isdir(root):
            return gaz
        for name in ["protein", "diet", "device", "allergy", "region"]:
            p = Path(root)/f"{name}.txt"
            if p.exists():
                with open(p, 'r', encoding='utf-8') as f:
                    gaz[name] = [line.strip() for line in f if line.strip()]
        return gaz

    def predict_intent(self, text: str) -> Dict[str, Any]:
        import unicodedata
        s = norm_text(text, lowercase=True, strip_dia=True)
        X = self.vectorizer.transform([s])
        prob = self.clf.predict_proba(X)[0]
        idx = int(np.argmax(prob))
        return {"name": self.labels[idx], "score": float(prob[idx])}

    def extract_slots(self, text: str) -> Dict[str, Any]:
        s = norm_text(text, lowercase=True, strip_dia=True)
        slots: Dict[str, Any] = {"servings": None, "time": None, "diet": None, "protein": None, "device": None, "allergy": []}
        # servings
        import re
        m = re.search(r"(\d+)\s*(nguoi|suat)", s)
        if m:
            slots["servings"] = int(m.group(1))
        # time (<=xx phut)
        m = re.search(r"(<=|<|duoi)\s*(\d{1,3})\s*(phut|p)?", s)
        if m:
            slots["time"] = f"<= {m.group(2)}"
        # protein from gazetteer
        for p in self.gaz.get("protein", []):
            if p in s:
                slots["protein"] = p
                break
        # diet
        for d in self.gaz.get("diet", []):
            if d in s:
                slots["diet"] = d
                break
        # device
        for d in self.gaz.get("device", []):
            if d in s:
                slots["device"] = d
                break
        # allergy (multi)
        for a in self.gaz.get("allergy", []):
            if a in s:
                slots["allergy"].append(a)
        return slots


class Retriever:
    def __init__(self, cfg: Dict[str, Any], ingredient_map: Optional[Dict[str, Any]] = None):
        self.cfg = cfg
        paths = cfg["paths"]
        # BM25
        with open(paths["bm25_path"], 'rb') as f:
            self.bm25 = pickle.load(f)
        with open(paths["corpus_path"], 'rb') as f:
            self.corpus_tokens = pickle.load(f)
        # FAISS
        import faiss
        self.faiss = faiss.read_index(paths["faiss_index"])
        self.emb = np.load(paths["embed_matrix"], mmap_mode="r")
        with open(paths["ids_path"], 'r', encoding='utf-8') as f:
            self.recipe_ids = json.load(f)["recipe_ids"]
        # Load recipes for metadata
        self.recipes = self._load_recipes(paths["recipes_dir"])
        self.mapping = ingredient_map or load_json(os.path.join(paths["mapping_dir"], "ingredient_to_sku.json"))

        self.idx_cfg = cfg["indexing"]
        self.ret_cfg = cfg["retrieval"]
        self.hy_cfg = cfg["hybrid"]

        # Prepare embedder for query
        from sentence_transformers import SentenceTransformer
        device_hint = resolve_encoder_device()
        self.embedder = SentenceTransformer(self.ret_cfg["embedder_model"], device=device_hint)
        self.encoder_device = device_hint

    def _load_recipes(self, recipes_dir: str) -> Dict[str, Any]:
        import glob
        result = {}
        for fp in glob.glob(os.path.join(recipes_dir, "*.json")):
            with open(fp, 'r', encoding='utf-8') as f:
                obj = json.load(f)
                result[obj["id"]] = obj
        return result

    def _recipe_index_text(self, r: Dict[str, Any]) -> str:
        parts = []
        for field in self.idx_cfg["fields"]:
            if field == "ingredients":
                names = [ing.get("name","") for ing in r.get("ingredients",[])]
                parts.append(" ".join(names))
            else:
                v = r.get(field, "")
                if isinstance(v, list): v = " ".join(map(str, v))
                parts.append(str(v))
        text = " . ".join(parts)
        # same normalization as index
        from .utils import norm_text
        return norm_text(text, lowercase=self.idx_cfg["lowercase"], strip_dia=self.idx_cfg["strip_diacritics"])

    def _tokenize(self, s: str) -> List[str]:
        mode = self.idx_cfg.get("use_vietnamese_tokenizer", "auto")
        if mode == "none":
            import re
            return re.findall(r"[a-z0-9]+", s.lower())
        try:
            from underthesea import word_tokenize
            if mode == "underthesea" or mode == "auto":
                return word_tokenize(s, format='text').split()
        except Exception:
            pass
        import re
        return re.findall(r"[a-z0-9]+", s.lower())

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        s = norm_text(query, lowercase=True, strip_dia=True)
        # BM25
        toks = self._tokenize(s)
        bm25_scores = self.bm25.get_scores(toks)
        k_bm25 = self.hy_cfg["k_bm25"]
        top_bm25_idx = np.argsort(bm25_scores)[::-1][:k_bm25]

        # Embedding
        q_emb = self.embedder.encode([s], normalize_embeddings=self.ret_cfg.get("normalize_embeddings", True))
        D, I = self.faiss.search(np.array(q_emb, dtype=np.float32), self.hy_cfg["k_emb"])  # (1, k)
        I = I[0]; D = D[0]

        # Merge with ordering + cap to keep latency predictable
        candidate_indices: List[int] = []
        seen = set()
        cap = int(self.hy_cfg.get("k_total", 0))

        def add_indices(indices: List[int]) -> bool:
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
            r = self.recipes[rid]
            # collect both scores (normalized later)
            bm = float(bm25_scores[idx])
            # semantic sim via cosine (FAISS IP already on normalized vectors ~ cosine)
            # we'll recompute q • d
            d_emb = self.emb[idx]
            sem = float(np.dot(q_emb[0], d_emb))
            cand.append({"id": rid, "bm25": bm, "semantic": sem, "recipe": r, "bm25_rank": None, "emb_rank": None})

        # Normalize & score rule-based
        if not cand:
            return []
        bm_vals = np.array([c["bm25"] for c in cand]);
        sem_vals = np.array([c["semantic"] for c in cand])
        # min-max normalize
        def mm(x):
            if x.size == 0: return x
            lo, hi = float(np.min(x)), float(np.max(x))
            if hi - lo < 1e-6:
                return np.zeros_like(x)+0.5
            return (x - lo) / (hi - lo)
        bm_n = mm(bm_vals); sem_n = mm(sem_vals)

        # business features
        out = []
        for i, c in enumerate(cand):
            r = c["recipe"]
            time_fit = 1.0  # default; refined with slots later
            diet_fit = 1.0
            availability_ratio = self._availability_ratio(r)
            promo_coverage = 0.0  # placeholder
            history_affinity = 0.0
            out.append({
                "id": c["id"],
                "bm25_n": float(bm_n[i]),
                "semantic_n": float(sem_n[i]),
                "time_fit": time_fit,
                "diet_fit": diet_fit,
                "availability_ratio": float(availability_ratio),
                "promo_coverage": float(promo_coverage),
                "history_affinity": float(history_affinity),
                "recipe": r
            })
        return out

    def _availability_ratio(self, recipe: Dict[str, Any]) -> float:
        # simple check: every ingredient exists in mapping (stock not considered here)
        mapping = self.mapping
        need = recipe.get("ingredients", [])
        if not need:
            return 0.0
        ok = 0
        for ing in need:
            name = ing.get("name", "").strip()
            if name in mapping and len(mapping[name]) > 0:
                ok += 1
        return ok/len(need)


class CartMapper:
    def __init__(self, cfg: Dict[str, Any], ingredient_map: Optional[Dict[str, Any]] = None):
        self.cfg = cfg
        paths = cfg["paths"]
        self.map = ingredient_map or load_json(os.path.join(paths["mapping_dir"], "ingredient_to_sku.json"))
        subs_path = os.path.join(paths["mapping_dir"], "substitutions.json")
        self.subs = load_json(subs_path) if os.path.exists(subs_path) else {}

    def suggest_cart(self, recipe: Dict[str, Any], servings: int) -> Dict[str, Any]:
        items = []
        total = 0
        notes = []
        for ing in recipe.get("ingredients", []):
            name = ing.get("name")
            mapping = self.map.get(name, [])
            if not mapping:
                items.append({"ingredient": name, "sku": None, "note": "no mapping"})
                continue
            best = mapping[0]
            unitSize = best.get("unitSize", {"qty":1, "unit":"unit"})
            ratio = best.get("ratio_per_serving", {"qty":1, "unit": unitSize.get("unit")})
            need_qty = float(ratio.get("qty",1)) * float(servings)
            # round packages
            packages = int(np.ceil(need_qty / float(unitSize.get("qty",1))))
            price = best.get("price", 0)  # optional if catalog merged later
            subtotal = packages * price
            total += subtotal
            items.append({
                "ingredient": name,
                "sku": best.get("sku"),
                "name": best.get("name"),
                "unitSize": unitSize,
                "need": {"qty": need_qty, "unit": ratio.get("unit")},
                "packages": packages,
                "price": price,
                "subtotal": subtotal,
                "stock_ok": True,
                "alt": []
            })
        return {"items": items, "estimated": total, "currency": "VND", "notes": notes}


def apply_slot_constraints(cand: List[Dict[str, Any]], slots: Dict[str, Any]) -> None:
    # adjust time_fit & diet_fit based on slots
    for c in cand:
        r = c["recipe"]
        # time
        tfit = 1.0
        if slots.get("time") and r.get("cook_time"):
            try:
                import re
                m = re.search(r"(\d+)", str(slots["time"]))
                if m:
                    limit = int(m.group(1))
                    t = int(r.get("cook_time", 999))
                    tfit = max(0.0, 1.0 - max(0, t - limit)/max(1.0, float(limit)))
            except Exception:
                pass
        c["time_fit"] = float(tfit)
        # diet
        dfit = 1.0
        if slots.get("diet"):
            dfit = 1.0 if slots["diet"] in r.get("diet", []) else 0.0
        c["diet_fit"] = float(dfit)
        # protein
        pfit = 1.0
        if slots.get("protein"):
            pfit = 1.0 if recipe_matches_protein(r, slots["protein"]) else 0.0
        c["protein_fit"] = float(pfit)
        # servings
        sfit = 1.0
        desired_serv = slots.get("servings")
        actual_serv = r.get("servings")
        if desired_serv and actual_serv:
            diff = abs(float(actual_serv) - float(desired_serv))
            denom = max(float(desired_serv), 1.0)
            sfit = max(0.0, 1.0 - diff/denom)
        c["servings_fit"] = float(sfit)


def score_candidates(cand: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    w = cfg.get("ranking", {})
    out = []
    for c in cand:
        s = (
            w.get("w_semantic", 0.35) * c.get("semantic_n", 0.0) +
            w.get("w_timefit",  0.20) * c.get("time_fit", 1.0) +
            w.get("w_dietfit",  0.15) * c.get("diet_fit", 1.0) +
            w.get("w_protein", 0.0) * c.get("protein_fit", 1.0) +
            w.get("w_servings", 0.0) * c.get("servings_fit", 1.0) +
            w.get("w_avail",    0.15) * c.get("availability_ratio", 0.0) +
            w.get("w_promo",    0.10) * c.get("promo_coverage", 0.0) +
            w.get("w_history",  0.05) * c.get("history_affinity", 0.0)
        )
        out.append({**c, "score": float(s)})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


class Pipeline:
    def __init__(self, cfg_path: str = "serverAI/config/app.yaml"):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
        paths = self.cfg["paths"]
        # modules
        gaz_dir = "serverAI/data/nlu/gazetteer"
        ingredient_map = load_json(os.path.join(paths["mapping_dir"], "ingredient_to_sku.json"))
        self.nlu = NLU(self.cfg["paths"]["nlu_model_dir"], gaz_dir)
        self.retriever = Retriever(self.cfg, ingredient_map=ingredient_map)
        self.mapper = CartMapper(self.cfg, ingredient_map=ingredient_map)
        # optional ranker
        self.ranker = None
        self.rank_features: List[str] = []
        self._load_ranker(paths.get("ranker_dir"))

    def query(self, text: str, top_k: int = 5) -> Dict[str, Any]:
        intent = self.nlu.predict_intent(text)
        slots = self.nlu.extract_slots(text)
        cands = self.retriever.retrieve(text)
        apply_slot_constraints(cands, slots)
        if self.ranker:
            ranked = self._rank_with_lgbm(cands)[:top_k]
        else:
            ranked = score_candidates(cands, self.cfg)[:top_k]
        # format
        out_cand = []
        for c in ranked:
            r = c["recipe"]
            out_cand.append({
                "id": r.get("id"),
                "title": r.get("title"),
                "summary": r.get("summary"),
                "image": r.get("image"),
                "cook_time": r.get("cook_time"),
                "servings": r.get("servings"),
                "score": c.get("score"),
                "score_factors": {
                    "semantic": c.get("semantic_n"),
                    "bm25_n":  c.get("bm25_n"),            # phải có! (đừng để default 0.5)
                    "time_fit": c.get("time_fit"),
                    "diet_fit": c.get("diet_fit"),
                    "availability_ratio": c.get("availability_ratio"),
                    "promo_coverage": c.get("promo_coverage", 0.0)
                }
            })
        return {
            "intents": [intent],
            "slots": slots,
            "candidates": out_cand,
            "explanations": [
                "Da ket hop BM25 + embedding, ap dung rang buoc thoi gian/diet va tinh kha dung nguyen lieu."
            ]
        }

    def suggest_cart(self, recipe_id: str, servings: int = 2) -> Dict[str, Any]:
        r = self.retriever.recipes.get(recipe_id)
        if not r:
            raise KeyError("recipe not found")
        cart = self.mapper.suggest_cart(r, servings)
        return {
            "recipe_id": recipe_id,
            "servings": servings,
            "items": cart["items"],
            "totals": {"estimated": cart["estimated"], "currency": cart["currency"]},
            "notes": cart["notes"]
        }

    # ---- Ranker helpers ----

    def _load_ranker(self, ranker_dir: Optional[str]) -> None:
        if not ranker_dir:
            return
        model_path = Path(ranker_dir)/"lgbm.txt"
        fmap_path = Path(ranker_dir)/"feature_map.json"
        if not model_path.exists():
            return
        try:
            import lightgbm as lgb
            self.ranker = lgb.Booster(model_file=str(model_path))
            if fmap_path.exists():
                with open(fmap_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.rank_features = data.get("features", [])
            if not self.rank_features:
                self.rank_features = ["semantic_n", "bm25_n", "time_fit", "diet_fit", "protein_fit", "servings_fit", "availability_ratio"]
            print(f"[INFO] Loaded ranker with {len(self.rank_features)} features from {model_path}")
        except Exception as e:
            print(f"[WARN] Failed to load ranker: {e}")
            self.ranker = None
            self.rank_features = []

    def _rank_with_lgbm(self, cand: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not cand or not self.ranker:
            return score_candidates(cand, self.cfg)
        feature_alias = {
            "semantic_sim": "semantic_n",
            "semantic": "semantic_n",
            "bm25": "bm25_n",
            "bm25_n": "bm25_n",
            "time_fit": "time_fit",
            "diet_fit": "diet_fit",
            "protein_fit": "protein_fit",
            "servings_fit": "servings_fit",
            "availability": "availability_ratio",
            "availability_ratio": "availability_ratio",
            "promo_coverage": "promo_coverage",
            "history_affinity": "history_affinity"
        }
        feats = []
        for c in cand:
            vec = []
            for name in self.rank_features:
                key = feature_alias.get(name, name)
                vec.append(float(c.get(key, 0.0)))
            feats.append(vec)
        feats_arr = np.asarray(feats, dtype=np.float32)
        try:
            scores = self.ranker.predict(feats_arr)
        except Exception as e:
            print(f"[WARN] Ranker prediction failed: {e}")
            return score_candidates(cand, self.cfg)
        out = []
        for c, s in zip(cand, scores):
            out.append({**c, "score": float(s)})
        out.sort(key=lambda x: x["score"], reverse=True)
        return out
