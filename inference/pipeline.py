# File: inference/pipeline.py
"""End-to-end inference pipeline: NLU → Context DST → Retrieval → Ranking → Response."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import yaml

from .cart import CartMapper
from .nlu_engine import NLU
from .ranking import apply_slot_constraints, score_candidates
from .retrieval import Retriever
from .utils import load_json, normalize_ingredient_mappings


def _load_mapping_fallback(mapping_dir: str) -> Any:
    for fn in ["ingredient_mappings.json", "ingredient_mapping.json"]:
        fp = os.path.join(mapping_dir, fn)
        if Path(fp).exists():
            return load_json(fp)
    raise FileNotFoundError(f"No ingredient mapping file found in `{mapping_dir}`")


class Pipeline:
    def __init__(
        self,
        cfg_path: str = "serverAI/config/app.yaml",
        product_catalog: Optional[Dict[str, Any]] = None,
        ingredient_map: Optional[Dict[str, Any]] = None,
        recipes: Optional[Dict[str, Any]] = None,
        recipe_fetcher: Optional[Callable[[List[str]], Dict[str, Any]]] = None,
    ):
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        paths = self.cfg["paths"]
        self.catalog = product_catalog or {}

        if ingredient_map:
            print(f"[INFO] Pipeline: Using Dynamic Map ({len(ingredient_map)} keys)")
            raw_map = ingredient_map
        else:
            print("[WARN] Pipeline: Fallback to Static JSON Map")
            raw_map = _load_mapping_fallback(paths["mapping_dir"])
        self.map_data = normalize_ingredient_mappings(raw_map)

        gaz_dir = "serverAI/data/nlu/gazetteer"
        self.nlu = NLU(self.cfg["paths"]["nlu_model_dir"], gaz_dir)
        self.retriever = Retriever(
            self.cfg,
            ingredient_map=self.map_data,
            product_catalog=self.catalog,
            recipes=recipes,
            recipe_fetcher=recipe_fetcher,
        )
        self.mapper = CartMapper(self.cfg, ingredient_map=self.map_data, product_catalog=self.catalog)
        self.ranker = None
        self.rank_features: List[str] = []
        self._load_ranker(paths.get("ranker_dir"))

    def query(self, text: str, context: Dict[str, Any] = None, top_k: int = 5) -> Dict[str, Any]:
        if context is None:
            context = {}

        intent = self.nlu.predict_intent(text)
        new_slots = self.nlu.extract_slots(text)

        if intent["name"] in ["greeting", "greeting_hello", "smalltalk_greeting"]:
            return {
                "intents": [intent],
                "slots": context.get("slots", {}),
                "candidates": [],
                "explanations": ["Xin chào! Mình là trợ lý ẩm thực thông minh. Bạn muốn tìm món ăn gì hôm nay?"],
            }

        if intent["name"] in ["goodbye"]:
            return {
                "intents": [intent],
                "slots": {},
                "candidates": [],
                "explanations": ["Tạm biệt bạn! Hẹn gặp lại trong bữa ăn tới nhé."],
            }

        current_slots = context.get("slots", {}).copy()
        merged_slots: Dict[str, Any] = {}
        intent_name = intent["name"]

        if intent_name in ["suggest_food", "suggest_food_by_ingredient"]:
            merged_slots = new_slots
            if "diet" in current_slots and "diet" not in merged_slots:
                merged_slots["diet"] = current_slots["diet"]
            if "allergy" in current_slots:
                merged_slots["allergy"] = current_slots["allergy"]
        elif intent_name in ["suggest_food_by_time", "ask_price", "filter_diet"]:
            merged_slots = current_slots
            for k, v in new_slots.items():
                if v:
                    merged_slots[k] = v
        else:
            merged_slots = current_slots

        search_query = text
        if intent_name in ["suggest_food_by_time", "ask_price", "filter_diet"]:
            relevant_context = []
            if "protein" in merged_slots:
                relevant_context.append(merged_slots["protein"])

            if relevant_context and len(text.split()) < 4:
                context_str = " ".join(relevant_context)
                search_query = f"{context_str} {text}"
                print(f"[INFO] Rewritten Query: '{text}' -> '{search_query}'")

        cands = self.retriever.retrieve(search_query)
        apply_slot_constraints(cands, merged_slots)

        if self.ranker:
            ranked = self._rank_with_lgbm(cands)[:top_k]
        else:
            ranked = score_candidates(cands, self.cfg)[:top_k]

        out_cand = []
        for c in ranked:
            r = c["recipe"]
            out_cand.append(
                {
                    "id": r.get("id"),
                    "title": r.get("title"),
                    "summary": r.get("summary"),
                    "image": r.get("image"),
                    "cook_time": r.get("cook_time"),
                    "servings": r.get("servings"),
                    "score": c.get("score"),
                    "score_factors": {
                        "semantic": c.get("semantic_n"),
                        "time_fit": c.get("time_fit"),
                        "availability": c.get("availability_ratio"),
                    },
                }
            )

        explanation = "Dưới đây là các gợi ý phù hợp nhất cho bạn."
        if intent_name == "ask_price":
            explanation = "Dưới đây là các món ăn kèm thông tin nguyên liệu. Bạn có thể chọn 'Suggest Cart' để xem giá chi tiết."
        elif not out_cand:
            explanation = "Tiếc quá, mình chưa tìm thấy món nào phù hợp với yêu cầu này. Bạn thử nguyên liệu khác xem sao?"

        return {
            "intents": [intent],
            "slots": merged_slots,
            "candidates": out_cand,
            "explanations": [explanation],
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
            "notes": cart["notes"],
        }

    def _load_ranker(self, ranker_dir: Optional[str]) -> None:
        if not ranker_dir:
            return
        model_path = Path(ranker_dir) / "lgbm.txt"
        if not model_path.exists():
            return
        try:
            import lightgbm as lgb

            self.ranker = lgb.Booster(model_file=str(model_path))
            self.rank_features = [
                "semantic_n",
                "bm25_n",
                "time_fit",
                "diet_fit",
                "protein_fit",
                "servings_fit",
                "availability_ratio",
            ]
            print(f"[INFO] Loaded ranker from {model_path}")
        except Exception as e:
            print(f"[WARN] Failed to load ranker: {e}")
            self.ranker = None

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
            "history_affinity": "history_affinity",
        }
        feats = []
        for c in cand:
            vec = [float(c.get(feature_alias.get(name, name), 0.0)) for name in self.rank_features]
            feats.append(vec)
        try:
            scores = self.ranker.predict(np.asarray(feats, dtype=np.float32))
            out = []
            for c, s in zip(cand, scores):
                out.append({**c, "score": float(s)})
            out.sort(key=lambda x: x["score"], reverse=True)
            return out
        except Exception:
            return score_candidates(cand, self.cfg)

