# File: inference/ranking.py
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .utils import norm_text


def recipe_matches_protein(recipe: Dict[str, Any], protein: str) -> bool:
    if not protein:
        return True
    target = norm_text(protein, lowercase=True, strip_dia=True)
    if not target:
        return True
    title = norm_text(recipe.get("title", ""), lowercase=True, strip_dia=True)
    if target in title:
        return True
    for tag in recipe.get("tags", []):
        if target in norm_text(str(tag), lowercase=True, strip_dia=True):
            return True
    for ing in recipe.get("ingredients", []):
        names = [ing.get("name", "")]
        names.extend(ing.get("aliases", []))
        for name in names:
            if target in norm_text(name, lowercase=True, strip_dia=True):
                return True
    return False


def apply_slot_constraints(cand: List[Dict[str, Any]], slots: Dict[str, Any]) -> None:
    for c in cand:
        r = c["recipe"]
        tfit = 1.0
        if slots.get("time") and r.get("cook_time"):
            try:
                import re

                m = re.search(r"(\\d+)", str(slots["time"]))
                if m:
                    limit = int(m.group(1))
                    t = int(r.get("cook_time", 999))
                    tfit = max(0.0, 1.0 - max(0, t - limit) / max(1.0, float(limit)))
            except Exception:
                pass
        c["time_fit"] = float(tfit)
        c["diet_fit"] = 1.0 if not slots.get("diet") else (1.0 if slots["diet"] in r.get("diet", []) else 0.0)
        c["protein_fit"] = 1.0 if not slots.get("protein") else (1.0 if recipe_matches_protein(r, slots["protein"]) else 0.0)
        c["servings_fit"] = 1.0


def score_candidates(cand: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    w = cfg.get("ranking", {})
    out = []
    for c in cand:
        s = (
            w.get("w_semantic", 0.35) * c.get("semantic_n", 0)
            + w.get("w_timefit", 0.2) * c.get("time_fit", 1)
            + w.get("w_dietfit", 0.15) * c.get("diet_fit", 1)
            + w.get("w_protein", 0.2) * c.get("protein_fit", 0)
            + w.get("w_avail", 0.05) * c.get("availability_ratio", 0)
        )
        out.append({**c, "score": float(s)})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out

