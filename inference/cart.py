# File: inference/cart.py
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .utils import normalize_ingredient_mappings


def normalize_to_base_unit(qty: float, unit: str) -> tuple[float, str]:
    unit = (unit or "").lower().strip()
    if unit in ["kg", "kilogram", "kilo"]:
        return qty * 1000.0, "g"
    if unit in ["g", "gr", "gram", "gam"]:
        return qty, "g"
    if unit in ["l", "lit", "lít", "liter"]:
        return qty * 1000.0, "ml"
    if unit in ["ml", "mililit"]:
        return qty, "ml"
    if unit in ["quả", "trái", "trứng", "cái"]:
        return qty, "quả"
    return qty, unit


class CartMapper:
    def __init__(
        self,
        cfg: Dict[str, Any],
        ingredient_map: Optional[Dict[str, Any]] = None,
        product_catalog: Optional[Dict[str, Any]] = None,
    ):
        self.cfg = cfg
        self.map = normalize_ingredient_mappings(ingredient_map or {})
        self.catalog = product_catalog or {}

    def suggest_cart(self, recipe: Dict[str, Any], servings: int) -> Dict[str, Any]:
        items, total = [], 0.0
        scale = float(servings) / (float(recipe.get("servings", 1)) or 1.0)
        for ing in recipe.get("ingredients", []):
            name = ing.get("name")
            qty_raw = float(ing.get("qty", 0) or 0) * scale
            unit_raw = ing.get("unit", "")
            qty, unit = normalize_to_base_unit(qty_raw, unit_raw)

            mapping_entries = self.map.get(name) or []
            prod = None
            chosen_mapping = None
            for m in mapping_entries:
                p = self.catalog.get(m.get("sku"))
                if not p:
                    continue
                if p.get("stock", 0) <= 0:
                    continue
                prod = p
                chosen_mapping = m
                break

            if (qty <= 0) and chosen_mapping:
                ratio = chosen_mapping.get("ratio_per_serving") or {}
                try:
                    ratio_qty = float(ratio.get("qty", 0) or 0) * float(servings)
                    ratio_unit = ratio.get("unit") or unit
                    if ratio_qty > 0:
                        qty, unit = normalize_to_base_unit(ratio_qty, ratio_unit)
                except Exception:
                    pass

            if not prod:
                items.append(
                    {
                        "ingredient": name,
                        "sku": None,
                        "name": f"{name} (N/A)",
                        "price": 0,
                        "subtotal": 0,
                        "is_missing": True,
                    }
                )
                continue

            prod_unit = (prod.get("measureUnit") or "").lower().strip()
            unit_size = float(prod.get("unitSize", 1) or 1)
            if qty <= 0 or unit_size <= 0:
                pkgs = 1
            elif prod_unit and (prod_unit == unit):
                pkgs = int(np.ceil(qty / unit_size))
            else:
                pkgs = 1

            subtotal = pkgs * float(prod.get("price", 0))
            total += subtotal
            items.append(
                {
                    "ingredient": name,
                    "sku": prod.get("sku"),
                    "name": prod.get("name"),
                    "packages": pkgs,
                    "price": prod.get("price"),
                    "subtotal": subtotal,
                    "stock_ok": True,
                }
            )
        return {"items": items, "estimated": float(total), "currency": "VND", "notes": []}
