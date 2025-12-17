import json
import os
import unicodedata
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- THÊM MỚI: Bắt buộc import underthesea ---
try:
    from underthesea import word_tokenize
except ImportError:
    raise ImportError("Thư viện 'underthesea' chưa được cài đặt. Vui lòng chạy: pip install underthesea")

def strip_diacritics(s: str) -> str:
    """Chuyển đổi tiếng Việt có dấu thành không dấu."""
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")

def norm_text(s: str, lowercase=True, strip_dia=True) -> str:
    """Chuẩn hóa văn bản đồng bộ."""
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = unicodedata.normalize('NFC', s)
    if lowercase:
        s = s.lower()
    if strip_dia:
        s = strip_diacritics(s)
    return s

def tokenize_vi(s: str) -> List[str]:
    """
    Hàm tách từ chuẩn cho tiếng Việt.
    Input: "gà kho gừng"
    Output: ["gà_kho", "gừng"] (Thay vì ["gà", "kho", "gừng"])
    """
    if not s: 
        return []
    
    # 1. Chuẩn hóa sơ bộ trước khi token (giữ dấu để tách từ đúng)
    s_clean = norm_text(s, lowercase=True, strip_dia=False)
    
    # 2. Tách từ bằng Underthesea
    # format="text" sẽ nối từ ghép bằng gạch dưới (ví dụ: củ_cải)
    tokens = word_tokenize(s_clean, format="text").split()
    
    return [t for t in tokens if t.strip()]

def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_recipes_any(recipes_path: str) -> List[Dict[str, Any]]:
    p = Path(recipes_path)

    if p.is_dir():
        candidates = [
            p / "recipies.json",
            p / "recipes.json",
            p / "recipe.json",
        ]
        found = next((c for c in candidates if c.exists() and c.is_file()), None)
        if found is not None:
            p = found
        else:
            js = sorted(p.glob("*.json"))
            if not js:
                return []
            p = js[0]

    if not p.exists():
        raise FileNotFoundError(f"Recipes path not found: {p}")

    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]

    if isinstance(data, dict):
        if isinstance(data.get("recipes"), list):
            return [r for r in data["recipes"] if isinstance(r, dict)]
        if isinstance(data.get("items"), list):
            return [r for r in data["items"] if isinstance(r, dict)]
        if all(isinstance(v, dict) for v in data.values()):
            return list(data.values())

    raise ValueError(f"Unsupported recipes format: root type={type(data)} at {p}")


def recipes_to_dict(recipes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in recipes:
        rid = r.get("id") or r.get("recipe_id")
        if rid is None:
            continue
        out[str(rid)] = r
    return out


def normalize_ingredient_mappings(raw: Any) -> Dict[str, List[Dict[str, Any]]]:
    """
    Chuẩn hoá mapping về dạng:
      { "<ingredient>": [ {"sku": "...", "ratio_per_serving": {"qty": 100, "unit": "g"}, ...}, ... ] }

    Hỗ trợ 2 kiểu input phổ biến:
    - MongoDB: dict[str, dict] (mỗi key trỏ tới 1 mapping entry)
    - JSON file ingredient_mappings.json: list[{ingredient_key, aliases, priority_skus, ...}]
    """
    if raw is None:
        return {}

    normalized: Dict[str, List[Dict[str, Any]]] = {}

    def add_key(key: Optional[str], entries: List[Dict[str, Any]]) -> None:
        if not key:
            return
        k = str(key).strip()
        if not k:
            return
        normalized[k] = entries

    if isinstance(raw, dict):
        for key, value in raw.items():
            if value is None:
                continue
            if isinstance(value, list):
                entries = [v for v in value if isinstance(v, dict) and v.get("sku")]
                add_key(key, entries)
                continue
            if isinstance(value, dict):
                sku = value.get("sku") or value.get("sku_ref")
                if not sku:
                    continue
                entry = {**value, "sku": sku}
                add_key(key, [entry])
                continue
        return normalized

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            ingredient_key = item.get("ingredient_key") or item.get("key") or item.get("ingredient")
            aliases = item.get("aliases") or []
            skus = item.get("priority_skus") or item.get("skus") or []

            entries: List[Dict[str, Any]] = []
            for sku in skus:
                if not sku:
                    continue
                entries.append(
                    {
                        "sku": sku,
                        "ratio_per_serving": item.get("ratio_per_serving"),
                        "is_staple": item.get("is_staple"),
                    }
                )

            add_key(ingredient_key, entries)
            for alias in aliases:
                add_key(alias, entries)

        return normalized

    raise ValueError(f"Unsupported ingredient mapping format: {type(raw)}")
