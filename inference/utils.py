import json
import os
import unicodedata
import re
from typing import List, Dict, Any

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