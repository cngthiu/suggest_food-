import json
import os
import unicodedata
import re
from typing import List, Dict, Any

def strip_diacritics(s: str) -> str:
    """
    Chuyển đổi tiếng Việt có dấu thành không dấu.
    Ví dụ: 'Món Cá' -> 'Mon Ca'
    """
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")

def norm_text(s: str, lowercase=True, strip_dia=True) -> str:
    """
    Chuẩn hóa văn bản đồng bộ cho cả Training và Inference.
    
    Các bước xử lý:
    1. An toàn: Chuyển None -> rỗng, số -> string.
    2. Khoảng trắng: Xóa khoảng trắng thừa ở giữa và 2 đầu.
    3. Unicode: Chuẩn hóa về NFC (dựng sẵn) để thống nhất mã.
    4. Tùy chọn: Chuyển thường (lowercase) và bỏ dấu (strip_dia).
    """
    if s is None:
        return ""
    
    # Đảm bảo là string
    s = str(s)
    
    # 1. Chuẩn hóa khoảng trắng (Replace multiple spaces with single space)
    # Đây là bước quan trọng nhất để fix lỗi vị trí (offset) khi train NER
    s = re.sub(r'\s+', ' ', s).strip()
    
    # 2. Chuẩn hóa Unicode về NFC (Normalization Form C)
    # Giúp đồng bộ font chữ, tránh lỗi so sánh chuỗi
    s = unicodedata.normalize('NFC', s)
    
    # 3. Chuyển chữ thường
    if lowercase:
        s = s.lower()
        
    # 4. Bỏ dấu tiếng Việt
    if strip_dia:
        s = strip_diacritics(s)
        
    return s

def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)