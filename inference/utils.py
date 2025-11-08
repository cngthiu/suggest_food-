# -*- coding: utf-8 -*-
import json, os, unicodedata
from typing import List, Dict, Any


def strip_diacritics(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def norm_text(s: str, lowercase=True, strip_dia=True) -> str:
    if s is None: return ""
    s = s.strip()
    if lowercase: s = s.lower()
    if strip_dia: s = strip_diacritics(s)
    return s


def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
