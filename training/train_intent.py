#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a lightweight intent classifier (TF-IDF + LogisticRegression).
Input: train.tsv, valid.tsv with format: text \t intent \t slots_json
Output: models/nlu_intent/{vectorizer.pkl, classifier.pkl, label_map.json}
"""

import os, json, argparse, sys, re
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
import joblib


def strip_diacritics(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def load_tsv(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = re.split(r"\s{2,}", line)
            if len(parts) < 2:
                continue
            text = parts[0]
            intent = parts[1]
            rows.append((text, intent))
    df = pd.DataFrame(rows, columns=["text", "intent"])
    return df


def preprocess_text(s: str, lowercase=True, remove_diacritics=True):
    s = s.strip()
    if lowercase:
        s = s.lower()
    if remove_diacritics:
        s = strip_diacritics(s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ngram_max", type=int, default=2)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    df_train = load_tsv(args.train)
    df_valid = load_tsv(args.valid)

    df_train["text"] = df_train["text"].apply(preprocess_text)
    df_valid["text"] = df_valid["text"].apply(preprocess_text)

    labels = sorted(df_train["intent"].unique().tolist())
    label_to_id = {l:i for i,l in enumerate(labels)}
    id_to_label = {i:l for l,i in label_to_id.items()}

    y_train = df_train["intent"].map(label_to_id).values
    y_valid = df_valid["intent"].map(label_to_id).fillna(-1).astype(int).values

    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, args.ngram_max), analyzer="char_wb")
    # char_wb n-grams hoạt động tốt cho tiếng Việt không dấu + biến thể chính tả ngắn.
    clf = LogisticRegression(max_iter=200, C=4.0, class_weight="balanced")

    pipe = Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf)
    ])

    pipe.fit(df_train["text"].values, y_train)

    y_pred = pipe.predict(df_valid["text"].values)
    macro_f1 = f1_score(y_valid, y_pred, average="macro")
    print("Macro-F1:", macro_f1)
    try:
        print(classification_report(y_valid, y_pred, target_names=labels))
    except Exception:
        pass

    # Save artifacts
    joblib.dump(pipe.named_steps["tfidf"], out_dir/"vectorizer.pkl")
    joblib.dump(pipe.named_steps["clf"], out_dir/"classifier.pkl")
    with open(out_dir/"label_map.json", "w", encoding="utf-8") as f:
        json.dump({"labels": labels}, f, ensure_ascii=False, indent=2)

    print("Saved to", out_dir)

if __name__ == "__main__":
    main()
