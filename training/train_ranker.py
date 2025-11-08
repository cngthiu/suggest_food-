#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a LightGBM ranking model using queries.jsonl & judgments.jsonl.
This is optional; MVP can use rule-based scores. 
Features generated on-the-fly: BM25 score, semantic similarity, time_fit, diet_fit.
"""

import os, json, argparse
from pathlib import Path
import numpy as np
import lightgbm as lgb

from typing import Dict, Any, List

# Minimal helpers (we will reuse logic from inference later if needed)

def load_jsonl(path: str):
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            items.append(json.loads(line))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queries', required=True)
    ap.add_argument('--judgments', required=True)
    ap.add_argument('--cache', required=True, help='.cache folder with ids, bm25, faiss, embeds')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: For brevity, we assume features are precomputed elsewhere.
    # In real use, you'd compute BM25 scores & cosine similarities per (query, recipe).
    # Here, we provide a stub training showcasing LightGBM API.

    # Dummy training to make script runnable; replace with real features later.
    X = np.random.rand(100, 6).astype(np.float32)
    y = np.random.rand(100).astype(np.float32)
    qid = np.array([10]*10 + [5]*5 + [20]*20 + [65]*65)  # group sizes sum to 100

    dataset = lgb.Dataset(X, label=y, group=qid)
    params = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[5])
    model = lgb.train(params, dataset, num_boost_round=50)
    model.save_model(str(out_dir/"lgbm.txt"))

    with open(out_dir/"feature_map.json", "w", encoding="utf-8") as f:
        json.dump({"features": ["semantic_sim", "bm25", "time_fit", "diet_fit", "availability", "promo"]}, f, indent=2)

    print("Saved ranker to", out_dir)

if __name__ == '__main__':
    main()
