#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train LightGBM Ranker (Fixed Version).
Update: Đã bổ sung feature 'protein_fit' và 'servings_fit' để model không bị 'mù' nguyên liệu.
"""

import os
import json
import argparse
import sys
import numpy as np
import lightgbm as lgb
from pathlib import Path
from tqdm import tqdm

# Thêm đường dẫn để import các module của hệ thống
sys.path.append(os.getcwd())

try:
    from serverAI.inference.pipeline import Pipeline, apply_slot_constraints
    from serverAI.inference.utils import norm_text
except ImportError:
    print("[ERROR] Không thể import module. Hãy chạy script từ thư mục gốc (nơi chứa folder serverAI).")
    sys.exit(1)

def load_jsonl(path: str):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            if "id" in obj:
                data[obj["id"]] = obj
    return data

def load_judgments(path: str):
    judgments = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            judgments.append(json.loads(line))
    return judgments

class FeatureExtractor:
    def __init__(self, cfg_path):
        print("Đang khởi tạo Pipeline để load tài nguyên...")
        self.pipe = Pipeline(cfg_path)
        self.retriever = self.pipe.retriever
        self.nlu = self.pipe.nlu
        self.recipe_map = self.retriever.recipes
        self.rid_to_idx = {rid: i for i, rid in enumerate(self.retriever.recipe_ids)}

    def compute_features(self, query_text, recipe_id):
        r = self.recipe_map.get(recipe_id)
        if not r: return None

        # 1. NLU Extraction
        slots = self.nlu.extract_slots(query_text)
        
        # 2. BM25 Score
        idx = self.rid_to_idx.get(recipe_id)
        if idx is None: return None
        q_toks = self.retriever._tokenize(norm_text(query_text))
        bm25_score = self.retriever.bm25.get_scores(q_toks)[idx]

        # 3. Semantic Score
        q_emb = self.retriever.embedder.encode([norm_text(query_text)], normalize_embeddings=True)[0]
        d_emb = self.retriever.emb[idx]
        semantic_score = float(np.dot(q_emb, d_emb))

        # 4. Constraint Fits (Quan trọng nhất!)
        cand = [{"recipe": r, "time_fit": 0, "diet_fit": 0, "protein_fit": 0, "servings_fit": 0}]
        apply_slot_constraints(cand, slots)
        c = cand[0]

        # 5. Availability
        avail_ratio = self.retriever._availability_ratio(r)

        return [
            semantic_score,   # 0
            bm25_score,       # 1
            c["time_fit"],    # 2
            c["diet_fit"],    # 3
            c["protein_fit"], # 4 <--- QUAN TRỌNG: Đã thêm vào
            c["servings_fit"],# 5 <--- QUAN TRỌNG: Đã thêm vào
            avail_ratio       # 6
        ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queries', default='serverAI/eval/queries.jsonl')
    ap.add_argument('--judgments', default='serverAI/eval/judgments.jsonl')
    ap.add_argument('--config', default='serverAI/config/app.yaml')
    ap.add_argument('--out', default='serverAI/models/ranker')
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    queries = load_jsonl(args.queries)
    judgments = load_judgments(args.judgments)
    
    grouped_data = {}
    for j in judgments:
        qid = j['query_id']
        if qid not in grouped_data: grouped_data[qid] = []
        grouped_data[qid].append((j['recipe_id'], j['label']))

    extractor = FeatureExtractor(args.config)
    X, y, group = [], [], []

    print(f"Extracting features for {len(grouped_data)} queries...")
    for qid, items in tqdm(grouped_data.items()):
        if qid not in queries: continue
        query_text = queries[qid]['text']
        cnt = 0
        for rid, label in items:
            feats = extractor.compute_features(query_text, rid)
            if feats:
                X.append(feats)
                y.append(int(label))
                cnt += 1
        if cnt > 0: group.append(cnt)

    X = np.array(X)
    y = np.array(y)
    group = np.array(group)

    print("Training LightGBM Lambdarank...")
    train_data = lgb.Dataset(X, label=y, group=group)
    
    # Feature names mới (7 features)
    feature_names = ["semantic_sim", "bm25", "time_fit", "diet_fit", "protein_fit", "servings_fit", "availability"]
    
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_data_in_leaf': 1,
        'verbose': -1
    }
    
    model = lgb.train(params, train_data, num_boost_round=100)
    model.save_model(str(out_dir / "lgbm.txt"))
    
    with open(out_dir / "feature_map.json", "w", encoding="utf-8") as f:
        json.dump({"features": feature_names}, f, indent=2)

    print(f"✅ Đã lưu model mới. Feature Importance:")
    for name, imp in zip(feature_names, model.feature_importance()):
        print(f"  - {name}: {imp}")

if __name__ == '__main__':
    main()