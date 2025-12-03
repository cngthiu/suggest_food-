#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI gán nhãn thủ công cho reranking (Learning-to-Rank)
Sinh ra 2 file: serverAI/eval/queries.jsonl và serverAI/eval/judgments.jsonl

Tính năng mới:
  - Hiển thị Summary để dễ đánh giá.
  - Tự động bỏ qua các câu đã gán nhãn (Resume).
  - Giao diện trực quan hơn.
"""

import os
import json
import argparse
import sys
from pathlib import Path

# Thêm đường dẫn gốc để import modules
sys.path.append(os.getcwd())

try:
    from serverAI.inference.pipeline import Pipeline
except ImportError:
    print("❌ Lỗi: Không tìm thấy module serverAI. Hãy chạy script từ thư mục gốc của dự án.")
    sys.exit(1)

def write_jsonl(path: str, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def append_jsonl(path: str, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_existing_queries(path):
    ids = set()
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    ids.add(obj.get('text')) # Dùng text làm key để check trùng
                except: pass
    return ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', required=True, help='Path file seed queries (txt)')
    ap.add_argument('--outq', default='serverAI/eval/queries.jsonl')
    ap.add_argument('--outj', default='serverAI/eval/judgments.jsonl')
    ap.add_argument('--config', default='serverAI/config/app.yaml')
    ap.add_argument('--topk', type=int, default=3)
    ap.add_argument('--append', action='store_true', help='Ghi nối tiếp (mặc định nên dùng)')
    args = ap.parse_args()

    # 1. Load Pipeline
    print(f"⏳ Đang khởi tạo Pipeline từ {args.config}...")
    try:
        pipe = Pipeline(args.config)
    except Exception as e:
        print(f"❌ Lỗi khởi tạo Pipeline: {e}")
        return

    # 2. Load Seed Queries
    if not os.path.exists(args.seed):
        print(f"❌ Không tìm thấy file seed: {args.seed}")
        return
        
    with open(args.seed, 'r', encoding='utf-8') as f:
        raw_queries = [line.strip() for line in f if line.strip()]

    # 3. Check Resume (Bỏ qua câu đã làm)
    existing_texts = load_existing_queries(args.outq)
    print(f"ℹ️  Đã có {len(existing_texts)} câu trong dữ liệu cũ.")

    q_buffer = []
    j_buffer = []

    print("\n=== BẮT ĐẦU GÁN NHÃN ===")
    print("Hướng dẫn: 3=Rất tốt, 2=Khá, 1=Hơi liên quan, 0=Sai/Bỏ qua")
    print("Nhấn 'q' hoặc Ctrl+C để thoát và lưu.\n")

    try:
        count = 0
        for qi, text in enumerate(raw_queries, start=1):
            # Skip nếu đã làm rồi
            if text in existing_texts:
                continue

            qid = f"q{len(existing_texts) + count + 1:03d}" # ID tăng dần: q001, q002...
            
            print(f"\n🔹 [{qid}] Query: \"{text}\"")
            
            # Gọi Pipeline
            out = pipe.query(text, top_k=args.topk)
            intents = out.get('intents', [])
            slots = out.get('slots', {})
            cands = out.get('candidates', [])

            # In thông tin NLU để kiểm tra
            intent_name = intents[0]['name'] if intents else "Unknown"
            print(f"   [NLU] Intent: {intent_name} | Slots: {slots}")

            if not cands:
                print("   ⚠️  Không tìm thấy ứng viên nào.")
                # Vẫn lưu query để biết là hệ thống fail
                q_buffer.append({"id": qid, "text": text, "slots": slots, "intent": intents[0] if intents else None})
                continue

            # In danh sách ứng viên
            print(f"   Found {len(cands)} candidates:")
            for idx, c in enumerate(cands, start=1):
                title = c.get('title', 'No Title')
                summary = c.get('summary', '')[:100] # Lấy 100 ký tự đầu
                time = c.get('cook_time', '?')
                score = c.get('score', 0.0)
                print(f"   ({idx}) {title.upper()} ({time}p)")
                print(f"       Info: {summary}...")
                print(f"       Score: {score:.4f} (ID: {c['id']})")

            # Nhập nhãn
            print("   👉 Nhập điểm (ví dụ: 3 2 0) tương ứng thứ tự trên, hoặc Enter từng dòng:")
            labels = []
            
            # Cách nhập nhanh: gõ "3 2 1" rồi enter
            val = input("      Labels > ").strip()
            if val.lower() == 'q': break
            
            if " " in val or len(val) == len(cands):
                # Xử lý nhập một lèo
                parts = val.replace(" ", "")
                for i, char in enumerate(parts):
                    if i >= len(cands): break
                    try:
                        lbl = int(char)
                        labels.append((cands[i]['id'], lbl))
                    except: pass
            else:
                # Xử lý nhập lẻ (nếu dòng trên trống hoặc sai)
                if val: 
                    try: labels.append((cands[0]['id'], int(val)))
                    except: pass
                
                start_idx = 1 if val else 0
                for idx in range(start_idx, len(cands)):
                    c = cands[idx]
                    while True:
                        v = input(f"      Label cho ({idx+1}) > ").strip()
                        if v.lower() == 'q': raise KeyboardInterrupt
                        if v == "": v = "0"
                        try:
                            lbl = int(v)
                            if 0 <= lbl <= 3:
                                labels.append((c['id'], lbl))
                                break
                        except: pass

            # Lưu vào buffer
            q_buffer.append({"id": qid, "text": text, "slots": slots, "intent": intents[0] if intents else None})
            for rid, lbl in labels:
                j_buffer.append({"query_id": qid, "recipe_id": rid, "label": lbl})
            
            count += 1
            
            # Auto save mỗi 5 câu để tránh mất điện
            if count % 5 == 0:
                append_jsonl(args.outq, q_buffer)
                append_jsonl(args.outj, j_buffer)
                q_buffer, j_buffer = [], []
                print("   (Đã autosave)")

    except KeyboardInterrupt:
        print("\n\nĐã dừng bởi người dùng.")

    # Lưu nốt phần còn lại
    if q_buffer:
        append_jsonl(args.outq, q_buffer)
        append_jsonl(args.outj, j_buffer)
        print(f"✅ Đã lưu {len(q_buffer)} queries mới vào {args.outq}")
    else:
        print("Không có dữ liệu mới để lưu.")

if __name__ == '__main__':
    main()