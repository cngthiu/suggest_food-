### `serverAI/tools/labeling_cli.py`
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI gán nhãn thủ công cho reranking (Learning-to-Rank)
Sinh ra 2 file: serverAI/eval/queries.jsonl và serverAI/eval/judgments.jsonl
Cách dùng:
  python serverAI/tools/labeling_cli.py \
    --seed serverAI/eval/queries_seed.txt \
    --outq serverAI/eval/queries.jsonl \
    --outj serverAI/eval/judgments.jsonl \
    --topk 5
"""

import os, json, argparse, sys
from pathlib import Path

# import pipeline
from serverAI.inference.pipeline import Pipeline


def write_jsonl(path: str, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "")


def append_jsonl(path: str, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', required=True, help='queries_seed.txt: mỗi dòng 1 câu truy vấn')
    ap.add_argument('--outq', default='serverAI/eval/queries.jsonl')
    ap.add_argument('--outj', default='serverAI/eval/judgments.jsonl')
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--append', action='store_true', help='ghi nối tiếp vào file out thay vì ghi đè')
    args = ap.parse_args()

    pipe = Pipeline('serverAI/config/app.yaml')

    # nạp seed queries
    with open(args.seed, 'r', encoding='utf-8') as f:
        raw_queries = [line.strip() for line in f if line.strip()]

    q_records = []
    j_records = []

    print("=== BẮT ĐẦU GÁN NHÃN (0=không liên quan, 1=hơi liên quan, 2=phù hợp, 3=rất phù hợp) ===")

    for qi, text in enumerate(raw_queries, start=1):
        qid = f"q{qi}"
        # chạy pipeline để lấy candidates
        out = pipe.query(text, top_k=args.topk)
        intents = out.get('intents', [])
        slots = out.get('slots', {})
        cands = out.get('candidates', [])

        # lưu query record
        q_records.append({"id": qid, "text": text, "slots": slots, "intent": intents[0] if intents else None})

        # hiển thị
        print(f"[Query {qid}] {text}")
        if slots:
            print("  Slots:", slots)
        print("  Ứng viên:")
        for idx, c in enumerate(cands, start=1):
            print(f"    ({idx}) {c['id']} | {c.get('title')} | time={c.get('cook_time')}p | score≈{c.get('score'):.3f}")

        # nhập nhãn cho từng ứng viên
        for idx, c in enumerate(cands, start=1):
            while True:
                try:
                    val = input(f"  → Nhập nhãn cho ({idx}) {c['id']} [0-3, Enter = bỏ qua=0]: ").strip()
                    if val == "":
                        label = 0
                    else:
                        label = int(val)
                        if label < 0 or label > 3:
                            raise ValueError
                    break
                except Exception:
                    print("    Nhập không hợp lệ, vui lòng gõ số 0..3 hoặc Enter")
            if label > 0:
                j_records.append({
                    "query_id": qid,
                    "recipe_id": c['id'],
                    "label": label
                })

    # ghi file
    if args.append and os.path.exists(args.outq):
        append_jsonl(args.outq, q_records)
    else:
        write_jsonl(args.outq, q_records)

    if args.append and os.path.exists(args.outj):
        append_jsonl(args.outj, j_records)
    else:
        write_jsonl(args.outj, j_records)

    print(f"✅ Đã ghi {len(q_records)} queries vào {args.outq}")
    print(f"✅ Đã ghi {len(j_records)} judgments vào {args.outj}")
    print("Gợi ý tiếp theo: train ranker hoặc đánh giá chất lượng (NDCG@k, Recall@k).")


if __name__ == '__main__':
    main()