#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script đồng bộ dữ liệu từ Intent (TSV) sang NER (JSON).
Tác dụng: Tận dụng dữ liệu đã viết cho Intent để train NER, tiết kiệm công sức.

Cách chạy:
  python serverAI/tools/sync_data.py
"""

import json
import os
import re
import sys
import unicodedata

# Cố gắng import hàm chuẩn hóa chung để đảm bảo đồng bộ
try:
    # Thêm thư mục gốc vào path để import
    sys.path.append(os.getcwd())
    from serverAI.inference.utils import norm_text
except ImportError:
    # Fallback nếu chạy trực tiếp mà không tìm thấy utils
    def strip_diacritics(s: str) -> str:
        s = unicodedata.normalize("NFD", s)
        return "".join(c for c in s if unicodedata.category(c) != "Mn")

    def norm_text(s: str, lowercase=True, strip_dia=True) -> str:
        if not s: return ""
        s = str(s)
        s = re.sub(r'\s+', ' ', s).strip()
        s = unicodedata.normalize('NFC', s)
        if lowercase: s = s.lower()
        if strip_dia: s = strip_diacritics(s)
        return s

# --- CẤU HÌNH MAPPING ---
# Ánh xạ từ tên Slot (trong Intent) sang nhãn Entity (cho NER)
SLOT_TO_ENTITY = {
    "protein": "FOOD",
    "ingredient": "FOOD",
    "species": "FOOD",
    "time": "TIME",
    "servings": "QUANTITY",
    "price": "PRICE",
    # "device": "DEVICE", # Bỏ comment nếu muốn train NER nhận diện thiết bị
    # "diet": "DIET"      # Bỏ comment nếu muốn train NER nhận diện chế độ ăn
}

def load_gazetteer(gaz_dir):
    """
    Load từ điển để biết các từ đồng nghĩa.
    Ví dụ: Slot là 'ga' -> Từ điển cho biết trong câu có thể là 'gà', 'thịt gà', 'cánh gà'...
    """
    mapping = {}
    if not os.path.exists(gaz_dir):
        return mapping
    
    for filename in os.listdir(gaz_dir):
        if not filename.endswith(".txt"): continue
        key = filename.replace(".txt", "") # protein, diet...
        mapping[key] = {}
        
        with open(os.path.join(gaz_dir, filename), 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if not parts: continue
                canonical = parts[0].strip() # Từ chuẩn (ví dụ: ga)
                variations = [p.strip() for p in parts] # Các biến thể
                mapping[key][canonical] = variations
    return mapping

def find_entity_in_text(text_norm, slot_value, slot_type, gaz_map):
    """
    Tìm vị trí chính xác của slot_value trong text_norm.
    Trả về: (start, end, label) hoặc None
    """
    candidates = []

    # 1. Nếu là FOOD (protein, ingredient...), dùng Gazetteer để mở rộng từ khóa tìm kiếm
    if slot_type in ['protein', 'ingredient', 'species']:
        # Tìm trong gazetteer xem slot_value (vd: 'ga') có những từ đồng nghĩa nào
        if slot_type in gaz_map and slot_value in gaz_map[slot_type]:
            candidates.extend(gaz_map[slot_type][slot_value])
        else:
            candidates.append(slot_value)
    
    # 2. Nếu là số liệu (TIME, QUANTITY, PRICE), cần parse số từ slot value
    elif slot_type in ['time', 'servings', 'price']:
        # Lấy phần số (ví dụ: "<=30" -> "30", "100k" -> "100")
        num_match = re.search(r'\d+', str(slot_value))
        if num_match:
            num = num_match.group(0)
            # Tạo các mẫu regex để tìm trong câu văn tự nhiên
            if SLOT_TO_ENTITY[slot_type] == "TIME":
                # Tìm: số + (khoảng trắng tùy ý) + từ chỉ thời gian
                # Vd: 30p, 30 phut, 30  phút
                candidates.append(rf"{num}\s*(phut|p|h|gio|tieng|h30|r|ruoi)") 
            elif SLOT_TO_ENTITY[slot_type] == "QUANTITY":
                candidates.append(rf"{num}\s*(nguoi|suat|phan|ban|thanh vien|bat|to)")
                candidates.append(rf"(cho|nhom)\s*{num}") # vd: cho 3
            elif SLOT_TO_ENTITY[slot_type] == "PRICE":
                candidates.append(rf"{num}\s*(k|nghin|ngan|d|vnd|dong|tram|trieu|tr)")
        
        # Luôn thêm chính giá trị gốc vào để tìm (fallback)
        candidates.append(re.escape(str(slot_value)))

    # 3. Quét text để tìm vị trí
    # Sắp xếp candidate dài trước để khớp chính xác nhất (greedy match)
    # Lưu ý: Với regex candidates, ta không sort theo len string mà để nguyên
    
    label = SLOT_TO_ENTITY.get(slot_type)
    if not label: return None

    for cand in candidates:
        # Nếu là regex pattern (chứa ký tự đặc biệt regex)
        try:
            # Chuẩn hóa candidate (trừ khi nó là regex pattern phức tạp)
            if '\\' not in cand and '(' not in cand: 
                cand = norm_text(cand)
            
            # Tìm kiếm
            match = re.search(cand, text_norm)
            if match:
                return [match.start(), match.end(), label]
        except re.error:
            continue

    return None

def process_tsv_file(file_path, gaz_map):
    """Đọc file TSV và chuyển đổi sang format NER"""
    samples = []
    print(f"Đang xử lý: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line: continue
            
            # Tách cột theo tab
            parts = line.split('\t')
            if len(parts) < 3: 
                # Thử tách bằng nhiều khoảng trắng nếu tab không hoạt động
                # (đề phòng copy paste bị lỗi format)
                # parts = re.split(r'\s{2,}', line) 
                if len(parts) < 3: continue

            raw_text = parts[0]
            # intent = parts[1] # Không cần
            json_str = parts[2]

            try:
                slots = json.loads(json_str)
            except json.JSONDecodeError:
                print(f"[WARN] Lỗi JSON dòng {line_idx+1}: {json_str}")
                continue

            # Chuẩn hóa text đầu vào (QUAN TRỌNG: Phải khớp với lúc training/inference)
            clean_text = norm_text(raw_text)
            
            entities = []
            
            # Duyệt qua từng slot đã gán nhãn trong Intent
            for slot_key, slot_val in slots.items():
                if slot_key not in SLOT_TO_ENTITY:
                    continue
                
                # Tìm vị trí thực thể trong câu
                ent = find_entity_in_text(clean_text, slot_val, slot_key, gaz_map)
                
                if ent:
                    # Kiểm tra trùng lặp (overlap) với các entity đã tìm thấy
                    is_overlap = False
                    for existing in entities:
                        # Logic: (Start1 < End2) and (Start2 < End1) là có overlap
                        if ent[0] < existing[1] and existing[0] < ent[1]:
                            is_overlap = True
                            break
                    
                    if not is_overlap:
                        entities.append(ent)
            
            # Chỉ thêm mẫu nào có ít nhất 1 entity
            if entities:
                samples.append({
                    "text": clean_text,
                    "entities": entities
                })

    return samples

def main():
    # Đường dẫn
    data_dir = "serverAI/data/nlu"
    train_path = os.path.join(data_dir, "train1.tsv")
    valid_path = os.path.join(data_dir, "valid1.tsv")
    gaz_dir = os.path.join(data_dir, "gazetteer")
    output_path = os.path.join(data_dir, "ner_train.json")

    # 1. Load Gazetteer
    gaz_map = load_gazetteer(gaz_dir)
    print(f"Đã load Gazetteer: {list(gaz_map.keys())}")

    # 2. Xử lý các file TSV
    all_data = []
    if os.path.exists(train_path):
        all_data.extend(process_tsv_file(train_path, gaz_map))
    if os.path.exists(valid_path):
        all_data.extend(process_tsv_file(valid_path, gaz_map))

    # 3. Loại bỏ dữ liệu trùng lặp (Deduplication)
    unique_data = {}
    for item in all_data:
        unique_data[item['text']] = item['entities']
    
    final_output = [{"text": k, "entities": v} for k, v in unique_data.items()]

    # 4. Lưu ra file JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Hoàn tất! Đã tạo file {output_path}")
    print(f"   Tổng số mẫu NER: {len(final_output)}")
    
    # In thử mẫu đầu tiên
    if final_output:
        print("\nVí dụ mẫu đầu tiên:")
        print(json.dumps(final_output[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()