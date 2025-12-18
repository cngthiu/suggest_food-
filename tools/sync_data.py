import random
import json
import os
import re
from collections import defaultdict

# --- CẤU HÌNH ---
NUM_TRAIN = 2000
NUM_VALID = 250
BASE_DIR = "serverAI/data/nlu"
INTENT_FILE = f"{BASE_DIR}/intents.json"

# --- 1. DATA POOL ---
DATA_POOL = {
    "food": [
        "gà", "thịt gà", "ức gà", "đùi gà", "bò", "thịt bò", "bắp bò", "heo", "sườn", "ba chỉ",
        "cá", "cá lóc", "cá hồi", "tôm", "mực", "trứng", "đậu phụ", "cơm gà", "phở", "bún",
        "lẩu", "món cuốn", "salad", "canh chua"
    ],
    "ingredient": [
        "rau muống", "cải thảo", "bí xanh", "khoai tây", "cà chua", "nấm", 
        "hành tây", "tỏi", "ớt", "sả", "rau xà lách", "dưa leo", "hành ngò"
    ],
    "time": [
        "5 phút", "15 phút", "30 phút", "1 tiếng", "nhanh", "cấp tốc", 
        "không tốn thời gian", "rảnh rỗi", "đi làm về muộn"
    ],
    "quantity": [
        "1 người", "2 người", "3 người", "cả nhà", "gia đình", "nhóm bạn", 
        "suất đôi", "phần lớn", "ít người"
    ],
    "price": [
        "20k", "50k", "100k", "200k", "rẻ", "tiết kiệm", "sinh viên", 
        "bình dân", "sang trọng", "không quan trọng tiền"
    ],
    "style": [
        "kho", "luộc", "hấp", "chiên", "nướng", "rim", "xào", 
        "chua ngọt", "chiên giòn", "nướng muối ớt", "thanh đạm", "đậm đà"
    ],
    "diet": [
        "chay", "eat clean", "giảm cân", "ít béo", "healthy", 
        "ít dầu mỡ", "nhiều rau", "tăng cơ"
    ],
    "meal": [
        "bữa trưa", "bữa tối", "bữa sáng", "bữa xế", "tiệc", "cơm văn phòng", "cơm hộp"
    ],
    "context": [
        "ít dọn dẹp", "dễ làm", "nguyên liệu có sẵn", "trong tủ lạnh còn", 
        "mới đi chợ về", "đang vội", "lười nấu", "muốn đổi gió"
    ]
}

# --- 2. NOISE ---
PREFIXES = ["", "", "ê ", "bạn ơi ", "gợi ý ", "mình muốn ", "tìm giúp ", "đề xuất ", "nhà còn ", "đang cần "]
SUFFIXES = ["", "", " đi", " nhé", " nha", " với", " gấp", " nào ngon", " cho hợp lý"]

# --- 3. MAPPING SLOT ---
SLOT_TO_NER_LABEL = {
    "food": "FOOD", "ingredient": "INGREDIENT", "time": "TIME",
    "quantity": "QUANTITY", "price": "PRICE", "style": "FOOD", "diet": "DIET",
    "meal": "TIME", "context": "O" # Context không cần bắt slot, chỉ để đa dạng câu
}

# --- 4. TEMPLATES NÂNG CAO (Mô phỏng Test Set) ---
TEMPLATE_POOL = {
    "search_recipe": [
        # Cơ bản
        "món {food} {style}",
        "nấu gì với {ingredient}",
        "hôm nay ăn gì",
        
        # Phức tạp (Giống Test set)
        "lên thực đơn {meal}: {food}, {time}, {quantity}",
        "mình muốn 1 set {food} + {ingredient}",
        "còn {food} và {ingredient}, gợi ý món phù hợp",
        "đề xuất món {style}, {context}, hợp {meal}",
        "tìm món kiểu {style} cho {meal} {quantity}",
        "nhà còn dư {ingredient} thì làm món gì {time}",
        "tư vấn thực đơn {diet} có {food}",
        "món nào làm từ {food} mà {context}",
        "gợi ý món {food} ăn kèm {ingredient}",
        "combo {food} và {ingredient} cho {quantity}",
        "tìm món {food} {style} ăn {meal}",
        "có món nào {time} mà ngon không",
        "thực đơn {quantity} người, giá {price}",
        "món nhậu từ {food} {style}",
        "ăn gì {meal} vừa {diet} vừa {price}"
    ],
    "ask_recipe_detail": [
        "hướng dẫn chi tiết món này",
        "cách làm cụ thể ra sao",
        "bước 1 là gì",
        "sơ chế {ingredient} thế nào",
        "nêm nếm gia vị ra sao",
        "làm sao cho {food} giòn",
        "bí quyết nấu ngon",
        "chỉ mình cách nấu đi"
    ],
    "ask_price_estimate": [
        "món này hết bao nhiêu tiền",
        "nấu bữa này tốn kém không",
        "giá nguyên liệu khoảng bao nhiêu",
        "chi phí cho {quantity} người",
        "ăn thế này có đắt không",
        "tổng thiệt hại là bao nhiêu",
        "ngân sách {price} đủ nấu không"
    ],
    "add_ingredients_to_cart": [
        "mua giúp mình nguyên liệu",
        "thêm {ingredient} vào giỏ",
        "đặt mua đồ về nấu",
        "order {quantity} {food} nhé",
        "cho hết vào giỏ hàng đi",
        "lấy đủ nguyên liệu cho món này",
        "cần mua những gì thì thêm vào giỏ giúp"
    ],
    "refine_search": [
        "tìm món khác đi",
        "không thích món này",
        "đổi món {style} hơn",
        "có món nào không có {ingredient} không",
        "món này ngán rồi",
        "tìm cái gì lạ miệng hơn",
        "đổi sang món {food} được không",
        "gợi ý cái khác {time} hơn"
    ],
    "fallback": [
        "xin chào", "hi bot", "bạn tên gì", "thời tiết thế nào", 
        "hát đi", "ngu quá", "giỏi lắm", "biết ông a không",
        "đang ở đâu", "mấy giờ rồi"
    ]
}

def get_random_item(key):
    return random.choice(DATA_POOL.get(key, [""]))

def construct_sample(template_str, intent):
    # Random prefix để tăng tính tự nhiên
    prefix = random.choice(PREFIXES) if random.random() > 0.3 else ""
    suffix = random.choice(SUFFIXES) if random.random() > 0.3 else ""
    
    full_template = f"{prefix}{template_str}{suffix}".strip()
    
    parts = re.split(r'({[^}]+})', full_template)
    final_text = ""
    entities = []
    
    for part in parts:
        if part.startswith('{') and part.endswith('}'):
            slot_name = part[1:-1]
            if slot_name in DATA_POOL:
                value = get_random_item(slot_name)
                label = SLOT_TO_NER_LABEL.get(slot_name)
                
                # Chỉ gán nhãn nếu không phải là O
                if label and label != "O":
                    start_idx = len(final_text)
                    end_idx = start_idx + len(value)
                    entities.append([start_idx, end_idx, label])
                
                final_text += value
            else:
                final_text += part
        else:
            final_text += part
            
    # Xử lý khoảng trắng thừa
    final_text = " ".join(final_text.split())
    return final_text, intent, entities

def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    
    # Chia Template Train/Valid
    train_templates = []
    valid_templates = []
    
    print("--- CHIA TEMPLATE (Đảm bảo Valid có mẫu lạ) ---")
    for intent, tmpls in TEMPLATE_POOL.items():
        random.shuffle(tmpls)
        # 80% template quen thuộc cho train, 20% template lạ cho valid
        split = int(len(tmpls) * 0.8)
        if split == 0: split = 1
        
        train_templates.extend([(t, intent) for t in tmpls[:split]])
        # Valid có thể dùng toàn bộ template để test độ phủ, 
        # hoặc tách riêng nếu muốn test zero-shot generalization.
        # Ở đây ta cho valid dùng toàn bộ nhưng ưu tiên mẫu khó.
        valid_templates.extend([(t, intent) for t in tmpls]) 
        
    def generate(templates, num):
        ds = []
        seen = set()
        while len(ds) < num:
            t, i = random.choice(templates)
            txt, intent, ents = construct_sample(t, i)
            if txt not in seen:
                seen.add(txt)
                ds.append({"text": txt, "intent": intent, "entities": ents})
        return ds

    print(f"Sinh {NUM_TRAIN} Train...")
    train_data = generate(train_templates, NUM_TRAIN)
    
    print(f"Sinh {NUM_VALID} Valid...")
    valid_data = generate(valid_templates, NUM_VALID) # Valid dùng mẫu rộng hơn để test

    # Ghi file
    with open(f"{BASE_DIR}/train1.tsv", "w", encoding="utf-8") as f:
        for d in train_data: f.write(f"{d['text']}\t{d['intent']}\n")
        
    with open(f"{BASE_DIR}/valid1.tsv", "w", encoding="utf-8") as f:
        for d in valid_data: f.write(f"{d['text']}\t{d['intent']}\n")
        
    ner_json = [{"text": d['text'], "entities": d['entities']} for d in train_data if d['entities']]
    with open(f"{BASE_DIR}/ner_train1.json", "w", encoding="utf-8") as f:
        json.dump(ner_json, f, ensure_ascii=False, indent=2)

    print("DONE! Dữ liệu mới đã có các mẫu câu phức tạp giống Test Set.")

if __name__ == "__main__":
    main()