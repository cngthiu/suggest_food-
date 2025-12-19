# import json
# import random
# import os
# import re
# from sklearn.model_selection import train_test_split

# # 1. CẤU HÌNH ĐƯỜNG DẪN (Sửa theo môi trường của bạn)
# DATA_DIR = "serverAI/data/nlu"
# GAZETTEER_DIR = os.path.join(DATA_DIR, "gazetteer")
# NUM_SAMPLES = 2500  # Tổng số mẫu sinh ra
# SEED = 42

# # 2. HÀM LOAD GAZETTEER TỪ FILE
# def load_gazetteer(file_name):
#     path = os.path.join(GAZETTEER_DIR, file_name)
#     if not os.path.exists(path):
#         return []
#     with open(path, 'r', encoding='utf-8') as f:
#         content = f.read()
#         # Loại bỏ các tag và tách từ dựa trên dấu | hoặc xuống dòng
#         lines = re.sub(r'\\', '', content).split('\n')
#         words = []
#         for line in lines:
#             parts = line.split('|')
#             words.extend([p.strip().lower() for p in parts if p.strip()])
#         return list(set(words))

# # 3. KHO DỮ LIỆU PHONG PHÚ
# DATA_POOL = {
#     "food": load_gazetteer("protein.txt"),
#     "diet": load_gazetteer("diet.txt"),
#     "device": load_gazetteer("device.txt"),
#     "time": ["15 phút", "30p", "1 tiếng", "45 phút", "nhanh", "cấp tốc", "20 phút", "1h", "tầm 30 phút"],
#     "price": ["50k", "100.000đ", "giá rẻ", "200 nghìn", "vừa túi tiền", "tầm 70k", "150 ngàn"],
#     "quantity": ["2 người", "4 phần ăn", "cả nhà", "3 thành viên", "1 suất", "gia đình 4 người", "cho 2 bé"]
# }

# # 4. TEMPLATES CHO 6 INTENTS
# TEMPLATES = [
#     # search_recipe
#     {"intent": "search_recipe", "tmpl": "tìm cách nấu {food} {diet}"},
#     {"intent": "search_recipe", "tmpl": "gợi ý món {food} làm trong {time}"},
#     {"intent": "search_recipe", "tmpl": "nấu món gì từ {food} cho {quantity}"},
#     {"intent": "search_recipe", "tmpl": "muốn ăn {food} {diet} khoảng {price}"},
    
#     # ask_recipe_detail
#     {"intent": "ask_recipe_detail", "tmpl": "hướng dẫn làm món {food}"},
#     {"intent": "ask_recipe_detail", "tmpl": "công thức nấu {food} chi tiết"},
#     {"intent": "ask_recipe_detail", "tmpl": "cách chế biến {food} như thế nào"},
    
#     # refine_search
#     {"intent": "refine_search", "tmpl": "nhưng mình muốn dùng {device}"},
#     {"intent": "refine_search", "tmpl": "tìm lại món {food} cho {quantity}"},
#     {"intent": "refine_search", "tmpl": "thêm điều kiện là {diet}"},
    
#     # add_ingredients_to_cart
#     {"intent": "add_ingredients_to_cart", "tmpl": "mua nguyên liệu nấu {food}"},
#     {"intent": "add_ingredients_to_cart", "tmpl": "cho thực phẩm làm {food} vào giỏ"},
#     {"intent": "add_ingredients_to_cart", "tmpl": "đặt hàng nguyên liệu cho món {food}"},
    
#     # ask_price_estimate
#     {"intent": "ask_price_estimate", "tmpl": "nấu {food} cho {quantity} hết bao nhiêu"},
#     {"intent": "ask_price_estimate", "tmpl": "chi phí làm {food} khoảng {price} đúng không"},
#     {"intent": "ask_price_estimate", "tmpl": "giá nguyên liệu món {food} hiện nay"},
    
#     # fallback
#     {"intent": "fallback", "tmpl": "xin chào"},
#     {"intent": "fallback", "tmpl": "bạn có thể làm gì"},
#     {"intent": "fallback", "tmpl": "thời tiết hôm nay thế nào"},
# ]

# LABEL_MAP = {
#     "food": "FOOD", "diet": "DIET", "time": "TIME", 
#     "price": "PRICE", "quantity": "QUANTITY", "device": "DEVICE"
# }

# def generate_samples(num_samples):
#     samples = []
#     seen_texts = set()
    
#     while len(samples) < num_samples:
#         t_obj = random.choice(TEMPLATES)
#         tmpl = t_obj["tmpl"]
#         intent = t_obj["intent"]
        
#         placeholders = re.findall(r"\{(.*?)\}", tmpl)
#         text = tmpl
#         entities = []
        
#         # Sắp xếp placeholders để thay thế không làm lệch index của các placeholder sau
#         # Tuy nhiên ở đây dùng replace 1 lần duy nhất cho mỗi placeholder là an toàn
#         for p in placeholders:
#             val = random.choice(DATA_POOL[p])
#             start_idx = text.find("{" + p + "}")
#             text = text.replace("{" + p + "}", val, 1)
#             end_idx = start_idx + len(val)
#             entities.append([start_idx, end_idx, LABEL_MAP[p]])
            
#         if text not in seen_texts:
#             samples.append({
#                 "text": text,
#                 "intent": intent, # Lưu lại intent để split stratified
#                 "entities": entities
#             })
#             seen_texts.add(text)
#     return samples

# def main():
#     print("🚀 Bắt đầu sinh dữ liệu NER...")
#     all_data = generate_samples(NUM_SAMPLES)
    
#     # Chia tập Train/Valid 80/20 có phân lớp (Stratified) theo Intent
#     intents_labels = [s["intent"] for s in all_data]
#     train_data, valid_data = train_test_split(
#         all_data, 
#         test_size=0.2, 
#         random_state=SEED, 
#         stratify=intents_labels
#     )
    
#     # Loại bỏ trường 'intent' trong file JSON cuối cùng (vì NER chỉ cần text và entities)
#     for s in train_data: s.pop("intent")
#     for s in valid_data: s.pop("intent")

#     # Lưu file
#     os.makedirs(DATA_DIR, exist_ok=True)
#     with open(os.path.join(DATA_DIR, 'ner_train.json'), 'w', encoding='utf-8') as f:
#         json.dump(train_data, f, ensure_ascii=False, indent=2)
    
#     with open(os.path.join(DATA_DIR, 'ner_valid.json'), 'w', encoding='utf-8') as f:
#         json.dump(valid_data, f, ensure_ascii=False, indent=2)

#     print(f"✅ Hoàn thành!")
#     print(f" - Tổng: {len(all_data)} mẫu")
#     print(f" - Train: {len(train_data)} mẫu tại {DATA_DIR}/ner_train.json")
#     print(f" - Valid: {len(valid_data)} mẫu tại {DATA_DIR}/ner_valid.json")

# if __name__ == "__main__":
#     main()
import json
import random
import os
import re
from sklearn.model_selection import train_test_split
from unidecode import unidecode

# 1. CẤU HÌNH
DATA_DIR = "serverAI/data/nlu"
GAZETTEER_DIR = os.path.join(DATA_DIR, "gazetteer")
RECIPE_FILE = "/media/congthieu/ubuntu_data/LTTM/MM/serverAI/data/recipes/recipies.json"
TOTAL_SAMPLES = 2000 
SEED = 42

# 2. LOAD DATA (Giữ nguyên logic cũ)
def load_gazetteer(file_name):
    path = os.path.join(GAZETTEER_DIR, file_name)
    if not os.path.exists(path): return []
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = re.sub(r'\\', '', content).split('\n')
        words = []
        for line in lines:
            parts = line.split('|')
            words.extend([p.strip().lower() for p in parts if p.strip()])
        return list(set(words))

def extract_from_recipes(file_path):
    if not os.path.exists(file_path): return [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        recipes = json.load(f)
    food_items = []
    diets = []
    for r in recipes:
        food_items.append(r['title'].lower())
        if 'search_keywords' in r: food_items.extend([kw.lower() for kw in r['search_keywords']])
        if 'diet' in r: diets.extend([d.lower() for d in r['diet']])
        for ing in r.get('ingredients', []): food_items.append(ing['name'].lower())
    return list(set(food_items)), list(set(diets))

# 3. KHO DỮ LIỆU TỔNG HỢP
raw_food, raw_diet = extract_from_recipes(RECIPE_FILE)
DATA_POOL = {
    "food": list(set(load_gazetteer("protein.txt") + raw_food)),
    "diet": list(set(load_gazetteer("diet.txt") + raw_diet)),
    "device": load_gazetteer("device.txt"),
    "time": ["10 phút", "15p", "30 phút", "1 tiếng", "45 phút", "20 phút", "nhanh", "cấp tốc", "siêu tốc"],
    "price": ["50k", "100.000đ", "giá rẻ", "200 nghìn", "vừa túi tiền", "tầm 80k", "sinh viên", "giá bình dân"],
    "quantity": ["2 người", "4 phần ăn", "cả nhà", "3 thành viên", "1 suất", "gia đình 4 người", "cho bé", "cho 2 người"]
}

LABEL_MAP = {
    "food": "FOOD", "diet": "DIET", "time": "TIME", 
    "price": "PRICE", "quantity": "QUANTITY", "device": "DEVICE"
}

# 4. TEMPLATES PHÂN LOẠI THEO NHÃN
# Chúng ta liệt kê các template và đánh dấu các nhãn mà nó chứa
TEMPLATES = [
    {"intent": "search_recipe", "tmpl": "tìm cách nấu {food} {diet}", "labels": ["food", "diet"]},
    {"intent": "search_recipe", "tmpl": "gợi ý món {food} làm trong {time}", "labels": ["food", "time"]},
    {"intent": "search_recipe", "tmpl": "muốn ăn {food} {diet} tầm {price}", "labels": ["food", "diet", "price"]},
    {"intent": "ask_recipe_detail", "tmpl": "hướng dẫn làm món {food}", "labels": ["food"]},
    {"intent": "refine_search", "tmpl": "nhưng mình muốn dùng {device}", "labels": ["device"]},
    {"intent": "refine_search", "tmpl": "tìm lại món {food} cho {quantity}", "labels": ["food", "quantity"]},
    {"intent": "add_ingredients_to_cart", "tmpl": "mua nguyên liệu nấu {food}", "labels": ["food"]},
    {"intent": "ask_price_estimate", "tmpl": "nấu {food} cho {quantity} hết bao nhiêu", "labels": ["food", "quantity"]},
    {"intent": "ask_price_estimate", "tmpl": "chi phí làm {food} khoảng {price}", "labels": ["food", "price"]},
    {"intent": "search_recipe", "tmpl": "có món {food} nào làm bằng {device} mất {time} không", "labels": ["food", "device", "time"]},
    {"intent": "search_recipe", "tmpl": "thực đơn {diet} cho {quantity} giá {price}", "labels": ["diet", "quantity", "price"]}
]

# 5. LOGIC SINH DỮ LIỆU CÂN BẰNG
def generate_balanced_samples(total_samples):
    samples = []
    seen_texts = set()
    # Khởi tạo bộ đếm nhãn
    label_counts = {l: 0 for l in LABEL_MAP.values()}
    
    print("🔄 Đang sinh dữ liệu cân bằng...")
    
    while len(samples) < total_samples:
        # Tìm nhãn đang có ít mẫu nhất
        min_label_key = min(label_counts, key=label_counts.get)
        
        # Lọc các template có chứa nhãn đang thiếu này
        suitable_templates = [t for t in TEMPLATES if any(LABEL_MAP[l] == min_label_key for l in t["labels"])]
        
        # Nếu không có template nào chứa nhãn đó (lỗi logic), chọn ngẫu nhiên
        if not suitable_templates:
            t_obj = random.choice(TEMPLATES)
        else:
            t_obj = random.choice(suitable_templates)
            
        text = t_obj["tmpl"]
        entities = []
        placeholders = re.findall(r"\{(.*?)\}", text)
        
        for p in placeholders:
            val = random.choice(DATA_POOL[p])
            start = text.find("{" + p + "}")
            text = text.replace("{" + p + "}", val, 1)
            label_name = LABEL_MAP[p]
            entities.append([start, start + len(val), label_name])
            label_counts[label_name] += 1 # Cập nhật bộ đếm khi sinh ra nhãn
            
        # Augmentation (25% không dấu)
        if random.random() < 0.25:
            text = unidecode(text)
            
        if text not in seen_texts:
            samples.append({"text": text, "intent": t_obj["intent"], "entities": entities})
            seen_texts.add(text)
            
    print("📊 Thống kê nhãn sau khi sinh:")
    for l, c in label_counts.items():
        print(f" - {l}: {c}")
    return samples

# 6. THỰC THI
def main():
    random.seed(SEED)
    all_data = generate_balanced_samples(TOTAL_SAMPLES)
    
    train_data, valid_data = train_test_split(
        all_data, test_size=0.15, random_state=SEED, 
        stratify=[s["intent"] for s in all_data]
    )
    
    for s in train_data + valid_data: s.pop("intent")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, 'ner_train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(os.path.join(DATA_DIR, 'ner_valid.json'), 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Đã lưu dữ liệu cân bằng vào ner_train.json và ner_valid.json")

if __name__ == "__main__":
    main()