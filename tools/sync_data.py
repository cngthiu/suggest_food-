import random
import json
import os
import re

# --- CẤU HÌNH ---
NUM_TRAIN = 3000  # Số mẫu tập train
NUM_VALID = 300   # Số mẫu tập valid
BASE_DIR = "serverAI/data/nlu"
INTENT_FILE = f"{BASE_DIR}/intents.json"

# --- KHO DỮ LIỆU PHONG PHÚ (Rich Data Pool) ---
DATA_POOL = {
    "protein": [
        "gà", "thịt gà", "cánh gà", "ức gà", "đùi gà", "cánh gà chiên nước mắm",
        "gà rang gừng", "gà kho gừng", "gà luộc", "gà nướng", "gà xào sả ớt",
        "bò", "thịt bò", "bắp bò", "ba chỉ bò", "gân bò", "bò xào hành tây",
        "bò lúc lắc", "bò nấu lagu", "bò hầm rau củ",
        "heo", "thịt heo", "sườn", "sườn non", "ba rọi", "ba rọi quay",
        "thịt xay", "chân giò", "thịt kho tàu", "thịt kho trứng", "thịt rang cháy cạnh",
        "cá", "cá diêu hồng", "cá basa", "cá lóc", "cá hồi", "cá rô phi",
        "cá kho tộ", "cá chiên giòn", "cá hấp xì dầu", "canh chua cá",
        "tôm", "tôm sú", "tôm thẻ", "tôm hùm", "tôm rim mặn ngọt", "tôm nướng",
        "mực", "mực ống", "mực hấp", "mực xào chua ngọt", "mực chiên giòn",
        "bạch tuộc",
        "trứng", "trứng gà", "trứng vịt", "trứng cút", "trứng chiên", "trứng hấp",
        "trứng kho thịt",
        "cơm gà", "cơm sườn", "cơm tấm", "cơm thịt kho",
        "phở bò", "phở gà",
        "bún bò", "bún riêu", "bún chả",
        "miến gà", "miến tôm",
        "cháo gà", "cháo tôm", "cháo thịt bằm",
        "lẩu gà", "lẩu hải sản", "lẩu bò nhúng dấm"
    ],

    "vege": [
        "rau muống", "cải thảo", "bắp cải", "bắp cải thịt", "bí xanh", "bí đỏ",
        "khoai tây", "khoai lang", "cà rốt", "cà chua", "đậu bắp", "nấm rơm", "nấm hương",
        "măng tây", "súp lơ", "rau cải", "rau muống xào tỏi", "rau muống luộc",
        "canh bí xanh nấu tôm", "canh rau cải thịt bằm", "canh rau dền", "canh cua rau đay",
        "mướp đắng", "mướp đắng xào trứng"
    ],

    "time": [
        "5 phút", "10 phút", "15 phút", "20 phút", "25 phút", "30 phút",
        "35 phút", "40 phút", "45 phút", "50 phút", "60 phút",
        "5p", "10p", "15p", "20p", "25p", "30p", "45p", "60p", "90p",
        "1 tiếng", "1 tiếng rưỡi", "2 tiếng", "hơn 1 tiếng",
        "nửa tiếng", "1h", "1h30p", "2h",
        "nấu nhanh", "siêu tốc", "cấp tốc", "ăn liền", "làm nhanh",
        "không mất nhiều thời gian", "làm trong giờ nghỉ trưa",
        "làm sau giờ làm", "dành cho bữa sáng vội", "chuẩn bị tối qua"
    ],

    "quantity": [
        "1 người", "2 người", "3 người", "4 người", "5 người", "6 người", "8 người", "10 người",
        "cho 1 người", "cho 2 người", "cho 3 người", "cho 4 người",
        "3 thành viên", "4 thành viên", "gia đình 3 người", "gia đình 4 người", "gia đình 5 người",
        "2 người lớn 1 trẻ em",
        "1 suất", "2 suất", "3 suất", "4 phần", "5 phần ăn", "6 phần ăn", "7 phần ăn", "10 phần ăn",
        "2 bát", "3 bát", "2 tô", "3 tô",
        "cả nhà", "2 vợ chồng", "cho bé", "cho 2 bé", "cho trẻ nhỏ",
        "đại gia đình", "cho mình tôi", "anh em trong phòng trọ"
    ],

    "price": [
        "20k", "30k", "40k", "50k", "60k", "70k", "80k",
        "100k", "120k", "150k", "180k", "200k", "250k", "300k", "400k", "500k", "800k", "1000k",
        "30 nghìn", "40 nghìn", "50 nghìn", "70 nghìn", "100 nghìn", "150 nghìn",
        "100 ngàn", "200 ngàn", "ba trăm nghìn",
        "50.000đ", "70.000đ", "100.000vnd", "150.000 đồng", "200.000 đồng",
        "300.000 đồng", "500.000 đồng", "1 triệu", "1tr", "1 triệu rưỡi", "nửa triệu",
        "rẻ", "bình dân", "tiết kiệm", "sinh viên", "cao cấp", "sang chảnh",
        "vừa túi tiền", "không quá đắt", "ăn sáng một bữa"
    ],

    "diet": [
        "chay", "keto", "eat clean", "low carb", "không dầu mỡ",
        "ít dầu mỡ", "thực dưỡng", "giảm cân", "ít calo", "nhiều rau",
        "ít tinh bột"
    ],

    "style": [
        "xào", "kho", "luộc", "hấp", "chiên", "nướng", "rim", "nấu canh", "làm gỏi",
        "xào tỏi", "xào sả ớt", "kho tiêu", "nướng muối ớt"
    ],

    "device": [
        "nồi chiên", "chảo", "nồi cơm điện", "lò vi sóng", "nồi áp suất", "máy xay",
        "nồi chiên không dầu", "bếp điện", "bếp ga"
    ]
}

# Mapping từ khóa trong DATA_POOL sang nhãn NER (Label)
# Chỉ map những nhãn mà file training/train_phobert.py hỗ trợ (FOOD, TIME, QUANTITY, PRICE)
SLOT_TO_NER_LABEL = {
    "protein": "FOOD",
    "vege": "FOOD",
    "style": "FOOD",  # Cách chế biến cũng coi là 1 phần của món ăn
    "time": "TIME",
    "quantity": "QUANTITY",
    "price": "PRICE",
    # "diet": "DIET",   # Nếu model hỗ trợ thì bật, hiện tại train_phobert.py chưa có
    # "device": "DEVICE"
}

# ========== TEMPLATE ==========
TEMPLATES = [
    # --- 1. SUGGEST_FOOD ---
    ("gợi ý món {protein} {style} cho {quantity}", 
     "suggest_food", {"protein": "protein", "style": "style", "quantity": "quantity"}),

    ("tôi muốn nấu món {protein} trong {time} giá {price}", 
     "suggest_food", {"protein": "protein", "time": "time", "price": "price"}),

    ("nấu {quantity} ăn món {protein} hết bao nhiêu tiền", 
     "suggest_food", {"protein": "protein", "quantity": "quantity"}),

    ("tìm món {protein} {time} khoảng {price}", 
     "suggest_food", {"protein": "protein", "time": "time", "price": "price"}),

    ("món {protein} nào nấu bằng {device} mất {time}", 
     "suggest_food", {"protein": "protein", "device": "device", "time": "time"}),

    ("hôm nay nên nấu gì từ {protein} cho {quantity}", 
     "suggest_food", {"protein": "protein", "quantity": "quantity"}),
     
    ("món ngon rẻ tầm {price}", 
     "suggest_food", {"price": "price"}),

    ("cơm sinh viên giá {price}", 
     "suggest_food", {"price": "price"}),

    # --- 2. SUGGEST_FOOD_BY_INGREDIENT ---
    ("hôm nay ăn {protein} với {vege} cho {quantity}", 
     "suggest_food_by_ingredient", {"protein": "protein", "vege": "vege", "quantity": "quantity"}),

    ("đổi gió ăn {protein} {style} kèm {vege} được không", 
     "suggest_food_by_ingredient", {"protein": "protein", "style": "style", "vege": "vege"}),

    ("nếu có {protein} và {vege} thì nấu món gì ngon", 
     "suggest_food_by_ingredient", {"protein": "protein", "vege": "vege"}),
     
    ("nhà còn dư {vege} thì nấu gì", 
     "suggest_food_by_ingredient", {"vege": "vege"}),

    # --- 3. SUGGEST_FOOD_BY_TIME ---
    ("món nào nấu nhanh dưới {time}", 
     "suggest_food_by_time", {"time": "time"}),

    ("tôi chỉ có {time} để nấu ăn thôi", 
     "suggest_food_by_time", {"time": "time"}),

    ("có món nào làm {time} xong không", 
     "suggest_food_by_time", {"time": "time"}),

    ("ăn gì trong {time} cho kịp giờ đi làm", 
     "suggest_food_by_time", {"time": "time"}),

    ("bữa tối muốn nấu gì trong vòng {time}", 
     "suggest_food_by_time", {"time": "time"}),

    # --- 4. ASK_PRICE ---
    ("món {protein} này giá bao nhiêu", 
     "ask_price", {"protein": "protein"}),
     
    ("chi phí để làm món {protein} {style}",
     "ask_price", {"protein": "protein", "style": "style"}),

    # --- 5. FILTER_DIET ---
    ("tư vấn thực đơn {diet} cho {quantity}", 
     "filter_diet", {"diet": "diet", "quantity": "quantity"}),

    ("món {protein} cho người ăn {diet}", 
     "filter_diet", {"protein": "protein", "diet": "diet"}),

    ("ăn kiểu {diet} thì nên ăn gì từ {protein}", 
     "filter_diet", {"diet": "diet", "protein": "protein"}),

    ("đổi món {diet} trong ngày mà vẫn rẻ như {price}", 
     "filter_diet", {"diet": "diet", "price": "price"}),
     
    ("món nào từ {protein} vừa {diet} vừa phù hợp {quantity}", 
     "filter_diet", {"protein": "protein", "diet": "diet", "quantity": "quantity"}),
     
    ("an {protein} {diet} trong {time} với ngân sách {price}",
     "suggest_food", {"protein": "protein", "diet": "diet", "time": "time", "price": "price"}),
     
    ("chỉ có {price} thì ăn món gì",
     "suggest_food", {"price": "price"}),
     
    ("ăn gì dưới {price}",
     "suggest_food", {"price": "price"})
]

def load_valid_intents():
    if os.path.exists(INTENT_FILE):
        with open(INTENT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return set(data.get("intents", []))
    else:
        print(f"CANH BAO: Khong tim thay {INTENT_FILE}. Su dung mac dinh.")
        return set()

def construct_sample(template, slot_map):
    """
    Hàm này thay thế các slot trong template bằng dữ liệu ngẫu nhiên,
    đồng thời tính toán vị trí (start, end) của các thực thể NER.
    """
    
    # Tách template thành các phần dựa trên dấu ngoặc nhọn {}
    # Ví dụ: "gợi ý món {protein} giá {price}" -> ['gợi ý món ', '{protein}', ' giá ', '{price}', '']
    parts = re.split(r'({[^}]+})', template)
    
    final_text = ""
    entities = []
    
    for part in parts:
        if part.startswith('{') and part.endswith('}'):
            # Đây là slot, ví dụ: "{protein}"
            slot_name = part[1:-1] # protein
            data_key = slot_map.get(slot_name) # Lấy key trong DATA_POOL
            
            if data_key and data_key in DATA_POOL:
                value = random.choice(DATA_POOL[data_key])
                
                # Kiểm tra xem slot này có nhãn NER tương ứng không
                label = SLOT_TO_NER_LABEL.get(data_key)
                
                if label:
                    start_idx = len(final_text)
                    end_idx = start_idx + len(value)
                    # Thêm vào danh sách entity: [start, end, label]
                    entities.append([start_idx, end_idx, label])
                
                final_text += value
            else:
                # Nếu không tìm thấy data (trường hợp hiếm), giữ nguyên
                final_text += part
        else:
            # Đây là text tĩnh
            final_text += part
            
    return final_text, entities

def generate_dataset(num_samples, valid_intents):
    tsv_rows = []
    ner_data = []
    used_intents = set()

    for _ in range(num_samples):
        tmpl, intent, slot_map = random.choice(TEMPLATES)
        
        # Chỉ check intent nếu file config tồn tại
        if valid_intents and intent not in valid_intents:
            continue
            
        used_intents.add(intent)
        
        # Sinh text và entity
        text, entities = construct_sample(tmpl, slot_map)
        
        # 1. Dữ liệu cho Intent Classification (TSV)
        tsv_rows.append(f"{text}\t{intent}")
        
        # 2. Dữ liệu cho NER (JSON)
        # Chỉ lưu nếu câu có entity
        if entities:
            ner_data.append({
                "text": text,
                "entities": entities
            })
            
    return tsv_rows, ner_data, used_intents

def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    
    # 1. Load intents
    valid_intents = load_valid_intents()
    print(f"Intents hop le: {valid_intents}")
    
    # 2. Sinh Train Data
    print(f"\nDang sinh {NUM_TRAIN} mau train...")
    train_tsv, train_ner, train_intents = generate_dataset(NUM_TRAIN, valid_intents)
    
    # Ghi file train.tsv
    with open(f"{BASE_DIR}/train.tsv", "w", encoding="utf-8") as f:
        f.write("\n".join(train_tsv))
        
    # Ghi file ner_train.json
    with open(f"{BASE_DIR}/ner_train.json", "w", encoding="utf-8") as f:
        json.dump(train_ner, f, ensure_ascii=False, indent=2)
        
    print(f"-> Da tao train.tsv ({len(train_tsv)} mau)")
    print(f"-> Da tao ner_train.json ({len(train_ner)} mau co thuc the)")
    print(f"-> Intents used: {train_intents}")

    # 3. Sinh Valid Data (Chỉ cần tsv cho intent, valid cho NER thường tách từ train lúc training)
    print(f"\nDang sinh {NUM_VALID} mau valid...")
    valid_tsv, _, _ = generate_dataset(NUM_VALID, valid_intents)
    
    with open(f"{BASE_DIR}/valid.tsv", "w", encoding="utf-8") as f:
        f.write("\n".join(valid_tsv))
    print(f"-> Da tao valid.tsv ({len(valid_tsv)} mau)")
    
    print("\nHOAN TAT! Da san sang huan luyen Intent & NER.")

if __name__ == "__main__":
    main()