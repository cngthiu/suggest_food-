import random
import json
import os

# --- CẤU HÌNH ---
NUM_TRAIN = 3000  # Tăng lên 3000 mẫu để phủ hết các trường hợp
NUM_VALID = 300

# --- KHO DỮ LIỆU PHONG PHÚ (Rich Data Pool) ---
DATA_POOL = {
    "protein": [
        # ga
        "ga", "thit ga", "canh ga", "uc ga", "dui ga", "canh ga chien nuoc mam",
        "ga rang gung", "ga kho gung", "ga luoc", "ga nuong", "ga xao sa ot",
        # bo
        "bo", "thit bo", "bap bo", "ba chi bo", "gan bo", "bo xao hanh tay",
        "bo luc lac", "bo nau lagu", "bo ham rau cu",
        # heo
        "heo", "thit heo", "suon", "suon non", "ba roi", "ba roi quay",
        "thit xay", "chan gio", "thit kho tau", "thit kho trung", "thit rang chay canh",
        # ca
        "ca", "ca dieu hong", "ca basa", "ca loc", "ca hoi", "ca ro phi",
        "ca kho to", "ca chien gion", "ca hap xi dau", "canh chua ca",
        # hai san
        "tom", "tom su", "tom the", "tom hum", "tom rim man ngot", "tom nuong",
        "muc", "muc ong", "muc hap", "muc xao chua ngot", "muc chien gion",
        "bach tuoc",
        # trung
        "trung", "trung ga", "trung vit", "trung cut", "trung chien", "trung hap",
        "trung kho thit",
        # mon mot dia / com bun pho
        "com ga", "com suon", "com tam", "com thit kho",
        "pho bo", "pho ga",
        "bun bo", "bun rieu", "bun cha",
        "mien ga", "mien tom",
        "chao ga", "chao tom", "chao thit bam",
        # lau
        "lau ga", "lau hai san", "lau bo nhung dam"
    ],

    "vege": [
        "rau muong", "cai thao", "bap cai", "bap cai thit", "bi xanh", "bi do",
        "khoai tay", "khoai lang", "ca rot", "ca chua", "dau bap", "nam rom", "nam huong",
        "mang tay", "sup lo", "rau cai", "rau muong xao toi", "rau muong luoc",
        "canh bi xanh nau tom", "canh rau cai thit bam", "canh rau den", "canh cua rau day",
        "muop dang", "muop dang xao trung"
    ],

    "time": [
        # phut
        "5 phut", "10 phut", "15 phut", "20 phut", "25 phut", "30 phut",
        "35 phut", "40 phut", "45 phut", "50 phut", "60 phut",
        "5p", "10p", "15p", "20p", "25p", "30p", "45p", "60p", "90p",
        # gio
        "1 tieng", "1 tieng ruoi", "2 tieng", "hon 1 tieng",
        "nua tieng", "1h", "1h30p", "2h",
        # dinh tinh
        "nau nhanh", "sieu toc", "cap toc", "an lien", "lam nhanh",
        "khong mat nhieu thoi gian", "lam trong gio nghi trua",
        "lam sau gio lam", "danh cho bua sang voi", "chuan bi toi qua"
    ],

    "quantity": [
        # so nguoi
        "1 nguoi", "2 nguoi", "3 nguoi", "4 nguoi", "5 nguoi", "6 nguoi", "8 nguoi", "10 nguoi",
        "cho 1 nguoi", "cho 2 nguoi", "cho 3 nguoi", "cho 4 nguoi",
        "3 thanh vien", "4 thanh vien", "gia dinh 3 nguoi", "gia dinh 4 nguoi", "gia dinh 5 nguoi",
        "2 nguoi lon 1 tre em",
        # suat / phan / bat / to
        "1 suat", "2 suat", "3 suat", "4 phan", "5 phan an", "6 phan an", "7 phan an", "10 phan an",
        "2 bat", "3 bat", "2 to", "3 to",
        # chung chung
        "ca nha", "2 vo chong", "cho be", "cho 2 be", "cho tre nho",
        "dai gia dinh", "cho minh toi", "anh em trong phong tro"
    ],

    "price": [
        # k / nghin
        "20k", "30k", "40k", "50k", "60k", "70k", "80k",
        "100k", "120k", "150k", "180k", "200k", "250k", "300k", "400k", "500k", "800k", "1000k",
        "30 nghin", "40 nghin", "50 nghin", "70 nghin", "100 nghin", "150 nghin",
        "100 ngan", "200 ngan", "ba tram nghin",
        # dong / vnd / trieu
        "50.000d", "70.000d", "100.000vnd", "150.000 dong", "200.000 dong",
        "300.000 dong", "500.000 dong", "1 trieu", "1tr", "1 trieu ruoi", "nua trieu",
        # dinh tinh
        "re", "binh dan", "tiet kiem", "sinh vien", "cao cap", "sang chanh",
        "vua tui tien", "khong qua dat", "an sang mot bua"
    ],

    "diet": [
        "chay", "keto", "eat clean", "low carb", "khong dau mo",
        "it dau mo", "thuc duong", "giam can", "it calo", "nhieu rau",
        "it tinh bot"
    ],

    "style": [
        "xao", "kho", "luoc", "hap", "chien", "nuong", "rim", "nau canh", "lam goi",
        "xao toi", "xao sa ot", "kho tieu", "nuong muoi ot"
    ],

    "device": [
        "noi chien", "chao", "noi com dien", "lo vi song", "noi ap suat", "may xay",
        "noi chien khong dau", "bep dien", "bep ga"
    ]
}

# ========== TEMPLATE (GIU KIEU BAN DA CO + MO RONG) ==========

# (cau_mau, intent, {slot_name_trong_json: key_trong_DATA_POOL hoac 'literal'})
TEMPLATES = [
    # 1. SUGGEST_RECIPE
    ("goi y mon {protein} {style} cho {quantity}", 
     "suggest_recipe", {"protein": "protein", "tags": "style", "servings": "quantity"}),

    ("toi muon nau mon {protein} trong {time} gia {price}", 
     "suggest_recipe", {"protein": "protein", "time": "time", "price": "price"}),

    ("nau {quantity} an mon {protein} het bao nhieu tien", 
     "suggest_recipe", {"protein": "protein", "servings": "quantity"}),

    ("tim mon {protein} {time} khoang {price}", 
     "suggest_recipe", {"protein": "protein", "time": "time", "price": "price"}),

    ("mon {protein} nao nau bang {device} mat {time}", 
     "suggest_recipe", {"protein": "protein", "device": "device", "time": "time"}),

    ("hom nay an {protein} voi {vege} cho {quantity}", 
     "suggest_recipe", {"protein": "protein", "ingredient": "vege", "servings": "quantity"}),

    ("hom nay nen nau gi tu {protein} cho {quantity}", 
     "suggest_recipe", {"protein": "protein", "servings": "quantity"}),

    ("doi gio an {protein} {style} kem {vege} duoc khong", 
     "suggest_recipe", {"protein": "protein", "tags": "style", "ingredient": "vege"}),

    ("neu co {protein} va {vege} thi nau mon gi ngon", 
     "suggest_recipe", {"protein": "protein", "ingredient": "vege"}),

    # 2. FILTER_TIME
    ("mon nao nau nhanh duoi {time}", 
     "filter_time", {"time": "time"}),

    ("toi chi co {time} de nau an thoi", 
     "filter_time", {"time": "time"}),

    ("co mon nao lam {time} xong khong", 
     "filter_time", {"time": "time"}),

    ("an gi trong {time} cho kip gio di lam", 
     "filter_time", {"time": "time"}),

    ("bua toi muon nau gi trong vong {time}", 
     "filter_time", {"time": "time"}),

    # 3. FILTER_PRICE / BUDGET
    ("an gi duoi {price}", 
     "filter_price", {"price": "price"}),

    ("mon ngon re tam {price}", 
     "suggest_recipe", {"price": "price"}),

    ("com sinh vien gia {price}", 
     "suggest_recipe", {"price": "price"}),

    ("chi co {price} thi an mon gi", 
     "filter_price", {"price": "price"}),

    ("an sang mot bua gia {price} thi nen an gi", 
     "filter_price", {"price": "price"}),

    # 4. FILTER_DIET
    ("tu van thuc don {diet} cho {quantity}", 
     "filter_diet", {"diet": "diet", "servings": "quantity"}),

    ("mon {protein} cho nguoi an {diet}", 
     "filter_diet", {"protein": "protein", "diet": "diet"}),

    ("an kieu {diet} thi nen an gi tu {protein}", 
     "filter_diet", {"diet": "diet", "protein": "protein"}),

    ("doi mon {diet} trong ngay ma van re nhu {price}", 
     "filter_diet", {"diet": "diet", "price": "price"}),

    # 5. KET HOP DIEU KIEN
    ("goi y mon {protein} {style} {diet} cho {quantity} trong {time}", 
     "suggest_recipe", {
         "protein": "protein", "tags": "style", "diet": "diet",
         "servings": "quantity", "time": "time"
     }),

    ("an {protein} {diet} trong {time} voi ngan sach {price}", 
     "suggest_recipe", {
         "protein": "protein", "diet": "diet",
         "time": "time", "price": "price"
     }),

    ("mon nao tu {protein} vua {diet} vua phu hop {quantity}", 
     "filter_diet", {"protein": "protein", "diet": "diet", "servings": "quantity"})
]

def generate_samples(num_samples):
    samples = []
    for _ in range(num_samples):
        tmpl, intent, slot_map = random.choice(TEMPLATES)
        text = tmpl
        slots = {}
        
        for placeholder, data_key in slot_map.items():
            value = random.choice(DATA_POOL.get(data_key, [""]))
            key_in_text = "{" + data_key + "}"
            text = text.replace(key_in_text, value)
            
            # Lưu nguyên văn giá trị vào slots để tool sync_data dễ tìm
            # (Logic parse số sẽ nằm ở pipeline.py lúc chạy thật)
            slots[placeholder] = value
                
        json_str = json.dumps(slots, ensure_ascii=False)
        samples.append(f"{text}\t{intent}\t{json_str}")
    return samples

def main():
    base_dir = "serverAI/data/nlu"
    os.makedirs(base_dir, exist_ok=True)
    
    print(f"Dang sinh {NUM_TRAIN} mau train mo rong...")
    train_data = generate_samples(NUM_TRAIN)
    with open(f"{base_dir}/train.tsv", "w", encoding="utf-8") as f:
        f.write("\n".join(train_data))
        
    print(f"Dang sinh {NUM_VALID} mau valid mo rong...")
    valid_data = generate_samples(NUM_VALID)
    with open(f"{base_dir}/valid.tsv", "w", encoding="utf-8") as f:
        f.write("\n".join(valid_data))
        
    print(f"Xong! Da tao file train.tsv va valid.tsv moi tai {base_dir}")

if __name__ == "__main__":
    main()