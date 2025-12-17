# import random
# import json
# import os

# # --- CẤU HÌNH ---
# NUM_TRAIN = 30  # Tăng lên 3000 mẫu để phủ hết các trường hợp
# NUM_VALID = 3

# # --- KHO DỮ LIỆU PHONG PHÚ (Rich Data Pool) ---
# DATA_POOL = {
#     "protein": [
#         # gà
#         "gà", "thịt gà", "cánh gà", "ức gà", "đùi gà", "cánh gà chiên nước mắm",
#         "gà rang gừng", "gà kho gừng", "gà luộc", "gà nướng", "gà xào sả ớt",
#         # bò
#         "bò", "thịt bò", "bắp bò", "ba chỉ bò", "gân bò", "bò xào hành tây",
#         "bò lúc lắc", "bò nấu lagu", "bò hầm rau củ",
#         # heo
#         "heo", "thịt heo", "sườn", "sườn non", "ba rọi", "ba rọi quay",
#         "thịt xay", "chân giò", "thịt kho tàu", "thịt kho trứng", "thịt rang cháy cạnh",
#         # cá
#         "cá", "cá diêu hồng", "cá basa", "cá lóc", "cá hồi", "cá rô phi",
#         "cá kho tộ", "cá chiên giòn", "cá hấp xì dầu", "canh chua cá",
#         # hải sản
#         "tôm", "tôm sú", "tôm thẻ", "tôm hùm", "tôm rim mặn ngọt", "tôm nướng",
#         "mực", "mực ống", "mực hấp", "mực xào chua ngọt", "mực chiên giòn",
#         "bạch tuộc",
#         # trứng (Đã bổ sung lại)
#         "trứng", "trứng gà", "trứng vịt", "trứng cút", "trứng chiên", "trứng hấp",
#         "trứng kho thịt",
#         # món một đĩa / cơm bún phở (Đã bổ sung lại)
#         "cơm gà", "cơm sườn", "cơm tấm", "cơm thịt kho",
#         "phở bò", "phở gà",
#         "bún bò", "bún riêu", "bún chả",
#         "miến gà", "miến tôm",
#         "cháo gà", "cháo tôm", "cháo thịt băm",
#         # lẩu
#         "lẩu gà", "lẩu hải sản", "lẩu bò nhúng dấm"
#     ],

#     "vege": [
#         "rau muống", "cải thảo", "bắp cải", "bắp cải cuộn thịt", "bí xanh", "bí đỏ",
#         "khoai tây", "khoai lang", "cà rốt", "cà chua", "đậu bắp", "nấm rơm", "nấm hương",
#         "măng tây", "súp lơ", "rau cải", "rau muống xào tỏi", "rau muống luộc",
#         "canh bí xanh nấu tôm", "canh rau cải thịt băm", "canh rau dền", "canh cua rau đay",
#         "mướp đắng", "mướp đắng xào trứng"
#     ],

#     "time": [
#         # phút
#         "5 phút", "10 phút", "15 phút", "20 phút", "25 phút", "30 phút",
#         "35 phút", "40 phút", "45 phút", "50 phút", "60 phút",
#         "5p", "10p", "15p", "20p", "25p", "30p", "45p", "60p", "90p",
#         # giờ
#         "1 tiếng", "1 tiếng rưỡi", "2 tiếng", "hơn 1 tiếng",
#         "nửa tiếng", "1h", "1h30p", "2h",
#         # định tính
#         "nấu nhanh", "siêu tốc", "cấp tốc", "ăn liền", "làm nhanh",
#         "không mất nhiều thời gian", "làm trong giờ nghỉ trưa",
#         "làm sau giờ làm", "dành cho bữa sáng vội", "chuẩn bị tối qua"
#     ],

#     "quantity": [
#         # số người
#         "1 người", "2 người", "3 người", "4 người", "5 người", "6 người", "8 người", "10 người",
#         "cho 1 người", "cho 2 người", "cho 3 người", "cho 4 người",
#         "3 thành viên", "4 thành viên", "gia đình 3 người", "gia đình 4 người", "gia đình 5 người",
#         "2 người lớn 1 trẻ em",
#         # suất / phần / bát / tô
#         "1 suất", "2 suất", "3 suất", "4 phần", "5 phần ăn", "6 phần ăn", "7 phần ăn", "10 phần ăn",
#         "2 bát", "3 bát", "2 tô", "3 tô",
#         # chung chung
#         "cả nhà", "2 vợ chồng", "cho bé", "cho 2 bé", "cho trẻ nhỏ",
#         "đại gia đình", "cho mình tôi", "anh em trong phòng trọ"
#     ],

#     "price": [
#         # k / nghìn
#         "20k", "30k", "40k", "50k", "60k", "70k", "80k",
#         "100k", "120k", "150k", "180k", "200k", "250k", "300k", "400k", "500k", "800k", "1000k",
#         "30 nghìn", "40 nghìn", "50 nghìn", "70 nghìn", "100 nghìn", "150 nghìn",
#         "100 ngàn", "200 ngàn", "ba trăm nghìn",
#         # đồng / vnd / triệu
#         "50.000đ", "70.000đ", "100.000vnđ", "150.000 đồng", "200.000 đồng",
#         "300.000 đồng", "500.000 đồng", "1 triệu", "1tr", "1 triệu rưỡi", "nửa triệu",
#         # định tính
#         "rẻ", "bình dân", "tiết kiệm", "sinh viên", "cao cấp", "sang chảnh",
#         "vừa túi tiền", "không quá đắt", "ăn sang một bữa"
#     ],

#     "diet": [
#         "chay", "keto", "eat clean", "low carb", "không dầu mỡ",
#         "ít dầu mỡ", "thực dưỡng", "giảm cân", "ít calo", "nhiều rau",
#         "ít tinh bột"
#     ],

#     "style": [
#         "xào", "kho", "luộc", "hấp", "chiên", "nướng", "rim", "nấu canh", "làm gỏi",
#         "xào tỏi", "xào sả ớt", "kho tiêu", "nướng muối ớt"
#     ],

#     "device": [
#         "nồi chiên", "chảo", "nồi cơm điện", "lò vi sóng", "nồi áp suất", "máy xay",
#         "nồi chiên không dầu", "bếp điện", "bếp ga"
#     ]
# }

# # ========== TEMPLATE (GIU KIEU BAN DA CO + MO RONG) ==========

# # (cau_mau, intent, {slot_name_trong_json: key_trong_DATA_POOL hoac 'literal'})
# TEMPLATES = [
#     # 1. SUGGEST_RECIPE
#     ("gợi ý món {protein} {style} cho {quantity}", 
#      "suggest_recipe", {"protein": "protein", "tags": "style", "servings": "quantity"}),

#     ("tôi muốn nấu món {protein} trong {time} giá {price}", 
#      "suggest_recipe", {"protein": "protein", "time": "time", "price": "price"}),

#     ("nấu {quantity} ăn món {protein} hết bao nhiêu tiền", 
#      "suggest_recipe", {"protein": "protein", "servings": "quantity"}),

#     ("tìm món {protein} {time} khoảng {price}", 
#      "suggest_recipe", {"protein": "protein", "time": "time", "price": "price"}),

#     ("món {protein} nào nấu bằng {device} mất {time}", 
#      "suggest_recipe", {"protein": "protein", "device": "device", "time": "time"}),

#     ("hôm nay ăn {protein} với {vege} cho {quantity}", 
#      "suggest_recipe", {"protein": "protein", "ingredient": "vege", "servings": "quantity"}),

#     ("hôm nay nên nấu gì từ {protein} cho {quantity}", 
#      "suggest_recipe", {"protein": "protein", "servings": "quantity"}),

#     ("đổi gió ăn {protein} {style} kèm {vege} được không", 
#      "suggest_recipe", {"protein": "protein", "tags": "style", "ingredient": "vege"}),

#     ("nếu có {protein} và {vege} thì nấu món gì ngon", 
#      "suggest_recipe", {"protein": "protein", "ingredient": "vege"}),

#     # 2. FILTER_TIME
#     ("món nào nấu nhanh dưới {time}", 
#      "filter_time", {"time": "time"}),

#     ("tôi chỉ có {time} để nấu ăn thôi", 
#      "filter_time", {"time": "time"}),

#     ("có món nào làm {time} xong không", 
#      "filter_time", {"time": "time"}),

#     ("ăn gì trong {time} cho kịp giờ đi làm", 
#      "filter_time", {"time": "time"}),

#     ("bữa tối muốn nấu gì trong vòng {time}", 
#      "filter_time", {"time": "time"}),

#     # 3. FILTER_PRICE / BUDGET
#     ("ăn gì dưới {price}", 
#      "filter_price", {"price": "price"}),

#     ("món ngon rẻ tầm {price}", 
#      "suggest_recipe", {"price": "price"}),

#     ("cơm sinh viên giá {price}", 
#      "suggest_recipe", {"price": "price"}),

#     ("chỉ có {price} thì ăn món gì", 
#      "filter_price", {"price": "price"}),

#     ("ăn sang một bữa giá {price} thì nên ăn gì", 
#      "filter_price", {"price": "price"}),

#     # 4. FILTER_DIET
#     ("tư vấn thực đơn {diet} cho {quantity}", 
#      "filter_diet", {"diet": "diet", "servings": "quantity"}),

#     ("món {protein} cho người ăn {diet}", 
#      "filter_diet", {"protein": "protein", "diet": "diet"}),

#     ("ăn kiểu {diet} thì nên ăn gì từ {protein}", 
#      "filter_diet", {"diet": "diet", "protein": "protein"}),

#     ("đổi món {diet} trong ngày mà vẫn rẻ như {price}", 
#      "filter_diet", {"diet": "diet", "price": "price"}),

#     # 5. KẾT HỢP ĐIỀU KIỆN
#     ("gợi ý món {protein} {style} {diet} cho {quantity} trong {time}", 
#      "suggest_recipe", {
#          "protein": "protein", "tags": "style", "diet": "diet",
#          "servings": "quantity", "time": "time"
#      }),

#     ("ăn {protein} {diet} trong {time} với ngân sách {price}", 
#      "suggest_recipe", {
#          "protein": "protein", "diet": "diet",
#          "time": "time", "price": "price"
#      }),

#     ("món nào từ {protein} vừa {diet} vừa phù hợp {quantity}", 
#      "filter_diet", {"protein": "protein", "diet": "diet", "servings": "quantity"})
# ]

# def generate_samples(num_samples):
#     samples = []
#     for _ in range(num_samples):
#         tmpl, intent, slot_map = random.choice(TEMPLATES)
#         text = tmpl
#         slots = {}
        
#         for placeholder, data_key in slot_map.items():
#             value = random.choice(DATA_POOL.get(data_key, [""]))
#             key_in_text = "{" + data_key + "}"
#             text = text.replace(key_in_text, value)
            
#             # Lưu nguyên văn giá trị vào slots để tool sync_data dễ tìm
#             # (Logic parse số sẽ nằm ở pipeline.py lúc chạy thật)
#             slots[placeholder] = value
                
#         json_str = json.dumps(slots, ensure_ascii=False)
#         samples.append(f"{text}\t{intent}\t{json_str}")
#     return samples

# def main():
#     base_dir = "serverAI/data/nlu"
#     os.makedirs(base_dir, exist_ok=True)
    
#     print(f"Dang sinh {NUM_TRAIN} mau train mo rong...")
#     train_data = generate_samples(NUM_TRAIN)
#     with open(f"{base_dir}/train.tsv", "w", encoding="utf-8") as f:
#         f.write("\n".join(train_data))
        
#     print(f"Dang sinh {NUM_VALID} mau valid mo rong...")
#     valid_data = generate_samples(NUM_VALID)
#     with open(f"{base_dir}/valid.tsv", "w", encoding="utf-8") as f:
#         f.write("\n".join(valid_data))
        
#     print(f"Xong! Da tao file train.tsv va valid.tsv moi tai {base_dir}")

# if __name__ == "__main__":
#     main()
import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "/media/congthieu/ubuntu_data/LTTM/MM/serverAI/data/nlu"  # sửa theo bạn
SEED = 42

OUT_TRAIN = os.path.join(DATA_DIR, "train2.tsv")
OUT_VALID = os.path.join(DATA_DIR, "valid2.tsv")

def load_data_tsv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep="\t", header=None, names=["text", "intent"], quoting=3)
    except Exception:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    rows.append([parts[0], parts[1]])
        df = pd.DataFrame(rows, columns=["text", "intent"])

    df["text"] = df["text"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["intent"] != "")]
    return df.reset_index(drop=True)

def norm_text(s: str) -> str:
    s = str(s).strip().lower()
    s = " ".join(s.split())
    return s

# ======================================================
# 1) Load 2 file
# ======================================================
df_train0 = load_data_tsv(os.path.join(DATA_DIR, "train.tsv"))
df_valid0 = load_data_tsv(os.path.join(DATA_DIR, "valid.tsv"))

print("Before merge:", df_train0.shape, df_valid0.shape)

# ======================================================
# 2) Merge
# ======================================================
df_all = pd.concat([df_train0, df_valid0], ignore_index=True)

# ======================================================
# 3) Remove duplicates (normalize text + intent)
# ======================================================
df_all["text_norm"] = df_all["text"].map(norm_text)
df_all["intent_norm"] = df_all["intent"].map(norm_text)

before = len(df_all)
df_all = df_all.drop_duplicates(subset=["text_norm", "intent_norm"]).reset_index(drop=True)
after = len(df_all)

print(f"Removed duplicates: {before - after} / {before} => remaining {after}")

# ======================================================
# 4) Stratified split 80/20
# ======================================================
train_df, valid_df = train_test_split(
    df_all,
    test_size=0.20,
    random_state=SEED,
    stratify=df_all["intent_norm"]
)

# ======================================================
# 5) Drop helper columns
# ======================================================
train_df = train_df.drop(columns=["text_norm", "intent_norm"]).reset_index(drop=True)
valid_df = valid_df.drop(columns=["text_norm", "intent_norm"]).reset_index(drop=True)

print("After split:", train_df.shape, valid_df.shape)
print("\nTrain label counts:\n", train_df["intent"].value_counts())
print("\nValid label counts:\n", valid_df["intent"].value_counts())

# ======================================================
# 6) SAVE FILES (TSV)
# ======================================================
os.makedirs(DATA_DIR, exist_ok=True)

train_df.to_csv(
    OUT_TRAIN,
    sep="\t",
    header=False,
    index=False,
    encoding="utf-8"
)

valid_df.to_csv(
    OUT_VALID,
    sep="\t",
    header=False,
    index=False,
    encoding="utf-8"
)

print("\nSaved files:")
print(" -", OUT_TRAIN)
print(" -", OUT_VALID)

