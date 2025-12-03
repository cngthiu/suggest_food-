import json
import random
import os
import sys
import unicodedata
import re

def norm_text(s):
    if not s: return ""
    s = str(s)
    # Chuẩn hóa khoảng trắng và Unicode
    s = re.sub(r'\s+', ' ', s).strip()
    s = unicodedata.normalize('NFC', s)
    s = s.lower()
    return s

def find_entity(text, substring, label):
    start = text.find(substring)
    if start == -1:
        return None
    end = start + len(substring)
    return [start, end, label]

import spacy
from spacy.training import offsets_to_biluo_tags
nlp = spacy.blank("vi")  # hoặc "xx" nếu chưa có vocab vi

def is_aligned(text, entities):
    """
    Kiểm tra xem các (start, end, label) có align đúng token boundary của spaCy không.
    Nếu có '-' trong BILUO tags -> misaligned.
    """
    doc = nlp.make_doc(text)
    spans = [(s, e, l) for (s, e, l) in entities]
    try:
        tags = offsets_to_biluo_tags(doc, spans)
    except Exception:
        # Có lỗi nặng (overlap, out-of-range, v.v.)
        return False
    return "-" not in tags

foods = [
    # Món chính cơ bản
    "gà", "cá", "bò", "heo", "tôm", "mực", "trứng", "sườn", "ba chỉ",
    "thịt băm", "đậu phụ", "nấm", "cá hồi", "cá lóc", "cá thu", "cá rô",
    "gà ta", "gà công nghiệp", "thịt bò mỹ", "thịt nạc vai", "chân giò",
    "sườn non", "ba rọi", "tôm sú", "mực ống", "bạch tuộc",

    # Món thịt/ hải sản chế biến sẵn
    "gà kho", "gà rang gừng", "gà chiên nước mắm", "gà luộc",
    "thịt kho tàu", "thịt kho trứng", "thịt rang cháy cạnh",
    "cá kho tộ", "cá chiên giòn", "cá hấp xì dầu",
    "sườn xào chua ngọt", "sườn ram mặn", "sườn nướng mật ong",
    "bò xào hành tây", "bò lúc lắc", "bò nấu lagu",
    "mực hấp", "mực xào chua ngọt", "mực chiên giòn",
    "tôm nướng", "tôm rang thịt", "tôm rim mặn ngọt",
    "trứng chiên", "trứng hấp", "trứng kho thịt",
    "đậu sốt cà chua", "đậu rán", "đậu kho nấm",

    # Món rau / canh
    "rau muống", "rau muống xào tỏi", "rau muống luộc",
    "bí xanh", "canh bí xanh nấu tôm",
    "rau cải", "canh rau cải thịt bằm",
    "bắp cải", "bắp cải xào",
    "su hào", "su hào xào trứng",
    "khoai tây", "khoai tây chiên",
    "cà chua", "canh cà chua trứng",
    "mướp đắng", "mướp đắng xào trứng",
    "canh chua cá", "canh bầu tôm", "canh măng",
    "canh rau dền", "canh cua rau đay",

    # Món một đĩa / món nước phổ biến
    "cơm gà", "cơm thịt kho", "cơm sườn", "cơm tấm",
    "bún bò", "bún riêu", "bún chả",
    "phở bò", "phở gà",
    "miến gà", "miến tôm",
    "cháo gà", "cháo tôm", "cháo thịt bằm",
    "lẩu gà", "lẩu hải sản", "lẩu bò nhúng dấm"
]

quantities = [
    # Số người
    "1 người", "2 người", "3 người", "4 người", "5 người", "6 người", "8 người", "10 người",
    "1 người ăn", "2 người ăn", "3 người lớn", "4 người lớn",
    "2 vợ chồng", "cả nhà", "cho bé", "cho 2 bé", "cho trẻ nhỏ",
    "đại gia đình", "nhóm 5 người", "3 bạn", "4 thành viên", "gia đình 3 người",
    "gia đình 4 người", "gia đình 5 người", "2 người lớn 1 trẻ em",
    "1 mẹ 1 con", "anh em trong phòng trọ",

    # Suất / phần
    "1 suất", "2 suất", "3 suất", "4 phần", "5 phần ăn",
    "6 phần ăn", "7 phần ăn", "10 phần ăn",
    "nửa con", "1 con", "1kg", "500g", "300g", "700g"
]

times = [
    # Phút
    "5 phút", "10 phút", "15 phút", "20 phút", "25 phút", "30 phút",
    "35 phút", "40 phút", "45 phút", "50 phút", "60 phút",
    "5p", "10p", "15p", "20p", "25p", "30p", "45p", "60p", "90p",

    # Giờ / khoảng
    "1 tiếng", "1 tiếng rưỡi", "2 tiếng", "hơn 1 tiếng",
    "nửa tiếng", "1h", "1.5h", "2h",

    # Chung chung / ngữ nghĩa
    "nấu nhanh", "siêu tốc", "cấp tốc", "trong tích tắc",
    "tốn ít thời gian", "không mất nhiều thời gian",
    "nấu trong giờ nghỉ trưa", "nấu sau giờ làm",
    "dành cho bữa sáng vội", "chuẩn bị trong buổi tối",
    "làm được trong giờ nghỉ"
]

prices = [
    # Nghìn / ngàn
    "30 nghìn", "40 nghìn", "50 nghìn", "70 nghìn", "80 nghìn",
    "100 ngàn", "120 ngàn", "150 ngàn", "200 ngàn", "250 ngàn",
    "300 ngàn", "400 ngàn", "500 ngàn",
    "30k", "40k", "50k", "60k", "70k", "80k",
    "100k", "120k", "150k", "180k", "200k", "250k", "300k", "400k", "500k", "800k", "1000k",

    # Đồng / triệu
    "50.000đ", "70.000đ", "100.000vnd", "150.000 đồng", "200.000 đồng",
    "300.000 đồng", "500.000 đồng", "1 triệu", "1 triệu rưỡi",

    # Ngân sách định tính
    "bình dân", "giá rẻ", "cao cấp", "tiết kiệm", "sinh viên",
    "vừa túi tiền", "không quá đắt", "thoải mái chi tiêu", "ăn sang một bữa"
]

modifiers = [
    "ngon", "bổ dưỡng", "thanh đạm", "đậm đà", "cay", "không cay",
    "ít dầu mỡ", "nhiều đạm", "giảm cân", "eat clean", "truyền thống",
    "ít ngọt", "ít mặn", "ít tinh bột", "nhiều rau", "nhiều chất xơ",
    "ít calo", "không chiên rán", "hấp dẫn", "lạ miệng", "dễ ăn",
    "phù hợp cho bé", "phù hợp người già", "phù hợp người ăn kiêng"
]

# --- CÁC MẪU CÂU (TEMPLATES) MỞ RỘNG ---

templates = [
    # === Giữ lại các template gốc ===
    "tôi muốn nấu món {FOOD} cho {QUANTITY} ăn trong {TIME} khoảng {PRICE}",
    "gợi ý món {FOOD} {TIME} giá dưới {PRICE} cho {QUANTITY}",
    "tìm thực đơn {FOOD} {MODIFIER} cho {QUANTITY} mất {TIME}",
    "cần làm món {FOOD} {MODIFIER} giá tầm {PRICE}",

    "món {FOOD} nào nấu nhanh trong {TIME}",
    "cách làm {FOOD} dưới {TIME}",
    "tìm món {FOOD} ăn liền {TIME}",
    "nấu {FOOD} cho {QUANTITY} ăn",
    "khẩu phần {FOOD} dành cho {QUANTITY}",
    "làm món {FOOD} đủ cho {QUANTITY}",
    "mua {FOOD} hết bao nhiêu tiền khoảng {PRICE}",
    "món {FOOD} ngon rẻ dưới {PRICE}",
    "ăn {FOOD} ngân sách {PRICE} quay đầu",

    "hôm nay ăn {FOOD} được không",
    "thèm {FOOD} quá",
    "có món {FOOD} nào {MODIFIER} không",
    "tư vấn thực đơn {QUANTITY} với {PRICE}",
    "bữa trưa {TIME} có món gì từ {FOOD}",
    "nhà có {QUANTITY} muốn ăn {FOOD} {TIME}",
    "chỉ có {PRICE} thì nấu món {FOOD} gì",
    "gợi ý món {MODIFIER} từ {FOOD}",

    # === Template đầy đủ thông tin hơn ===
    "gợi ý thực đơn từ {FOOD} cho {QUANTITY} trong {TIME} với ngân sách khoảng {PRICE}",
    "muốn nấu {FOOD} cho {QUANTITY} ăn, thời gian {TIME}, chi phí {PRICE} thì làm món gì",
    "nấu món {FOOD} vừa {MODIFIER} cho {QUANTITY}, làm trong {TIME}, tiền khoảng {PRICE}",
    "tư vấn món từ {FOOD} phù hợp {MODIFIER} cho {QUANTITY} trong {TIME}, tầm giá {PRICE}",
    "tôi có {PRICE}, muốn nấu {FOOD} {MODIFIER} cho {QUANTITY} trong {TIME}",

    # === Template ưu tiên thời gian ===
    "có món {FOOD} nào {MODIFIER} nấu trong {TIME} không",
    "món {FOOD} nào làm được trong {TIME} cho {QUANTITY}",
    "cần món {FOOD} nấu siêu nhanh {TIME} cho {QUANTITY}",
    "bữa tối cần món {FOOD} làm {TIME} là xong",
    "gợi ý món {FOOD} làm nhanh trong {TIME} mà vẫn {MODIFIER}",

    # === Template ưu tiên giá tiền ===
    "chỉ có khoảng {PRICE} thì nên mua gì từ {FOOD} cho {QUANTITY}",
    "ngân sách {PRICE} thì nấu món {FOOD} nào {MODIFIER}",
    "{PRICE} đủ để nấu món {FOOD} cho {QUANTITY} không",
    "muốn ăn {FOOD} {MODIFIER} mà giá {PRICE} thì có món nào",
    "tìm món {FOOD} giá {PRICE} ăn cho {QUANTITY}",

    # === Template ưu tiên khẩu vị / chế độ ăn ===
    "tôi muốn ăn {FOOD} kiểu {MODIFIER} cho {QUANTITY} trong {TIME}",
    "món {FOOD} nào {MODIFIER} phù hợp {QUANTITY}",
    "gợi ý thực đơn {MODIFIER} với nguyên liệu chính là {FOOD} cho {QUANTITY}",
    "muốn ăn {FOOD} nhưng phải {MODIFIER}, thời gian nấu khoảng {TIME}",
    "có món {FOOD} nào vừa {MODIFIER} vừa hợp cho {QUANTITY} không",

    # === Câu hỏi tự nhiên, hội thoại ===
    "hôm nay nên nấu {FOOD} gì cho {QUANTITY}",
    "tối nay ăn {FOOD} được không, {PRICE} có đủ không",
    "trưa nay muốn ăn {FOOD} {MODIFIER} mà chỉ có {PRICE}",
    "mai có khách {QUANTITY}, nên làm món {FOOD} nào {MODIFIER}",
    "không biết {TIME} có kịp nấu {FOOD} cho {QUANTITY} không",

    # === Dạng 'chọn món' ===
    "giữa {FOOD} và các món khác thì nên chọn gì cho {QUANTITY} với {PRICE}",
    "từ {FOOD} có thể biến tấu thành món gì {MODIFIER} cho {QUANTITY}",
    "đổi gió với {FOOD} {MODIFIER} cho {QUANTITY}, thời gian nấu {TIME}",
    "món {FOOD} nào hợp ăn cơm cho {QUANTITY}, giá khoảng {PRICE}",
    "tìm món mặn từ {FOOD} ăn với cơm cho {QUANTITY}",

    # === Dạng 'planning / meal prep' ===
    "muốn nấu sẵn {FOOD} cho {QUANTITY} mang đi làm trong {TIME}",
    "meal prep {FOOD} {MODIFIER} cho {QUANTITY} ăn trong {TIME}",
    "chuẩn bị bữa ăn {MODIFIER} từ {FOOD} cho {QUANTITY}, ngân sách {PRICE}",
    "lên thực đơn có {FOOD} {MODIFIER} cho {QUANTITY} trong khoảng {TIME}",
    "chuẩn bị bữa cơm với {FOOD} cho {QUANTITY}, tốn khoảng {PRICE}",

    # === Dạng siêu ngắn, tự nhiên ===
    "có gợi ý món {FOOD} nào {MODIFIER} không",
    "ăn {FOOD} gì cho {QUANTITY} nhanh trong {TIME}",
    "mua {FOOD} khoảng {PRICE} đủ cho {QUANTITY} không",
    "món {FOOD} nào dễ làm cho {QUANTITY}",
    "làm sao nấu {FOOD} vừa {MODIFIER} vừa rẻ khoảng {PRICE}"
]


def generate_dataset(num_samples=1000):
    dataset = []
    generated_texts = set()
    attempts = 0
    
    while len(dataset) < num_samples and attempts < num_samples * 5:
        attempts += 1
        
        template = random.choice(templates)
        f_val = random.choice(foods)
        q_val = random.choice(quantities)
        t_val = random.choice(times)
        p_val = random.choice(prices)
        m_val = random.choice(modifiers)
        
        raw_text = template.replace("{FOOD}", f_val) \
                           .replace("{QUANTITY}", q_val) \
                           .replace("{TIME}", t_val) \
                           .replace("{PRICE}", p_val) \
                           .replace("{MODIFIER}", m_val)
        
        clean_text = norm_text(raw_text)
        if clean_text in generated_texts:
            continue
            
        entities = []
        
        def add_ent(val, label):
            val_clean = norm_text(val)
            if val_clean in clean_text:
                ent = find_entity(clean_text, val_clean, label)
                if ent:
                    entities.append(ent)

        if "{FOOD}" in template: add_ent(f_val, "FOOD")
        if "{QUANTITY}" in template: add_ent(q_val, "QUANTITY")
        if "{TIME}" in template: add_ent(t_val, "TIME")
        if "{PRICE}" in template: add_ent(p_val, "PRICE")
        
        if len(entities) > 0:
            # 🔥 FIX QUAN TRỌNG: sort entities theo start offset
            entities.sort(key=lambda e: e[0])
            
            if is_aligned(clean_text, entities):
                dataset.append({
                    "text": clean_text,
                    "entities": entities
                })
                generated_texts.add(clean_text)
    
    return dataset

if __name__ == "__main__":
    TARGET = 2000
    print(f"Đang sinh {TARGET} mẫu dữ liệu...")
    
    data = generate_dataset(TARGET)
    
    # Tạo thư mục nếu chưa có
    output_dir = "serverAI/data/nlu"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "ner_train_1000.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Đã tạo thành công file: {output_path}")
    print(f"   Tổng số mẫu thực tế: {len(data)}")