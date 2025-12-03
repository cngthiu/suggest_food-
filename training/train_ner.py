"""
Script huấn luyện mô hình NER (Named Entity Recognition) sử dụng SpaCy.
Input: File JSON chứa dữ liệu đã gán nhãn (dạng list các dict).
Output: Folder model spacy (ví dụ: serverAI/models/ner_model).
"""

import os
import json
import random
import argparse
import sys
from pathlib import Path

import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding
from spacy.training import offsets_to_biluo_tags

# Thêm đường dẫn để import được module serverAI từ thư mục gốc
sys.path.append(os.getcwd())

try:
    from serverAI.inference.utils import norm_text
    print("✅ Đã load thành công hàm norm_text từ utils.")
except ImportError:
    print("⚠️ Cảnh báo: Không tìm thấy serverAI.inference.utils. Code sẽ chạy mà không kiểm tra chuẩn hóa.")
    def norm_text(s, **kwargs): return s


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def validate_data(data):
    print(f"\n--- Đang kiểm tra chất lượng dữ liệu ({len(data)} mẫu) ---")
    dirty_count = 0
    for i, item in enumerate(data):
        raw_text = item.get("text", "")
        # dùng cùng logic với generator
        clean_text = norm_text(raw_text, lowercase=True)

        if raw_text != clean_text:
            dirty_count += 1
            if dirty_count <= 3:
                print(f"[CẢNH BÁO] Mẫu #{i+1} chưa được chuẩn hóa!")
                print(f"  Gốc : {repr(raw_text)}")
                print(f"  Chuẩn: {repr(clean_text)}")
                print("  -> Hãy sửa lại file json để text giống dòng 'Chuẩn', và cập nhật lại vị trí entity.")

    if dirty_count > 0:
        print(f"⚠️  Tổng cộng {dirty_count} mẫu chưa chuẩn hóa. Mô hình có thể hoạt động kém chính xác.")
        print("💡 Gợi ý: Dùng script chuẩn hóa dữ liệu trước khi gán nhãn.")
    else:
        print("✅ Dữ liệu sạch và đồng bộ với norm_text.")


def check_alignment(nlp, data, max_show=10):
    """
    Kiểm tra entity nào bị lệch token boundary.
    In ra một số mẫu lỗi để bạn sửa offset trong JSON.
    """
    print("\n--- Kiểm tra entity alignment ---")
    bad = 0

    for i, item in enumerate(data):
        text = item["text"]
        entities = item.get("entities", [])
        spans = [(start, end, label) for start, end, label in entities]
        doc = nlp.make_doc(text)

        try:
            tags = offsets_to_biluo_tags(doc, spans)
        except Exception as e:
            bad += 1
            if bad <= max_show:
                print(f"[LỖI HARD] Mẫu #{i+1}: {repr(text)}")
                print("  Entities:", spans)
                print("  -> Exception:", e)
            continue

        if "-" in tags:
            bad += 1
            if bad <= max_show:
                print(f"[MISALIGNED] Mẫu #{i+1}: {repr(text)}")
                print("  Entities:", spans)
                print("  Tags    :", tags)
                for (start, end, label) in spans:
                    print(f"    {label}: [{start}, {end}] -> {repr(text[start:end])}")

    if bad == 0:
        print("✅ Không có entity bị misaligned.")
    else:
        print(f"⚠️ Có tổng cộng khoảng {bad} mẫu có entity misaligned (hiển thị tối đa {max_show}).")
        print("   -> Những entity này sẽ bị spaCy bỏ qua khi train. Nên fix offset trong JSON.")


def evaluate_ner(nlp, data):
    """
    Đánh giá mô hình NER trên một tập dữ liệu (dev set).
    Trả về: precision, recall, f1 (cho entities).
    """
    if not data:
        return 0.0, 0.0, 0.0

    examples = []
    for item in data:
        text = item["text"]
        ents = item.get("entities", [])
        doc = nlp.make_doc(text)
        try:
            ex = Example.from_dict(doc, {"entities": ents})
            examples.append(ex)
        except Exception as e:
            # Bỏ qua mẫu lỗi alignment nặng
            continue

    if not examples:
        return 0.0, 0.0, 0.0

    scores = nlp.evaluate(examples)  # <-- trả về dict
    return scores.get("ents_p", 0.0), scores.get("ents_r", 0.0), scores.get("ents_f", 0.0)



def train(data_path, output_dir, n_iter=30, drop=0.3):
    # 1. Load và Validate dữ liệu
    TRAIN_DATA = load_data(data_path)
    validate_data(TRAIN_DATA)

    # 2. Khởi tạo mô hình SpaCy trắng (Blank Language Model)
    try:
        nlp = spacy.blank("vi")
        print("Load language: Vietnamese (vi)")
    except Exception:
        nlp = spacy.blank("xx")
        print("Load language: Multi-language (xx)")

    # Check alignment một lần trước khi train
    check_alignment(nlp, TRAIN_DATA)

    # 3. Chia train/dev (80/20)
    random.shuffle(TRAIN_DATA)
    split = int(len(TRAIN_DATA) * 0.8)
    train_data = TRAIN_DATA[:split]
    dev_data = TRAIN_DATA[split:]
    print(f"\n--- Chia dữ liệu: {len(train_data)} train / {len(dev_data)} dev ---")

    # 4. Tạo pipeline NER
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # 5. Thêm nhãn (Labels) vào mô hình
    for example in TRAIN_DATA:
        for ent in example.get("entities", []):
            ner.add_label(ent[2])

    # 6. Huấn luyện
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    print(f"\n--- Bắt đầu huấn luyện ({n_iter} vòng) ---")
    best_f = -1.0  # Để lưu best F1 trên dev
    output_path = Path(output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    with nlp.disable_pipes(*other_pipes):
        # Với spaCy 3.x (3.8.11), khuyến nghị dùng initialize
        optimizer = nlp.initialize()

        for itn in range(n_iter):
            random.shuffle(train_data)
            losses = {}

            # Tạo lại compounding mỗi epoch cho rõ ràng
            sizes = compounding(4.0, 32.0, 1.001)
            batches = minibatch(train_data, size=sizes)

            for batch in batches:
                texts = [d["text"] for d in batch]
                annotations = [{"entities": d["entities"]} for d in batch]

                examples = []
                for text, ann in zip(texts, annotations):
                    doc = nlp.make_doc(text)
                    try:
                        example = Example.from_dict(doc, ann)
                        examples.append(example)
                    except Exception as e:
                        # Bỏ qua mẫu lỗi alignment nặng
                        print(f"[TRAIN] Bỏ qua mẫu lỗi: {repr(text)} - {e}")

                if not examples:
                    continue

                nlp.update(
                    examples,
                    drop=drop,
                    losses=losses,
                    sgd=optimizer,
                )

            # Đánh giá dev mỗi vòng (hoặc mỗi 5 vòng nếu muốn giảm log)
            p, r, f = evaluate_ner(nlp, dev_data)
            msg = f"Vòng {itn + 1:3d} | Loss: {losses.get('ner', 0.0):8.3f} | Dev P: {p:5.3f} R: {r:5.3f} F1: {f:5.3f}"
            print(msg)

            # Lưu best model nếu F1 tốt hơn
            if f > best_f:
                best_f = f
                best_path = output_path / "best_model"
                nlp.to_disk(best_path)
                print(f"  👉 Cập nhật best model (F1={best_f:.3f}) tại: {best_path}")

    # 7. Lưu "last model" (mô hình sau epoch cuối)
    nlp.meta["name"] = "food_ner_model"
    last_model_path = output_path / "last_model"
    nlp.to_disk(last_model_path)
    print(f"\n🎉 Đã lưu last model tại: {last_model_path}")
    print(f"🔎 Best dev F1: {best_f:.3f} (model lưu tại: {output_path / 'best_model'})")

    # 8. Test nhanh với 1 câu
    try:
        loaded_nlp = spacy.load(best_path if best_f >= 0 else last_model_path)
    except Exception:
        loaded_nlp = spacy.load(last_model_path)

    test_text = TRAIN_DATA[0]['text'] if TRAIN_DATA else "nấu món cá 3 người ăn"
    doc = loaded_nlp(test_text)
    print(f"\nTest nhanh:")
    print(f"Input: {test_text}")
    print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])


def main():
    ap = argparse.ArgumentParser(description="Train NER Model for SmartShop AI")
    ap.add_argument("--data", required=True, help="Đường dẫn file json dữ liệu train")
    ap.add_argument("--output", required=True, help="Thư mục lưu model output")
    ap.add_argument("--iter", type=int, default=30, help="Số vòng lặp huấn luyện (default: 30)")
    args = ap.parse_args()

    train(args.data, args.output, args.iter)


if __name__ == "__main__":
    main()
