import os 
import json
import argparse
import sys
import numpy as np
from pathlib import Path
from datasets import Dataset
import evaluate
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification,
    RobertaTokenizerFast
)

# Thêm đường dẫn gốc để import utils
sys.path.append(os.getcwd())
try:
    from serverAI.inference.utils import norm_text
except ImportError:
    def norm_text(s, **kwargs): return s

# Định nghĩa nhãn (Labels) - Cần khớp với dữ liệu NER của bạn
LABEL_LIST = [
    "O", 
    "B-FOOD", "I-FOOD", 
    "B-TIME", "I-TIME", 
    "B-QUANTITY", "I-QUANTITY", 
    "B-PRICE", "I-PRICE"
]
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}

def prepare_data(data_path):
    """Đọc dữ liệu từ JSON và chuyển đổi sang định dạng tokens/ner_tags"""
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    formatted_data = {"tokens": [], "ner_tags": []}
    
    for item in raw_data:
        text = item['text']
        entities = item['entities']
        
        # Tokenize cơ bản bằng khoảng trắng (pre-tokenization)
        words = text.split()
        tags = ["O"] * len(words)
        
        # Tạo map từ vị trí ký tự sang index của từ
        char_to_word_idx = []
        cursor = 0
        for i, w in enumerate(words):
            char_to_word_idx.extend([i] * len(w))
            cursor += len(w)
            if i < len(words) - 1:
                char_to_word_idx.append(-1) # Khoảng trắng
                cursor += 1
        
        # Gán nhãn cho từng từ
        for start, end, label in entities:
            try:
                start_word_idx = char_to_word_idx[start]
                end_word_idx = char_to_word_idx[end - 1]
                
                if start_word_idx == -1 or end_word_idx == -1: continue
                
                tags[start_word_idx] = f"B-{label}"
                for i in range(start_word_idx + 1, end_word_idx + 1):
                    tags[i] = f"I-{label}"
            except IndexError:
                continue
                
        tag_ids = [LABEL2ID.get(t, 0) for t in tags]
        formatted_data["tokens"].append(words)
        formatted_data["ner_tags"].append(tag_ids)
        
    return Dataset.from_dict(formatted_data)

def tokenize_and_align(examples, tokenizer):
    """Tokenize lại bằng PhoBERT tokenizer và căn chỉnh nhãn"""
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True
    )
    
    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100) # Token đặc biệt ([CLS], [SEP])
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx]) # Token đầu tiên của từ
            else:
                label_ids.append(-100) # Các sub-token tiếp theo (bỏ qua khi tính loss)
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    metric = evaluate.load("seqeval")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="File ner_train.json")
    ap.add_argument("--output", required=True, help="Folder lưu model")
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()

    model_checkpoint = "vinai/phobert-base-v2"
    print(f"Loading {model_checkpoint}...")
    
    # --- PHẦN SỬA LỖI QUAN TRỌNG ---
    # PhoBERT/RoBERTa bắt buộc cần add_prefix_space=True khi dùng pre-tokenized inputs
    try:
        # Thử load AutoTokenizer Fast với tham số bắt buộc
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, add_prefix_space=True)
    except Exception as e:
        print(f"[WARN] Không thể load AutoTokenizer Fast: {e}")
        tokenizer = None

    # Nếu AutoTokenizer không trả về Fast tokenizer, ép dùng RobertaTokenizerFast
    if tokenizer is None or not tokenizer.is_fast:
        print("[INFO] Đang chuyển sang dùng RobertaTokenizerFast trực tiếp...")
        try:
            from transformers import RobertaTokenizerFast
            tokenizer = RobertaTokenizerFast.from_pretrained(model_checkpoint, add_prefix_space=True)
        except Exception as e:
            print(f"[ERROR] Không thể load Tokenizer: {e}")
            print("Gợi ý: pip install transformers tokenizers sentencepiece")
            sys.exit(1)
            
    if not tokenizer.is_fast:
        print("[ERROR] Tokenizer hiện tại không phải Fast version. Code cần hàm word_ids() để chạy.")
        sys.exit(1)
    # -------------------------------

    print("Chuẩn bị dữ liệu...")
    raw_dataset = prepare_data(args.data)
    # Chia train/test
    dataset = raw_dataset.train_test_split(test_size=0.1)
    
    tokenized_datasets = dataset.map(
        lambda x: tokenize_and_align(x, tokenizer), 
        batched=True
    )

    print("Cấu hình model...")
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint, 
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    training_args = TrainingArguments(
        output_dir=args.output,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=2, # Chỉ giữ 2 model tốt nhất để tiết kiệm ổ cứng
        logging_steps=10
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Bắt đầu huấn luyện...")
    trainer.train()

    print(f"Lưu model tại {args.output}...")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print("✅ Hoàn tất!")

if __name__ == "__main__":
    main()