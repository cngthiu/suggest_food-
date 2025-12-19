import os 
import json
import argparse
import sys
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict
import evaluate
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)

# Thêm đường dẫn gốc để import utils nếu cần
sys.path.append(os.getcwd())

# 1. DANH SÁCH NHÃN ĐẦY ĐỦ (Phải khớp chính xác với LABEL_MAP trong script sinh dữ liệu)
LABEL_LIST = [
    "O", 
    "B-FOOD", "I-FOOD", 
    "B-DIET", "I-DIET", 
    "B-TIME", "I-TIME", 
    "B-QUANTITY", "I-QUANTITY", 
    "B-PRICE", "I-PRICE",
    "B-DEVICE", "I-DEVICE"
]
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}

def prepare_data(data_path):
    """Đọc dữ liệu từ JSON và chuyển đổi sang định dạng tokens/ner_tags chuẩn HuggingFace"""
    if not os.path.exists(data_path):
        print(f"[ERROR] Không tìm thấy file: {data_path}")
        return None

    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    formatted_data = {"tokens": [], "ner_tags": []}
    
    for item in raw_data:
        text = item['text']
        entities = item['entities']
        
        # Pre-tokenization cơ bản theo khoảng trắng
        words = text.split()
        tags = ["O"] * len(words)
        
        # Map vị trí ký tự sang index của từ để gán nhãn chính xác
        char_to_word_idx = []
        for i, w in enumerate(words):
            char_to_word_idx.extend([i] * len(w))
            char_to_word_idx.append(-1) # Cho khoảng trắng
        
        for start, end, label in entities:
            try:
                # Tìm word index của ký tự bắt đầu và kết thúc
                start_word_idx = char_to_word_idx[start]
                end_word_idx = char_to_word_idx[end - 1]
                
                if start_word_idx == -1 or end_word_idx == -1: continue
                
                tags[start_word_idx] = f"B-{label}"
                for i in range(start_word_idx + 1, end_word_idx + 1):
                    if i < len(tags):
                        tags[i] = f"I-{label}"
            except (IndexError, KeyError):
                continue
                
        tag_ids = [LABEL2ID.get(t, 0) for t in tags]
        formatted_data["tokens"].append(words)
        formatted_data["ner_tags"].append(tag_ids)
        
    return Dataset.from_dict(formatted_data)

def tokenize_and_align(examples, tokenizer):
    """Tokenize sub-words và căn chỉnh nhãn cho PhoBERT"""
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True,
        max_length=128
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100) # [CLS], [SEP]
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx]) # Token đầu tiên của một từ
            else:
                label_ids.append(-100) # Các sub-tokens còn lại
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
    ap.add_argument("--train", required=True, help="Đường dẫn file ner_train.json")
    ap.add_argument("--valid", required=True, help="Đường dẫn file ner_valid.json")
    ap.add_argument("--output", required=True, help="Thư mục lưu model sau khi train")
    ap.add_argument("--epochs", type=int, default=10) # Tăng lên 10 để hội tụ tốt hơn
    args = ap.parse_args()

    model_checkpoint = "vinai/phobert-base-v2"
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, add_prefix_space=True)

    print("--- Đang chuẩn bị dữ liệu NER ---")
    train_ds = prepare_data(args.train)
    valid_ds = prepare_data(args.valid)
    
    ds = DatasetDict({"train": train_ds, "validation": valid_ds})
    
    tokenized_datasets = ds.map(
        lambda x: tokenize_and_align(x, tokenizer), 
        batched=True
    )

    # Load Model
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint, 
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # Cấu hình Trainer
    training_args = TrainingArguments(
        output_dir=args.output,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5, # Tăng nhẹ LR cho data sinh
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True, # Tự động giữ model có F1 cao nhất
        metric_for_best_model="f1",
        logging_steps=20
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("--- Bắt đầu huấn luyện PhoBERT NER ---")
    trainer.train()

    # Lưu model cuối cùng
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"✅ Đã lưu model tại: {args.output}")

if __name__ == "__main__":
    main()