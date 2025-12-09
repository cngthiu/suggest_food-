import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
import json

def load_data(path):
    df = pd.read_csv(path, sep='\t', names=['text', 'intent'])
    # Xử lý nếu file tsv có header hoặc format lạ
    if len(df.columns) < 2:
        # Fallback đọc thủ công nếu pandas lỗi format
        rows = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    rows.append(parts[:2])
        df = pd.DataFrame(rows, columns=['text', 'intent'])
    return df

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

def main():
    # Cấu hình
    DATA_DIR = "serverAI/data/nlu"
    OUT_DIR = "serverAI/models/intent_phobert"
    MODEL_NAME = "vinai/phobert-base-v2"
    
    # 1. Chuẩn bị dữ liệu
    print("[INFO] Loading data...")
    df_train = load_data(os.path.join(DATA_DIR, "train.tsv"))
    df_valid = load_data(os.path.join(DATA_DIR, "valid.tsv"))
    
    # Tạo label map
    labels = sorted(list(set(df_train['intent'].unique()) | set(df_valid['intent'].unique())))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}
    
    print(f"[INFO] Labels: {labels}")
    
    # Lưu label map để dùng lúc inference
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(OUT_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": labels}, f, ensure_ascii=False)

    # 2. Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=64)

    # Convert pandas -> HuggingFace Dataset
    from datasets import Dataset
    train_ds = Dataset.from_pandas(df_train)
    valid_ds = Dataset.from_pandas(df_valid)
    
    # Map label
    train_ds = train_ds.map(lambda x: {"label": label2id[x["intent"]]})
    valid_ds = valid_ds.map(lambda x: {"label": label2id[x["intent"]]})
    
    # Tokenize
    tokenized_train = train_ds.map(preprocess_function, batched=True)
    tokenized_valid = valid_ds.map(preprocess_function, batched=True)

    # 3. Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    # 4. Train
    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("[INFO] Training Intent Model...")
    trainer.train()
    
    print(f"[INFO] Saving model to {OUT_DIR}...")
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

if __name__ == "__main__":
    main()