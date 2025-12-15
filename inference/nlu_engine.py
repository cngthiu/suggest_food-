#file: serverAI/inference/nlu_engine.py
import os
import json
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Any
from transformers import AutoTokenizer
from .utils import norm_text

class NLU:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir) / "nlu_onnx"
        
        # --- 1. LOAD INTENT MODEL (ONNX) ---
        print("[INFO] NLU: Loading Intent Model (ONNX)...")
        try:
            # Load Tokenizer (Dùng chung tokenizer PhoBERT cho cả 2 model vì cùng gốc)
            # Lưu ý: Cần trỏ đúng đường dẫn tokenizer đã lưu
            self.tokenizer = AutoTokenizer.from_pretrained("serverAI/models/intent_phobert") 
            
            intent_path = self.model_dir / "intent_model.quant.onnx"
            if not intent_path.exists(): intent_path = self.model_dir / "intent_model.onnx"
            
            self.sess_intent = ort.InferenceSession(str(intent_path))
            
            with open(self.model_dir / "intent_labels.json", "r", encoding="utf-8") as f:
                self.intent_labels = json.load(f)["labels"]
        except Exception as e:
            print(f"[ERROR] Intent Load Failed: {e}")
            self.sess_intent = None

        # --- 2. LOAD NER MODEL (ONNX) ---
        print("[INFO] NLU: Loading NER Model (ONNX)...")
        try:
            ner_path = self.model_dir / "ner_model.quant.onnx"
            if not ner_path.exists(): ner_path = self.model_dir / "ner_model.onnx"
            
            self.sess_ner = ort.InferenceSession(str(ner_path))
            # Hardcode hoặc load label từ file config
            self.ner_labels = ["O", "B-FOOD", "I-FOOD", "B-TIME", "I-TIME", "B-QUANTITY", "I-QUANTITY", "B-PRICE", "I-PRICE"]
        except Exception as e:
            print(f"[ERROR] NER Load Failed: {e}")
            self.sess_ner = None

    def predict_intent(self, text: str) -> Dict[str, Any]:
        if not self.sess_intent:
            return {"name": "unknown", "score": 0.0}
            
        s = norm_text(text, lowercase=True, strip_dia=False) # PhoBERT cần có dấu nếu model train có dấu
        inputs = self.tokenizer(s, return_tensors="np", padding=True, truncation=True)
        
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
        
        logits = self.sess_intent.run(None, ort_inputs)[0]
        probs = self._softmax(logits[0])
        idx = np.argmax(probs)
        
        return {"name": self.intent_labels[idx], "score": float(probs[idx])}

    def extract_slots(self, text: str) -> Dict[str, Any]:
        """Trích xuất thực thể dùng ONNX Runtime."""
        slots = {
            "servings": None, "time": None, "price": None, 
            "diet": None, "protein": None, "device": None, "allergy": []
        }
        
        if not self.sess_ner:
            return slots

        # 1. Pre-process
        # Chuẩn hóa text nhẹ
        s = " ".join(text.split()) 
        
        # 2. Tokenize & Prepare Input
        inputs = self.tokenizer(s, return_tensors="np")
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }

        # 3. Inference (Siêu nhanh)
        logits = self.ort_session.run(None, ort_inputs)[0] # Shape: (1, seq_len, num_labels)
        predictions = np.argmax(logits, axis=2)[0]
        
        # 4. Decode Tags
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # Logic ghép token wordpieces (vd: "thịt_" "gà") thành từ hoàn chỉnh
        # PhoBERT dùng "@@" hoặc "_" tùy version, ta xử lý cơ bản:
        extracted_entities = []
        current_entity = {"label": None, "words": []}
        
        for idx, (token, label_id) in enumerate(zip(tokens, predictions)):
            if token in ["<s>", "</s>", "<pad>"]: continue
            
            label_name = self.ner_labels[label_id] if label_id < len(self.ner_labels) else "O"
            
            # Xử lý subword (những từ bắt đầu bằng @@ trong một số tokenizer)
            clean_token = token.replace("@@", "")
            if token.startswith(" ") or token == tokens[0]: # SentencePiece prefix
                clean_token = token.replace(" ", "")

            if label_name.startswith("B-"):
                if current_entity["label"]:
                    extracted_entities.append(current_entity)
                current_entity = {"label": label_name[2:], "words": [clean_token]}
            elif label_name.startswith("I-") and current_entity["label"] == label_name[2:]:
                current_entity["words"].append(clean_token)
            else:
                if current_entity["label"]:
                    extracted_entities.append(current_entity)
                    current_entity = {"label": None, "words": []}
        
        # Append entity cuối cùng nếu có
        if current_entity["label"]:
            extracted_entities.append(current_entity)

        # 5. Map to Slots
        import re
        foods = []
        for ent in extracted_entities:
            val = " ".join(ent["words"]).replace("_", " ").strip()
            lbl = ent["label"]
            
            if lbl == "FOOD":
                foods.append(val)
            elif lbl == "TIME":
                m = re.search(r"\d+", val)
                if m: slots["time"] = f"<= {m.group(0)}"
            elif lbl == "QUANTITY":
                m = re.search(r"\d+", val)
                if m: slots["servings"] = int(m.group(0))
        
        if foods:
            slots["protein"] = " ".join(dict.fromkeys(foods)) # Remove duplicates

        return slots
    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()