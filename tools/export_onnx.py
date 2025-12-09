import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import onnxruntime
from pathlib import Path
import os

# Cấu hình đường dẫn
MODEL_DIR = "serverAI/models/phobert_ner"  # Folder chứa model PyTorch đã train
OUTPUT_ONNX_PATH = "serverAI/models/nlu_onnx/ner_model.onnx"
OUTPUT_QUANT_PATH = "serverAI/models/nlu_onnx/ner_model.quant.onnx"

def export_to_onnx():
    print(f"[INFO] Loading model from {MODEL_DIR}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
        model.eval()
    except Exception as e:
        print(f"[ERROR] Không tìm thấy model PyTorch. Hãy chạy 'python training/train_phobert.py' trước.\nLỗi: {e}")
        return

    # Tạo thư mục output
    Path(OUTPUT_ONNX_PATH).parent.mkdir(parents=True, exist_ok=True)

    # 1. Export sang ONNX
    print("[INFO] Exporting to ONNX...")
    dummy_input = "hôm nay ăn gì"
    inputs = tokenizer(dummy_input, return_tensors="pt")
    
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        OUTPUT_ONNX_PATH,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
    )
    print(f"[SUCCESS] Saved raw ONNX to {OUTPUT_ONNX_PATH}")

    # 2. Quantization (Nén model int8 để chạy nhanh trên CPU)
    print("[INFO] Quantizing model (Float32 -> Int8)...")
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    quantize_dynamic(
        model_input=OUTPUT_ONNX_PATH,
        model_output=OUTPUT_QUANT_PATH,
        weight_type=QuantType.QUInt8,
    )
    print(f"[SUCCESS] Saved QUANTIZED ONNX to {OUTPUT_QUANT_PATH}")
    print("=> Hãy dùng file .quant.onnx cho Production!")

    # Lưu luôn tokenizer vào folder onnx để tiện loading
    tokenizer.save_pretrained(Path(OUTPUT_ONNX_PATH).parent)

if __name__ == "__main__":
    export_to_onnx()