import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import os

MODEL_DIR = "serverAI/models/intent_phobert"
OUTPUT_ONNX_PATH = "serverAI/models/nlu_onnx/intent_model.onnx"
OUTPUT_QUANT_PATH = "serverAI/models/nlu_onnx/intent_model.quant.onnx"

def main():
    print(f"[INFO] Exporting Intent Model from {MODEL_DIR}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.eval()
    except Exception as e:
        print(f"[ERROR] Load model failed: {e}")
        return

    Path(OUTPUT_ONNX_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    # Export
    dummy_input = tokenizer("hôm nay ăn gì", return_tensors="pt")
    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        OUTPUT_ONNX_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size"}
        },
        opset_version=14
    )
    
    # Quantize
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(OUTPUT_ONNX_PATH, OUTPUT_QUANT_PATH, weight_type=QuantType.QUInt8)
    print(f"[SUCCESS] Exported Intent ONNX to {OUTPUT_QUANT_PATH}")

    # Copy label_map.json sang folder onnx để NLU engine load
    import shutil
    src_label = os.path.join(MODEL_DIR, "label_map.json")
    dst_label = os.path.join(os.path.dirname(OUTPUT_ONNX_PATH), "intent_labels.json")
    if os.path.exists(src_label):
        shutil.copy(src_label, dst_label)
        print("[INFO] Copied label_map.json -> intent_labels.json")

if __name__ == "__main__":
    main()
