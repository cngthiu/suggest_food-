import sys
import os
import json
import numpy as np

# Thêm đường dẫn gốc để import modules
sys.path.append(os.getcwd())

try:
    from serverAI.inference.pipeline import Pipeline, apply_slot_constraints, score_candidates
    from serverAI.inference.utils import norm_text
except ImportError:
    print("❌ Lỗi: Hãy chạy script từ thư mục gốc của dự án (python serverAI/tools/debug_query.py)")
    sys.exit(1)

# --- CẤU HÌNH TEST ---
QUERY = "Nấu gì với gà, cho 3 người ăn trong 20 phút"
# Một món gà có trong data mà lẽ ra phải được chọn (đã tạo ở bước generate_30_recipes)
EXPECTED_ID = "uc-ga-ap-chao-15p" 

def main():
    print(f"\n🔍 DEBUGGING QUERY: \"{QUERY}\"\n")

    # 1. Load Pipeline
    print("⏳ [1] Đang khởi tạo Pipeline...")
    try:
        pipe = Pipeline("serverAI/config/app.yaml")
        print("   ✅ Pipeline loaded.")
    except Exception as e:
        print(f"   ❌ Lỗi load pipeline: {e}")
        return

    # 2. Kiểm tra xem món Gà có trong kho dữ liệu chưa
    print(f"\n📦 [2] Kiểm tra món mục tiêu: '{EXPECTED_ID}'")
    if EXPECTED_ID in pipe.retriever.recipes:
        r = pipe.retriever.recipes[EXPECTED_ID]
        print(f"   ✅ Có trong kho dữ liệu.")
        print(f"      Title: {r['title']}")
        print(f"      Time: {r['cook_time']}p | Ingredients: {[i['name'] for i in r['ingredients']]}")
    else:
        print(f"   ❌ KHÔNG TÌM THẤY '{EXPECTED_ID}' trong Index.")
        print("      👉 Nguyên nhân: Bạn chưa chạy 'python serverAI/features/build_index.py' sau khi tạo món mới.")
        print("      👉 Giải pháp: Chạy lại build_index.py ngay.")
        return

    # 3. Kiểm tra NLU
    print(f"\n🧠 [3] Kiểm tra NLU (Hiểu ý định)")
    slots = pipe.nlu.extract_slots(QUERY)
    print(f"   Slots: {slots}")
    if slots.get('protein') != 'ga':
        print("   ⚠️  Cảnh báo: NLU không bắt được 'ga'. Kiểm tra lại training NER.")
    else:
        print("   ✅ NLU hoạt động tốt.")

    # 4. Kiểm tra Điểm số thô (Raw Scores)
    print(f"\nww [4] Kiểm tra điểm số thô của '{EXPECTED_ID}' với Query")
    
    # Tính điểm BM25 & Semantic thủ công
    try:
        # Lấy index của doc
        doc_idx = pipe.retriever.recipe_ids.index(EXPECTED_ID)
        
        # Tokenize query
        q_toks = pipe.retriever._tokenize(norm_text(QUERY))
        # Get BM25
        bm25_score = pipe.retriever.bm25.get_scores(q_toks)[doc_idx]
        
        # Get Semantic
        q_emb = pipe.retriever.embedder.encode([norm_text(QUERY)], normalize_embeddings=True)[0]
        d_emb = pipe.retriever.emb[doc_idx]
        sem_score = np.dot(q_emb, d_emb)
        
        print(f"   - BM25 Score (Từ khóa): {bm25_score:.4f} (Cao > 3.0 là tốt)")
        print(f"   - Semantic Score (Ngữ nghĩa): {sem_score:.4f} (Cao > 0.5 là tốt)")
        
        if bm25_score < 1.0 and sem_score < 0.4:
            print("   ⚠️  Điểm quá thấp. Có thể do từ khóa trong recipe không khớp với 'gà'/'15 phút'.")
            
    except ValueError:
        print("   ❌ Lỗi: ID không khớp trong Index (Cần chạy lại build_index.py)")
        return

    # 5. Chạy Retrieval thực tế
    print(f"\n🔎 [5] Chạy Retrieval (Tìm kiếm thô)")
    # Lấy top 50 ứng viên thô
    pipe.retriever.hy_cfg['k_total'] = 100 # Mở rộng để debug
    cands = pipe.retriever.retrieve(QUERY)
    
    found_at = -1
    for idx, c in enumerate(cands):
        if c['id'] == EXPECTED_ID:
            found_at = idx
            break
            
    if found_at != -1:
        print(f"   ✅ Tìm thấy '{EXPECTED_ID}' ở vị trí thứ {found_at + 1} trong danh sách thô.")
    else:
        print(f"   ❌ KHÔNG tìm thấy '{EXPECTED_ID}' trong Top 100 ứng viên thô.")
        print("      👉 Nguyên nhân: Retrieval Model (BM25/Embedding) thấy món này không liên quan.")
        return

    # 6. Kiểm tra Ranking/Filter
    print(f"\n⚖️  [6] Kiểm tra Xếp hạng & Lọc (Ranking)")
    
    # Giả lập danh sách chỉ gồm món Gà (đúng) và món Mực (sai - đang bị lên top)
    WRONG_ID = "muc-hap-hanh-15p" # Món sai mà hệ thống đang trả về
    
    debug_cands = []
    # Lấy object món Gà
    target_c = next((c for c in cands if c['id'] == EXPECTED_ID), None)
    if target_c: debug_cands.append(target_c)
    
    # Lấy object món Mực (nếu có trong ds tìm kiếm)
    wrong_c = next((c for c in cands if c['id'] == WRONG_ID), None)
    if wrong_c: debug_cands.append(wrong_c)
    
    if not debug_cands:
        print("   (Không lấy được candidate để so sánh)")
        return

    # Áp dụng constraints
    apply_slot_constraints(debug_cands, slots)
    
    # Chấm điểm
    if pipe.ranker:
        print("   🤖 Đang dùng: AI Ranker (LightGBM)")
        ranked = pipe._rank_with_lgbm(debug_cands)
    else:
        print("   rule-based Đang dùng: Rule-based Ranking")
        ranked = score_candidates(debug_cands, pipe.cfg)

    print(f"\n   {'ID':<25} | {'Score':<8} | {'ProteinFit':<10} | {'TimeFit':<8}")
    print("-" * 60)
    for c in ranked:
        p_fit = c.get('protein_fit', 0)
        t_fit = c.get('time_fit', 0)
        sc = c.get('score', 0)
        print(f"   {c['id']:<25} | {sc:.4f}   | {p_fit:<10} | {t_fit:<8}")

    # KẾT LUẬN
    top_id = ranked[0]['id']
    if top_id == EXPECTED_ID:
        print(f"\n✅ Kết quả Debug: Hệ thống ĐÚNG. Món '{EXPECTED_ID}' đang đứng đầu.")
    else:
        print(f"\n❌ Kết quả Debug: Hệ thống SAI. Món '{top_id}' đang đứng đầu.")
        if pipe.ranker:
            print("   👉 Nguyên nhân: Ranker (LightGBM) đang học sai. Nó chấm điểm món sai cao hơn dù ProteinFit thấp.")
            print("   👉 Giải pháp tạm thời: Tắt Ranker (đổi tên file lgbm.txt) để dùng Rule-based.")

if __name__ == "__main__":
    main()