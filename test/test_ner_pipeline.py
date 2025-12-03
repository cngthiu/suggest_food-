# tools/check_db.py
import os
import sys
from pymongo import MongoClient
from dotenv import load_dotenv

# Load biến môi trường
load_dotenv()

def inspect_db():
    uri = "mongodb+srv://thieulk23:thieulk23@cluster0.es7pd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    db_name= "test"
    col_name = "products"

    print(f"--- KẾT NỐI ĐẾN: {db_name}.{col_name} ---")
    
    try:
        client = MongoClient(uri)
        db = client[db_name]
        
        # 1. Kiểm tra collection có tồn tại không
        cols = db.list_collection_names()
        if col_name not in cols:
            print(f"[LỖI] Không tìm thấy collection '{col_name}' trong database '{db_name}'.")
            print(f"   -> Các collection hiện có: {cols}")
            return

        col = db[col_name]
        count = col.count_documents({})
        print(f"[OK] Tìm thấy {count} documents.")

        if count == 0:
            print("[CẢNH BÁO] Collection rỗng!")
            return

        # 2. Lấy mẫu 1 document để xem tên trường
        sample = col.find_one()
        print("\n--- CẤU TRÚC DỮ LIỆU MẪU (1 SẢN PHẨM) ---")
        for key, val in sample.items():
            print(f"  - {key}: {type(val).__name__}  (Ví dụ: {str(val)[:50]}...)")

        print("\n-------------------------------------------")
        print("HÃY DÙNG TÊN CÁC TRƯỜNG Ở TRÊN ĐỂ CẬP NHẬT VÀO db_connector.py")

    except Exception as e:
        print(f"[LỖI KẾT NỐI] {e}")

if __name__ == "__main__":
    inspect_db()