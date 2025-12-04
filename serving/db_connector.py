# File: serverAI/serving/db_connector.py
import pymongo
import re
from typing import Dict, Any

class ProductDatabase:
    def __init__(self, uri: str, db_name: str):
        self.client = pymongo.MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db["products"] 

    def _parse_unit_size(self, name: str, unit: str) -> float:
        """Hàm phụ trợ: Tự đoán trọng lượng nếu DB thiếu trường unitSize"""
        # Ưu tiên tìm trong tên (VD: ... 500g)
        text = (name + " " + unit).lower()
        
        # Tìm kg
        m_kg = re.search(r"(\d+(\.\d+)?)(\s?)kg", text)
        if m_kg: return float(m_kg.group(1))
        
        # Tìm g (gam)
        m_g = re.search(r"(\d+)(\s?)g", text)
        if m_g: return float(m_g.group(1)) / 1000.0
        
        # Tìm ml/lít
        m_l = re.search(r"(\d+(\.\d+)?)(\s?)l", text)
        if m_l: return float(m_l.group(1))
        m_ml = re.search(r"(\d+)(\s?)ml", text)
        if m_ml: return float(m_ml.group(1)) / 1000.0
        
        return 1.0 # Mặc định 1 đơn vị

    def get_full_mapping_logic(self):
        """Lấy logic mapping từ DB"""
        mapping = {}
        # Nếu chưa có collection ingredient_mappings, trả về rỗng để code không crash
        if "ingredient_mappings" not in self.db.list_collection_names():
            return {}

        cursor = self.db["ingredient_mappings"].find({"status": "active"})
        for doc in cursor:
            main_key = doc.get("ingredient_key")
            if not main_key: continue
            
            logic = doc.get("logic", {})
            data = {
                "sku": doc.get("sku_ref"),
                "ratio_per_serving": {"qty": logic.get("ratio_per_serving", 100), "unit": logic.get("ratio_unit", "g")},
            }
            mapping[main_key] = data
            # Map alias
            for alias in doc.get("aliases", []):
                mapping[alias] = data
        return mapping

    def get_product_catalog(self) -> Dict[str, Any]:
        """
        Load sản phẩm khớp với cấu trúc thực tế:
        - id: dùng field 'sku'
        - price: dùng field 'price' (int)
        """
        catalog = {}
        try:
            cursor = self.collection.find({})
            for doc in cursor:
                # 1. CẤU TRÚC THỰC TẾ: Lấy 'sku' làm khóa chính
                sku = doc.get("sku")
                if not sku:
                    # Fallback phòng hờ dữ liệu cũ
                    sku = doc.get("reference_id") 
                
                if sku:
                    # 2. Xử lý Unit Size (Quan trọng để tính số lượng)
                    u_size = doc.get("unitSize")
                    if not u_size:
                        u_size = self._parse_unit_size(doc.get("name", ""), doc.get("unit", ""))

                    catalog[sku] = {
                        "sku": sku,
                        "name": doc.get("name", ""),
                        "price": float(doc.get("price", 0)), # Lấy đúng trường price
                        "stock": int(doc.get("stock", 0)),
                        "unit": doc.get("unit", "gói"),
                        "unitSize": u_size, 
                        "image": doc.get("image", [])
                    }
            print(f"[DB] Loaded {len(catalog)} products from MongoDB.")
            return catalog
        except Exception as e:
            print(f"[DB] Error loading catalog: {e}")
            return {}