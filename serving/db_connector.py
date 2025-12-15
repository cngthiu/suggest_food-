# File: serverAI/serving/db_connector.py
import pymongo
from typing import Dict, Any

class ProductDatabase:
    def __init__(self, uri: str, db_name: str):
        self.client = pymongo.MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db["products"] 

    def get_full_mapping_logic(self):
        """Lấy logic mapping từ DB"""
        mapping = {}
        if "ingredient_mappings" not in self.db.list_collection_names():
            return {}

        cursor = self.db["ingredient_mappings"].find({"status": "active"})
        for doc in cursor:
            main_key = doc.get("ingredient_key")
            if not main_key: continue
            
            logic = doc.get("logic", {})
            data = {
                "sku": doc.get("sku_ref"),
                # Mặc định mapping trả về theo chuẩn g/ml/quả
                "ratio_per_serving": {"qty": logic.get("ratio_per_serving", 100), "unit": logic.get("ratio_unit", "g")},
            }
            mapping[main_key] = data
            for alias in doc.get("aliases", []):
                mapping[alias] = data
        return mapping

    def get_product_catalog(self) -> Dict[str, Any]:
        """
        Load sản phẩm với cấu trúc chuẩn: g, ml, quả.
        Sử dụng field 'net_weight' và 'measure_unit' từ DB.
        """
        catalog = {}
        try:
            cursor = self.collection.find({})
            for doc in cursor:
                sku = doc.get("sku") or doc.get("reference_id")
                
                if sku:
                    # Lấy trực tiếp trọng lượng tịnh và đơn vị đo từ DB
                    # Nếu thiếu, fallback về 1 đơn vị
                    net_weight = float(doc.get("net_weight", 1))
                    measure_unit = doc.get("measure_unit", "g").lower() # Chuẩn hóa về chữ thường
                    
                    # Logic chuẩn hóa đơn vị (nếu dữ liệu bẩn lọt vào)
                    if measure_unit in ["gram", "grams"]: measure_unit = "g"
                    if measure_unit in ["lit", "lít", "liter"]: 
                        # Nếu lỡ có lít, convert về ml ngay tại đây
                        measure_unit = "ml" 
                        net_weight *= 1000
                    
                    catalog[sku] = {
                        "sku": sku,
                        "name": doc.get("name", ""),
                        "price": float(doc.get("price", 0)),
                        "stock": int(doc.get("stock", 0)),
                        "unit": doc.get("unit", "gói"), # Đơn vị đóng gói (Hộp, Túi, Khay)
                        
                        # Hai trường quan trọng nhất để tính toán
                        "unitSize": net_weight, 
                        "measureUnit": measure_unit, 
                        
                        "image": doc.get("image", [])
                    }
            print(f"[DB] Loaded {len(catalog)} products. Standardized to g/ml/qua.")
            return catalog
        except Exception as e:
            print(f"[DB] Error loading catalog: {e}")
            return {}