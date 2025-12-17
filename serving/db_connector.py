# File: serverAI/serving/db_connector.py
import pymongo
from typing import Dict, Any, List, Optional

class ProductDatabase:
    def __init__(self, uri: str, db_name: str, products_collection: str = "products"):
        self.client = pymongo.MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[products_collection]

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

    def _resolve_recipes_collection(self, preferred: Optional[str] = None) -> Optional[str]:
        names = set(self.db.list_collection_names())
        if preferred and preferred in names:
            return preferred
        for cand in ["recipes", "recipies", "recipe"]:
            if cand in names:
                return cand
        return None

    def get_recipes_dict(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Load toàn bộ recipes lên memory theo dict[id] để phục vụ inference.
        """
        col = self._resolve_recipes_collection(collection_name)
        if not col:
            print("[DB] No recipes collection found (expected: recipes/recipies/recipe).")
            return {}

        out: Dict[str, Any] = {}
        try:
            for doc in self.db[col].find({}):
                rid = doc.get("id") or doc.get("recipe_id") or doc.get("reference_id")
                if not rid and "_id" in doc:
                    rid = str(doc["_id"])
                if not rid:
                    continue
                doc.pop("_id", None)
                out[str(rid)] = doc
            print(f"[DB] Loaded {len(out)} recipes from `{col}`.")
            return out
        except Exception as e:
            print(f"[DB] Error loading recipes: {e}")
            return {}

    def get_recipes_by_ids(self, ids: List[str], collection_name: Optional[str] = None) -> Dict[str, Any]:
        col = self._resolve_recipes_collection(collection_name)
        if not col:
            return {}
        if not ids:
            return {}
        wanted = [str(x) for x in ids if x]
        try:
            cursor = self.db[col].find({"id": {"$in": wanted}})
            out: Dict[str, Any] = {}
            for doc in cursor:
                rid = doc.get("id") or doc.get("recipe_id") or doc.get("reference_id")
                if not rid and "_id" in doc:
                    rid = str(doc["_id"])
                if not rid:
                    continue
                doc.pop("_id", None)
                out[str(rid)] = doc
            return out
        except Exception:
            return {}
