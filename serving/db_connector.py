# serving/db_connector.py
import os
from pymongo import MongoClient
from typing import Dict, Any

class ProductDatabase:
    def __init__(self, uri: str, db_name: str):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db["products"]

    def get_product_catalog(self) -> Dict[str, Any]:
        """
        Lấy toàn bộ sản phẩm active để cache vào RAM.
        Output format: { "sku1": {price: 100, stock: 5, name: "..."}, ... }
        """
        cursor = self.collection.find({"status": "active"}) # Chỉ lấy sp đang bán
        catalog = {}
        for doc in cursor:
            sku = doc.get("sku")
            if sku:
                catalog[sku] = {
                    "sku": sku,
                    "name": doc.get("name"),
                    "price": doc.get("price_vnd", 0), # Giá hiện tại
                    "stock": doc.get("inventory_qty", 0), # Tồn kho
                    "unit": doc.get("unit", "cái"),
                    "unitSize": doc.get("net_weight", 1), # VD: 500 (gram)
                    "image": doc.get("thumbnail_url", "")
                }
        return catalog