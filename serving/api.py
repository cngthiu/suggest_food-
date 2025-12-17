# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv

from serverAI.inference.pipeline import Pipeline
from serverAI.serving.db_connector import ProductDatabase

load_dotenv()

app = FastAPI(title="SmartShop AI Context-Aware")

# Global Pipeline Instance
PIPELINE = None

@app.on_event("startup")
async def startup_event():
    global PIPELINE
    try:
        mongo_uri = os.getenv("MONGO_URI", "mongodb+srv://thieulk23:thieulk23@cluster0.es7pd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
        db_name = os.getenv("DBNAME", "test")
        products_col = os.getenv("PRODUCTS_COLLECTION", "products")
        recipes_col = os.getenv("RECIPES_COLLECTION")
        
        print("[INFO] Connecting to MongoDB...")
        db_conn = ProductDatabase(mongo_uri, db_name, products_collection=products_col)
        
        # Load real-time data
        live_catalog = db_conn.get_product_catalog()
        live_mapping = db_conn.get_full_mapping_logic()
        live_recipes = db_conn.get_recipes_dict(collection_name=recipes_col)
        
        print(f"[INIT] Catalog size: {len(live_catalog)}, Mapping size: {len(live_mapping)}, Recipes size: {len(live_recipes)}")
        
        PIPELINE = Pipeline(
            "serverAI/config/app.yaml",
            product_catalog=live_catalog,
            ingredient_map=live_mapping,
            recipes=(live_recipes or None),
        )
        print("[INIT] Pipeline Ready!")
    except Exception as e:
        print("[ERROR] Init failed:", e)

# --- Define Request Model với Context ---
class QueryReq(BaseModel):
    text: str
    context: Optional[Dict[str, Any]] = None # Client gửi lại context nhận được từ turn trước
    limits: Optional[Dict[str, Any]] = None

@app.post("/assistant/query")
async def assistant_query(req: QueryReq):
    if not PIPELINE:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    top_k = int((req.limits or {}).get("top_k", 5))
    
    # Lấy context từ request, nếu không có thì tạo mới
    client_context = req.context or {}
    
    try:
        # Truyền context vào hàm query mới cập nhật
        out = PIPELINE.query(req.text, context=client_context, top_k=top_k)
        
        # Output giờ đây sẽ chứa field "slots" đã được merge (Merged Slots)
        # Client CẦN lưu field này để gửi lại trong request tiếp theo
        return {**out, "latency_ms": None}
    except Exception as e:
        print(f"[ERROR] Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recipes/{rid}/suggest-cart")
async def suggest_cart(rid: str, servings: int = 2):
    if not PIPELINE:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    try:
        out = PIPELINE.suggest_cart(rid, servings)
        out["latency_ms"] = None
        return out
    except KeyError:
        raise HTTPException(status_code=404, detail="Recipe not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
