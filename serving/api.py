# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv

from serverAI.inference.pipeline import Pipeline
from serverAI.serving.db_connector import ProductDatabase
load_dotenv()

app = FastAPI(title="SmartShop AI")

# Warm-up
PIPELINE = None

@app.on_event("startup")
async def startup_event():
    global PIPELINE
    try:
        # --- BƯỚC 1: KẾT NỐI DB & LẤY CATALOG TẠI ĐÂY ---
        mongo_uri ="mongodb+srv://thieulk23:thieulk23@cluster0.es7pd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        db_name = "test"
        
        print("[INFO] Connecting to MongoDB...")
        db_conn = ProductDatabase(mongo_uri, db_name)
        live_catalog = db_conn.get_product_catalog()
        print(f"[INFO] Loaded {len(live_catalog)} products from DB.")
        
        # --- BƯỚC 2: TRUYỀN CATALOG VÀO PIPELINE (DEPENDENCY INJECTION) ---
        PIPELINE = Pipeline("serverAI/config/app.yaml", product_catalog=live_catalog)
        
    except Exception as e:
        print("[WARN] Pipeline init failed:", e)
        PIPELINE = None
    try:
        PIPELINE = Pipeline("serverAI/config/app.yaml")
    except Exception as e:
        print("[WARN] Pipeline init failed:", e)
        PIPELINE = None

class QueryReq(BaseModel):
    text: str
    context: Optional[dict] = None
    limits: Optional[dict] = None

@app.post("/assistant/query")
async def assistant_query(req: QueryReq):
    if not PIPELINE:
        raise HTTPException(status_code=503, detail={"error": "INDEX_NOT_READY", "message": "Build index or init pipeline failed"})
    top_k = int((req.limits or {}).get("top_k", 5))
    try:
        out = PIPELINE.query(req.text, top_k=top_k)
        return {**out, "latency_ms": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recipes/{rid}/suggest-cart")
async def suggest_cart(rid: str, servings: int = 2):
    if not PIPELINE:
        raise HTTPException(status_code=503, detail={"error": "INDEX_NOT_READY", "message": "Build index or init pipeline failed"})
    try:
        out = PIPELINE.suggest_cart(rid, servings)
        out["latency_ms"] = None
        return out
    except KeyError:
        raise HTTPException(status_code=404, detail="recipe not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
