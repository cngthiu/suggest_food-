# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv

from serverAI.inference.pipeline import Pipeline

load_dotenv()

app = FastAPI(title="SmartShop AI")

# Warm-up
PIPELINE = None

@app.on_event("startup")
async def startup_event():
    global PIPELINE
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
