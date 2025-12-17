# File: inference/pipeline.py
"""End-to-end inference pipeline: NLU → Context DST → Retrieval → Ranking → Response."""

import os, json, pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import numpy as np
import yaml
from .nlu_engine import NLU
from .utils import norm_text, load_json, load_recipes_any, recipes_to_dict, normalize_ingredient_mappings
from underthesea import word_tokenize

# --- Helper Functions (Giữ nguyên) ---
def vi_tokenize(s: str) -> List[str]:
    if not s: return []
    return word_tokenize(norm_text(s), format='text').split()

def resolve_encoder_device() -> str:
    pref = os.environ.get("SENTENCE_EMB_DEVICE", "auto").lower()
    if pref != "auto": return pref
    try:
        import torch
        if torch.cuda.is_available(): return "cuda"
    except Exception: pass
    return "cpu"

def recipe_matches_protein(recipe: Dict[str, Any], protein: str) -> bool:
    if not protein: return True
    target = norm_text(protein, lowercase=True, strip_dia=True)
    if not target: return True
    title = norm_text(recipe.get("title", ""), lowercase=True, strip_dia=True)
    if target in title: return True
    for tag in recipe.get("tags", []):
        if target in norm_text(str(tag), lowercase=True, strip_dia=True): return True
    for ing in recipe.get("ingredients", []):
        names = [ing.get("name", "")]
        names.extend(ing.get("aliases", []))
        for name in names:
            if target in norm_text(name, lowercase=True, strip_dia=True): return True
    return False

def _load_mapping_fallback(mapping_dir: str) -> Any:
    for fn in ["ingredient_mappings.json", "ingredient_mapping.json"]:
        fp = os.path.join(mapping_dir, fn)
        if Path(fp).exists():
            return load_json(fp)
    raise FileNotFoundError(f"No ingredient mapping file found in `{mapping_dir}`")

# --- Classes (Retriever, CartMapper, utils) giữ nguyên logic cũ ---
# (Để tiết kiệm không gian, tôi chỉ viết lại class Pipeline và các hàm cần thiết, 
# các class Retriever, CartMapper bạn giữ nguyên như file gốc).

class Retriever:
    def __init__(
        self,
        cfg: Dict[str, Any],
        ingredient_map: Optional[Dict[str, Any]] = None,
        product_catalog: Optional[Dict[str, Any]] = None,
        recipes: Optional[Dict[str, Any]] = None,
        recipe_fetcher: Optional[Callable[[List[str]], Dict[str, Any]]] = None,
    ):
        self.cfg = cfg
        paths = cfg["paths"]
        for key in ["bm25_path", "corpus_path", "ids_path"]:
            if not Path(paths[key]).exists():
                raise FileNotFoundError(
                    f"Missing artifact `{key}` at `{paths[key]}`. "
                    "Bạn cần chạy `python -m serverAI.features.build_index --config serverAI/config/app.yaml` để build cache."
                )

        with open(paths["bm25_path"], "rb") as f:
            self.bm25 = pickle.load(f)
        with open(paths["corpus_path"], "rb") as f:
            self.corpus_tokens = pickle.load(f)
        with open(paths["ids_path"], "r", encoding="utf-8") as f:
            self.recipe_ids = json.load(f)["recipe_ids"]

        self.semantic_enabled = False
        self.faiss = None
        self.emb = None
        self.embedder = None

        if recipes is not None:
            self.recipes = recipes
        elif recipe_fetcher is not None:
            self.recipes = recipe_fetcher([str(rid) for rid in self.recipe_ids])
        else:
            self.recipes = self._load_recipes(paths["recipes_dir"])

        raw_mapping = ingredient_map
        if not raw_mapping:
            raw_mapping = _load_mapping_fallback(paths["mapping_dir"])
        self.mapping = normalize_ingredient_mappings(raw_mapping)

        self.catalog = product_catalog or {}
        self.idx_cfg = cfg["indexing"]
        self.ret_cfg = cfg["retrieval"]
        self.hy_cfg = cfg["hybrid"]
        self.encoder_device = resolve_encoder_device()

        embedder_model = (self.ret_cfg.get("embedder_model") or "").strip()
        if embedder_model and Path(paths.get("faiss_index", "")).exists() and Path(paths.get("embed_matrix", "")).exists():
            try:
                import faiss
                from sentence_transformers import SentenceTransformer

                self.faiss = faiss.read_index(paths["faiss_index"])
                self.emb = np.load(paths["embed_matrix"], mmap_mode="r")
                self.embedder = SentenceTransformer(embedder_model, device=self.encoder_device)
                self.semantic_enabled = True
            except Exception as e:
                print(f"[WARN] Semantic retrieval disabled: {e}")

    def _tokenize(self, s: str) -> List[str]: return vi_tokenize(s)
    def _load_recipes(self, recipes_dir: str) -> Dict[str, Any]:
        recipes = load_recipes_any(recipes_dir)
        return recipes_to_dict(recipes)
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        # ... (Giữ nguyên logic retrieve cũ) ...
        s = norm_text(query, lowercase=True, strip_dia=True)
        toks = self._tokenize(s)
        bm25_scores = self.bm25.get_scores(toks)
        k_bm25 = self.hy_cfg["k_bm25"]
        top_bm25_idx = np.argsort(bm25_scores)[::-1][:k_bm25]

        q_emb = None
        I = np.array([], dtype=np.int64)
        if self.semantic_enabled and self.embedder is not None and self.faiss is not None:
            q_emb = self.embedder.encode([s], normalize_embeddings=self.ret_cfg.get("normalize_embeddings", True))
            _, I = self.faiss.search(np.array(q_emb, dtype=np.float32), self.hy_cfg["k_emb"])
            I = I[0]

        candidate_indices: List[int] = []
        seen = set()
        cap = int(self.hy_cfg.get("k_total", 0))
        def add_indices(indices):
            for idx in indices:
                if idx < 0 or idx in seen: continue
                candidate_indices.append(int(idx))
                seen.add(int(idx))
                if cap and len(candidate_indices) >= cap: return True
            return False
        if not add_indices(top_bm25_idx.tolist()):
            add_indices(I.tolist())
        cand = []
        for idx in candidate_indices:
            rid = self.recipe_ids[idx]
            recipe = self.recipes.get(str(rid))
            if not recipe:
                continue
            semantic = 0.0
            if self.semantic_enabled and q_emb is not None and self.emb is not None:
                semantic = float(np.dot(q_emb[0], self.emb[idx]))
            cand.append(
                {
                    "id": rid,
                    "bm25": float(bm25_scores[idx]),
                    "semantic": semantic,
                    "recipe": recipe,
                }
            )
        if not cand: return []
        bm_vals = np.array([c["bm25"] for c in cand])
        sem_vals = np.array([c["semantic"] for c in cand])
        def mm(x): 
            if x.size == 0: return x
            return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-9)
        bm_n = mm(bm_vals)
        sem_n = mm(sem_vals)
        out = []
        for i, c in enumerate(cand):
            out.append({
                "id": c["id"], "bm25_n": float(bm_n[i]), "semantic_n": float(sem_n[i]),
                "time_fit": 1.0, "diet_fit": 1.0, "availability_ratio": self._availability_ratio(c["recipe"]),
                "promo_coverage": 0.0, "history_affinity": 0.0, "recipe": c["recipe"]
            })
        return out
    def _availability_ratio(self, recipe):
        # ... (Giữ nguyên logic cũ) ...
        need = recipe.get("ingredients", [])
        if not need: return 0.0
        total_weight, score = 0, 0
        for ing in need:
            weight = 3.0 if ing.get("type") == "main" else 1.0
            total_weight += weight
            mappings = self.mapping.get(ing.get("name", "").strip(), [])
            is_in_stock = False
            if self.catalog:
                for m in mappings:
                    if self.catalog.get(m.get("sku"), {}).get("stock", 0) > 0:
                        is_in_stock = True; break
            elif mappings: is_in_stock = True
            if is_in_stock: score += weight
        return score / total_weight if total_weight > 0 else 0.0

def normalize_to_base_unit(qty: float, unit: str) -> tuple[float, str]:
    # ... (Giữ nguyên logic cũ) ...
    unit = unit.lower().strip()
    if unit in ["kg", "kilogram", "kilo"]: return qty * 1000.0, "g"
    if unit in ["g", "gr", "gram", "gam"]: return qty, "g"
    if unit in ["l", "lit", "lít", "liter"]: return qty * 1000.0, "ml"
    if unit in ["ml", "mililit"]: return qty, "ml"
    if unit in ["quả", "trái", "trứng", "cái"]: return qty, "quả"
    return qty, unit

class CartMapper:
    # ... (Giữ nguyên code class CartMapper từ file gốc) ...
    def __init__(self, cfg, ingredient_map=None, product_catalog=None):
        self.cfg = cfg
        self.map = normalize_ingredient_mappings(ingredient_map or {})
        self.catalog = product_catalog or {}
    def suggest_cart(self, recipe, servings):
        # ... (Giữ nguyên logic cũ) ...
        items, total = [], 0
        scale = float(servings) / (float(recipe.get("servings", 1)) or 1.0)
        for ing in recipe.get("ingredients", []):
            name = ing.get("name")
            qty_raw = float(ing.get("qty", 0) or 0) * scale
            unit_raw = ing.get("unit", "")
            qty, unit = normalize_to_base_unit(qty_raw, unit_raw)

            mapping_entries = self.map.get(name) or []
            prod = None
            chosen_mapping = None
            for m in mapping_entries:
                p = self.catalog.get(m.get("sku"))
                if not p:
                    continue
                if p.get("stock", 0) <= 0:
                    continue
                prod = p
                chosen_mapping = m
                break

            if (qty <= 0) and chosen_mapping:
                ratio = (chosen_mapping.get("ratio_per_serving") or {})
                try:
                    ratio_qty = float(ratio.get("qty", 0) or 0) * float(servings)
                    ratio_unit = ratio.get("unit") or unit
                    if ratio_qty > 0:
                        qty, unit = normalize_to_base_unit(ratio_qty, ratio_unit)
                except Exception:
                    pass
            if not prod:
                items.append({"ingredient": name, "sku": None, "name": f"{name} (N/A)", "price": 0, "subtotal": 0, "is_missing": True})
                continue

            prod_unit = (prod.get("measureUnit") or "").lower().strip()
            unit_size = float(prod.get("unitSize", 1) or 1)
            if qty <= 0 or unit_size <= 0:
                pkgs = 1
            elif prod_unit and (prod_unit == unit):
                pkgs = int(np.ceil(qty / unit_size))
            else:
                pkgs = 1

            subtotal = pkgs * float(prod.get("price", 0))
            total += subtotal
            items.append({"ingredient": name, "sku": prod.get("sku"), "name": prod.get("name"), "packages": pkgs, "price": prod.get("price"), "subtotal": subtotal, "stock_ok": True})
        return {"items": items, "estimated": total, "currency": "VND", "notes": []}

def apply_slot_constraints(cand: List[Dict[str, Any]], slots: Dict[str, Any]) -> None:
    # ... (Giữ nguyên logic cũ) ...
    for c in cand:
        r = c["recipe"]
        tfit = 1.0
        if slots.get("time") and r.get("cook_time"):
            try:
                import re
                m = re.search(r"(\d+)", str(slots["time"]))
                if m:
                    limit = int(m.group(1))
                    t = int(r.get("cook_time", 999))
                    tfit = max(0.0, 1.0 - max(0, t - limit)/max(1.0, float(limit)))
            except: pass
        c["time_fit"] = float(tfit)
        c["diet_fit"] = 1.0 if not slots.get("diet") else (1.0 if slots["diet"] in r.get("diet", []) else 0.0)
        c["protein_fit"] = 1.0 if not slots.get("protein") else (1.0 if recipe_matches_protein(r, slots["protein"]) else 0.0)
        # Servings fit logic simplified
        c["servings_fit"] = 1.0

def score_candidates(cand: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    # ... (Giữ nguyên logic cũ) ...
    w = cfg.get("ranking", {})
    out = []
    for c in cand:
        s = (w.get("w_semantic", 0.35)*c.get("semantic_n", 0) + w.get("w_timefit", 0.2)*c.get("time_fit", 1) + 
             w.get("w_dietfit", 0.15)*c.get("diet_fit", 1) + w.get("w_protein", 0.2)*c.get("protein_fit", 0) + 
             w.get("w_avail", 0.05)*c.get("availability_ratio", 0))
        out.append({**c, "score": float(s)})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out

# --- UPDATED PIPELINE CLASS STARTS HERE ---
class Pipeline:
    def __init__(
        self,
        cfg_path: str = "serverAI/config/app.yaml",
        product_catalog: Optional[Dict[str, Any]] = None,
        ingredient_map: Optional[Dict[str, Any]] = None,
        recipes: Optional[Dict[str, Any]] = None,
        recipe_fetcher: Optional[Callable[[List[str]], Dict[str, Any]]] = None,
    ):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
        paths = self.cfg["paths"]
        self.catalog = product_catalog or {}
        
        if ingredient_map:
            print(f"[INFO] Pipeline: Using Dynamic Map ({len(ingredient_map)} keys)")
            raw_map = ingredient_map
        else:
            print("[WARN] Pipeline: Fallback to Static JSON Map")
            raw_map = _load_mapping_fallback(paths["mapping_dir"])
        self.map_data = normalize_ingredient_mappings(raw_map)

        gaz_dir = "serverAI/data/nlu/gazetteer"
        self.nlu = NLU(self.cfg["paths"]["nlu_model_dir"], gaz_dir)
        self.retriever = Retriever(
            self.cfg,
            ingredient_map=self.map_data,
            product_catalog=self.catalog,
            recipes=recipes,
            recipe_fetcher=recipe_fetcher,
        )
        self.mapper = CartMapper(self.cfg, ingredient_map=self.map_data, product_catalog=self.catalog)
        self.ranker = None
        self.rank_features: List[str] = []
        self._load_ranker(paths.get("ranker_dir"))

    # --- HÀM QUERY ĐÃ ĐƯỢC NÂNG CẤP ---
    def query(self, text: str, context: Dict[str, Any] = None, top_k: int = 5) -> Dict[str, Any]:
        """
        Xử lý truy vấn có ngữ cảnh (Context-Aware).
        Logic: 
        1. NLU trích xuất Intent/Slots mới.
        2. Quyết định Merge vào Context cũ hay Reset Context.
        3. Làm giàu truy vấn (Query Enrichment) nếu cần.
        4. Trả về response kèm context mới.
        """
        if context is None:
            context = {}

        # 1. NLU Prediction
        intent = self.nlu.predict_intent(text)
        new_slots = self.nlu.extract_slots(text)
        
        # 2. Xử lý logic Greeting & Goodbye sớm
        if intent["name"] in ["greeting", "greeting_hello", "smalltalk_greeting"]:
            return {
                "intents": [intent],
                "slots": context.get("slots", {}), # Giữ nguyên slot cũ phòng khi user quay lại
                "candidates": [],
                "explanations": ["Xin chào! Mình là trợ lý ẩm thực thông minh. Bạn muốn tìm món ăn gì hôm nay?"]
            }

        if intent["name"] in ["goodbye"]:
            # Có thể clear context ở đây nếu muốn session kết thúc
            return {
                "intents": [intent],
                "slots": {}, # Reset slots khi tạm biệt
                "candidates": [],
                "explanations": ["Tạm biệt bạn! Hẹn gặp lại trong bữa ăn tới nhé."]
            }

        # 3. Dialogue State Tracking (Context Merging)
        current_slots = context.get("slots", {}).copy()
        merged_slots = {}
        intent_name = intent["name"]

        # Strategy: INTENT SCOPING
        # Nhóm "New Search": User muốn tìm món mới -> Reset các slot món ăn, giữ lại diet/allergy (nếu có profile)
        if intent_name in ["suggest_food", "suggest_food_by_ingredient"]:
            merged_slots = new_slots
            # (Optional) Giữ lại diet/allergy nếu user đã set trước đó
            if "diet" in current_slots and "diet" not in merged_slots:
                merged_slots["diet"] = current_slots["diet"]
            if "allergy" in current_slots:
                merged_slots["allergy"] = current_slots["allergy"]

        # Nhóm "Refine": User hỏi chi tiết (giá, thời gian) -> Merge slot mới vào cũ
        elif intent_name in ["suggest_food_by_time", "ask_price", "filter_diet"]:
            merged_slots = current_slots
            for k, v in new_slots.items():
                if v: merged_slots[k] = v
        
        # Nhóm khác: Mặc định giữ context cũ
        else:
            merged_slots = current_slots

        # 4. Query Enrichment (Quan trọng cho Retrieval)
        # Nếu câu query ngắn (vd: "giá bao nhiêu"), ta cần nối thêm tên món từ context (vd: "gà")
        # để bộ tìm kiếm BM25/Embedding hoạt động đúng.
        search_query = text
        if intent_name in ["suggest_food_by_time", "ask_price", "filter_diet"]:
            relevant_context = []
            if "protein" in merged_slots: relevant_context.append(merged_slots["protein"])
            # Có thể thêm các từ khóa khác nếu cần
            
            if relevant_context and len(text.split()) < 4: # Chỉ rewrite nếu câu hỏi ngắn
                context_str = " ".join(relevant_context)
                search_query = f"{context_str} {text}"
                print(f"[INFO] Rewritten Query: '{text}' -> '{search_query}'")

        # 5. Retrieval & Ranking
        cands = self.retriever.retrieve(search_query)
        apply_slot_constraints(cands, merged_slots)
        
        if self.ranker:
            ranked = self._rank_with_lgbm(cands)[:top_k]
        else:
            ranked = score_candidates(cands, self.cfg)[:top_k]
        
        # 6. Format Output
        out_cand = []
        for c in ranked:
            r = c["recipe"]
            out_cand.append({
                "id": r.get("id"),
                "title": r.get("title"),
                "summary": r.get("summary"),
                "image": r.get("image"),
                "cook_time": r.get("cook_time"),
                "servings": r.get("servings"),
                "score": c.get("score"),
                "score_factors": {
                    "semantic": c.get("semantic_n"),
                    "time_fit": c.get("time_fit"),
                    "availability": c.get("availability_ratio")
                }
            })
            
        # Tùy chỉnh câu trả lời dựa trên intent
        explanation = "Dưới đây là các gợi ý phù hợp nhất cho bạn."
        if intent_name == "ask_price":
            explanation = "Dưới đây là các món ăn kèm thông tin nguyên liệu. Bạn có thể chọn 'Suggest Cart' để xem giá chi tiết."
        elif not out_cand:
            explanation = "Tiếc quá, mình chưa tìm thấy món nào phù hợp với yêu cầu này. Bạn thử nguyên liệu khác xem sao?"

        return {
            "intents": [intent],
            "slots": merged_slots, # Trả về context mới để Client lưu
            "candidates": out_cand,
            "explanations": [explanation]
        }

    def suggest_cart(self, recipe_id: str, servings: int = 2) -> Dict[str, Any]:
        r = self.retriever.recipes.get(recipe_id)
        if not r: raise KeyError("recipe not found")
        cart = self.mapper.suggest_cart(r, servings)
        return {
            "recipe_id": recipe_id,
            "servings": servings,
            "items": cart["items"],
            "totals": {"estimated": cart["estimated"], "currency": cart["currency"]},
            "notes": cart["notes"]
        }

    def _load_ranker(self, ranker_dir: Optional[str]) -> None:
        if not ranker_dir: return
        model_path = Path(ranker_dir)/"lgbm.txt"
        if not model_path.exists(): return
        try:
            import lightgbm as lgb
            self.ranker = lgb.Booster(model_file=str(model_path))
            self.rank_features = ["semantic_n", "bm25_n", "time_fit", "diet_fit", "protein_fit", "servings_fit", "availability_ratio"]
            print(f"[INFO] Loaded ranker from {model_path}")
        except Exception as e:
            print(f"[WARN] Failed to load ranker: {e}")
            self.ranker = None

    def _rank_with_lgbm(self, cand: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not cand or not self.ranker: return score_candidates(cand, self.cfg)
        feature_alias = {
            "semantic_sim": "semantic_n", "semantic": "semantic_n", "bm25": "bm25_n", "bm25_n": "bm25_n",
            "time_fit": "time_fit", "diet_fit": "diet_fit", "protein_fit": "protein_fit", 
            "servings_fit": "servings_fit", "availability": "availability_ratio", "availability_ratio": "availability_ratio",
            "promo_coverage": "promo_coverage", "history_affinity": "history_affinity"
        }
        feats = []
        for c in cand:
            vec = [float(c.get(feature_alias.get(name, name), 0.0)) for name in self.rank_features]
            feats.append(vec)
        try:
            scores = self.ranker.predict(np.asarray(feats, dtype=np.float32))
            out = []
            for c, s in zip(cand, scores):
                out.append({**c, "score": float(s)})
            out.sort(key=lambda x: x["score"], reverse=True)
            return out
        except Exception:
            return score_candidates(cand, self.cfg)
