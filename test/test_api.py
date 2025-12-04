import requests
import json

# Đảm bảo server đang chạy (uvicorn serverAI.serving.api:app --reload)
BASE_URL = "http://localhost:8000"

def test_query(text):
    print(f"\n🔹 Testing Query: '{text}'")
    url = f"{BASE_URL}/assistant/query"
    payload = {
        "text": text,
        "limits": {"top_k": 3}
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # In kết quả NLU
        intents = data.get("intents", [])
        intent_name = intents[0].get("name", "Unknown") if intents else "Unknown"
        
        print(f"   [NLU] Intent: {intent_name}")
        print(f"   [NLU] Slots: {data.get('slots')}")
        
        # Nếu bot trả lời (chào hỏi) thay vì tìm món
        if not data.get("candidates") and data.get("explanations"):
             print(f"   [Bot]: {data['explanations'][0]}")
             return None

        # In món ăn gợi ý
        candidates = data.get("candidates", [])
        print(f"   Found {len(candidates)} recipes:")
        for i, c in enumerate(candidates, 1):
            print(f"   {i}. {c['title']} (ID: {c['id']}) - Score: {c['score']:.4f}")
            
        return candidates[0]['id'] if candidates else None
        
    except requests.exceptions.ConnectionError:
        print("   ❌ Error: Không kết nối được Server. Hãy kiểm tra 'uvicorn' đã chạy chưa.")
        return None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def test_cart(recipe_id, servings=2):
    print(f"\n🛒 Testing Cart for '{recipe_id}' ({servings} people)")
    url = f"{BASE_URL}/recipes/{recipe_id}/suggest-cart"
    params = {"servings": servings}
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        totals = data.get('totals', {})
        print(f"   Dự kiến chi phí: {totals.get('estimated', 0):,.0f} {totals.get('currency', 'VND')}")
        
        print("   Danh sách mua sắm:")
        items = data.get('items', [])
        
        if not items:
            print("   (Giỏ hàng rỗng)")
            
        for item in items:
            qty = item.get('packages', 0)
            
            # Sử dụng pack_unit từ Backend trả về
            # Nếu backend cũ chưa trả về pack_unit, fallback về 'unitSize' -> 'unit'
            unit = item.get('pack_unit')
            if not unit:
                unit_size = item.get('unitSize')
                if isinstance(unit_size, dict):
                    unit = unit_size.get('unit', 'gói')
                else:
                    unit = 'gói'
            
            name = item.get('name') or item.get('ingredient')
            price = item.get('price', 0)
            subtotal = item.get('subtotal', 0)
            
            print(f"     - {qty} {unit} x {name:<30} : {price:,.0f}đ/sp  => {subtotal:,.0f}đ")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    # 1. Hỏi tìm món gà
    top_recipe_id = test_query("Trời nóng nên tôi muốn nấu món canh chua cá, 3 thành viên ăn")
    
    # 2. Nếu tìm thấy, tạo giỏ hàng cho món đó
    if top_recipe_id:
        test_cart(top_recipe_id, servings=2)