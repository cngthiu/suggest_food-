import requests
import json

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
        intent = data.get("intents", [{}])[0].get("name", "Unknown")
        print(f"   [NLU] Intent: {intent}")
        print(f"   [NLU] Slots: {data.get('slots')}")
        
        # In món ăn gợi ý
        candidates = data.get("candidates", [])
        print(f"   Found {len(candidates)} recipes:")
        for i, c in enumerate(candidates, 1):
            print(f"   {i}. {c['title']} (ID: {c['id']}) - Score: {c['score']:.4f}")
            
        return candidates[0]['id'] if candidates else None
        
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
        
        print(f"   Estimated Cost: {data['totals']['estimated']:,} {data['totals']['currency']}")
        print("   Shopping List:")
        for item in data['items']:
            qty = item.get('packages', 0)
            unit = item.get('unitSize', {}).get('unit', 'unit')
            name = item['name'] or item['ingredient']
            print(f"     - {qty} {unit} x {name}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    # 1. Hỏi tìm món gà
    top_recipe_id = test_query("Xin chao, ")
    
    # 2. Nếu tìm thấy, tạo giỏ hàng cho món đó
    if top_recipe_id:
        test_cart(top_recipe_id, servings=4)