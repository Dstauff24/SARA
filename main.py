from flask import Flask, request, jsonify
import os, re, json, requests
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
from datetime import datetime
from seller_memory_service import get_seller_memory, update_seller_memory
import tiktoken

# === Load Environment ===
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))
RENTCAST_API_KEY = os.getenv("RENTCAST_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

app = Flask(__name__)
conversation_memory = {"history": []}
MEMORY_LIMIT = 5

def safe_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def get_rentcast_valuation(address):
    headers = {"X-API-Key": RENTCAST_API_KEY}
    params = {"address": address}
    try:
        response = requests.get("https://api.rentcast.io/v1/properties/valuations", headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            return {
                "estimated_arv": safe_number(data.get("valuation")),
                "valuation_range_low": safe_number(data.get("valuationRangeLow")),
                "valuation_range_high": safe_number(data.get("valuationRangeHigh")),
                "price_per_sqft": safe_number(data.get("pricePerSqft")),
                "arv_source": "rentcast"
            }
    except Exception as e:
        print(f"[RentCast ARV Error] {e}")
    return {}

def get_rentcast_rent(address):
    headers = {"X-API-Key": RENTCAST_API_KEY}
    params = {"address": address}
    try:
        response = requests.get("https://api.rentcast.io/v1/properties/rental-estimates", headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            return {
                "estimated_rent": safe_number(data.get("rent")),
                "cap_rate": safe_number(data.get("capRate"))
            }
    except Exception as e:
        print(f"[RentCast Rent Error] {e}")
    return {}

tone_map = {
    "angry": ["this is ridiculous", "pissed", "you people", "frustrated"],
    "skeptical": ["not sure", "sounds like a scam", "don’t believe"],
    "curious": ["wondering", "what would you offer", "explain"],
    "hesitant": ["i don’t know", "maybe", "thinking"],
    "urgent": ["need to sell fast", "asap", "foreclosure"],
    "emotional": ["passed", "divorce", "lost job", "emotional"],
    "motivated": ["ready to go", "want to sell", "just want out"],
    "doubtful": ["too low", "never take that"],
    "withdrawn": ["leave me alone", "not interested"],
    "friendly": ["hey", "thanks for calling"],
    "direct": ["how much", "what’s the offer"],
    "neutral": []
}

def detect_tone(text):
    lowered = text.lower()
    for tone, keywords in tone_map.items():
        if any(kw in lowered for kw in keywords):
            return tone
    return "neutral"

def detect_seller_intent(text):
    text = text.lower()
    if any(kw in text for kw in ["how much", "offer", "price"]):
        return "price_sensitive"
    if any(kw in text for kw in ["foreclosure", "behind", "bank"]):
        return "distressed"
    if any(kw in text for kw in ["maybe", "thinking", "not sure"]):
        return "on_fence"
    if any(kw in text for kw in ["stop calling", "not interested"]):
        return "cold"
    if any(kw in text for kw in ["vacant", "tenant", "rented"]):
        return "landlord"
    return "general_inquiry"

def detect_system_flags(text):
    lowered = text.lower()
    system_flags = []
    repair_indicators = [
        ("roof", ["leak", "old", "bad", "replace", "issues", "problem", "not sure"]),
        ("hvac", ["not working", "old", "no ac", "broken", "needs", "unsure"]),
        ("plumbing", ["leak", "pipes", "bad", "old", "replace"]),
        ("electrical", ["rewire", "outlets", "old", "breaker", "unsafe"]),
        ("foundation", ["crack", "settle", "issue", "sinking", "problem"])
    ]
    for system, red_flags in repair_indicators:
        if system in lowered:
            for flag in red_flags:
                if flag in lowered:
                    system_flags.append(system)
                    break
    return system_flags

def estimate_repair_cost(memory, tone, text):
    sqft = memory.get("square_footage")
    try:
        sqft = int(sqft)
    except:
        sqft = 1200
    is_detailed = any(kw in text.lower() for kw in ["new roof", "new hvac", "renovated", "updated"])
    year_built = memory.get("year_built")
    try:
        is_newer = int(year_built) >= 2000
    except:
        is_newer = False
    base_cost = 20 * sqft if is_detailed and is_newer else 35 * sqft
    reason = "Used light rehab estimate due to detailed upgrades and newer build." if is_detailed and is_newer else "Defaulted to medium rehab + buffer due to vague or incomplete details."
    system_flags = detect_system_flags(text)
    system_cost = min(len(system_flags), 2) * 7000
    total_cost = (base_cost + system_cost) * 1.10
    return round(total_cost), reason

def calculate_offer(arv, repair_cost, roi):
    try:
        arv, repair_cost = float(arv), float(repair_cost)
        fees = arv * 0.06
        hold = 0.01 * (arv - repair_cost) * 3
        profit = roi * (arv - repair_cost)
        return round(arv - (fees + hold + repair_cost + profit), 2)
    except:
        return None

def extract_asking_price(text):
    matches = re.findall(r'\$?\s?(\d{5,7})', text.replace(',', ''))
    try:
        return int(matches[0]) if matches else None
    except:
        return None

def extract_offer_from_reply(text, asking_price=None, repair_cost=None):
    if not text:
        return None
    cleaned_text = text.replace(',', '')
    matches = re.findall(r'\$?\s?(\d{4,7})', cleaned_text)
    try:
        amounts = [int(m) for m in matches]
        if asking_price:
            amounts = [a for a in amounts if abs(a - asking_price) > 100]
        if repair_cost:
            amounts = [a for a in amounts if abs(a - repair_cost) > 100]
        return amounts[0] if amounts else None
    except:
        return None

def score_lead(tone, intent):
    score = 1
    if tone in ["motivated", "urgent"]:
        score += 2
    elif tone in ["curious", "direct"]:
        score += 1
    if intent in ["price_sensitive", "distressed"]:
        score += 2
    elif intent == "on_fence":
        score += 1
    return min(score, 5)

def num_tokens_from_messages(messages, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    total = 3
    for m in messages:
        total += 3
        for k, v in m.items():
            total += len(encoding.encode(v))
    return total

def summarize_messages(messages):
    prompt = [
        {"role": "system", "content": "Summarize key points like motivation, condition, timeline, pricing. Be natural."},
        {"role": "user", "content": "\n".join(messages)}
    ]
    return client.chat.completions.create(
        model="gpt-4", messages=prompt, temperature=0.5
    ).choices[0].message.content

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    seller_input = data.get("seller_input", "")
    phone = data.get("phone_number")
    if not seller_input or not phone:
        return jsonify({"error": "Missing seller_input or phone_number"}), 400

    memory = get_seller_memory(phone) or {}
    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    tone = detect_tone(seller_input)
    intent = detect_seller_intent(seller_input)

    address = data.get("property_address") or memory.get("property_address")
    if address:
        valuation_data = get_rentcast_valuation(address)
        rent_data = get_rentcast_rent(address)
    else:
        valuation_data, rent_data = {}, {}

    arv = safe_number(valuation_data.get("estimated_arv") or data.get("estimated_arv") or memory.get("estimated_arv"))
    repair_cost = data.get("repair_cost") or memory.get("repair_cost")
    repair_reason = memory.get("repair_reason")
    if not repair_cost:
        repair_cost, repair_reason = estimate_repair_cost(memory, tone, seller_input)

    min_offer = calculate_offer(arv, repair_cost, 0.30) if arv and repair_cost else None
    max_offer = calculate_offer(arv, repair_cost, 0.15) if arv and repair_cost else None

    try:
        embed = client.embeddings.create(input=[seller_input], model="text-embedding-3-small").data[0].embedding
        results = index.query(vector=embed, top_k=3, include_metadata=True)
        top_examples = [r.metadata["response"] for r in results.matches]
    except:
        top_examples = []

    summary = summarize_messages([m["content"] for m in conversation_memory["history"] if m["role"] == "user"])

    system_prompt = f'''
You are SARA, a smart, calm, emotionally intelligent acquisitions expert.
Seller tone: {tone}
Seller intent: {intent}
Use this offer range: Min Offer = ${min_offer}, Max Offer = ${max_offer}
Based on the seller input, we estimated repair costs to be ${repair_cost}.
Reason: {repair_reason}
NEPQ context: {"; ".join(top_examples) if top_examples else "No NEPQ examples found."}
Avoid ROI %. Emphasize cost logic, risk, and rehab needs.
'''

    messages = [{"role": "system", "content": system_prompt}] + conversation_memory["history"]
    while num_tokens_from_messages(messages) > 3000:
        messages.pop(1)

    response = client.chat.completions.create(model="gpt-4", messages=messages, temperature=0.7)
    reply = response.choices[0].message.content
    conversation_memory["history"].append({"role": "assistant", "content": reply})

    asking_price = extract_asking_price(seller_input)
    verbal_offer = extract_offer_from_reply(reply, asking_price, repair_cost)

    payload = {
        "phone_number": phone,
        "property_address": address,
        "conversation_log": conversation_memory["history"],  # ✅ Already a native list
        "call_summary": summary,
        "last_updated": datetime.utcnow().isoformat(),
        "follow_up_reason": memory.get("follow_up_reason"),
        "follow_up_set_by": memory.get("follow_up_set_by"),
        "asking_price": safe_number(asking_price or memory.get("asking_price")),
        "estimated_arv": safe_number(arv),
        "repair_cost": safe_number(repair_cost),
        "condition_notes": memory.get("condition_notes"),
        "strategy_flags": memory.get("strategy_flags") or [],
        "offer_history": memory.get("offer_history") or [],  # ✅ Make sure this is a list
        "verbal_offer_amount": safe_number(verbal_offer),
        "min_offer_amount": safe_number(min_offer),
        "max_offer_amount": safe_number(max_offer),
        "bedrooms": memory.get("bedrooms"),
        "bathrooms": memory.get("bathrooms"),
        "square_footage": memory.get("square_footage"),
        "year_built": memory.get("year_built"),
        "lead_source": memory.get("lead_source"),
        "is_deal": memory.get("is_deal", False),
        "summary_history": memory.get("summary_history") or [],  # ✅ Make sure this is a list
        "conversation_stage": memory.get("conversation_stage"),
        "next_follow_up_date": memory.get("next_follow_up_date"),
        "lead_status": memory.get("lead_status"),
        "motivation_score": memory.get("motivation_score"),
        "personality_tag": memory.get("personality_tag"),
        "timeline_to_sell": memory.get("timeline_to_sell"),
        "contradiction_flags": memory.get("contradiction_flags") or [],
        "lead_score": score_lead(tone, intent),
        "valuation_range_low": safe_number(valuation_data.get("valuation_range_low")),
        "valuation_range_high": safe_number(valuation_data.get("valuation_range_high")),
        "price_per_sqft": safe_number(valuation_data.get("price_per_sqft")),
        "estimated_rent": safe_number(rent_data.get("estimated_rent")),
        "cap_rate": safe_number(rent_data.get("cap_rate")),
        "arv_source": valuation_data.get("arv_source") or data.get("arv_source") or "manual"
}

    print("\n==== Payload to Supabase ====")
    print(json.dumps(payload, indent=2, default=str))
    print("=============================\n")

    update_success = update_seller_memory(phone, payload)

    if not update_success:
        print("❌ Supabase update failed. Sending 500 response to client.")
        return jsonify({
        "error": "Failed to update seller memory in Supabase.",
        "supabase_payload": payload
         }), 500

    return jsonify({
        "content": reply,
        "tone": tone,
        "intent": intent,
        "summary": summary,
        "nepq_examples": top_examples,
        "min_offer": min_offer,
        "max_offer": max_offer,
        "verbal_offer": verbal_offer,
        "repair_cost": repair_cost,
        "repair_reason": repair_reason
})

@app.route("/", methods=["GET"])
def index():
    return "✅ SARA Webhook is live!"

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")



