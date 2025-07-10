from flask import Flask, request, jsonify
import os, re, json, requests
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
import tiktoken
from datetime import datetime, timedelta
from seller_memory_service import get_seller_memory, update_seller_memory

# Load environment
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))
RENTCAST_API_KEY = os.getenv("RENTCAST_API_KEY")

app = Flask(__name__)
conversation_memory = {"history": []}
MEMORY_LIMIT = 5

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

def get_property_valuation(address):
    url = f"https://api.rentcast.io/v1/avm/value?address={address}"
    headers = {"X-Api-Key": RENTCAST_API_KEY, "Accept": "application/json"}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return {
                "estimated_arv": data.get("price"),
                "valuation_range_low": data.get("priceRangeLow"),
                "valuation_range_high": data.get("priceRangeHigh"),
                "price_per_sqft": data.get("pricePerSqft"),
                "arv_source": "RentCast"
            }
    except Exception as e:
        print(f"[RentCast ARV Error] {e}")
    return {}

def get_rental_estimate(address):
    url = f"https://api.rentcast.io/v1/avm/rent/long-term?address={address}"
    headers = {"X-Api-Key": RENTCAST_API_KEY, "Accept": "application/json"}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return {
                "estimated_rent": data.get("rent"),
                "cap_rate": data.get("marketCapRate")
            }
    except Exception as e:
        print(f"[RentCast Rent Error] {e}")
    return {}
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

    tone = tone or "neutral"
    lowered = text.lower()
    is_vague = any(phrase in lowered for phrase in ["great shape", "just needs paint", "just cosmetic", "a little touching up"])
    is_detailed = any(phrase in lowered for phrase in ["new roof", "new hvac", "remodeled", "renovated", "fully updated"])
    year_built = memory.get("year_built")
    try:
        is_newer = int(year_built) >= 2000
    except:
        is_newer = False

    if is_detailed and is_newer:
        base_cost = 20 * sqft
        reason = "Used light rehab estimate due to detailed upgrades and newer build."
    else:
        base_cost = 35 * sqft
        reason = "Defaulted to medium rehab + buffer due to vague or incomplete details."

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
def extract_asking_price(text):
    matches = re.findall(r'\$?\s?(\d{5,7})', text.replace(',', ''))
    try:
        return int(matches[0]) if matches else None
    except:
        return None

def extract_offer_from_reply(text, asking_price=None):
    if not text:
        return None
    cleaned_text = text.replace(',', '')
    matches = re.findall(r'\$?\s?(\d{4,7})', cleaned_text)
    try:
        amounts = [int(m) for m in matches]
        if asking_price:
            amounts = [a for a in amounts if abs(a - asking_price) > 100]
        return amounts[0] if amounts else None
    except:
        return None

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

    # RentCast ARV + Rent if we have an address and no ARV yet
    property_address = data.get("property_address") or memory.get("property_address")
    if property_address and not memory.get("estimated_arv"):
        valuation_data = get_property_valuation(property_address)
        rental_data = get_rental_estimate(property_address)
        memory.update({**valuation_data, **rental_data})
        memory["property_address"] = property_address  # make sure it's stored

    try:
        embed = client.embeddings.create(input=[seller_input], model="text-embedding-3-small").data[0].embedding
        results = index.query(vector=embed, top_k=3, include_metadata=True)
        top_examples = [r.metadata["response"] for r in results.matches]
    except:
        top_examples = []

    # Conservative repair cost if not yet set
    repair_cost = data.get("repair_cost") or memory.get("repair_cost")
    repair_reason = memory.get("repair_reason")
    if not repair_cost:
        repair_cost, repair_reason = estimate_repair_cost(memory, tone, seller_input)

    arv = memory.get("estimated_arv") or data.get("estimated_arv")
    min_offer = calculate_offer(arv, repair_cost, 0.30) if arv and repair_cost else None
    max_offer = calculate_offer(arv, repair_cost, 0.15) if arv and repair_cost else None

    summary = summarize_messages([m["content"] for m in conversation_memory["history"] if m["role"] == "user"])

    system_prompt = f'''
You are SARA, a smart, calm, emotionally intelligent acquisitions expert.
Seller tone: {tone}
Seller intent: {intent}
Use this offer range: Min Offer = ${min_offer}, Max Offer = ${max_offer}
Based on the seller input, we estimated repair costs to be ${repair_cost}.
Reason: {repair_reason}
Explain how you arrived at that estimate clearly during the repair review stage.
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
    verbal_offer = extract_offer_from_reply(reply, asking_price)

    payload = {
        "phone_number": phone,
        "conversation_log": conversation_memory["history"],
        "call_summary": summary,
        "verbal_offer_amount": verbal_offer,
        "min_offer_amount": min_offer,
        "max_offer_amount": max_offer,
        "asking_price": asking_price or memory.get("asking_price"),
        "repair_cost": repair_cost,
        "repair_reason": repair_reason,
        "estimated_arv": arv,
        "valuation_range_low": memory.get("valuation_range_low"),
        "valuation_range_high": memory.get("valuation_range_high"),
        "price_per_sqft": memory.get("price_per_sqft"),
        "estimated_rent": memory.get("estimated_rent"),
        "cap_rate": memory.get("cap_rate"),
        "arv_source": memory.get("arv_source"),
        "property_address": property_address,
        "tone": tone,
        "intent": intent
    }

    print("\n==== Payload to Supabase ====")
    print(json.dumps(payload, indent=2, default=str))
    print("=============================\n")

    update_seller_memory(phone, payload)

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








