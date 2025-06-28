from flask import Flask, request, jsonify
import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
import tiktoken
from seller_memory_service import get_seller_memory, update_seller_memory
from datetime import datetime
import json
import pprint

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX")

# Initialize clients
client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

# Flask App
app = Flask(__name__)

# Short-Term Memory
conversation_memory = {"history": []}
MEMORY_LIMIT = 5

# Tone Mapping
tone_map = {
    "angry": ["ridiculous", "pissed", "you people", "frustrated"],
    "skeptical": ["not sure", "scam", "don’t believe"],
    "curious": ["wondering", "what would you offer", "can you explain"],
    "hesitant": ["don’t know", "maybe", "thinking about it"],
    "urgent": ["sell fast", "asap", "foreclosure", "eviction"],
    "emotional": ["mom passed", "divorce", "lost job", "hard time"],
    "motivated": ["ready to go", "want to sell", "just want out"],
    "doubtful": ["no way", "too low", "never take that"],
    "withdrawn": ["leave me alone", "stop calling", "not interested"],
    "neutral": [],
    "friendly": ["hey", "thanks for calling", "no worries"],
    "direct": ["how much", "what’s the offer", "cut to it"]
}

def detect_tone(text):
    lowered = text.lower()
    for tone, keywords in tone_map.items():
        if any(keyword in lowered for keyword in keywords):
            return tone
    return "neutral"

def detect_seller_intent(text):
    lowered = text.lower()
    if any(kw in lowered for kw in ["how much", "offer", "price", "what would you give"]):
        return "price_sensitive"
    if any(kw in lowered for kw in ["foreclosure", "behind", "bank", "notice"]):
        return "distressed"
    if any(kw in lowered for kw in ["maybe", "thinking", "not sure", "depends"]):
        return "on_fence"
    if any(kw in lowered for kw in ["stop calling", "not interested", "leave me alone"]):
        return "cold"
    if any(kw in lowered for kw in ["vacant", "tenant", "rented", "investment"]):
        return "landlord"
    return "general_inquiry"

def detect_contradiction(input_text, seller_data):
    contradictions = []
    if seller_data:
        if seller_data.get("asking_price") and str(seller_data["asking_price"]) not in input_text:
            if any(w in input_text for w in ["price", "$", "want", "need"]):
                contradictions.append("asking_price")
        if seller_data.get("condition_notes") and "roof" in seller_data["condition_notes"].lower():
            if "roof is fine" in input_text.lower() or "no issues" in input_text.lower():
                contradictions.append("condition_notes")
    return contradictions

def generate_summary(user_messages):
    prompt = [
        {"role": "system", "content": "Summarize the following seller conversation highlighting motivation, condition, timeline, and pricing:"},
        {"role": "user", "content": "\n".join(user_messages)}
    ]
    response = client.chat.completions.create(model="gpt-4", messages=prompt, temperature=0.5)
    return response.choices[0].message.content

def num_tokens_from_messages(messages, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = 0
    for msg in messages:
        tokens += 3 + sum(len(encoding.encode(v)) for v in msg.values())
    return tokens + 3

def calculate_investor_price(arv, repair_cost, target_roi):
    fees = arv * 0.06
    hold = 0.01 * (arv - repair_cost) * 3
    profit = target_roi * (arv - repair_cost)
    return round(arv - (fees + hold + repair_cost + profit), 2)

def generate_update_payload(data, existing, conversation, summary, verbal, min_offer, max_offer):
    summary_history = existing.get("summary_history", [])
    if isinstance(summary_history, str):
        try:
            summary_history = json.loads(summary_history)
        except:
            summary_history = []
    summary_history.append({"timestamp": datetime.utcnow().isoformat(), "summary": summary})

    offer_history = existing.get("offer_history", [])
    if verbal:
        offer_history.append({
            "amount": round(verbal, 2),
            "timestamp": datetime.utcnow().isoformat()
        })

    return {
        "phone_number": data.get("phone_number"),
        "conversation_log": conversation,
        "call_summary": summary,
        "summary_history": summary_history,
        "asking_price": data.get("asking_price") or existing.get("asking_price"),
        "repair_cost": data.get("repair_cost") or existing.get("repair_cost"),
        "estimated_arv": data.get("estimated_arv") or data.get("arv") or existing.get("estimated_arv"),
        "min_offer_amount": round(min_offer, 2) if min_offer else existing.get("min_offer_amount"),
        "max_offer_amount": round(max_offer, 2) if max_offer else existing.get("max_offer_amount"),
        "verbal_offer_amount": round(verbal, 2) if verbal else existing.get("verbal_offer_amount"),
        "offer_history": offer_history,
        "follow_up_date": data.get("follow_up_date") or existing.get("follow_up_date"),
        "follow_up_reason": data.get("follow_up_reason") or existing.get("follow_up_reason"),
        "follow_up_set_by": data.get("follow_up_set_by") or existing.get("follow_up_set_by"),
        "property_address": data.get("property_address") or existing.get("property_address"),
        "condition_notes": data.get("condition_notes") or existing.get("condition_notes"),
        "lead_source": data.get("lead_source") or existing.get("lead_source"),
        "bedrooms": data.get("bedrooms") or existing.get("bedrooms"),
        "bathrooms": data.get("bathrooms") or existing.get("bathrooms"),
        "square_footage": data.get("square_footage") or existing.get("square_footage"),
        "year_built": data.get("year_built") or existing.get("year_built")
    }

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    seller_input = data.get("seller_input", "")
    phone = data.get("phone_number")

    if not seller_input or not phone:
        return jsonify({"error": "Missing input or phone_number"}), 400

    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    tone = detect_tone(seller_input)
    intent = detect_seller_intent(seller_input)
    seller_data = get_seller_memory(phone) or {}

    try:
        vector = client.embeddings.create(input=[seller_input], model="text-embedding-3-small").data[0].embedding
        results = index.query(vector=vector, top_k=3, include_metadata=True)
        top_pairs = [m.metadata["response"] for m in results.matches]
    except:
        top_pairs = []

    verbal_offer = min_offer = max_offer = None
    arv = data.get("arv") or seller_data.get("estimated_arv")
    repair = data.get("repair_cost") or seller_data.get("repair_cost")

    if arv and repair:
        try:
            arv = float(arv)
            repair = float(repair)
            min_offer = calculate_investor_price(arv, repair, 0.30)
            max_offer = calculate_investor_price(arv, repair, 0.15)
            verbal_offer = round(max_offer * 0.95, 2)
        except:
            pass

    summary = generate_summary([m["content"] for m in conversation_memory["history"] if m["role"] == "user"])
    contradictions = detect_contradiction(seller_input, seller_data)
    system_prompt = f"""
⚠️ Contradictions: {', '.join(contradictions)}\n
Last Summary: {summary}\n
You are SARA, a sharp and emotionally intelligent acquisitions expert.
Tone: {tone}, Intent: {intent}
NEPQ Examples: {"; ".join(top_pairs) if top_pairs else "None"}
Verbal Offer: ${verbal_offer}, Range: ${min_offer} – ${max_offer}
"""

    messages = [{"role": "system", "content": system_prompt}] + conversation_memory["history"]
    while num_tokens_from_messages(messages) > 3000:
        messages.pop(1)

    try:
        reply = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        ).choices[0].message.content
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    conversation_memory["history"].append({"role": "assistant", "content": reply})

    payload = generate_update_payload(data, seller_data, conversation_memory["history"], summary, verbal_offer, min_offer, max_offer)
    pprint.PrettyPrinter(indent=2).pprint(payload)
    update_seller_memory(phone, payload)

    return jsonify({
        "content": reply,
        "tone": tone,
        "intent": intent,
        "summary": summary,
        "nepq_examples": top_pairs,
        "reasoning": f"Start at ${min_offer}, negotiate up to ${max_offer}.",
        "contradictions": contradictions
    })

@app.route("/", methods=["GET"])
def index():
    return "✅ SARA is live."

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")








