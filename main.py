from flask import Flask, request, jsonify
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
import tiktoken
from datetime import datetime
from seller_memory_service import get_seller_memory, update_seller_memory

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX")

client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

app = Flask(__name__)
conversation_memory = { "history": [] }
MEMORY_LIMIT = 5

tone_map = {
    "angry": ["ridiculous", "pissed", "frustrated"],
    "skeptical": ["scam", "don’t believe"],
    "curious": ["wondering", "what would you offer"],
    "hesitant": ["i don’t know", "maybe"],
    "urgent": ["asap", "foreclosure"],
    "emotional": ["passed", "divorce", "lost job"],
    "motivated": ["ready to go", "want to sell"],
    "doubtful": ["too low", "not enough"],
    "withdrawn": ["stop calling", "not interested"],
    "neutral": [],
    "friendly": ["thanks", "no worries"],
    "direct": ["what’s the offer", "how much"]
}

def detect_tone(input_text):
    lowered = input_text.lower()
    for tone, keywords in tone_map.items():
        if any(k in lowered for k in keywords):
            return tone
    return "neutral"

def detect_seller_intent(text):
    text = text.lower()
    if any(kw in text for kw in ["how much", "offer", "price"]):
        return "price_sensitive"
    elif any(kw in text for kw in ["foreclosure", "behind"]):
        return "distressed"
    elif any(kw in text for kw in ["maybe", "not sure"]):
        return "on_fence"
    elif any(kw in text for kw in ["stop calling", "not interested"]):
        return "cold"
    elif any(kw in text for kw in ["vacant", "tenant"]):
        return "landlord"
    return "general_inquiry"

def detect_contradiction(input_text, seller_data):
    contradictions = []
    if seller_data:
        if seller_data.get("asking_price") and str(seller_data["asking_price"]) not in input_text:
            if any(word in input_text.lower() for word in ["price", "$", "want"]):
                contradictions.append("asking_price")
        if seller_data.get("condition_notes") and "roof" in seller_data["condition_notes"].lower():
            if "roof is fine" in input_text.lower():
                contradictions.append("condition_notes")
    return contradictions

def generate_summary(user_messages):
    summary_prompt = [
        { "role": "system", "content": "Summarize the seller's key points: motivation, condition, timeline, pricing." },
        { "role": "user", "content": "\n".join(user_messages) }
    ]
    response = client.chat.completions.create(
        model="gpt-4", messages=summary_prompt, temperature=0.5
    )
    return response.choices[0].message.content

def num_tokens_from_messages(messages, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    tokens_per_message = 3
    tokens_per_name = 1
    total = 0
    for m in messages:
        total += tokens_per_message + sum(len(encoding.encode(v)) for v in m.values())
        if "name" in m: total += tokens_per_name
    return total + 3

def calculate_investor_price(arv, repair_cost, target_roi):
    realtor_fees = arv * 0.06
    holding_costs = 0.01 * (arv - repair_cost) * 3
    profit = target_roi * (arv - repair_cost)
    return round(arv - (realtor_fees + holding_costs + repair_cost + profit), 2)

def generate_update_payload(data, existing, convo, call_summary, min_offer, max_offer, verbal_offer):
    summary_history = existing.get("summary_history")
    if isinstance(summary_history, str):
        try: summary_history = json.loads(summary_history)
        except: summary_history = []
    if not summary_history: summary_history = []
    summary_history.append({ "timestamp": datetime.utcnow().isoformat(), "summary": call_summary })

    offer_history = existing.get("offer_history") or []
    if verbal_offer:
        offer_history.append({ "amount": round(verbal_offer, 2), "timestamp": datetime.utcnow().isoformat() })

    merged_payload = {
        "conversation_log": convo,
        "call_summary": call_summary,
        "summary_history": summary_history,
        "min_offer_amount": round(min_offer, 2) if min_offer else None,
        "max_offer_amount": round(max_offer, 2) if max_offer else None,
        "verbal_offer_amount": round(verbal_offer, 2) if verbal_offer else None,
        "offer_history": offer_history,
    }

    for key in [
        "asking_price", "repair_cost", "estimated_arv", "follow_up_date",
        "follow_up_reason", "follow_up_set_by", "property_address", "condition_notes",
        "lead_source", "bedrooms", "bathrooms", "square_footage", "year_built", "conversation_stage"
    ]:
        if key in data and data[key] is not None:
            merged_payload[key] = data[key]
        elif existing.get(key):
            merged_payload[key] = existing[key]

    return merged_payload

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    seller_input = data.get("seller_input", "")
    arv = data.get("estimated_arv") or data.get("arv")
    repair_cost = data.get("repair_cost")
    phone = data.get("phone_number")

    if not seller_input or not phone:
        return jsonify({"error": "Missing required fields"}), 400

    seller_data = get_seller_memory(phone)
    conversation_memory["history"].append({ "role": "user", "content": seller_input })
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    try:
        vector = client.embeddings.create(input=[seller_input], model="text-embedding-3-small").data[0].embedding
        result = index.query(vector=vector, top_k=3, include_metadata=True)
        top_pairs = [match.metadata["response"] for match in result.matches]
    except:
        top_pairs = []

    seller_tone = detect_tone(seller_input)
    seller_intent = detect_seller_intent(seller_input)
    contradictions = detect_contradiction(seller_input, seller_data)
    contradiction_note = f"⚠️ Seller contradiction(s): {', '.join(contradictions)}" if contradictions else ""

    min_offer = max_offer = verbal_offer = None
    investor_offer = ""
    if arv and repair_cost:
        try:
            arv = float(arv)
            repair_cost = float(repair_cost)
            min_offer = calculate_investor_price(arv, repair_cost, 0.30)
            max_offer = calculate_investor_price(arv, repair_cost, 0.10)
            verbal_offer = min_offer
            investor_offer = f"Start at ${min_offer}, negotiate to max ${max_offer}."
        except:
            pass

    summary = generate_summary([m["content"] for m in conversation_memory["history"] if m["role"] == "user"])

    system_prompt = f"""
{contradiction_note}
Summary of previous convo:
{summary}

You are SARA, a warm, confident, strategic acquisitions expert.
Seller Tone: {seller_tone}
Seller Intent: {seller_intent}
Offer Strategy: {investor_offer}

Embed NEPQ-style language like:
{"; ".join(top_pairs) if top_pairs else "No NEPQ examples found."}

Avoid ROI talk. Justify price via repair risk, timeline, costs.
"""

    messages = [{"role": "system", "content": system_prompt}] + conversation_memory["history"]
    while num_tokens_from_messages(messages) > 3000:
        messages.pop(1)

    try:
        reply = client.chat.completions.create(
            model="gpt-4", messages=messages, temperature=0.7
        ).choices[0].message.content
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    conversation_memory["history"].append({ "role": "assistant", "content": reply })

    payload = generate_update_payload(data, seller_data or {}, conversation_memory["history"], summary, min_offer, max_offer, verbal_offer)
    update_seller_memory(phone, payload)

    return jsonify({
        "content": reply,
        "tone": seller_tone,
        "intent": seller_intent,
        "summary": summary,
        "nepq_examples": top_pairs,
        "reasoning": investor_offer,
        "contradictions": contradictions
    })

@app.route("/", methods=["GET"])
def index():
    return "✅ SARA Webhook is running!"

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")



