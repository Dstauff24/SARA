# main.py
from flask import Flask, request, jsonify
import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
import tiktoken
from seller_memory_service import get_seller_memory, update_seller_memory
from memory_summarizer import summarize_and_trim_memory
from datetime import datetime

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

# Tone Detection
TONE_MAP = {
    "angry": ["this is ridiculous", "i’m pissed", "you people", "frustrated"],
    "skeptical": ["not sure", "sounds like a scam", "don’t believe"],
    "curious": ["i’m just wondering", "what would you offer", "can you explain"],
    "hesitant": ["i don’t know", "maybe", "thinking about it"],
    "urgent": ["need to sell fast", "asap", "foreclosure", "eviction"],
    "emotional": ["my mom passed", "divorce", "lost job", "hard time"],
    "motivated": ["ready to go", "want to sell", "just want out"],
    "doubtful": ["no way that’s enough", "that’s too low", "i’ll never take that"],
    "withdrawn": ["leave me alone", "stop calling", "not interested"],
    "neutral": [],
    "friendly": ["hey", "thanks for calling", "no worries"],
    "direct": ["how much", "what’s the offer", "let’s cut to it"]
}

def detect_tone(text):
    text = text.lower()
    for tone, keywords in TONE_MAP.items():
        if any(k in text for k in keywords):
            return tone
    return "neutral"

def detect_seller_intent(text):
    lowered = text.lower()
    if any(k in lowered for k in ["how much", "offer", "price", "what would you give"]):
        return "price_sensitive"
    elif any(k in lowered for k in ["foreclosure", "behind", "bank", "notice"]):
        return "distressed"
    elif any(k in lowered for k in ["maybe", "thinking", "not sure", "depends"]):
        return "on_fence"
    elif any(k in lowered for k in ["stop calling", "not interested", "leave me alone"]):
        return "cold"
    elif any(k in lowered for k in ["vacant", "tenant", "rented", "investment"]):
        return "landlord"
    return "general_inquiry"

def detect_contradiction(new_input, old_data):
    if not old_data:
        return []
    contradictions = []
    if old_data.get("asking_price") and str(old_data["asking_price"]) not in new_input:
        if any(w in new_input for w in ["price", "$", "want", "need"]):
            contradictions.append("asking_price")
    if old_data.get("condition_notes") and "roof" in old_data["condition_notes"].lower():
        if "roof is fine" in new_input.lower() or "no issues" in new_input.lower():
            contradictions.append("condition_notes")
    return contradictions

def calculate_investor_price(arv, repair_cost, roi):
    realtor_fees = arv * 0.06
    hold_costs = 0.01 * (arv - repair_cost) * 3
    profit = roi * (arv - repair_cost)
    return round(arv - (realtor_fees + hold_costs + repair_cost + profit), 2)

def num_tokens_from_messages(messages, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = 0
    for msg in messages:
        tokens += 3 + sum(len(encoding.encode(v)) for v in msg.values())
    return tokens + 3

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    seller_input = data.get("seller_input")
    phone = data.get("phone_number")
    if not seller_input or not phone:
        return jsonify({"error": "Missing fields"}), 400

    seller_tone = detect_tone(seller_input)
    seller_intent = detect_seller_intent(seller_input)
    seller_data = get_seller_memory(phone)

    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    try:
        summarized, trimmed = summarize_and_trim_memory(phone, conversation_memory["history"])
        conversation_memory["history"] = trimmed
    except Exception as e:
        summarized = None

    try:
        vector = client.embeddings.create(
            input=[seller_input],
            model="text-embedding-3-small"
        ).data[0].embedding
        top = index.query(vector=vector, top_k=3, include_metadata=True)
        top_pairs = [m.metadata["response"] for m in top.matches]
    except:
        top_pairs = []

    offer_amount, offer_text = None, ""
    try:
        arv, rc = float(data.get("arv", 0)), float(data.get("repair_cost", 0))
        start = calculate_investor_price(arv, rc, 0.30)
        max_offer = calculate_investor_price(arv, rc, 0.10)
        offer_amount = start
        offer_text = f"Start at ${start}, max out at ${max_offer}."
    except:
        pass

    contradictions = detect_contradiction(seller_input, seller_data)
    contradiction_note = f"⚠️ Contradiction: {', '.join(contradictions)}." if contradictions else ""

    system_prompt = f"""
{contradiction_note}
Summary: {summarized or "No prior summary."}
Tone: {seller_tone}
Intent: {seller_intent}
Offer Guidance: {offer_text}
Use these NEPQ examples naturally:
{' ; '.join(top_pairs) if top_pairs else 'No examples.'}
"""

    messages = [{"role": "system", "content": system_prompt}] + conversation_memory["history"]
    while num_tokens_from_messages(messages) > 3000:
        messages.pop(1)

    response = client.chat.completions.create(
        model="gpt-4", messages=messages, temperature=0.7
    )
    reply = response.choices[0].message.content
    conversation_memory["history"].append({"role": "assistant", "content": reply})

    payload = {
        "conversation_log": conversation_memory["history"],
        "call_summary": summarized,
        "summary_history": seller_data.get("summary_history", []) + [{"summary": summarized, "timestamp": datetime.utcnow().isoformat()}] if summarized else seller_data.get("summary_history", []),
        "last_offer_amount": offer_amount,
        "asking_price": data.get("asking_price") or seller_data.get("asking_price"),
        "repair_cost": data.get("repair_cost") or seller_data.get("repair_cost"),
        "estimated_arv": data.get("arv") or seller_data.get("estimated_arv"),
        "follow_up_date": data.get("follow_up_date") or seller_data.get("follow_up_date"),
        "follow_up_reason": data.get("follow_up_reason") or seller_data.get("follow_up_reason"),
        "follow_up_set_by": data.get("follow_up_set_by") or seller_data.get("follow_up_set_by"),
        "condition_notes": data.get("condition_notes") or seller_data.get("condition_notes"),
        "property_address": data.get("property_address") or seller_data.get("property_address"),
        "bedrooms": data.get("bedrooms") or seller_data.get("bedrooms"),
        "bathrooms": data.get("bathrooms") or seller_data.get("bathrooms"),
        "square_footage": data.get("square_footage") or seller_data.get("square_footage"),
        "year_built": data.get("year_built") or seller_data.get("year_built"),
        "lead_source": data.get("lead_source") or seller_data.get("lead_source"),
        "phone_number": phone
    }

    if offer_amount:
        payload["offer_history"] = seller_data.get("offer_history", []) + [{"amount": offer_amount, "timestamp": datetime.utcnow().isoformat()}]

    update_seller_memory(phone, payload)

    return jsonify({
        "content": reply,
        "tone": seller_tone,
        "intent": seller_intent,
        "summary": summarized,
        "reasoning": offer_text,
        "nepq_examples": top_pairs,
        "contradictions": contradictions
    })

@app.route("/", methods=["GET"])
def index():
    return "✅ SARA Webhook is running!"

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")







