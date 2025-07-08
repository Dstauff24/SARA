from flask import Flask, request, jsonify
import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
import tiktoken
import re
from datetime import datetime
import json
from seller_memory_service import get_seller_memory, update_seller_memory

# Load env variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX")

# Init services
client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)
app = Flask(__name__)

conversation_memory = {"history": []}
MEMORY_LIMIT = 5

# --- UTILITY FUNCTIONS ---

def detect_tone(text):
    tone_map = {
        "angry": ["this is ridiculous", "pissed", "frustrated"],
        "skeptical": ["not sure", "scam", "don’t believe"],
        "curious": ["just wondering", "what would you offer"],
        "hesitant": ["don’t know", "maybe", "thinking"],
        "urgent": ["sell fast", "asap", "foreclosure"],
        "emotional": ["passed", "divorce", "lost job"],
        "motivated": ["ready", "want to sell", "just want out"],
        "doubtful": ["too low", "never take"],
        "withdrawn": ["stop calling", "not interested"],
        "friendly": ["hey", "thanks for calling"],
        "direct": ["how much", "what’s the offer"]
    }
    lowered = text.lower()
    for tone, keywords in tone_map.items():
        if any(k in lowered for k in keywords):
            return tone
    return "neutral"

def detect_seller_intent(text):
    text = text.lower()
    if any(k in text for k in ["how much", "offer", "price"]): return "price_sensitive"
    if any(k in text for k in ["foreclosure", "behind", "notice"]): return "distressed"
    if any(k in text for k in ["maybe", "thinking", "not sure"]): return "on_fence"
    if any(k in text for k in ["stop calling", "not interested"]): return "cold"
    if any(k in text for k in ["vacant", "tenant", "investment"]): return "landlord"
    return "general_inquiry"

def extract_asking_price(text):
    matches = re.findall(r'\$?\s?(\d{3,7})(?:[^\d]|$)', text.replace(',', ''))
    numbers = [int(n) for n in matches if int(n) > 1000]
    return numbers[0] if numbers else None

def num_tokens_from_messages(messages, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    return sum(len(enc.encode(m["content"])) + 3 for m in messages) + 3

def calculate_investor_price(arv, repair, roi):
    fees = arv * 0.06
    hold = 0.01 * (arv - repair) * 3
    profit = roi * (arv - repair)
    return round(arv - (fees + hold + repair + profit), 2)

def generate_summary(msgs):
    messages = [
        {"role": "system", "content": "Summarize the seller’s key points: motivation, timeline, price, condition."},
        {"role": "user", "content": "\n".join(msgs)}
    ]
    return client.chat.completions.create(model="gpt-4", messages=messages).choices[0].message.content

def generate_update_payload(data, memory, convo, summary, verbal, min_offer, max_offer):
    now = datetime.utcnow().isoformat()
    summary_history = memory.get("summary_history", [])
    if isinstance(summary_history, str):
        try: summary_history = json.loads(summary_history)
        except: summary_history = []
    summary_history.append({"timestamp": now, "summary": summary})

    offer_history = memory.get("offer_history", [])
    if isinstance(offer_history, str):
        try: offer_history = json.loads(offer_history)
        except: offer_history = []
    if verbal:
        offer_history.append({"amount": verbal, "timestamp": now})

    return {
        "phone_number": data.get("phone_number"),
        "conversation_log": convo,
        "call_summary": summary,
        "summary_history": summary_history,
        "offer_history": offer_history,
        "verbal_offer_amount": verbal or memory.get("verbal_offer_amount"),
        "min_offer_amount": min_offer or memory.get("min_offer_amount"),
        "max_offer_amount": max_offer or memory.get("max_offer_amount"),
        "asking_price": data.get("asking_price") or extract_asking_price(data.get("seller_input", "")) or memory.get("asking_price"),
        "estimated_arv": data.get("estimated_arv") or memory.get("estimated_arv"),
        "repair_cost": data.get("repair_cost") or memory.get("repair_cost"),
        "condition_notes": data.get("condition_notes") or memory.get("condition_notes"),
        "property_address": data.get("property_address") or memory.get("property_address"),
        "lead_source": data.get("lead_source") or memory.get("lead_source"),
        "bedrooms": data.get("bedrooms") or memory.get("bedrooms"),
        "bathrooms": data.get("bathrooms") or memory.get("bathrooms"),
        "square_footage": data.get("square_footage") or memory.get("square_footage"),
        "year_built": data.get("year_built") or memory.get("year_built"),
        "follow_up_date": data.get("follow_up_date") or memory.get("follow_up_date"),
        "follow_up_reason": data.get("follow_up_reason") or memory.get("follow_up_reason"),
        "follow_up_set_by": data.get("follow_up_set_by") or memory.get("follow_up_set_by"),
        "conversation_stage": memory.get("conversation_stage", "Introduction + Rapport")
    }

# --- ROUTE ---

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    phone = data.get("phone_number")
    seller_input = data.get("seller_input", "")
    if not phone or not seller_input:
        return jsonify({"error": "Missing seller_input or phone_number"}), 400

    memory = get_seller_memory(phone) or {}
    conversation_memory["history"].append({"role": "user", "content": seller_input})
    conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    seller_tone = detect_tone(seller_input)
    seller_intent = detect_seller_intent(seller_input)

    try:
        vector = client.embeddings.create(input=[seller_input], model="text-embedding-3-small").data[0].embedding
        matches = index.query(vector=vector, top_k=3, include_metadata=True)
        top_pairs = [m.metadata["response"] for m in matches.matches]
    except:
        top_pairs = []

    try:
        arv = float(data.get("estimated_arv") or memory.get("estimated_arv") or 0)
        repair = float(data.get("repair_cost") or memory.get("repair_cost") or 0)
        min_offer = calculate_investor_price(arv, repair, 0.30)
        max_offer = calculate_investor_price(arv, repair, 0.15)
        verbal_offer = min_offer  # default first offer
    except:
        min_offer = max_offer = verbal_offer = None

    summary = generate_summary([m["content"] for m in conversation_memory["history"] if m["role"] == "user"])

    walkthrough = 'Once we agree on terms, we’ll verify condition — nothing for you to worry about now.'
    prompt = f"""
You are SARA, a calm and strategic acquisitions expert.

Tone: {seller_tone}
Intent: {seller_intent}

NEPQ Framing:
{"; ".join(top_pairs) if top_pairs else "No NEPQ examples."}

Walkthrough Logic: {walkthrough}
Offer Strategy: Start at ${min_offer}, max ${max_offer}.
"""

    messages = [{"role": "system", "content": prompt}] + conversation_memory["history"]
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
    payload = generate_update_payload(data, memory, conversation_memory["history"], summary, verbal_offer, min_offer, max_offer)
    update_seller_memory(phone, payload)

    return jsonify({
        "content": reply,
        "tone": seller_tone,
        "intent": seller_intent,
        "summary": summary,
        "reasoning": f"${min_offer} to ${max_offer}",
        "offers": {
            "verbal": verbal_offer,
            "min_offer": min_offer,
            "max_offer": max_offer
        }
    })

@app.route("/", methods=["GET"])
def home():
    return "✅ SARA is live"

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")









