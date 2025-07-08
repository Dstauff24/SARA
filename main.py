from flask import Flask, request, jsonify
import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
import tiktoken
from seller_memory_service import get_seller_memory, update_seller_memory
from datetime import datetime
import json
import re

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX")

# Initialize clients
client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

app = Flask(__name__)
conversation_memory = {"history": []}
MEMORY_LIMIT = 5

tone_map = {
    "angry": ["ridiculous", "pissed", "you people", "frustrated"],
    "skeptical": ["scam", "not sure", "don’t believe"],
    "curious": ["wondering", "offer", "explain"],
    "hesitant": ["don’t know", "maybe", "thinking"],
    "urgent": ["asap", "foreclosure", "eviction"],
    "emotional": ["passed", "divorce", "lost job", "hard time"],
    "motivated": ["ready", "want to sell", "just want out"],
    "doubtful": ["too low", "never take that"],
    "withdrawn": ["leave me alone", "stop calling"],
    "neutral": [],
    "friendly": ["hey", "thanks", "no worries"],
    "direct": ["how much", "what’s the offer"]
}

def detect_tone(text):
    text = text.lower()
    for tone, keywords in tone_map.items():
        if any(k in text for k in keywords):
            return tone
    return "neutral"

def detect_seller_intent(text):
    text = text.lower()
    if any(k in text for k in ["how much", "offer", "price"]):
        return "price_sensitive"
    if any(k in text for k in ["foreclosure", "behind", "bank"]):
        return "distressed"
    if any(k in text for k in ["maybe", "thinking", "not sure"]):
        return "on_fence"
    if any(k in text for k in ["stop calling", "not interested"]):
        return "cold"
    if any(k in text for k in ["vacant", "tenant", "rented"]):
        return "landlord"
    return "general_inquiry"

def extract_asking_price(text):
    numbers = re.findall(r'\$?\s?(\d{3,6})', text.replace(',', ''))
    if numbers:
        price_candidates = [int(n) for n in numbers if int(n) > 10000]
        return max(price_candidates) if price_candidates else None
    return None

def summarize_messages(user_messages):
    prompt = [
        {"role": "system", "content": "Summarize the seller's messages below. Focus on property condition, price expectations, motivation, and any timeline or situational details. Be natural and concise."},
        {"role": "user", "content": "\n".join(user_messages)}
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=prompt,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

def num_tokens_from_messages(messages, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    tokens_per_message = 3
    tokens_per_name = 1
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    return num_tokens + 3

def calculate_offer_range(arv, repair_cost):
    try:
        arv = float(arv)
        repair_cost = float(repair_cost)
        realtor_fees = arv * 0.06
        holding_costs = 0.01 * (arv - repair_cost) * 3
        min_profit = 0.15 * (arv - repair_cost)
        max_profit = 0.30 * (arv - repair_cost)
        min_offer = arv - (realtor_fees + holding_costs + repair_cost + min_profit)
        max_offer = arv - (realtor_fees + holding_costs + repair_cost + max_profit)
        return round(min_offer, 2), round(max_offer, 2)
    except:
        return None, None

def generate_update_payload(data, existing, history, summary, verbal_offer, min_offer, max_offer):
    merged = existing.copy() if existing else {}
    summary_history = merged.get("summary_history")
    if isinstance(summary_history, str):
        try:
            summary_history = json.loads(summary_history)
        except:
            summary_history = []
    if not summary_history:
        summary_history = []
    summary_history.append({
        "timestamp": datetime.utcnow().isoformat(),
        "summary": summary
    })

    offer_history = merged.get("offer_history") or []
    if verbal_offer:
        offer_history.append({
            "amount": round(verbal_offer, 2),
            "timestamp": datetime.utcnow().isoformat()
        })

    asking_price = data.get("asking_price") or extract_asking_price(data.get("seller_input", ""))
    condition_notes = data.get("condition_notes") or merged.get("condition_notes")

    return {
        "phone_number": data.get("phone_number"),
        "property_address": data.get("property_address") or merged.get("property_address"),
        "estimated_arv": data.get("estimated_arv") or data.get("arv") or merged.get("estimated_arv"),
        "asking_price": asking_price or merged.get("asking_price"),
        "repair_cost": data.get("repair_cost") or merged.get("repair_cost"),
        "bedrooms": data.get("bedrooms") or merged.get("bedrooms"),
        "bathrooms": data.get("bathrooms") or merged.get("bathrooms"),
        "square_footage": data.get("square_footage") or merged.get("square_footage"),
        "year_built": data.get("year_built") or merged.get("year_built"),
        "lead_source": data.get("lead_source") or merged.get("lead_source"),
        "condition_notes": condition_notes,
        "conversation_log": history,
        "call_summary": summary,
        "summary_history": summary_history,
        "offer_history": offer_history,
        "verbal_offer_amount": verbal_offer or merged.get("verbal_offer_amount"),
        "min_offer_amount": min_offer or merged.get("min_offer_amount"),
        "max_offer_amount": max_offer or merged.get("max_offer_amount"),
        "follow_up_date": data.get("follow_up_date") or merged.get("follow_up_date"),
        "follow_up_reason": data.get("follow_up_reason") or merged.get("follow_up_reason"),
        "follow_up_set_by": data.get("follow_up_set_by") or merged.get("follow_up_set_by"),
        "conversation_stage": merged.get("conversation_stage") or "Introduction + Rapport"
    }

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    phone = data.get("phone_number")
    seller_input = data.get("seller_input", "")
    arv = data.get("arv") or data.get("estimated_arv")
    repair = data.get("repair_cost")

    if not phone or not seller_input:
        return jsonify({"error": "Missing phone_number or seller_input"}), 400

    tone = detect_tone(seller_input)
    intent = detect_seller_intent(seller_input)
    memory = get_seller_memory(phone)

    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    try:
        vector = client.embeddings.create(
            input=[seller_input],
            model="text-embedding-3-small"
        ).data[0].embedding
        result = index.query(vector=vector, top_k=3, include_metadata=True)
        top_nepq = [m.metadata["response"] for m in result.matches]
    except:
        top_nepq = []

    min_offer, max_offer = calculate_offer_range(arv, repair)
    verbal_offer = min_offer if min_offer else None

    summary = summarize_messages([m["content"] for m in conversation_memory["history"] if m["role"] == "user"])

    system_prompt = f"""
You are SARA, a sharp, emotionally intelligent AI trained in NEPQ sales and real estate acquisitions.
Seller Tone: {tone}
Seller Intent: {intent}

Summary: {summary}

Use max 3 counteroffers. Avoid talking ROI %. Frame in terms of risk, cost, time.
Example Objection Responses: {" | ".join(top_nepq) if top_nepq else "None"}
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages += conversation_memory["history"]

    while num_tokens_from_messages(messages) > 3000:
        messages.pop(1)

    try:
        res = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        )
        reply = res.choices[0].message.content
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    conversation_memory["history"].append({"role": "assistant", "content": reply})

    update_payload = generate_update_payload(data, memory, conversation_memory["history"], summary, verbal_offer, min_offer, max_offer)
    update_seller_memory(phone, update_payload)

    return jsonify({
        "content": reply,
        "tone": tone,
        "intent": intent,
        "summary": summary,
        "verbal_offer": verbal_offer,
        "min_offer": min_offer,
        "max_offer": max_offer
    })

@app.route("/", methods=["GET"])
def index():
    return "✅ SARA Webhook is running."

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")







