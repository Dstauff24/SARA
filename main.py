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

client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

app = Flask(__name__)

conversation_memory = {"history": []}
MEMORY_LIMIT = 5

tone_map = {
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
    lowered = text.lower()
    for tone, keywords in tone_map.items():
        if any(keyword in lowered for keyword in keywords):
            return tone
    return "neutral"

def detect_intent(text):
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

def detect_contradictions(text, data):
    contradictions = []
    if data:
        if data.get("asking_price") and str(data["asking_price"]) not in text:
            if any(word in text for word in ["price", "$", "want", "need"]):
                contradictions.append("asking_price")
        if data.get("condition_notes") and "roof" in data["condition_notes"].lower():
            if "roof is fine" in text.lower() or "no issues" in text.lower():
                contradictions.append("condition_notes")
    return contradictions

def extract_asking_price(text):
    matches = re.findall(r'\$\s?(\d{2,7})|(\d{5,7})', text.replace(',', ''))
    numbers = []
    for match in matches:
        for group in match:
            if group:
                try:
                    numbers.append(int(group))
                except ValueError:
                    continue
    if numbers:
        return max(numbers)
    return None

def extract_condition_notes(text):
    keywords = ["roof", "hvac", "kitchen", "foundation", "floor", "window", "bathroom", "paint", "carpet"]
    notes = [kw for kw in keywords if kw in text.lower()]
    return ", ".join(notes) if notes else None

def generate_summary(user_messages):
    prompt = [
        {"role": "system", "content": "Summarize the seller's motivation, property condition, pricing, and timeline in 2-3 sentences."},
        {"role": "user", "content": "\n".join(user_messages)}
    ]
    res = client.chat.completions.create(model="gpt-4", messages=prompt, temperature=0.5)
    return res.choices[0].message.content.strip()

def num_tokens(messages, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    num = 0
    for msg in messages:
        num += 3
        for k, v in msg.items():
            num += len(encoding.encode(v))
    return num + 3

def calculate_offer(arv, repairs, roi):
    try:
        arv = float(arv)
        repairs = float(repairs)
        fees = arv * 0.06
        hold = 0.01 * (arv - repairs) * 3
        profit = roi * (arv - repairs)
        return round(arv - (fees + hold + repairs + profit), 2)
    except:
        return None

def generate_update_payload(data, old, history, summary, verbal, min_offer, max_offer):
    summary_log = old.get("summary_history")
    if isinstance(summary_log, str):
        try:
            summary_log = json.loads(summary_log)
        except:
            summary_log = []
    if not summary_log:
        summary_log = []
    summary_log.append({"timestamp": datetime.utcnow().isoformat(), "summary": summary})

    offer_log = old.get("offer_history") or []
    if verbal:
        offer_log.append({"amount": round(verbal, 2), "timestamp": datetime.utcnow().isoformat()})

    asking_price = data.get("asking_price") or extract_asking_price(data.get("seller_input", ""))
    condition_notes = data.get("condition_notes") or extract_condition_notes(data.get("seller_input", ""))
    stage = old.get("conversation_stage") or "Introduction + Rapport"

    return {
        "phone_number": data.get("phone_number"),
        "conversation_log": history,
        "call_summary": summary,
        "summary_history": summary_log,
        "offer_history": offer_log,
        "verbal_offer_amount": verbal,
        "min_offer_amount": min_offer,
        "max_offer_amount": max_offer,
        "asking_price": asking_price,
        "repair_cost": data.get("repair_cost"),
        "estimated_arv": data.get("estimated_arv") or data.get("arv"),
        "follow_up_date": data.get("follow_up_date"),
        "follow_up_reason": data.get("follow_up_reason"),
        "follow_up_set_by": data.get("follow_up_set_by"),
        "property_address": data.get("property_address"),
        "condition_notes": condition_notes,
        "lead_source": data.get("lead_source"),
        "bedrooms": data.get("bedrooms"),
        "bathrooms": data.get("bathrooms"),
        "square_footage": data.get("square_footage"),
        "year_built": data.get("year_built"),
        "conversation_stage": stage
    }

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    seller_input = data.get("seller_input", "")
    phone = data.get("phone_number")
    arv = data.get("arv") or data.get("estimated_arv")
    repair = data.get("repair_cost")

    if not seller_input or not phone:
        return jsonify({"error": "Missing required fields"}), 400

    tone = detect_tone(seller_input)
    intent = detect_intent(seller_input)
    memory = get_seller_memory(phone)

    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    try:
        vector = client.embeddings.create(input=[seller_input], model="text-embedding-3-small").data[0].embedding
        result = index.query(vector=vector, top_k=3, include_metadata=True)
        top_pairs = [match.metadata["response"] for match in result.matches]
    except:
        top_pairs = []

    min_offer = max_offer = verbal_offer = None
    investor_logic = ""
    if arv and repair:
        try:
            min_offer = calculate_offer(arv, repair, 0.30)
            max_offer = calculate_offer(arv, repair, 0.15)
            investor_logic = f"Start at ${min_offer}, negotiate up to ${max_offer}."
            verbal_offer = min_offer
        except:
            pass

    summary = generate_summary([m["content"] for m in conversation_memory["history"] if m["role"] == "user"])
    contradictions = detect_contradictions(seller_input, memory)

    walkthrough = "Once we agree on terms, we’ll verify condition — nothing for you to worry about now."
    prompt = f"""
You are SARA, a warm and confident real estate acquisitions expert.
Seller Tone: {tone}
Seller Intent: {intent}
Negotiation Instructions: {investor_logic}
Walkthrough Guidance: {walkthrough}
Prior Summary: {summary}
Contradictions: {', '.join(contradictions) if contradictions else 'None'}

NEPQ Examples: {top_pairs if top_pairs else 'None'}
"""

    messages = [{"role": "system", "content": prompt}] + conversation_memory["history"]
    while num_tokens(messages) > 3000:
        messages.pop(1)

    try:
        response = client.chat.completions.create(model="gpt-4", messages=messages, temperature=0.7)
        reply = response.choices[0].message.content
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    conversation_memory["history"].append({"role": "assistant", "content": reply})
    update_payload = generate_update_payload(data, memory or {}, conversation_memory["history"], summary, verbal_offer, min_offer, max_offer)
    update_seller_memory(phone, update_payload)

    return jsonify({
        "content": reply,
        "tone": tone,
        "intent": intent,
        "summary": summary,
        "reasoning": investor_logic,
        "contradictions": contradictions,
        "nepq_examples": top_pairs
    })

@app.route("/", methods=["GET"])
def index():
    return "✅ SARA Webhook is running!"

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")









