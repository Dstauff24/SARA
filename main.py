from flask import Flask, request, jsonify
import os
import re
import json
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
import tiktoken
from seller_memory_service import get_seller_memory, update_seller_memory
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
        if any(k in lowered for k in keywords):
            return tone
    return "neutral"

def detect_intent(text):
    text = text.lower()
    if any(k in text for k in ["how much", "offer", "price"]):
        return "price_sensitive"
    elif any(k in text for k in ["foreclosure", "behind", "bank", "notice"]):
        return "distressed"
    elif any(k in text for k in ["maybe", "thinking", "not sure"]):
        return "on_fence"
    elif any(k in text for k in ["stop calling", "not interested", "leave me alone"]):
        return "cold"
    elif any(k in text for k in ["vacant", "tenant", "rented", "investment"]):
        return "landlord"
    return "general_inquiry"

def detect_contradiction(text, memory):
    contradictions = []
    if memory:
        if memory.get("asking_price") and str(memory["asking_price"]) not in text:
            if any(word in text for word in ["price", "$", "want", "need"]):
                contradictions.append("asking_price")
        if memory.get("condition_notes") and "roof" in memory["condition_notes"].lower():
            if "roof is fine" in text.lower() or "no issues" in text.lower():
                contradictions.append("condition_notes")
    return contradictions

def summarize_messages(user_messages):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Summarize the following seller conversation (motivation, pricing, repairs, timeline). Be clear, specific, and natural."},
            {"role": "user", "content": "\n".join(user_messages)}
        ]
    )
    return response.choices[0].message.content

def extract_asking_price(text):
    matches = re.findall(r'\$?(\d{5,7})', text.replace(",", ""))
    numbers = [int(m) for m in matches if m.isdigit()]
    return max(numbers) if numbers else None

def extract_condition_notes(text):
    indicators = ["roof", "kitchen", "hvac", "ac", "bathroom", "carpet", "paint", "foundation", "windows", "leak"]
    return ". ".join([s.strip() for s in re.split(r'[.?!]', text) if any(w in s.lower() for w in indicators)])

def extract_verbal_offer(text):
    matches = re.findall(r'\$([0-9,]+)', text)
    try:
        values = [int(m.replace(",", "")) for m in matches]
        return max(values) if values else None
    except:
        return None

def calculate_price(arv, repairs, roi):
    arv = float(arv)
    repairs = float(repairs)
    fees = arv * 0.06
    hold = 0.01 * (arv - repairs) * 3
    profit = roi * (arv - repairs)
    return round(arv - (fees + hold + repairs + profit), 2)

def generate_update_payload(data, memory, history, summary, verbal_offer, min_offer, max_offer):
    def merge(field):
        return data.get(field) or memory.get(field)

    summary_history = memory.get("summary_history")
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

    offer_history = memory.get("offer_history") or []
    if verbal_offer:
        offer_history.append({
            "amount": round(verbal_offer, 2),
            "timestamp": datetime.utcnow().isoformat()
        })

    return {
        "conversation_log": history,
        "call_summary": summary,
        "summary_history": summary_history,
        "offer_history": offer_history,
        "asking_price": merge("asking_price") or extract_asking_price(data.get("seller_input", "")),
        "repair_cost": merge("repair_cost"),
        "estimated_arv": merge("estimated_arv"),
        "condition_notes": merge("condition_notes") or extract_condition_notes(data.get("seller_input", "")),
        "verbal_offer_amount": round(verbal_offer, 2) if verbal_offer else memory.get("verbal_offer_amount"),
        "min_offer_amount": round(min_offer, 2) if min_offer else memory.get("min_offer_amount"),
        "max_offer_amount": round(max_offer, 2) if max_offer else memory.get("max_offer_amount"),
        "phone_number": data.get("phone_number"),
        "property_address": merge("property_address"),
        "lead_source": merge("lead_source"),
        "bedrooms": merge("bedrooms"),
        "bathrooms": merge("bathrooms"),
        "square_footage": merge("square_footage"),
        "year_built": merge("year_built"),
        "follow_up_date": merge("follow_up_date"),
        "follow_up_reason": merge("follow_up_reason"),
        "follow_up_set_by": merge("follow_up_set_by"),
        "conversation_stage": merge("conversation_stage") or "Introduction + Rapport"
    }

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    seller_input = data.get("seller_input", "")
    phone = data.get("phone_number")

    if not seller_input or not phone:
        return jsonify({"error": "Missing input or phone"}), 400

    tone = detect_tone(seller_input)
    intent = detect_intent(seller_input)
    memory = get_seller_memory(phone)

    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    try:
        vector = client.embeddings.create(input=[seller_input], model="text-embedding-3-small").data[0].embedding
        results = index.query(vector=vector, top_k=3, include_metadata=True)
        top_pairs = [r.metadata["response"] for r in results.matches]
    except:
        top_pairs = []

    arv = data.get("estimated_arv") or memory.get("estimated_arv")
    repairs = data.get("repair_cost") or memory.get("repair_cost")

    min_offer = max_offer = verbal_offer = None
    if arv and repairs:
        try:
            min_offer = calculate_price(arv, repairs, 0.30)
            max_offer = calculate_price(arv, repairs, 0.15)
        except:
            pass

    summary = summarize_messages([m["content"] for m in conversation_memory["history"] if m["role"] == "user"])
    contradictions = detect_contradiction(seller_input, memory)

    walkthrough = "Once we agree on terms, we’ll verify condition — nothing for you to worry about now."
    investor_note = f"Start at ${min_offer}, negotiate up to ${max_offer}" if min_offer and max_offer else ""

    system_prompt = f"""
⚠️ Contradictions: {', '.join(contradictions)}\n
Summary so far: {summary}
Seller Tone: {tone} | Intent: {intent}
Negotiation Instructions: {investor_note}
Walkthrough Instructions: {walkthrough}
Examples: {"; ".join(top_pairs) if top_pairs else "None"}
Avoid ROI %. Focus on risk/cost.
Max 3 counters. Sound human, warm, strategic.
"""

    messages = [{"role": "system", "content": system_prompt}] + conversation_memory["history"]
    while sum(len(m["content"]) for m in messages) > 3000:
        messages.pop(1)

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7
    )
    reply = response.choices[0].message.content
    conversation_memory["history"].append({"role": "assistant", "content": reply})
    verbal_offer = extract_verbal_offer(reply) or min_offer

    payload = generate_update_payload(data, memory or {}, conversation_memory["history"], summary, verbal_offer, min_offer, max_offer)
    update_seller_memory(phone, payload)

    return jsonify({
        "content": reply,
        "tone": tone,
        "intent": intent,
        "summary": summary,
        "contradictions": contradictions,
        "min_offer": min_offer,
        "max_offer": max_offer,
        "verbal_offer": verbal_offer
    })

@app.route("/", methods=["GET"])
def home():
    return "✅ SARA Webhook Running"

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")





