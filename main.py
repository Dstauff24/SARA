from flask import Flask, request, jsonify
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from datetime import datetime
import tiktoken
from seller_memory_service import get_seller_memory, update_seller_memory

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
conversation_memory = {"history": []}
MEMORY_LIMIT = 5

# Tone Detection
tone_map = {
    "angry": ["ridiculous", "pissed", "frustrated"],
    "skeptical": ["scam", "don’t believe", "sounds fake"],
    "curious": ["wondering", "what would you offer", "explain"],
    "hesitant": ["maybe", "not sure", "thinking"],
    "urgent": ["asap", "foreclosure", "eviction"],
    "emotional": ["passed", "divorce", "lost job"],
    "motivated": ["ready to go", "just want out"],
    "doubtful": ["too low", "never take"],
    "withdrawn": ["stop calling", "leave me alone"],
    "neutral": [],
    "friendly": ["hey", "thanks", "no worries"],
    "direct": ["how much", "let’s cut to it"]
}

def detect_tone(text):
    text = text.lower()
    for tone, keywords in tone_map.items():
        if any(k in text for k in keywords):
            return tone
    return "neutral"

def detect_seller_intent(text):
    lowered = text.lower()
    if any(kw in lowered for kw in ["offer", "price", "how much"]):
        return "price_sensitive"
    elif any(kw in lowered for kw in ["behind", "foreclosure"]):
        return "distressed"
    elif any(kw in lowered for kw in ["maybe", "not sure"]):
        return "on_fence"
    elif any(kw in lowered for kw in ["leave me alone", "not interested"]):
        return "cold"
    elif any(kw in lowered for kw in ["tenant", "rented"]):
        return "landlord"
    return "general_inquiry"

def extract_asking_price(text):
    import re
    numbers = [int(n.replace(',', '')) for n in re.findall(r'\$\s?(\d{2,7})|(\d{5,7})', text.replace(',', '')) if n]
    plausible_prices = [n for tup in numbers for n in tup if n and 10000 < int(n) < 2000000]
    return max(plausible_prices, default=None) if plausible_prices else None

def generate_summary(messages):
    prompt = [
        {"role": "system", "content": "Summarize this seller conversation, including condition, motivation, and price."},
        {"role": "user", "content": "\n".join(messages)}
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=prompt,
        temperature=0.4
    )
    return response.choices[0].message.content

def detect_contradiction(text, memory):
    contradictions = []
    if memory.get("asking_price"):
        if str(memory["asking_price"]) not in text and any(x in text for x in ["price", "$", "want", "need"]):
            contradictions.append("asking_price")
    if memory.get("condition_notes") and "roof" in memory["condition_notes"].lower():
        if "roof is fine" in text.lower() or "no issues" in text.lower():
            contradictions.append("condition_notes")
    return contradictions

def num_tokens_from_messages(messages, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    tokens = 0
    for msg in messages:
        tokens += 4  # base per msg
        for key, val in msg.items():
            tokens += len(enc.encode(val))
    return tokens + 2

def calculate_offer_range(arv, repair_cost):
    try:
        arv = float(arv)
        repair_cost = float(repair_cost)
        fees = arv * 0.06
        hold = 0.01 * (arv - repair_cost) * 3
        min_offer = arv - (fees + hold + repair_cost + 0.15 * (arv - repair_cost))
        max_offer = arv - (fees + hold + repair_cost + 0.10 * (arv - repair_cost))
        return round(min_offer), round(max_offer)
    except:
        return None, None

def generate_update_payload(data, memory, conv, summary, offer_amount, min_offer, max_offer):
    summary_history = memory.get("summary_history")
    if isinstance(summary_history, str):
        try: summary_history = json.loads(summary_history)
        except: summary_history = []
    if not summary_history: summary_history = []

    summary_history.append({
        "timestamp": datetime.utcnow().isoformat(),
        "summary": summary
    })

    offer_history = memory.get("offer_history") or []
    if offer_amount:
        offer_history.append({
            "amount": offer_amount,
            "timestamp": datetime.utcnow().isoformat()
        })

    payload = {
        "conversation_log": conv,
        "call_summary": summary,
        "summary_history": summary_history,
        "follow_up_date": data.get("follow_up_date"),
        "follow_up_reason": data.get("follow_up_reason"),
        "follow_up_set_by": data.get("follow_up_set_by"),
        "phone_number": data.get("phone_number"),
        "property_address": data.get("property_address"),
        "condition_notes": data.get("condition_notes"),
        "lead_source": data.get("lead_source"),
        "bedrooms": data.get("bedrooms"),
        "bathrooms": data.get("bathrooms"),
        "square_footage": data.get("square_footage"),
        "year_built": data.get("year_built"),
        "conversation_stage": memory.get("conversation_stage", "Introduction + Rapport")
    }

    # Persist ARV
    payload["estimated_arv"] = data.get("estimated_arv") or memory.get("estimated_arv")

    # Extract & persist asking price
    asking_price = data.get("asking_price") or extract_asking_price(data.get("seller_input", ""))
    payload["asking_price"] = asking_price if asking_price else memory.get("asking_price")

    # Repair Cost
    payload["repair_cost"] = data.get("repair_cost") or memory.get("repair_cost")

    # Offers
    payload["verbal_offer_amount"] = offer_amount or memory.get("verbal_offer_amount")
    payload["min_offer_amount"] = min_offer or memory.get("min_offer_amount")
    payload["max_offer_amount"] = max_offer or memory.get("max_offer_amount")
    payload["offer_history"] = offer_history

    return payload

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    seller_input = data.get("seller_input", "")
    phone = data.get("phone_number", "")
    arv = data.get("estimated_arv")
    repair = data.get("repair_cost")

    if not phone or not seller_input:
        return jsonify({"error": "Missing phone number or seller input"}), 400

    memory = get_seller_memory(phone) or {}
    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    # Embedding search
    try:
        vector = client.embeddings.create(input=[seller_input], model="text-embedding-3-small").data[0].embedding
        results = index.query(vector=vector, top_k=3, include_metadata=True)
        top_examples = [m.metadata["response"] for m in results.matches]
    except:
        top_examples = []

    # Offer logic
    offer_text = ""
    verbal_offer = None
    min_offer = None
    max_offer = None
    if arv and repair:
        min_offer, max_offer = calculate_offer_range(arv, repair)
        if min_offer and max_offer:
            offer_text = f"Start at ${min_offer}, negotiate up to ${max_offer}."
            verbal_offer = min_offer

    summary = generate_summary([m["content"] for m in conversation_memory["history"] if m["role"] == "user"])
    contradictions = detect_contradiction(seller_input, memory)

    walkthrough = 'Once we agree on terms, we’ll verify condition — nothing for you to worry about now.'
    system_prompt = f"""
You are SARA, a warm and strategic real estate acquisitions expert.

Tone: {detect_tone(seller_input)}
Intent: {detect_seller_intent(seller_input)}
Summary: {summary}
Negotiation: {offer_text}
Walkthrough: {walkthrough}
Contradictions: {', '.join(contradictions) if contradictions else 'None'}
Examples: {"; ".join(top_examples) if top_examples else "None"}
"""

    messages = [{"role": "system", "content": system_prompt}] + conversation_memory["history"]
    while num_tokens_from_messages(messages) > 3000:
        messages.pop(1)

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        )
        reply = response.choices[0].message.content
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    conversation_memory["history"].append({"role": "assistant", "content": reply})
    update_payload = generate_update_payload(data, memory, conversation_memory["history"], summary, verbal_offer, min_offer, max_offer)
    update_seller_memory(phone, update_payload)

    return jsonify({
        "content": reply,
        "tone": detect_tone(seller_input),
        "intent": detect_seller_intent(seller_input),
        "summary": summary,
        "contradictions": contradictions,
        "reasoning": offer_text,
        "nepq_examples": top_examples
    })

@app.route("/", methods=["GET"])
def index():
    return "✅ SARA Webhook is live!"

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")








