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

def detect_tone(input_text):
    tone_map = {
        "angry": ["this is ridiculous", "pissed", "frustrated"],
        "skeptical": ["not sure", "scam", "don’t believe"],
        "curious": ["wondering", "offer", "explain"],
        "hesitant": ["don’t know", "maybe", "thinking"],
        "urgent": ["sell fast", "asap", "foreclosure"],
        "emotional": ["passed", "divorce", "lost job"],
        "motivated": ["ready", "want to sell", "just want out"],
        "doubtful": ["too low", "never take that"],
        "withdrawn": ["leave me alone", "stop calling"],
        "friendly": ["thanks", "no worries"],
        "direct": ["how much", "what’s the offer"]
    }
    lowered = input_text.lower()
    for tone, keywords in tone_map.items():
        if any(keyword in lowered for keyword in keywords):
            return tone
    return "neutral"

def detect_seller_intent(text):
    text = text.lower()
    if any(kw in text for kw in ["how much", "offer", "price"]):
        return "price_sensitive"
    elif any(kw in text for kw in ["foreclosure", "behind", "bank"]):
        return "distressed"
    elif any(kw in text for kw in ["maybe", "thinking", "not sure"]):
        return "on_fence"
    elif any(kw in text for kw in ["stop calling", "not interested"]):
        return "cold"
    elif any(kw in text for kw in ["tenant", "rented"]):
        return "landlord"
    else:
        return "general_inquiry"

def extract_numeric_value(text):
    match = re.search(r"\$?(\d{2,3}(?:,\d{3})*|\d+)(?:\.\d{1,2})?", text.replace(",", ""))
    if match:
        return float(match.group(1))
    return None

def extract_updated_asking_price(text):
    price = extract_numeric_value(text)
    if price and any(kw in text.lower() for kw in ["asking", "want", "hoping", "need", "looking"]):
        return price
    return None

def detect_contradiction(seller_input, seller_data):
    contradictions = []
    if seller_data:
        if seller_data.get("asking_price") and str(seller_data["asking_price"]) not in seller_input:
            if any(word in seller_input for word in ["price", "$", "want", "need"]):
                contradictions.append("asking_price")
    return contradictions

def generate_summary(user_messages):
    prompt = [
        {"role": "system", "content": "Summarize the conversation: motivation, condition, timeline, and price."},
        {"role": "user", "content": "\n".join(user_messages)}
    ]
    response = client.chat.completions.create(model="gpt-4", messages=prompt, temperature=0.5)
    return response.choices[0].message.content

def num_tokens_from_messages(messages, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = 0
    for msg in messages:
        tokens += 3
        for k, v in msg.items():
            tokens += len(encoding.encode(v))
    return tokens + 3

def calculate_investor_price(arv, repair_cost, target_roi):
    realtor_fees = arv * 0.06
    holding_costs = 0.01 * (arv - repair_cost) * 3
    investor_profit = target_roi * (arv - repair_cost)
    max_price = arv - (realtor_fees + holding_costs + repair_cost + investor_profit)
    return round(max_price, 2)

def generate_update_payload(data, seller_data, convo, summary, verbal_offer, min_offer, max_offer):
    summary_history = seller_data.get("summary_history") if seller_data else []
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

    offer_history = seller_data.get("offer_history") if seller_data else []
    if not offer_history:
        offer_history = []
    if verbal_offer:
        offer_history.append({
            "amount": verbal_offer,
            "timestamp": datetime.utcnow().isoformat()
        })

    return {
        "conversation_log": convo,
        "call_summary": summary,
        "summary_history": summary_history,
        "asking_price": data.get("asking_price"),
        "repair_cost": data.get("repair_cost"),
        "estimated_arv": data.get("estimated_arv") or data.get("arv"),
        "verbal_offer_amount": verbal_offer,
        "min_offer_amount": min_offer,
        "max_offer_amount": max_offer,
        "offer_history": offer_history,
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
        "conversation_stage": data.get("conversation_stage")
    }

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    seller_input = data.get("seller_input", "")
    phone_number = data.get("phone_number")
    arv = float(data.get("arv") or data.get("estimated_arv") or 0)
    repair_cost = float(data.get("repair_cost") or 0)

    if not seller_input or not phone_number:
        return jsonify({"error": "Missing required fields"}), 400

    seller_data = get_seller_memory(phone_number)
    stage = (seller_data or {}).get("conversation_stage", "Introduction + Rapport")

    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    seller_tone = detect_tone(seller_input)
    seller_intent = detect_seller_intent(seller_input)
    new_asking_price = extract_updated_asking_price(seller_input)

    try:
        vector = client.embeddings.create(
            input=[seller_input],
            model="text-embedding-3-small"
        ).data[0].embedding
        result = index.query(vector=vector, top_k=3, include_metadata=True)
        top_pairs = [m.metadata["response"] for m in result.matches]
    except:
        top_pairs = []

    min_offer = max_offer = verbal_offer = None
    if arv > 0 and repair_cost >= 0:
        max_offer = calculate_investor_price(arv, repair_cost, 0.15)
        min_offer = calculate_investor_price(arv, repair_cost, 0.30)
        verbal_offer = min_offer  # starting offer

    call_summary = generate_summary([m["content"] for m in conversation_memory["history"] if m["role"] == "user"])
    contradictions = detect_contradiction(seller_input, seller_data)
    contradiction_note = f"⚠️ Seller contradiction(s) noted: {', '.join(contradictions)}." if contradictions else ""

    walkthrough = """Use language like: "Once we agree on terms, we’ll verify condition — nothing for you to worry about now.""" 

    system_prompt = f"""
{contradiction_note}
Previous Summary:
{call_summary}

You are SARA, a sharp and emotionally intelligent real estate acquisitions expert.
Seller Tone: {seller_tone}
Seller Intent: {seller_intent}
Stage: {stage}
Negotiation Instructions:
Start at ${min_offer}, max at ${max_offer}

Walkthrough Guidance:
{walkthrough}

NEPQ examples:
{"; ".join(top_pairs) if top_pairs else "No NEPQ matches returned."}
"""

    messages = [{"role": "system", "content": system_prompt}] + conversation_memory["history"]
    while num_tokens_from_messages(messages) > 3000:
        messages.pop(1)

    try:
        response = client.chat.completions.create(model="gpt-4", messages=messages, temperature=0.7)
        reply = response.choices[0].message.content
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    conversation_memory["history"].append({"role": "assistant", "content": reply})
    if new_asking_price:
        data["asking_price"] = new_asking_price

    payload = generate_update_payload(data, seller_data or {}, conversation_memory["history"], call_summary, verbal_offer, min_offer, max_offer)
    update_seller_memory(phone_number, payload)

    return jsonify({
        "content": reply,
        "tone": seller_tone,
        "intent": seller_intent,
        "summary": call_summary,
        "reasoning": f"Start at ${min_offer}, max ${max_offer}",
        "contradictions": contradictions
    })

@app.route("/", methods=["GET"])
def index():
    return "✅ SARA Webhook is running!"

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")








