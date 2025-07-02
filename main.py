# === START OF MAIN.PY ===
from flask import Flask, request, jsonify
import os
import json
import re
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

def extract_asking_price(text):
    price_match = re.search(r'\$?(\d{2,3}(?:,\d{3})+|\d{4,6})(?:\.\d{1,2})?', text.replace(",", ""))
    if price_match:
        try:
            return int(float(price_match.group(1)))
        except ValueError:
            return None
    return None

def detect_tone(input_text):
    tone_map = {
        "angry": ["this is ridiculous", "i’m pissed"],
        "motivated": ["ready to go", "want to sell"],
        "curious": ["wondering", "offer"],
        "hesitant": ["don’t know", "maybe", "thinking"],
        "neutral": []
    }
    lowered = input_text.lower()
    for tone, keywords in tone_map.items():
        if any(k in lowered for k in keywords):
            return tone
    return "neutral"

def detect_seller_intent(text):
    text = text.lower()
    if "offer" in text or "price" in text:
        return "price_sensitive"
    elif "vacant" in text or "tenant" in text:
        return "landlord"
    elif "foreclosure" in text:
        return "distressed"
    elif "maybe" in text or "not sure" in text:
        return "on_fence"
    elif "not interested" in text:
        return "cold"
    else:
        return "general_inquiry"

def detect_contradiction(seller_input, seller_data):
    contradictions = []
    if seller_data:
        if seller_data.get("asking_price") and str(seller_data["asking_price"]) not in seller_input:
            if any(w in seller_input for w in ["price", "$", "want", "need"]):
                contradictions.append("asking_price")
    return contradictions

def num_tokens_from_messages(messages, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += 3
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
    num_tokens += 3
    return num_tokens

def calculate_investor_price(arv, repair_cost, target_roi):
    realtor_fees = arv * 0.06
    holding = 0.01 * (arv - repair_cost) * 3
    profit = target_roi * (arv - repair_cost)
    return round(arv - (realtor_fees + holding + repair_cost + profit), 2)

def generate_summary(user_messages):
    prompt = [
        {"role": "system", "content": "Summarize key motivation, price, condition, timeline"},
        {"role": "user", "content": "\n".join(user_messages)}
    ]
    response = client.chat.completions.create(model="gpt-4", messages=prompt)
    return response.choices[0].message.content

def generate_update_payload(data, seller_data, conversation_history, call_summary, min_offer, max_offer, verbal_offer):
    summary_history = seller_data.get("summary_history") or []
    if isinstance(summary_history, str):
        try:
            summary_history = json.loads(summary_history)
        except:
            summary_history = []
    summary_history.append({
        "timestamp": datetime.utcnow().isoformat(),
        "summary": call_summary
    })

    offer_history = seller_data.get("offer_history") or []
    if verbal_offer:
        offer_history.append({
            "amount": round(verbal_offer, 2),
            "timestamp": datetime.utcnow().isoformat()
        })

    merged = {
        "conversation_log": conversation_history,
        "call_summary": call_summary,
        "summary_history": summary_history,
        "phone_number": data.get("phone_number"),
        "estimated_arv": data.get("estimated_arv") or seller_data.get("estimated_arv"),
        "property_address": data.get("property_address") or seller_data.get("property_address"),
        "condition_notes": data.get("condition_notes") or seller_data.get("condition_notes"),
        "lead_source": data.get("lead_source") or seller_data.get("lead_source"),
        "bedrooms": data.get("bedrooms") or seller_data.get("bedrooms"),
        "bathrooms": data.get("bathrooms") or seller_data.get("bathrooms"),
        "square_footage": data.get("square_footage") or seller_data.get("square_footage"),
        "year_built": data.get("year_built") or seller_data.get("year_built"),
        "follow_up_date": data.get("follow_up_date") or seller_data.get("follow_up_date"),
        "follow_up_reason": data.get("follow_up_reason") or seller_data.get("follow_up_reason"),
        "follow_up_set_by": data.get("follow_up_set_by") or seller_data.get("follow_up_set_by"),
        "min_offer_amount": min_offer or seller_data.get("min_offer_amount"),
        "max_offer_amount": max_offer or seller_data.get("max_offer_amount"),
        "verbal_offer_amount": verbal_offer or seller_data.get("verbal_offer_amount"),
        "offer_history": offer_history
    }

    # Dynamically detect asking price if not directly supplied
    asking_price = data.get("asking_price")
    if not asking_price:
        asking_price = extract_asking_price(data.get("seller_input", ""))
    merged["asking_price"] = asking_price or seller_data.get("asking_price")

    return merged

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    seller_input = data.get("seller_input", "")
    arv = data.get("estimated_arv")
    repair_cost = data.get("repair_cost")
    phone = data.get("phone_number")

    if not seller_input or not phone:
        return jsonify({"error": "Missing required fields"}), 400

    seller_data = get_seller_memory(phone)
    tone = detect_tone(seller_input)
    intent = detect_seller_intent(seller_input)
    contradiction = detect_contradiction(seller_input, seller_data)

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
    offer_reasoning = ""
    if arv and repair_cost:
        try:
            arv = float(arv)
            repair_cost = float(repair_cost)
            max_offer = calculate_investor_price(arv, repair_cost, 0.15)
            min_offer = calculate_investor_price(arv, repair_cost, 0.30)
            verbal_offer = min_offer
            offer_reasoning = f"Start at ${min_offer}, negotiate up to ${max_offer}."
        except:
            pass

    call_summary = generate_summary([m["content"] for m in conversation_memory["history"] if m["role"] == "user"])
    system_prompt = f"""
⚠️ Contradictions: {', '.join(contradiction)}.
Summary: {call_summary}
You are SARA, an emotionally intelligent real estate acquisition expert.
Tone: {tone} | Intent: {intent}
Negotiation: {offer_reasoning}
"""
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_memory["history"])

    while num_tokens_from_messages(messages) > 3000:
        messages.pop(1)

    response = client.chat.completions.create(model="gpt-4", messages=messages)
    reply = response.choices[0].message.content
    conversation_memory["history"].append({"role": "assistant", "content": reply})

    update_payload = generate_update_payload(data, seller_data or {}, conversation_memory["history"], call_summary, min_offer, max_offer, verbal_offer)
    update_seller_memory(phone, update_payload)

    return jsonify({
        "content": reply,
        "tone": tone,
        "intent": intent,
        "summary": call_summary,
        "reasoning": offer_reasoning,
        "contradictions": contradiction,
        "nepq_examples": top_pairs
    })

@app.route("/", methods=["GET"])
def index():
    return "✅ SARA Webhook is live!"

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")
# === END OF MAIN.PY ===






