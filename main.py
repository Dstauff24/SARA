from flask import Flask, request, jsonify
import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
import tiktoken
from seller_memory_service import get_seller_memory, update_seller_memory
from datetime import datetime
import json

# Load environment
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index = os.getenv("PINECONE_INDEX")

client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index)

app = Flask(__name__)
conversation_memory = {"history": []}
MEMORY_LIMIT = 5

def num_tokens_from_messages(messages, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    tokens_per_message, tokens_per_name, num_tokens = 3, 1, 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    return num_tokens + 3

@app.route("/", methods=["GET"])
def index():
    return "✅ SARA Webhook running!"

def calculate_offer(arv, repair_cost, roi):
    fees = arv * 0.06
    holding = 0.01 * (arv - repair_cost) * 3
    profit = roi * (arv - repair_cost)
    return round(arv - (fees + holding + repair_cost + profit), 2)

def detect_tone(text):
    lowered = text.lower()
    tone_map = {
        "angry": ["pissed", "frustrated"], "skeptical": ["scam", "don't believe"],
        "curious": ["wondering", "what would you offer"], "hesitant": ["maybe", "not sure"],
        "urgent": ["asap", "foreclosure"], "emotional": ["passed", "divorce", "lost job"],
        "motivated": ["ready", "just want out"], "doubtful": ["too low"],
        "withdrawn": ["leave me alone"], "neutral": [], "friendly": ["thanks"], "direct": ["how much"]
    }
    for tone, keywords in tone_map.items():
        if any(k in lowered for k in keywords): return tone
    return "neutral"

def detect_intent(text):
    lowered = text.lower()
    if any(k in lowered for k in ["how much", "offer"]): return "price_sensitive"
    if any(k in lowered for k in ["foreclosure", "behind"]): return "distressed"
    if any(k in lowered for k in ["maybe", "not sure"]): return "on_fence"
    if any(k in lowered for k in ["leave me alone", "stop calling"]): return "cold"
    if any(k in lowered for k in ["vacant", "tenant"]): return "landlord"
    return "general_inquiry"

def detect_contradictions(text, seller_data):
    contradictions = []
    if seller_data.get("asking_price") and str(seller_data["asking_price"]) not in text:
        if any(w in text for w in ["price", "$", "want"]):
            contradictions.append("asking_price")
    if seller_data.get("condition_notes") and "roof" in seller_data["condition_notes"].lower():
        if "roof is fine" in text.lower():
            contradictions.append("condition_notes")
    return contradictions

def summarize_conversation(user_msgs):
    prompt = [{"role": "system", "content": "Summarize key points from this seller conversation."},
              {"role": "user", "content": "\n".join(user_msgs)}]
    return client.chat.completions.create(model="gpt-4", messages=prompt).choices[0].message.content

def generate_update_payload(data, seller_data, conversation_history, summary, min_offer, max_offer, verbal_offer):
    history_log = seller_data.get("summary_history")
    if isinstance(history_log, str):
        try: history_log = json.loads(history_log)
        except: history_log = []
    if not history_log: history_log = []
    history_log.append({"timestamp": datetime.utcnow().isoformat(), "summary": summary})

    offer_log = seller_data.get("offer_history") or []
    if verbal_offer:
        offer_log.append({"amount": round(verbal_offer, 2), "timestamp": datetime.utcnow().isoformat()})

    def safe_get(field): return data[field] if data.get(field) is not None else seller_data.get(field)

    return {
        "phone_number": safe_get("phone_number"),
        "conversation_log": conversation_history,
        "call_summary": summary,
        "summary_history": history_log,
        "offer_history": offer_log,
        "min_offer_amount": min_offer,
        "max_offer_amount": max_offer,
        "verbal_offer_amount": verbal_offer,
        "asking_price": safe_get("asking_price"),
        "repair_cost": safe_get("repair_cost"),
        "estimated_arv": safe_get("estimated_arv") or data.get("arv"),
        "follow_up_date": safe_get("follow_up_date"),
        "follow_up_reason": safe_get("follow_up_reason"),
        "follow_up_set_by": safe_get("follow_up_set_by"),
        "property_address": safe_get("property_address"),
        "condition_notes": safe_get("condition_notes"),
        "lead_source": safe_get("lead_source"),
        "bedrooms": safe_get("bedrooms"),
        "bathrooms": safe_get("bathrooms"),
        "square_footage": safe_get("square_footage"),
        "year_built": safe_get("year_built"),
        "conversation_stage": seller_data.get("conversation_stage", "Introduction + Rapport")
    }

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    seller_input = data.get("seller_input", "")
    phone = data.get("phone_number")
    arv = float(data.get("estimated_arv") or data.get("arv") or 0)
    repair = float(data.get("repair_cost") or 0)

    if not seller_input or not phone:
        return jsonify({"error": "Missing phone or input"}), 400

    seller_data = get_seller_memory(phone) or {}
    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    try:
        vector = client.embeddings.create(input=[seller_input], model="text-embedding-3-small").data[0].embedding
        top_pairs = [m.metadata["response"] for m in index.query(vector=vector, top_k=3, include_metadata=True).matches]
    except: top_pairs = []

    min_offer, max_offer, verbal_offer = None, None, None
    if arv and repair:
        min_offer = calculate_offer(arv, repair, 0.30)
        max_offer = calculate_offer(arv, repair, 0.15)
        verbal_offer = min_offer

    call_summary = summarize_conversation([m["content"] for m in conversation_memory["history"] if m["role"] == "user"])
    contradictions = detect_contradictions(seller_input, seller_data)
    contradiction_note = f"⚠️ Seller contradiction(s) noted: {', '.join(contradictions)}." if contradictions else ""

    walkthrough = "Use language like: 'Once we agree on terms, we’ll verify condition — nothing for you to worry about now.'"

    system_prompt = f"""
{contradiction_note}
Summary: {call_summary}
Tone: {detect_tone(seller_input)}
Intent: {detect_intent(seller_input)}
Instructions: Start @ ${min_offer}, negotiate up to ${max_offer}
{walkthrough}
NEPQ Examples: {'; '.join(top_pairs) if top_pairs else 'No matches.'}
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

    conversation_memory["history"].append({"role": "assistant", "content": reply})
    update_payload = generate_update_payload(data, seller_data, conversation_memory["history"], call_summary, min_offer, max_offer, verbal_offer)
    update_seller_memory(phone, update_payload)

    return jsonify({
        "content": reply,
        "tone": detect_tone(seller_input),
        "intent": detect_intent(seller_input),
        "summary": call_summary,
        "reasoning": f"Start @ ${min_offer}, max ${max_offer}",
        "contradictions": contradictions
    })

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")










