from flask import Flask, request, jsonify
import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
import tiktoken
from seller_memory_service import get_seller_memory, update_seller_memory
from datetime import datetime
import re
import json

# Load env vars
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

tone_map = {
    "angry": ["ridiculous", "pissed", "frustrated"],
    "skeptical": ["scam", "not sure", "don’t believe"],
    "curious": ["wondering", "offer", "explain"],
    "hesitant": ["maybe", "thinking", "idk"],
    "urgent": ["asap", "foreclosure", "eviction"],
    "emotional": ["passed", "divorce", "lost job"],
    "motivated": ["ready", "want to sell", "out"],
    "doubtful": ["too low", "never take", "not enough"],
    "withdrawn": ["leave me alone", "stop calling"],
    "friendly": ["hey", "thanks", "appreciate"],
    "direct": ["how much", "offer", "cut to it"],
    "neutral": []
}

def detect_tone(text):
    text = text.lower()
    for tone, keywords in tone_map.items():
        if any(k in text for k in keywords):
            return tone
    return "neutral"

def detect_seller_intent(text):
    text = text.lower()
    if any(k in text for k in ["offer", "price", "how much"]): return "price_sensitive"
    if any(k in text for k in ["foreclosure", "behind", "bank"]): return "distressed"
    if any(k in text for k in ["maybe", "thinking", "not sure"]): return "on_fence"
    if any(k in text for k in ["stop calling", "not interested"]): return "cold"
    if any(k in text for k in ["tenant", "rented", "investment"]): return "landlord"
    return "general_inquiry"

def detect_contradiction(seller_input, seller_data):
    contradictions = []
    if seller_data:
        if seller_data.get("asking_price"):
            try:
                asking_price = float(seller_data["asking_price"])
                if str(int(asking_price)) not in seller_input and "$" in seller_input:
                    contradictions.append("asking_price")
            except:
                pass
        if seller_data.get("condition_notes") and "roof" in seller_data["condition_notes"].lower():
            if "roof is fine" in seller_input.lower() or "no issues" in seller_input.lower():
                contradictions.append("condition_notes")
    return contradictions

def extract_asking_price(text):
    matches = re.findall(r'\$?\s?(\d{3,7})', text.replace(',', ''))
    numbers = [int(m) for m in matches if m.isdigit()]
    if numbers:
        return max(numbers)
    return None

def extract_condition_notes(text):
    if any(k in text.lower() for k in ["roof", "hvac", "paint", "kitchen", "carpet", "bathroom", "floor", "leak"]):
        return text
    return None

def summarize_messages(user_messages):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Summarize the seller's responses focusing on motivation, condition, price, and timeline."},
            {"role": "user", "content": "\n".join(user_messages)}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content

def num_tokens_from_messages(messages, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return sum(len(encoding.encode(m.get("content", ""))) + 4 for m in messages)

def calculate_offer(arv, repair_cost, roi):
    fees = arv * 0.06
    hold = 0.01 * (arv - repair_cost) * 3
    profit = roi * (arv - repair_cost)
    return round(arv - (fees + hold + repair_cost + profit), 2)

def generate_update_payload(data, memory, history, summary, verbal, min_offer, max_offer):
    summary_history = memory.get("summary_history", [])
    if isinstance(summary_history, str):
        try:
            summary_history = json.loads(summary_history)
        except:
            summary_history = []

    summary_history.append({"timestamp": datetime.utcnow().isoformat(), "summary": summary})

    offer_history = memory.get("offer_history", [])
    if verbal:
        offer_history.append({"amount": round(verbal, 2), "timestamp": datetime.utcnow().isoformat()})

    asking_price = data.get("asking_price") or extract_asking_price(data.get("seller_input", ""))
    condition_notes = data.get("condition_notes") or extract_condition_notes(data.get("seller_input", ""))

    return {
        "conversation_log": history,
        "call_summary": summary,
        "summary_history": summary_history,
        "offer_history": offer_history,
        "verbal_offer_amount": round(verbal, 2) if verbal else None,
        "min_offer_amount": round(min_offer, 2) if min_offer else None,
        "max_offer_amount": round(max_offer, 2) if max_offer else None,
        "asking_price": asking_price or memory.get("asking_price"),
        "repair_cost": data.get("repair_cost") or memory.get("repair_cost"),
        "estimated_arv": data.get("estimated_arv") or data.get("arv") or memory.get("estimated_arv"),
        "follow_up_date": data.get("follow_up_date") or memory.get("follow_up_date"),
        "follow_up_reason": data.get("follow_up_reason") or memory.get("follow_up_reason"),
        "follow_up_set_by": data.get("follow_up_set_by") or memory.get("follow_up_set_by"),
        "phone_number": data.get("phone_number"),
        "property_address": data.get("property_address") or memory.get("property_address"),
        "condition_notes": condition_notes or memory.get("condition_notes"),
        "lead_source": data.get("lead_source") or memory.get("lead_source"),
        "bedrooms": data.get("bedrooms") or memory.get("bedrooms"),
        "bathrooms": data.get("bathrooms") or memory.get("bathrooms"),
        "square_footage": data.get("square_footage") or memory.get("square_footage"),
        "year_built": data.get("year_built") or memory.get("year_built"),
        "conversation_stage": memory.get("conversation_stage", "Introduction + Rapport")
    }

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    seller_input = data.get("seller_input", "")
    phone = data.get("phone_number")
    if not seller_input or not phone:
        return jsonify({"error": "Missing seller_input or phone_number"}), 400

    memory = get_seller_memory(phone) or {}
    tone = detect_tone(seller_input)
    intent = detect_seller_intent(seller_input)
    contradictions = detect_contradiction(seller_input, memory)

    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    try:
        vector = client.embeddings.create(input=[seller_input], model="text-embedding-3-small").data[0].embedding
        result = index.query(vector=vector, top_k=3, include_metadata=True)
        examples = [m.metadata["response"] for m in result.matches]
    except:
        examples = []

    arv = float(data.get("arv") or data.get("estimated_arv") or memory.get("estimated_arv") or 0)
    repair = float(data.get("repair_cost") or memory.get("repair_cost") or 0)
    min_offer = calculate_offer(arv, repair, 0.30) if arv and repair else None
    max_offer = calculate_offer(arv, repair, 0.15) if arv and repair else None
    verbal_offer = min_offer if min_offer else None

    summary = summarize_messages([m["content"] for m in conversation_memory["history"] if m["role"] == "user"])

    prompt = f"""
⚠️ Contradictions: {', '.join(contradictions)}\n\nSummary:\n{summary}\n
Tone: {tone} | Intent: {intent}
Start at ${min_offer}, up to ${max_offer}.\n
NEPQ Examples: {"; ".join(examples) if examples else "None."}\n
Avoid ROI %, use cost/risk framing. Be strategic and emotionally intelligent.\n
Walkthrough: Once we agree on terms, we’ll verify condition — nothing for you to worry about now.
"""

    messages = [{"role": "system", "content": prompt}] + conversation_memory["history"]
    while num_tokens_from_messages(messages) > 3000:
        messages.pop(1)

    try:
        response = client.chat.completions.create(model="gpt-4", messages=messages, temperature=0.7)
        reply = response.choices[0].message.content
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
        "min_offer": min_offer,
        "max_offer": max_offer,
        "verbal_offer": verbal_offer,
        "contradictions": contradictions
    })

@app.route("/", methods=["GET"])
def index():
    return "✅ SARA is live and ready."

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")







