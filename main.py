from flask import Flask, request, jsonify
import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
import tiktoken
import re
import json
from datetime import datetime
from seller_memory_service import get_seller_memory, update_seller_memory

# Load env
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
    "angry": ["ridiculous", "pissed", "you people", "frustrated"],
    "skeptical": ["not sure", "scam", "don’t believe"],
    "curious": ["wondering", "offer", "explain"],
    "hesitant": ["don’t know", "maybe", "thinking"],
    "urgent": ["fast", "asap", "foreclosure", "eviction"],
    "emotional": ["mom passed", "divorce", "lost job", "hard time"],
    "motivated": ["ready", "want to sell", "just want out"],
    "doubtful": ["too low", "never take that"],
    "withdrawn": ["leave alone", "stop calling", "not interested"],
    "neutral": [],
    "friendly": ["hey", "thanks", "no worries"],
    "direct": ["how much", "offer", "cut to it"]
}

def detect_tone(text):
    text = text.lower()
    for tone, keywords in tone_map.items():
        if any(k in text for k in keywords):
            return tone
    return "neutral"

def detect_intent(text):
    text = text.lower()
    if any(k in text for k in ["how much", "offer", "price"]): return "price_sensitive"
    elif any(k in text for k in ["foreclosure", "behind", "bank"]): return "distressed"
    elif any(k in text for k in ["maybe", "thinking", "not sure"]): return "on_fence"
    elif any(k in text for k in ["stop calling", "not interested"]): return "cold"
    elif any(k in text for k in ["vacant", "tenant", "rented"]): return "landlord"
    else: return "general_inquiry"

def detect_contradiction(text, data):
    contradictions = []
    if data:
        if data.get("asking_price") and str(data["asking_price"]) not in text and any(w in text for w in ["price", "$", "want", "need"]):
            contradictions.append("asking_price")
        if data.get("condition_notes") and "roof" in data["condition_notes"].lower():
            if "roof is fine" in text.lower() or "no issues" in text.lower():
                contradictions.append("condition_notes")
    return contradictions

def extract_price_from_text(text):
    text = text.lower().replace(",", "").replace("$", "")
    matches = re.findall(r"(\d{5,6})", text)
    for match in matches:
        try:
            num = int(match)
            if 10000 < num < 1000000:
                return num
        except:
            continue
    return None

def generate_summary(messages):
    prompt = [{"role": "system", "content": "Summarize seller motivation, price, condition, timeline."},
              {"role": "user", "content": "\n".join(messages)}]
    result = client.chat.completions.create(model="gpt-4", messages=prompt, temperature=0.5)
    return result.choices[0].message.content

def num_tokens_from_messages(messages, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    tokens_per_message = 3
    tokens_per_name = 1
    total = 0
    for msg in messages:
        total += tokens_per_message
        for key, val in msg.items():
            total += len(encoding.encode(val))
            if key == "name": total += tokens_per_name
    return total + 3

def calculate_investor_price(arv, repair, roi):
    fees = arv * 0.06
    hold = 0.01 * (arv - repair) * 3
    profit = roi * (arv - repair)
    max_price = arv - (fees + hold + repair + profit)
    return round(max_price, 2)

def generate_update_payload(data, seller_data, convo, summary, min_offer, max_offer, verbal_offer, stage):
    summary_history = seller_data.get("summary_history")
    if isinstance(summary_history, str):
        try: summary_history = json.loads(summary_history)
        except: summary_history = []
    if not summary_history: summary_history = []

    summary_history.append({
        "timestamp": datetime.utcnow().isoformat(),
        "summary": summary
    })

    offer_history = seller_data.get("offer_history") or []
    if verbal_offer:
        offer_history.append({
            "amount": round(verbal_offer, 2),
            "timestamp": datetime.utcnow().isoformat()
        })

    return {
        "conversation_log": convo,
        "call_summary": summary,
        "summary_history": summary_history,
        "asking_price": data.get("asking_price"),
        "repair_cost": data.get("repair_cost"),
        "estimated_arv": data.get("estimated_arv"),
        "min_offer_amount": min_offer,
        "max_offer_amount": max_offer,
        "verbal_offer_amount": verbal_offer,
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
        "conversation_stage": stage
    }

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    seller_input = data.get("seller_input", "")
    phone = data.get("phone_number")
    arv = data.get("estimated_arv")
    repair = data.get("repair_cost")

    if not seller_input or not phone:
        return jsonify({"error": "Missing required fields"}), 400

    extracted_price = extract_price_from_text(seller_input)
    if extracted_price and not data.get("asking_price"):
        data["asking_price"] = extracted_price

    tone = detect_tone(seller_input)
    intent = detect_intent(seller_input)
    seller_data = get_seller_memory(phone)

    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    try:
        vector = client.embeddings.create(input=[seller_input], model="text-embedding-3-small").data[0].embedding
        result = index.query(vector=vector, top_k=3, include_metadata=True)
        top_pairs = [m.metadata["response"] for m in result.matches]
    except:
        top_pairs = []

    min_offer = max_offer = verbal_offer = None
    if arv and repair:
        try:
            arv = float(arv)
            repair = float(repair)
            initial_offer = calculate_investor_price(arv, repair, 0.30)
            final_offer = calculate_investor_price(arv, repair, 0.15)
            verbal_offer = initial_offer
            min_offer = initial_offer
            max_offer = final_offer
        except:
            pass

    summary = generate_summary([m["content"] for m in conversation_memory["history"] if m["role"] == "user"])
    contradictions = detect_contradiction(seller_input, seller_data)
    note = f"⚠️ Seller contradiction(s) noted: {', '.join(contradictions)}." if contradictions else ""

    stage = seller_data.get("conversation_stage", "Introduction + Rapport")

    walkthrough = "Once we agree on terms, we’ll verify condition — nothing for you to worry about now."
    system_prompt = f"""{note}
Summary so far: {summary}
You are SARA, an emotionally intelligent wholesaling assistant.
Seller Tone: {tone}
Seller Intent: {intent}
Current Stage: {stage}
Walkthrough: {walkthrough}
Start at ${min_offer}, negotiate to ${max_offer} max.
Embed NEPQ if relevant: {"; ".join(top_pairs) if top_pairs else "No NEPQ available."}
"""

    messages = [{"role": "system", "content": system_prompt}] + conversation_memory["history"]

    while num_tokens_from_messages(messages) > 3000:
        messages.pop(1)

    try:
        result = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        )
        reply = result.choices[0].message.content
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    conversation_memory["history"].append({"role": "assistant", "content": reply})

    payload = generate_update_payload(
        data=data,
        seller_data=seller_data or {},
        convo=conversation_memory["history"],
        summary=summary,
        min_offer=min_offer,
        max_offer=max_offer,
        verbal_offer=verbal_offer,
        stage=stage
    )

    update_seller_memory(phone, payload)

    return jsonify({
        "content": reply,
        "tone": tone,
        "intent": intent,
        "summary": summary,
        "reasoning": f"Start at ${min_offer}, max out at ${max_offer}",
        "verbal_offer_amount": verbal_offer,
        "contradictions": contradictions
    })

@app.route("/", methods=["GET"])
def index():
    return "✅ SARA Webhook is running!"

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")







