from flask import Flask, request, jsonify
import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
import tiktoken
import json
from datetime import datetime
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

conversation_memory = {
    "history": []
}
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

def detect_seller_intent(text):
    lowered = text.lower()
    if any(kw in lowered for kw in ["how much", "offer", "price", "what would you give"]):
        return "price_sensitive"
    elif any(kw in lowered for kw in ["foreclosure", "behind", "bank", "notice"]):
        return "distressed"
    elif any(kw in lowered for kw in ["maybe", "thinking", "not sure", "depends"]):
        return "on_fence"
    elif any(kw in lowered for kw in ["stop calling", "not interested", "leave me alone"]):
        return "cold"
    elif any(kw in lowered for kw in ["vacant", "tenant", "rented", "investment"]):
        return "landlord"
    else:
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

def num_tokens_from_messages(messages, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += 3
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
    num_tokens += 3
    return num_tokens

def calculate_offer_prices(arv, repair_cost):
    def calc(roi): 
        realtor_fees = arv * 0.06
        holding_costs = 0.01 * (arv - repair_cost) * 3
        profit = roi * (arv - repair_cost)
        return round(arv - (realtor_fees + holding_costs + repair_cost + profit), 2)
    min_offer = calc(0.30)
    max_offer = calc(0.15)
    return min_offer, max_offer

def generate_summary(user_texts):
    prompt = [
        {"role": "system", "content": "Summarize the following conversation for key details: motivation, condition, timeline, pricing. Be natural and concise."},
        {"role": "user", "content": "\n".join(user_texts)}
    ]
    result = client.chat.completions.create(model="gpt-4", messages=prompt, temperature=0.5)
    return result.choices[0].message.content

def extract_asking_price(text):
    import re
    price_match = re.search(r'\$?(\d{3,6})', text.replace(",", ""))
    if price_match:
        return int(price_match.group(1))
    return None

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    seller_input = data.get("seller_input", "")
    phone_number = data.get("phone_number", "")

    if not seller_input or not phone_number:
        return jsonify({"error": "Missing seller_input or phone_number"}), 400

    seller_data = get_seller_memory(phone_number) or {}
    seller_tone = detect_tone(seller_input)
    seller_intent = detect_seller_intent(seller_input)
    contradictions = detect_contradiction(seller_input, seller_data)

    # Handle memory
    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    # NEPQ logic
    try:
        vector = client.embeddings.create(input=[seller_input], model="text-embedding-3-small").data[0].embedding
        result = index.query(vector=vector, top_k=3, include_metadata=True)
        top_pairs = [match.metadata["response"] for match in result.matches]
    except:
        top_pairs = []

    # Offers
    arv = float(data.get("arv") or seller_data.get("estimated_arv") or 0)
    repair_cost = float(data.get("repair_cost") or seller_data.get("repair_cost") or 0)
    min_offer, max_offer = None, None
    verbal_offer = None

    if arv and repair_cost:
        min_offer, max_offer = calculate_offer_prices(arv, repair_cost)
        verbal_offer = round((min_offer + max_offer) / 2)

    # Summarize
    call_summary = generate_summary([m["content"] for m in conversation_memory["history"] if m["role"] == "user"])
    summary_history = seller_data.get("summary_history", [])
    if isinstance(summary_history, str):
        try: summary_history = json.loads(summary_history)
        except: summary_history = []
    summary_history.append({"timestamp": datetime.utcnow().isoformat(), "summary": call_summary})

    offer_history = seller_data.get("offer_history", [])
    if verbal_offer:
        offer_history.append({
            "amount": verbal_offer,
            "timestamp": datetime.utcnow().isoformat()
        })

    # Merge memory fields safely
    merged_data = {
        "conversation_log": conversation_memory["history"],
        "call_summary": call_summary,
        "summary_history": summary_history,
        "follow_up_date": data.get("follow_up_date") or seller_data.get("follow_up_date"),
        "follow_up_reason": data.get("follow_up_reason") or seller_data.get("follow_up_reason"),
        "follow_up_set_by": data.get("follow_up_set_by") or seller_data.get("follow_up_set_by"),
        "phone_number": phone_number,
        "property_address": data.get("property_address") or seller_data.get("property_address"),
        "condition_notes": data.get("condition_notes") or seller_data.get("condition_notes"),
        "lead_source": data.get("lead_source") or seller_data.get("lead_source"),
        "bedrooms": data.get("bedrooms") or seller_data.get("bedrooms"),
        "bathrooms": data.get("bathrooms") or seller_data.get("bathrooms"),
        "square_footage": data.get("square_footage") or seller_data.get("square_footage"),
        "year_built": data.get("year_built") or seller_data.get("year_built"),
        "estimated_arv": arv or seller_data.get("estimated_arv"),
        "repair_cost": repair_cost or seller_data.get("repair_cost"),
        "asking_price": extract_asking_price(seller_input) or data.get("asking_price") or seller_data.get("asking_price"),
        "min_offer_amount": min_offer or seller_data.get("min_offer_amount"),
        "max_offer_amount": max_offer or seller_data.get("max_offer_amount"),
        "verbal_offer_amount": verbal_offer or seller_data.get("verbal_offer_amount"),
        "offer_history": offer_history,
        "conversation_stage": seller_data.get("conversation_stage") or "Introduction + Rapport"
    }

    update_seller_memory(phone_number, merged_data)

    walkthrough_note = 'Once we agree on terms, we’ll verify the condition — nothing for you to worry about now.'
    system_prompt = f"""
You are SARA, a sharp and emotionally intelligent real estate acquisitions expert.
Seller Tone: {seller_tone}
Seller Intent: {seller_intent}
Walkthrough Guidance: {walkthrough_note}
{f"⚠️ Contradictions detected: {', '.join(contradictions)}" if contradictions else ""}
Call Summary: {call_summary}
Embed NEPQ logic. Avoid ROI %. Offer range ${min_offer}–${max_offer} if asked. Be warm, clear, and strategic.
"""

    messages = [{"role": "system", "content": system_prompt}] + conversation_memory["history"]
    while num_tokens_from_messages(messages) > 3000:
        messages.pop(1)

    response = client.chat.completions.create(model="gpt-4", messages=messages, temperature=0.7)
    reply = response.choices[0].message.content
    conversation_memory["history"].append({"role": "assistant", "content": reply})

    return jsonify({
        "content": reply,
        "tone": seller_tone,
        "intent": seller_intent,
        "summary": call_summary,
        "nepq_examples": top_pairs,
        "reasoning": f"Start at ${min_offer}, negotiate to ${max_offer}.",
        "contradictions": contradictions
    })

@app.route("/", methods=["GET"])
def index():
    return "✅ SARA Webhook is running!"

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")











