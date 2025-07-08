from flask import Flask, request, jsonify
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
import tiktoken
from seller_memory_service import get_seller_memory, update_seller_memory
from datetime import datetime
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
    "angry": ["this is ridiculous", "pissed", "frustrated"],
    "skeptical": ["not sure", "sounds like a scam"],
    "curious": ["wondering", "what would you offer"],
    "hesitant": ["maybe", "not sure"],
    "urgent": ["need to sell fast", "foreclosure"],
    "emotional": ["passed", "divorce", "lost job"],
    "motivated": ["ready to go", "just want out"],
    "doubtful": ["too low", "never take that"],
    "withdrawn": ["leave me alone", "not interested"],
    "neutral": [],
    "friendly": ["hey", "thanks for calling"],
    "direct": ["how much", "what’s the offer"]
}

def detect_tone(text):
    lowered = text.lower()
    for tone, keywords in tone_map.items():
        if any(kw in lowered for kw in keywords):
            return tone
    return "neutral"

def detect_seller_intent(text):
    lowered = text.lower()
    if any(kw in lowered for kw in ["how much", "offer", "price"]):
        return "price_sensitive"
    elif any(kw in lowered for kw in ["foreclosure", "behind", "bank"]):
        return "distressed"
    elif any(kw in lowered for kw in ["maybe", "thinking", "not sure"]):
        return "on_fence"
    elif any(kw in lowered for kw in ["stop calling", "not interested"]):
        return "cold"
    elif any(kw in lowered for kw in ["vacant", "tenant", "rented"]):
        return "landlord"
    return "general_inquiry"

def detect_contradiction(text, data):
    contradictions = []
    if data:
        if data.get("asking_price") and str(data["asking_price"]) not in text:
            if any(kw in text for kw in ["price", "$", "want", "need"]):
                contradictions.append("asking_price")
        if data.get("condition_notes") and "roof" in data["condition_notes"].lower():
            if "roof is fine" in text.lower() or "no issues" in text.lower():
                contradictions.append("condition_notes")
    return contradictions

def extract_asking_price(text):
    text = text.lower()
    match = re.search(r"\$?([1-9][0-9]{4,6})", text)
    if match:
        number = int(match.group(1))
        if number > 40000:  # Ignore house numbers, square feet
            return number
    return None

def extract_condition_notes(text):
    condition_keywords = ["roof", "hvac", "kitchen", "paint", "floor", "window", "bathroom"]
    matches = [kw for kw in condition_keywords if kw in text.lower()]
    return ", ".join(set(matches)) if matches else None

def generate_summary(messages):
    prompt = [
        {"role": "system", "content": "Summarize the seller conversation: motivation, price, condition, urgency."},
        {"role": "user", "content": "\n".join(messages)}
    ]
    res = client.chat.completions.create(
        model="gpt-4",
        messages=prompt,
        temperature=0.5
    )
    return res.choices[0].message.content

def num_tokens_from_messages(messages, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        for key, value in message.items():
            num_tokens += len(encoding.encode(value)) + 4
    return num_tokens

def calculate_investor_price(arv, repairs, roi):
    fees = arv * 0.06
    hold = 0.01 * (arv - repairs) * 3
    profit = roi * (arv - repairs)
    return round(arv - (fees + hold + repairs + profit), 2)

def generate_update_payload(data, existing_data, history, summary, verbal_offer, min_offer, max_offer):
    summary_history = existing_data.get("summary_history")
    if isinstance(summary_history, str):
        try:
            summary_history = json.loads(summary_history)
        except:
            summary_history = []
    if not summary_history:
        summary_history = []
    summary_history.append({"timestamp": datetime.utcnow().isoformat(), "summary": summary})

    offer_history = existing_data.get("offer_history") or []
    if verbal_offer:
        offer_history.append({"amount": verbal_offer, "timestamp": datetime.utcnow().isoformat()})

    def preserve(field):
        return data.get(field) if data.get(field) is not None else existing_data.get(field)

    return {
        "phone_number": data.get("phone_number"),
        "conversation_log": history,
        "call_summary": summary,
        "summary_history": summary_history,
        "asking_price": preserve("asking_price"),
        "repair_cost": preserve("repair_cost"),
        "condition_notes": preserve("condition_notes"),
        "property_address": preserve("property_address"),
        "lead_source": preserve("lead_source"),
        "bedrooms": preserve("bedrooms"),
        "bathrooms": preserve("bathrooms"),
        "square_footage": preserve("square_footage"),
        "year_built": preserve("year_built"),
        "estimated_arv": preserve("estimated_arv"),
        "follow_up_date": preserve("follow_up_date"),
        "follow_up_reason": preserve("follow_up_reason"),
        "follow_up_set_by": preserve("follow_up_set_by"),
        "conversation_stage": preserve("conversation_stage"),
        "verbal_offer_amount": verbal_offer if verbal_offer else existing_data.get("verbal_offer_amount"),
        "min_offer_amount": min_offer if min_offer else existing_data.get("min_offer_amount"),
        "max_offer_amount": max_offer if max_offer else existing_data.get("max_offer_amount"),
        "offer_history": offer_history
    }

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    seller_input = data.get("seller_input", "")
    phone = data.get("phone_number")
    if not seller_input or not phone:
        return jsonify({"error": "Missing fields"}), 400

    seller_data = get_seller_memory(phone) or {}
    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    try:
        vector = client.embeddings.create(input=[seller_input], model="text-embedding-3-small").data[0].embedding
        result = index.query(vector=vector, top_k=3, include_metadata=True)
        top_pairs = [match.metadata["response"] for match in result.matches]
    except:
        top_pairs = []

    seller_tone = detect_tone(seller_input)
    seller_intent = detect_seller_intent(seller_input)
    contradictions = detect_contradiction(seller_input, seller_data)
    contradiction_note = f"⚠️ Contradictions: {', '.join(contradictions)}." if contradictions else ""

    asking_price = extract_asking_price(seller_input)
    condition_notes = extract_condition_notes(seller_input)
    arv = data.get("estimated_arv") or seller_data.get("estimated_arv")
    repair = data.get("repair_cost") or seller_data.get("repair_cost")

    min_offer = max_offer = verbal_offer = None
    investor_offer = ""
    if arv and repair:
        try:
            arv = float(arv)
            repair = float(repair)
            min_offer = calculate_investor_price(arv, repair, 0.30)
            max_offer = calculate_investor_price(arv, repair, 0.15)
            verbal_offer = min_offer
            investor_offer = f"Start at ${min_offer}, go to ${max_offer}."
        except:
            pass

    summary = generate_summary([m["content"] for m in conversation_memory["history"] if m["role"] == "user"])

    system_prompt = f"""
{contradiction_note}
Summary: {summary}
You are SARA, an elite real estate acquisitions assistant.
Seller Tone: {seller_tone}
Intent: {seller_intent}
NEPQ: {top_pairs}
Investor Offer: {investor_offer}
"""

    messages = [{"role": "system", "content": system_prompt}] + conversation_memory["history"]
    while num_tokens_from_messages(messages) > 3000:
        messages.pop(1)

    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        )
        reply = completion.choices[0].message.content
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    conversation_memory["history"].append({"role": "assistant", "content": reply})

    update_payload = generate_update_payload(
        {**data, "asking_price": asking_price, "condition_notes": condition_notes},
        seller_data,
        conversation_memory["history"],
        summary,
        verbal_offer,
        min_offer,
        max_offer
    )

    update_seller_memory(phone, update_payload)

    return jsonify({
        "content": reply,
        "tone": seller_tone,
        "intent": seller_intent,
        "summary": summary,
        "reasoning": investor_offer,
        "contradictions": contradictions
    })

@app.route("/", methods=["GET"])
def index():
    return "✅ SARA is running!"

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")









