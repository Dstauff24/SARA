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
def detect_tone(input_text):
    lowered = input_text.lower()
    for tone, keywords in tone_map.items():
        if any(keyword in lowered for keyword in keywords):
            return tone
    return "neutral"

def detect_seller_intent(text):
    text = text.lower()
    if any(kw in text for kw in ["how much", "offer", "price", "what would you give"]):
        return "price_sensitive"
    elif any(kw in text for kw in ["foreclosure", "behind", "bank", "notice"]):
        return "distressed"
    elif any(kw in text for kw in ["maybe", "thinking", "not sure", "depends"]):
        return "on_fence"
    elif any(kw in text for kw in ["stop calling", "not interested", "leave me alone"]):
        return "cold"
    elif any(kw in text for kw in ["vacant", "tenant", "rented", "investment"]):
        return "landlord"
    else:
        return "general_inquiry"

def detect_contradiction(seller_input, seller_data):
    contradictions = []
    if seller_data:
        if seller_data.get("asking_price") and str(seller_data["asking_price"]) not in seller_input:
            if any(word in seller_input for word in ["price", "$", "want", "need"]):
                contradictions.append("asking_price")
        if seller_data.get("condition_notes") and "roof" in seller_data["condition_notes"].lower():
            if "roof is fine" in seller_input.lower() or "no issues" in seller_input.lower():
                contradictions.append("condition_notes")
    return contradictions

def extract_asking_price(text):
    matches = re.findall(r"\$?\s?(\d{4,7})", text.replace(',', ''))
    if matches:
        numbers = [int(n) for n in matches]
        for n in numbers:
            if 20000 < n < 2000000:
                return n
    return None

def extract_verbal_offer(text):
    matches = re.findall(r"\$?\s?(\d{4,7})", text.replace(',', ''))
    if matches:
        numbers = [int(n) for n in matches]
        for n in numbers:
            if 10000 < n < 2000000:
                return n
    return None
@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    seller_input = data.get("seller_input", "")
    phone_number = data.get("phone_number")
    if not seller_input or not phone_number:
        return jsonify({"error": "Missing seller_input or phone_number"}), 400

    memory = get_seller_memory(phone_number) or {}
    conversation_stage = memory.get("conversation_stage", "Introduction + Rapport")

    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    # Tone, intent, contradiction
    tone = detect_tone(seller_input)
    intent = detect_seller_intent(seller_input)
    contradictions = detect_contradiction(seller_input, memory)
    contradiction_note = f"⚠️ Seller contradiction(s): {', '.join(contradictions)}." if contradictions else ""

    # Embedding lookup
    try:
        vector = client.embeddings.create(
            input=[seller_input],
            model="text-embedding-3-small"
        ).data[0].embedding
        result = index.query(vector=vector, top_k=3, include_metadata=True)
        top_pairs = [match.metadata["response"] for match in result.matches]
    except:
        top_pairs = []

    # Repair cost / ARV / Offer logic
    arv = data.get("estimated_arv") or memory.get("estimated_arv")
    repair = data.get("repair_cost") or memory.get("repair_cost")
    try:
        min_offer = calculate_offer(float(arv), float(repair), 0.30)
        max_offer = calculate_offer(float(arv), float(repair), 0.15)
        investor_offer_instruction = f"Start at ${min_offer}, max at ${max_offer}"
    except:
        min_offer = max_offer = investor_offer_instruction = None

    summary = summarize_messages([m["content"] for m in conversation_memory["history"] if m["role"] == "user"])
    walkthrough = "Once we agree on terms, we’ll verify condition — nothing for you to worry about now."

    system_prompt = f"""
{contradiction_note}
Summary so far: {summary}
You are SARA, a sharp, emotionally intelligent acquisitions expert.
Seller tone: {tone} | Seller intent: {intent}
Conversation stage: {conversation_stage}
Offer guidance: {investor_offer_instruction}
Walkthrough logic: {walkthrough}
NEPQ examples: {"; ".join(top_pairs) if top_pairs else "No NEPQ examples available."}
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

    # Extract dynamic values
    asking_price = data.get("asking_price") or extract_asking_price(seller_input)
    verbal_offer = extract_verbal_offer(reply)

    # Build summary history
    summary_history = memory.get("summary_history", [])
    if isinstance(summary_history, str):
        try:
            summary_history = json.loads(summary_history)
        except:
            summary_history = []
    summary_history.append({
        "timestamp": datetime.utcnow().isoformat(),
        "summary": summary
    })

    # Offer history
    offer_history = memory.get("offer_history", [])
    if isinstance(offer_history, str):
        try:
            offer_history = json.loads(offer_history)
        except:
            offer_history = []
    if verbal_offer:
        offer_history.append({
            "amount": verbal_offer,
            "timestamp": datetime.utcnow().isoformat()
        })

    update_payload = {
        "conversation_log": conversation_memory["history"],
        "call_summary": summary,
        "summary_history": summary_history,
        "asking_price": asking_price,
        "repair_cost": repair,
        "estimated_arv": arv,
        "verbal_offer_amount": verbal_offer,
        "min_offer_amount": min_offer,
        "max_offer_amount": max_offer,
        "offer_history": offer_history,
        "follow_up_date": data.get("follow_up_date"),
        "follow_up_reason": data.get("follow_up_reason"),
        "follow_up_set_by": data.get("follow_up_set_by"),
        "property_address": data.get("property_address") or memory.get("property_address"),
        "condition_notes": data.get("condition_notes") or memory.get("condition_notes"),
        "lead_source": data.get("lead_source") or memory.get("lead_source"),
        "bedrooms": data.get("bedrooms") or memory.get("bedrooms"),
        "bathrooms": data.get("bathrooms") or memory.get("bathrooms"),
        "square_footage": data.get("square_footage") or memory.get("square_footage"),
        "year_built": data.get("year_built") or memory.get("year_built"),
        "conversation_stage": conversation_stage,
        "phone_number": phone_number
    }

    update_seller_memory(phone_number, update_payload)

    return jsonify({
        "content": reply,
        "tone": tone,
        "intent": intent,
        "summary": summary,
        "verbal_offer": verbal_offer,
        "min_offer": min_offer,
        "max_offer": max_offer,
        "asking_price": asking_price,
        "contradictions": contradictions
    })

@app.route("/", methods=["GET"])
def index():
    return "✅ SARA is live!"

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")







