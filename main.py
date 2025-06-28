from flask import Flask, request, jsonify
import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
import tiktoken
from seller_memory_service import get_seller_memory, update_seller_memory
from datetime import datetime
import json

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

# Short-Term Memory
conversation_memory = {
    "history": []
}
MEMORY_LIMIT = 5

# Tone Map
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

def detect_contradiction(input_text, seller_data):
    contradictions = []
    if seller_data:
        if seller_data.get("asking_price") and str(seller_data["asking_price"]) not in input_text:
            if any(word in input_text for word in ["price", "$", "want", "need"]):
                contradictions.append("asking_price")
        if seller_data.get("condition_notes") and "roof" in seller_data["condition_notes"].lower():
            if "roof is fine" in input_text.lower() or "no issues" in input_text.lower():
                contradictions.append("condition_notes")
    return contradictions

def generate_summary(user_messages):
    prompt = [
        {"role": "system", "content": "Summarize this seller conversation focusing on condition, timeline, price, and motivation."},
        {"role": "user", "content": "\n".join(user_messages)}
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=prompt,
        temperature=0.5
    )
    return response.choices[0].message.content

def num_tokens_from_messages(messages, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    tokens_per_message = 3
    tokens_per_name = 1
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

def calculate_investor_price(arv, repair_cost, roi):
    fees = arv * 0.06
    hold = 0.01 * (arv - repair_cost) * 3
    profit = roi * (arv - repair_cost)
    return round(arv - (fees + hold + repair_cost + profit), 2)

def generate_update_payload(data, seller_data, conversation_history, summary, offer_amount):
    def retain(key):
        return data.get(key) if data.get(key) is not None else seller_data.get(key)

    summary_history = seller_data.get("summary_history")
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

    offer_history = seller_data.get("offer_history") or []
    if offer_amount:
        offer_history.append({
            "amount": round(offer_amount, 2),
            "timestamp": datetime.utcnow().isoformat()
        })

    return {
        "conversation_log": conversation_history,
        "call_summary": summary,
        "summary_history": summary_history,
        "asking_price": retain("asking_price"),
        "repair_cost": retain("repair_cost"),
        "estimated_arv": retain("arv"),
        "last_offer_amount": round(offer_amount, 2) if offer_amount else seller_data.get("last_offer_amount"),
        "offer_history": offer_history,
        "follow_up_date": retain("follow_up_date"),
        "follow_up_reason": retain("follow_up_reason"),
        "follow_up_set_by": retain("follow_up_set_by"),
        "phone_number": data.get("phone_number"),
        "property_address": retain("property_address"),
        "condition_notes": retain("condition_notes"),
        "lead_source": retain("lead_source"),
        "bedrooms": retain("bedrooms"),
        "bathrooms": retain("bathrooms"),
        "square_footage": retain("square_footage"),
        "year_built": retain("year_built")
    }

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    seller_input = data.get("seller_input", "")
    phone_number = data.get("phone_number")
    arv = data.get("arv")
    repair_cost = data.get("repair_cost")

    if not seller_input or not phone_number:
        return jsonify({"error": "Missing input"}), 400

    seller_tone = detect_tone(seller_input)
    seller_intent = detect_seller_intent(seller_input)
    seller_data = get_seller_memory(phone_number)

    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    try:
        vector = client.embeddings.create(input=[seller_input], model="text-embedding-3-small").data[0].embedding
        result = index.query(vector=vector, top_k=3, include_metadata=True)
        top_pairs = [match.metadata["response"] for match in result.matches]
    except:
        top_pairs = []

    offer_amount = None
    investor_offer = ""
    if arv and repair_cost:
        try:
            arv = float(arv)
            repair_cost = float(repair_cost)
            offer_start = calculate_investor_price(arv, repair_cost, 0.30)
            offer_cap = calculate_investor_price(arv, repair_cost, 0.10)
            investor_offer = f"Start at ${offer_start}, cap at ${offer_cap}."
            offer_amount = offer_start
        except:
            pass

    summary = generate_summary([m["content"] for m in conversation_memory["history"] if m["role"] == "user"])
    contradictions = detect_contradiction(seller_input, seller_data)

    system_prompt = f"""
You are SARA, an emotionally intelligent real estate acquisitions assistant.

⚠️ Contradictions: {', '.join(contradictions)} if any.
Summary so far: {summary}
Seller Tone: {seller_tone}, Intent: {seller_intent}
Negotiation: {investor_offer}

NEPQ Guidance:
{'; '.join(top_pairs) if top_pairs else "No relevant training found."}
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_memory["history"])

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
    update_payload = generate_update_payload(data, seller_data or {}, conversation_memory["history"], summary, offer_amount)
    update_seller_memory(phone_number, update_payload)

    return jsonify({
        "content": reply,
        "tone": seller_tone,
        "intent": seller_intent,
        "summary": summary,
        "nepq_examples": top_pairs,
        "reasoning": investor_offer,
        "contradictions": contradictions
    })

@app.route("/", methods=["GET"])
def index():
    return "✅ SARA Webhook is live!"

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")


