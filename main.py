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
    return "general_inquiry"

def detect_contradiction(input_text, previous_data):
    contradictions = []
    if previous_data:
        if previous_data.get("asking_price") and str(previous_data["asking_price"]) not in input_text:
            if any(word in input_text for word in ["price", "$", "want", "need"]):
                contradictions.append("asking_price")
        if previous_data.get("condition_notes") and "roof" in previous_data["condition_notes"].lower():
            if "roof is fine" in input_text.lower() or "no issues" in input_text.lower():
                contradictions.append("condition_notes")
    return contradictions

def generate_summary(user_messages):
    messages = [
        {"role": "system", "content": "Summarize the following conversation from a seller to highlight key points like motivation, condition, timeline, and pricing. Be concise and natural."},
        {"role": "user", "content": "\n".join(user_messages)}
    ]
    response = client.chat.completions.create(model="gpt-4", messages=messages, temperature=0.5)
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
    return num_tokens + 3

def calculate_investor_price(arv, repair_cost, roi):
    fees = arv * 0.06
    holding = 0.01 * (arv - repair_cost) * 3
    profit = roi * (arv - repair_cost)
    return round(arv - (fees + holding + repair_cost + profit), 2)

def generate_update_payload(data, existing, conversation_history, call_summary, verbal_offer, min_offer, max_offer):
    summary_history = existing.get("summary_history")
    if isinstance(summary_history, str):
        try: summary_history = json.loads(summary_history)
        except: summary_history = []
    if not summary_history: summary_history = []

    summary_history.append({
        "timestamp": datetime.utcnow().isoformat(),
        "summary": call_summary
    })

    offer_history = existing.get("offer_history") or []
    if verbal_offer:
        offer_history.append({
            "amount": round(verbal_offer, 2),
            "timestamp": datetime.utcnow().isoformat()
        })

    return {
        "conversation_log": conversation_history,
        "call_summary": call_summary,
        "summary_history": summary_history,
        "asking_price": data.get("asking_price"),
        "repair_cost": data.get("repair_cost"),
        "estimated_arv": data.get("arv"),
        "verbal_offer_amount": round(verbal_offer, 2) if verbal_offer else None,
        "min_offer_amount": round(min_offer, 2) if min_offer else None,
        "max_offer_amount": round(max_offer, 2) if max_offer else None,
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
        "year_built": data.get("year_built")
    }

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    seller_input = data.get("seller_input", "")
    phone_number = data.get("phone_number")
    arv = data.get("arv")
    repair = data.get("repair_cost")

    if not seller_input or not phone_number:
        return jsonify({"error": "Missing seller_input or phone_number"}), 400

    seller_tone = detect_tone(seller_input)
    seller_intent = detect_seller_intent(seller_input)
    memory = get_seller_memory(phone_number)

    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    try:
        embedding = client.embeddings.create(input=[seller_input], model="text-embedding-3-small").data[0].embedding
        result = index.query(vector=embedding, top_k=3, include_metadata=True)
        top_nepq = [match.metadata["response"] for match in result.matches]
    except:
        top_nepq = []

    min_offer = max_offer = verbal_offer = None
    if arv and repair:
        try:
            arv = float(arv)
            repair = float(repair)
            min_offer = calculate_investor_price(arv, repair, 0.30)
            max_offer = calculate_investor_price(arv, repair, 0.15)
            verbal_offer = min_offer
        except:
            pass

    summary = generate_summary([m["content"] for m in conversation_memory["history"] if m["role"] == "user"])
    contradictions = detect_contradiction(seller_input, memory)
    contradiction_note = f"⚠️ Seller contradiction(s) noted: {', '.join(contradictions)}" if contradictions else ""

    walkthrough_logic = '''
You are a virtual wholesaling assistant. Do not push for in-person walkthroughs unless final steps are reached.
Use language like: "Once we agree on terms, we’ll verify condition — nothing for you to worry about now."
'''

    prompt = f"""{contradiction_note}
Previous Summary:
{summary}

You are SARA, a sharp and emotionally intelligent real estate acquisitions expert.
Seller Tone: {seller_tone}
Seller Intent: {seller_intent}
Negotiation Instructions:
Start at ${min_offer}, negotiate up to ${max_offer}.

Walkthrough Guidance:
{walkthrough}

NEPQ Integration:
{"; ".join(top_nepq) if top_nepq else "No NEPQ matches found."}

Avoid ROI % talk. Frame with real cost + risk. Max 3 counteroffers. Stay human + strategic."""

    messages = [{"role": "system", "content": prompt}]
    messages.extend(conversation_memory["history"])
    while num_tokens_from_messages(messages) > 3000:
        messages.pop(1)

    try:
        chat_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        )
        reply = chat_response.choices[0].message.content
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    conversation_memory["history"].append({"role": "assistant", "content": reply})

    payload = generate_update_payload(data, memory or {}, conversation_memory["history"], summary, verbal_offer, min_offer, max_offer)
    update_seller_memory(phone_number, payload)

    return jsonify({
        "content": reply,
        "tone": seller_tone,
        "intent": seller_intent,
        "summary": summary,
        "nepq_examples": top_nepq,
        "reasoning": f"Start at ${min_offer}, max ${max_offer}",
        "contradictions": contradictions
    })

@app.route("/", methods=["GET"])
def index():
    return "✅ SARA Webhook running"

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")

