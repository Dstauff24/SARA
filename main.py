from flask import Flask, request, jsonify
import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
import tiktoken
from seller_memory_service import get_seller_memory, update_seller_memory
from datetime import datetime
import json
import pprint

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
    "angry": ["ridiculous", "pissed", "you people", "frustrated"],
    "skeptical": ["scam", "don’t believe", "sounds fishy"],
    "curious": ["just wondering", "what would you offer", "can you explain"],
    "hesitant": ["i don’t know", "maybe", "thinking"],
    "urgent": ["sell fast", "asap", "foreclosure", "eviction"],
    "emotional": ["mom passed", "divorce", "lost job", "hard time"],
    "motivated": ["ready to go", "want to sell", "just want out"],
    "doubtful": ["too low", "never take that"],
    "withdrawn": ["leave me alone", "not interested"],
    "neutral": [],
    "friendly": ["thanks for calling", "no worries"],
    "direct": ["how much", "what’s the offer"]
}

def detect_tone(text):
    text = text.lower()
    for tone, phrases in tone_map.items():
        if any(p in text for p in phrases):
            return tone
    return "neutral"

def detect_seller_intent(text):
    text = text.lower()
    if any(k in text for k in ["offer", "price", "what would you give"]):
        return "price_sensitive"
    if any(k in text for k in ["foreclosure", "behind", "bank", "notice"]):
        return "distressed"
    if any(k in text for k in ["maybe", "thinking", "not sure", "depends"]):
        return "on_fence"
    if any(k in text for k in ["stop calling", "not interested"]):
        return "cold"
    if any(k in text for k in ["vacant", "tenant", "investment"]):
        return "landlord"
    return "general_inquiry"

def detect_contradiction(input_text, memory):
    contradictions = []
    if memory:
        if memory.get("asking_price") and str(memory["asking_price"]) not in input_text:
            if any(w in input_text.lower() for w in ["price", "$", "want", "need"]):
                contradictions.append("asking_price")
        if memory.get("condition_notes") and "roof" in memory["condition_notes"].lower():
            if "roof is fine" in input_text.lower() or "no issues" in input_text.lower():
                contradictions.append("condition_notes")
    return contradictions

def num_tokens_from_messages(messages, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += 3
        for key, val in message.items():
            num_tokens += len(encoding.encode(val))
            if key == "name":
                num_tokens += 1
    return num_tokens + 3

def generate_summary(messages):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Summarize the conversation for key points: motivation, timeline, condition, and pricing."},
            {"role": "user", "content": "\n".join(messages)}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content

def calculate_investor_price(arv, repair_cost, target_roi):
    realtor_fees = arv * 0.06
    hold_cost = 0.01 * (arv - repair_cost) * 3
    target_profit = target_roi * (arv - repair_cost)
    return round(arv - (realtor_fees + hold_cost + repair_cost + target_profit), 2)

def generate_update_payload(data, memory, history, summary, offer_amount):
    summary_history = memory.get("summary_history")
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

    offer_history = memory.get("offer_history") or []
    if offer_amount:
        offer_history.append({
            "amount": round(offer_amount, 2),
            "timestamp": datetime.utcnow().isoformat()
        })

    return {
        "conversation_log": history,
        "call_summary": summary,
        "summary_history": summary_history,
        "asking_price": data.get("asking_price"),
        "repair_cost": data.get("repair_cost"),
        "estimated_arv": data.get("arv"),
        "last_offer_amount": round(offer_amount, 2) if offer_amount else None,
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
    phone = data.get("phone_number")
    input_text = data.get("seller_input", "")
    arv = data.get("arv")
    repair_cost = data.get("repair_cost")

    if not phone or not input_text:
        return jsonify({"error": "Missing phone number or seller input"}), 400

    tone = detect_tone(input_text)
    intent = detect_seller_intent(input_text)
    memory = get_seller_memory(phone)

    conversation_memory["history"].append({"role": "user", "content": input_text})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    try:
        vector = client.embeddings.create(
            input=[input_text],
            model="text-embedding-3-small"
        ).data[0].embedding
        result = index.query(vector=vector, top_k=3, include_metadata=True)
        top_pairs = [r.metadata["response"] for r in result.matches]
    except:
        top_pairs = []

    investor_offer = ""
    offer_amount = None
    if arv and repair_cost:
        try:
            arv = float(arv)
            repair_cost = float(repair_cost)
            initial_offer = calculate_investor_price(arv, repair_cost, 0.30)
            max_offer = calculate_investor_price(arv, repair_cost, 0.10)
            investor_offer = f"Start at ${initial_offer}, negotiate up to ${max_offer}."
            offer_amount = max_offer
        except:
            pass

    summary = generate_summary([m["content"] for m in conversation_memory["history"] if m["role"] == "user"])
    contradictions = detect_contradiction(input_text, memory)
    contradiction_note = f"⚠️ Contradictions noted: {', '.join(contradictions)}" if contradictions else ""

    system_prompt = f"""
{contradiction_note}
Last Summary: {summary}
You are SARA, a strategic wholesaling assistant.
Seller Tone: {tone}
Seller Intent: {intent}
Negotiation Strategy: {investor_offer}
Walkthrough Guidance: Don’t schedule until terms are agreed.
NEPQ examples: {"; ".join(top_pairs) if top_pairs else "None."}
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

    update_payload = generate_update_payload(data, memory or {}, conversation_memory["history"], summary, offer_amount)

    print("🚨 Supabase Payload:")
    pprint.pprint(update_payload)

    update_seller_memory(phone, update_payload)

    return jsonify({
        "content": reply,
        "tone": tone,
        "intent": intent,
        "summary": summary,
        "nepq_examples": top_pairs,
        "reasoning": investor_offer,
        "contradictions": contradictions
    })

@app.route("/", methods=["GET"])
def index():
    return "✅ SARA Webhook is active."

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")

