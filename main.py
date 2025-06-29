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
pp = pprint.PrettyPrinter(indent=2)

# Conversation flow stages
conversation_stages = [
    "introduction",
    "setting_stage",
    "motivation",
    "wholesaler_vs_traditional",
    "wholesale_process",
    "review_arv",
    "home_tour",
    "review_repairs",
    "offer",
    "next_steps"
]

# Tone detection
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
        if any(kw in lowered for kw in keywords):
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

def generate_summary(user_messages):
    prompt = [
        {"role": "system", "content": "Summarize the seller's motivation, timeline, condition, and price point."},
        {"role": "user", "content": "\n".join(user_messages)}
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=prompt,
        temperature=0.5
    )
    return response.choices[0].message.content

def calculate_investor_price(arv, repair_cost, target_roi):
    realtor_fees = arv * 0.06
    holding_costs = 0.01 * (arv - repair_cost) * 3
    investor_profit = target_roi * (arv - repair_cost)
    max_price = arv - (realtor_fees + holding_costs + repair_cost + investor_profit)
    return round(max_price, 2)

def num_tokens_from_messages(messages, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    tokens_per_message = 3
    tokens_per_name = 1
    total = 0
    for message in messages:
        total += tokens_per_message
        for key, value in message.items():
            total += len(encoding.encode(value))
            if key == "name":
                total += tokens_per_name
    return total + 3

def get_next_stage(current_stage):
    try:
        index = conversation_stages.index(current_stage)
        return conversation_stages[index + 1] if index + 1 < len(conversation_stages) else "offer"
    except:
        return "introduction"

def generate_update_payload(data, seller_data, history, summary, min_offer, max_offer, verbal_offer, stage):
    summary_history = seller_data.get("summary_history")
    if isinstance(summary_history, str):
        try:
            summary_history = json.loads(summary_history)
        except:
            summary_history = []
    summary_history = summary_history or []
    summary_history.append({"timestamp": datetime.utcnow().isoformat(), "summary": summary})

    offer_history = seller_data.get("offer_history") or []
    if verbal_offer:
        offer_history.append({
            "amount": round(verbal_offer, 2),
            "timestamp": datetime.utcnow().isoformat()
        })

    return {
        "conversation_log": history,
        "call_summary": summary,
        "summary_history": summary_history,
        "asking_price": data.get("asking_price") or seller_data.get("asking_price"),
        "repair_cost": data.get("repair_cost") or seller_data.get("repair_cost"),
        "estimated_arv": data.get("estimated_arv") or data.get("arv") or seller_data.get("estimated_arv"),
        "min_offer_amount": min_offer,
        "max_offer_amount": max_offer,
        "verbal_offer_amount": verbal_offer,
        "offer_history": offer_history,
        "follow_up_date": data.get("follow_up_date") or seller_data.get("follow_up_date"),
        "follow_up_reason": data.get("follow_up_reason") or seller_data.get("follow_up_reason"),
        "follow_up_set_by": data.get("follow_up_set_by") or seller_data.get("follow_up_set_by"),
        "phone_number": data.get("phone_number"),
        "property_address": data.get("property_address") or seller_data.get("property_address"),
        "condition_notes": data.get("condition_notes") or seller_data.get("condition_notes"),
        "lead_source": data.get("lead_source") or seller_data.get("lead_source"),
        "bedrooms": data.get("bedrooms") or seller_data.get("bedrooms"),
        "bathrooms": data.get("bathrooms") or seller_data.get("bathrooms"),
        "square_footage": data.get("square_footage") or seller_data.get("square_footage"),
        "year_built": data.get("year_built") or seller_data.get("year_built"),
        "conversation_stage": stage
    }

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    seller_input = data.get("seller_input", "")
    phone = data.get("phone_number")
    if not seller_input or not phone:
        return jsonify({"error": "Missing seller_input or phone_number"}), 400

    seller_data = get_seller_memory(phone) or {}
    conversation_stage = get_next_stage(seller_data.get("conversation_stage", "introduction"))
    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    try:
        vector = client.embeddings.create(input=[seller_input], model="text-embedding-3-small").data[0].embedding
        result = index.query(vector=vector, top_k=3, include_metadata=True)
        top_pairs = [m.metadata["response"] for m in result.matches]
    except:
        top_pairs = []

    tone = detect_tone(seller_input)
    intent = detect_seller_intent(seller_input)
    contradiction_note = detect_contradiction(seller_input, seller_data)
    contradiction_text = f"⚠️ Seller contradiction(s): {', '.join(contradiction_note)}." if contradiction_note else ""

    arv = float(data.get("arv") or seller_data.get("estimated_arv") or 0)
    repairs = float(data.get("repair_cost") or seller_data.get("repair_cost") or 0)
    min_offer = calculate_investor_price(arv, repairs, 0.30) if arv and repairs else None
    max_offer = calculate_investor_price(arv, repairs, 0.10) if arv and repairs else None
    verbal_offer = min_offer  # Initial for now

    summary = generate_summary([m["content"] for m in conversation_memory["history"] if m["role"] == "user"])

    prompt = f"""
{contradiction_text}
Previous Summary:
{summary}
Current Stage: {conversation_stage}
Seller Tone: {tone}
Seller Intent: {intent}

You are SARA, an experienced and emotionally aware real estate acquisitions expert.
Respond using NEPQ-style framing and guide the conversation naturally from this stage.
Negotiation Range: ${min_offer} to ${max_offer}
Include relevant NEPQ if useful:
{'; '.join(top_pairs) if top_pairs else 'No NEPQ examples found.'}
"""

    messages = [{"role": "system", "content": prompt}]
    messages.extend(conversation_memory["history"])
    while num_tokens_from_messages(messages) > 3000:
        messages.pop(1)

    try:
        result = client.chat.completions.create(model="gpt-4", messages=messages, temperature=0.7)
        reply = result.choices[0].message.content
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    conversation_memory["history"].append({"role": "assistant", "content": reply})

    update_payload = generate_update_payload(
        data=data,
        seller_data=seller_data,
        history=conversation_memory["history"],
        summary=summary,
        min_offer=min_offer,
        max_offer=max_offer,
        verbal_offer=verbal_offer,
        stage=conversation_stage
    )

    print("🚨 DEBUG Payload:")
    pp.pprint(update_payload)

    update_seller_memory(phone, update_payload)

    return jsonify({
        "content": reply,
        "tone": tone,
        "intent": intent,
        "summary": summary,
        "reasoning": f"Start at ${min_offer}, max ${max_offer}",
        "contradictions": contradiction_note,
        "conversation_stage": conversation_stage
    })

@app.route("/", methods=["GET"])
def index():
    return "✅ SARA Webhook is live"

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")









