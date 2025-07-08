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

def detect_contradiction(input_text, memory):
    contradictions = []
    if memory:
        if memory.get("asking_price") and str(memory["asking_price"]) not in input_text:
            if any(word in input_text for word in ["price", "$", "want", "need"]):
                contradictions.append("asking_price")
        if memory.get("condition_notes") and "roof" in memory["condition_notes"].lower():
            if "roof is fine" in input_text.lower() or "no issues" in input_text.lower():
                contradictions.append("condition_notes")
    return contradictions

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

def calculate_offer_range(arv, repair_cost):
    try:
        arv = float(arv)
        repair_cost = float(repair_cost)
        realtor_fees = arv * 0.06
        holding_costs = 0.01 * (arv - repair_cost) * 3
        min_profit = 0.15 * (arv - repair_cost)
        max_profit = 0.30 * (arv - repair_cost)
        min_offer = arv - (realtor_fees + holding_costs + repair_cost + min_profit)
        max_offer = arv - (realtor_fees + holding_costs + repair_cost + max_profit)
        return round(min_offer, 2), round(max_offer, 2)
    except:
        return None, None
def generate_summary(user_messages):
    summary_prompt = [
        {"role": "system", "content": "Summarize this seller conversation for motivation, condition, timeline, and pricing. Be concise and conversational."},
        {"role": "user", "content": "\n".join(user_messages)}
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=summary_prompt,
        temperature=0.5
    )
    return response.choices[0].message.content

def extract_fields_from_text(text):
    asking_price_match = re.search(r"\$?(\d{2,6})(?!\s?square)", text.replace(",", ""))
    condition_match = re.search(r"(needs.*|outdated.*|repairs.*|roof.*|hvac.*|paint.*|carpet.*)", text, re.IGNORECASE)

    asking_price = int(asking_price_match.group(1)) if asking_price_match else None
    condition_notes = condition_match.group(0) if condition_match else None

    return asking_price, condition_notes

def generate_update_payload(data, memory, conversation, summary, verbal_offer, min_offer, max_offer):
    summary_history = memory.get("summary_history")
    if isinstance(summary_history, str):
        try:
            summary_history = json.loads(summary_history)
        except:
            summary_history = []
    summary_history = summary_history or []
    summary_history.append({"timestamp": datetime.utcnow().isoformat(), "summary": summary})

    offer_history = memory.get("offer_history") or []
    if verbal_offer:
        offer_history.append({
            "amount": round(verbal_offer, 2),
            "timestamp": datetime.utcnow().isoformat()
        })

    return {
        "conversation_log": conversation,
        "call_summary": summary,
        "summary_history": summary_history,
        "asking_price": data.get("asking_price"),
        "repair_cost": data.get("repair_cost"),
        "estimated_arv": data.get("estimated_arv") or data.get("arv"),
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
        "year_built": data.get("year_built"),
        "conversation_stage": memory.get("conversation_stage", "Introduction + Rapport")
    }
@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    seller_input = data.get("seller_input", "")
    phone_number = data.get("phone_number")

    if not seller_input or not phone_number:
        return jsonify({"error": "Missing required fields"}), 400

    memory = get_seller_memory(phone_number) or {}

    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    seller_tone = detect_tone(seller_input)
    seller_intent = detect_seller_intent(seller_input)
    contradictions = detect_contradiction(seller_input, memory)

    asking_price, condition_notes = extract_fields_from_text(seller_input)
    if asking_price:
        data["asking_price"] = asking_price
    if condition_notes:
        data["condition_notes"] = condition_notes

    try:
        vector = client.embeddings.create(
            input=[seller_input],
            model="text-embedding-3-small"
        ).data[0].embedding

        result = index.query(vector=vector, top_k=3, include_metadata=True)
        top_pairs = [match.metadata["response"] for match in result.matches]
    except:
        top_pairs = []

    arv = data.get("arv") or data.get("estimated_arv") or memory.get("estimated_arv")
    repair_cost = data.get("repair_cost") or memory.get("repair_cost")

    min_offer = max_offer = verbal_offer = None
    negotiation_notes = ""
    if arv and repair_cost:
        try:
            arv = float(arv)
            repair_cost = float(repair_cost)
            max_offer = calculate_investor_price(arv, repair_cost, 0.10)
            min_offer = calculate_investor_price(arv, repair_cost, 0.30)
            verbal_offer = min_offer
            negotiation_notes = f"Start at ${min_offer}, negotiate up to ${max_offer}."
        except:
            pass

    call_summary = generate_summary([m["content"] for m in conversation_memory["history"] if m["role"] == "user"])
    contradiction_note = f"⚠️ Seller contradiction(s) noted: {', '.join(contradictions)}." if contradictions else ""

    walkthrough = "Once we agree on terms, we’ll verify condition — nothing for you to worry about now."

    system_prompt = f"""
{contradiction_note}
Previous Summary:
{call_summary}

You are SARA, a strategic and emotionally intelligent real estate acquisitions expert.
Seller Tone: {seller_tone}
Seller Intent: {seller_intent}
Negotiation Instructions:
{negotiation_notes}

Walkthrough Guidance:
{walkthrough}

Embed the following NEPQ-style examples into your natural conversation:
{"; ".join(top_pairs) if top_pairs else "No NEPQ matches returned."}

Avoid ROI %. Frame our position in terms of real costs and risk. Max 3 counteroffers.
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
    update_payload = generate_update_payload(data, memory, conversation_memory["history"], call_summary, verbal_offer, min_offer, max_offer)

    print("📦 SUPABASE UPDATE PAYLOAD:")
    print(json.dumps(update_payload, indent=2))

    update_seller_memory(phone_number, update_payload)

    return jsonify({
        "content": reply,
        "tone": seller_tone,
        "intent": seller_intent,
        "summary": call_summary,
        "nepq_examples": top_pairs,
        "reasoning": negotiation_notes,
        "contradictions": contradictions
    })

@app.route("/", methods=["GET"])
def index():
    return "✅ SARA Webhook is running!"

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")








