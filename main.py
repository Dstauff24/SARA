# main.py

from flask import Flask, request, jsonify
import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
import tiktoken
from datetime import datetime
from seller_memory_service import get_seller_memory, update_seller_memory
from memory_summarizer import summarize_and_trim_memory

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

# Seller Tone Mapping
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

def calculate_investor_price(arv, repair_cost, target_roi):
    realtor_fees = arv * 0.06
    holding_costs = 0.01 * (arv - repair_cost) * 3
    investor_profit = target_roi * (arv - repair_cost)
    max_price = arv - (realtor_fees + holding_costs + repair_cost + investor_profit)
    return round(max_price, 2)

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    seller_input = data.get("seller_input", "")
    arv = data.get("arv")
    repair_cost = data.get("repair_cost")
    phone_number = data.get("phone_number")

    if not seller_input or not phone_number:
        return jsonify({"error": "Missing required fields"}), 400

    seller_tone = detect_tone(seller_input)
    seller_intent = detect_seller_intent(seller_input)
    seller_data = get_seller_memory(phone_number)

    trimmed_memory, summary = summarize_and_trim_memory(phone_number, conversation_memory.get("history", []))
    conversation_memory["history"] = trimmed_memory or []
    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    try:
        vector = client.embeddings.create(
            input=[seller_input],
            model="text-embedding-3-small"
        ).data[0].embedding
        result = index.query(vector=vector, top_k=3, include_metadata=True)
        top_pairs = [match.metadata["response"] for match in result.matches]
    except:
        top_pairs = []

    investor_offer = ""
    offer_amount = None
    if arv and repair_cost:
        try:
            arv = float(arv)
            repair_cost = float(repair_cost)
            initial_offer = calculate_investor_price(arv, repair_cost, 0.30)
            hard_cap_offer = calculate_investor_price(arv, repair_cost, 0.10)
            investor_offer = f"Start at ${initial_offer}, cap at ${hard_cap_offer}."
            offer_amount = initial_offer
        except:
            pass

    contradictions = detect_contradiction(seller_input, seller_data)
    contradiction_note = f"⚠️ Seller contradiction(s) noted: {', '.join(contradictions)}." if contradictions else ""

    walkthrough_logic = """
You are a virtual wholesaling assistant. Do not push for in-person walkthroughs unless final steps are reached.
Use language like: "Once we agree on terms, we’ll verify condition — nothing for you to worry about now."
"""

    system_prompt = f"""
{contradiction_note}

You are SARA, a sharp and emotionally intelligent real estate acquisitions expert.
Seller Tone: {seller_tone}
Seller Intent: {seller_intent}
Negotiation Instructions:
{investor_offer}

Walkthrough Guidance:
{walkthrough_logic}

Embed the following NEPQ-style examples into your natural conversation:
{"; ".join(top_pairs) if top_pairs else "No NEPQ matches returned."}

Avoid talking about ROI %. Frame our position in terms of real costs and risk.
Max 3 total counteroffers. Sound human, strategic, and calm.
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

    offer_history = seller_data.get("offer_history") if seller_data else []
    if offer_amount:
        offer_history = offer_history or []
        offer_history.append({"amount": offer_amount, "timestamp": datetime.utcnow().isoformat()})

    update_payload = {
        "conversation_log": conversation_memory["history"],
        "call_summary": summary,
        "summary_history": summary,
        "asking_price": data.get("asking_price"),
        "repair_cost": repair_cost,
        "estimated_arv": arv,
        "last_offer_amount": offer_amount,
        "offer_history": offer_history,
        "follow_up_date": data.get("follow_up_date"),
        "follow_up_reason": data.get("follow_up_reason"),
        "follow_up_set_by": data.get("follow_up_set_by"),
        "property_address": data.get("property_address"),
        "condition_notes": data.get("condition_notes"),
        "bedrooms": data.get("bedrooms"),
        "bathrooms": data.get("bathrooms"),
        "square_footage": data.get("square_footage"),
        "year_built": data.get("year_built"),
        "lead_source": data.get("lead_source"),
        "phone_number": phone_number
    }

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
    return "✅ SARA Webhook is running!"

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")




