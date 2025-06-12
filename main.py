from flask import Flask, request, jsonify
import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
from datetime import datetime
import tiktoken
from seller_memory_service import get_seller_memory, update_seller_memory
from memory_summarizer import summarize_and_trim_memory

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX")

client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

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
        if any(kw in lowered for kw in keywords):
            return tone
    return "neutral"

def detect_seller_intent(text):
    text = text.lower()
    if any(kw in text for kw in ["how much", "offer", "price"]):
        return "price_sensitive"
    elif any(kw in text for kw in ["foreclosure", "behind", "bank"]):
        return "distressed"
    elif any(kw in text for kw in ["maybe", "thinking", "not sure", "depends"]):
        return "on_fence"
    elif any(kw in text for kw in ["stop calling", "not interested", "leave me alone"]):
        return "cold"
    elif any(kw in text for kw in ["vacant", "tenant", "rented", "investment"]):
        return "landlord"
    return "general_inquiry"

def detect_contradiction(seller_input, memory):
    contradictions = []
    if memory:
        if memory.get("asking_price") and str(memory["asking_price"]) not in seller_input:
            if any(word in seller_input for word in ["price", "$", "want", "need"]):
                contradictions.append("asking_price")
        if memory.get("condition_notes") and "roof" in memory["condition_notes"].lower():
            if "roof is fine" in seller_input.lower() or "new roof" in seller_input.lower():
                contradictions.append("condition_notes")
    return contradictions

def num_tokens_from_messages(messages, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    tokens_per_message = 3
    tokens_per_name = 1
    num_tokens = 0
    for msg in messages:
        num_tokens += tokens_per_message
        for k, v in msg.items():
            num_tokens += len(encoding.encode(v))
            if k == "name":
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
    seller_input = data.get("seller_input")
    phone_number = data.get("phone_number")

    if not seller_input or not phone_number:
        return jsonify({"error": "Missing seller_input or phone_number"}), 400

    # Append to memory
    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    # Retrieve prior memory
    seller_data = get_seller_memory(phone_number)

    # Summarize if long
    conversation_memory["history"], memory_summary = summarize_and_trim_memory(
        phone_number, conversation_memory["history"]
    )

    seller_tone = detect_tone(seller_input)
    seller_intent = detect_seller_intent(seller_input)
    contradictions = detect_contradiction(seller_input, seller_data)

    # NEPQ retrieval
    try:
        vector = client.embeddings.create(
            input=[seller_input],
            model="text-embedding-3-small"
        ).data[0].embedding
        result = index.query(vector=vector, top_k=3, include_metadata=True)
        top_pairs = [match.metadata["response"] for match in result.matches]
    except:
        top_pairs = []

    # Offer logic
    investor_offer = ""
    offer_amount = None
    reasoning = ""
    if data.get("arv") and data.get("repair_cost"):
        try:
            arv = float(data["arv"])
            repair_cost = float(data["repair_cost"])
            start_offer = calculate_investor_price(arv, repair_cost, 0.30)
            max_offer = calculate_investor_price(arv, repair_cost, 0.15)
            hard_cap = calculate_investor_price(arv, repair_cost, 0.10)
            offer_amount = start_offer
            investor_offer = f"Start at ${start_offer}, negotiate up to ${max_offer}."
            reasoning = investor_offer
        except:
            pass

    # System prompt
    contradiction_note = f"⚠️ Contradictions: {', '.join(contradictions)}" if contradictions else ""
    walkthrough_guidance = "Once we agree on terms, we’ll verify condition — nothing for you to worry about now."

    system_prompt = f"""
{contradiction_note}
Previous Summary:
{memory_summary or 'No summary available.'}

You are SARA, a friendly and strategic real estate acquisitions assistant.
Seller Tone: {seller_tone}
Seller Intent: {seller_intent}
Negotiation: {investor_offer}

Walkthrough:
{walkthrough_guidance}

NEPQ Guidance:
{"; ".join(top_pairs) if top_pairs else "No NEPQ returned."}
""".strip()

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

    # Offer history
    offer_history = seller_data.get("offer_history", []) if seller_data else []
    if offer_amount:
        offer_history.append({
            "amount": offer_amount,
            "timestamp": datetime.utcnow().isoformat()
        })

    # Summary tracking
    summary_history = seller_data.get("summary_history", []) if seller_data else []
    if memory_summary:
        summary_history.append({
            "summary": memory_summary,
            "timestamp": datetime.utcnow().isoformat()
        })

    # Update Supabase
    update_payload = {
        "phone_number": phone_number,
        "conversation_log": conversation_memory["history"],
        "call_summary": memory_summary,
        "summary_history": summary_history,
        "last_offer_amount": offer_amount,
        "asking_price": data.get("asking_price"),
        "repair_cost": data.get("repair_cost"),
        "estimated_arv": data.get("arv"),
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
        "offer_history": offer_history
    }

    update_seller_memory(phone_number, update_payload)

    return jsonify({
        "content": reply,
        "tone": seller_tone,
        "intent": seller_intent,
        "summary": memory_summary,
        "nepq_examples": top_pairs,
        "reasoning": reasoning,
        "contradictions": contradictions
    })

@app.route("/", methods=["GET"])
def index():
    return "✅ SARA Webhook is live!"

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")






