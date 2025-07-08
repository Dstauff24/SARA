from flask import Flask, request, jsonify
import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
import tiktoken
import re
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

def extract_asking_price(text):
    matches = re.findall(r"\$?\s?(\d{2,7})", text.replace(",", ""))
    numbers = []
    for match in matches:
        try:
            num = int(match)
            if 10000 < num < 1000000:
                numbers.append(num)
        except:
            continue
    return max(numbers) if numbers else None

def extract_condition_notes(text):
    condition_keywords = ["roof", "hvac", "foundation", "plumbing", "electrical", "kitchen", "bathroom", "paint", "floor", "window"]
    sentences = re.split(r'[.?!]', text)
    notes = [s.strip() for s in sentences if any(word in s.lower() for word in condition_keywords)]
    return " ".join(notes) if notes else None

def summarize_messages(messages):
    prompt = [
        {"role": "system", "content": "Summarize the following seller conversation. Highlight motivation, pricing, timeline, and condition."},
        {"role": "user", "content": "\n".join(messages)}
    ]
    response = client.chat.completions.create(model="gpt-4", messages=prompt, temperature=0.5)
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
def calculate_investor_price(arv, repair_cost, target_roi):
    try:
        arv = float(arv)
        repair_cost = float(repair_cost)
        realtor_fees = arv * 0.06
        holding_costs = 0.01 * (arv - repair_cost) * 3
        investor_profit = target_roi * (arv - repair_cost)
        max_price = arv - (realtor_fees + holding_costs + repair_cost + investor_profit)
        return round(max_price, 2)
    except:
        return None

def generate_update_payload(data, memory, conversation_history, summary, verbal_offer, min_offer, max_offer):
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
    if verbal_offer:
        offer_history.append({
            "amount": round(verbal_offer, 2),
            "timestamp": datetime.utcnow().isoformat()
        })

    return {
        "conversation_log": conversation_history,
        "call_summary": summary,
        "summary_history": summary_history,
        "asking_price": data.get("asking_price") or extract_asking_price(data.get("seller_input", "")),
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
        "condition_notes": data.get("condition_notes") or extract_condition_notes(data.get("seller_input", "")),
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
        return jsonify({"error": "Missing seller_input or phone_number"}), 400

    memory = get_seller_memory(phone_number) or {}
    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    seller_tone = detect_tone(seller_input)
    seller_intent = detect_seller_intent(seller_input)
    contradictions = detect_contradiction(seller_input, memory)

    summary = generate_summary([m["content"] for m in conversation_memory["history"] if m["role"] == "user"])

    # Embedding and NEPQ retrieval
    try:
        vector = client.embeddings.create(input=[seller_input], model="text-embedding-3-small").data[0].embedding
        result = index.query(vector=vector, top_k=3, include_metadata=True)
        top_pairs = [match.metadata["response"] for match in result.matches]
    except:
        top_pairs = []

    # Offer logic
    arv = data.get("estimated_arv") or data.get("arv")
    repair_cost = data.get("repair_cost")
    verbal_offer = min_offer = max_offer = None
    investor_offer_text = ""

    if arv and repair_cost:
        try:
            arv = float(arv)
            repair_cost = float(repair_cost)
            min_offer = calculate_investor_price(arv, repair_cost, 0.30)
            max_offer = calculate_investor_price(arv, repair_cost, 0.15)
            verbal_offer = min_offer  # Starting point
            investor_offer_text = f"Start at ${min_offer}, negotiate up to ${max_offer}."
        except:
            pass

    stage = memory.get("conversation_stage", "Introduction + Rapport")
    walkthrough = "Once we agree on terms, we’ll verify condition — nothing for you to worry about now."

    system_prompt = f"""
⚠️ Seller contradiction(s): {', '.join(contradictions)}.

Summary so far:
{summary}

You are SARA, an elite real estate acquisitions expert. Your tone should match the seller's: {seller_tone}. They are likely {seller_intent}.

You're in the conversation stage: {stage}. Your job is to move it forward naturally while sticking to the flow.

Investor Math:
{investor_offer_text}

Walkthrough guidance:
{walkthrough}

NEPQ responses to guide you:
{"; ".join(top_pairs) if top_pairs else "No NEPQ matches."}
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

    update_payload = generate_update_payload(data, memory, conversation_memory["history"], summary, verbal_offer, min_offer, max_offer)
    update_seller_memory(phone_number, update_payload)

    return jsonify({
        "content": reply,
        "tone": seller_tone,
        "intent": seller_intent,
        "summary": summary,
        "nepq_examples": top_pairs,
        "reasoning": investor_offer_text,
        "contradictions": contradictions
    })

@app.route("/", methods=["GET"])
def index():
    return "✅ SARA Webhook is running!"

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")






