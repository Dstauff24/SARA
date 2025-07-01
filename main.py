from flask import Flask, request, jsonify
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
import tiktoken
from datetime import datetime
from seller_memory_service import get_seller_memory, update_seller_memory

# Load env vars
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX")

# Clients
client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

app = Flask(__name__)
conversation_memory = {"history": []}
MEMORY_LIMIT = 5

# Tone mapping
tone_map = {
    "angry": ["ridiculous", "pissed", "you people", "frustrated"],
    "skeptical": ["not sure", "scam", "don’t believe"],
    "curious": ["wondering", "offer", "explain"],
    "hesitant": ["don’t know", "maybe", "thinking"],
    "urgent": ["sell fast", "asap", "foreclosure", "eviction"],
    "emotional": ["passed", "divorce", "lost job"],
    "motivated": ["ready", "want to sell", "just want out"],
    "doubtful": ["too low", "never take"],
    "withdrawn": ["leave alone", "stop calling"],
    "neutral": [],
    "friendly": ["hey", "thanks", "no worries"],
    "direct": ["how much", "offer", "cut to it"]
}

def detect_tone(text):
    lowered = text.lower()
    for tone, keywords in tone_map.items():
        if any(kw in lowered for kw in keywords):
            return tone
    return "neutral"

def detect_intent(text):
    lowered = text.lower()
    if any(kw in lowered for kw in ["how much", "offer", "price"]):
        return "price_sensitive"
    if any(kw in lowered for kw in ["foreclosure", "behind", "bank"]):
        return "distressed"
    if any(kw in lowered for kw in ["maybe", "thinking", "not sure"]):
        return "on_fence"
    if any(kw in lowered for kw in ["not interested", "leave me"]):
        return "cold"
    if any(kw in lowered for kw in ["tenant", "vacant", "investment"]):
        return "landlord"
    return "general_inquiry"

def detect_contradiction(new_input, seller_data):
    contradictions = []
    if seller_data:
        if "asking_price" in seller_data and str(seller_data["asking_price"]) not in new_input:
            if any(kw in new_input for kw in ["price", "$", "want"]):
                contradictions.append("asking_price")
    return contradictions

def extract_price(text):
    import re
    match = re.search(r"\$?(\d{2,6})([kK]?)", text)
    if match:
        num = int(match.group(1))
        if match.group(2).lower() == "k":
            num *= 1000
        return num
    return None

def generate_summary(user_messages):
    summary_prompt = [
        {"role": "system", "content": "Summarize the seller’s situation, motivation, condition, timeline, and price."},
        {"role": "user", "content": "\n".join(user_messages)}
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=summary_prompt,
        temperature=0.5
    )
    return response.choices[0].message.content

def calculate_offer_range(arv, repair, roi_lo=0.15, roi_hi=0.30):
    fees = arv * 0.06
    holding = 0.01 * (arv - repair) * 3
    min_offer = arv - (fees + holding + repair + roi_lo * (arv - repair))
    max_offer = arv - (fees + holding + repair + roi_hi * (arv - repair))
    return round(min_offer, 2), round(max_offer, 2)

def num_tokens(messages):
    encoding = tiktoken.encoding_for_model("gpt-4")
    return sum(len(encoding.encode(m["content"])) + 3 for m in messages) + 3

def generate_update_payload(data, seller_data, convo, summary, verbal_offer, min_offer, max_offer):
    summary_history = seller_data.get("summary_history", [])
    if isinstance(summary_history, str):
        try: summary_history = json.loads(summary_history)
        except: summary_history = []
    summary_history.append({
        "timestamp": datetime.utcnow().isoformat(),
        "summary": summary
    })

    offer_history = seller_data.get("offer_history", [])
    if verbal_offer:
        offer_history.append({
            "amount": round(verbal_offer, 2),
            "timestamp": datetime.utcnow().isoformat()
        })

    def merge(key, default=None):
        return data.get(key) if key in data and data.get(key) is not None else seller_data.get(key, default)

    asking_price = extract_price(data.get("seller_input", "")) or merge("asking_price")

    return {
        "phone_number": merge("phone_number"),
        "asking_price": asking_price,
        "repair_cost": merge("repair_cost"),
        "estimated_arv": merge("estimated_arv") or data.get("arv"),
        "property_address": merge("property_address"),
        "condition_notes": merge("condition_notes"),
        "lead_source": merge("lead_source"),
        "bedrooms": merge("bedrooms"),
        "bathrooms": merge("bathrooms"),
        "square_footage": merge("square_footage"),
        "year_built": merge("year_built"),
        "follow_up_date": merge("follow_up_date"),
        "follow_up_reason": merge("follow_up_reason"),
        "follow_up_set_by": merge("follow_up_set_by"),
        "conversation_log": convo,
        "call_summary": summary,
        "summary_history": summary_history,
        "verbal_offer_amount": round(verbal_offer, 2) if verbal_offer else None,
        "min_offer_amount": round(min_offer, 2) if min_offer else None,
        "max_offer_amount": round(max_offer, 2) if max_offer else None,
        "offer_history": offer_history
    }

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    seller_input = data.get("seller_input", "")
    phone = data.get("phone_number")
    arv = float(data.get("arv", 0))
    repair = float(data.get("repair_cost", 0))

    if not seller_input or not phone:
        return jsonify({"error": "Missing required fields"}), 400

    seller_data = get_seller_memory(phone) or {}
    tone = detect_tone(seller_input)
    intent = detect_intent(seller_input)
    contradiction = detect_contradiction(seller_input, seller_data)

    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    try:
        embedding = client.embeddings.create(input=[seller_input], model="text-embedding-3-small").data[0].embedding
        matches = index.query(vector=embedding, top_k=3, include_metadata=True)
        examples = [m.metadata["response"] for m in matches.matches]
    except:
        examples = []

    min_offer, max_offer = None, None
    verbal_offer = None
    if arv and repair:
        min_offer, max_offer = calculate_offer_range(arv, repair)
        verbal_offer = min_offer

    summary = generate_summary([m["content"] for m in conversation_memory["history"] if m["role"] == "user"])

    prompt = f"""
You are SARA, a sharp and emotionally intelligent real estate acquisitions expert.

Tone: {tone}
Intent: {intent}
Summary: {summary}
Contradictions: {', '.join(contradiction) if contradiction else 'None'}

Negotiation Guidance:
Start at ${max_offer}, negotiate up to ${min_offer}.

Walkthrough: Once we agree on terms, we’ll verify condition — nothing for you to worry about now.
NEPQ examples: {"; ".join(examples) if examples else "None"}
"""

    messages = [{"role": "system", "content": prompt}] + conversation_memory["history"]
    while num_tokens(messages) > 3000:
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
    update_payload = generate_update_payload(data, seller_data, conversation_memory["history"], summary, verbal_offer, min_offer, max_offer)

    import pprint
    pprint.PrettyPrinter(indent=2).pprint(update_payload)

    update_seller_memory(phone, update_payload)

    return jsonify({
        "content": reply,
        "tone": tone,
        "intent": intent,
        "summary": summary,
        "nepq_examples": examples,
        "reasoning": f"Start at ${max_offer}, negotiate up to ${min_offer}",
        "contradictions": contradiction
    })

@app.route("/", methods=["GET"])
def index():
    return "✅ SARA Webhook is running!"

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")





