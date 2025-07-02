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

# Initialize clients
client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

app = Flask(__name__)
conversation_memory = { "history": [] }
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
    price_match = re.search(r"\$?([0-9]{2,4}(?:[,\.]?[0-9]{3})?)", text.replace(",", ""))
    if price_match:
        try:
            return int(float(price_match.group(1)))
        except:
            return None
    return None

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
    summary_prompt = [
        {"role": "system", "content": "Summarize the following conversation from a seller to highlight key points like motivation, condition, timeline, and pricing. Be concise and natural."},
        {"role": "user", "content": "\n".join(user_messages)}
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=summary_prompt,
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

def calculate_investor_price(arv, repair_cost, target_roi):
    realtor_fees = arv * 0.06
    holding_costs = 0.01 * (arv - repair_cost) * 3
    investor_profit = target_roi * (arv - repair_cost)
    max_price = arv - (realtor_fees + holding_costs + repair_cost + investor_profit)
    return round(max_price, 2)
def generate_update_payload(data, seller_data, conversation_history, call_summary, offer_data):
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
        "summary": call_summary
    })

    offer_history = seller_data.get("offer_history") or []
    if offer_data["verbal_offer_amount"]:
        offer_history.append({
            "amount": round(offer_data["verbal_offer_amount"], 2),
            "timestamp": datetime.utcnow().isoformat()
        })

    # Safely preserve existing values unless new ones are explicitly provided
    def safe_update(field, default=None):
        return data.get(field) if data.get(field) is not None else seller_data.get(field, default)

    return {
        "conversation_log": conversation_history,
        "call_summary": call_summary,
        "summary_history": summary_history,
        "asking_price": extract_asking_price(data.get("seller_input", "")) or safe_update("asking_price"),
        "repair_cost": safe_update("repair_cost"),
        "estimated_arv": safe_update("estimated_arv") or safe_update("arv"),
        "min_offer_amount": offer_data["min_offer_amount"],
        "max_offer_amount": offer_data["max_offer_amount"],
        "verbal_offer_amount": offer_data["verbal_offer_amount"],
        "offer_history": offer_history,
        "follow_up_date": safe_update("follow_up_date"),
        "follow_up_reason": safe_update("follow_up_reason"),
        "follow_up_set_by": safe_update("follow_up_set_by"),
        "phone_number": safe_update("phone_number"),
        "property_address": safe_update("property_address"),
        "condition_notes": safe_update("condition_notes"),
        "lead_source": safe_update("lead_source"),
        "bedrooms": safe_update("bedrooms"),
        "bathrooms": safe_update("bathrooms"),
        "square_footage": safe_update("square_footage"),
        "year_built": safe_update("year_built"),
        "conversation_stage": safe_update("conversation_stage")
    }
    update_payload = generate_update_payload(
        data,
        seller_data or {},
        conversation_memory["history"],
        call_summary,
        offer_data
    )

    import pprint
    pp = pprint.PrettyPrinter(indent=2)
    print("🚨 DEBUG: Payload to Supabase:")
    pp.pprint(update_payload)

    update_seller_memory(phone_number, update_payload)

    return jsonify({
        "content": reply,
        "tone": seller_tone,
        "intent": seller_intent,
        "summary": call_summary,
        "nepq_examples": top_pairs,
        "reasoning": investor_offer,
        "contradictions": contradictions
    })

@app.route("/", methods=["GET"])
def index():
    return "✅ SARA Webhook is running!"

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0"









