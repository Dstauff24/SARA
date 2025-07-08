from flask import Flask, request, jsonify
import os
import re
import json
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
from datetime import datetime
from seller_memory_service import get_seller_memory, update_seller_memory
import tiktoken

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
STAGES = [
    "Introduction + Rapport", "Setting the Stage", "Seller Motivation",
    "Wholesaler vs Agent", "Wholesale Process", "Review ARV",
    "Home Tour", "Review Repairs", "Your Offer", "Let's Get Started"
]

def extract_asking_price(text):
    matches = re.findall(r"\$?\s?([1-9][0-9]{4,6})", text.replace(",", ""))
    for match in matches:
        price = int(match)
        if 30000 <= price <= 2000000:
            return price
    return None

def extract_condition_notes(text):
    keywords = ["roof", "hvac", "foundation", "kitchen", "floor", "paint", "carpet", "plumbing", "windows"]
    matches = [word for word in keywords if word in text.lower()]
    return ", ".join(set(matches)) if matches else None

def calculate_price(arv, repairs, roi):
    try:
        arv = float(arv)
        repairs = float(repairs)
        realtor = arv * 0.06
        hold = 0.01 * (arv - repairs) * 3
        profit = roi * (arv - repairs)
        return round(arv - (realtor + hold + repairs + profit), 2)
    except:
        return None

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

def get_next_stage(current):
    try:
        idx = STAGES.index(current)
        return STAGES[min(idx + 1, len(STAGES) - 1)]
    except:
        return STAGES[0]

def summarize_history(messages):
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    prompt = [{"role": "system", "content": "Summarize the key seller points from this conversation (motivation, condition, price, timeline):"},
              {"role": "user", "content": "\n".join(user_msgs)}]
    response = client.chat.completions.create(model="gpt-4", messages=prompt, temperature=0.5)
    return response.choices[0].message.content

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    seller_input = data.get("seller_input", "")
    phone = data.get("phone_number", "")
    seller_data = get_seller_memory(phone) or {}

    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    condition_notes = data.get("condition_notes") or seller_data.get("condition_notes") or extract_condition_notes(seller_input)
    asking_price = data.get("asking_price") or extract_asking_price(seller_input) or seller_data.get("asking_price")
    estimated_arv = data.get("estimated_arv") or seller_data.get("estimated_arv")
    repair_cost = data.get("repair_cost") or seller_data.get("repair_cost")

    min_offer = max_offer = verbal_offer = None
    if estimated_arv and repair_cost:
        min_offer = calculate_price(estimated_arv, repair_cost, 0.30)
        max_offer = calculate_price(estimated_arv, repair_cost, 0.15)
        verbal_offer = min_offer  # First offer usually starts at min

    call_summary = summarize_history(conversation_memory["history"])
    conversation_stage = seller_data.get("conversation_stage", "Introduction + Rapport")
    next_stage = get_next_stage(conversation_stage)

    system_prompt = f"""
You are SARA, a smart, emotionally intelligent real estate acquisitions assistant. You're currently at the conversation stage: {conversation_stage}.
Ask only questions relevant to this stage: {conversation_stage}. Use NEPQ and soft skills. Once that part is complete, guide seller to next stage: {next_stage}.
If seller is highly motivated but resists structured flow, skip forward to offer.
Don't talk about ROI. Frame things in terms of real cost and condition.
Current ARV: {estimated_arv}, Repairs: {repair_cost}, Min offer: {min_offer}, Max offer: {max_offer}
    """.strip()

    messages = [{"role": "system", "content": system_prompt}] + conversation_memory["history"]
    while num_tokens_from_messages(messages) > 3000:
        messages.pop(1)

    response = client.chat.completions.create(model="gpt-4", messages=messages, temperature=0.7)
    reply = response.choices[0].message.content
    conversation_memory["history"].append({"role": "assistant", "content": reply})

    offer_history = seller_data.get("offer_history", [])
    if verbal_offer:
        offer_history.append({
            "amount": verbal_offer,
            "timestamp": datetime.utcnow().isoformat()
        })

    summary_history = seller_data.get("summary_history", [])
    if isinstance(summary_history, str):
        try:
            summary_history = json.loads(summary_history)
        except:
            summary_history = []
    summary_history.append({
        "timestamp": datetime.utcnow().isoformat(),
        "summary": call_summary
    })

    payload = {
        "phone_number": phone,
        "conversation_log": conversation_memory["history"],
        "call_summary": call_summary,
        "summary_history": summary_history,
        "asking_price": asking_price,
        "repair_cost": repair_cost,
        "estimated_arv": estimated_arv,
        "condition_notes": condition_notes,
        "property_address": data.get("property_address") or seller_data.get("property_address"),
        "lead_source": data.get("lead_source") or seller_data.get("lead_source"),
        "square_footage": data.get("square_footage") or seller_data.get("square_footage"),
        "bedrooms": data.get("bedrooms") or seller_data.get("bedrooms"),
        "bathrooms": data.get("bathrooms") or seller_data.get("bathrooms"),
        "year_built": data.get("year_built") or seller_data.get("year_built"),
        "follow_up_date": data.get("follow_up_date") or seller_data.get("follow_up_date"),
        "follow_up_reason": data.get("follow_up_reason") or seller_data.get("follow_up_reason"),
        "follow_up_set_by": data.get("follow_up_set_by") or seller_data.get("follow_up_set_by"),
        "conversation_stage": next_stage,
        "min_offer_amount": min_offer,
        "max_offer_amount": max_offer,
        "verbal_offer_amount": verbal_offer,
        "offer_history": offer_history
    }

    update_seller_memory(phone, payload)

    return jsonify({
        "content": reply,
        "summary": call_summary,
        "verbal_offer": verbal_offer,
        "min_offer": min_offer,
        "max_offer": max_offer,
        "stage": next_stage
    })

@app.route("/", methods=["GET"])
def index():
    return "✅ SARA Webhook running!"

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")












