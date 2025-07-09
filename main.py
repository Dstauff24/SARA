from flask import Flask, request, jsonify
import os, re, json
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
import tiktoken
from datetime import datetime, timedelta
from seller_memory_service import get_seller_memory, update_seller_memory

# Load environment
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

app = Flask(__name__)
conversation_memory = {"history": []}
MEMORY_LIMIT = 5

tone_map = {
    "angry": ["this is ridiculous", "pissed", "you people", "frustrated"],
    "skeptical": ["not sure", "sounds like a scam", "don’t believe"],
    "curious": ["wondering", "what would you offer", "explain"],
    "hesitant": ["i don’t know", "maybe", "thinking"],
    "urgent": ["need to sell fast", "asap", "foreclosure"],
    "emotional": ["passed", "divorce", "lost job"],
    "motivated": ["ready to go", "want to sell", "just want out"],
    "doubtful": ["too low", "never take that"],
    "withdrawn": ["leave me alone", "not interested"],
    "friendly": ["hey", "thanks for calling"],
    "direct": ["how much", "what’s the offer"],
    "neutral": []
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
    if any(kw in text for kw in ["foreclosure", "behind", "bank"]):
        return "distressed"
    if any(kw in text for kw in ["maybe", "thinking", "not sure"]):
        return "on_fence"
    if any(kw in text for kw in ["stop calling", "not interested"]):
        return "cold"
    if any(kw in text for kw in ["vacant", "tenant", "rented"]):
        return "landlord"
    return "general_inquiry"

def extract_motivation_score(text):
    lowered = text.lower()
    if any(kw in lowered for kw in ["asap", "right away", "urgent", "need to sell fast"]):
        return 9
    elif any(kw in lowered for kw in ["just looking", "no rush", "maybe later"]):
        return 2
    return 5

def extract_personality_tag(text):
    lowered = text.lower()
    if "i don’t trust" in lowered or "i don't trust" in lowered:
        return "skeptical"
    elif "fair deal" in lowered:
        return "direct"
    elif "been through" in lowered or "emotional" in lowered:
        return "emotional"
    return "neutral"

def extract_timeline(text):
    lowered = text.lower()
    if "asap" in lowered or "right away" in lowered:
        return "ASAP"
    if "30 days" in lowered:
        return "30 days"
    if "90 days" in lowered or "3 months" in lowered:
        return "90+ days"
    return None

def detect_contradictions(text, history):
    flags = []
    if "asap" in text.lower() and "waiting it out" in history.lower():
        flags.append("ASAP+Waiting")
    if "no repairs" in text.lower() and "kitchen needs work" in history.lower():
        flags.append("NoRepairs+Kitchen")
    return flags

def determine_lead_status(score, timeline):
    if score >= 8 and timeline == "ASAP":
        return "hot"
    if score <= 3:
        return "cold"
    return "warm"

def calculate_follow_up_date(lead_status):
    today = datetime.utcnow()
    delays = {
        "hot": 2,
        "warm": 7,
        "cold": 30,
        "needs_research": 14,
        "dead": None
    }
    days = delays.get(lead_status)
    return (today + timedelta(days=days)).isoformat() if days else None

def follow_up_suggestion(lead_status, timeline):
    if lead_status == "cold" or timeline in ["30 days", "90+ days"]:
        return "Totally understand. Would it make sense for me to check back in a few weeks?"
    if lead_status == "needs_research":
        return "Sounds like there's still some figuring out to do — I can follow up in a couple weeks if that helps?"
    return ""

def detect_strategy_flags(text, arv, repair_cost, asking_price):
    flags = []
    text = text.lower()
    try:
        arv = float(arv)
        repair_cost = float(repair_cost)
        novation_limit = (arv - repair_cost) * 0.9 - 30000
        if asking_price and asking_price <= novation_limit:
            flags.append("novation_candidate")
    except:
        pass
    if any(kw in text for kw in ["mortgage", "payment", "monthly", "rent"]):
        flags.append("creative_finance")
    if any(kw in text for kw in ["listing", "full price", "market value"]):
        flags.append("needs_agent_follow_up")
    return flags

def calculate_lead_score(motivation, timeline, tone, intent):
    score = 0
    score += min(max(motivation, 0), 10) * 4
    if timeline == "ASAP":
        score += 25
    elif timeline == "30 days":
        score += 15
    elif timeline == "90+ days":
        score += 5
    high = ["motivated", "friendly"]
    mid = ["direct", "curious", "hesitant"]
    low = ["cold", "withdrawn", "angry", "skeptical"]
    if tone in high:
        score += 20
    elif tone in mid:
        score += 10
    elif tone in low:
        score += 0
    if intent == "distressed":
        score += 15
    elif intent in ["landlord", "on_fence"]:
        score += 10
    elif intent == "cold":
        score += 0
    else:
        score += 5
    return min(score, 100)
def extract_asking_price(text):
    matches = re.findall(r'\$?\s?(\d{5,7})', text.replace(',', ''))
    try:
        return int(matches[0]) if matches else None
    except:
        return None

def extract_condition_notes(text):
    notes = []
    if "roof" in text.lower():
        notes.append("roof mentioned")
    if "hvac" in text.lower():
        notes.append("hvac mentioned")
    if "kitchen" in text.lower():
        notes.append("kitchen updates")
    return ", ".join(notes) if notes else None

def extract_offer_from_reply(text, asking_price=None):
    if not text:
        return None
    cleaned_text = text.replace(',', '')
    matches = re.findall(r'\$?\s?(\d{4,7})', cleaned_text)
    try:
        amounts = [int(m) for m in matches]
        if asking_price:
            amounts = [a for a in amounts if abs(a - asking_price) > 100]
        return amounts[0] if amounts else None
    except:
        return None

def summarize_messages(messages):
    prompt = [
        {"role": "system", "content": "Summarize key points like motivation, condition, timeline, pricing. Be natural."},
        {"role": "user", "content": "\n".join(messages)}
    ]
    return client.chat.completions.create(
        model="gpt-4", messages=prompt, temperature=0.5
    ).choices[0].message.content

def calculate_offer(arv, repair_cost, roi):
    try:
        arv, repair_cost = float(arv), float(repair_cost)
        fees = arv * 0.06
        hold = 0.01 * (arv - repair_cost) * 3
        profit = roi * (arv - repair_cost)
        return round(arv - (fees + hold + repair_cost + profit), 2)
    except:
        return None

def num_tokens_from_messages(messages, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    total = 3
    for m in messages:
        total += 3
        for k, v in m.items():
            total += len(encoding.encode(v))
    return total

def generate_update_payload(data, memory, history, summary, verbal, min_offer, max_offer, extra_fields):
    summary_history = memory.get("summary_history", [])
    if isinstance(summary_history, str):
        try:
            summary_history = json.loads(summary_history)
        except:
            summary_history = []
    summary_history.append({
        "timestamp": datetime.utcnow().isoformat(),
        "summary": summary
    })

    offer_history = memory.get("offer_history", [])
    if verbal:
        offer_history.append({
            "amount": verbal,
            "timestamp": datetime.utcnow().isoformat()
        })

    existing = memory or {}
    asking_price = data.get("asking_price") or extract_asking_price(data.get("seller_input", ""))
    condition_notes = data.get("condition_notes") or extract_condition_notes(data.get("seller_input", ""))

    return {
        "phone_number": data.get("phone_number"),
        "conversation_log": history,
        "call_summary": summary,
        "summary_history": summary_history,
        "offer_history": offer_history,
        "verbal_offer_amount": verbal or existing.get("verbal_offer_amount"),
        "min_offer_amount": min_offer or existing.get("min_offer_amount"),
        "max_offer_amount": max_offer or existing.get("max_offer_amount"),
        "asking_price": asking_price or existing.get("asking_price"),
        "repair_cost": data.get("repair_cost") or existing.get("repair_cost"),
        "estimated_arv": data.get("estimated_arv") or existing.get("estimated_arv"),
        "condition_notes": condition_notes or existing.get("condition_notes"),
        "follow_up_date": extra_fields.get("follow_up_date") or existing.get("follow_up_date"),
        "follow_up_reason": extra_fields.get("follow_up_reason") or existing.get("follow_up_reason"),
        "follow_up_set_by": data.get("follow_up_set_by") or existing.get("follow_up_set_by"),
        "property_address": data.get("property_address") or existing.get("property_address"),
        "lead_source": data.get("lead_source") or existing.get("lead_source"),
        "square_footage": data.get("square_footage") or existing.get("square_footage"),
        "bedrooms": data.get("bedrooms") or existing.get("bedrooms"),
        "bathrooms": data.get("bathrooms") or existing.get("bathrooms"),
        "year_built": data.get("year_built") or existing.get("year_built"),
        "conversation_stage": existing.get("conversation_stage", "Introduction + Rapport"),
        "motivation_score": extra_fields.get("motivation_score"),
        "timeline_to_sell": extra_fields.get("timeline_to_sell"),
        "personality_tag": extra_fields.get("personality_tag"),
        "lead_status": extra_fields.get("lead_status"),
        "lead_score": extra_fields.get("lead_score"),
        "strategy_flags": extra_fields.get("strategy_flags"),
        "contradictions": extra_fields.get("contradictions")
    }

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    seller_input = data.get("seller_input", "")
    phone = data.get("phone_number")
    if not seller_input or not phone:
        return jsonify({"error": "Missing seller_input or phone_number"}), 400

    memory = get_seller_memory(phone)
    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    tone = detect_tone(seller_input)
    intent = detect_seller_intent(seller_input)
    motivation_score = extract_motivation_score(seller_input)
    timeline = extract_timeline(seller_input)
    personality = extract_personality_tag(seller_input)
    contradictions = detect_contradictions(seller_input, str(memory.get("conversation_log", "")))
    arv = data.get("estimated_arv") or memory.get("estimated_arv")
    repair = data.get("repair_cost") or memory.get("repair_cost")
    asking_price = extract_asking_price(seller_input) or memory.get("asking_price")
    strategy_flags = detect_strategy_flags(seller_input, arv, repair, asking_price)
    lead_status = determine_lead_status(motivation_score, timeline)
    follow_up_date = calculate_follow_up_date(lead_status)
    lead_score = calculate_lead_score(motivation_score, timeline, tone, intent)

    try:
        embed = client.embeddings.create(input=[seller_input], model="text-embedding-3-small").data[0].embedding
        results = index.query(vector=embed, top_k=3, include_metadata=True)
        top_examples = [r.metadata["response"] for r in results.matches]
    except:
        top_examples = []

    min_offer = calculate_offer(arv, repair, 0.30) if arv and repair else None
    max_offer = calculate_offer(arv, repair, 0.15) if arv and repair else None
    summary = summarize_messages([m["content"] for m in conversation_memory["history"] if m["role"] == "user"])

    system_prompt = f"""
You are SARA, a calm, smart acquisitions expert.
Tone: {tone}, Intent: {intent}, Personality: {personality}
Use this negotiation range: Min ${min_offer} — Max ${max_offer}
NEPQ: {"; ".join(top_examples) if top_examples else "None found"}
Avoid technical language. Focus on simplicity, certainty, and seller value.
"""

    messages = [{"role": "system", "content": system_prompt}] + conversation_memory["history"]
    while num_tokens_from_messages(messages) > 3000:
        messages.pop(1)

    reply = client.chat.completions.create(model="gpt-4", messages=messages, temperature=0.7).choices[0].message.content
    conversation_memory["history"].append({"role": "assistant", "content": reply})
    verbal_offer = extract_offer_from_reply(reply, asking_price)

    extras = {
        "motivation_score": motivation_score,
        "timeline_to_sell": timeline,
        "personality_tag": personality,
        "lead_status": lead_status,
        "lead_score": lead_score,
        "strategy_flags": strategy_flags,
        "contradictions": contradictions,
        "follow_up_date": follow_up_date,
        "follow_up_reason": follow_up_suggestion(lead_status, timeline)
    }

    payload = generate_update_payload(data, memory or {}, conversation_memory["history"], summary, verbal_offer, min_offer, max_offer, extras)
    update_seller_memory(phone, payload)

    return jsonify({
        "content": reply,
        "tone": tone,
        "intent": intent,
        "lead_score": lead_score,
        "summary": summary,
        "nepq_examples": top_examples,
        "min_offer": min_offer,
        "max_offer": max_offer,
        "verbal_offer": verbal_offer,
        "strategy_flags": strategy_flags,
        "lead_status": lead_status
    })

@app.route("/", methods=["GET"])
def index():
    return "✅ SARA Webhook is live!"

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")









