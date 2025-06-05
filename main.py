
from flask import Flask, request, jsonify
import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
import tiktoken

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

def determine_rehab_level(description):
    description = description.lower()
    if any(keyword in description for keyword in ["foundation", "roof", "gut", "full remodel", "mold", "asbestos", "plumbing", "electrical", "structural"]):
        return "heavy", 55
    elif any(keyword in description for keyword in ["kitchen", "bathroom", "cabinets", "flooring", "windows", "hvac"]):
        return "medium", 35
    elif any(keyword in description for keyword in ["paint", "carpet", "trashout", "fixtures", "light", "clean", "landscaping"]):
        return "light", 20
    else:
        return "full_gut", 70

def estimate_repair_cost(sqft, condition_description):
    level, rate = determine_rehab_level(condition_description)
    raw_cost = sqft * rate
    total_cost = round(raw_cost * 1.10, 2)  # Add 10% buffer
    return total_cost, level

def calculate_investor_price(arv, repair_cost, target_roi):
    realtor_fees = arv * 0.06
    holding_costs = 0.01 * (arv - repair_cost) * 3
    investor_profit = target_roi * (arv - repair_cost)
    return round(arv - (realtor_fees + holding_costs + repair_cost + investor_profit), 2)

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    seller_input = data.get("seller_input", "")
    arv = data.get("arv")
    sqft = data.get("sqft")
    condition_description = data.get("condition_description", "")

    if not seller_input:
        return jsonify({"error": "Missing seller_input"}), 400

    seller_tone = detect_tone(seller_input)
    seller_intent = detect_seller_intent(seller_input)

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

    repair_cost, repair_level = 0, "unknown"
    if sqft and condition_description:
        try:
            sqft = float(sqft)
            repair_cost, repair_level = estimate_repair_cost(sqft, condition_description)
        except:
            repair_cost = 0

    investor_offer = ""
    if arv and repair_cost:
        try:
            arv = float(arv)
            initial_offer = calculate_investor_price(arv, repair_cost, 0.30)
            min_offer = calculate_investor_price(arv, repair_cost, 0.15)
            investor_offer = (
                f"SARA will start the negotiation at ${initial_offer} and can go up to around ${min_offer}. "
                f"She will not tell the seller this max but use seller tone to negotiate toward it. "
                f"The backend investor must also account for repairs, holding costs, taxes, agent fees, and still make a return."
            )
        except:
            investor_offer = ""

    system_prompt = f'''
You are SARA, a sharp and emotionally intelligent real estate acquisitions expert.
Seller Tone: {seller_tone}
Seller Intent: {seller_intent}
Repair Level: {repair_level.upper()}
Repair Estimate: ${repair_cost}

Negotiation Instructions:
{investor_offer}

Embed NEPQ-style examples:
{"; ".join(top_pairs) if top_pairs else "No NEPQ matches found."}

Do NOT mention ROI percentage to seller. You can say investors factor in agent fees, repairs, taxes, holding costs, and profit margin.
Limit to 3 counteroffers. Be calm, human, and strategic.
'''

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

    return jsonify({
        "content": reply,
        "tone": seller_tone,
        "intent": seller_intent,
        "repair_level": repair_level,
        "repair_cost": repair_cost,
        "nepq_examples": top_pairs,
        "reasoning": investor_offer
    })

@app.route("/", methods=["GET"])
def index():
    return "✅ SARA with repair logic is running!"

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")

