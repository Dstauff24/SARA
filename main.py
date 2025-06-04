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
pinecone_index = os.getenv("PINECONE_INDEX")

client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index)

app = Flask(__name__)

# Short-Term Memory
conversation_memory = {"history": []}
MEMORY_LIMIT = 5

# Tone mapping
tone_map = {
    "angry": ["this is ridiculous", "pissed", "you people", "frustrated"],
    "skeptical": ["not sure", "sounds like a scam", "don’t believe"],
    "curious": ["i’m just wondering", "what would you offer", "can you explain"],
    "hesitant": ["i don’t know", "maybe", "thinking about it"],
    "urgent": ["need to sell fast", "asap", "foreclosure", "eviction"],
    "emotional": ["my mom passed", "divorce", "lost job", "hard time"],
    "motivated": ["ready to go", "want to sell", "just want out"],
    "doubtful": ["no way", "that’s too low", "never take that"],
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
    return "general_inquiry"

# ROI-based Offer Logic
def calculate_offer_logic(arv, repair_cost, wholesale_fee=10000, min_roi=0.15):
    arv = float(arv)
    repair_cost = float(repair_cost)
    realtor_fees = arv * 0.06

    for investor_price in range(1000000, 0, -1000):
        holding_costs = investor_price * 0.01 * 3
        cost_basis = investor_price + repair_cost
        net_proceeds = arv - holding_costs - realtor_fees
        profit = net_proceeds - cost_basis
        roi = profit / cost_basis if cost_basis else 0

        if roi >= min_roi:
            return {
                "max_buyer_price": investor_price,
                "max_seller_offer": investor_price - wholesale_fee,
                "roi_percent": round(roi * 100, 2)
            }
    return {}

# Token counting logic
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

@app.route("/", methods=["GET"])
def index():
    return "✅ SARA Webhook is running!"

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    seller_input = data.get("seller_input", "")
    arv = data.get("arv")
    repair_cost = data.get("repair_cost")

    if not seller_input:
        return jsonify({"error": "Missing seller_input"}), 400

    tone = detect_tone(seller_input)
    intent = detect_seller_intent(seller_input)

    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    # Pinecone NEPQ Retrieval
    try:
        embed = client.embeddings.create(model="text-embedding-3-small", input=[seller_input])
        vector = embed.data[0].embedding
        result = index.query(vector=vector, top_k=1, include_metadata=True)
        training_pair = result.matches[0].metadata["response"] if result.matches else ""
    except:
        training_pair = ""

    # ROI Reasoning
    reasoning = ""
    if intent == "price_sensitive" and arv and repair_cost:
        logic = calculate_offer_logic(arv, repair_cost)
        if logic:
            reasoning = f"Our cash offer factors in resale value, repairs, and investor ROI. Based on a $ARV of {arv} and ${repair_cost} in repairs, the investor would buy at around ${logic['max_buyer_price']}, which means your offer could land around ${logic['max_seller_offer']}."

    # Build Prompt
    messages = [{"role": "system", "content": f"""
You are SARA, a human-sounding, emotionally intelligent real estate acquisitions expert.
Tone: {tone}
Intent: {intent}
If relevant, use this reasoning: {reasoning}
"""}]
    messages.extend(conversation_memory["history"])
    if training_pair:
        messages.append({"role": "assistant", "content": training_pair})

    while num_tokens_from_messages(messages) > 3000:
        messages.pop(1)

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.6
        )
        reply = response.choices[0].message.content
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    conversation_memory["history"].append({"role": "assistant", "content": reply})

    return jsonify({
        "content": reply,
        "tone": tone,
        "intent": intent,
        "reasoning": reasoning
    })

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")
