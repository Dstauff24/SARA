from flask import Flask, request, jsonify
import os
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
from pinecone import Pinecone

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX")

# Initialize clients
client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

# Flask app
app = Flask(__name__)
conversation_memory = {"history": []}
MEMORY_LIMIT = 5

tone_map = {
    "angry": ["ridiculous", "pissed", "frustrated"],
    "skeptical": ["scam", "not sure", "believe that"],
    "curious": ["wondering", "offer", "explain"],
    "hesitant": ["maybe", "not sure", "thinking"],
    "urgent": ["asap", "foreclosure", "eviction", "quick"],
    "emotional": ["divorce", "passed away", "job loss"],
    "motivated": ["want to sell", "get it done", "ready"],
    "doubtful": ["too low", "never accept"],
    "withdrawn": ["stop calling", "not interested"],
    "neutral": [],
    "friendly": ["hey", "thanks for calling", "great"],
    "direct": ["how much", "what’s the number", "let’s talk money"]
}

def detect_tone(text):
    lowered = text.lower()
    for tone, phrases in tone_map.items():
        if any(p in lowered for p in phrases):
            return tone
    return "neutral"

def detect_seller_intent(text):
    text = text.lower()
    if any(kw in text for kw in ["how much", "offer", "price"]):
        return "price_sensitive"
    if any(kw in text for kw in ["foreclosure", "behind", "bank"]):
        return "distressed"
    if any(kw in text for kw in ["maybe", "not sure", "depends"]):
        return "on_fence"
    if any(kw in text for kw in ["stop calling", "not interested"]):
        return "cold"
    if any(kw in text for kw in ["tenant", "vacant", "rented"]):
        return "landlord"
    return "general_inquiry"

def num_tokens_from_messages(messages, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    tokens_per_message = 3
    tokens_per_name = 1
    total_tokens = 0
    for msg in messages:
        total_tokens += tokens_per_message
        for key, val in msg.items():
            total_tokens += len(encoding.encode(val))
            if key == "name":
                total_tokens += tokens_per_name
    total_tokens += 3
    return total_tokens

def calculate_roi_offers(arv, repair_cost, seller_price=None):
    holding_costs = arv * 0.01 * 3
    realtor_fees = arv * 0.06
    net_proceeds = arv - holding_costs - realtor_fees

    def investor_price(roi): return round((net_proceeds / (1 + roi)) - repair_cost, 2)

    anchor = investor_price(0.30)
    ceiling = investor_price(0.15)

    if seller_price:
        seller_price = float(seller_price)
        roi = (net_proceeds - (seller_price + repair_cost)) / (seller_price + repair_cost)
        if roi >= 0.30:
            anchor = round(seller_price * 0.98, 2)
        elif roi >= 0.15:
            second_offer = round(seller_price * 0.98, 2)
        else:
            second_offer = ceiling
    else:
        second_offer = None

    return {
        "anchor_offer": anchor,
        "max_ceiling": ceiling,
        "second_offer": second_offer
    }

@app.route("/", methods=["GET"])
def health_check():
    return "✅ SARA Webhook is running!"

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    seller_input = data.get("seller_input", "")
    arv = float(data.get("arv", 0))
    repair_cost = float(data.get("repair_cost", 0))
    seller_price = float(data.get("seller_price", 0)) if data.get("seller_price") else None

    if not seller_input or arv == 0 or repair_cost == 0:
        return jsonify({"error": "Missing seller_input, arv, or repair_cost"}), 400

    tone = detect_tone(seller_input)
    intent = detect_seller_intent(seller_input)
    conversation_memory["history"].append({"role": "user", "content": seller_input})

    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    try:
        embedding = client.embeddings.create(input=[seller_input], model="text-embedding-3-small").data[0].embedding
        query = index.query(vector=embedding, top_k=3, include_metadata=True)
        examples = [match.metadata['pair'] for match in query['matches']]
    except Exception as e:
        examples = []

    strategy = calculate_roi_offers(arv, repair_cost, seller_price)
    reasoning = f"Our initial offer is based on a 30% ROI for our investor, which would be around ${strategy['anchor_offer']}. We’ll always aim to meet your needs, so if we’re close, we may improve that offer depending on the situation."

    system_prompt = f"""
You are SARA, a persuasive and emotionally aware real estate acquisitions specialist.
Seller tone: {tone}
Intent: {intent}
NEPQ examples to use: {" | ".join(examples) if examples else "No match"}
Incorporate this reasoning into your logic:
{reasoning}
Only present the offer at the 30% ROI level unless the seller names a price that meets ROI threshold, in which case counter at 2% lower. Don’t reveal your full ceiling. Negotiate up to 3 times, never above 15% ROI ceiling.
"""

    messages = [{"role": "system", "content": system_prompt}] + conversation_memory["history"]
    while num_tokens_from_messages(messages) > 3000:
        messages.pop(1)

    try:
        chat_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.6
        )
        reply = chat_response.choices[0].message.content
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    conversation_memory["history"].append({"role": "assistant", "content": reply})
    return jsonify({
        "content": reply,
        "tone": tone,
        "intent": intent,
        "anchor_offer": strategy["anchor_offer"],
        "max_ceiling": strategy["max_ceiling"],
        "examples_used": examples,
        "reasoning": reasoning
    })

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")
