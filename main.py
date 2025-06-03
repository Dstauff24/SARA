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
pinecone_env = os.getenv("PINECONE_ENV")
pinecone_index_name = os.getenv("PINECONE_INDEX")

# Initialize clients
client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

# Flask app
app = Flask(__name__)

# Short-term memory setup
conversation_memory = {"history": []}
MEMORY_LIMIT = 5

# Count tokens
def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))

# Detect seller intent
def detect_intent(text):
    text = text.lower()
    if "lowball" in text or "too low" in text:
        return "price_sensitive"
    elif "makes sense" in text or "depends" in text:
        return "open_to_offer"
    elif "not selling" in text or "not interested" in text:
        return "not_motivated"
    elif "cash" in text or "quick close" in text:
        return "motivated_cash"
    elif "creative" in text or "terms" in text or "seller finance" in text:
        return "creative_finance"
    return "general"

# Build system prompt based on seller tone
def build_system_prompt(intent):
    base = "You are SARA, a friendly and strategic real estate acquisitions expert. You speak like a human rep, using emotional intelligence, curiosity, and active listening."
    if intent == "price_sensitive":
        base += " The seller is price sensitive. Ask their ideal number and educate softly."
    elif intent == "open_to_offer":
        base += " The seller is cautiously curious. Explore their motivation and build trust."
    elif intent == "not_motivated":
        base += " The seller may be disinterested. Use curiosity to see if anything changed."
    elif intent == "motivated_cash":
        base += " The seller is motivated and open to cash. Emphasize speed and certainty."
    elif intent == "creative_finance":
        base += " The seller may be open to creative terms. Test gently with seller finance language."
    return base

@app.route("/")
def index():
    return "✅ SARA Webhook is running!"

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    seller_input = data.get("seller_input", "")
    if not seller_input:
        return jsonify({"error": "Missing seller_input"}), 400

    # Store input
    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    # Analyze tone and build dynamic system prompt
    intent = detect_intent(seller_input)
    system_prompt = build_system_prompt(intent)

    # Embed seller input to retrieve NEPQ strategy
    embedding = client.embeddings.create(
        input=seller_input,
        model="text-embedding-3-small"
    )
    vector = embedding.data[0].embedding

    pinecone_results = index.query(
        vector=vector,
        top_k=3,
        include_metadata=True
    )

    # Add retrieved NEPQ pairs
    nepq_context = []
    for match in pinecone_results.get("matches", []):
        if "text" in match["metadata"]:
            nepq_context.append({"role": "system", "content": f"(NEPQ Strategy) {match['metadata']['text']}"})

    # Construct prompt
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(nepq_context)
    messages.extend(conversation_memory["history"])

    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.6
        )
        reply = completion.choices[0].message.content
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Store assistant response
    conversation_memory["history"].append({"role": "assistant", "content": reply})
    return jsonify({"content": reply})

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")
