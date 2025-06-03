from flask import Flask, request, jsonify
import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
from typing import List
import tiktoken

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
pinecone_index_name = os.getenv("PINECONE_INDEX")

# Initialize services
client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

app = Flask(__name__)

# Short-term memory store
conversation_memory = {"history": []}
MEMORY_LIMIT = 5

# --- Utility Functions ---

def count_tokens(messages) -> int:
    enc = tiktoken.encoding_for_model("gpt-4")
    token_count = 0
    for msg in messages:
        token_count += 4  # base per-message overhead
        token_count += len(enc.encode(msg["content"]))
    return token_count

def fetch_nepq_responses(seller_input: str, top_k: int = 3) -> List[str]:
    try:
        embedded = client.embeddings.create(
            model="text-embedding-3-small",
            input=seller_input
        ).data[0].embedding

        pinecone_results = index.query(vector=embedded, top_k=top_k, include_metadata=True)
        responses = [match['metadata']['response'] for match in pinecone_results['matches']]
        return responses
    except Exception as e:
        print(f"Error fetching NEPQ responses: {e}")
        return []

# --- Seller Intent Tagging ---

def detect_seller_intent(user_input: str) -> str:
    lowered = user_input.lower()
    if any(x in lowered for x in ["need to", "have to", "must sell", "urgent"]):
        return "motivated"
    elif any(x in lowered for x in ["just curious", "not sure", "maybe", "thinking"]):
        return "not_ready"
    elif any(x in lowered for x in ["price", "money", "offer", "worth"]):
        return "price_sensitive"
    elif any(x in lowered for x in ["agent", "realtor", "listed", "mls"]):
        return "listed_or_considering"
    else:
        return "unknown"

# --- Routes ---

@app.route("/")
def index_route():
    return "✅ SARA Webhook is running!"

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    seller_input = data.get("seller_input", "")

    if not seller_input:
        return jsonify({"error": "Missing seller_input"}), 400

    # Track memory
    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    # Seller intent
    intent_tag = detect_seller_intent(seller_input)

    # System prompt
    system_prompt = f"""You are SARA, a friendly and strategic real estate acquisitions expert.
Use emotional intelligence, NEPQ questions, and conversational tone.
Tag for seller intent: [{intent_tag}]. Respond thoughtfully and with empathy.
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_memory["history"])

    # Pull relevant NEPQ training pair examples
    nepq_responses = fetch_nepq_responses(seller_input)
    for resp in nepq_responses:
        messages.append({"role": "assistant", "content": resp})

    # Ensure token limit safety
    while count_tokens(messages) > 7000:
        messages.pop(1)  # remove oldest user/assistant pair (after system)

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
    return jsonify({"content": reply})

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")
