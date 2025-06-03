from flask import Flask, request, jsonify
import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone import ServerlessSpec
from openai import OpenAI
from openai.types import CreateEmbeddingResponse

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
pinecone_index = os.getenv("PINECONE_INDEX")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index)

# Initialize Flask app
app = Flask(__name__)

# Short-term in-memory conversation memory
conversation_memory = {
    "history": []
}
MEMORY_LIMIT = 5

@app.route("/")
def index():
    return "✅ SARA Webhook is running!"

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    seller_input = data.get("seller_input", "")

    if not seller_input:
        return jsonify({"error": "Missing seller_input"}), 400

    # Append seller input to memory
    conversation_memory["history"].append({"role": "user", "content": seller_input})

    # Trim to last MEMORY_LIMIT exchanges
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    # Embed seller input to query Pinecone
    try:
        embed_response = client.embeddings.create(
            input=[seller_input],
            model="text-embedding-3-small"
        )
        seller_embedding = embed_response.data[0].embedding
    except Exception as e:
        return jsonify({"error": f"Embedding error: {str(e)}"}), 500

    # Search Pinecone for relevant NEPQ pair
    try:
        search_results = index.query(
            vector=seller_embedding,
            top_k=1,
            include_metadata=True
        )
        top_result = search_results["matches"][0]["metadata"]["pair"]
    except Exception as e:
        top_result = None  # Fail gracefully

    # Construct message stack
    messages = [{"role": "system", "content": "You are SARA, a friendly and strategic real estate acquisitions expert. You speak like a human rep, using emotional intelligence, curiosity, and active listening."}]

    # If NEPQ pair was found, insert before conversation history
    if top_result:
        messages.append({
            "role": "system",
            "content": f"When relevant, guide the conversation using this strategy: {top_result}"
        })

    # Add conversation memory
    messages.extend(conversation_memory["history"])

    # Call OpenAI
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.6
        )
        reply = response.choices[0].message.content
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Store reply in memory
    conversation_memory["history"].append({"role": "assistant", "content": reply})

    return jsonify({"content": reply})

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")
