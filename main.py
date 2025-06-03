from flask import Flask, request, jsonify
import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
pinecone_index = os.getenv("PINECONE_INDEX")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Initialize Flask app
app = Flask(__name__)

# In-memory short-term memory for session
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

    # System prompt
    messages = [{"role": "system", "content": "You are SARA, a friendly and strategic real estate acquisitions expert. You speak like a human rep, using emotional intelligence, curiosity, and active listening."}]
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

    # Append assistant reply to memory
    conversation_memory["history"].append({"role": "assistant", "content": reply})

    return jsonify({"content": reply})

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")
