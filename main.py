from flask import Flask, request, jsonify
import os
from openai import OpenAI
from pinecone import Pinecone

app = Flask(__name__)

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

@app.route("/")
def home():
    return "✅ SARA Webhook is running!"

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json()
        user_input = data.get("seller_input")

        if not user_input:
            return jsonify({"error": "Missing seller_input"}), 400

        # Basic response
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are SARA, a helpful real estate acquisitions assistant."},
                {"role": "user", "content": user_input}
            ]
        )

        answer = response.choices[0].message.content
        return jsonify({"content": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
