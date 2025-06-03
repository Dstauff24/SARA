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

# Initialize Flask app
app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

# Short-term memory
conversation_memory = {"history": []}
MEMORY_LIMIT = 5

# Count tokens for message management
def count_tokens(messages):
    enc = tiktoken.encoding_for_model("gpt-4")
    return sum(len(enc.encode(msg["content"])) for msg in messages)

# Detect seller emotional tone
def detect_seller_tone(seller_input):
    prompt = f"""
    Classify the emotional tone of the following seller message into one of the following categories:
    Skeptical, Guarded, Defensive, Curious, Optimistic, Indecisive, Frustrated, Distracted, Confused, Motivated, Withdrawn, Open.

    Seller Message: "{seller_input}"

    Respond only with the category name.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Tone detection error:", e)
        return "Neutral"

# Adjust system prompt based on tone
def get_system_prompt(tone):
    styles = {
        "Skeptical": "Be transparent and build trust without pressure.",
        "Guarded": "Use gentle NEPQ questions and respect their boundaries.",
        "Defensive": "Be empathetic, patient, and disarming.",
        "Curious": "Be engaging and informative with NEPQ.",
        "Optimistic": "Match their energy and support their enthusiasm.",
        "Indecisive": "Ask clarifying questions to uncover their motives.",
        "Frustrated": "Stay calm and help solve their core concern.",
        "Distracted": "Be direct, simple, and keep attention focused.",
        "Confused": "Use analogies and simple NEPQ phrasing to explain.",
        "Motivated": "Show urgency and a strong understanding of their goals.",
        "Withdrawn": "Be patient, light, and curious. Don’t pressure.",
        "Open": "Lean in, ask discovery questions, and go deep with NEPQ."
    }
    return (
        f"You are SARA, a warm and strategic real estate acquisitions expert. "
        f"Use emotional intelligence, NEPQ methodology, and sound reasoning to move the deal forward. "
        f"Seller tone: {tone}. {styles.get(tone, '')}"
    )

# Retrieve NEPQ training phrases from Pinecone
def retrieve_nepq_pairs(query_text):
    try:
        embedding = client.embeddings.create(
            model="text-embedding-ada-002",
            input=[query_text]
        ).data[0].embedding

        result = index.query(vector=embedding, top_k=3, include_metadata=True)
        return [match["metadata"]["text"] for match in result["matches"]]
    except Exception as e:
        print("Pinecone error:", e)
        return []

@app.route("/")
def index():
    return "✅ SARA Webhook is running!"

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    seller_input = data.get("seller_input", "")

    if not seller_input:
        return jsonify({"error": "Missing seller_input"}), 400

    # Add to memory
    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    # Detect tone + get system prompt
    seller_tone = detect_seller_tone(seller_input)
    system_prompt = get_system_prompt(seller_tone)

    # Retrieve NEPQ training pair suggestions
    nepq_suggestions = retrieve_nepq_pairs(seller_input)
    if nepq_suggestions:
        conversation_memory["history"].append({
            "role": "system",
            "content": f"Use these NEPQ ideas in your approach: {', '.join(nepq_suggestions)}"
        })

    # Build conversation
    messages = [{"role": "system", "content": system_prompt}] + conversation_memory["history"]

    # Trim if tokens are too high
    while count_tokens(messages) > 4000:
        conversation_memory["history"] = conversation_memory["history"][2:]
        messages = [{"role": "system", "content": system_prompt}] + conversation_memory["history"]

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
    app.run(debug=False, host="0.0.0.0", port=8080)
