from flask import Flask, request, jsonify
import os
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
from pinecone import Pinecone, ServerlessSpec
from pinecone import Index
from openai import OpenAI
from uuid import uuid4

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
pinecone_index_name = os.getenv("PINECONE_INDEX")

client = OpenAI(api_key=openai_api_key)

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

# Flask app
app = Flask(__name__)

# In-memory memory
conversation_memory = {
    "history": []
}
MEMORY_LIMIT = 5

# Tone detection keywords
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

# Intent detection (simple keyword-based logic)
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

# Token limiter
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
def health_check():
    return "✅ SARA Webhook is running!"

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    seller_input = data.get("seller_input", "")
    if not seller_input:
        return jsonify({"error": "Missing seller_input"}), 400

    # Detect tone and intent
    seller_tone = detect_tone(seller_input)
    seller_intent = detect_seller_intent(seller_input)

    # Append to memory
    conversation_memory["history"].append({"role": "user", "content": seller_input})
    if len(conversation_memory["history"]) > MEMORY_LIMIT * 2:
        conversation_memory["history"] = conversation_memory["history"][-MEMORY_LIMIT * 2:]

    # Pinecone NEPQ retrieval
    try:
        query_response = index.query(
            vector=client.embeddings.create(input=[seller_input], model="text-embedding-3-small").data[0].embedding,
            top_k=3,
            include_metadata=True
        )
        top_pairs = [match['metadata']['pair'] for match in query_response['matches']]
    except Exception as e:
        top_pairs = []

    # Mock reasoning (ARV/repair logic placeholder)
    if seller_intent == "price_sensitive":
        reasoning = "Just so you know, our offer will factor in repairs and what the home could resell for after updates. It’s not just a random number—we calculate ROI and resale risk too."
    elif seller_intent == "distressed":
        reasoning = "We’ve worked with sellers in tough spots before—foreclosure, liens, even tax defaults—and can move quickly with minimal hassle."
    else:
        reasoning = "Our process is tailored to your situation and goals. Let’s talk it through so you feel confident about what happens next."

    # Construct system prompt
    system_prompt = f"""
You are SARA, an emotionally intelligent and strategic real estate acquisitions expert.
Tone of the seller: {seller_tone}
Intent: {seller_intent}
Your reply should include emotional awareness, sales psychology, and the following NEPQ-style examples:
{"; ".join(top_pairs) if top_pairs else "No retrieved pairs"}
Also, speak like a human—not a chatbot—and gently include this reasoning if applicable:
{reasoning}
"""

    # Final prompt stack
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_memory["history"])

    # Trim based on token count
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

    # Store assistant reply
    conversation_memory["history"].append({"role": "assistant", "content": reply})

    return jsonify({
        "content": reply,
        "tone": seller_tone,
        "intent": seller_intent,
        "nepq_examples": top_pairs,
        "reasoning": reasoning
    })

if __name__ == "__main__":
    app.run(debug=False, port=8080, host="0.0.0.0")
