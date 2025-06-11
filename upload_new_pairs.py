import json
import os
import openai
from pinecone import Pinecone

# Load environment variables
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX = os.environ["PINECONE_INDEX"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Initialize Pinecone and OpenAI
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
openai.api_key = OPENAI_API_KEY

# Load JSON files
json_paths = [
    "./NEPQ_Pairs_97_to_111_Part1.json",
    "./NEPQ_Pairs_112_to_126_Part2.json"
]

# Embedding helper
def embed_text(text):
    response = openai.Embedding.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response["data"][0]["embedding"]

# Upload to Pinecone
for path in json_paths:
    with open(path, "r") as f:
        data = json.load(f)

    upserts = []
    for item in data:
        combined_text = f"{item['seller_statement']} {item['response']} {item['follow_up_prompt']}"
        embedding = embed_text(combined_text)

        metadata = {
            "seller_statement": item["seller_statement"],
            "response": item["response"],
            "follow_up_prompt": item["follow_up_prompt"],
            "tone": item["tone"],
            "category": item["category"]
        }

        upserts.append((item["id"], embedding, metadata))

    # Push batch to Pinecone
    index.upsert(vectors=upserts)

print("✅ New NEPQ pairs successfully uploaded.")
