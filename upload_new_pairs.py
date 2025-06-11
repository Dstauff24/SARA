import json
import os
import pinecone
import openai

# Load environment variables
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = os.environ["PINECONE_ENV"]
INDEX_NAME = os.environ["PINECONE_INDEX"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Initialize clients
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(INDEX_NAME)
openai.api_key = OPENAI_API_KEY

# Load JSON files
json_paths = [
    "./NEPQ_Pairs_97_to_111_Part1.json",
    "./NEPQ_Pairs_112_to_126_Part2.json"
]

# Prepare data
def embed_text(text):
    response = openai.Embedding.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response["data"][0]["embedding"]

for path in json_paths:
    with open(path, "r") as f:
        data = json.load(f)

    for item in data:
        combined_text = f"{item['seller_statement']} {item['response']} {item['follow_up_prompt']}"
        embedding = embed_text(combined_text)

        index.upsert([
            (
                item["id"],
                embedding,
                {
                    "seller_statement": item["seller_statement"],
                    "response": item["response"],
                    "follow_up_prompt": item["follow_up_prompt"],
                    "tone": item["tone"],
                    "category": item["category"]
                }
            )
        ])

print("✅ New NEPQ training pairs uploaded to Pinecone.")
