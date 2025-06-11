import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Setup keys and clients
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Define files to load
json_files = [
    "data/NEPQ_Pairs_97_to_111_Part1.json",
    "data/NEPQ_Pairs_112_to_126_Part2.json"
]

# Embedding function
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        model=model,
        input=[text]
    )
    return response.data[0].embedding

# Upload logic
for file_path in json_files:
    with open(file_path, "r") as f:
        training_data = json.load(f)

    vectors = []
    for pair in training_data:
        embedding = get_embedding(pair["seller_statement"])
        vectors.append({
            "id": pair["id"],
            "values": embedding,
            "metadata": {
                "tone": pair["tone"],
                "category": pair["category"],
                "response": pair["response"],
                "follow_up_prompt": pair["follow_up_prompt"]
            }
        })

    index.upsert(vectors=vectors)
    print(f"✅ Uploaded {len(vectors)} vectors from {file_path}")

print("🎉 All selected NEPQ pairs uploaded successfully.")
