import os
import json
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME")

# Initialize OpenAI client
openai = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Target JSON files
json_paths = [
    "./data/NEPQ_Pairs_97_to_111_Part1.json",
    "./data/NEPQ_Pairs_112_to_126_Part2.json"
]

# Embedding function
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# Upload each JSON file
for json_path in json_paths:
    with open(json_path, "r") as f:
        training_pairs = json.load(f)

    vectors = []
    for pair in training_pairs:
        seller_statement = pair["seller_statement"]
        vector = get_embedding(seller_statement)
        vectors.append({
            "id": pair["id"],
            "values": vector,
            "metadata": {
                "tone": pair["tone"],
                "category": pair["category"],
                "response": pair["response"],
                "follow_up_prompt": pair["follow_up_prompt"]
            }
        })

    # Upsert to Pinecone
    index.upsert(vectors=vectors)

print("Upload complete.")
