import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index = os.getenv("PINECONE_INDEX")

# Initialize OpenAI and Pinecone
client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index)

# Directory with NEPQ data
data_dir = "./data"

# Upload data in batches
for file_name in os.listdir(data_dir):
    if file_name.endswith(".json"):
        print(f"📄 Uploading {file_name}")
        with open(os.path.join(data_dir, file_name), "r") as f:
            data = json.load(f)

        vectors = []
        for item in data:
            try:
                embed = client.embeddings.create(
                    input=[item["seller_statement"]],
                    model="text-embedding-3-small"
                )
                vector = embed.data[0].embedding
                vectors.append({
                    "id": item["id"],
                    "values": vector,
                    "metadata": {
                        "seller_statement": item["seller_statement"],
                        "response": item["response"],
                        "follow_up_prompt": item["follow_up_prompt"],
                        "tone": item["tone"],
                        "category": item["category"]
                    }
                })
            except Exception as e:
                print(f"❌ Error embedding {item['id']}: {e}")

        index.upsert(vectors=vectors)
        print(f"✅ Uploaded {len(vectors)} vectors from {file_name}")
