# memory_summarizer.py

import os
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

RECENT_TURNS_TO_KEEP = 6

def summarize_and_trim_memory(phone_number, memory):
    """
    Summarizes older messages in memory and returns trimmed memory + summary blob for Supabase.
    """
    if len(memory) <= RECENT_TURNS_TO_KEEP:
        return memory, None

    try:
        user_messages = [m["content"] for m in memory if m["role"] == "user"][:-RECENT_TURNS_TO_KEEP]
        assistant_messages = [m["content"] for m in memory if m["role"] == "assistant"][:-RECENT_TURNS_TO_KEEP]

        combined = ""
        for u, a in zip(user_messages, assistant_messages):
            combined += f"Seller said: {u}\nAssistant replied: {a}\n"

        messages = [
            {"role": "system", "content": "Summarize this seller conversation, focusing on pricing, condition, timeline, and motivation."},
            {"role": "user", "content": combined}
        ]

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3
        )

        summary = response.choices[0].message.content

        # Create memory summary blob
        summary_entry = {
            "summary": summary,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Return the last RECENT_TURNS_TO_KEEP turns + summary object
        trimmed_memory = memory[-RECENT_TURNS_TO_KEEP:]
        return trimmed_memory, summary_entry

    except Exception as e:
        print(f"[Summarizer Error] {e}")
        return memory, None


