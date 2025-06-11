# memory_summarizer.py

import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import os
from seller_memory_service import update_seller_memory

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

RECENT_TURNS_TO_KEEP = 4  # Keep last N turns
MAX_SUMMARY_TOKENS = 300  # Optional cap for summary length

def summarize_and_trim_memory(phone_number: str, memory: list):
    # ✅ Fix for the TypeError: ensure memory is a list of dicts
    if isinstance(memory, str):
        try:
            memory = json.loads(memory)
        except Exception as e:
            print(f"[Memory Summarizer Error] Could not parse memory JSON: {e}")
            memory = []

    # Extract only user messages for summarization
    user_messages = [m["content"] for m in memory if m.get("role") == "user"][:-RECENT_TURNS_TO_KEEP]
    recent_turns = memory[-RECENT_TURNS_TO_KEEP:]

    if not user_messages:
        return memory, None

    summary_prompt = [
        {"role": "system", "content": "Summarize the seller’s inputs in a natural way. Focus on their motivation, condition of the home, timing, and price expectations."},
        {"role": "user", "content": "\n".join(user_messages)}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=summary_prompt,
            temperature=0.5,
            max_tokens=MAX_SUMMARY_TOKENS
        )
        summary = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Memory Summarizer Error] GPT summarization failed: {e}")
        summary = None

    # Rebuild memory with summary + recent turns
    new_memory = []
    if summary:
        new_memory.append({"role": "system", "content": f"Memory Summary: {summary}"})
        update_seller_memory(phone_number, {"call_summary": summary})

    new_memory.extend(recent_turns)
    return new_memory, summary



