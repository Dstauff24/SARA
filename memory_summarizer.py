# memory_summarizer.py

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# How many most recent GPT-user turns to keep
RECENT_TURNS_TO_KEEP = 4

def summarize_and_trim_memory(phone_number, memory):
    """
    Summarizes all but the last N turns of the conversation and trims memory.
    Returns a trimmed memory list and the new summary string.
    """
    if not memory:
        return memory, None

    # Isolate only the seller (user) messages
    user_messages = [m["content"] for m in memory if m["role"] == "user"]
    assistant_messages = [m["content"] for m in memory if m["role"] == "assistant"]

    # Exclude the last N turns from summary
    if len(user_messages) < RECENT_TURNS_TO_KEEP:
        return memory, None

    summary_input = "\n".join(user_messages[:-RECENT_TURNS_TO_KEEP])
    assistant_context = "\n".join(assistant_messages[:-RECENT_TURNS_TO_KEEP])

    prompt = [
        {"role": "system", "content": "You are summarizing a real estate seller conversation. Highlight key points like motivation, condition, timeline, and pricing."},
        {"role": "user", "content": summary_input + "\n\n" + assistant_context}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=prompt,
            temperature=0.4
        )
        summary = response.choices[0].message.content.strip()
    except Exception as e:
        summary = None

    # Keep only the last N messages
    trimmed_history = memory[-RECENT_TURNS_TO_KEEP * 2:]  # 2x for user + assistant pairs
    return trimmed_history, summary



