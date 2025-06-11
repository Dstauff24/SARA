# memory_summarizer.py

from openai import OpenAI
from datetime import datetime
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

RECENT_TURNS_TO_KEEP = 4

def summarize_and_trim_memory(phone_number, memory):
    """
    Summarize earlier parts of the conversation and trim history for storage.
    Returns the new memory list and summary log for saving to summary_history.
    """

    # Separate earlier messages to summarize
    user_messages = [m["content"] for m in memory if isinstance(m, dict) and m.get("role") == "user"]
    assistant_messages = [m["content"] for m in memory if isinstance(m, dict) and m.get("role") == "assistant"]

    if len(user_messages) < RECENT_TURNS_TO_KEEP or len(assistant_messages) < RECENT_TURNS_TO_KEEP:
        return memory, None  # Not enough content to summarize

    summary_input = "\n".join(user_messages[:-RECENT_TURNS_TO_KEEP] + assistant_messages[:-RECENT_TURNS_TO_KEEP])
    summary_prompt = [
        {
            "role": "system",
            "content": "Summarize the earlier part of this seller conversation, highlighting condition, motivation, pricing, and timeline details. Use natural tone and keep it concise."
        },
        {
            "role": "user",
            "content": summary_input
        }
    ]

    try:
        summary_response = client.chat.completions.create(
            model="gpt-4",
            messages=summary_prompt,
            temperature=0.5
        )
        summary_text = summary_response.choices[0].message.content

        # Keep the most recent memory entries
        recent_memory = memory[-RECENT_TURNS_TO_KEEP * 2:]

        summary_log = {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": summary_text
        }

        return recent_memory, summary_log

    except Exception as e:
        print(f"[Summarizer Error] {e}")
        return memory, None

