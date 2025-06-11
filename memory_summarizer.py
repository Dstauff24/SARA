# memory_summarizer.py

from datetime import datetime
from openai import OpenAI
import tiktoken
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MEMORY_SUMMARY_THRESHOLD = 6  # Number of messages before summarizing
RECENT_TURNS_TO_KEEP = 4      # Number of turns to retain after summarizing

def summarize_and_trim_memory(memory: list, summary_history: list):
    """
    Summarizes older messages and returns a cleaned conversation + updated summary history.
    """
    if len(memory) < MEMORY_SUMMARY_THRESHOLD:
        return memory, summary_history

    # Extract old user messages only
    user_messages = [m["content"] for m in memory if m["role"] == "user"][:-RECENT_TURNS_TO_KEEP]

    summary_prompt = [
        {"role": "system", "content": "Summarize this conversation for a real estate lead: focus on price, condition, motivation, and timeline."},
        {"role": "user", "content": "\n".join(user_messages)}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=summary_prompt,
            temperature=0.3
        )
        summary_text = response.choices[0].message.content
    except Exception as e:
        summary_text = f"Auto-summary failed: {str(e)}"

    summary_entry = {
        "summary": summary_text,
        "timestamp": datetime.utcnow().isoformat()
    }

    # Trim and inject summary into memory
    trimmed_memory = memory[-RECENT_TURNS_TO_KEEP:]
    trimmed_memory.insert(0, {"role": "system", "content": f"Previous Summary:\n{summary_text}"})

    # Update summary history
    updated_summary_history = summary_history or []
    updated_summary_history.append(summary_entry)

    return trimmed_memory, updated_summary_history


