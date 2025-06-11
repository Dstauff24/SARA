# memory_summarizer.py

from openai import OpenAI

# Assumes OpenAI key is already loaded via environment
client = OpenAI()

def summarize_conversation(messages: list[str]) -> str:
    """
    Summarize a list of seller messages into a concise recap that captures:
    - Motivation
    - Timeline
    - Price expectations
    - Property condition
    """
    prompt = [
        {"role": "system", "content": "Summarize the following seller messages for internal use by an acquisitions rep. Capture motivation, timeline, condition, and pricing if mentioned."},
        {"role": "user", "content": "\n".join(messages)}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=prompt,
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error generating summary: {e}]"

def reduce_conversation_log(full_log: list[dict], summary: str, keep_turns: int = 4) -> list[dict]:
    """
    Reduces the conversation memory to a summarized version.
    - Keeps the last `keep_turns` user/assistant messages
    - Prepends the generated summary as a system message
    """
    if not full_log:
        return [{"role": "system", "content": f"Conversation Summary: {summary}"}]

    trimmed_log = full_log[-(keep_turns * 2):]  # Approx. 4 user/assistant turns
    summary_entry = {"role": "system", "content": f"Conversation Summary: {summary}"}
    return [summary_entry] + trimmed_log

