"""
llm_client.py - Shared LLM client

Used by agent, multi_agent, and vault_summarizer.
Degrades gracefully when connection fails.
"""
import requests
from config import LLAMA_SERVER_URL

class LLMUnavailableError(Exception):
    pass

def call_llm(
    messages:    list[dict],
    system:      str   = "",
    max_tokens:  int   = 600,
    temperature: float = 0.1,
    base_url:    str   = "",
) -> str:
    url = (base_url or LLAMA_SERVER_URL).rstrip("/")
    full_messages = (
        [{"role": "system", "content": system}] if system else []
    ) + messages
    payload = {
        "model":       "local",
        "messages":    full_messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
        "stream":      False,
    }
    try:
        r = requests.post(f"{url}/chat/completions", json=payload, timeout=180)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.ConnectionError as e:
        raise LLMUnavailableError("llama-server is offline or unreachable") from e

def tok(text: str) -> int:
    """Estimativa rápida de tokens (4 chars ≈ 1 token)."""
    return max(1, len(text) // 4)
