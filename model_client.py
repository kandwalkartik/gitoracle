"""
model_client.py — GitOracle LLM Backend Connector
==================================================
Backend-agnostic LLM client factory.

Supported backends:
  - ollama  : Local Ollama via its OpenAI-compatible /v1 endpoint
  - groq    : Groq cloud API (fast inference, free tier available)

Both return a client whose .chat.completions.create() signature is identical,
so the rest of the codebase never needs to know which backend is active.
"""

import sys
from config import CFG
from logger import get_logger

log = get_logger("ModelClient")


def get_client_and_model():
    """
    Returns (client, model_name, backend_name).

    The client implements the OpenAI chat completions interface regardless
    of which backend is chosen — swap backends by changing .env only.
    """
    if CFG.llm_backend == "groq":
        try:
            from groq import Groq
        except ImportError:
            log.error("groq package not installed. Run: pip install groq>=0.8.0")
            sys.exit(1)
        client = Groq(api_key=CFG.groq_api_key)
        log.info(f"Backend: Groq | Model: {CFG.groq_model}")
        return client, CFG.groq_model, "groq"

    # Default: Ollama via OpenAI-compatible endpoint
    # api_key="ollama" is a required-but-unchecked placeholder.
    from openai import OpenAI
    client = OpenAI(base_url=CFG.ollama_base_url, api_key="ollama")
    log.info(f"Backend: Ollama @ {CFG.ollama_base_url} | Model: {CFG.ollama_model}")
    return client, CFG.ollama_model, "ollama"


def check_connection(client, model: str, backend: str) -> bool:
    """
    Pre-flight connectivity check. Sends a minimal request and reports
    a clear, backend-specific action if it fails.

    Returns True on success, False on failure.
    """
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        log.success(f"Connected to {backend} ({model})")
        return True
    except Exception as e:
        if backend == "ollama":
            log.error(f"Cannot reach Ollama at {CFG.ollama_base_url}")
            log.error(f"  1) Start Ollama:       ollama serve")
            log.error(f"  2) Pull the model:     ollama pull {model}")
        else:
            log.error(f"Cannot reach Groq. Check GROQ_API_KEY in your .env file.")
            log.error(f"  Underlying error: {e}")
        return False
