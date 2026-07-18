"""
config.py — GitOracle Centralized Configuration
================================================
Single source of truth for all environment variables.

Option C behaviour:
  1. First tries to load a value from the .env file (silent, automatic)
  2. If a critical variable is missing, prompts the user on the terminal
  3. Offers to save the answer back into .env so they're never asked again

This means the first run is guided and interactive; every run after that
is fully automatic.
"""

import os
import sys
import getpass
from dataclasses import dataclass, field
from dotenv import load_dotenv, set_key

# Absolute path to .env — always next to this file, regardless of cwd
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

# Load whatever is already in .env (override=False so real env vars win)
load_dotenv(dotenv_path=_ENV_PATH, override=False)


# ── Prompt helper ─────────────────────────────────────────────────────────────

def _prompt_and_save(
    env_key: str,
    label: str,
    hint: str = "",
    secret: bool = False,
    optional: bool = False,
) -> str:
    """
    Ask the user for a missing environment variable value on the terminal.
    Offers to save it into .env so they are not asked again next time.

    Args:
        env_key  : The variable name, e.g. "GITHUB_TOKEN"
        label    : Human-readable name shown in the prompt
        hint     : Where to get the value (URL / instructions)
        secret   : If True, input is hidden (like a password)
        optional : If True, the user can skip by pressing Enter

    Returns:
        The value entered, or "" if skipped.
    """
    print()
    print(f"  [Setup] {label} is not configured.")
    if hint:
        print(f"  [Hint]  {hint}")
    if optional:
        print(f"  Press Enter to skip (limited functionality).")

    try:
        if secret:
            value = getpass.getpass(f"  Enter {env_key}: ").strip()
        else:
            value = input(f"  Enter {env_key}: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n  Skipped.")
        return ""

    if not value:
        if optional:
            print(f"  Skipped. You can add {env_key} to your .env file later.")
        return ""

    # Offer to save into .env
    try:
        save = input(f"  Save to .env so you're not asked again? [Y/n]: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        save = "n"

    if save in ("", "y", "yes"):
        # Ensure .env file exists before set_key tries to write it
        if not os.path.exists(_ENV_PATH):
            open(_ENV_PATH, "w").close()
        set_key(_ENV_PATH, env_key, value)
        # Also inject into current process environment so the rest of the
        # session sees it without needing to reload the file
        os.environ[env_key] = value
        print(f"  Saved to .env")
    else:
        os.environ[env_key] = value   # use for this session only
        print(f"  Using for this session only.")

    return value


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class Config:
    # -- GitHub ---------------------------------------------------------------
    github_token: str = ""          # Optional; unlocks 5,000 req/hr vs 60
    github_api: str = "https://api.github.com"

    # -- LLM Backend ----------------------------------------------------------
    llm_backend: str = "ollama"     # "ollama" | "groq"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen2.5:3b"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # -- Cache ----------------------------------------------------------------
    cache_ttl_seconds: int = 60
    cache_enabled: bool = True

    # -- History --------------------------------------------------------------
    max_history_turns: int = 20

    # -- Session Store --------------------------------------------------------
    sessions_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sessions"
    ))

    # -- Logging --------------------------------------------------------------
    log_level: str = "INFO"


# ── Loader with Option C fallback ─────────────────────────────────────────────

def load_config(interactive: bool = True) -> Config:
    """
    Build a Config from environment variables.

    Option C flow for each critical variable:
      1. Read from environment / .env  →  use silently if present
      2. Missing?  →  prompt user on terminal
      3. Offer to save the answer into .env for all future runs

    Args:
        interactive: Set False in automated/test contexts to disable prompting.
    """

    # ── Determine backend first (affects which keys are critical) ────────────
    llm_backend = os.environ.get("LLM_BACKEND", "ollama").lower()

    # ── GITHUB_TOKEN (optional but strongly recommended) ─────────────────────
    github_token = os.environ.get("GITHUB_TOKEN", "")
    if not github_token and interactive:
        github_token = _prompt_and_save(
            env_key="GITHUB_TOKEN",
            label="GitHub Personal Access Token",
            hint=(
                "Get one free at: github.com → Settings → Developer settings\n"
                "  → Personal access tokens → Generate new token (classic)\n"
                "  Tick 'public_repo' scope, then copy the token."
            ),
            secret=True,
            optional=True,
        )

    # ── GROQ_API_KEY (required only if LLM_BACKEND=groq) ─────────────────────
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    if llm_backend == "groq" and not groq_api_key:
        if interactive:
            groq_api_key = _prompt_and_save(
                env_key="GROQ_API_KEY",
                label="Groq API Key",
                hint="Get a free key at: console.groq.com → API Keys",
                secret=True,
                optional=False,
            )
        if not groq_api_key:
            print(
                "\n[config] ERROR: LLM_BACKEND=groq but GROQ_API_KEY is not set.\n"
                "  Either provide your Groq key or switch LLM_BACKEND=ollama in .env.",
                file=sys.stderr,
            )
            sys.exit(1)

    # ── Build Config ──────────────────────────────────────────────────────────
    cfg = Config(
        github_token=github_token,
        github_api=os.environ.get("GITHUB_API", "https://api.github.com"),
        llm_backend=llm_backend,
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        ollama_model=os.environ.get("OLLAMA_MODEL", "qwen2.5:3b"),
        groq_api_key=groq_api_key,
        groq_model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
        cache_ttl_seconds=int(os.environ.get("CACHE_TTL_SECONDS", "60")),
        cache_enabled=os.environ.get("CACHE_ENABLED", "true").lower() == "true",
        max_history_turns=int(os.environ.get("MAX_HISTORY_TURNS", "20")),
        log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    )

    return cfg


# ── Module-level singleton ────────────────────────────────────────────────────
# Import this directly in other modules:
#   from config import CFG
CFG = load_config()
