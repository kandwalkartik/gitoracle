"""
conversation_store.py — GitOracle Session Persistence
======================================================
Saves and loads conversation history to/from JSON files so sessions
survive terminal restarts.

Design:
  - Each session is a file: sessions/<session_id>.json
  - History format matches the OpenAI messages list exactly
  - Automatic session ID based on date (one session per day by default)
  - History is trimmed to CFG.max_history_turns on save to prevent bloat

Usage:
    from conversation_store import ConversationStore
    store = ConversationStore()
    store.load(history)           # fills existing list in-place
    store.save(history)           # persists to disk
    store.list_sessions()         # show all saved sessions
    store.clear()                 # delete current session file
"""

import os
import json
import datetime
from typing import Optional

from config import CFG
from logger import get_logger

log = get_logger("Store")


class ConversationStore:
    """JSON-backed conversation history store."""

    def __init__(self, session_id: Optional[str] = None):
        """
        Args:
            session_id: Custom session name. Defaults to today's date
                        (YYYY-MM-DD), giving one persistent session per day.
        """
        self.sessions_dir = CFG.sessions_dir
        os.makedirs(self.sessions_dir, exist_ok=True)
        self.session_id = session_id or datetime.date.today().isoformat()
        self._path = os.path.join(self.sessions_dir, f"{self.session_id}.json")

    # ── Public API ──────────────────────────────────────────────────────────

    def load(self, history: list) -> list:
        """
        Populate `history` in-place from the saved session file.
        Returns the same list (for chaining).
        """
        if not os.path.exists(self._path):
            log.debug(f"No saved session found for '{self.session_id}' — starting fresh.")
            return history

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            history.extend(saved)
            log.info(f"Loaded {len(saved)} messages from session '{self.session_id}'.")
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Could not load session '{self.session_id}': {e}")

        return history

    def save(self, history: list):
        """
        Persist `history` to disk, trimmed to the most recent
        CFG.max_history_turns turn-pairs (user + assistant = 1 pair).
        """
        trimmed = self._trim(history)
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(trimmed, f, ensure_ascii=False, indent=2)
            log.debug(f"Session '{self.session_id}' saved ({len(trimmed)} messages).")
        except OSError as e:
            log.warning(f"Could not save session '{self.session_id}': {e}")

    def clear(self):
        """Delete the current session file."""
        if os.path.exists(self._path):
            os.remove(self._path)
            log.info(f"Session '{self.session_id}' cleared.")
        else:
            log.warning(f"No session file found for '{self.session_id}'.")

    def list_sessions(self) -> list[str]:
        """Return a list of all saved session IDs (file stems)."""
        if not os.path.isdir(self.sessions_dir):
            return []
        return [
            f[:-5] for f in os.listdir(self.sessions_dir)
            if f.endswith(".json")
        ]

    # ── Internal ────────────────────────────────────────────────────────────

    def _trim(self, history: list) -> list:
        """
        Keep only the last N turn-pairs from history.
        A turn-pair is one user message + one assistant message = 2 entries.
        Tool messages are counted separately (they can accompany assistant turns).
        """
        max_msgs = CFG.max_history_turns * 3  # user + assistant + possible tool msgs per turn
        if len(history) <= max_msgs:
            return history
        trimmed = history[-max_msgs:]
        log.debug(f"History trimmed from {len(history)} → {len(trimmed)} messages.")
        return trimmed
