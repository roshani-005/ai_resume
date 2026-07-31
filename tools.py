"""
memory.py
---------
Lightweight per-session memory so a recruiter can analyze multiple
candidates against the same job description in one session, and the
agent can reference prior analyses (e.g. "how does this candidate
compare to the last one you showed me?").

In-memory only — swap the dict for Redis/a DB for multi-instance deployments.
"""

from collections import defaultdict
from typing import Dict, List

_session_store: Dict[str, List[dict]] = defaultdict(list)

MAX_HISTORY_PER_SESSION = 20


def add_analysis(session_id: str, record: dict) -> None:
    """Store a completed resume analysis for a session."""
    history = _session_store[session_id]
    history.append(record)
    if len(history) > MAX_HISTORY_PER_SESSION:
        history.pop(0)


def get_history(session_id: str) -> List[dict]:
    """Return all analyses run in this session, most recent last."""
    return _session_store.get(session_id, [])


def get_last_analysis(session_id: str) -> dict | None:
    history = _session_store.get(session_id, [])
    return history[-1] if history else None


def clear_session(session_id: str) -> None:
    _session_store.pop(session_id, None)
