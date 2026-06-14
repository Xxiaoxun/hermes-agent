"""Anthropic prompt caching strategy.

Two strategies are available:

* ``system_and_3`` (legacy) — 4 ``cache_control`` breakpoints: system prompt
  + last 3 non-system messages, all at the same TTL.
* ``prefix_match`` (recommended) — 2 breakpoints: system prompt (1h TTL) +
  last stable non-ephemeral message (5m TTL).  Designed for prefix-match
  semantics: the conversation prefix from start to the last stable message
  is cached as a single unit, achieving ~99% hit rate on append-only
  histories.

Configuration (``config.yaml``):

.. code-block:: yaml

   prompt_caching:
     strategy: "prefix_match"      # or "system_and_3"
     system_ttl: "1h"              # TTL for system-message breakpoint
     conversation_ttl: "5m"        # TTL for conversation breakpoint

Pure functions — no class state, no AIAgent dependency.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

from agent.turn_composer import is_ephemeral_message


# ── Internal helpers ─────────────────────────────────────────────

def _apply_cache_marker(msg: dict, cache_marker: dict, native_anthropic: bool = False) -> None:
    """Add cache_control to a single message, handling all format variations."""
    role = msg.get("role", "")
    content = msg.get("content")

    if role == "tool":
        if native_anthropic:
            msg["cache_control"] = cache_marker
        return

    if content is None or content == "":
        msg["cache_control"] = cache_marker
        return

    if isinstance(content, str):
        msg["content"] = [
            {"type": "text", "text": content, "cache_control": cache_marker}
        ]
        return

    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict):
            last["cache_control"] = cache_marker


def _build_marker(ttl: str) -> Dict[str, str]:
    """Build a cache_control marker dict for the given TTL ('5m' or '1h')."""
    marker: Dict[str, str] = {"type": "ephemeral"}
    if ttl == "1h":
        marker["ttl"] = "1h"
    return marker


# ── Strategy: system_and_3 (legacy) ─────────────────────────────

def _apply_system_and_3(
    messages: List[Dict[str, Any]],
    marker: Dict[str, str],
    native_anthropic: bool = False,
) -> List[Dict[str, Any]]:
    """Legacy strategy: system prompt + last 3 non-system messages."""
    if not messages:
        return messages

    breakpoints_used = 0

    if messages[0].get("role") == "system":
        _apply_cache_marker(messages[0], marker, native_anthropic=native_anthropic)
        breakpoints_used += 1

    remaining = 4 - breakpoints_used
    non_sys = [i for i in range(len(messages)) if messages[i].get("role") != "system"]
    for idx in non_sys[-remaining:]:
        _apply_cache_marker(messages[idx], marker, native_anthropic=native_anthropic)

    return messages


# ── Strategy: prefix_match (recommended) ─────────────────────────

def _apply_prefix_match(
    messages: List[Dict[str, Any]],
    system_marker: Dict[str, str],
    conversation_marker: Dict[str, str],
    native_anthropic: bool = False,
) -> List[Dict[str, Any]]:
    """Prefix-match strategy: 2 breakpoints for ~99% cache coverage.

    Breakpoint 1 (system, long TTL): caches the static prefix
    (tools + system prompt).  Always byte-stable when the system
    prompt is frozen.

    Breakpoint 2 (last stable message, short TTL): caches the full
    conversation prefix up to the last non-ephemeral message.  On
    append-only histories, this prefix grows by one message per turn,
    so the cached prefix from turn N hits on turn N+1 — only the
    newly appended tokens miss.

    Design reference: DeepSeek-Reasonix ``anthropic.go:255-269`` —
    "Max 4 breakpoints; we use ≤2."
    """
    # ── Breakpoint 1: system message (long TTL) ──
    if messages and messages[0].get("role") == "system":
        _apply_cache_marker(messages[0], system_marker, native_anthropic=native_anthropic)

    # ── Breakpoint 2: last stable non-ephemeral message (short TTL) ──
    # Walk backwards to find the last message NOT injected by TurnComposer.
    # Ephemeral messages change every turn — placing a breakpoint on them
    # would always miss.
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "system":
            continue  # already handled
        if not is_ephemeral_message(msg):
            _apply_cache_marker(msg, conversation_marker, native_anthropic=native_anthropic)
            break

    return messages


# ── Public API ───────────────────────────────────────────────────

def apply_anthropic_cache_control(
    api_messages: List[Dict[str, Any]],
    cache_ttl: str = "5m",
    native_anthropic: bool = False,
    strategy: str = "system_and_3",
    system_ttl: str = "1h",
    conversation_ttl: str = "5m",
) -> List[Dict[str, Any]]:
    """Apply caching strategy to messages for Anthropic-compatible models.

    Args:
        api_messages: Messages to annotate (not mutated).
        cache_ttl: Default TTL for legacy ``system_and_3`` strategy.
        native_anthropic: True when using the native Anthropic API
            (vs OpenRouter / OpenAI-wire envelope).
        strategy: ``"system_and_3"`` (legacy) or ``"prefix_match"``
            (recommended).
        system_ttl: TTL for the system-message breakpoint in
            ``prefix_match`` mode.
        conversation_ttl: TTL for the conversation breakpoint in
            ``prefix_match`` mode.

    Returns:
        Deep copy of messages with cache_control breakpoints injected.
    """
    messages = copy.deepcopy(api_messages)
    if not messages:
        return messages

    if strategy == "prefix_match":
        sys_marker = _build_marker(system_ttl)
        conv_marker = _build_marker(conversation_ttl)
        return _apply_prefix_match(messages, sys_marker, conv_marker, native_anthropic)
    else:
        # Default / legacy: system_and_3
        marker = _build_marker(cache_ttl)
        return _apply_system_and_3(messages, marker, native_anthropic)
