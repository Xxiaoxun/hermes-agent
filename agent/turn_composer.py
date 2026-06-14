"""Turn Composer — ephemeral tail-injection without prefix mutation.

Injects transient per-turn data (memory updates, ephemeral system prompt,
background-job completions, active goals) into the **tail** of the API
request as a *separate* user message — never into the system prompt and
never by mutating existing historical messages.

Design reference: DeepSeek-Reasonix ``internal/control/input.go`` ``Compose()``.

Key invariants:
* The system prompt remains byte-stable across turns.
* Historical messages are append-only (never modified mid-session).
* Injected ephemeral content does NOT enter persisted session history —
  it only affects the API-request copy.
* When the last historical message is already ``user``, ephemeral content
  is merged into it (to preserve strict role alternation).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Prefixes used to identify ephemeral messages so the breakpoint strategy
# can skip them (they change every turn).
_EPHEMERAL_PREFIXES = (
    "<memory-update>",
    "<active-goal>",
    "<background-jobs>",
    "<ephemeral-system>",
)


def is_ephemeral_message(msg: Dict[str, Any]) -> bool:
    """Return True if *msg* was injected by :class:`TurnComposer`."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content.startswith(_EPHEMERAL_PREFIXES)
    if isinstance(content, list):
        # Anthropic-format content blocks
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")
                if text.startswith(_EPHEMERAL_PREFIXES):
                    return True
    return False


class TurnComposer:
    """Accumulates per-turn ephemeral data and emits it as tail messages.

    Usage::

        composer = TurnComposer()
        composer.queue_memory("learned that X is deprecated")
        composer.set_ephemeral_system("Channel: #general — keep answers short.")

        # At API-call time:
        tail = composer.build_tail_messages(existing_messages=messages)
        api_messages = messages + tail   # or merge if last role == user
        composer.drain()                  # clear pending state
    """

    def __init__(self) -> None:
        self._pending_memory: List[str] = []
        self._pending_ephemeral_system: List[str] = []
        self._pending_notifications: List[str] = []
        self._active_goal: Optional[str] = None
        self._active_goal_status: Optional[str] = None  # "running" | None

    # ── Queue methods ────────────────────────────────────────────

    def queue_memory(self, note: str) -> None:
        """Queue a memory-update note for injection into the next turn.

        The note takes effect immediately (this turn) without touching the
        cached system prefix.  On the next session the memory will be
        folded into the prefix naturally.
        """
        if note and note.strip():
            self._pending_memory.append(note.strip())

    def set_ephemeral_system(self, text: str) -> None:
        """Set (or append) ephemeral system-prompt additions.

        Replaces the old ``agent.ephemeral_system_prompt`` mutation path.
        Content is injected into the tail, not the system message.
        """
        if text and text.strip():
            self._pending_ephemeral_system.append(text.strip())

    def queue_notification(self, text: str) -> None:
        """Queue a background-job completion notification."""
        if text and text.strip():
            self._pending_notifications.append(text.strip())

    def set_active_goal(self, goal: Optional[str], status: str = "running") -> None:
        """Set the active goal block (injected when status == 'running')."""
        self._active_goal = goal
        self._active_goal_status = status if goal else None

    # ── Composition ──────────────────────────────────────────────

    def has_pending(self) -> bool:
        """Return True if any ephemeral data is queued."""
        return bool(
            self._pending_memory
            or self._pending_ephemeral_system
            or self._pending_notifications
            or (self._active_goal and self._active_goal_status == "running")
        )

    def build_tail_messages(
        self,
        existing_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Build ephemeral tail messages for the API request.

        Returns a list of messages to append **after** the conversation
        history.  Does NOT modify *existing_messages*.

        When the last existing message is ``user``, the ephemeral content
        is returned as a merge directive (dict with ``_merge_into_last``
        key) so the caller can append it to the existing user message
        instead of creating a new ``user`` message that would break role
        alternation.
        """
        blocks = self._compose_blocks()
        if not blocks:
            return []

        content = "\n\n".join(blocks)

        # Check last message role to preserve alternation
        last_role = None
        if existing_messages:
            last_role = existing_messages[-1].get("role")

        if last_role == "user":
            # Merge into existing user message — caller handles the merge
            return [{"_merge_into_last": True, "content": content}]
        else:
            # Safe to append as a new user message
            return [{"role": "user", "content": content}]

    def _compose_blocks(self) -> List[str]:
        """Collect all pending ephemeral data into text blocks."""
        blocks: List[str] = []

        # 1. Active goal
        if self._active_goal and self._active_goal_status == "running":
            blocks.append(f"<active-goal>\n{self._active_goal}\n</active-goal>")

        # 2. Memory updates
        if self._pending_memory:
            lines = "\n".join(f"- {n}" for n in self._pending_memory)
            blocks.append(
                "<memory-update>\n"
                "The following memory changes were just made and apply from now on:\n"
                f"{lines}\n"
                "</memory-update>"
            )

        # 3. Ephemeral system prompt (replaces agent.ephemeral_system_prompt)
        if self._pending_ephemeral_system:
            blocks.append(
                "<ephemeral-system>\n"
                + "\n\n".join(self._pending_ephemeral_system)
                + "\n</ephemeral-system>"
            )

        # 4. Background job completions
        if self._pending_notifications:
            lines = "\n".join(self._pending_notifications)
            blocks.append(f"<background-jobs>\n{lines}\n</background-jobs>")

        return blocks

    def drain(self) -> None:
        """Clear all pending ephemeral data after it has been emitted.

        Called after the API request is assembled.  Goal state is preserved
        (it persists across turns until explicitly cleared).
        """
        self._pending_memory.clear()
        self._pending_ephemeral_system.clear()
        self._pending_notifications.clear()

    def clear_goal(self) -> None:
        """Clear the active goal."""
        self._active_goal = None
        self._active_goal_status = None

    # ── Inspection ───────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of pending state (for debugging)."""
        return {
            "pending_memory": list(self._pending_memory),
            "pending_ephemeral_system": list(self._pending_ephemeral_system),
            "pending_notifications": list(self._pending_notifications),
            "active_goal": self._active_goal,
            "active_goal_status": self._active_goal_status,
        }


# ── Helper for conversation_loop.py ──────────────────────────────

def apply_tail_messages(
    api_messages: List[Dict[str, Any]],
    tail: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply tail messages from :meth:`TurnComposer.build_tail_messages`.

    Handles the merge-into-last-user-message case and appends otherwise.
    Returns the updated message list.  Does not mutate *api_messages*.
    """
    if not tail:
        return api_messages

    result = list(api_messages)  # shallow copy

    for tmsg in tail:
        if tmsg.get("_merge_into_last"):
            # Merge content into the last user message
            content_to_add = tmsg["content"]
            for i in range(len(result) - 1, -1, -1):
                if result[i].get("role") == "user":
                    existing = result[i].get("content", "")
                    if isinstance(existing, str):
                        result[i] = {
                            **result[i],
                            "content": existing + "\n\n" + content_to_add,
                        }
                    elif isinstance(existing, list):
                        # Anthropic content blocks — append a new text block
                        result[i] = {
                            **result[i],
                            "content": existing + [
                                {"type": "text", "text": content_to_add}
                            ],
                        }
                    break
        else:
            result.append(tmsg)

    return result


__all__ = [
    "TurnComposer",
    "is_ephemeral_message",
    "apply_tail_messages",
]
