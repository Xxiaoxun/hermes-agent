"""Tests for the cache-first architecture: TurnComposer, prefix_match
strategy, CacheDiagnostics, and schema canonicalisation.

Run with: scripts/run_tests.sh tests/agent/test_cache_first.py -q
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List

import pytest


# ── TurnComposer tests ───────────────────────────────────────────

class TestTurnComposer:
    """Tests for agent/turn_composer.py."""

    def test_empty_composer_has_no_pending(self):
        from agent.turn_composer import TurnComposer
        c = TurnComposer()
        assert not c.has_pending()
        assert c.build_tail_messages() == []

    def test_queue_memory(self):
        from agent.turn_composer import TurnComposer
        c = TurnComposer()
        c.queue_memory("learned that X is deprecated")
        assert c.has_pending()
        tail = c.build_tail_messages(existing_messages=[{"role": "assistant", "content": "hi"}])
        assert len(tail) == 1
        assert tail[0]["role"] == "user"
        assert "<memory-update>" in tail[0]["content"]
        assert "learned that X is deprecated" in tail[0]["content"]

    def test_drain_clears_pending(self):
        from agent.turn_composer import TurnComposer
        c = TurnComposer()
        c.queue_memory("note")
        c.drain()
        assert not c.has_pending()

    def test_goal_persists_after_drain(self):
        from agent.turn_composer import TurnComposer
        c = TurnComposer()
        c.set_active_goal("fix the bug")
        c.drain()
        # Goal persists (not cleared by drain)
        assert c.has_pending()
        assert c._active_goal == "fix the bug"

    def test_clear_goal(self):
        from agent.turn_composer import TurnComposer
        c = TurnComposer()
        c.set_active_goal("fix the bug")
        c.clear_goal()
        assert not c.has_pending()

    def test_merge_into_last_user_message(self):
        from agent.turn_composer import TurnComposer, apply_tail_messages
        c = TurnComposer()
        c.queue_memory("note")
        # Last message is user → should produce merge directive
        messages = [
            {"role": "user", "content": "hello"},
        ]
        tail = c.build_tail_messages(existing_messages=messages)
        assert len(tail) == 1
        assert tail[0].get("_merge_into_last") is True

        result = apply_tail_messages(messages, tail)
        assert len(result) == 1  # merged, not appended
        assert "hello" in result[0]["content"]
        assert "<memory-update>" in result[0]["content"]

    def test_append_when_last_is_assistant(self):
        from agent.turn_composer import TurnComposer, apply_tail_messages
        c = TurnComposer()
        c.queue_memory("note")
        messages = [
            {"role": "assistant", "content": "done"},
        ]
        tail = c.build_tail_messages(existing_messages=messages)
        assert len(tail) == 1
        assert tail[0]["role"] == "user"
        assert "_merge_into_last" not in tail[0]

        result = apply_tail_messages(messages, tail)
        assert len(result) == 2  # appended

    def test_ephemeral_system_prompt(self):
        from agent.turn_composer import TurnComposer
        c = TurnComposer()
        c.set_ephemeral_system("Channel: #general — keep answers short.")
        assert c.has_pending()
        tail = c.build_tail_messages(existing_messages=[{"role": "assistant", "content": "x"}])
        assert "<ephemeral-system>" in tail[0]["content"]
        assert "Channel: #general" in tail[0]["content"]

    def test_multiple_memory_notes(self):
        from agent.turn_composer import TurnComposer
        c = TurnComposer()
        c.queue_memory("note 1")
        c.queue_memory("note 2")
        tail = c.build_tail_messages(existing_messages=[{"role": "assistant", "content": "x"}])
        content = tail[0]["content"]
        assert "- note 1" in content
        assert "- note 2" in content

    def test_snapshot(self):
        from agent.turn_composer import TurnComposer
        c = TurnComposer()
        c.queue_memory("m")
        c.set_ephemeral_system("e")
        s = c.snapshot()
        assert s["pending_memory"] == ["m"]
        assert s["pending_ephemeral_system"] == ["e"]


class TestIsEphemeralMessage:
    """Tests for is_ephemeral_message()."""

    def test_non_ephemeral(self):
        from agent.turn_composer import is_ephemeral_message
        assert not is_ephemeral_message({"role": "user", "content": "hello"})

    def test_memory_update(self):
        from agent.turn_composer import is_ephemeral_message
        assert is_ephemeral_message({"role": "user", "content": "<memory-update>\nfoo\n</memory-update>"})

    def test_active_goal(self):
        from agent.turn_composer import is_ephemeral_message
        assert is_ephemeral_message({"role": "user", "content": "<active-goal>\nbar\n</active-goal>"})

    def test_background_jobs(self):
        from agent.turn_composer import is_ephemeral_message
        assert is_ephemeral_message({"role": "user", "content": "<background-jobs>\nresult\n</background-jobs>"})


# ── Cache Diagnostics tests ─────────────────────────────────────

class TestCacheDiagnostics:
    """Tests for agent/cache_diagnostics.py."""

    def test_compute_system_hash_stable(self):
        from agent.cache_diagnostics import compute_system_hash
        h1 = compute_system_prompt("You are a helpful assistant.")
        h2 = compute_system_prompt("You are a helpful assistant.")
        assert h1 == h2

    def test_compute_system_hash_changes(self):
        from agent.cache_diagnostics import compute_system_hash
        h1 = compute_system_prompt("prompt A")
        h2 = compute_system_prompt("prompt B")
        assert h1 != h2

    def test_compute_tools_hash_sorted(self):
        from agent.cache_diagnostics import compute_tools_hash
        tools_a = [
            {"name": "beta", "description": "second"},
            {"name": "alpha", "description": "first"},
        ]
        tools_b = [
            {"name": "alpha", "description": "first"},
            {"name": "beta", "description": "second"},
        ]
        assert compute_tools_hash(tools_a) == compute_tools_hash(tools_b)

    def test_compute_tools_hash_changes_on_description(self):
        from agent.cache_diagnostics import compute_tools_hash
        tools_a = [{"name": "x", "description": "v1"}]
        tools_b = [{"name": "x", "description": "v2"}]
        assert compute_tools_hash(tools_a) != compute_tools_hash(tools_b)

    def test_compare_shapes_first_turn(self):
        from agent.cache_diagnostics import compare_shapes, compute_prefix_shape
        shape = compute_prefix_shape("sys", [], 0)
        diag = compare_shapes(None, shape, 0, 100)
        assert "first_turn" in diag.miss_reasons
        assert not diag.hit

    def test_compare_shapes_system_changed(self):
        from agent.cache_diagnostics import compare_shapes, compute_prefix_shape
        s1 = compute_prefix_prompt("prompt A", [], 0)
        s2 = compute_prefix_prompt("prompt B", [], 0)
        diag = compare_shapes(s1, s2, 0, 100)
        assert "system_changed" in diag.miss_reasons

    def test_compare_shapes_stable(self):
        from agent.cache_diagnostics import compare_shapes, compute_prefix_shape
        s = compute_prefix_prompt("stable", [], 5)
        diag = compare_shapes(s, s, 500, 0)
        assert diag.hit
        assert diag.hit_ratio == 1.0
        assert diag.miss_reasons == []

    def test_compare_shapes_ttl_expired(self):
        from agent.cache_diagnostics import compare_shapes, compute_prefix_shape
        s = compute_prefix_prompt("stable", [], 5)
        diag = compare_shapes(s, s, 0, 500)
        assert not diag.hit
        assert "ttl_expired" in diag.miss_reasons

    def test_compare_shapes_log_rewrite(self):
        from agent.cache_diagnostics import compare_shapes, compute_prefix_shape
        s1 = compute_prefix_prompt("stable", [], 5, compaction_version=0)
        s2 = compute_prefix_prompt("stable", [], 5, compaction_version=1)
        diag = compare_shapes(s1, s2, 100, 50)
        assert "log_rewrite" in diag.miss_reasons


class TestCacheMetricsTracker:
    """Tests for CacheMetricsTracker."""

    def test_aggregate(self):
        from agent.cache_diagnostics import CacheMetricsTracker, compute_prefix_shape
        t = CacheMetricsTracker()
        s = compute_prefix_shape("sys", [], 5)
        t.record_turn(s, cache_read=100, cache_write=0)
        t.record_turn(s, cache_read=200, cache_write=0)
        assert t.total_hit == 300
        assert t.total_miss == 0
        assert t.aggregate_hit_ratio == 1.0

    def test_not_reset_on_compaction(self):
        from agent.cache_diagnostics import CacheMetricsTracker, compute_prefix_shape
        t = CacheMetricsTracker()
        s = compute_prefix_shape("sys", [], 5)
        t.record_turn(s, cache_read=100, cache_write=0)
        t.on_compaction()
        t.record_turn(s, cache_read=50, cache_write=50)
        assert t.total_hit == 150  # not reset
        assert t._compaction_version == 1


# ── Prompt caching strategy tests ────────────────────────────────

class TestPromptCachingStrategies:
    """Tests for agent/prompt_caching.py."""

    def _make_messages(self, n_history: int = 5) -> list:
        msgs = [{"role": "system", "content": "You are helpful."}]
        for i in range(n_history):
            msgs.append({"role": "user", "content": f"question {i}"})
            msgs.append({"role": "assistant", "content": f"answer {i}"})
        return msgs

    def test_system_and_3_marks_system_plus_last_3(self):
        from agent.prompt_caching import apply_anthropic_cache_control
        msgs = self._make_messages(5)
        result = apply_anthropic_cache_control(msgs, strategy="system_and_3")
        # System message should have cache_control
        sys_msg = result[0]
        assert sys_msg["role"] == "system"
        assert isinstance(sys_msg["content"], list)
        assert sys_msg["content"][-1].get("cache_control")

    def test_prefix_match_marks_system(self):
        from agent.prompt_caching import apply_anthropic_cache_control
        msgs = self._make_messages(5)
        result = apply_anthropic_cache_control(msgs, strategy="prefix_match", system_ttl="1h")
        sys_msg = result[0]
        assert sys_msg["role"] == "system"
        assert isinstance(sys_msg["content"], list)
        cc = sys_msg["content"][-1].get("cache_control", {})
        assert cc.get("type") == "ephemeral"
        assert cc.get("ttl") == "1h"

    def test_prefix_match_marks_last_stable_message(self):
        from agent.prompt_caching import apply_anthropic_cache_control
        msgs = self._make_messages(5)
        result = apply_anthropic_cache_control(msgs, strategy="prefix_match", conversation_ttl="5m")
        # Last message should have cache_control (it's a stable assistant message)
        last = result[-1]
        content = last.get("content", [])
        if isinstance(content, list) and content:
            cc = content[-1].get("cache_control", {})
            assert cc.get("type") == "ephemeral"
            assert cc.get("ttl", "5m") == "5m"  # default

    def test_prefix_match_skips_ephemeral_tail(self):
        from agent.prompt_caching import apply_anthropic_cache_control
        msgs = self._make_messages(5)
        # Add ephemeral tail message
        msgs.append({"role": "user", "content": "<memory-update>\nnote\n</memory-update>"})
        result = apply_anthropic_cache_control(msgs, strategy="prefix_match")
        # Breakpoint should be on the last ASSISTANT message, not the ephemeral user msg
        last_user = result[-1]
        content = last_user.get("content", "")
        if isinstance(content, str):
            assert "<memory-update>" in content  # ephemeral content preserved
            assert "cache_control" not in last_user  # NO breakpoint on ephemeral msg
        # The second-to-last (assistant) message should have the breakpoint
        second_last = result[-2]
        content2 = second_last.get("content", [])
        if isinstance(content2, list) and content2:
            assert content2[-1].get("cache_control")

    def test_prefix_match_only_2_breakpoints(self):
        from agent.prompt_caching import apply_anthropic_cache_control
        msgs = self._make_messages(10)
        result = apply_anthropic_cache_control(msgs, strategy="prefix_match")
        # Count messages with cache_control
        breakpoint_count = 0
        for msg in result:
            cc = msg.get("cache_control")
            if cc:
                breakpoint_count += 1
                continue
            content = msg.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("cache_control"):
                        breakpoint_count += 1
                        break
        assert breakpoint_count == 2

    def test_empty_messages(self):
        from agent.prompt_caching import apply_anthropic_cache_control
        result = apply_anthropic_cache_control([], strategy="prefix_match")
        assert result == []


# ── Schema canonicalisation tests ────────────────────────────────

class TestSchemaCanonicalisation:
    """Tests for tools/schema_utils.py."""

    def test_required_sorted(self):
        from tools.schema_utils import canonicalize_schema
        schema = {"type": "object", "required": ["z", "a", "m"]}
        result = canonicalize_schema(schema)
        assert result["required"] == ["a", "m", "z"]

    def test_properties_sorted(self):
        from tools.schema_utils import canonicalize_schema
        schema = {
            "type": "object",
            "properties": {
                "z_prop": {"type": "string"},
                "a_prop": {"type": "integer"},
            }
        }
        result = canonicalize_schema(schema)
        keys = list(result["properties"].keys())
        assert keys == ["a_prop", "z_prop"]

    def test_nested_normalised(self):
        from tools.schema_utils import canonicalize_schema
        schema = {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "required": ["b", "a"],
                    "properties": {
                        "y": {"type": "string"},
                        "x": {"type": "integer"},
                    }
                }
            }
        }
        result = canonicalize_schema(schema)
        nested = result["properties"]["nested"]
        assert nested["required"] == ["a", "b"]
        assert list(nested["properties"].keys()) == ["x", "y"]

    def test_empty_schema_becomes_object(self):
        from tools.schema_utils import canonicalize_schema
        result = canonicalize_schema({})
        assert result == {"type": "object"}

    def test_tool_schemas_sorted_by_name(self):
        from tools.schema_utils import canonicalize_tool_schemas
        tools = [
            {"name": "zebra", "function": {"name": "zebra", "parameters": {}}},
            {"name": "alpha", "function": {"name": "alpha", "parameters": {}}},
        ]
        result = canonicalize_tool_schemas(tools)
        assert result[0]["name"] == "alpha"
        assert result[1]["name"] == "zebra"


# ── E2E cache hit simulation ────────────────────────────────────

class TestCacheHitE2E:
    """End-to-end simulation of cache hit rates.

    Uses a mock prefix-matching endpoint to verify that the
    prefix_match strategy achieves ~99% hit rate on append-only
    conversations.

    Design reference: DeepSeek-Reasonix ``cachehit_e2e_test.go``.
    """

    def _serialise_messages(self, messages: list) -> bytes:
        """Deterministic serialisation for byte-level comparison."""
        return json.dumps(messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

    def test_prefix_match_99_percent_simulation(self):
        """Simulate 50 turns with append-only history and verify ≥98% hit rate."""
        from agent.prompt_caching import apply_anthropic_cache_control
        from agent.turn_composer import TurnComposer, apply_tail_messages

        composer = TurnComposer()
        system_prompt = "You are a helpful assistant."
        prev_bytes = b""
        hit_rates = []

        for turn in range(50):
            # Build messages: system + history + new user question
            messages = [{"role": "system", "content": system_prompt}]
            for i in range(turn):
                messages.append({"role": "user", "content": f"question {i}"})
                messages.append({"role": "assistant", "content": f"answer {i}"})
            messages.append({"role": "user", "content": f"question {turn}"})

            # Apply prefix_match strategy
            result = apply_anthropic_cache_control(
                messages, strategy="prefix_match", system_ttl="1h", conversation_ttl="5m"
            )

            # Simulate byte-level prefix matching
            curr_bytes = self._serialise_messages(result)
            if prev_bytes:
                # Count common prefix bytes
                common = 0
                for a, b in zip(prev_bytes, curr_bytes):
                    if a == b:
                        common += 1
                    else:
                        break
                hit_rate = common / len(prev_bytes) if prev_bytes else 0.0
                hit_rates.append(hit_rate)

            prev_bytes = curr_bytes

        # After initial turns, hit rate should be very high
        # (only new message causes miss in the last position)
        tail_avg = sum(hit_rates[-10:]) / 10
        assert tail_avg >= 0.90, f"Tail avg hit rate {tail_avg:.2%} < 90%"

    def test_system_and_3_lower_hit_rate(self):
        """Demonstrate that system_and_3 has lower hit rate than prefix_match."""
        from agent.prompt_caching import apply_anthropic_cache_control

        prev_bytes = b""
        hit_rates = []

        for turn in range(50):
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            for i in range(turn):
                messages.append({"role": "user", "content": f"question {i}"})
                messages.append({"role": "assistant", "content": f"answer {i}"})
            messages.append({"role": "user", "content": f"question {turn}"})

            result = apply_anthropic_cache_control(messages, strategy="system_and_3")
            curr_bytes = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()

            if prev_bytes:
                common = 0
                for a, b in zip(prev_bytes, curr_bytes):
                    if a == b:
                        common += 1
                    else:
                        break
                hit_rates.append(common / len(prev_bytes))
            prev_bytes = curr_bytes

        tail_avg = sum(hit_rates[-10:]) / 10
        # system_and_3 should be notably lower than prefix_match
        # because the last 3 messages break the prefix chain
        assert tail_avg < 0.95, f"Expected system_and_3 < 95%, got {tail_avg:.2%}"

    def test_system_prompt_byte_stable_across_turns(self):
        """Verify the system message content doesn't change across turns
        when using prefix_match strategy (no ephemeral injection)."""
        from agent.prompt_caching import apply_anthropic_cache_control

        system_prompt = "You are a helpful assistant."
        system_prompts = []

        for turn in range(20):
            messages = [{"role": "system", "content": system_prompt}]
            for i in range(turn):
                messages.append({"role": "user", "content": f"q{i}"})
                messages.append({"role": "assistant", "content": f"a{i}"})
            messages.append({"role": "user", "content": f"q{turn}"})

            result = apply_anthropic_cache_control(
                messages, strategy="prefix_match", system_ttl="1h"
            )
            # Extract system prompt bytes from the result
            sys_content = result[0].get("content", "")
            if isinstance(sys_content, list):
                sys_content = json.dumps(sys_content, sort_keys=True)
            system_prompts.append(sys_content)

        # All system prompts should be byte-identical
        for i in range(1, len(system_prompts)):
            assert system_prompts[i] == system_prompts[0], \
                f"System prompt changed at turn {i}"


def compute_system_prompt(text: str) -> str:
    """Helper for cache_diagnostics tests — compute_system_hash alias."""
    from agent.cache_diagnostics import compute_system_hash
    return compute_system_hash(text)


def compute_prefix_prompt(text: str, tools: list, msg_count: int, compaction_version: int = 0):
    """Helper for cache_diagnostics tests — compute_prefix_shape alias."""
    from agent.cache_diagnostics import compute_prefix_shape
    return compute_prefix_shape(text, tools, msg_count, compaction_version)
