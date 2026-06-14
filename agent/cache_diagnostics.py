"""Cache diagnostics — prefix shape tracking and miss-reason attribution.

Tracks the SHA-256 fingerprints of cacheable prefix components (system prompt,
tool schemas) across turns and computes ``CacheDiagnostics`` explaining *why*
a cache miss occurred.

Design reference: DeepSeek-Reasonix ``internal/agent/cache_shape.go`` +
``CompareShape()``.

Pure data + functions — no AIAgent dependency, no I/O.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Prefix Shape ─────────────────────────────────────────────────

@dataclass
class PrefixShape:
    """Fingerprint of the cacheable prefix sent to the provider.

    Captured **before** the API call so consecutive shapes can be compared
    to diagnose cache-miss reasons.
    """
    system_hash: str            # SHA-256[:16] of the system prompt string
    tools_hash: str             # SHA-256[:16] of normalised tool schemas
    combined_hash: str          # SHA-256[:16] of system + tools combined
    compaction_version: int     # bumped on each compaction / prune event
    tool_schema_tokens: int     # estimated token count of tool schemas
    message_count: int          # number of non-system messages


# ── Cache Diagnostics ────────────────────────────────────────────

@dataclass
class CacheDiagnostics:
    """Per-turn cache-miss diagnosis attached to usage events."""
    hit: bool
    miss_reasons: List[str] = field(default_factory=list)
    prev_shape: Optional[PrefixShape] = None
    curr_shape: Optional[PrefixShape] = None
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def hit_ratio(self) -> float:
        total = self.cache_read_tokens + self.cache_write_tokens
        if total == 0:
            return 0.0
        return self.cache_read_tokens / total

    @property
    def effective_breakpoints(self) -> int:
        """How many breakpoints are contributing to cache hits."""
        if not self.curr_shape:
            return 0
        count = 0
        if self.curr_shape.system_hash:
            count += 1
        if self.curr_shape.message_count > 0:
            count += 1  # conversation prefix breakpoint
        return count

    @property
    def prefix_coverage_hint(self) -> str:
        """Human-readable estimate of prefix coverage."""
        ratio = self.hit_ratio
        if ratio >= 0.95:
            return "excellent"
        if ratio >= 0.80:
            return "good"
        if ratio >= 0.50:
            return "moderate"
        return "poor"


# ── Hashing helpers ──────────────────────────────────────────────

def _sha256_short(data: str, length: int = 16) -> str:
    """Return the first *length* hex chars of SHA-256(data)."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:length]


def compute_system_hash(system_prompt: str) -> str:
    """SHA-256 fingerprint of the system prompt."""
    return _sha256_short(system_prompt)


def compute_tools_hash(tools: List[Dict[str, Any]]) -> str:
    """SHA-256 fingerprint of normalised tool schemas.

    Normalisation:
    * Sort tool list by ``name``.
    * Sort ``required`` arrays in ``parameters``.
    * Deterministic JSON serialisation (sort_keys, no extra whitespace).
    """
    if not tools:
        return ""

    normalised = []
    for t in sorted(tools, key=lambda x: x.get("name", x.get("function", {}).get("name", ""))):
        entry = dict(t)
        # Normalise function.parameters.required if present
        func = entry.get("function")
        if isinstance(func, dict):
            params = func.get("parameters")
            if isinstance(params, dict):
                params = dict(params)
                req = params.get("required")
                if isinstance(req, list):
                    params["required"] = sorted(req)
                func = {**func, "parameters": params}
                entry = {**entry, "function": func}
        normalised.append(entry)

    serialised = json.dumps(normalised, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _sha256_short(serialised)


# ── Shape computation ────────────────────────────────────────────

def compute_prefix_shape(
    system_prompt: str,
    tools: List[Dict[str, Any]],
    message_count: int = 0,
    compaction_version: int = 0,
    tool_schema_tokens: int = 0,
) -> PrefixShape:
    """Build a :class:`PrefixShape` snapshot for the current turn."""
    sys_h = compute_system_hash(system_prompt)
    tools_h = compute_tools_hash(tools)
    combined = _sha256_short(sys_h + tools_h)
    return PrefixShape(
        system_hash=sys_h,
        tools_hash=tools_h,
        combined_hash=combined,
        compaction_version=compaction_version,
        tool_schema_tokens=tool_schema_tokens,
        message_count=message_count,
    )


# ── Comparison ───────────────────────────────────────────────────

def compare_shapes(
    prev: Optional[PrefixShape],
    curr: PrefixShape,
    cache_read: int = 0,
    cache_write: int = 0,
) -> CacheDiagnostics:
    """Diagnose cache-miss reasons by comparing consecutive shapes.

    Returns a :class:`CacheDiagnostics` with human-readable miss reasons
    inspired by DeepSeek-Reasonix ``CompareShape()``.

    Possible miss reasons:
    * ``first_turn``        — no previous shape available
    * ``system_changed``    — system prompt hash differs
    * ``tools_changed``     — tool schema hash differs
    * ``log_rewrite``       — compaction / prune occurred
    * ``ttl_expired``       — cache_write > 0 with cache_read == 0
    """
    reasons: List[str] = []

    if prev is None:
        reasons.append("first_turn")
    else:
        if prev.system_hash != curr.system_hash:
            reasons.append("system_changed")
        if prev.tools_hash != curr.tools_hash:
            reasons.append("tools_changed")
        if prev.compaction_version != curr.compaction_version:
            reasons.append("log_rewrite")

    # TTL / general miss — content matched but provider cache expired
    if cache_write > 0 and cache_read == 0 and not reasons:
        reasons.append("ttl_expired")

    return CacheDiagnostics(
        hit=cache_read > 0,
        miss_reasons=reasons,
        prev_shape=prev,
        curr_shape=curr,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )


# ── Session-level aggregate tracker ─────────────────────────────

class CacheMetricsTracker:
    """Accumulates cache hit/miss tokens across a session.

    Mirrors DeepSeek-Reasonix ``agent.go`` ``sessCacheHit``/``sessCacheMiss``
    atomic accumulators — NOT reset on compaction, so the aggregate ratio
    is a steady cost-oriented metric.
    """

    def __init__(self) -> None:
        self.total_hit: int = 0
        self.total_miss: int = 0
        self._prev_shape: Optional[PrefixShape] = None
        self._compaction_version: int = 0
        self._turn_count: int = 0
        self._diagnostics: List[CacheDiagnostics] = []

    def on_compaction(self) -> None:
        """Bump compaction version (called after compression/prune)."""
        self._compaction_version += 1

    def record_turn(
        self,
        shape: PrefixShape,
        cache_read: int = 0,
        cache_write: int = 0,
    ) -> CacheDiagnostics:
        """Record one turn's cache outcome and return diagnostics."""
        diag = compare_shapes(self._prev_shape, shape, cache_read, cache_write)
        self.total_hit += cache_read
        self.total_miss += cache_write
        self._prev_shape = shape
        self._turn_count += 1
        self._diagnostics.append(diag)

        if not diag.hit and diag.miss_reasons:
            logger.info(
                "Cache miss [turn %d]: reasons=%s read=%d write=%d ratio=%.1f%%",
                self._turn_count,
                ",".join(diag.miss_reasons),
                cache_read,
                cache_write,
                diag.hit_ratio * 100,
            )
        return diag

    @property
    def aggregate_hit_ratio(self) -> float:
        """Session-wide cache hit ratio (not reset on compaction)."""
        total = self.total_hit + self.total_miss
        if total == 0:
            return 0.0
        return self.total_hit / total

    def tail_hit_ratio(self, last_n: int = 10) -> float:
        """Average hit ratio of the last *last_n* turns."""
        recent = self._diagnostics[-last_n:] if self._diagnostics else []
        if not recent:
            return 0.0
        return sum(d.hit_ratio for d in recent) / len(recent)

    @property
    def diagnostics_summary(self) -> Dict[str, Any]:
        """Summary dict for /cache command display."""
        from collections import Counter
        reason_counts: Counter = Counter()
        for d in self._diagnostics:
            for r in d.miss_reasons:
                reason_counts[r] += 1

        return {
            "turns": self._turn_count,
            "total_cache_read_tokens": self.total_hit,
            "total_cache_write_tokens": self.total_miss,
            "aggregate_hit_ratio": f"{self.aggregate_hit_ratio:.1%}",
            "compaction_version": self._compaction_version,
            "miss_reasons": dict(reason_counts),
        }


__all__ = [
    "PrefixShape",
    "CacheDiagnostics",
    "CacheMetricsTracker",
    "compute_prefix_shape",
    "compute_system_hash",
    "compute_tools_hash",
    "compare_shapes",
]
