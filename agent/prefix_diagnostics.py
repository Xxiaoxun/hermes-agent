"""Prefix shape diagnostics for cache hit optimization.

Hashes the cacheable portions of each API request (system prompt + tool schemas)
and compares them turn-over-turn to explain cache misses.

Design reference: DeepSeek-Reasonix internal/agent/cache_shape.go
"""
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PrefixShape:
    """Snapshot of the cacheable prefix state."""
    system_hash: str
    tools_hash: str
    prefix_hash: str
    tool_schema_tokens: int


def short_hash(obj: Any) -> str:
    """SHA-256 of JSON-serialized object, truncated to 16 hex chars."""
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def normalize_tool_schemas(schemas: List[Dict]) -> List[Dict]:
    """Normalize and sort tool schemas for deterministic prefix matching.

    Uses canonicalize_tool_schemas for recursive property normalization,
    matching the same function used in the API path.
    """
    try:
        from tools.schema_utils import canonicalize_tool_schemas
        return canonicalize_tool_schemas(schemas)
    except ImportError:
        return sorted(
            schemas,
            key=lambda s: (
                s.get("function", {}).get("name", ""),
                s.get("function", {}).get("description", ""),
                json.dumps(s.get("function", {}).get("parameters", {}), sort_keys=True),
            ),
        )



    Ensures MCP server reconnect order doesn't break the cache prefix.
    Reference: DeepSeek-Reasonix normalizeToolSchemas() sorts by
    (Name, Description, Parameters) lexicographic order.
    """
    return sorted(
        schemas,
        key=lambda s: (
            s.get("function", {}).get("name", ""),
            s.get("function", {}).get("description", ""),
            json.dumps(
                s.get("function", {}).get("parameters", {}),
                sort_keys=True
            ),
        ),
    )


def capture_shape(system_prompt: str, tool_schemas: List[Dict]) -> PrefixShape:
    """Snapshot the current cacheable prefix state.

    Args:
        system_prompt: The full system prompt text.
        tool_schemas: List of tool schema dicts (OpenAI format).

    Returns:
        PrefixShape with hashes of system prompt and normalized tools.
    """
    normalized = normalize_tool_schemas(tool_schemas)
    tools_json = json.dumps(normalized, sort_keys=True)
    return PrefixShape(
        system_hash=short_hash(system_prompt),
        tools_hash=short_hash(tools_json),
        prefix_hash=short_hash({"system": system_prompt, "tools": tools_json}),
        tool_schema_tokens=len(tools_json) // 4,
    )


def compare_shape(
    prev: Optional[PrefixShape],
    cur: PrefixShape,
) -> Dict[str, Any]:
    """Describe what changed between two prefix shapes.

    Three possible change reasons (matching DeepSeek-Reasonix):
    - 'system': system prompt content changed (soul update, context change)
    - 'tools': tool schema changed (MCP connect/disconnect, tool enable/disable)

    Returns dict with prefix_changed, reasons, and diagnostic hashes.
    """
    reasons = []
    if prev and prev.system_hash != cur.system_hash:
        reasons.append("system")
    if prev and prev.tools_hash != cur.tools_hash:
        reasons.append("tools")

    if reasons:
        logger.info(
            "Cache prefix changed: %s (system=%s tools=%s tool_tokens=%d)",
            "+".join(reasons),
            cur.system_hash[:8],
            cur.tools_hash[:8],
            cur.tool_schema_tokens,
        )

    return {
        "prefix_changed": len(reasons) > 0,
        "reasons": reasons,
        "system_hash": cur.system_hash,
        "tools_hash": cur.tools_hash,
        "tool_schema_tokens": cur.tool_schema_tokens,
    }
