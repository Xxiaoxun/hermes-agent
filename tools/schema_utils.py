"""Tool schema canonicalization — byte-stable schemas for cache efficiency.

Recursively normalises JSON Schema objects so that logically equivalent
schemas always produce the same byte sequence, regardless of the order
they were constructed in.

Design reference: DeepSeek-Reasonix ``internal/agent/schema_canonicalize.go``.

Key normalisations:
* ``required`` arrays sorted alphabetically.
* ``dependentRequired`` sub-arrays sorted alphabetically.
* ``properties`` keys sorted alphabetically (recursive).
* ``items`` normalised recursively.
* Empty schemas become ``{"type": "object"}``.

Pure functions — no I/O, no agent dependency.
"""

from __future__ import annotations

from typing import Any, Dict


def canonicalize_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively normalise a JSON Schema for byte-stable hashing.

    Returns a new dict; the input is never mutated.
    """
    if not isinstance(schema, dict):
        return schema

    result: Dict[str, Any] = {}

    for key, value in schema.items():
        if key == "required" and isinstance(value, list):
            result[key] = sorted(value)
        elif key == "dependentRequired" and isinstance(value, dict):
            result[key] = {
                k: sorted(v) if isinstance(v, list) else v
                for k, v in sorted(value.items())
            }
        elif key == "properties" and isinstance(value, dict):
            result[key] = {
                k: canonicalize_schema(v)
                for k, v in sorted(value.items())
            }
        elif key == "items" and isinstance(value, dict):
            result[key] = canonicalize_schema(value)
        elif key == "anyOf" and isinstance(value, list):
            result[key] = [canonicalize_schema(v) if isinstance(v, dict) else v for v in value]
        elif key == "oneOf" and isinstance(value, list):
            result[key] = [canonicalize_schema(v) if isinstance(v, dict) else v for v in value]
        elif key == "allOf" and isinstance(value, list):
            result[key] = [canonicalize_schema(v) if isinstance(v, dict) else v for v in value]
        else:
            result[key] = value

    # Empty schema → minimal valid schema
    if not result:
        return {"type": "object"}

    return result


def canonicalize_tool_schemas(tools: list) -> list:
    """Normalise a list of tool definitions for stable API requests.

    Each tool's ``function.parameters`` is canonicalised, and the list is
    sorted by tool name for deterministic ordering.
    """
    normalised = []
    for tool in tools:
        t = dict(tool)
        func = t.get("function")
        if isinstance(func, dict):
            func = dict(func)
            params = func.get("parameters")
            if isinstance(params, dict):
                func["parameters"] = canonicalize_schema(params)
            t["function"] = func
        normalised.append(t)

    # Sort by name for deterministic ordering
    normalised.sort(key=lambda x: x.get("name", x.get("function", {}).get("name", "")))
    return normalised


__all__ = [
    "canonicalize_schema",
    "canonicalize_tool_schemas",
]
