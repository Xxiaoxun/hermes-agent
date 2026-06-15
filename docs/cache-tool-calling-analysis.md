# Deep-Dive: Tool Calling Patterns & Cache Impact — DeepSeek-Reasonix vs Hermes Agent

> Generated: 2026-06-15 | Source: DeepSeek-Reasonix `internal/agent/agent.go`, `internal/agent/cache_shape.go`, `internal/tool/tool.go`, `internal/provider/schema_canonicalize.go` | Hermes `agent/conversation_loop.py`, `agent/tool_executor.py`, `agent/prompt_caching.py`, `agent/prefix_diagnostics.py`, `agent/chat_completion_helpers.py`, `tools/tool_output_limits.py`, `tools/tool_result_storage.py`, `tools/registry.py`, `tools/schema_utils.py`, `tools/mcp_tool.py`

---

## 1. Tool Result Size Management

### DeepSeek-Reasonix

**Hard 32 KB cap with head+tail preservation** — applied in `executeOne()` before the result ever enters the session:

```go
// agent.go:29
const maxToolOutputBytes = 32 * 1024

// agent.go:1575-1586
func truncateToolOutput(s string) (string, string) {
    if len(s) <= maxToolOutputBytes {
        return s, ""
    }
    keep := maxToolOutputBytes / 2
    head := snapToRuneBoundary(s, 0, keep)
    tail := snapToRuneBoundary(s, len(s)-keep, len(s))
    omitted := len(s) - len(head) - len(tail)
    notice := fmt.Sprintf("tool output truncated: %d of %d bytes elided", omitted, len(s))
    body := head + fmt.Sprintf("\n\n…[truncated %d of %d bytes — rerun with narrower args to see the middle]…\n\n", omitted, len(s)) + tail
    return body, notice
}
```

**Applied at two points in `executeOne()` (agent.go:1394-1414):**
- Error path: `truncateToolOutput(fmt.Sprintf("error: %v\n%s", err, detail))`
- Success path: `truncateToolOutput(result)`

**Key property:** The 32 KB cap runs *inside* the tool execution function, **before** the result is appended to `session.Messages`. A 5 MB tool output never touches the context window.

### Hermes Agent

Hermes uses a **three-layer defense** — but none are as aggressive as Reasonix's 32 KB hard cap:

#### Layer 1: Per-tool self-truncation (`tools/tool_output_limits.py`)
```python
# tool_output_limits.py:39-41
DEFAULT_MAX_BYTES = 50_000       # terminal_tool cap (~50 KB, 1.5× Reasonix)
DEFAULT_MAX_LINES = 2000         # read_file pagination cap
DEFAULT_MAX_LINE_LENGTH = 2000   # per-line length cap
```

Individual tools (terminal, read_file) truncate their own output. Configurable via `config.yaml`.

#### Layer 2: Per-result persistence (`tools/tool_result_storage.py`)
```python
# tool_result_storage.py:122-178
def maybe_persist_tool_result(content, tool_name, tool_use_id, env=None, ...):
    # If content > threshold: write full output to /tmp/hermes-results/{id}.txt
    # Replace in-context content with <persisted-output> preview + file path
    # Model can read_file the full output later
```

Instead of truncating, Hermes writes large results to disk and replaces them with a preview + reference. The model sees ~500 chars of preview + a file path.

#### Layer 3: Per-turn aggregate budget (`tool_result_storage.py:181-232`)
```python
# tool_result_storage.py
MAX_TURN_BUDGET_CHARS = 200_000  # 200 KB aggregate per turn
```

If all tool results in one turn exceed 200 KB combined, the largest non-persisted results are spilled to disk.

#### Layer 4: Phase 1 pre-compression pruning (`context_compressor.py`)
```python
# context_compressor.py:149-150
_PRUNED_TOOL_PLACEHOLDER = "[Old tool output cleared to save context space]"

# context_compressor.py:850-950 — _prune_old_tool_outputs()
# - Deduplicates identical tool results (keeps newest, replaces older with back-ref)
# - Replaces old tool results with informative summaries:
#     "[terminal] ran `npm test` -> exit 0, 47 lines output"
#     "[read_file] read config.py from line 1 (1,200 chars)"
# - Truncates large tool_call arguments in assistant messages outside the protected tail
```

### Gap Analysis

| Dimension | DeepSeek-Reasonix | Hermes Agent | Gap |
|---|---|---|---|
| **Cap point** | Inside `executeOne()`, before session append | Per-tool (Layer 1), post-execution (Layer 2), post-turn (Layer 3) | Reasonix caps earlier; Hermes lets large results through to `messages` then retrofits |
| **Hard cap** | 32 KB per result | 50 KB per-tool (configurable), 200 KB per-turn | Hermes is 1.5–6× more permissive |
| **Preservation** | Head+tail (16 KB each) | Preview + disk file reference | Both preserve signal; Hermes is more sophisticated (disk persistence) |
| **Token waste** | ~8K tokens max per result | ~12.5K tokens per result (50 KB/4) | Reasonix wastes less context |
| **Context bloat risk** | Bounded at 32 KB | A single tool result can be 50 KB before Layer 2 fires | **GAP**: A 49 KB tool result stays inline in Hermes |

**Critical gap:** Hermes has no equivalent of Reasonix's 32 KB hard cap that runs *before* the result enters the messages list. If a tool returns 49 KB (under the 50 KB default), it goes straight into context. Only `maybe_persist_tool_result` catches it, and that requires a working sandbox environment.

---

## 2. Tool Call Retry Patterns

### DeepSeek-Reasonix

**No automatic retry of failed tool calls.** The model sees the error and decides what to do:

```go
// agent.go:1394-1404
if err != nil {
    detail := result
    if !json.Valid([]byte(call.Arguments)) {
        detail = strings.TrimRight(detail, "\n") + "\nThe arguments were not valid JSON..."
    }
    body, truncMsg := truncateToolOutput(fmt.Sprintf("error: %v\n%s", err, detail))
    return toolOutcome{output: body, errMsg: firstLine(err.Error()), ...}
}
```

**Storm breaker** — detects repeated identical failures (agent.go:1198-1249):
```go
const stormBreakThreshold = 3

// Keys on (tool_name, error_message) — NOT args
// A stuck model reworks args cosmetically while failing identically
func batchStormSignature(calls []provider.ToolCall, outcomes []toolOutcome) (string, bool) {
    // sig = "tool1\0error1\0tool2\0error2\0..."
}
```

After 3 identical failures, the result is rewritten with a `[loop guard]` directive telling the model to change approach.

**Stream recovery** (agent.go:571-588):
```go
if interrupted && streamRecoveries < maxStreamRecoveries {
    streamRecoveries++
    a.session.Add(provider.Message{Role: provider.RoleUser, Content: streamRecoveryMessage(...)})
    step-- // recovery retries do not consume the tool-round maxSteps budget
    continue
}
```

### Hermes Agent

**No automatic retry of failed tool calls either.** Failed results go back to the model as-is. But Hermes has extensive retry logic for *API-level* failures:

```python
# conversation_loop.py:1671-1702 — Truncated tool call retry
if truncated_tool_call_retries < 3:
    truncated_tool_call_retries += 1
    # Boost max_tokens on each retry
    _tc_boost = _tc_boost_base * (truncated_tool_call_retries + 1)
    agent._ephemeral_max_output_tokens = min(_tc_boost, _tc_boost_cap)
    # Don't append the broken response to messages
    continue
```

```python
# conversation_loop.py:1509-1651 — finish_reason=length continuation
# Appends continuation prompt, retries up to 3 times:
_get_continuation_prompt(is_partial_stub, dropped_tools)
```

### Cache Impact

**DeepSeek-Reasonix:** A failed tool call + model retry appends the error as a `role=tool` message, then the model's new response. The prefix up to the error is stable — cache hit for the unchanged portion.

**Hermes:** API-level retries (truncated tool calls, stream interruptions) are handled by *not appending* the broken response to messages, then retrying the same API call. This preserves the cache prefix. But continuation prompts (`finish_reason=length`) add a new `role=user` message that shifts the prefix boundary.

**Key difference:** Reasonix's storm breaker is purely session-level (no cache impact). Hermes's continuation prompts are `role=user` messages that break the prefix cache for the next turn.

---

## 3. Tool Schema Field Ordering

### DeepSeek-Reasonix

**Double-sorting: at registry level AND at cache-shape level.**

Registry `Schemas()` sorts by name alphabetically (tool.go:196-217):
```go
func (r *Registry) Schemas() []provider.ToolSchema {
    names := make([]string, len(r.order))
    copy(names, r.order)
    sort.Strings(names)  // Alphabetical sort
    // ... build output in sorted order
}
```

Cache shape normalizer sorts by (Name, Description, Parameters) (cache_shape.go:51-63):
```go
func normalizeToolSchemas(schemas []provider.ToolSchema) []provider.ToolSchema {
    sort.Slice(out, func(i, j int) bool {
        if out[i].Name != out[j].Name { return out[i].Name < out[j].Name }
        if out[i].Description != out[j].Description { return out[i].Description < out[j].Description }
        return string(out[i].Parameters) < string(out[j].Parameters)
    })
}
```

**Schema canonicalization** (schema_canonicalize.go:10-28):
```go
func CanonicalizeSchema(raw json.RawMessage) json.RawMessage {
    // - required arrays sorted alphabetically
    // - dependentRequired sub-arrays sorted
    // - Empty schemas → {"type":"object"}
}
```

Applied once at `Add()` time, cached in `r.canon[name]`.

### Hermes Agent

**Sorting at two points:**

1. `registry.get_definitions()` sorts by tool name (registry.py:354):
```python
for name in sorted(tool_names):  # Alphabetical
```

2. `build_api_kwargs()` sorts by (name, description) at request time (chat_completion_helpers.py:562-569):
```python
tools_for_api = sorted(
    tools_for_api,
    key=lambda t: (
        (t.get("function", {}) or {}).get("name", ""),
        (t.get("function", {}) or {}).get("description", ""),
    ),
)
```

**Schema canonicalization** (`tools/schema_utils.py:24-62`):
```python
def canonicalize_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    # - required arrays sorted alphabetically
    # - dependentRequired sub-arrays sorted
    # - properties keys sorted alphabetically (recursive)
    # - items normalised recursively
    # - Empty schemas → {"type": "object"}
```

**Prefix diagnostics** (`agent/prefix_diagnostics.py:32-49`):
```python
def normalize_tool_schemas(schemas: List[Dict]) -> List[Dict]:
    return sorted(schemas, key=lambda s: (
        s.get("function", {}).get("name", ""),
        s.get("function", {}).get("description", ""),
        json.dumps(s.get("function", {}).get("parameters", {}), sort_keys=True),
    ))
```

### Gap Analysis

| Dimension | DeepSeek-Reasonix | Hermes Agent | Gap |
|---|---|---|---|
| **Sort key** | (Name, Description, Parameters) | (Name, Description) in API; (Name, Description, Parameters) in diagnostics | **GAP**: API request sorts by 2 keys, diagnostics by 3. If two tools share name+description but differ in parameters, API order is nondeterministic |
| **Schema canonicalization** | At `Add()` time, cached | Available but **NOT applied at API request time** | **GAP**: `canonicalize_tool_schemas()` exists but `build_api_kwargs()` doesn't call it |
| **Property order** | `properties` keys NOT sorted | `properties` keys sorted alphabetically | Hermes is better here |
| **`required` arrays** | Sorted | Sorted | Parity |

**Critical gap:** Hermes's `canonicalize_tool_schemas()` (which sorts `properties` keys and `required` arrays) is never called in the API request path. The `build_api_kwargs()` function only sorts tools by (name, description), but the parameter schemas themselves are sent as-is from registration. If an MCP server returns `properties` in different order on reconnect, the tool schema bytes change, breaking the cache prefix.

---

## 4. Tool Call ID Stability

### DeepSeek-Reasonix

Tool call IDs come from the **provider's streaming response**. The OpenAI provider synthesizes stable IDs when the gateway doesn't provide them (openai.go:500-504):

```go
if tc.ID == "" {
    // Some OpenAI-compatible gateways stream tool calls by index with no id.
    // Synthesize a stable one so the result can be paired back to its call
    tc.ID = fmt.Sprintf("call_%d", idx)
}
```

The ID is deterministic: `call_0`, `call_1`, etc. based on streaming order.

Tool results are paired by ID when IDs are distinct, by position when they're empty/duplicate (provider.go:213-239):
```go
func pairToolResults(calls []ToolCall, avail []Message) []Message {
    if idDistinct(calls) {
        byID := make(map[string]Message)
        // pair by ID
    }
    // fallback: pair by position
}
```

### Hermes Agent

Tool call IDs come from the **provider response** and are stored as-is:

```python
# run_agent.py:3246-3250
@staticmethod
def _get_tool_call_id_static(tc) -> str:
    if isinstance(tc, dict):
        return tc.get("call_id", "") or tc.get("id", "") or ""
    return getattr(tc, "call_id", "") or getattr(tc, "id", "") or ""
```

For Codex/Responses API, Hermes generates deterministic IDs when missing (chat_completion_helpers.py:1007):
```python
call_id = agent._deterministic_call_id(_fn_name, _fn_args, len(tool_calls))
```

### Cache Impact

Tool call IDs appear in:
1. `assistant.tool_calls[].id` — part of the assistant message
2. `tool.tool_call_id` — part of the tool result message

**Both systems use provider-issued IDs.** These are stable within a session (the model's tool_calls IDs are deterministic for the same prompt), so they don't break the cache prefix between turns. The IDs change when the model generates different tool calls, which is expected.

**No gap here** — both systems handle IDs the same way.

---

## 5. Parallel vs Sequential Tool Execution

### DeepSeek-Reasonix

**Smart batching with read-only parallelism** (agent.go:1085-1120):

```go
func (a *Agent) executeBatch(ctx context.Context, calls []provider.ToolCall) []string {
    for _, batch := range partitionToolCalls(a.tools, calls) {
        if batch.parallel && batch.end-batch.start > 1 {
            runParallel(batch.start, batch.end, run)  // up to 8 goroutines
            continue
        }
        for i := batch.start; i < batch.end; i++ {
            run(i)  // sequential
        }
    }
}
```

Partitioning logic (agent.go:1155-1171):
```go
func partitionToolCalls(r *tool.Registry, calls []provider.ToolCall) []toolCallBatch {
    // Contiguous known ReadOnly tools → parallel batch
    // Unknown or writer tools → single-call serial batch
    // complete_step/todo_write never join parallel (they read evidence ledger)
}
```

**Max parallelism: 8 goroutines** (agent.go:1182).

### Hermes Agent

**Two execution modes** (tool_executor.py):

1. **Concurrent** (`execute_tool_calls_concurrent`): ThreadPoolExecutor with up to 8 workers
2. **Sequential** (`execute_tool_calls_sequential`): One tool at a time

Gating logic (tool_dispatch_helpers.py:103-146):
```python
def _should_parallelize_tool_batch(tool_calls) -> bool:
    # - Single tool → sequential
    # - Interactive tools (clarify) → sequential
    # - Path-scoped tools (read_file, write_file, patch) → parallel if paths don't overlap
    # - Known safe tools → parallel
    # - MCP tools → parallel if server opted in
    # - Everything else → sequential
```

### Cache Impact

**Both systems preserve tool result ordering.** Results are appended to messages in the original call order regardless of execution parallelism. This means the cache prefix sees the same message sequence whether tools ran in parallel or not.

**No cache gap** — parallelism is an execution optimization that doesn't affect the message structure.

---

## 6. Tool Result Injection Pattern

### DeepSeek-Reasonix

Tool results are appended as `role=tool` messages in **call order** (agent.go:664-672):

```go
results := a.executeBatch(ctx, calls)
for i, call := range calls {
    a.session.Add(provider.Message{
        Role:       provider.RoleTool,
        Content:    results[i],
        ToolCallID: call.ID,
        Name:       call.Name,
    })
}
```

**Order is deterministic:** always matches the model's `tool_calls` array order.

### Hermes Agent

Tool results are appended as `role=tool` messages in **call order** (tool_executor.py:748):

```python
messages.append(make_tool_result_message(name, _tool_content, tc.id))
```

Where `make_tool_result_message` (tool_dispatch_helpers.py:320-343):
```python
def make_tool_result_message(name: str, content: Any, tool_call_id: str) -> dict:
    return {
        "role": "tool",
        "name": name,
        "tool_name": name,      # Internal field, stripped before API
        "content": wrapped,     # May be wrapped in <untrusted_tool_result> for web tools
        "tool_call_id": tool_call_id,
    }
```

**Key difference:** Hermes adds an extra `tool_name` field and wraps untrusted tool results in `<untrusted_tool_result>` delimiters. The `tool_name` field is stripped before the API call, but the wrapping changes the content bytes.

### Cache Impact

**Order is deterministic in both systems.** The order of tool results in the messages list matches the model's tool_calls order, so the prefix is stable.

**No cache gap** — both systems inject tool results identically (role=tool, in call order).

---

## 7. MCP Server Tool Hot-Reload

### DeepSeek-Reasonix

MCP tools are namespaced as `mcp__<server>__<tool>` (tool.go:129):
```go
const MCPNamePrefix = "mcp__"
```

On disconnect, all tools with the server's prefix are removed (tool.go:149-166):
```go
func (r *Registry) RemovePrefix(prefix string) int {
    // Removes from r.order and r.tools
}
```

On reconnect, tools are re-added in the order the server provides them. But `Schemas()` always sorts alphabetically, so the order is stable.

### Hermes Agent

MCP tool refresh (mcp_tool.py:1287-1353):
```python
async def _refresh_tools(self):
    old_tool_names = set(self._registered_tool_names)
    new_mcp_tools = (await self.session.list_tools()).tools

    # Remove stale tools
    stale_tool_names = old_tool_names - {new_names}
    for tool_name in stale_tool_names:
        registry.deregister(tool_name)

    # Re-register fresh tools
    self._registered_tool_names = _register_server_tools(self.name, self, self._config)
```

The registry increments `_generation` on every mutation (register/deregister), so caches downstream are invalidated.

### Cache Impact

**Both systems sort tool schemas alphabetically at request time**, so MCP reconnect order doesn't break the cache prefix. However:

**GAP in Hermes:** The `_generation` counter invalidates downstream caches (e.g., `_tool_search_scope_cache`), but the actual API request rebuilds tools from `agent.tools` which is set at session start. If an MCP server reconnects mid-session and changes the tool list, the next API call will have a different tool schema set — which is correct behavior, but the transition point causes a cache miss.

**GAP in Hermes:** The `canonicalize_tool_schemas()` function exists but is never called in the API path. If an MCP server reconnects and returns the same tools but with `properties` in different order, the tool schema bytes change, causing a spurious cache miss.

---

## 8. Streaming vs Non-Streaming Cache Impact

### DeepSeek-Reasonix

Always streams (provider.go:406):
```go
Stream(ctx context.Context, req Request) (<-chan Chunk, error)
```

Requests include `stream_options` (openai.go:547-560):
```go
type chatRequest struct {
    Stream        bool           `json:"stream"`
    StreamOptions *streamOptions `json:"stream_options,omitempty"`
}

type streamOptions struct {
    IncludeUsage bool `json:"include_usage"`
}
```

Cache stats are received via `ChunkUsage` events and accumulated per-session (agent.go:1031-1032):
```go
case provider.ChunkUsage:
    usage = chunk.Usage
    a.sessCacheHit.Add(int64(chunk.Usage.CacheHitTokens))
    a.sessCacheMiss.Add(int64(chunk.Usage.CacheMissTokens))
```

### Hermes Agent

Supports both streaming and non-streaming. Streaming always includes usage (chat_completion_helpers.py:1828):
```python
stream_kwargs = {
    **api_kwargs,
    "stream": True,
    "stream_options": {"include_usage": True},
}
```

Cache stats are tracked per-session (conversation_loop.py:1806-1809):
```python
agent._session_cache_hit_cumulative += canonical_usage.cache_read_tokens
_miss = max(0, (canonical_usage.prompt_tokens or 0) - (canonical_usage.cache_read_tokens or 0))
agent._session_cache_miss_cumulative += _miss
```

### Cache Impact

**Both systems request `include_usage: true` in streaming mode**, which is required for providers to return cache hit/miss stats in the final streaming chunk.

**Streaming vs non-streaming has NO impact on cache hit rates.** The cache is determined by the request content (system prompt, tool schemas, message history), not by whether the response is streamed. The `stream_options` field is part of the request but doesn't affect the prefix.

**No gap** — both systems handle streaming cache stats correctly.

---

## Summary of Gaps

### High-Impact Gaps (Worth Fixing)

1. **No hard pre-context cap in Hermes** — Reasonix caps at 32 KB before results enter context; Hermes's closest equivalent is 50 KB per-tool + 200 KB per-turn, applied after execution. A single 49 KB tool result stays inline.
   - **Fix:** Add a `maxToolOutputBytes` check in `execute_tool_calls_sequential`/`concurrent` before appending to messages. Or lower `DEFAULT_MAX_BYTES` to 32 KB.

2. **`canonicalize_tool_schemas()` not called in API path** — The function exists in `tools/schema_utils.py` but `build_api_kwargs()` only sorts by (name, description), not normalizing parameter schemas.
   - **Fix:** Call `canonicalize_tool_schemas(tools_for_api)` in `build_api_kwargs()` before sending.

3. **Sort key mismatch** — API sorts by 2 keys (name, description); diagnostics sort by 3 keys (name, description, parameters). Two tools with identical name+description could have nondeterministic order.
   - **Fix:** Use the same 3-key sort in `build_api_kwargs()`.

### Medium-Impact Gaps

4. **Continuation prompts break cache prefix** — `finish_reason=length` adds a `role=user` continuation message, shifting the prefix boundary. Reasonix handles this more gracefully (stream recovery reuses the same prefix).
   - **Fix:** Consider using `role=system` or a metadata field instead of `role=user` for continuation prompts.

5. **`tool_name` extra field in tool messages** — Hermes adds `tool_name` alongside `name` in tool result messages. While `tool_name` is stripped before the API call, any inconsistency in stripping could cause prefix drift.
   - **Fix:** Verify stripping is consistent across all transports.

### Low-Impact Gaps (Informational)

6. **Hermes wraps untrusted tool results** — `<untrusted_tool_result>` wrapping changes content bytes. This is intentional security behavior, not a bug, but it means identical web search results from different turns have different bytes.
   - **Note:** This is by design for prompt injection defense.

7. **Both systems lack `required` array normalization at request time** — Reasonix normalizes `required` arrays at `Add()` time; Hermes has the function but doesn't call it in the API path.
   - **Fix:** Covered by gap #2 above.

---

## Architecture Comparison

```
DeepSeek-Reasonix Tool Result Flow:
┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────┐
│  Tool     │───▶│truncateTool  │───▶│ session.Add │───▶│ Provider │
│ Execute   │    │Output (32KB) │    │ (role=tool) │    │ Request  │
└──────────┘    └──────────────┘    └─────────────┘    └──────────┘
                 ▲ Hard cap BEFORE entering context

Hermes Agent Tool Result Flow:
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│  Tool     │───▶│ Per-tool cap │───▶│ maybe_persist│───▶│enforce_turn  │───▶│ Provider │
│ Execute   │    │ (50KB deflt) │    │_tool_result  │    │_budget(200K) │    │ Request  │
└──────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────┘
                 ▲ Tool-level      ▲ Post-exec         ▲ Post-turn
                 │ (optional)      │ (needs sandbox)    │ (aggregate)
                 │                 │                    │
                 └── Results CAN reach context if under all thresholds ──┘
```

```
DeepSeek-Reasonix Schema Canonicalization:
┌──────────┐    ┌──────────────────┐    ┌─────────────┐
│ MCP/Tool │───▶│ Add() + Canon()  │───▶│ canon map   │
│ Register │    │ (once at startup)│    │ (cached)    │
└──────────┘    └──────────────────┘    └─────────────┘
                                               │
                                        ┌──────▼──────┐
                                        │ Schemas()   │
                                        │ sort by name│
                                        └─────────────┘

Hermes Agent Schema Canonicalization:
┌──────────┐    ┌──────────────────┐    ┌─────────────┐
│ MCP/Tool │───▶│ register()       │───▶│ _tools dict │
│ Register │    │ (raw schema)     │    │ (uncanon)   │
└──────────┘    └──────────────────┘    └─────────────┘
                                               │
                                        ┌──────▼──────┐
                                        │build_api_   │
                                        │kwargs()     │
                                        │sort(name,   │
                                        │ description)│
                                        └─────────────┘
                                        ⚠️ No canonicalize!
```
