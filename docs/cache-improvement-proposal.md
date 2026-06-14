# Hermes 缓存命中率提升方案（v3 · 终版）

> **核心发现**: Anthropic 的缓存机制是**前缀匹配系统**（prefix-match），不是分段系统。
> 理论天花板不是 90%，而是 **~99%**——与 DeepSeek-Reasonix 相同。
> Hermes 当前缓存命中率低的根因不是 4 断点限制，而是**断点策略错误 + 历史消息被篡改**。
>
> 作者：Reasonix · 2026-06-14 · v3

---

## 一、范式转变：之前理解错了什么

### v2 的错误假设

v2 文档认为：
> "Anthropic 使用显式 `cache_control` 断点，最多 4 个，仅被标记的区域可以被缓存。理论天花板 ~90%。"

**这是错的。** 这个理解把 Anthropic 的缓存当成了"分段缓存"（segment-based），实际上它是**前缀缓存**（prefix-based）。

### 正确的缓存模型

**Anthropic 的每个断点标记的是"从请求开头到此处的前缀"**，不是"两个断点之间的片段"。

```
请求结构:  [tools] → [system] → [msg0] → [msg1] → ... → [msgN]
                                                           ↑
断点放在 msgN 上 → 缓存的不是 msgN 本身，而是从 tools 到 msgN 的整个前缀
```

多个断点创建的是**嵌套前缀**（nested prefixes），不是独立片段：

```
断点 1 (system)       → 缓存前缀: [tools + system]
断点 2 (msgN)         → 缓存前缀: [tools + system + msg0..msgN]

断点 2 是断点 1 的超集。两个前缀独立缓存。
```

### DeepSeek-Reasonix 的证明

Reasonix 的 Anthropic 适配器**只用 2 个断点**（`anthropic.go:255-269`）：

```go
// Max 4 breakpoints; we use ≤2.
system[n-1].CacheControl = ephemeral()          // 断点 1: system 最后一个 block
msgs[n-1].Content[k-1].CacheControl = ephemeral() // 断点 2: 最后一条消息的最后 block
```

且 Reasonix 在 Anthropic 上也能实现高缓存命中率。**4 个断点全用反而浪费钱**——多余的断点是冗余前缀，增加 `cache_creation_input_tokens` 开销而无额外覆盖收益。

### 99% 是怎么来的

**前缀匹配 + append-only 历史 = 近乎完美的缓存命中**：

```
Turn N:   [sys] [m0] [m1] ... [mN-1] [mN]
          └──────────────────────────────┘
          断点 2 缓存此前缀

Turn N+1: [sys] [m0] [m1] ... [mN-1] [mN] [mN+1]  ← 新增
          └──────────────────────────────────┘
          前缀匹配! sys→mN 全部命中 (0.1x)
          只有 mN+1 是 miss (1.25x)

命中率 = N / (N+1)
N=10  → 91%
N=50  → 98%
N=100 → 99%
```

**这就是 DeepSeek-Reasonix 99% 的全部秘密。不是什么高级算法，是正确的断点策略 + append-only 历史。**

---

## 二、Hermes 当前的真正问题

Hermes 的缓存命中率低，**不是因为 4 断点限制不够用**，而是因为两个根本性错误：

### 错误 1: 断点放在了会变化的消息上

当前策略 `system_and_3`（`agent/prompt_caching.py:56-78`）：

```
断点 1: system          ← ✓ 稳定（如果 ephemeral_system_prompt 为空）
断点 2: msg[N-2]        ← ✓ 稳定
断点 3: msg[N-1]        ← ✓ 稳定
断点 4: msg[N] (最后)   ← ✗ 每轮被 Compose 注入不同内容 → 永远 miss
```

**问题**：断点 4 放在最后一条消息上，而这条消息每轮都被 `pre_llm_call` 注入和 memory prefetch 修改。前缀匹配要求从头到断点的字节完全相同——最后一条消息变化 → 前缀断裂 → 断点 4 永远 miss。

更严重的是，由于前缀匹配是**从头开始逐字节比较**的，断点 4 的 miss 本身不是最大问题——问题是**断点 2 和 3 的覆盖范围太小**（只覆盖了最近 2-3 条消息），大量的中间历史消息没有被任何前缀覆盖。

### 错误 2: 系统提示的 volatile 层导致前缀不稳定

`agent/system_prompt.py:340-378` 的 volatile 层包含：
- Memory snapshot — 可能在会话中变化
- External memory provider block — 每个 session 不同
- Model/Provider 信息

更严重的是，`ephemeral_system_prompt`（`conversation_loop.py:736-737`）在每轮拼接到系统消息中，导致**系统消息每轮都可能不同** → 断点 1 也可能 miss。

**当断点 1（system）和断点 4（last msg）都 miss 时，只剩下 2 个有效断点覆盖极少的历史消息 → 命中率 ~60-70%。**

### 对比 Reasonix 的做法

| 维度 | Reasonix | Hermes 当前 |
|------|---------|------------|
| 断点策略 | 2 个：system + last msg | 4 个：system + last 3 msgs |
| 系统提示 | 启动时一次组装，永不修改 | volatile 层 + ephemeral_system_prompt 每轮变异 |
| 历史消息 | append-only，永不修改 | pre_llm_call 注入最后一条用户消息 → 被修改 |
| 断点有效性 | 2/2 = 100% | 1-2/4 = 25-50% |
| 缓存命中率 | ~99% | ~60-70% |

---

## 三、终极方案：达到 99% 的三步架构

### 第一步：冻结系统提示（消除系统前缀变异）

**目标**: 断点 1（system）100% 有效

**方案**: 移除 volatile 层中所有可能变化的内容，通过 Turn Composer 注入到会话尾部。

```python
# agent/system_prompt.py — 修改 build_system_prompt_parts()

def build_system_prompt_parts(agent, system_message, ...):
    # ── STABLE tier (不变) ──
    stable = _build_stable_tier(agent)  # identity, tool guidance, skills, etc.
    
    # ── CONTEXT tier (不变) ──
    context = _build_context_tier(system_message, context_files)
    
    # ── 原 VOLATILE tier 中的内容 → 全部移除或冻结 ──
    # Memory snapshot: 在 session 启动时加载一次，之后冻结
    # External memory provider: 同上，启动时加载一次
    # Date timestamp: 保持（日期级别足够，PR #20451）
    # Model/Provider: 保持（session 内不变）
    # Session ID: 保持（session 内不变）
    
    return "\n\n".join(filter(None, [stable, context, volatile_frozen]))
```

**关键变更**:
1. **Memory snapshot** 在 session 启动时加载一次并冻结。会话期间的 memory 写入通过 `pendingMemory` 队列排队，由 Turn Composer 注入用户消息。
2. **External memory provider block** 同上——启动时构建，session 内不变。
3. **`ephemeral_system_prompt` 不再拼接到系统消息**——通过 Turn Composer 注入到会话尾部。

**效果**: 系统提示在 session 全程字节稳定 → 断点 1 始终命中。

### 第二步：注入模式重构（消除历史消息变异）

**目标**: 断点 2（last msg）100% 有效

**核心问题**: 当前 `Compose()` 将 ephemeral 数据注入到最后一条用户消息，导致该消息每轮变化。如果断点放在最后一条消息上，断点永远 miss。

**解决方案**: 将 ephemeral 内容作为一个**独立的尾部消息**追加，断点放在倒数第二条（稳定的历史消息）上。

```python
# agent/turn_composer.py (新文件)

class TurnComposer:
    """将瞬态数据作为独立消息追加到会话尾部。
    
    设计原则:
    - 历史消息永不修改（append-only）
    - 瞬态数据追加为独立的 user 消息
    - 断点放在最后一条稳定的历史消息上（不是瞬态消息）
    """
    
    def __init__(self):
        self._pending_memory: list[str] = []
        self._pending_ephemeral: list[str] = []
        self._active_goal: str | None = None
    
    def queue_memory(self, note: str):
        """中途写入的内存笔记。"""
        self._pending_memory.append(note)
    
    def set_ephemeral_system(self, text: str):
        """替代 ephemeral_system_prompt。"""
        self._pending_ephemeral.append(text)
    
    def build_tail_messages(self) -> list[dict]:
        """构建要追加的瞬态消息列表。
        
        返回的消息追加到历史消息之后，不修改任何已有消息。
        """
        blocks = []
        
        if self._active_goal:
            blocks.append(f"<active-goal>\n{self._active_goal}\n</active-goal>")
        
        if self._pending_memory:
            lines = "\n".join(f"- {n}" for n in self._pending_memory)
            blocks.append(f"<memory-update>\n{lines}\n</memory-update>")
            self._pending_memory.clear()
        
        if self._pending_ephemeral:
            blocks.append("\n\n".join(self._pending_ephemeral))
            self._pending_ephemeral.clear()
        
        if not blocks:
            return []
        
        # 作为独立的 user 消息追加
        return [{"role": "user", "content": "\n\n".join(blocks)}]
```

**消息结构变化**:

```
之前 (system_and_3):
  [system + ephemeral] [m0] [m1] ... [mN-1] [mN + injection]  ← mN 被修改
  断点:  system  ---     ---   ---     ---     mN (永远 miss)

之后:
  [system (冻结)] [m0] [m1] ... [mN] [user: ephemeral]  ← mN 不变
  断点:  system  ---   ---   ---    mN (稳定!)     无断点
```

**关键**: 断点放在 `mN`（最后一条稳定的历史消息）上，而不是 `ephemeral` 消息上。`ephemeral` 消息没有断点，每轮变化 → miss，但它的 token 占比极小（通常 <500 tokens）。

### 第三步：2-断点策略（替代 system_and_3）

**目标**: 最大化前缀覆盖

```python
# agent/prompt_caching.py — 新策略

def prefix_match_strategy(messages: list[dict], cache_ttl: str = "5m",
                           native_anthropic: bool = True) -> list[dict]:
    """前缀匹配优化的 2-断点策略。
    
    参考: DeepSeek-Reasonix internal/provider/anthropic/anthropic.go:255-269
    
    断点 1: system message (1h TTL — 最稳定的内容)
    断点 2: 最后一条稳定消息 (5m TTL)
    
    "稳定消息" = 不被 Turn Composer 注入修改的消息。
    如果最后一条消息是 Composer 注入的 ephemeral 消息，断点放在倒数第二条。
    """
    result = copy.deepcopy(messages)
    breakpoints_used = 0
    
    # 断点 1: system message — 1h TTL
    system_marker = {"type": "ephemeral", "ttl": "1h"}
    for msg in result:
        if msg.get("role") == "system":
            _apply_cache_marker(msg, system_marker, native_anthropic)
            breakpoints_used += 1
            break
    
    # 断点 2: 最后一条稳定消息 — 5m TTL
    marker = {"type": "ephemeral"}
    non_system = [m for m in result if m.get("role") != "system"]
    if non_system:
        # 从后往前找第一条非 ephemeral 注入的消息
        for msg in reversed(non_system):
            if not _is_ephemeral_injection(msg):
                _apply_cache_marker(msg, marker, native_anthropic)
                breakpoints_used += 1
                break
    
    return result

def _is_ephemeral_injection(msg: dict) -> bool:
    """判断消息是否为 Turn Composer 注入的 ephemeral 内容。"""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content.startswith("<memory-update>") or \
               content.startswith("<active-goal>") or \
               content.startswith("<background-jobs>")
    return False
```

**为什么只用 2 个断点**:

```
断点 1 (system, 1h)     → 缓存前缀: [tools + system]
断点 2 (mN, 5m)         → 缓存前缀: [tools + system + m0...mN]

Turn N+1:
  前缀匹配: tools ✓ → system ✓ → m0 ✓ → ... → mN ✓
  全部命中! 只有 [ephemeral_msg] + [assistant_response] 是 miss.

命中率 = (tools + system + m0...mN) / (tools + system + m0...mN + ephemeral + response)
       ≈ 99% (ephemeral 消息通常 <500 tokens)
```

**为什么不需要 4 个断点**: 在前缀匹配模型下，断点 2 已经覆盖了从头到 mN 的全部内容。断点 3 和 4 是冗余的——它们标记的前缀被断点 2 的前缀所包含（是子集）。多余的断点只增加 `cache_creation_input_tokens` 开销。

---

## 四、与 v2 方案的对比

| 方案 | v2 预估 | v3 预估 | 原因 |
|------|--------|--------|------|
| Anthropic 路径 | ~85-88% | **~99%** | 前缀匹配 + 2 断点 + append-only |
| DeepSeek 路径 | ~95-97% | **~99%** | 同上（自动前缀缓存天然匹配） |
| OpenRouter + Claude | ~80% | **~95%** | content-part markers 的前缀行为 |

**差距消除原因**: v2 基于"分段缓存"的错误假设，认为 4 断点只能覆盖 4 个有限区域。v3 基于"前缀缓存"的正确理解，认识到 2 个嵌套前缀即可覆盖整个会话。

---

## 五、完整实施计划

### Phase 0: 诊断先行（1 周）

**Prefix Shape 诊断系统** — 与 v2 相同，但增加一个关键指标：

```python
@dataclass
class CacheDiagnostics:
    hit: bool
    miss_reasons: list[str]
    effective_breakpoints: int    # 有效断点数量（系统 + 稳定历史）
    prefix_coverage: float        # 前缀覆盖比例 (prefix_tokens / total_tokens)
    ephemeral_overhead: int       # ephemeral 注入的 token 开销
```

`prefix_coverage` 是核心指标：它直接告诉你"多少 token 被前缀缓存覆盖"。

### Phase 1: 冻结系统提示 + Turn Composer（2-3 周）

这是**最核心的改动**，分三个子步骤：

#### 1a. 创建 Turn Composer

```python
# agent/turn_composer.py (新文件)
# 如上第三节所述
```

#### 1b. 冻结系统提示的 volatile 层

```python
# agent/system_prompt.py

# 修改前:
def _build_volatile_tier(agent, ...):
    memory_snapshot = _load_memory_snapshot()  # 可能每 session 不同
    external_memory = _load_external_memory()   # 每 session 不同
    ...

# 修改后:
def _build_volatile_tier(agent, ...):
    # Memory snapshot: 启动时加载一次，之后由 Turn Composer 管理变更
    memory_snapshot = agent._frozen_memory_snapshot  # 启动时缓存
    # External memory: 同上
    external_memory = agent._frozen_external_memory
    # Timestamp: 保持日期级别（已是稳定的）
    ...
```

#### 1c. 消除 ephemeral_system_prompt 对系统消息的污染

```python
# agent/conversation_loop.py — 修改前:
effective_system = active_system_prompt or ""
if agent.ephemeral_system_prompt:
    effective_system = (effective_system + "\n\n" + agent.ephemeral_system_prompt).strip()
api_messages = [{"role": "system", "content": effective_system}] + api_messages

# 修改后:
api_messages = [{"role": "system", "content": active_system_prompt or ""}] + api_messages
# ephemeral_system_prompt 由 Turn Composer 作为尾部消息注入
tail_messages = agent.turn_composer.build_tail_messages()
if tail_messages:
    api_messages = api_messages + tail_messages
```

### Phase 2: 2-断点策略（1 周）

```python
# agent/prompt_caching.py

def prefix_match_strategy(messages, cache_ttl="5m", native_anthropic=True):
    """替代 system_and_3 的 2-断点前缀匹配策略。"""
    # 断点 1: system (1h TTL)
    # 断点 2: 最后一条稳定消息 (5m TTL)
    ...
```

**配置化**:

```yaml
# config.yaml
prompt_caching:
  strategy: "prefix_match"    # 新增: "system_and_3" | "prefix_match"
  system_ttl: "1h"            # 系统提示断点 TTL
  conversation_ttl: "5m"      # 对话断点 TTL
```

**向后兼容**: `strategy: "system_and_3"` 保留为默认值，用户可切换到 `"prefix_match"`。

### Phase 3: 结构保留式压缩（2-3 周）

与 v2 方案 B 相同，但适配新的前缀匹配模型：

```
压缩前: [sys] [m0] ... [summary_old] [mK] ... [mN] [ephemeral]
                          ↑ 断点 2
压缩后: [sys] [m0] ... [summary_old] [summary_new] [mN-2] [mN-1] [mN] [ephemeral]
                                      ↑ 新断点 2
```

压缩后，断点 2 移到 `mN`（尾部保护区内最后一条稳定消息），前缀从 `summary_new` 开始匹配。压缩边界处有一次 cache miss（summary_new 是新内容），之后恢复稳态。

**关键改进**: 压缩时**不重载 memory**（因为系统提示已冻结），避免 v2 中"压缩后系统提示变化"的问题。

### Phase 4: 冷启动优化 + Tool Schema 规范化（1-2 周）

与 v2 方案 E 和 F 相同。

---

## 六、消息结构对比（完整示例）

### 50 轮会话的第 51 轮

**当前 Hermes (`system_and_3`)**:
```
[system + ephemeral]          ← 断点 1: 可能 miss（ephemeral 变化）
[m0] [m1] ... [m46]          ← 无断点覆盖，全价
[m47]                         ← 断点 2: ✓
[m48]                         ← 断点 3: ✓
[m49 + injection]             ← 断点 4: ✗（injection 导致 miss）
[m50]                         ← 新 assistant response

断点覆盖:  system + m47 + m48 = ~3/51 消息 ≈ 6%
实际命中:  ~60-70%（加上 TTL 内的累积）
```

**改进后 (`prefix_match` + Turn Composer)**:
```
[system (冻结)]               ← 断点 1 (1h): ✓ 始终命中
[m0] [m1] ... [m49]          ← 被断点 2 的前缀覆盖
[m49]                         ← 断点 2 (5m): ✓ 前缀 = system + m0...m49
[ephemeral (Compose)]         ← 无断点，~300 tokens miss
[assistant response]          ← 新内容

断点覆盖:  system + m0...m49 = ~50/52 消息 ≈ 96%
实际命中:  ~99%（ephemeral 消息极短）
```

---

## 七、成本分析

### Anthropic 定价回顾

| 类型 | 价格倍数 | 说明 |
|------|---------|------|
| 首次写入 (cache write) | 1.25× | 比普通输入贵 25% |
| 缓存读取 (cache read) | 0.1× | 比普通输入便宜 90% |
| 普通输入 | 1.0× | 基准价格 |

### 50 轮会话的成本对比

**当前 (`system_and_3`，假设 65% 命中率)**:

```
每轮 prompt tokens ≈ 10,000
- 缓存读取: 6,500 tokens × 0.1× = 650
- 缓存写入/未命中: 3,500 tokens × 1.25× = 4,375
- 每轮成本: 5,025 单位
- 50 轮总成本: 251,250 单位
```

**改进后 (`prefix_match`，假设 99% 命中率)**:

```
首轮: 10,000 tokens × 1.25× = 12,500 (全量写入)
后续每轮: - 缓存读取: 9,900 tokens × 0.1× = 990
          - 新内容: 100 tokens × 1.25× = 125
          - 每轮成本: 1,115 单位
- 50 轮总成本: 12,500 + 49 × 1,115 = 67,135 单位
```

**节省**: (251,250 - 67,135) / 251,250 = **73% 成本降低**

---

## 八、风险与缓解

### 风险 1: Ephemeral 消息导致角色交替破坏

**场景**: Turn Composer 追加 `user` 消息，如果前一条也是 `user`，破坏交替规则。

**缓解**: 在追加前检查最后一条消息的角色。如果是 `user`，将 ephemeral 内容合并到最后一条用户消息的末尾（用分隔符）。如果是 `assistant`，正常追加。

```python
def build_tail_messages(self, existing_messages: list[dict]) -> list[dict]:
    tail_content = self._compose_content()
    if not tail_content:
        return []
    
    last_role = existing_messages[-1]["role"] if existing_messages else None
    if last_role == "user":
        # 合并到最后一条用户消息，不创建新消息
        return [{"_merge_into_last": True, "content": tail_content}]
    else:
        # 正常追加
        return [{"role": "user", "content": tail_content}]
```

### 风险 2: Gateway 每轮创建新 AIAgent → 系统提示可能不同

**场景**: Gateway 的 `_restore_or_build_system_prompt()` 从 SQLite 恢复系统提示，但如果是新 session（无存储的提示），会重新构建。

**缓解**: 
- 已有 session：从 DB 恢复 → 字节相同 ✓
- 新 session：构建后写入 DB → 后续轮次从 DB 恢复 ✓
- **关键**: 冻结 volatile 层后，即使重新构建，结果也与 DB 中的相同（因为 memory 等已在启动时固定）

### 风险 3: 压缩后的前缀断裂

**场景**: 压缩重写消息历史 → 前缀缓存 miss。

**缓解**: 这是**唯一允许的缓存断裂点**（与 DeepSeek-Reasonix 的设计一致）。压缩后：
1. 新的摘要消息写入历史
2. 断点 2 移到新的最后一条稳定消息
3. 首轮 miss → 后续恢复稳态

**经济门控**: 目标区域 <400 tokens 跳过压缩（不值得 API 调用）。

### 风险 4: OpenRouter/OpenAI-wire 路径的行为差异

**场景**: OpenRouter 使用 OpenAI wire envelope，`cache_control` 放在 content parts 上而非消息级别。

**缓解**: `prefix_match` 策略在 OpenAI-wire 下的行为需要单独验证。`_apply_cache_marker` 已处理 content part 格式差异。Tool messages 在非 native Anthropic 上跳过标记（已有逻辑）。可能需要针对 OpenRouter 做 A/B 测试。

---

## 九、测试策略

### E2E 缓存命中率测试

```python
class MockAnthropicEndpoint:
    """模拟 Anthropic 的前缀缓存行为。
    
    追踪连续请求之间的字节级前缀匹配长度。
    参考: DeepSeek-Reasonix cachehit_e2e_test.go
    """
    
    def __init__(self):
        self._prev_request_bytes: bytes = b""
    
    def compute_cache_hit(self, request_bytes: bytes) -> tuple[int, int]:
        """返回 (hit_tokens, miss_tokens)。"""
        common = 0
        for a, b in zip(self._prev_request_bytes, request_bytes):
            if a == b:
                common += 1
            else:
                break
        self._prev_request_bytes = request_bytes
        # 粗略估算: 4 chars ≈ 1 token
        hit = common // 4
        miss = (len(request_bytes) - common) // 4
        return hit, miss

def test_prefix_match_99_percent():
    """验证 prefix_match 策略在 50 轮会话中达到 ≥98% 命中率。"""
    endpoint = MockAnthropicEndpoint()
    conversation = Conversation()
    
    hit_rates = []
    for turn in range(50):
        request = conversation.build_request(strategy="prefix_match")
        hit, miss = endpoint.compute_cache_hit(request.to_bytes())
        if turn > 0:  # 跳过首轮（无缓存）
            hit_rates.append(hit / (hit + miss))
    
    avg_tail = sum(hit_rates[-10:]) / 10  # 最后 10 轮平均
    assert avg_tail >= 0.98, f"Tail avg hit rate {avg_tail:.2%} < 98%"

def test_system_prompt_byte_stable():
    """验证系统提示在 50 轮中字节完全相同。"""
    agent = create_test_agent()
    prompts = []
    for _ in range(50):
        agent.simulate_turn()
        prompts.append(agent.get_system_prompt_bytes())
    
    for i in range(1, len(prompts)):
        assert prompts[i] == prompts[0], f"System prompt changed at turn {i}"

def test_ephemeral_doesnt_break_prefix():
    """验证 ephemeral 注入不影响历史消息前缀。"""
    ...

def test_compression_preserves_prefix_stability():
    """验证压缩后前缀在下一轮恢复稳定。"""
    ...
```

### 发布门控

参考 DeepSeek-Reasonix 的 `TestReleaseCacheHitGuard`：

```python
def test_release_cache_hit_guard():
    """发布门控：标准场景尾部平均 <95% 则失败。"""
    scenarios = [
        "plain", "tool_loop", "long_conversation",
        "mixed", "with_compression", "with_ephemeral",
    ]
    for scenario in scenarios:
        rate = run_scenario(scenario, turns=30)
        tail_avg = rate.tail_average(last_n=10)
        assert tail_avg >= 0.95, \
            f"Scenario '{scenario}': tail avg {tail_avg:.2%} < 95%"
```

---

## 十、实施优先级

```
Phase 0 — 诊断（1 周）
  └── Prefix Shape 诊断系统 + /cache 命令 + 发布门控测试

Phase 1 — 冻结 + 注入（2-3 周）     ← 核心改动，贡献 ~90% 的命中率提升
  ├── 1a. Turn Composer (agent/turn_composer.py)
  ├── 1b. 冻结系统提示 volatile 层 (agent/system_prompt.py)
  └── 1c. 消除 ephemeral_system_prompt 污染 (agent/conversation_loop.py)

Phase 2 — 断点策略（1 周）
  └── 2-断点 prefix_match 策略 (agent/prompt_caching.py)

Phase 3 — 压缩优化（2-3 周）
  └── 结构保留式压缩 + 经济门控 + 连续压缩防护

Phase 4 — 附带优化（1-2 周）
  ├── 冷启动裁剪
  └── Tool Schema 规范化
```

**Phase 1 + Phase 2 合计 3-4 周，可将 Anthropic 路径命中率从 ~65% 提升到 ~99%。** 这是投入产出比最高的改动。

---

## 十一、总结

### 为什么之前的方案估错了天花板

v2 文档将 Anthropic 的缓存理解为"分段系统"——4 个断点覆盖 4 个有限区域。实际上它是**前缀系统**——2 个嵌套断点即可覆盖整个会话历史。

### 99% 的三个支柱

1. **冻结系统提示** — 系统消息字节稳定 → 断点 1（system）始终命中
2. **Append-only 历史 + Turn Composer** — 历史消息永不修改，瞬态数据追加为尾部消息 → 断点 2（last stable msg）始终命中
3. **2-断点 prefix_match 策略** — 前缀匹配模型下，2 个嵌套前缀覆盖整个会话 → 命中率 = (N-1)/N ≈ 99%

### 关键洞察

> **Anthropic 的 4 断点限制不是瓶颈。**
> **真正的瓶颈是：系统提示是否字节稳定 × 历史消息是否只追加不修改。**
> 
> 满足这两个条件，2 个断点即可 99%。
> 不满足，4 个断点也只有 60-70%。

DeepSeek-Reasonix 的 99% 不是 DeepSeek 的专利——它是正确架构的必然结果。
