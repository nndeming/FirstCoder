# 工具设计

[English](TOOLS_DESIGN.md)

## 问题与边界

工具让模型可以请求本地操作，但不会让 provider 代码直接拥有文件系统或 shell 权限。本层负责三件事：模型可见定义、本地执行器、可选权限元数据；最终 allow/ask/deny 的策略不属于这里。

## 端到端例子：读取 `view`

```text
create_builtin_registry(project_root)
  -> Tool(definition, executor, permission spec)
  -> create_session_tool_registry 注入 task_boundary/retrieve_archive
  -> PermissionAwareToolRegistry 包装 dispatch
  -> AgentLoop 筛选当前可用的工具声明，放进 ChatRequest.tools
  -> provider 返回 ToolCall(name="view", arguments=...)
  -> ToolExecutor 通过 session 预检权限，调度 registry 执行 -> ToolResult
  -> ToolExecutor 记录 role=tool 结果，AgentLoop 再请求模型
```

JSON Schema 经 `ChatRequest.tools` 发送，由 provider adapter 转成各家原生 `tools` 格式。它不会追加进 system prompt，既不会重复占对话 token，也绝不是安全边界。

## 核心契约

`ToolDefinition` 定义在 `providers/types.py`；`tools/types.py` 定义工具及结果的数据类：

| 类型 | 含义 |
| --- | --- |
| `ToolDefinition` | 模型看到的 name、description、JSON-Schema 风格 parameters |
| `Tool` | definition + 本地 executor + 可选 `ToolPermissionSpec` |
| `ToolResult` | 统一的 `name`、`ok`、`content`、`data`、`error` |
| `ToolPermissionSpec` | 如何从真实 arguments 推导权限请求 |

executor 返回 `ToolResult`；`ToolRegistry.execute` 还会把未知工具、参数错误和执行异常
转换成失败结果。因此模型能读取失败原因，主循环也能保存这次调用的结果。

### 两个同名的 ToolExecutor

`tools.types.ToolExecutor` 是函数协议，说明 `Tool.executor` 应接收参数并返回 `ToolResult`。
`agent.tool_execution.ToolExecutor` 是运行时调度器，负责整批工具的校验、权限暂停/恢复、
并行执行和结果记录。前者描述一个函数，后者安排这些函数何时执行。

从执行入口往下读时，按以下职责区分：

| 对象 | 负责什么 |
| --- | --- |
| `agent.tool_execution.ToolExecutor` | 决定执行、暂停、并行或转后台，并记录结果 |
| `PermissionAwareToolRegistry` | 根据工具声明构造权限请求，交给 `PermissionManager` 裁决 |
| `ToolRegistry` | 按名字找到工具，调用 `Tool.executor`，把异常转换为失败结果 |

## Registry 的创建与包装

`tools/builtin.py` 的 `create_builtin_registry` 组装 inspection、mutation、execution、network、git、interaction 等类别。`utils/introspection.py` 从函数签名得到基础 schema，随后覆盖为给模型阅读的 curated description，避免把 Python docstring 生硬扔过去。

原始 builtin 不是最终 registry。`create_session_tool_registry` 会注入 `task_boundary`、按需注入 `retrieve_archive`，有 manager 时再以 `PermissionAwareToolRegistry` 包一层。因为这些工具需要 session 状态，所以不能混进无状态全局 builtin。

装配期 `app.factory` 还可能通过 `McpToolProvider` 把 MCP 工具与 builtins 合并，再进入 session registry。对 loop 来说仍是同一张工具面，不是第二套 agent。

### 隐藏的控制面工具

`firstcoder.tools.hidden.HIDDEN_TOOL_STATUS_NAMES` 是唯一名单，列出既不应暴露给主模型、也不应刷进人机活动流的内部工具（当前是 `task_boundary`）。Session registry 保留它们供专用运行时控制器使用；`AgentLoop` 会过滤 Schema，并拒绝主模型幻觉返回的调用。不要在 UI 里散落工具名特判。

## 执行规则

`ToolRegistry.execute(name, arguments)` 查找并执行工具，同时规范化失败。
`PermissionAwareToolRegistry.preflight` 生成权限请求并获得 allow/ask/deny，预检本身不执行工具。
主循环经 `ToolExecutor` 调度时，`ASK` 会保存原调用并暂停，用户回答后再恢复。
直接调用权限包装的 `execute` 时，`ASK` 返回结构化确认信号，同样不会执行工具。

支持的直接修改到达 executor 前，`ToolExecutor` 还会通过 `tools.review.build_prewrite_review` 构造可信 diff 与文件快照。这是控制面行为，不是模型可见工具；覆盖 `write`、`edit`、`apply_patch`、`delete`，刻意排除无法安全预计算影响的 `shell`。恢复 Apply 时会再次校验快照，并始终执行本地保留的原始 `ToolCall`。

loop 必须保持 provider 对话的配对顺序：

```text
assistant(tool_call id=call_1) -> tool(tool_call_id=call_1)
```

被拒绝、跳过、失败也要有第二条消息。不要在 UI 里伪造 tool result，也不要在 context 压缩时删掉它。

## 特殊工具

- TaskPlan 有四个模型可见工具：`task_list` 读取权威快照和 revision；`task_create` 创建计划或追加任务；`task_update` 按稳定任务 ID 原子修改 status、owner 或依赖；`task_revise` 仅在任务语义内容变化时修改措辞。三个写工具都需要 `expected_revision`；
- `linear` 按稳定 order 推导执行顺序，`dag` 使用显式依赖。模型不会提交 ready/blocked 派生状态，也不能为汇报普通进度而替换整份计划；
- 成功的 TaskPlan 变更恰好追加一个 `task_plan_updated` 快照，因此 `SessionView.task_plan`、resume/fork 与 TUI 共用一份持久事实模型。发生 revision 冲突时，先 `task_list` 再用其 revision 重试；
- session schema 是严格兼容边界：resume/fork 会拒绝旧版本、缺少版本或未来版本的日志；不存在旧工具结果迁移或 fallback；
- `think` 记录结构化 reasoning，不改工作区；
- `task_boundary` 是隐藏分类器专用的内部控制工具；hash 由程序生成，主模型不能调用；
- `retrieve_archive` 仅能有界读取当前 session 的归档内容；
- `web_search` 使用托管的 Parallel MCP endpoint，配置后可回退 Exa；它是内建工具，不是用户 MCP 配置中的 server。

它们不是“顺手的小命令”，而是运行时参与者；改输出 schema 等于改变 context 和测试契约。

## 依赖规则

`tools`（以及 `permissions`、`utils`）可以为共享取消/用户输入类型 import `firstcoder.runtime`，但**不得** import `firstcoder.agent`。若 DTO 需要被 loop 上下层共用，放进 `runtime/`。

## 安全新增工具

1. 写小而真诚的 executor，返回 `ToolResult`。
2. 推导/校验 schema，并给出清晰 description。
3. 触碰本地或网络资源时注册 `ToolPermissionSpec`。
4. 放进正确 builtin 分类，别给 loop 加专属特判。
5. 覆盖成功、非法参数、denied/ask，必要时覆盖 provider 序列。

```sh
.venv/bin/python -m pytest tests/test_tools.py tests/test_schema.py \
  tests/test_introspection.py tests/test_execution_tools.py \
  tests/test_permission_registry.py tests/test_prewrite_review.py -q
```

## 排错地图

| 现象 | 先查 |
| --- | --- |
| 模型没看到工具 | builtin/session registry 或 provider capabilities |
| 参数 schema 不对 | introspection 或 curated definition |
| 没有预期确认就执行 | permission spec/wrapper/policy |
| 调用后 provider 拒绝历史 | tool result 缺失或 id 不匹配 |
| 单测能跑、真 session 不能 | session registry 装配 |

关联：[架构说明](ARCHITECTURE.zh-CN.md)、[权限设计](PERMISSIONS_DESIGN.zh-CN.md)、[Provider 设计](PROVIDERS_DESIGN.zh-CN.md)。
