# Tools Design

[中文版本](TOOLS_DESIGN.zh-CN.md)

## Problem and Non-Goals

Tools let a model request local actions without giving provider code direct
filesystem or shell access. This layer owns three things: a model-visible
definition, a local executor, and optional permission metadata. It does not
decide policy; `PermissionManager` combines policy and remembered grants.

## End-to-End Example: `view` a File

```text
create_builtin_registry(project_root)
  -> Tool(definition, executor, permission spec)
  -> create_session_tool_registry injects task_boundary/retrieve_archive
  -> PermissionAwareToolRegistry wraps dispatch
  -> AgentLoop filters currently available definitions into ChatRequest.tools
  -> provider returns ToolCall(name="view", arguments=...)
  -> ToolExecutor preflights through the session and dispatches the registry -> ToolResult
  -> ToolExecutor records role=tool; AgentLoop asks the provider again
```

The JSON Schema travels as `ChatRequest.tools`; provider adapters turn it into
their native `tools` representation. The schema is not appended to the system
prompt, so it is neither duplicated conversation text nor a security boundary.

## Core Contract

`ToolDefinition` is defined in `providers/types.py`; `tools/types.py` defines
the tool and result dataclasses:

| Type | Meaning |
| --- | --- |
| `ToolDefinition` | name, description, JSON-Schema-like parameters visible to the model |
| `Tool` | definition + local executor + optional `ToolPermissionSpec` |
| `ToolResult` | normalized `name`, `ok`, `content`, `data`, and `error` |
| `ToolPermissionSpec` | how to derive a permission request from concrete arguments |

An executor returns `ToolResult`. `ToolRegistry.execute` also converts unknown
names, invalid arguments, and execution exceptions into failed results. The
model can read the failure reason, and the loop can record the outcome.

### The Two Types Named ToolExecutor

`tools.types.ToolExecutor` is the callable protocol for `Tool.executor`:
accept arguments and return a `ToolResult`. `agent.tool_execution.ToolExecutor`
is the runtime scheduler for batches, validation, permission pauses and resumes,
parallel execution, and result recording. One describes a function; the other
decides when those functions run.

| Object | Responsibility |
| --- | --- |
| `agent.tool_execution.ToolExecutor` | Schedule execution, pauses, parallel/background work, and record results |
| `PermissionAwareToolRegistry` | Build permission requests and obtain decisions from `PermissionManager` |
| `ToolRegistry` | Find a tool by name, call `Tool.executor`, and normalize failures |

## Building and Wrapping a Registry

`create_builtin_registry` in `tools/builtin.py` assembles groups: inspection,
mutation, execution, network, git, and interaction tools. Function signatures
become a baseline schema through `utils/introspection.py`; curated descriptions
are then applied so tool instructions are useful to a model rather than raw
Python docstrings.

The raw builtins are never the whole runtime registry. `create_session_tool_registry`
adds `task_boundary`, optionally `retrieve_archive`, and wraps the base
`ToolRegistry` in `PermissionAwareToolRegistry` whenever a manager exists.
Session injection is essential because those tools need session state and must
not be globally stateless.

At composition time, `app.factory` may also merge MCP tools via
`McpToolProvider` before the session registry is built. That is still one tool
surface for the loop—not a second agent.

### Hidden control-plane tools

`firstcoder.tools.hidden.HIDDEN_TOOL_STATUS_NAMES` is the single list of tools
that stay out of both the main model surface and noisy human activity streams
(currently `task_boundary`). The session registry retains them for dedicated
runtime controllers, while `AgentLoop` filters their schemas and rejects any
hallucinated main-model call. Do not scatter tool-name checks through the UI.

## Execution Rules

`ToolRegistry.execute(name, arguments)` finds and executes a tool, normalizing
failures. `PermissionAwareToolRegistry.preflight` builds a request and obtains
an allow/ask/deny decision without executing the tool. During loop scheduling,
`ToolExecutor` stores the original call and pauses on `ASK`, then resumes after
the answer. Calling the permission wrapper's `execute` directly returns a
structured confirmation signal on `ASK` and likewise does not execute the tool.

Before a supported direct mutation reaches the executor, `ToolExecutor` also
uses `tools.review.build_prewrite_review` to construct a trusted diff and file
snapshots. This is control-plane behavior, not a model-visible tool. It covers
`write`, `edit`, `apply_patch`, and `delete`; it deliberately excludes `shell`,
whose effect cannot be safely precomputed. A resumed Apply validates snapshots
again and always executes the locally retained original `ToolCall`.

The agent loop guarantees legal conversation ordering:

```text
assistant(tool_call id=call_1) -> tool(tool_call_id=call_1)
```

Denied, skipped, and failed calls still receive the second message. Never
invent a tool result in UI code or remove one during context compaction.

## Special Tools

- TaskPlan has four model-visible tools: `task_list` reads the authoritative
  snapshot and revision; `task_create` creates a plan or appends tasks;
  `task_update` atomically changes status, owner, or dependencies by stable
  task ID; and `task_revise` changes task wording only when its semantic
  content changes. The three write tools require `expected_revision`.
- `linear` plans derive execution order from stable task order; `dag` plans use
  explicit dependencies. The model never submits derived ready/blocked state
  or a replacement snapshot to report ordinary progress.
- A successful TaskPlan mutation appends exactly one `task_plan_updated`
  snapshot, so `SessionView.task_plan`, resume/fork, and the TUI share one
  durable fact model. On a revision conflict, call `task_list` and retry with
  its revision.
- Session schemas are a strict compatibility boundary: resume/fork reject old,
  missing-version, and future-version logs. There is no plan migration or
  fallback from older tool results.
- `think` records internal structured reasoning without mutating the workspace.
- `task_boundary` is an internal runtime control tool used only by the hidden
  classifier; hashes are generated program-side. The main model cannot call it.
- `retrieve_archive` reads bounded archived output only from the current
  session.
- `web_search` uses a hosted Parallel MCP endpoint and may fall back to Exa
  when configured. It is built in, not a server from the user's MCP config.

These are runtime participants, not merely convenience commands. Treat a
change to their output schema as a compatibility change for context and tests.

## Dependency Rule

`tools` (like `permissions` and `utils`) may import `firstcoder.runtime` for
shared cancel/user-input types. It must **not** import `firstcoder.agent`. If a
DTO is needed both below and above the loop, put it in `runtime/`.

## Add a Tool Safely

1. Write a small executor returning a truthful `ToolResult`.
2. Derive/validate its schema and give it a curated description.
3. Declare `ToolPermissionSpec` at registration time when it touches local or
   network resources.
4. Add it to the correct builtin group; do not add loop-only special cases.
5. Test success, invalid arguments, denied/ask behavior, and provider-visible
   sequence when relevant.

```sh
.venv/bin/python -m pytest tests/test_tools.py tests/test_schema.py \
  tests/test_introspection.py tests/test_execution_tools.py \
  tests/test_permission_registry.py tests/test_prewrite_review.py -q
```

## Failure Diagnosis

| Observation | Likely owner |
| --- | --- |
| tool not offered to model | builtin/session registry or provider capabilities |
| model sees wrong parameters | schema generation or curated definition |
| executor ran without expected confirmation | permission spec/wrapper/policy |
| provider rejects history after a call | missing/mismatched tool result id |
| tool works in a unit test but not a session | session-scoped registry assembly |

Related: [Architecture](ARCHITECTURE.md), [Permissions](PERMISSIONS_DESIGN.md), and [Providers](PROVIDERS_DESIGN.md).
