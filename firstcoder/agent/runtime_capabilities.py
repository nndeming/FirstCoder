"""Agent 运行能力配置与 benchmark 工具路由。"""

from __future__ import annotations

import re
from dataclasses import dataclass

BENCHMARK_BACKGROUND_TOOL_NAMES = frozenset(
    {
        "diagnostics",
        "shell",
        "python_exec",
        "fetch",
        "delegate",
    }
)
PLANNING_TOOL_NAMES = frozenset(
    {"task_create", "task_update", "task_revise", "task_list"}
)

_COMPLEX_TASK_KEYWORDS = re.compile(
    r"\b(?:"
    r"compile|configure|install|service|daemon|server|docker|qemu|ssh|"
    r"repository[- ]wide|multiple files?|end[- ]to[- ]end|migration|"
    r"integration|benchmark|refactor"
    r")\b",
    re.IGNORECASE,
)
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")


@dataclass(frozen=True, slots=True)
class AgentRuntimeCapabilities:
    """由 app/factory 统一装配的运行能力，而不是散落的 benchmark 分支。"""

    allow_user_input: bool = True                       # 是否允许向用户提问
    enable_completion_gate: bool = False                # 完成门控(benchmark模式下要求模型需要根据证据判定任务是否真正完成)
    enable_stagnation_guard: bool = False               # 停滞检查(防止模型反复执行相同操作)
    enable_delegate_tool: bool = True                   # 是否开放子agent委派能力
    expose_planning_tools: bool = True                  # 是否开放规划工具集
    expose_think_tool: bool = True                      # 是否开放think工具
    expose_web_search_tool: bool = True                 # 是否开放联网搜索
    enable_process_tools: bool = True                   # 是否启用进程管理工具
    background_tool_names: frozenset[str] | None = None # 后台执行工具白名单

    @classmethod
    def interactive(cls) -> "AgentRuntimeCapabilities":
        return cls()

    @classmethod
    def benchmark(cls, task: str) -> "AgentRuntimeCapabilities":
        complex_task = benchmark_task_is_complex(task)
        return cls(
            allow_user_input=False,
            enable_completion_gate=True,
            enable_stagnation_guard=True,
            enable_delegate_tool=complex_task,
            expose_planning_tools=complex_task,
            expose_think_tool=False,
            expose_web_search_tool=False,
            enable_process_tools=True,
            background_tool_names=BENCHMARK_BACKGROUND_TOOL_NAMES,
        )


def benchmark_task_is_complex(task: str) -> bool:
    """保守识别值得暴露 TaskPlan/delegate 的多步骤 benchmark 任务。"""

    text = task.strip()
    if len(text) >= 1200:
        return True
    list_items = sum(1 for line in text.splitlines() if _LIST_ITEM_RE.match(line))
    if list_items >= 3:
        return True
    keyword_hits = len(_COMPLEX_TASK_KEYWORDS.findall(text))
    return keyword_hits >= 2
