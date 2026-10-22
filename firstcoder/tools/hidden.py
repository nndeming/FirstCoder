"""不应出现在普通用户可见活动流中的工具。"""

from __future__ import annotations

# 内部控制工具：代理需要使用，但对用户来说会产生多余信息。
HIDDEN_TOOL_STATUS_NAMES: frozenset[str] = frozenset({"task_boundary"})
