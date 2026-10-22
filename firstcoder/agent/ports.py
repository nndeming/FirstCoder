"""代理编排边界使用的稳定协议接口。"""

from __future__ import annotations

from typing import Protocol

from firstcoder.context.manager import ContextCompactRequest, ContextCompactResult


class ContextManagerLike(Protocol):
    """AgentLoop 所需的最小上下文窗口管理接口。"""

    def compact_if_needed(self, request: ContextCompactRequest) -> ContextCompactResult: ...
