"""代理、工具、权限与工具函数共享的运行时基础组件。

这些类型放在 `agent/` 之外，避免底层模块反向导入上层编排包。
"""

from firstcoder.runtime.cancellation import (
    AgentCancelledError,
    CancellationToken,
    cancellation_context,
    current_cancellation_token,
)
from firstcoder.runtime.user_input import (
    UserInputOption,
    UserInputRequest,
    user_input_request_from_tool_result,
)

__all__ = [
    "AgentCancelledError",
    "CancellationToken",
    "UserInputOption",
    "UserInputRequest",
    "cancellation_context",
    "current_cancellation_token",
    "user_input_request_from_tool_result",
]
