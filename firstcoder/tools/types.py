"""工具层共享类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from firstcoder.providers.types import ToolDefinition

if TYPE_CHECKING:
    from firstcoder.permissions.types import PermissionAction


@dataclass(slots=True)
class ToolResult:
    """一次工具执行后的结构化结果。

    agent 后续会把这个结果转换成 `role="tool"` 的消息，再发回模型。
    `content` 是给模型看的主要文本，`data` 保留结构化信息，方便 UI 展示。
    """

    name: str
    ok: bool
    content: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class ToolHandler(Protocol):
    """单个工具函数的调用协议，对应 Tool.executor。

    每个工具最终都是一个接收关键字参数、返回 `ToolResult` 的函数。
    它描述函数签名；批次调度、权限暂停和事件由 agent.tool_execution.ToolExecutor 负责。
    """

    def __call__(self, *args: Any, **kwargs: Any) -> ToolResult: ...


@dataclass(slots=True)
class ToolPermissionSpec:
    """工具的程序侧权限声明。

    这个声明不进入模型可见 schema，只给权限 wrapper 用来构造 `PermissionRequest`。
    目标可由参数名、固定值或 target_builder 提取；cwd_arg 指定工作目录参数。
    """

    action: PermissionAction
    target_arg: str | None = None
    target_value: str | None = None
    target_builder: Callable[[dict[str, Any]], str] | None = None
    cwd_arg: str | None = None
    reason: str = ""
    allow_always: bool = True
    allow_auto: bool = True


@dataclass(slots=True)
class Tool:
    """一个完整工具由模型可见 schema 和本地 executor 组成。"""

    definition: ToolDefinition
    executor: ToolHandler
    permission: ToolPermissionSpec | None = None

    @property
    def name(self) -> str:
        return self.definition.name


def make_error_result(name: str, message: str, **data: Any) -> ToolResult:
    """创建统一的失败结果，避免各工具重复拼接错误结构。"""

    return ToolResult(name=name, ok=False, content=message, data=data, error=message)


def make_text_result(name: str, content: str, **data: Any) -> ToolResult:
    """创建统一的成功文本结果。"""

    return ToolResult(name=name, ok=True, content=content, data=data)
