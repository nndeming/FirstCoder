"""FirstCoder 内部会话事实模型。

这些模型表示长期会话事实，不等同于某个 provider 的请求格式。provider 请求由
`ContextBuilder` 在每轮调用前投影出来。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from firstcoder.planning.models import TaskPlan

if TYPE_CHECKING:
    from firstcoder.context.checkpoint import Checkpoint


MessageRole = Literal["user", "assistant", "tool", "system_meta"]
PartKind = Literal[
    "text",                  # 普通对话文本 (user/assistant 的话, L1 修剪的对象)
    "tool_call",             # assistant 发起的工具调用, 与 tool_result 必须成对出现
    "tool_result",           # 工具执行结果, 生命周期判定与 L2/L3 压缩的主要对象
    "checkpoint_summary",    # L4 摘要消息的内容 part, 由 checkpoint 投影而来
    "compaction_event_ref",  # 对压缩事件的引用 part, 不占正文只留溯源指针
    "archive_placeholder",   # L3 占位符: 原文已进归档, 只留生命周期说明和 archive_id
]


def utc_now_iso() -> str:
    """返回稳定的 UTC ISO 时间字符串，统一 JSONL 里的时间格式。"""

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class MessagePart:
    """一条消息(`AgentMessage`)内部的"最小内容单元"
    """
    id: str                 # Part的编号
    message_id: str         # 归属的消息ID
    kind: PartKind | str    # 类型
    content: str            # 正文(文本/JSON/占位符)
    metadata: dict[str, Any] = field(default_factory=dict)  # 附加信息

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MessagePart":
        return cls(
            id=str(value["id"]),
            message_id=str(value["message_id"]),
            kind=str(value["kind"]),
            content=str(value.get("content", "")),
            metadata=dict(value.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "message_id": self.message_id,
            "kind": self.kind,
            "content": self.content,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class AgentMessage:
    """一条会话消息 (事实层最小记录单位), 由若干 MessagePart 组成, 落盘进 JSONL 后不再修改."""
    id: str
    session_id: str
    role: MessageRole | str
    parts: list[MessagePart]
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AgentMessage":
        return cls(
            id=str(value["id"]),
            session_id=str(value["session_id"]),
            role=str(value["role"]),
            parts=[MessagePart.from_dict(part) for part in value.get("parts", [])],
            created_at=str(value.get("created_at") or utc_now_iso()),
            metadata=dict(value.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "parts": [part.to_dict() for part in self.parts],
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


def latest_user_message_id(messages: list[AgentMessage]) -> str | None:
    """Return the latest user message ID, if the history contains one."""

    for message in reversed(messages):
        if message.role == "user":
            return message.id
    return None


@dataclass(slots=True)
class SessionView:
    """由事件日志重放得到的当前会话视图: 是投影的输入, 不是 provider 请求格式."""

    session_id: str
    messages: list[AgentMessage] = field(default_factory=list)
    checkpoints: list[Checkpoint] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    task_plan: TaskPlan | None = None
