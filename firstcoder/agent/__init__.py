"""agent 主循环与会话编排层。"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from firstcoder.agent.loop import AgentLoop
    from firstcoder.agent.session import AgentSession

__all__ = ["AgentLoop", "AgentSession"]


def __getattr__(name: str):
    if name == "AgentLoop":
        from firstcoder.agent.loop import AgentLoop

        return AgentLoop
    if name == "AgentSession":
        from firstcoder.agent.session import AgentSession

        return AgentSession
    raise AttributeError(name)
