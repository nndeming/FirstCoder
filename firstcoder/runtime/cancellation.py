"""Cooperative cancellation primitives for agent turns and tool execution."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass(slots=True)
class CancellationToken:
    """Small thread-safe cancellation flag shared by UI, loop, and tools."""
    # 语义上相当于 Java 的 volatile: 本类只依赖其 "可见性" --
    # 一个线程 cancel() 之后, 其他线程 is_set() 必能读到最新值.
    # 实现上 Event 内部是锁 (比 volatile 重, 还附带原子性与 wait 能力),
    # 但本项目是纯写/纯读, 无复合操作, 恰好只需可见性.
    _event: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise AgentCancelledError()


class AgentCancelledError(RuntimeError):
    """Raised when an agent turn is interrupted by the user."""

    def __init__(self, message: str = "Agent turn was interrupted.") -> None:
        super().__init__(message)


_LOCAL = threading.local()


def current_cancellation_token() -> CancellationToken | None:
    return getattr(_LOCAL, "token", None)


class cancellation_context:
    """Temporarily expose a cancellation token to synchronous tool executors."""

    def __init__(self, token: CancellationToken | None) -> None:
        self.token = token
        self.previous: CancellationToken | None = None

    # with 要求实现__enter__ 和 __exit__ 两个方法
    def __enter__(self) -> None:
        self.previous = current_cancellation_token()
        _LOCAL.token = self.token

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        _LOCAL.token = self.previous
