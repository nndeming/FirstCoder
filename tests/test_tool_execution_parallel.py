from __future__ import annotations

from unittest.mock import MagicMock, patch

from firstcoder.agent.session import AgentSession
from firstcoder.agent.tool_execution import ToolExecutor
from firstcoder.context.store import JsonlSessionStore
from firstcoder.providers.types import ToolCall
from firstcoder.tools.types import ToolResult


def _fake_execute(self, tool_call: ToolCall) -> ToolResult:
    return ToolResult(name=tool_call.name, ok=True, content="ok")


def _make_tool_executor(tmp_path, *, permission_manager=None):
    store = JsonlSessionStore(tmp_path)
    session = AgentSession.create(
        store=store,
        session_id="sess_parallel",
        permission_manager=permission_manager,
    )
    executor = ToolExecutor(
        session=session,
        settlement=MagicMock(),
        emit_event=MagicMock(),
        check_cancelled=lambda: None,
        cancellation_token=None,
        tag_task_boundary_messages=MagicMock(),
        emit_settlements=MagicMock(),
    )
    return executor, session


def _read_calls(*names: str) -> list[ToolCall]:
    return [
        ToolCall(id=f"call_{name}", name=name, arguments={"pattern": "test"})
        for name in names
    ]


def test_parallel_batch_skips_double_permission_check(tmp_path) -> None:
    """每次并行工具调用只应在预检阶段进行一次权限检查。"""

    executor, session = _make_tool_executor(tmp_path)
    calls = _read_calls("grep", "tree", "glob")

    with (
        patch.object(AgentSession, "execute_tool_call_after_permission_confirmation", new=_fake_execute),
        patch.object(AgentSession, "execute_tool_call") as mock_direct,
    ):
        results = executor.execute_parallel_readonly_batch(calls)

    assert len(results) == 3
    assert all(result.ok for result in results)
    mock_direct.assert_not_called()


def test_single_tool_parallel_batch_skips_thread_pool(tmp_path) -> None:
    """批次中只有一个可并行工具时，应使用 execute_single，而非 ThreadPoolExecutor。"""

    executor, _ = _make_tool_executor(tmp_path)
    calls = _read_calls("grep")

    with patch.object(ToolExecutor, "execute_single", return_value=ToolResult(name="grep", ok=True, content="ok")) as mock_single:
        with patch("firstcoder.agent.tool_execution.ThreadPoolExecutor") as mock_pool:
            state = executor.execute_interactive(calls)

    mock_single.assert_called_once()
    mock_pool.assert_not_called()
    assert state.pending_input is None


def test_multi_tool_parallel_batch_uses_thread_pool(tmp_path) -> None:
    """多个可并行工具仍应使用 ThreadPoolExecutor。"""

    executor, _ = _make_tool_executor(tmp_path)
    calls = _read_calls("grep", "glob", "tree")

    with patch.object(AgentSession, "execute_tool_call_after_permission_confirmation", new=_fake_execute):
        state = executor.execute_interactive(calls)

    assert state.pending_input is None
    view = executor.session.store.rebuild_session_view("sess_parallel")
    tool_results = [
        part
        for msg in view.messages
        for part in msg.parts
        if part.kind == "tool_result"
    ]
    assert len(tool_results) == 3


def test_interleaved_batch_runs_sequential_between_parallel(tmp_path) -> None:
    """[grep, write, grep] 应依次以并行安全模式运行 grep、串行写入、再次运行 grep。"""

    executor, _ = _make_tool_executor(tmp_path)
    calls = [
        ToolCall(id="call_grep1", name="grep", arguments={"pattern": "test"}),
        ToolCall(id="call_grep2", name="grep", arguments={"pattern": "test2"}),
        ToolCall(id="call_write", name="write", arguments={"path": "out.txt", "content": "hello"}),
        ToolCall(id="call_grep3", name="grep", arguments={"pattern": "test3"}),
    ]

    with patch.object(AgentSession, "execute_tool_call_after_permission_confirmation", new=_fake_execute):
        state = executor.execute_interactive(calls)

    assert state.pending_input is None
    view = executor.session.store.rebuild_session_view("sess_parallel")
    tool_results = [
        part
        for msg in view.messages
        for part in msg.parts
        if part.kind == "tool_result"
    ]
    assert len(tool_results) >= 3
