"""持久 shell 会话（ShellSessionManager + shell 工具的持久模式）行为测试。

覆盖方案 A 的核心承诺：env/cwd/后台作业跨调用保留，env/cwd 覆盖不污染会话，
超时/exit 后会话可回收重建，以及会话 cwd 越出沙箱时自动回到项目根。
"""

from __future__ import annotations

import shutil

import pytest

from firstcoder.agent.shell_session import ShellSessionManager
from firstcoder.tools import create_builtin_registry

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="持久 shell 会话需要 bash")


@pytest.fixture
def manager():
    session_manager = ShellSessionManager()
    yield session_manager
    session_manager.shutdown()


@pytest.fixture
def registry(tmp_path, manager):
    return create_builtin_registry(
        tmp_path,
        include_execution_tools=True,
        shell_session_manager=manager,
    )


def test_shell_session_preserves_environment_across_calls(registry):
    first = registry.execute("shell", {"command": "export FC_TEST_PERSIST=hello"})
    assert first.ok is True

    second = registry.execute("shell", {"command": "echo \"$FC_TEST_PERSIST\""})

    assert second.ok is True
    assert second.content == "hello"
    assert second.data["persistent_session"] is True


def test_shell_session_preserves_cwd_across_calls(registry, tmp_path):
    (tmp_path / "subdir").mkdir()
    first = registry.execute("shell", {"command": "cd subdir"})
    assert first.ok is True

    second = registry.execute("shell", {"command": "pwd"})

    assert second.ok is True
    assert second.data["cwd"] == "subdir"


def test_shell_session_explicit_cwd_does_not_move_session(registry, tmp_path):
    (tmp_path / "elsewhere").mkdir()
    moved = registry.execute("shell", {"command": "cd elsewhere"})
    assert moved.ok is True

    # 显式 cwd 只作用于本条命令，不改变会话的当前目录
    scoped = registry.execute("shell", {"command": "pwd", "cwd": "."})
    assert scoped.data["cwd"] == "elsewhere"


def test_shell_session_env_override_does_not_leak(registry):
    scoped = registry.execute(
        "shell",
        {"command": "echo \"$FC_TEST_SCOPED\"", "env": {"FC_TEST_SCOPED": "yes"}},
    )
    assert scoped.ok is True
    assert scoped.data["stdout"].strip() == "yes"

    leaked = registry.execute("shell", {"command": "echo \"x${FC_TEST_SCOPED}x\""})
    assert leaked.data["stdout"].strip() == "xx"


def test_shell_session_reports_nonzero_exit(registry):
    result = registry.execute("shell", {"command": "(exit 3)"})

    assert result.ok is False
    assert result.error == "命令退出码为 3"
    assert result.data["exit_code"] == 3


def test_shell_session_recovers_after_exit_command(registry):
    died = registry.execute("shell", {"command": "exit 0"})
    assert died.ok is False
    assert "持久 shell 会话已结束" in died.error

    rebuilt = registry.execute("shell", {"command": "echo back"})
    assert rebuilt.ok is True
    assert rebuilt.data["stdout"].strip() == "back"
    assert "重新" in rebuilt.content


def test_shell_session_timeout_resets_session(registry, manager):
    registry.execute("shell", {"command": "export FC_TEST_TIMEOUT=1"})
    timed_out = registry.execute("shell", {"command": "sleep 30", "timeout_seconds": 1})

    assert timed_out.ok is False
    assert timed_out.error == "命令执行超时"
    assert "会话" in timed_out.content

    # 超时会话已被回收，下次调用惰性重建，env 状态丢失
    rebuilt = registry.execute("shell", {"command": "echo \"x${FC_TEST_TIMEOUT}x\""})
    assert rebuilt.ok is True
    assert rebuilt.data["stdout"].strip() == "xx"


def test_shell_session_cd_outside_root_returns_to_root(registry):
    escaped = registry.execute("shell", {"command": "cd .."})

    assert escaped.ok is True
    assert "项目根之外" in escaped.content

    back = registry.execute("shell", {"command": "pwd"})
    assert back.data["cwd"] == "."


def test_shell_session_separates_stdout_and_stderr(registry):
    result = registry.execute("shell", {"command": "echo out; echo err >&2"})

    assert result.ok is True
    assert result.data["stdout"] == "out\n"
    assert result.data["stderr"] == "err\n"


def test_shell_session_background_job_survives_between_calls(registry):
    started = registry.execute("shell", {"command": "sleep 30 & echo started"})
    assert started.ok is True

    jobs = registry.execute("shell", {"command": "jobs"})
    assert "sleep 30" in jobs.content


def test_shell_manager_reuses_alive_session(manager, tmp_path):
    first, created_first = manager.get_or_create("k", cwd=tmp_path, env={})
    second, created_second = manager.get_or_create("k", cwd=tmp_path, env={})

    assert created_first is True
    assert created_second is False
    assert first is second


def test_shell_manager_recreates_dead_session(manager, tmp_path):
    first, _ = manager.get_or_create("k", cwd=tmp_path, env={})
    first.close()

    second, created = manager.get_or_create("k", cwd=tmp_path, env={})

    assert created is True
    assert second is not first
