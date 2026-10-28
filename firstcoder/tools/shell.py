"""`shell` 工具。"""

from __future__ import annotations

import shlex
from pathlib import Path

from firstcoder.agent.shell_session import ShellSessionManager
from firstcoder.permissions.types import PermissionAction
from firstcoder.runtime.cancellation import current_cancellation_token
from firstcoder.tools.command_result import command_tool_result
from firstcoder.tools.types import Tool, ToolPermissionSpec, ToolResult, make_error_result
from firstcoder.utils.introspection import tool_from_function
from firstcoder.utils.execution_sandbox import ExecutionSandbox
from firstcoder.utils.sandbox_access import SandboxAccess
from firstcoder.utils.subprocess import CommandResult

DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_MAX_OUTPUT_CHARS = 20000

# shell 工具在全 app 共享的持久会话 key：同一进程内的多次 shell 调用
# 复用同一个长驻 bash，从而保留 env/cwd/后台作业状态。
_DEFAULT_SESSION_KEY = "default"


def create_shell_tool(
    root: str | Path,
    *,
    access: SandboxAccess | None = None,
    session_manager: ShellSessionManager | None = None,
) -> Tool:
    """创建命令执行工具。

    这是高风险工具：调用方必须在用户明确开启执行权限后才能注册它。
    传入 session_manager 时启用持久 shell 会话（命令在同一个长驻 bash 中执行，
    env、cwd、后台作业跨调用保留）；否则退化为每次调用起新进程。
    """

    sandbox = ExecutionSandbox(root, access=access)

    def shell(
        command: str,
        cwd: str = ".",
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        env: dict[str, str] | None = None,
    ) -> ToolResult:
        """在项目内执行 shell 命令；高风险，需显式启用。"""

        if timeout_seconds <= 0:
            return make_error_result("shell", "timeout_seconds 必须大于 0")
        if max_output_chars <= 0:
            return make_error_result("shell", "max_output_chars 必须大于 0")

        try:
            workdir = sandbox.resolve_cwd(cwd)
            accepted_env, rejected_env = sandbox.prepare_env_overrides(env)
        except ValueError as exc:
            return make_error_result("shell", str(exc))
        if rejected_env:
            return make_error_result(
                "shell",
                "拒绝传入敏感环境变量：" + ", ".join(rejected_env),
                rejected_env_keys=list(rejected_env),
            )

        session_notes: list[str] = []
        if session_manager is not None:
            result, effective_cwd = _run_in_session(
                sandbox,
                session_manager,
                command=command,
                workdir=workdir,
                explicit_cwd=cwd.strip() not in ("", "."),
                accepted_env=accepted_env,
                timeout_seconds=timeout_seconds,
                max_output_chars=max_output_chars,
                notes=session_notes,
            )
        else:
            result = sandbox.run(
                command,
                cwd=workdir,
                timeout_seconds=timeout_seconds,
                max_output_chars=max_output_chars,
                shell=True,
                extra_env=accepted_env,
            )
            effective_cwd = workdir

        data = {
            "command": command,
            "cwd": sandbox.relative(effective_cwd) or ".",
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "stdout_truncated": result.stdout_truncated,
            "stderr_truncated": result.stderr_truncated,
            "env_keys": sorted(accepted_env),
            "persistent_session": session_manager is not None,
        }

        tool_result = command_tool_result(
            "shell",
            result,
            data=data,
            nonzero_error=f"命令退出码为 {result.exit_code}",
            success_fallback=f"命令退出码：{result.exit_code}",
        )
        if session_notes:
            tool_result.content += "\n\n[session]\n" + "\n".join(session_notes)
        if result.error == "命令执行超时":
            tool_result.content += (
                "\n\n[agent guidance]\n长编译/安装可显式提高 timeout_seconds（例如 900）；"
                "长期服务请改用 process_start，并用 readiness 条件验证。"
            )
            if session_manager is not None:
                tool_result.content += (
                    "\n超时会回收整条持久 shell 会话（env/cwd 状态丢失），"
                    "下次调用会自动重建。"
                )
        return tool_result

    tool = tool_from_function(shell)
    tool.definition.parameters["properties"]["env"] = {
        "type": "object",
        "additionalProperties": {"type": "string"},
        "description": "Optional non-sensitive environment overrides for this command.",
    }
    tool.definition.parameters["properties"]["timeout_seconds"]["description"] = (
        "Timeout in seconds. Default 120; use up to about 900 for installs/compiles. "
        "Use process_start for a service that should remain running."
    )
    if session_manager is not None:
        tool.definition.description += (
            " Commands run in a persistent shell session: environment variables, the working "
            "directory, shell functions, and `&` background jobs all carry over between calls. "
            "cwd defaults to the session's current directory; pass an explicit cwd to run "
            "elsewhere without moving the session."
        )
        tool.definition.parameters["properties"]["cwd"]["description"] = (
            "Workspace-relative directory. Default \".\" means the persistent session's "
            "current directory (project root on first use); an explicit value runs the "
            "command there without changing the session directory."
        )
    tool.permission = ToolPermissionSpec(
        action=PermissionAction.EXECUTE_SHELL,
        target_arg="command",
        cwd_arg="cwd",
        reason="执行 shell 命令需要用户确认。",
    )
    return tool


def _run_in_session(
    sandbox: ExecutionSandbox,
    manager: ShellSessionManager,
    *,
    command: str,
    workdir: Path,
    explicit_cwd: bool,
    accepted_env: dict[str, str],
    timeout_seconds: int,
    max_output_chars: int,
    notes: list[str],
) -> tuple[CommandResult, Path]:
    """在持久会话中执行命令，并处理 cwd/env 覆盖与沙箱回退。

    覆盖（env、显式 cwd）通过子 shell 包裹实现，只作用于本条命令，
    不污染会话的长期状态；会话自身的 cwd 只跟随命令里真实的 cd。
    """

    try:
        session, created = manager.get_or_create(
            _DEFAULT_SESSION_KEY,
            cwd=workdir,
            env=sandbox.build_env(),
        )
    except RuntimeError as exc:
        return (
            CommandResult(
                exit_code=-1,
                stdout="",
                stderr="",
                stdout_truncated=False,
                stderr_truncated=False,
                ok=False,
                error=str(exc),
            ),
            workdir,
        )
    if created:
        notes.append("持久 shell 会话已（重新）启动，之前的 env/cwd/后台作业状态已清空。")

    wrapped = command
    prefixes: list[str] = []
    if explicit_cwd:
        # as_posix：Git Bash 不认识反斜杠路径。
        prefixes.append(f"cd {shlex.quote(workdir.as_posix())}")
    if accepted_env:
        exports = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(accepted_env.items()))
        prefixes.append(f"export {exports}")
    if prefixes:
        wrapped = "( " + "; ".join(prefixes) + "; " + command + "\n)"

    result = session.run(
        wrapped,
        timeout_seconds=timeout_seconds,
        max_output_chars=max_output_chars,
        cancellation_token=current_cancellation_token(),
    )

    # 命令可能把会话 cd 到了项目根之外：沙箱不允许，静默 cd 回项目根并提示。
    # 会话已死（如命令含 exit）时无法 cd，直接把报告的 cwd 回落到项目根。
    session_cwd = session.cwd
    if not _within_root(session_cwd, sandbox.root):
        if session.alive:
            session.run(
                f"cd {shlex.quote(sandbox.root.as_posix())}",
                timeout_seconds=10,
                max_output_chars=1000,
            )
            notes.append(f"命令把工作目录切到了项目根之外（{session_cwd}），会话已自动回到项目根。")
        session_cwd = sandbox.root
    return result, session_cwd


def _within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False
