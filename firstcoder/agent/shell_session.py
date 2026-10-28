"""跨工具调用存活的持久 shell 会话。

对标 opencode 的 persistent-pty：shell 工具不再每次新起一次性进程，而是向一个
长驻 bash 写入命令。环境变量、cwd、shell 函数、`&` 后台作业在同一会话内自然保留。

实现上用管道 + 哨兵标记而不是 PTY：

- Windows 上 PTY 需要 ConPTY/pywinpty，管道方案零额外依赖（有 Git Bash 即可）；
- 每条命令结束后由 shell 自己打印随机化哨兵（退出码、cwd、stderr 结束标记），
  输出边界不依赖对命令文本的解析，多行命令、引号嵌套都不影响；
- 代价是无法运行 vim/top 这类真 TUI 程序——这类需求本来就应该走 process_start；
- 同理，从 stdin 读输入的命令（read、交互式确认提示）会吃掉协议行，应当避免。

会话进程与子命令运行在同一进程组；超时或取消时整条会话被回收（状态丢失），
下一次 shell 调用会惰性重建。持久的后台服务请使用 process_start（独立进程组、
日志落盘），不要依赖本模块的会话存活期。
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, cast

from firstcoder.runtime.cancellation import CancellationToken
from firstcoder.utils.subprocess import CommandResult, process_group_kwargs, terminate_process_group
from firstcoder.utils.text import truncate_head_tail

# 等待哨兵时的轮询间隔：既要能及时响应取消，又不至于空转占 CPU。
_POLL_INTERVAL_SECONDS = 0.05


class PersistentShellSession:
    """一个长驻 bash 进程，通过 stdin/stdout/stderr 管道执行命令。

    线程模型：两个守护线程分别持续排空 stdout/stderr 到队列（避免管道写满
    造成死锁），`run()` 在持锁状态下消费队列直到看见本条命令的哨兵。
    """

    def __init__(self, *, cwd: Path, env: dict[str, str]) -> None:
        bash = shutil.which("bash")
        if bash is None:
            raise RuntimeError("未找到 bash；持久 shell 会话需要 bash（Windows 上可用 Git Bash）")
        # 每条会话的哨兵带随机 token，命令输出里恰好出现哨兵文本的概率可忽略。
        token = uuid.uuid4().hex[:12]
        self._end_marker = f"__FC_SHELL_END_{token}__"
        self._err_marker = f"__FC_SHELL_ERR_{token}__"
        self._pwd_marker = f"__FC_SHELL_PWD_{token}__"

        self._process = subprocess.Popen(
            # --noprofile --norc：避免用户 rc 文件污染会话初始状态，行为可预期。
            [bash, "--noprofile", "--norc"],
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,  # 行缓冲，哨兵行能尽快到达读取线程
            **cast(Any, process_group_kwargs()),
        )
        self._cwd = cwd
        self._lock = threading.Lock()
        self._stdout_queue: queue.Queue[str | None] = queue.Queue()
        self._stderr_queue: queue.Queue[str | None] = queue.Queue()
        self._stdout_thread = _start_reader(self._process.stdout, self._stdout_queue)
        self._stderr_thread = _start_reader(self._process.stderr, self._stderr_queue)

    @property
    def alive(self) -> bool:
        return self._process.poll() is None

    @property
    def cwd(self) -> Path:
        """最近一次命令结束后 shell 所在的工作目录。"""
        return self._cwd

    def run(
        self,
        command: str,
        *,
        timeout_seconds: int,
        max_output_chars: int,
        cancellation_token: CancellationToken | None = None,
    ) -> CommandResult:
        """在持久 shell 中执行一条命令，返回与 run_command 相同结构的结果。"""
        with self._lock:
            return self._run_locked(
                command,
                timeout_seconds=timeout_seconds,
                max_output_chars=max_output_chars,
                cancellation_token=cancellation_token,
            )

    def close(self) -> None:
        """终止整个会话进程组（含 shell 里用 & 拉起的后台作业）。"""
        if self.alive:
            terminate_process_group(self._process)
        for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass

    def _run_locked(
        self,
        command: str,
        *,
        timeout_seconds: int,
        max_output_chars: int,
        cancellation_token: CancellationToken | None,
    ) -> CommandResult:
        # 上一条命令遗留的后台作业输出可能仍留在队列里；丢弃以保证哨兵协议对齐。
        _drain(self._stdout_queue)
        _drain(self._stderr_queue)

        # 命令脚本：先跑用户命令，再由 shell 自己依次打印
        # stdout 哨兵（携带退出码）→ stderr 哨兵 → cwd 哨兵。
        # stderr 也有哨兵，因此能把本条命令的 stderr 与后台作业的杂散输出切开，
        # 而不必依赖"等一小会儿再读"这种不可靠的时序假设。
        payload = (
            f"{command}\n"
            f"__fc_rc=$?\n"
            f"printf '\\n{self._end_marker}%s\\n' \"$__fc_rc\"\n"
            f"printf '\\n{self._err_marker}\\n' >&2\n"
            # Git Bash 下 pwd 输出 MSYS 路径（/d/...），-W 给出 Windows 路径，
            # 供沙箱边界检查使用；POSIX 上 -W 不存在，回退普通 pwd。
            f"printf '{self._pwd_marker}%s\\n' \"$(pwd -W 2>/dev/null || pwd)\"\n"
        )
        try:
            assert self._process.stdin is not None
            self._process.stdin.write(payload)
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            return CommandResult(
                exit_code=-1,
                stdout="",
                stderr="",
                stdout_truncated=False,
                stderr_truncated=False,
                ok=False,
                error=f"持久 shell 会话不可用：{exc}",
            )

        deadline = time.monotonic() + timeout_seconds
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        def expired() -> bool:
            return time.monotonic() >= deadline

        def cancelled() -> bool:
            return cancellation_token is not None and cancellation_token.is_cancelled

        # 第一阶段：读 stdout 直到结束哨兵，解析退出码。
        exit_code: int | None = None
        while exit_code is None:
            if cancelled():
                return self._abort(stdout_lines, stderr_lines, max_output_chars, error="命令已中断")
            if expired():
                return self._abort(stdout_lines, stderr_lines, max_output_chars, error="命令执行超时")
            line = _take(self._stdout_queue, deadline)
            if line is _DEADLINE:
                continue
            if line is None:
                return self._session_died(stdout_lines, stderr_lines, max_output_chars)
            if line.startswith(self._end_marker):
                exit_code = _parse_exit_code(line[len(self._end_marker):])
            else:
                stdout_lines.append(line)

        # 第二阶段：读 stderr 直到 stderr 哨兵。
        while True:
            if expired():
                return self._abort(stdout_lines, stderr_lines, max_output_chars, error="命令执行超时")
            line = _take(self._stderr_queue, deadline)
            if line is _DEADLINE:
                continue
            if line is None:
                return self._session_died(stdout_lines, stderr_lines, max_output_chars)
            if line.startswith(self._err_marker):
                break
            stderr_lines.append(line)

        # 第三阶段：读 stdout 上的 cwd 哨兵，更新会话 cwd 供工具层做沙箱检查。
        while True:
            if expired():
                return self._abort(stdout_lines, stderr_lines, max_output_chars, error="命令执行超时")
            line = _take(self._stdout_queue, deadline)
            if line is _DEADLINE:
                continue
            if line is None:
                return self._session_died(stdout_lines, stderr_lines, max_output_chars)
            if line.startswith(self._pwd_marker):
                raw_cwd = line[len(self._pwd_marker):].strip()
                if raw_cwd:
                    self._cwd = Path(raw_cwd)
                break
            stdout_lines.append(line)

        # 哨兵 printf 以 '\n' 开头：若命令输出本身以换行结尾，会多出一个空行，
        # 把它剥掉，保持与一次性执行相同的输出形态。
        _strip_trailing_blank(stdout_lines)
        _strip_trailing_blank(stderr_lines)
        stdout, stdout_truncated = truncate_head_tail("".join(stdout_lines), max_output_chars)
        stderr, stderr_truncated = truncate_head_tail("".join(stderr_lines), max_output_chars)
        return CommandResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            ok=exit_code == 0,
        )

    def _abort(
        self,
        stdout_lines: list[str],
        stderr_lines: list[str],
        max_output_chars: int,
        *,
        error: str,
    ) -> CommandResult:
        """超时/取消时回收整条会话。

        管道方案无法只打断当前命令而不伤及 shell 本身（尤其 Windows），
        因此终止整个进程组；工具层会在下次调用时惰性重建会话。
        """
        self.close()
        stdout, stdout_truncated = truncate_head_tail("".join(stdout_lines), max_output_chars)
        stderr, stderr_truncated = truncate_head_tail("".join(stderr_lines), max_output_chars)
        return CommandResult(
            exit_code=-1,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            ok=False,
            error=error,
        )

    def _session_died(
        self,
        stdout_lines: list[str],
        stderr_lines: list[str],
        max_output_chars: int,
    ) -> CommandResult:
        """shell 进程意外退出（如命令里执行了 exit）：尽力回收残余输出。"""
        exit_code = self._process.poll()
        for line in _drain(self._stderr_queue):
            if line is not None and not line.startswith(self._err_marker):
                stderr_lines.append(line)
        stdout, stdout_truncated = truncate_head_tail("".join(stdout_lines), max_output_chars)
        stderr, stderr_truncated = truncate_head_tail("".join(stderr_lines), max_output_chars)
        return CommandResult(
            exit_code=exit_code if exit_code is not None else -1,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            ok=False,
            error="持久 shell 会话已结束（命令中可能包含 exit）；下次调用会自动重建会话",
        )


class ShellSessionManager:
    """按 key 管理持久 shell 会话；shell 工具用固定 key 共享应用级会话。

    与 ProcessManager 一样挂在 app 生命周期上：TUI 卸载时统一 shutdown。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, PersistentShellSession] = {}

    def get_or_create(
        self,
        key: str,
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> tuple[PersistentShellSession, bool]:
        """返回存活会话；已死或不存在时重建。第二个返回值表示是否新建。"""
        with self._lock:
            session = self._sessions.get(key)
            if session is not None and session.alive:
                return session, False
            if session is not None:
                session.close()
            session = PersistentShellSession(cwd=cwd, env=env)
            self._sessions[key] = session
            return session, True

    def shutdown(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()


class _Deadline:
    """_take 的轮询超时哨兵，与 EOF 的 None 区分。"""


_DEADLINE = _Deadline()


def _start_reader(stream: Any, target: queue.Queue[str | None]) -> threading.Thread:
    """守护线程：持续把管道内容按行送入队列，EOF 时送 None。"""

    def _pump() -> None:
        try:
            while True:
                line = stream.readline()
                if line == "":
                    break
                target.put(line)
        except (OSError, ValueError):
            # 会话被 close() 时管道从另一侧关闭，readline 抛错属正常退出路径。
            pass
        target.put(None)

    thread = threading.Thread(target=_pump, daemon=True)
    thread.start()
    return thread


def _take(source: queue.Queue[str | None], deadline: float) -> str | None | _Deadline:
    """在剩余时间内取一行；轮询超时返回 _DEADLINE 以便调用方检查取消。"""
    try:
        return source.get(timeout=max(min(deadline - time.monotonic(), _POLL_INTERVAL_SECONDS), 0.001))
    except queue.Empty:
        return _DEADLINE


def _drain(source: queue.Queue[str | None]) -> list[str | None]:
    items: list[str | None] = []
    while True:
        try:
            items.append(source.get_nowait())
        except queue.Empty:
            return items


def _parse_exit_code(raw: str) -> int:
    try:
        return int(raw.strip())
    except ValueError:
        return -1


def _strip_trailing_blank(lines: list[str]) -> None:
    if lines and lines[-1].strip() == "":
        lines.pop()
