"""第四阶段：为可修改文件的后台子代理提供 Git 工作树隔离。

管理器负责底层 Git 操作，使编码角色（``coder``）不触碰父会话工作树。
每个隔离任务使用独立工作树与专用分支，工作树存放在仓库的公共 Git 目录下
（``<git-common-dir>/fc-worktrees/<name>``）。这样既不会出现在父目录的
``git status`` 中，也不在普通工具可见的沙箱路径范围内。

本模块落实以下约束，详见 docs/async-subagents-dag-plan.md 第四阶段：

- 只能为真实 Git 仓库创建工作树。
- 创建与比较差异均不修改父会话工作树。
- 除非指定 ``force=True``，否则拒绝删除含未提交改动的工作树。
- 不自动合并，也不将隔离差异应用到父会话工作树。
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from firstcoder.utils.execution_sandbox import ExecutionSandbox

WORKTREE_DIRNAME = "fc-worktrees"
_DIFF_STAT_LIMIT = 8000


class WorktreeError(RuntimeError):
    """Git 工作树操作无法安全完成时抛出的异常。"""


@dataclass(slots=True)
class Worktree:
    """由管理器持有的单个隔离工作树。"""

    name: str
    path: Path
    branch: str
    base_ref: str


@dataclass(slots=True)
class WorktreeDiff:
    """隔离工作树内未提交改动的摘要。"""

    stat: str
    files_changed: list[str]
    has_changes: bool

    def render(self) -> str:
        if not self.has_changes:
            return "(worktree has no changes)"
        return self.stat


def _run_git(cwd: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    """使用经过清理的环境变量运行 Git 命令。

    复用 ``ExecutionSandbox.build_env``，防止含密钥的环境变量泄漏到子进程，
    与项目其他子进程的处理方式保持一致。
    """

    env = ExecutionSandbox(cwd).build_env()
    try:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        return subprocess.CompletedProcess(["git", *args], returncode=1, stdout="", stderr=str(exc))


def is_git_repo(path: str | Path) -> bool:
    """当 ``path`` 位于真实 Git 工作树内时返回 True。"""

    root = Path(path)
    if not root.exists():
        return False
    result = _run_git(root, ["rev-parse", "--is-inside-work-tree"])
    return result.returncode == 0 and result.stdout.strip() == "true"


class WorktreeManager:
    """为后台修改任务创建和清理隔离的 Git 工作树。"""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()

    # -- 查找工作树 --------------------------------------------------------

    def available(self, *, base_ref: str = "HEAD") -> bool:
        """判断当前项目根目录是否支持创建隔离工作树。

        新初始化且没有提交的仓库虽然是 Git 工作树，但只有基础引用能够解析为提交时，
        ``git worktree add ... HEAD`` 才能成功。将此类仓库视为不可用，
        便于在派发后台编码任务前直接拒绝，避免启动注定失败的任务。
        """

        if not is_git_repo(self.project_root):
            return False
        result = _run_git(self.project_root, ["rev-parse", "--verify", f"{base_ref}^{{commit}}"])
        return result.returncode == 0

    def _common_git_dir(self) -> Path:
        result = _run_git(self.project_root, ["rev-parse", "--git-common-dir"])
        if result.returncode != 0:
            raise WorktreeError(result.stderr.strip() or "无法定位 git 目录；当前不是 git 仓库。")
        raw = result.stdout.strip() or ".git"
        common = Path(raw)
        if not common.is_absolute():
            common = (self.project_root / common).resolve()
        return common

    def worktrees_root(self) -> Path:
        return self._common_git_dir() / WORKTREE_DIRNAME

    # -- 生命周期 ----------------------------------------------------------

    def create(self, name: str, *, base_ref: str = "HEAD") -> Worktree:
        """为指定任务或会话名创建全新的工作树与分支。

        项目不是 Git 仓库、目标路径已存在或 Git 拒绝操作时，抛出 ``WorktreeError``。
        创建过程不修改父会话工作树，因此父目录存在未提交改动也不影响创建。
        """

        safe_name = _sanitize_name(name)
        if not safe_name:
            raise WorktreeError("worktree 名称不能为空。")
        if not is_git_repo(self.project_root):
            raise WorktreeError("当前项目不是 git 仓库，无法创建隔离 worktree。")
        if not self.available(base_ref=base_ref):
            raise WorktreeError(f"无法解析 worktree 基准提交：{base_ref}")

        target = self.worktrees_root() / safe_name
        if target.exists():
            raise WorktreeError(f"worktree 路径已存在：{target}")
        target.parent.mkdir(parents=True, exist_ok=True)

        branch = f"fc/subagent/{safe_name}"
        result = _run_git(
            self.project_root,
            ["worktree", "add", "-q", "-b", branch, str(target), base_ref],
        )
        if result.returncode != 0:
            raise WorktreeError(result.stderr.strip() or "git worktree add 失败。")
        return Worktree(name=safe_name, path=target.resolve(), branch=branch, base_ref=base_ref)

    def diff(self, worktree: Worktree) -> WorktreeDiff:
        """汇总未提交改动，包括未跟踪文件。

        先用 ``git add -A -N`` 让新增文件出现在统计中，再读取相对 ``HEAD`` 的
        ``--stat`` 与 ``--name-status``。仅修改隔离工作树的索引，不触碰父会话。
        """

        add_result = _run_git(worktree.path, ["add", "-A", "-N"])
        if add_result.returncode != 0:
            raise WorktreeError(add_result.stderr.strip() or "git add -N 失败。")
        stat_result = _run_git(worktree.path, ["diff", "--stat", "HEAD"])
        names_result = _run_git(worktree.path, ["diff", "--name-status", "HEAD"])
        if stat_result.returncode != 0:
            raise WorktreeError(stat_result.stderr.strip() or "git diff --stat 失败。")
        if names_result.returncode != 0:
            raise WorktreeError(names_result.stderr.strip() or "git diff --name-status 失败。")
        stat = stat_result.stdout.strip()
        files = _parse_name_status(names_result.stdout)
        has_changes = bool(files) or bool(stat)
        if len(stat) > _DIFF_STAT_LIMIT:
            stat = stat[:_DIFF_STAT_LIMIT] + "\n…(diff stat truncated)"
        return WorktreeDiff(stat=stat, files_changed=files, has_changes=has_changes)

    def is_dirty(self, worktree: Worktree) -> bool:
        result = _run_git(worktree.path, ["status", "--porcelain"])
        return bool(result.stdout.strip())

    def remove(self, worktree: Worktree, *, force: bool = False) -> None:
        """移除工作树。

        与 Git 默认行为一致，除非指定 ``force=True``，否则拒绝丢弃未提交改动，
        防止隔离编码代理的产物被静默删除。
        """

        if not force and self.is_dirty(worktree):
            raise WorktreeError("worktree 有未提交改动；请先审查/保存，或用 force=True 显式丢弃。")
        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(str(worktree.path))
        result = _run_git(self.project_root, args)
        if result.returncode != 0:
            raise WorktreeError(result.stderr.strip() or "git worktree remove 失败。")
        _run_git(self.project_root, ["worktree", "prune"])


def _sanitize_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in str(name).strip())
    return cleaned.strip("-")


def _parse_name_status(output: str) -> list[str]:
    files: list[str] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        # 对重命名或复制记录（R100 old new），最后一项是目标路径。
        files.append(parts[-1])
    return files
