"""第二阶段委派工具的子代理执行器。

首版实现将职责边界限制为：

- 为子代理创建带元数据标记的新会话；
- 按配置限制工具访问；
- 不向子代理提供委派工具，避免递归；
- 前台执行只向父会话返回精简摘要；
- 后台执行交由第一阶段的通用异步运行机制处理。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from firstcoder.agent.loop_limits import AgentLoopLimits
from firstcoder.agent.session import AgentSession
from firstcoder.agent.worktree import Worktree, WorktreeDiff, WorktreeError, WorktreeManager
from firstcoder.context.identity import new_session_id
from firstcoder.context.store import JsonlSessionStore
from firstcoder.permissions.grants import PermissionGrantStore
from firstcoder.permissions.manager import PermissionManager
from firstcoder.permissions.policy import DefaultPermissionPolicy
from firstcoder.permissions.types import PermissionAction, PermissionGrant, PermissionMode, PermissionScopeType
from firstcoder.providers.base import ChatProvider
from firstcoder.providers.types import ChatResponse, MainRequestOptions
from firstcoder.skills.models import SkillCatalog
from firstcoder.tools.types import Tool
from firstcoder.utils.sandbox_access import SandboxAccess, SandboxAccessMode

SubagentRole = Literal["researcher", "reviewer", "tester", "coder"]

READ_ONLY_TOOL_NAMES = frozenset(
    {
        "ls",
        "view",
        "grep",
        "glob",
        "tree",
        "read_multi",
        "git_status",
        "git_diff",
        "git_log",
        "diagnostics",
        "think",
        "retrieve_archive",
    }
)
REVIEWER_TOOL_NAMES = frozenset({"view", "grep", "git_status", "git_diff", "git_log", "read_multi", "think", "retrieve_archive"})
TESTER_TOOL_NAMES = READ_ONLY_TOOL_NAMES | frozenset({"shell", "python_exec"})
CODER_TOOL_NAMES = TESTER_TOOL_NAMES | frozenset({"write", "edit", "delete", "apply_patch"})


@dataclass(frozen=True, slots=True)
class SubagentProfile:
    role: SubagentRole
    description: str
    allowed_tool_names: frozenset[str]
    allow_background: bool = True
    requires_worktree: bool = False


SUBAGENT_PROFILES: dict[str, SubagentProfile] = {
    "researcher": SubagentProfile(
        role="researcher",
        description="Read-only codebase exploration and evidence collection.",
        allowed_tool_names=READ_ONLY_TOOL_NAMES,
    ),
    "reviewer": SubagentProfile(
        role="reviewer",
        description="Read-only review of diffs, call sites, and risks.",
        allowed_tool_names=REVIEWER_TOOL_NAMES,
    ),
    "tester": SubagentProfile(
        role="tester",
        description="Validation-focused investigation with diagnostics and approved execution tools.",
        allowed_tool_names=TESTER_TOOL_NAMES,
    ),
    "coder": SubagentProfile(
        role="coder",
        description=("Implementation work. Background coding runs inside an isolated git worktree " "so it can never mutate the parent working tree."),
        allowed_tool_names=CODER_TOOL_NAMES,
        allow_background=True,
        requires_worktree=True,
    ),
}

SUBAGENT_ROLE_LIMITS: dict[str, AgentLoopLimits] = {
    "researcher": AgentLoopLimits(
        max_tool_rounds=16,
        max_provider_calls=32,
        max_turn_seconds=480,
    ),
    "reviewer": AgentLoopLimits(
        max_tool_rounds=10,
        max_provider_calls=20,
        max_turn_seconds=300,
    ),
    "tester": AgentLoopLimits(
        max_tool_rounds=30,
        max_provider_calls=60,
        max_turn_seconds=900,
    ),
    "coder": AgentLoopLimits(
        max_tool_rounds=40,
        max_provider_calls=80,
        max_turn_seconds=1200,
    ),
}


@dataclass(slots=True)
class SubagentRequest:
    role: SubagentRole
    task: str
    parent_session_id: str
    parent_task_hash: str | None = None
    parent_summary: str | None = None
    path_hints: list[str] = field(default_factory=list)
    run_in_background: bool = False
    isolate_worktree: bool = False


@dataclass(slots=True)
class SubagentResult:
    ok: bool
    role: SubagentRole
    child_session_id: str
    summary: str
    evidence: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    error: str | None = None
    worktree_path: str | None = None
    worktree_branch: str | None = None
    diff_summary: str | None = None

    def to_data(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "child_session_id": self.child_session_id,
            "summary": self.summary,
            "evidence": list(self.evidence),
            "files_changed": list(self.files_changed),
            "error": self.error,
            "worktree_path": self.worktree_path,
            "worktree_branch": self.worktree_branch,
            "diff_summary": self.diff_summary,
        }


class SubagentRunner:
    """为委派任务创建并运行隔离的子 AgentLoop 会话。"""

    def __init__(
        self,
        *,
        store: JsonlSessionStore,
        provider: ChatProvider,
        tools: list[Tool],
        project_root: str | Path | None = None,
        agents_md: str = "",
        skill_catalog: SkillCatalog | None = None,
        permission_manager: PermissionManager | None = None,
        sandbox_access: SandboxAccess | None = None,
        request_options: MainRequestOptions | None = None,
        limits: AgentLoopLimits | None = None,
    ) -> None:
        self.store = store
        self.provider = provider
        self.tools = list(tools)
        self.project_root = Path(project_root).resolve() if project_root is not None else None
        self.agents_md = agents_md
        self.skill_catalog = skill_catalog or SkillCatalog()
        self.permission_manager = permission_manager
        self.sandbox_access = sandbox_access or SandboxAccess()
        self.request_options = request_options or MainRequestOptions()
        self.limits = limits
        self.profile_map = SUBAGENT_PROFILES

    def profile(self, role: str) -> SubagentProfile | None:
        return SUBAGENT_PROFILES.get(str(role))

    def tools_for_role(self, role: str) -> list[Tool]:
        profile = self.profile(role)
        if profile is None:
            return []
        return [tool for tool in self.tools if tool.name in profile.allowed_tool_names and tool.name != "delegate"]

    def limits_for_role(self, role: str) -> AgentLoopLimits:
        """显式覆盖优先，否则按 reviewer/tester 等角色分配独立预算。"""

        if self.limits is not None:
            return self.limits
        return SUBAGENT_ROLE_LIMITS.get(
            str(role),
            AgentLoopLimits(max_tool_rounds=20, max_provider_calls=40, max_turn_seconds=600),
        )

    def run(self, request: SubagentRequest) -> SubagentResult:
        profile = self.profile(request.role)
        if profile is None:
            return SubagentResult(
                ok=False,
                role=request.role,
                child_session_id="",
                summary=f"Unknown subagent role: {request.role}",
                error="unknown_role",
            )
        if request.run_in_background and not profile.allow_background:
            return SubagentResult(
                ok=False,
                role=request.role,
                child_session_id="",
                summary=f"{request.role} 不支持后台执行。",
                error="background_not_allowed",
            )

        if self._needs_worktree(request, profile=profile):
            return self._run_isolated(request, profile=profile)
        return self._run_inline(request, profile=profile)

    def _needs_worktree(self, request: SubagentRequest, *, profile: SubagentProfile) -> bool:
        """判断本次执行是否必须使用隔离的 Git 工作树。

        可修改文件的角色在后台运行时必须隔离，确保后台任务不会触碰父会话工作树。
        调用方也可通过 ``request.isolate_worktree`` 显式要求隔离；
        父会话的 ToolExecutor 将编码委派转入后台时会使用该字段。
        """

        if request.isolate_worktree:
            return True
        return bool(profile.requires_worktree and request.run_in_background)

    def _run_inline(self, request: SubagentRequest, *, profile: SubagentProfile) -> SubagentResult:
        """第二阶段的原始行为：子代理使用以父会话目录为根的工具。"""

        child_session = self.create_child_session(request, profile=profile)
        prompt = self._child_prompt(request, profile=profile)
        from firstcoder.agent.loop import AgentLoop

        loop = AgentLoop(
            session=child_session,
            provider=self.provider,
            tools=self.tools_for_role(request.role),
            limits=self.limits_for_role(request.role),
            request_options=self.request_options,
            background_manager=None,
            enable_delegate_tool=False,
        )
        try:
            result = asyncio.run(loop.run_user_turn(prompt))
            response = result.response
            if response is None:
                raise RuntimeError("subagent paused for user input")
        except Exception as exc:  # noqa: BLE001 - delegate must return a tool result, not break parent loop
            return SubagentResult(
                ok=False,
                role=request.role,
                child_session_id=child_session.session_id,
                summary=f"Subagent failed: {exc}",
                error=str(exc),
            )
        content = response.content.strip() or "Subagent finished without text output."
        return SubagentResult(
            ok=True,
            role=request.role,
            child_session_id=child_session.session_id,
            summary=content,
        )

    def _run_isolated(self, request: SubagentRequest, *, profile: SubagentProfile) -> SubagentResult:
        """第四阶段：在专用 Git 工作树中运行可修改文件的子代理。

        为子代理重新创建以隔离工作树为根的工具，以及 ``project_root`` 指向该路径的
        子 ``PermissionManager``，确保其无法读写父会话工作树。任务结束后保留工作树，
        并返回差异摘要供父会话显式审查，不自动合并。
        """

        if self.project_root is None:
            return SubagentResult(
                ok=False,
                role=request.role,
                child_session_id="",
                summary="无法隔离执行：未知项目根目录。",
                error="worktree_unavailable",
            )
        manager = WorktreeManager(self.project_root)
        if not manager.available():
            return SubagentResult(
                ok=False,
                role=request.role,
                child_session_id="",
                summary="无法隔离执行：当前项目不是 git 仓库，后台 coder 需要 worktree 隔离。",
                error="worktree_unavailable",
            )

        session_id = new_session_id()
        try:
            worktree = manager.create(session_id)
        except WorktreeError as exc:
            return SubagentResult(
                ok=False,
                role=request.role,
                child_session_id=session_id,
                summary=f"创建隔离 worktree 失败：{exc}",
                error="worktree_create_failed",
            )

        try:
            child_session = self._create_isolated_child_session(request, profile=profile, worktree=worktree, session_id=session_id)
            prompt = self._child_prompt(request, profile=profile, worktree=worktree)
            from firstcoder.agent.loop import AgentLoop

            loop = AgentLoop(
                session=child_session,
                provider=self.provider,
                tools=self._worktree_child_tools(worktree.path, profile=profile, access=child_session.sandbox_access),
                limits=self.limits_for_role(request.role),
                request_options=self.request_options,
                background_manager=None,
                enable_delegate_tool=False,
            )
            try:
                result = asyncio.run(loop.run_user_turn(prompt))
                response = result.response
                if response is None:
                    diff = manager.diff(worktree)
                    return SubagentResult(
                        ok=False,
                        role=request.role,
                        child_session_id=session_id,
                        summary="隔离 coder 等待用户输入，无法在后台继续。",
                        error="waiting_for_user_input",
                        files_changed=diff.files_changed,
                        worktree_path=str(worktree.path),
                        worktree_branch=worktree.branch,
                        diff_summary=diff.render(),
                    )
            except Exception as exc:  # noqa: BLE001 - never break the parent loop
                diff = manager.diff(worktree)
                return SubagentResult(
                    ok=False,
                    role=request.role,
                    child_session_id=session_id,
                    summary=f"隔离 coder 执行失败：{exc}",
                    error=str(exc),
                    files_changed=diff.files_changed,
                    worktree_path=str(worktree.path),
                    worktree_branch=worktree.branch,
                    diff_summary=diff.render(),
                )
            diff = manager.diff(worktree)
            content = response.content.strip() or "Subagent finished without text output."
            summary = self._compose_isolated_summary(content, worktree=worktree, diff=diff)
            return SubagentResult(
                ok=True,
                role=request.role,
                child_session_id=session_id,
                summary=summary,
                files_changed=diff.files_changed,
                worktree_path=str(worktree.path),
                worktree_branch=worktree.branch,
                diff_summary=diff.render(),
            )
        except Exception as exc:  # noqa: BLE001 - defensive: setup failures must not break parent loop
            return SubagentResult(
                ok=False,
                role=request.role,
                child_session_id=session_id,
                summary=f"隔离执行初始化失败：{exc}",
                error=str(exc),
                worktree_path=str(worktree.path),
                worktree_branch=worktree.branch,
            )

    def create_child_session(self, request: SubagentRequest, *, profile: SubagentProfile) -> AgentSession:
        session_id = new_session_id()
        child = AgentSession.create(
            store=self.store,
            session_id=session_id,
            agents_md=self.agents_md,
            skill_catalog=self.skill_catalog,
            tools=self._supplied_tools_for_child(profile.role),
            permission_manager=self.permission_manager,
            sandbox_access=self.sandbox_access,
        )
        child.writer.append_session_metadata_updated(
            parent_session_id=request.parent_session_id,
            parent_task_hash=request.parent_task_hash,
            delegate_role=profile.role,
            delegate_task=request.task,
        )
        return child

    def _supplied_tools_for_child(self, role: str) -> list[Tool]:
        """传给 AgentSession.create 的工具，不包含会话保留工具。

        ``retrieve_archive`` 由 ``create_session_tool_registry`` 为每个会话单独注入，
        传入父会话的工具实例会触发保留名称检查。
        """

        return [tool for tool in self.tools_for_role(role) if tool.name != "retrieve_archive"]

    def _create_isolated_child_session(
        self,
        request: SubagentRequest,
        *,
        profile: SubagentProfile,
        worktree: Worktree,
        session_id: str,
    ) -> AgentSession:
        """创建工具与权限均以隔离工作树为根的子会话。

        子会话使用独立的 ``PermissionManager``，或父权限管理器的私有副本，
        其 ``project_root`` 指向隔离工作树。因此，路径、命令与 Git 操作的
        权限判断都基于隔离目录，不使用父会话工作目录。
        """

        permission_manager = self._child_permission_manager(worktree.path)
        # PROJECT 沙箱将每个文件工具的实际访问范围限制在隔离工作树内，
        # 即使权限策略自动允许树内写入，也不会突破该范围。
        sandbox_access = SandboxAccess(mode=SandboxAccessMode.PROJECT)
        child = AgentSession.create(
            store=self.store,
            session_id=session_id,
            agents_md=self.agents_md,
            skill_catalog=self.skill_catalog,
            tools=self._worktree_child_tools(worktree.path, profile=profile, access=sandbox_access, for_registry=True),
            permission_manager=permission_manager,
            sandbox_access=sandbox_access,
        )
        # 隔离的后台编码代理没有交互用户，逐次写入审查会让任务无法继续。
        # 工作树差异改由父会话审查，因此这里关闭会暂停执行的
        # 写入前审查流程。
        child.require_prewrite_review = False
        child.writer.append_session_metadata_updated(
            parent_session_id=request.parent_session_id,
            parent_task_hash=request.parent_task_hash,
            delegate_role=profile.role,
            delegate_task=request.task,
            worktree_path=str(worktree.path),
            worktree_branch=worktree.branch,
        )
        return child

    def _child_permission_manager(self, root) -> PermissionManager:
        """创建以隔离工作树为根、可自主执行任务的权限管理器。

        所有路径、命令与 Git 操作的权限判断均以隔离工作树为准。AGGRESSIVE 模式
        自动允许树内写入及安全验证命令白名单，使后台编码代理无需交互也能推进任务。
        敏感路径、删除操作与危险命令仍需确认，此时任务会暂停，不会越过沙箱。
        使用全新的授权存储，避免继承仅适用于父会话的授权。
        """

        grants = PermissionGrantStore()
        root_value = str(root)
        for action in (PermissionAction.WRITE_PATH, PermissionAction.DELETE_PATH):
            grants.add(
                PermissionGrant(
                    id=f"grant_subagent_{action.value}",
                    effect="allow",
                    action=action,
                    scope_type=PermissionScopeType.PATH_TREE,
                    scope_value=root_value,
                    created_at="runtime",
                    reason="Isolated background coder may mutate only its dedicated worktree.",
                )
            )
        return PermissionManager(
            policy=DefaultPermissionPolicy(root),
            grants=grants,
            mode=PermissionMode.AGGRESSIVE,
        )

    def _worktree_child_tools(
        self,
        root,
        *,
        profile: SubagentProfile,
        access: SandboxAccess,
        for_registry: bool = False,
    ) -> list[Tool]:
        """按子代理角色重新创建以隔离工作树为根的工具。

        不复用父会话工具，确保工具内部的路径沙箱指向隔离工作树。
        传入 ``AgentSession`` 时，通过 ``for_registry`` 排除会话保留工具名。
        """

        from firstcoder.tools.builtin import create_builtin_registry

        registry = create_builtin_registry(
            root,
            include_mutation_tools=True,
            include_execution_tools=True,
            include_network_tools=True,
            access=access,
        )
        allowed = profile.allowed_tool_names
        tools = [tool for tool in registry.tools() if tool.name in allowed and tool.name != "delegate"]
        if for_registry:
            tools = [tool for tool in tools if tool.name != "retrieve_archive"]
        return tools

    def _child_prompt(
        self,
        request: SubagentRequest,
        *,
        profile: SubagentProfile,
        worktree: Worktree | None = None,
    ) -> str:
        hints = "\n".join(f"- {hint}" for hint in request.path_hints if str(hint).strip())
        summary = request.parent_summary.strip() if request.parent_summary else "(none provided)"
        if worktree is not None:
            root = str(worktree.path)
            isolation = (
                "You are running inside an ISOLATED git worktree. All edits stay on branch "
                f"{worktree.branch} and never touch the parent working tree. Implement the task, "
                "then summarize what you changed. Do not attempt to merge or push.\n"
            )
        else:
            root = str(self.project_root) if self.project_root is not None else "(current project root)"
            isolation = ""
        return (
            f"You are a FirstCoder subagent with role: {profile.role}.\n"
            f"Role scope: {profile.description}\n"
            f"Project root: {root}\n"
            f"{isolation}"
            "Do not call delegate or spawn nested subagents.\n"
            "Return a compact final report with: summary, evidence, files changed, and risks.\n\n"
            f"Parent summary:\n{summary}\n\n"
            f"Path hints:\n{hints or '(none)'}\n\n"
            f"Task:\n{request.task}"
        )

    def _compose_isolated_summary(self, content: str, *, worktree: Worktree, diff: "WorktreeDiff") -> str:
        parts = [
            content,
            "",
            "--- isolated worktree ---",
            f"path: {worktree.path}",
            f"branch: {worktree.branch}",
            "diff:",
            diff.render(),
        ]
        return "\n".join(parts)
