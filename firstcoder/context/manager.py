"""上下文窗口压缩触发编排。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Literal, Protocol

from firstcoder.context.checkpoint import Checkpoint
from firstcoder.context.compaction import CompactionEvent, CompactionPipeline, CompactionRequest, CompactionResult
from firstcoder.context.context_builder import InvalidCheckpointBoundaryError
from firstcoder.context.fallback import CompactFallbackPolicy, FallbackStep
from firstcoder.context.identity import session_view_fingerprint
from firstcoder.context.llm_compact import LlmCompactCandidate, LlmCompactEvent, LlmCompactRequest
from firstcoder.context.models import AgentMessage, SessionView
from firstcoder.context.runtime_state import SessionRuntimeState, auto_compact_circuit_is_open
from firstcoder.context.store import JsonlSessionStore
from firstcoder.context.token_budget import ContextBudget
from firstcoder.context.tool_sequence import InvalidToolCallSequenceError
from firstcoder.context.triggers import ContextCompactionConfig, evaluate_context_triggers
from firstcoder.context.writer import SessionEventWriter


class ContextWindowTrigger(StrEnum):
    AUTO = "auto"
    TASK_HASH_CHANGED = "task_hash_changed"
    PROMPT_TOO_LONG = "prompt_too_long"
    MANUAL = "manual"


class ContextCompactMode(StrEnum):
    AUTO = "auto"
    MANUAL = "manual"


ManagerStatus = Literal["success", "skipped", "failed"]


class ProgrammaticCompactor(Protocol):
    """L1-L3 程序化压缩器的接口, 具体实现是 CompactionPipeline (compaction.py)."""

    def compact(self, request: CompactionRequest) -> CompactionResult: ...


class L4Compactor(Protocol):
    """L4 LLM 摘要的接口, 具体实现是 LlmCompactService (llm_compact.py); 生成与提交分开, 便于先校验再落盘."""

    def generate_candidate(self, request: LlmCompactRequest) -> LlmCompactCandidate: ...

    def commit_candidate(
        self,
        candidate: LlmCompactCandidate,
        *,
        runtime_state: SessionRuntimeState,
    ) -> Checkpoint: ...


@dataclass(slots=True)
class ContextCompactRequest:
    view: SessionView
    runtime_state: SessionRuntimeState
    budget: ContextBudget
    estimate_budget: Callable[[SessionView], ContextBudget]
    trigger: ContextWindowTrigger | str = ContextWindowTrigger.AUTO
    mode: ContextCompactMode | str = ContextCompactMode.AUTO
    current_turn: int = 0
    target_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ContextCompactResult:
    status: ManagerStatus
    reason: str
    view: SessionView
    before_tokens: int
    after_tokens: int
    programmatic_event: CompactionEvent | None = None
    l4_event: LlmCompactEvent | None = None
    fallback_steps: list[dict[str, object]] | None = None
    final_failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _CandidateOutcome:
    candidate: LlmCompactCandidate
    event: LlmCompactEvent
    view: SessionView
    input_tokens: int


@dataclass(slots=True)
class ContextWindowManager:
    """用一次 provider-facing budget 编排 L1-L4.

    编排入口: 先过触发闸门 (水位线/熔断/无效压缩跳过), 再走 L1-L3 程序化压缩,
    压不到目标就升级 L4 模型摘要, L4 失败按 fallback 策略收场. 全程只追加事件,
    事实文件保持 append-only.
    """

    store: JsonlSessionStore
    pipeline: ProgrammaticCompactor | None = None
    l4_service: L4Compactor | None = None
    config: ContextCompactionConfig | None = None
    fallback_policy: CompactFallbackPolicy = CompactFallbackPolicy()

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = ContextCompactionConfig()
        if self.pipeline is None:
            self.pipeline = CompactionPipeline(
                root=self.store.root,
                large_tool_result_tokens=self.config.large_tool_result_tokens,
                cold_turn_distance=self.config.cold_turn_distance,
                cold_preview_chars=self.config.cold_preview_chars,
            )

    def compact_if_needed(self, request: ContextCompactRequest) -> ContextCompactResult:
        """压缩编排入口: 先过触发闸门(阈值/熔断/无效跳过), 再走 L1-L3 程序化压缩,
        压不到目标就升级 L4 模型摘要; 全程落盘压缩事件, 失败按 fallback 策略收场。"""

        assert self.config is not None
        assert self.pipeline is not None

        # 获取压缩触发类型(trigger) 模式(mode)
        trigger = ContextWindowTrigger(request.trigger)
        mode = ContextCompactMode(request.mode)

        # 压缩前token数
        before_tokens = request.budget.input_tokens
        # 当前视图的指纹
        input_fingerprint = session_view_fingerprint(request.view)
        # 当前连续失败次数
        auto_failure_count_before = request.runtime_state.auto_compact_failure_count

        # 目前实现比较简陋, 就比较watermark
        decision = evaluate_context_triggers(
            request.view,
            self.config,
            input_tokens=before_tokens,
            high_watermark=request.budget.high_watermark,
            low_watermark=request.budget.low_watermark,
        )

        # AUTO模式, 且input_tokens未达到high_watermark
        if trigger == ContextWindowTrigger.AUTO and not decision.should_compact:
            return self._unchanged("skipped", "under_threshold", request, before_tokens)

        # AUTO模式, 且自动熔断开启
        if (
            trigger == ContextWindowTrigger.AUTO
            and mode == ContextCompactMode.AUTO
            and auto_compact_circuit_is_open(request.runtime_state)
        ):
            return self._unchanged("skipped", "circuit_open", request, before_tokens)

        # AUTO模式, 且当前指纹上次压缩过但无效
        if (
            trigger == ContextWindowTrigger.AUTO
            and request.runtime_state.last_no_effect_compaction_fingerprint == input_fingerprint
        ):
            return self._unchanged("skipped", "skipped_no_effect", request, before_tokens)

        target_tokens = _target_tokens(request, trigger)
        if request.budget.fixed_tokens >= request.budget.low_watermark:
            # 如果fixed_tokens >= low_watermark, 就判定失败
            # 因为压缩只压history_tokens
            return ContextCompactResult(
                status="failed",
                reason="fixed_context_over_budget",
                view=request.view,
                before_tokens=before_tokens,
                after_tokens=before_tokens,
                final_failure_reason="fixed_context_over_budget",
            )

        # TASK_HASH_CHANGED情况下必须走完l2, l3
        required_levels: tuple[Literal["l1", "l2", "l3"], ...] = (
            ("l2", "l3") if trigger == ContextWindowTrigger.TASK_HASH_CHANGED else ()
        )
        # 跑程序化流水线压缩
        programmatic = self.pipeline.compact(
            CompactionRequest(
                view=request.view,
                active_task_hash=request.runtime_state.active_task_hash,
                target_tokens=target_tokens,
                current_turn=request.current_turn,
                estimate_tokens=lambda candidate: request.estimate_budget(candidate).input_tokens,
                consumed_tool_result_part_ids=frozenset(
                    request.runtime_state.consumed_tool_result_part_ids
                ),
                required_levels=required_levels,
                l2_result_target_tokens=self.config.l2_result_target_tokens,
                force_route_current_text=_force_route_current_text_for_trigger(trigger),
                force_old_task_compaction=trigger == ContextWindowTrigger.TASK_HASH_CHANGED,
            )
        )

        # 估计"压缩后"的视图token
        after_tokens = request.estimate_budget(programmatic.view).input_tokens

        if trigger == ContextWindowTrigger.AUTO and programmatic.event.noop and after_tokens < target_tokens:
            # AUTO自动触发 + nnop + after_tokens达到目标
            request.runtime_state.last_no_effect_compaction_fingerprint = input_fingerprint
            SessionEventWriter(store=self.store, session_id=request.view.session_id).append_compaction_skipped(
                trigger=trigger.value,
                input_fingerprint=input_fingerprint,
                reason="skipped_no_effect",
            )
            return ContextCompactResult(
                status="skipped",
                reason="skipped_no_effect",
                view=programmatic.view,
                before_tokens=before_tokens,
                after_tokens=after_tokens,
                programmatic_event=programmatic.event,
            )

        self._record_programmatic_event(
            session_id=request.view.session_id,
            trigger=trigger,
            target_tokens=target_tokens,
            event=programmatic.event,
        )

        # after_tokens达到目标 并且 压缩触发原因不是provider判定为超长
        if after_tokens < target_tokens and trigger != ContextWindowTrigger.PROMPT_TOO_LONG:
            self._record_auto_success_if_needed(request=request, mode=mode)
            return ContextCompactResult(
                status="success",
                reason=_result_reason(trigger=trigger, auto_reason=decision.reason),
                view=programmatic.view,
                before_tokens=before_tokens,
                after_tokens=after_tokens,
                programmatic_event=programmatic.event,
            )

        if self.l4_service is None:
            # 这里是走到了L4, L4需要使用大模型进行压缩, 此处判断相关的大模型服务不存在, 需要报错
            return self._final_l4_failure(
                request=request,
                trigger=trigger,
                mode=mode,
                target_tokens=target_tokens,
                programmatic=programmatic,
                before_tokens=before_tokens,
                after_tokens=after_tokens,
                before_failure_count=auto_failure_count_before,
                event=LlmCompactEvent(
                    status="failed",
                    source_fingerprint=programmatic.event.input_fingerprint,
                    failure_reason="l4_service_missing",
                ),
                reason="l4_service_missing",
            )

        # L4级别模型摘要压缩
        outcome = self._generate_validate_commit(
            request=request,
            l4_request=LlmCompactRequest(
                view=programmatic.view,
                runtime_state=request.runtime_state,
                consumed_tool_result_part_ids=frozenset(
                    request.runtime_state.consumed_tool_result_part_ids
                ),
                mode=mode.value,
            ),
            target_tokens=target_tokens,
        )
        if outcome.event.status != "success":
            return self._run_fallback(
                request=request,
                trigger=trigger,
                mode=mode,
                target_tokens=target_tokens,
                programmatic=programmatic,
                outcome=outcome,
                before_failure_count=auto_failure_count_before,
            )

        self._record_l4_event(
            session_id=request.view.session_id,
            trigger=trigger,
            target_tokens=target_tokens,
            event=outcome.event,
        )
        self._record_auto_success_if_needed(request=request, mode=mode)
        return ContextCompactResult(
            status="success",
            reason=_result_reason(trigger=trigger, auto_reason=decision.reason),
            view=outcome.view,
            before_tokens=before_tokens,
            after_tokens=outcome.input_tokens,
            programmatic_event=programmatic.event,
            l4_event=outcome.event,
        )

    def _generate_validate_commit(
        self,
        *,
        request: ContextCompactRequest,
        l4_request: LlmCompactRequest,
        target_tokens: int,
    ) -> _CandidateOutcome:
        """生成 L4 压缩候选, 校验有效性, 达标后提交落盘并重建视图.

        分四步: 生成候选 -> 估算校验 -> 是否达标 -> 提交并重建. 任何一步
        不满足都返回 failed 结果, 由调用方决定降级策略.
        """

        assert self.l4_service is not None
        # 第一步: 让 l4_service 生成候选 checkpoint, 此时尚未落盘
        candidate = self.l4_service.generate_candidate(l4_request)
        event = candidate.event
        if event.status != "success" or candidate.checkpoint is None:
            reason = event.failure_reason or event.status
            # 原因细化: 边界处的工具结果未被消费, 且当前输入已超过容量时, 报更具体的原因
            if reason == "unconsumed_boundary" and request.budget.input_tokens > request.budget.input_capacity:
                reason = "unconsumed_result_over_budget"
            failed = replace(event, status="failed", failure_reason=reason, final_failure_reason=reason)
            return _CandidateOutcome(
                candidate=candidate,
                event=failed,
                view=l4_request.view,
                input_tokens=request.estimate_budget(l4_request.view).input_tokens,
            )

        # 第二步: 构造套用候选 checkpoint 的预演视图, 在不落盘的前提下估算压缩后 token
        candidate_view = _view_with_checkpoint(l4_request.view, candidate.checkpoint)
        try:
            candidate_budget = request.estimate_budget(candidate_view)
        except (InvalidCheckpointBoundaryError, InvalidToolCallSequenceError):
            # 校验失败: checkpoint 边界非法或工具调用序列断裂, 候选不可提交
            failed = replace(
                event,
                status="failed",
                failure_reason="invalid_tool_sequence",
                checkpoint_id=None,
                final_failure_reason="invalid_tool_sequence",
            )
            return _CandidateOutcome(
                candidate=candidate,
                event=failed,
                view=l4_request.view,
                input_tokens=request.estimate_budget(l4_request.view).input_tokens,
            )
        # 第三步: 候选压缩后仍不低于目标值, 说明 L4 摘要没有压到位
        if candidate_budget.input_tokens >= target_tokens:
            failed = replace(
                event,
                status="failed",
                failure_reason="still_over_budget",
                checkpoint_id=None,
                final_failure_reason="still_over_budget",
            )
            return _CandidateOutcome(
                candidate=candidate,
                event=failed,
                view=l4_request.view,
                input_tokens=candidate_budget.input_tokens,
            )

        # 第四步: 校验通过, 提交候选 (checkpoint 作为事件落盘), 再从 store 重放事件得到权威视图
        self.l4_service.commit_candidate(candidate, runtime_state=request.runtime_state)
        rebuilt_view = self.store.rebuild_session_view(request.view.session_id)
        rebuilt_budget = request.estimate_budget(rebuilt_view)
        return _CandidateOutcome(
            candidate=candidate,
            event=event,
            view=rebuilt_view,
            input_tokens=rebuilt_budget.input_tokens,
        )

    def _run_fallback(
        self,
        *,
        request: ContextCompactRequest,
        trigger: ContextWindowTrigger,
        mode: ContextCompactMode,
        target_tokens: int,
        programmatic: CompactionResult,
        outcome: _CandidateOutcome,
        before_failure_count: int,
    ) -> ContextCompactResult:
        """L4 失败后的降级处理: 按 fallback 策略选择补救动作, 尽可能把上下文压回目标内.

        策略分两档: stronger_programmatic (再跑一轮全量 L1-L3, 之后仍可重试 L4)
        和 retry_l4_stronger_summary (换更强的摘要模式重试 L4); 都救不回来就汇总成
        最终失败. 这么做是因为 L4 单次失败多半是输入问题而非终局, 值得用更便宜的手段再试.
        """

        assert self.pipeline is not None
        assert self.config is not None
        reason = outcome.event.failure_reason or outcome.event.status
        action = self.fallback_policy.action_for(reason)
        steps: list[dict[str, object]] = []
        current_programmatic = programmatic
        before_tokens = request.budget.input_tokens

        if action == "stronger_programmatic":
            stronger = self.pipeline.compact(
                CompactionRequest(
                    view=programmatic.view,
                    active_task_hash=request.runtime_state.active_task_hash,
                    target_tokens=target_tokens,
                    current_turn=request.current_turn,
                    estimate_tokens=lambda candidate: request.estimate_budget(candidate).input_tokens,
                    consumed_tool_result_part_ids=frozenset(
                        request.runtime_state.consumed_tool_result_part_ids
                    ),
                    enabled_levels=("l1", "l2", "l3"),
                    required_levels=("l2", "l3") if trigger == ContextWindowTrigger.TASK_HASH_CHANGED else (),
                    l2_result_target_tokens=self.config.l2_result_target_tokens,
                    force_route_current_text=_force_route_current_text_for_trigger(trigger),
                    force_old_task_compaction=trigger == ContextWindowTrigger.TASK_HASH_CHANGED,
                )
            )
            self._record_programmatic_event(
                session_id=request.view.session_id,
                trigger=trigger,
                target_tokens=target_tokens,
                event=stronger.event,
            )
            stronger_tokens = request.estimate_budget(stronger.view).input_tokens
            stronger_status = "success" if stronger_tokens < target_tokens else "failed"
            steps.append(
                FallbackStep(
                    step=1,
                    reason=reason,
                    action=action,
                    before_tokens=outcome.input_tokens,
                    after_tokens=stronger_tokens,
                    status=stronger_status,
                    error=None if stronger_status == "success" else "still_over_budget",
                ).to_dict()
            )
            current_programmatic = stronger
            if stronger_status == "success":
                event = _with_fallback(
                    replace(outcome.event, status="success", failure_reason="fallback_success"),
                    fallback_steps=steps,
                    final_failure_reason=None,
                )
                self._record_l4_event(
                    session_id=request.view.session_id,
                    trigger=trigger,
                    target_tokens=target_tokens,
                    event=event,
                )
                self._record_auto_success_if_needed(request=request, mode=mode)
                return ContextCompactResult(
                    status="success",
                    reason=_result_reason(trigger=trigger, auto_reason=reason),
                    view=stronger.view,
                    before_tokens=before_tokens,
                    after_tokens=stronger_tokens,
                    programmatic_event=stronger.event,
                    l4_event=event,
                    fallback_steps=steps,
                )

        if action in {"stronger_programmatic", "retry_l4_stronger_summary"}:
            retry = self._generate_validate_commit(
                request=request,
                l4_request=LlmCompactRequest(
                    view=current_programmatic.view,
                    runtime_state=request.runtime_state,
                    consumed_tool_result_part_ids=frozenset(
                        request.runtime_state.consumed_tool_result_part_ids
                    ),
                    mode=mode.value,
                    summary_mode="stronger",
                ),
                target_tokens=target_tokens,
            )
            steps.append(
                FallbackStep(
                    step=len(steps) + 1,
                    reason=retry.event.failure_reason or retry.event.status,
                    action="retry_l4_stronger_summary",
                    before_tokens=request.estimate_budget(current_programmatic.view).input_tokens,
                    after_tokens=retry.input_tokens,
                    status="success" if retry.event.status == "success" else "failed",
                    error=retry.event.failure_reason if retry.event.status != "success" else None,
                ).to_dict()
            )
            final_reason = None if retry.event.status == "success" else retry.event.failure_reason
            event = _with_fallback(retry.event, fallback_steps=steps, final_failure_reason=final_reason)
            self._record_l4_event(
                session_id=request.view.session_id,
                trigger=trigger,
                target_tokens=target_tokens,
                event=event,
            )
            if retry.event.status == "success":
                self._record_auto_success_if_needed(request=request, mode=mode)
            else:
                self._record_auto_failure_if_needed(
                    request=request,
                    mode=mode,
                    before_failure_count=before_failure_count,
                    failure_reason=final_reason or "failed",
                )
            return ContextCompactResult(
                status="success" if retry.event.status == "success" else "failed",
                reason=_result_reason(trigger=trigger, auto_reason=reason),
                view=retry.view,
                before_tokens=before_tokens,
                after_tokens=retry.input_tokens,
                programmatic_event=current_programmatic.event,
                l4_event=event,
                fallback_steps=steps,
                final_failure_reason=final_reason,
            )

        steps.append(
            FallbackStep(
                step=1,
                reason=reason,
                action=action,
                before_tokens=outcome.input_tokens,
                after_tokens=outcome.input_tokens,
                status="failed",
                error=reason,
            ).to_dict()
        )
        return self._final_l4_failure(
            request=request,
            trigger=trigger,
            mode=mode,
            target_tokens=target_tokens,
            programmatic=programmatic,
            before_tokens=before_tokens,
            after_tokens=outcome.input_tokens,
            before_failure_count=before_failure_count,
            event=_with_fallback(outcome.event, fallback_steps=steps, final_failure_reason=reason),
            reason=reason,
            fallback_steps=steps,
        )

    def _final_l4_failure(
        self,
        *,
        request: ContextCompactRequest,
        trigger: ContextWindowTrigger,
        mode: ContextCompactMode,
        target_tokens: int,
        programmatic: CompactionResult,
        before_tokens: int,
        after_tokens: int,
        before_failure_count: int,
        event: LlmCompactEvent,
        reason: str,
        fallback_steps: list[dict[str, object]] | None = None,
    ) -> ContextCompactResult:
        """所有手段 (含 fallback) 都失败后的收场: 记录最终失败事件并按需推进熔断计数.

        返回的视图保持程序化压缩后的状态而不是回滚, 因为 L1-L3 的改动已经作为事件落盘,
        且它们本身是有益的; 失败只意味着 L4 没补上, 不该把已省下的 token 还回去.
        """

        event = replace(event, final_failure_reason=reason)
        self._record_l4_event(
            session_id=request.view.session_id,
            trigger=trigger,
            target_tokens=target_tokens,
            event=event,
        )
        self._record_auto_failure_if_needed(
            request=request,
            mode=mode,
            before_failure_count=before_failure_count,
            failure_reason=reason,
        )
        return ContextCompactResult(
            status="failed",
            reason=reason,
            view=programmatic.view,
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            programmatic_event=programmatic.event,
            l4_event=event,
            fallback_steps=fallback_steps,
            final_failure_reason=reason,
        )

    def _unchanged(
        self,
        status: ManagerStatus,
        reason: str,
        request: ContextCompactRequest,
        tokens: int,
    ) -> ContextCompactResult:
        return ContextCompactResult(status, reason, request.view, tokens, tokens)

    def _record_auto_failure_if_needed(
        self,
        *,
        request: ContextCompactRequest,
        mode: ContextCompactMode,
        before_failure_count: int,
        failure_reason: str,
    ) -> None:
        if mode != ContextCompactMode.AUTO:
            return
        if request.runtime_state.auto_compact_failure_count > before_failure_count:
            return
        request.runtime_state.record_auto_compact_failure(failure_reason)

    def _record_auto_success_if_needed(
        self,
        *,
        request: ContextCompactRequest,
        mode: ContextCompactMode,
    ) -> None:
        if mode == ContextCompactMode.AUTO:
            request.runtime_state.record_auto_compact_success()

    def _record_programmatic_event(
        self,
        *,
        session_id: str,
        trigger: ContextWindowTrigger,
        target_tokens: int,
        event: CompactionEvent,
    ) -> None:
        SessionEventWriter(store=self.store, session_id=session_id).append_compaction_completed(
            trigger=trigger.value,
            target_tokens=target_tokens,
            event=event,
        )

    def _record_l4_event(
        self,
        *,
        session_id: str,
        trigger: ContextWindowTrigger,
        target_tokens: int,
        event: LlmCompactEvent,
    ) -> None:
        SessionEventWriter(store=self.store, session_id=session_id).append_llm_compaction_completed(
            trigger=trigger.value,
            target_tokens=target_tokens,
            event=event,
        )


def _target_tokens(request: ContextCompactRequest, trigger: ContextWindowTrigger) -> int:
    # 调用方显式指定
    if request.target_tokens is not None:
        return request.target_tokens
    # 任务切换之后, 压缩到low_watermark的2/3
    if trigger == ContextWindowTrigger.TASK_HASH_CHANGED:
        return max(1, request.budget.low_watermark * 2 // 3)
    # 其他情况, 压缩到low_watermark
    return request.budget.low_watermark


def _view_with_checkpoint(view: SessionView, checkpoint: Checkpoint) -> SessionView:
    return SessionView(
        session_id=view.session_id,
        messages=[AgentMessage.from_dict(message.to_dict()) for message in view.messages],
        checkpoints=[*view.checkpoints, Checkpoint.from_dict(checkpoint.to_dict())],
        metadata=dict(view.metadata),
        task_plan=view.task_plan,
    )


def _result_reason(*, trigger: ContextWindowTrigger, auto_reason: str) -> str:
    return auto_reason if trigger == ContextWindowTrigger.AUTO else trigger.value


def _force_route_current_text_for_trigger(trigger: ContextWindowTrigger) -> bool:
    return trigger in {ContextWindowTrigger.MANUAL, ContextWindowTrigger.PROMPT_TOO_LONG}


def _with_fallback(
    event: LlmCompactEvent,
    *,
    fallback_steps: list[dict[str, object]],
    final_failure_reason: str | None,
) -> LlmCompactEvent:
    return replace(
        event,
        fallback_steps=fallback_steps,
        final_failure_reason=final_failure_reason,
    )
