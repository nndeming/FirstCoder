"""L4 LLM compact 的实现: LLM 交接摘要, 有损兜底, L1-L3 压不到目标才启用.

产出是 checkpoint 事件, 只追加进 append-only 的事实文件, 不改历史事件; resume 时由
ContextBuilder 把它投影成一条摘要消息. 这里只负责"生成候选 -> 校验边界 -> 提交落盘",
压缩时机由 ContextWindowManager 决定.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Literal

from firstcoder.context.checkpoint import Checkpoint, CheckpointIndex, checkpoint_summary_content
from firstcoder.context.events import SessionEvent
from firstcoder.context.identity import new_event_id, stable_json_hash
from firstcoder.context.models import AgentMessage, MessagePart, SessionView
from firstcoder.context.retry_policy import CompactRetryPolicy
from firstcoder.context.runtime_state import SessionRuntimeState, auto_compact_circuit_is_open
from firstcoder.context.store import JsonlSessionStore
from firstcoder.context.tool_sequence import InvalidToolCallSequenceError, validate_tool_call_sequence
from firstcoder.context.versions import CHECKPOINT_STRATEGY_VERSION

CompactMode = Literal["auto", "manual"]


CODING_HANDOFF_HEADINGS: tuple[str, ...] = (
    "## 当前目标",
    "## 已知事实与硬约束",
    "## 已确认的决定及理由",
    "## 相关文件与当前实现状态",
    "## 已运行命令及有效结果",
    "## 当前错误与未解决事项",
    "## 下一步（可立即执行）",
)


class PromptTooLongError(RuntimeError):
    pass


class CompactTimeoutError(RuntimeError):
    pass


class NoSummaryError(RuntimeError):
    pass


class InvalidLlmCheckpointBoundaryError(ValueError):
    """L4 summarizer 返回的 checkpoint 边界会破坏 resume 投影。"""


class UnconsumedLlmCheckpointBoundaryError(InvalidLlmCheckpointBoundaryError):
    """L4 边界会把模型还没消费过的 tool result 藏进摘要里, 违反"没看过的不能压"红线."""


class LlmSourceFingerprintMismatchError(ValueError):
    """调用方传入的 expected source fingerprint 与当前 view 不一致。"""


@dataclass(frozen=True, slots=True)
class LlmCompactSummary:
    summary: str
    tail_start_message_id: str
    covered_until_message_id: str


class LlmCompactSummarizer(Protocol):
    """摘要生成器协议。

    真实实现后续可以适配任意 provider；当前上下文层只依赖这个窄协议，避免把 OpenAI、
    Anthropic 等外部消息格式提前泄漏进 checkpoint 写入逻辑。
    """

    def summarize(self, messages: list[AgentMessage], *, summary_mode: str = "default") -> LlmCompactSummary: ...


@dataclass(slots=True)
class LlmCompactRequest:
    view: SessionView
    runtime_state: SessionRuntimeState
    consumed_tool_result_part_ids: frozenset[str]
    mode: CompactMode = "auto"
    expected_source_fingerprint: str | None = None
    summary_mode: str = "default"


@dataclass(frozen=True, slots=True)
class LlmCompactEvent:
    status: Literal["success", "failed", "skipped"]
    source_fingerprint: str
    retry_count: int = 0
    failure_reason: str | None = None
    checkpoint_id: str | None = None
    fallback_steps: list[dict[str, object]] | None = None
    final_failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class LlmCompactCandidate:
    checkpoint: Checkpoint | None
    event: LlmCompactEvent


@dataclass(slots=True)
class LlmCompactService:
    """L4 压缩服务: 调 summarizer 生成 checkpoint 候选, 校验边界后才允许提交落盘."""

    store: JsonlSessionStore
    summarizer: LlmCompactSummarizer
    retry_policy: CompactRetryPolicy = CompactRetryPolicy()
    auto_failure_limit: int = 3

    def generate_candidate(self, request: LlmCompactRequest) -> LlmCompactCandidate:
        """让 LLM 生成压缩候选 checkpoint, 只生成不落盘, 失败按策略重试.

        入口先过三道快速返回: 期望指纹校验, 重复源跳过, 熔断器跳过. 之后进入
        重试循环, 只有可重试的错误 (prompt 过长, 超时, 无摘要) 才会重试,
        边界类错误 (unconsumed_boundary, invalid_tool_sequence) 直接失败.
        """

        # 构造 L4 源: 只含会话消息 (不含 system prompt 和工具 schema), 并计算源指纹
        source = _build_l4_source(request.view)
        source_messages = source.messages
        source_fingerprint = _source_fingerprint(request.view.session_id, source)
        # 调用方传了期望指纹且不匹配时抛异常, 防止基于过期视图生成摘要
        if request.expected_source_fingerprint and request.expected_source_fingerprint != source_fingerprint:
            raise LlmSourceFingerprintMismatchError(
                "expected_source_fingerprint does not match current L4 source",
            )

        # 源与上次压缩时完全相同, 再压一遍不会产生新内容, 直接跳过
        if request.runtime_state.last_compaction_input_fingerprint == source_fingerprint:
            return LlmCompactCandidate(
                checkpoint=None,
                event=LlmCompactEvent(
                    status="skipped",
                    source_fingerprint=source_fingerprint,
                    failure_reason="duplicate_source",
                ),
            )

        # auto 模式下熔断器开着时直接跳过, 避免连续失败烧 token
        if request.mode == "auto" and auto_compact_circuit_is_open(request.runtime_state):
            return LlmCompactCandidate(
                checkpoint=None,
                event=LlmCompactEvent(
                    status="skipped",
                    source_fingerprint=source_fingerprint,
                    failure_reason="circuit_open",
                ),
            )

        attempts = 0
        retries = 0
        # 重试循环: 摘要 + 边界校验 + 构造 checkpoint, 成功即返回
        while True:
            attempts += 1
            try:
                summary = _summarize(
                    self.summarizer,    # 这里会传入系统提示词
                    source_messages,
                    summary_mode=request.summary_mode,
                )
                _validate_summary_boundary(
                    summary,
                    source=source,
                    consumed_tool_result_part_ids=request.consumed_tool_result_part_ids,
                )
                checkpoint = _candidate_checkpoint(
                    request.view,
                    summary=summary,
                    source=source,
                    source_fingerprint=source_fingerprint,
                    retry_count=retries,
                )
                return LlmCompactCandidate(
                    checkpoint=checkpoint,
                    event=LlmCompactEvent(
                        status="success",
                        source_fingerprint=source_fingerprint,
                        retry_count=retries,
                        checkpoint_id=checkpoint.id,
                    ),
                )
            # 边界类错误不可重试: 重试只会得到同样的边界, 直接失败
            except UnconsumedLlmCheckpointBoundaryError:
                return _failed_candidate(source_fingerprint, retries, "unconsumed_boundary")
            except InvalidLlmCheckpointBoundaryError:
                return _failed_candidate(source_fingerprint, retries, "invalid_tool_sequence")
            # 可重试错误: 按重试策略决定是否再来一轮
            except (PromptTooLongError, CompactTimeoutError, NoSummaryError) as error:
                reason = _failure_reason(error)
                decision = self.retry_policy.decide(reason, attempt=attempts)
                if not decision.should_retry:
                    return _failed_candidate(source_fingerprint, retries, reason)
                retries += 1

    def commit_candidate(
        self,
        candidate: LlmCompactCandidate,
        *,
        runtime_state: SessionRuntimeState,
    ) -> Checkpoint:
        checkpoint = candidate.checkpoint
        if checkpoint is None or candidate.event.status != "success":
            raise ValueError("only successful L4 candidates can be committed")
        self.store.append_event(
            SessionEvent(
                id=new_event_id(),
                session_id=checkpoint.session_id,
                type="checkpoint_created",
                payload=checkpoint.to_dict(),
            )
        )
        runtime_state.latest_checkpoint_id = checkpoint.id
        runtime_state.last_compaction_input_fingerprint = candidate.event.source_fingerprint
        return checkpoint


@dataclass(frozen=True, slots=True)
class L4Source:
    messages: list[AgentMessage]
    base_checkpoint_id: str | None = None
    tail_message_ids: tuple[str, ...] = ()


def _conversation_messages_only(view: SessionView) -> list[AgentMessage]:
    """L4 摘要只看会话历史。

    system prompt、工具 schema 和 provider 能力属于 stable prefix/cache 输入，不属于可被 LLM
    总结折叠的历史。如果把它们混入 summary，resume 时容易污染系统提示词保护边界。
    """

    return [message for message in view.messages if message.role != "system_meta"]


def _build_l4_source(view: SessionView) -> L4Source:
    messages = _conversation_messages_only(view)
    checkpoint = CheckpointIndex(view.checkpoints).latest()
    if checkpoint is None:
        return L4Source(messages=messages, tail_message_ids=tuple(message.id for message in messages))

    for index, message in enumerate(messages):
        if message.id == checkpoint.tail_start_message_id:
            tail = messages[index:]
            return L4Source(
                messages=[_checkpoint_summary_message(view.session_id, checkpoint), *tail],
                base_checkpoint_id=checkpoint.id,
                tail_message_ids=tuple(message.id for message in tail),
            )
    raise InvalidLlmCheckpointBoundaryError(
        f"latest checkpoint tail_start_message_id not found: {checkpoint.tail_start_message_id}",
    )


def _checkpoint_summary_message(session_id: str, checkpoint: Checkpoint) -> AgentMessage:
    message_id = f"{checkpoint.id}_summary"
    return AgentMessage(
        id=message_id,
        session_id=session_id,
        role="user",
        parts=[
            MessagePart(
                id=f"part_{message_id}",
                message_id=message_id,
                kind="checkpoint_summary",
                content=checkpoint_summary_content(checkpoint),
                metadata={"checkpoint_id": checkpoint.id},
            )
        ],
        created_at=checkpoint.created_at,
        metadata={"checkpoint_id": checkpoint.id, "synthetic": True},
    )


def _validate_summary_boundary(
    summary: LlmCompactSummary,
    *,
    source: L4Source,
    consumed_tool_result_part_ids: frozenset[str],
) -> None:
    if source.base_checkpoint_id is None:
        valid_ids = {message.id for message in source.messages}
    else:
        valid_ids = set(source.tail_message_ids)

    if summary.tail_start_message_id not in valid_ids:
        raise InvalidLlmCheckpointBoundaryError(
            "tail_start_message_id must stay within current L4 input tail",
        )
    if summary.covered_until_message_id not in valid_ids:
        raise InvalidLlmCheckpointBoundaryError(
            "covered_until_message_id must stay within current L4 input tail",
        )

    tail_order = {message_id: index for index, message_id in enumerate(source.tail_message_ids)}
    if tail_order[summary.covered_until_message_id] >= tail_order[summary.tail_start_message_id]:
        raise InvalidLlmCheckpointBoundaryError(
            "covered_until_message_id must be before tail_start_message_id",
        )

    tail_messages = _source_tail_messages(source)
    tail_start_index = tail_order[summary.tail_start_message_id]
    try:
        validate_tool_call_sequence(tail_messages[tail_start_index:])
    except InvalidToolCallSequenceError as error:
        raise InvalidLlmCheckpointBoundaryError(
            "checkpoint tail would break assistant tool_call/tool_result sequence",
        ) from error

    earliest_protected = _earliest_unconsumed_transaction_index(
        tail_messages,
        consumed_tool_result_part_ids=consumed_tool_result_part_ids,
    )
    if earliest_protected is not None and tail_start_index > earliest_protected:
        raise UnconsumedLlmCheckpointBoundaryError(
            "checkpoint tail would cover an unconsumed tool transaction",
        )


def _earliest_unconsumed_transaction_index(
    messages: list[AgentMessage],
    *,
    consumed_tool_result_part_ids: frozenset[str],
) -> int | None:
    assistant_by_call_id = {
        str(part.metadata.get("tool_call_id")): index
        for index, message in enumerate(messages)
        if message.role == "assistant"
        for part in message.parts
        if part.kind == "tool_call" and part.metadata.get("tool_call_id")
    }
    earliest: int | None = None
    for index, message in enumerate(messages):
        if message.role != "tool":
            continue
        for part in message.parts:
            if part.kind != "tool_result" or part.id in consumed_tool_result_part_ids:
                continue
            start = assistant_by_call_id.get(str(part.metadata.get("tool_call_id")), index)
            earliest = start if earliest is None else min(earliest, start)
    return earliest


def _source_tail_messages(source: L4Source) -> list[AgentMessage]:
    if source.base_checkpoint_id is None:
        return source.messages
    tail_ids = set(source.tail_message_ids)
    return [message for message in source.messages if message.id in tail_ids]


def _source_fingerprint(session_id: str, source: L4Source) -> str:
    return stable_json_hash(
        {
            "session_id": session_id,
            "strategy_version": CHECKPOINT_STRATEGY_VERSION,
            "base_checkpoint_id": source.base_checkpoint_id,
            "tail_message_ids": list(source.tail_message_ids),
            "messages": [message.to_dict() for message in source.messages],
        },
        length=24,
    )


def _candidate_checkpoint(
    view: SessionView,
    *,
    summary: LlmCompactSummary,
    source: L4Source,
    source_fingerprint: str,
    retry_count: int,
) -> Checkpoint:
    return Checkpoint(
        id="",
        session_id=view.session_id,
        summary=summary.summary,
        tail_start_message_id=summary.tail_start_message_id,
        covered_until_message_id=summary.covered_until_message_id,
        source_fingerprint=source_fingerprint,
        sequence=max((checkpoint.sequence for checkpoint in view.checkpoints), default=0) + 1,
        metadata={
            "created_by": "l4_llm_compact",
            "summary_prompt_scope": "conversation_history_only",
            "retry_count": retry_count,
            "base_checkpoint_id": source.base_checkpoint_id,
            "source_message_ids": [message.id for message in source.messages],
        },
    )


def _failed_candidate(
    source_fingerprint: str,
    retry_count: int,
    failure_reason: str,
) -> LlmCompactCandidate:
    return LlmCompactCandidate(
        checkpoint=None,
        event=LlmCompactEvent(
            status="failed",
            source_fingerprint=source_fingerprint,
            retry_count=retry_count,
            failure_reason=failure_reason,
        ),
    )


def _summarize(
    summarizer: LlmCompactSummarizer,
    messages: list[AgentMessage],
    *,
    summary_mode: str,
) -> LlmCompactSummary:
    summary = summarizer.summarize(messages, summary_mode=summary_mode)
    return LlmCompactSummary(
        summary=normalize_coding_handoff(summary.summary),
        tail_start_message_id=summary.tail_start_message_id,
        covered_until_message_id=summary.covered_until_message_id,
    )


def normalize_coding_handoff(summary: str) -> str:
    """把 provider 的输出规整成固定的 L4 交接摘要格式.

    模型只负责给散文, checkpoint 的公开结构由本地代码兜底: 命中的小节保留正文
    (重复小节取后出现的), 缺失小节显式标 `无`, 无法识别的 Markdown 标题降级为
    正文, 保证产出的交接摘要恰好包含七个固定小节各一次.
    """

    bodies: dict[str, list[str]] = {heading: [] for heading in CODING_HANDOFF_HEADINGS}
    current: str | None = None
    preamble: list[str] = []
    for line in summary.strip().splitlines():
        heading = line.strip()
        if heading in bodies:
            current = heading
            continue
        if heading.startswith("##"):
            # Do not emit an extra heading into a checkpoint whose schema is
            # deliberately fixed. Keep the model's information as body text.
            line = heading.lstrip("#").strip()
        if current is None:
            preamble.append(line)
        else:
            bodies[current].append(line)

    if preamble:
        bodies[CODING_HANDOFF_HEADINGS[0]].extend(preamble)

    sections: list[str] = []
    for heading in CODING_HANDOFF_HEADINGS:
        body = "\n".join(bodies[heading]).strip()
        sections.append(f"{heading}\n{body or '无'}")
    return "\n\n".join(sections)


def _failure_reason(error: Exception) -> str:
    if isinstance(error, PromptTooLongError):
        return "prompt_too_long"
    if isinstance(error, CompactTimeoutError):
        return "timeout"
    if isinstance(error, NoSummaryError):
        return "no_summary"
    return "provider_error"
