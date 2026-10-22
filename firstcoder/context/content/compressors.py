"""确定性内容压缩器。

这一层只做可预测的轻量替换，不调用 LLM。目标是先降低 token 占用，并用 metadata
记录压缩状态，避免后续 pipeline 对同一 part 反复处理。
"""

from __future__ import annotations

from typing import Any

from firstcoder.context.content.router import (
    RouteCompactResult,
    RouteContentType,
    RouteContext,
)
from firstcoder.context.identity import content_fingerprint
from firstcoder.context.models import MessagePart, utc_now_iso
from firstcoder.context.token_budget import estimate_text_tokens
from firstcoder.context.versions import COMPACTION_STRATEGY_VERSION


def compact_old_task_part(part: MessagePart) -> MessagePart:
    """生成 L1 修剪后的 part: 内容清空, 原样保留 id 与元数据。

    L1 是"刻意遗忘"而非摘要——原文仍留在 JSONL 里可查, 但有效视图中该 part
    不再有任何可见文本; ContextBuilder 对含此类 part 的尾部只输出一条
    汇总标记, 不再逐条投影原文。
    """

    metadata = _compacted_metadata(
        part,
        state="trimmed",
        compacted_by="l1_old_task_dialogue",
    )
    return MessagePart(
        id=part.id,
        message_id=part.message_id,
        kind=part.kind,
        content="",
        metadata=metadata,
    )


class PlainTextRouteCompressor:
    """工具派生输出的确定性兜底压缩器。

    专用搜索、差异、构建、JSON、代码及 HTML 压缩器不适用时，保留首尾内容和
    明确的词元元数据。仅供 L2 的工具结果路由调用，不用于普通对话或新读取的源文件。
    """

    def compact(self, part: MessagePart, context: RouteContext) -> RouteCompactResult | None:
        preview = part.content[: context.preview_chars]
        tail_preview = part.content[-context.preview_chars :] if len(part.content) > context.preview_chars else ""
        preview_tokens = estimate_text_tokens(preview)
        tail_preview_tokens = estimate_text_tokens(tail_preview) if tail_preview else 0
        content = "\n".join(
            [
                "[Derived tool result compacted]",
                f"part_id={part.id}",
                f"original_tokens={estimate_text_tokens(part.content)}",
                f"preview_tokens={preview_tokens}",
                f"preview={preview}",
                f"tail_preview_tokens={tail_preview_tokens}",
                f"tail_preview={tail_preview}",
            ]
        )
        return RouteCompactResult(
            content=content,
            content_type=RouteContentType.PLAIN_TEXT,
            compacted_by="l2_current_task_cold",
            metadata={
                "preview": preview,
                "preview_tokens": preview_tokens,
                "tail_preview": tail_preview,
                "tail_preview_tokens": tail_preview_tokens,
            },
        )


def _compacted_metadata(part: MessagePart, *, state: str, compacted_by: str) -> dict[str, Any]:
    metadata = dict(part.metadata)
    metadata.update(
        {
            "original_tokens": estimate_text_tokens(part.content),
            "content_fingerprint": content_fingerprint(part.content),
            "compaction_state": state,
            "compacted_by": compacted_by,
            "compacted_at": utc_now_iso(),
            "compaction_strategy_version": COMPACTION_STRATEGY_VERSION,
        }
    )
    return metadata
