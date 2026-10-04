"""上下文 token 预算的集中估算。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from firstcoder.context.budget_defaults import DEFAULT_CONTEXT_WINDOW, DEFAULT_OUTPUT_RESERVE
from firstcoder.providers.types import ChatMessage, ToolDefinition


IMAGE_INPUT_TOKEN_ESTIMATE = 1_024


@dataclass(frozen=True, slots=True)
class ContextBudget:
    context_window: int
    output_reserve: int
    input_capacity: int
    fixed_tokens: int
    history_tokens: int
    input_tokens: int
    high_watermark: int
    low_watermark: int
    source: Literal["configured", "assumed"]


def estimate_text_tokens(text: str) -> int:
    """第一版使用字符数近似 token。

    这里有意不绑定具体 tokenizer, 避免 context 层过早依赖 provider。
    """

    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def build_context_budget(
    *,
    messages: list[ChatMessage],
    tools: list[ToolDefinition],
    context_window: int | None,
    max_output_tokens: int | None,
) -> ContextBudget:
    """给一次待发的请求算 token 账本, 产出压缩决策要用的预算表。

    输入侧可用量 = 窗口打95折(留估算误差的安全垫) - 给模型回复的预留;
    system 消息 + 工具定义记为 fixed(每次必带, 不可压缩), 其余记为 history(可压缩);
    high_watermark(90%容量)是触发压缩的高压线, low_watermark(72%)是压缩的目标线,
    两档拉开是为了压完留出增长余量, 避免刚压完又超线的抖动。
    """

    # 上下文窗口总大小, 目前默认200K(token数)
    resolved_window = DEFAULT_CONTEXT_WINDOW if context_window is None else context_window
    # 给模型回复预留的输出空间(token 数)
    output_reserve = DEFAULT_OUTPUT_RESERVE if max_output_tokens is None else max_output_tokens
    if resolved_window <= 0 or output_reserve <= 0:
        raise ValueError("context window and output reserve must be positive")

    # 留5%是给估算误差准备的缓冲带, 这一版近似4个字符≈1个token
    usable_window = int(resolved_window * 0.95)
    input_capacity = usable_window - output_reserve
    if input_capacity <= 0:
        raise ValueError("output reserve must be smaller than usable context window")

    # 粗略估算token
    fixed_tokens = sum(_estimate_chat_message_tokens(message) for message in messages if message.role == "system")
    fixed_tokens += _estimate_tool_definition_tokens(tools)
    history_tokens = sum(_estimate_chat_message_tokens(message) for message in messages if message.role != "system")
    high_watermark = int(input_capacity * 0.90)
    low_watermark = int(input_capacity * 0.72)
    if low_watermark >= high_watermark:
        raise ValueError("low watermark must be below high watermark")

    return ContextBudget(
        context_window=resolved_window,
        output_reserve=output_reserve,
        input_capacity=input_capacity,
        fixed_tokens=fixed_tokens,
        history_tokens=history_tokens,
        input_tokens=fixed_tokens + history_tokens,
        high_watermark=high_watermark,
        low_watermark=low_watermark,
        source="configured" if context_window is not None else "assumed",
    )


def _estimate_chat_message_tokens(message: ChatMessage) -> int:
    tokens = estimate_text_tokens(message.content)
    tokens += estimate_text_tokens(message.name or "")
    tokens += estimate_text_tokens(message.tool_call_id or "")
    tokens += sum(
        estimate_text_tokens(call.name + json.dumps(call.arguments, ensure_ascii=False, sort_keys=True))
        for call in message.tool_calls
    )
    tokens += sum(
        IMAGE_INPUT_TOKEN_ESTIMATE
        for part in message.content_parts or []
        if part.type == "image"
    )
    return tokens


def _estimate_tool_definition_tokens(tools: list[ToolDefinition]) -> int:
    return sum(
        estimate_text_tokens(
            json.dumps(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        for tool in tools
    )
