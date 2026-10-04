"""L2 内容路由压缩框架。

这一层只负责“识别内容类型 -> 分发到对应压缩器 -> 验证压缩收益 -> 统一写 metadata”。
具体的 search、diff、build、json、code、html 算法会按第 14 步逐个补齐，避免把
路由边界和具体压缩策略耦合在一起。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from firstcoder.context.identity import content_fingerprint
from firstcoder.context.models import MessagePart, utc_now_iso
from firstcoder.context.token_budget import estimate_text_tokens
from firstcoder.context.versions import COMPACTION_STRATEGY_VERSION


class RouteContentType(str, Enum):
    SEARCH_RESULTS = "search_results"
    GIT_DIFF = "git_diff"
    BUILD_OUTPUT = "build_output"
    JSON_ARRAY = "json_array"
    JSON_OBJECT = "json_object"
    SOURCE_CODE = "source_code"
    HTML = "html"
    PLAIN_TEXT = "plain_text"


@dataclass(slots=True)
class RouteDetection:
    content_type: RouteContentType
    confidence: float
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class RouteContext:
    detection: RouteDetection
    preview_chars: int = 160


@dataclass(slots=True)
class RouteCompactResult:
    content: str
    content_type: RouteContentType
    compacted_by: str
    metadata: dict[str, object] = field(default_factory=dict)


class RouteCompressor(Protocol):
    def compact(self, part: MessagePart, context: RouteContext) -> RouteCompactResult | None:
        """返回压缩结果；不适合压缩时返回 None。"""


@dataclass(slots=True)
class RouteCompactRouter:
    compressors: dict[RouteContentType, RouteCompressor] = field(default_factory=dict)
    min_original_tokens: int = 40
    preview_chars: int = 160

    def compact_part(self, part: MessagePart) -> MessagePart | None:
        """路由压缩入口: 对单个 tool_result part 尝试按内容类型压缩, 不适合就返回 None。

        三种"不适合"的情况: 原文太小(min_original_tokens 以下)、没有可用的
        压缩器、压缩结果没有真的变小。成功时返回新 part -- id/message_id/kind
        保持不变, 只替换 content 并补充溯源 metadata。
        """

        # 原文太小就不值得压
        original_tokens = estimate_text_tokens(part.content)
        if original_tokens < self.min_original_tokens:
            return None

        # 探测内容类型(grep 输出/搜索列表/diff/纯文本...), tool_name 作为辅助提示
        detection = detect_route_content_type(part.content, tool_name=_tool_name(part))
        route_content_type = detection.content_type
        # 按探测到的类型找专用压缩器, 没有就回退到纯文本压缩器
        compressor = self.compressors.get(route_content_type)
        fallback_from: RouteContentType | None = None
        if compressor is None and route_content_type is not RouteContentType.PLAIN_TEXT:
            compressor = self.compressors.get(RouteContentType.PLAIN_TEXT)
            if compressor is not None:
                fallback_from = route_content_type
        if compressor is None:
            return None

        route_result = compressor.compact(part, RouteContext(detection=detection, preview_chars=self.preview_chars))
        if route_result is None:
            return None

        # 压缩后必须真的变小, 没变小就等于白压
        replacement_tokens = estimate_text_tokens(route_result.content)
        if replacement_tokens >= original_tokens:
            return None

        # 补溯源信息: 指纹校验内容, 压缩器自己还可以再追加额外字段
        metadata = dict(part.metadata)
        metadata.update(
            {
                "original_tokens": original_tokens,
                "replacement_tokens": replacement_tokens,
                "content_fingerprint": content_fingerprint(part.content),
                "compaction_state": "route_compacted",
                "compacted_by": route_result.compacted_by,
                "compacted_at": utc_now_iso(),
                "compaction_strategy_version": COMPACTION_STRATEGY_VERSION,
                "content_type": route_result.content_type.value,
                "detected_content_type": route_content_type.value,
                "route_confidence": detection.confidence,
                "route_metadata": detection.metadata,
            }
        )
        if fallback_from is not None:
            metadata["route_fallback_from"] = fallback_from.value
        metadata.update(route_result.metadata)

        return MessagePart(
            id=part.id,
            message_id=part.message_id,
            kind=part.kind,
            content=route_result.content,
            metadata=metadata,
        )


_SEARCH_RESULT_PATTERN = re.compile(r"^[^\s:][^:\n]*:\d+:", re.MULTILINE)
_DIFF_HEADER_PATTERN = re.compile(r"^(diff --git|--- a/|\+\+\+ b/|@@\s+-\d+)", re.MULTILINE)
_BUILD_OUTPUT_PATTERN = re.compile(
    r"(FAILED|ERROR|Traceback \(most recent call last\)|pytest|npm ERR!|cargo test|warning:)",
    re.IGNORECASE,
)
_HTML_PATTERN = re.compile(r"<!doctype\s+html|<html[\s>]|<body[\s>]", re.IGNORECASE)
_CODE_PATTERN = re.compile(
    r"^\s*(def|class|import|from|function|const|let|export|interface|type|fn|struct|impl|package)\b",
    re.MULTILINE,
)


def detect_route_content_type(content: str, *, tool_name: str | None = None) -> RouteDetection:
    """探测 tool_result 的内容类型, 供路由阶段选择对应的压缩器。

    本函数只做分类, 不做压缩: 输出的 content_type(8 选 1) 交给 compact_part
    去 compressors 字典查对应的专用压缩器, 压缩器里写死了该类型的保留/删除规则。

    判定流程为级联, 命中即停:
    1. 工具名提示(grep/rg -> 搜索结果, git_diff/diff -> diff);
    2. JSON 实际解析(成功即 JSON 数组/对象);
    3. 正则特征级联(diff 头 -> HTML -> 文件:行号 -> 代码 -> 构建输出,
       按特征具体程度排序, 越具体越靠前);
    4. 全部未命中 -> 纯文本。

    confidence 是每条判定规则手工标定的证据强度(1.0 JSON 解析 / 0.95 工具名 /
    0.85~0.6 正则 / 0.5 默认), 不随文本变化, 不是概率; 目前仅随压缩 metadata
    落盘留痕, 不参与路由决策。
    """

    stripped = content.strip()
    if not stripped:
        # 空内容没有压缩价值, 类型无所谓, 置信度给 0
        return RouteDetection(RouteContentType.PLAIN_TEXT, 0.0)

    # 工具名是最强信号: grep/rg 的输出基本可判定为搜索结果, 无需再检查内容
    tool_hint = (tool_name or "").lower()
    if tool_hint in {"grep", "rg"}:
        return RouteDetection(RouteContentType.SEARCH_RESULTS, 0.95, {"source": "tool_hint"})
    if tool_hint in {"git_diff", "diff"}:
        return RouteDetection(RouteContentType.GIT_DIFF, 0.95, {"source": "tool_hint"})

    # JSON 通过实际解析判定, 解析成功即完全确定
    json_detection = _detect_json(stripped)
    if json_detection is not None:
        return json_detection

    # 正则级联: 特征越具体越靠前(置信度也越高); 代码特征最泛, 放最后且置信度最低
    if _DIFF_HEADER_PATTERN.search(stripped):
        return RouteDetection(RouteContentType.GIT_DIFF, 0.85)
    if _HTML_PATTERN.search(stripped[:3000]):
        return RouteDetection(RouteContentType.HTML, 0.85)
    if _SEARCH_RESULT_PATTERN.search(stripped):
        return RouteDetection(RouteContentType.SEARCH_RESULTS, 0.8)
    if _CODE_PATTERN.search(stripped):
        return RouteDetection(RouteContentType.SOURCE_CODE, 0.6)
    # 构建输出要凑"工具名 + 内容特征"两个信号, 比只中内容特征的更可信
    if tool_hint in {"shell", "pytest"} and _BUILD_OUTPUT_PATTERN.search(stripped):
        return RouteDetection(RouteContentType.BUILD_OUTPUT, 0.75, {"source": "tool_hint"})
    if _BUILD_OUTPUT_PATTERN.search(stripped):
        return RouteDetection(RouteContentType.BUILD_OUTPUT, 0.65)
    return RouteDetection(RouteContentType.PLAIN_TEXT, 0.5)


def _detect_json(content: str) -> RouteDetection | None:
    if not content.startswith(("[", "{")):
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, list):
        return RouteDetection(RouteContentType.JSON_ARRAY, 1.0, {"item_count": len(parsed)})
    if isinstance(parsed, dict):
        return RouteDetection(RouteContentType.JSON_OBJECT, 1.0, {"keys": list(parsed.keys())[:20]})
    return None


def _tool_name(part: MessagePart) -> str | None:
    value = part.metadata.get("tool_name")
    return str(value) if value is not None else None
