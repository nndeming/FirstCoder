"""有效尾部 tool_result 的生命周期分类.

纯函数模块: 只解读内置读取/修改类工具成功结果里的结构化数据 (参数, 路径, 内容指纹),
不读正文猜语义, 不碰文件系统, 也不改会话状态, 因此调用方可以在重放后安全地重建同一份索引.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
from pathlib import PureWindowsPath
from typing import Any, cast

from firstcoder.context.models import AgentMessage, MessagePart


class ToolResultLifecycle(StrEnum):
    FRESH = "fresh"            # 新鲜结果: 任何压缩层都不碰; 失败/未知结果一律按此处理 (fail-open)
    STALE = "stale"            # 读取之后源文件又被改过, 内容不再可信, 属垃圾级必须清理
    SUPERSEDED = "superseded"  # 同一目标的读取被更新的读取覆盖, 属垃圾级必须清理
    DERIVED = "derived"        # 工具派生输出 (非源码读取), 是 L2 路由压缩和 L3 可选清理的对象
    DUPLICATE = "duplicate"    # 内容指纹与更新的 derived 结果重复, 属垃圾级必须清理


@dataclass(frozen=True, slots=True)
class ToolResultLifecycleRecord:
    message_id: str
    part_id: str
    lifecycle: ToolResultLifecycle
    reason: str
    content_fingerprint: str
    duplicate_of_part_id: str | None = None
    source_targets: tuple[SourceReadTarget, ...] = ()


@dataclass(frozen=True, slots=True)
class SourceReadTarget:
    path: str
    start_line: int | None
    end_line: int | None
    is_full_file: bool = False


@dataclass(frozen=True, slots=True)
class _ToolCall:
    name: str
    arguments: dict[str, Any]


def index_tool_result_lifecycles(
    messages: list[AgentMessage],
    *,
    current_turn: int | None = None,
) -> dict[tuple[str, str], ToolResultLifecycleRecord]:
    """为尾部消息里的每个`tool_result`标注生命周期状态, 以 (message_id, part_id) 为键返回索引。

    分类只看确定性的工具时间线(配对、消费、派生关系), 拿不准的一律按 fresh 处理。
    """

    del current_turn
    tool_calls = _index_tool_calls(messages)
    records: dict[tuple[str, str], ToolResultLifecycleRecord] = {}
    source_reads: list[tuple[tuple[str, str], SourceReadTarget]] = []
    derived_parts: list[tuple[str, str]] = []

    for message in messages:
        if message.role != "tool":
            continue
        for part in message.parts:
            if part.kind != "tool_result":
                continue

            key = (message.id, part.id)
            call = _call_for_result(part, tool_calls)
            tool_name = call.name if call is not None else None
            fingerprint = _content_sha256(part.content)
            lifecycle = ToolResultLifecycle.DERIVED
            reason = "derived_tool_output"
            targets: tuple[SourceReadTarget, ...] = ()

            # fail-open: 失败结果既不能证明读过也不能证明改过, 即使内容与别的输出
            # 恰好相同也必须按 fresh 保留, 宁可不压也不误删.
            if part.metadata.get("ok") is not True:
                lifecycle = ToolResultLifecycle.FRESH
                reason = "failed_or_unknown_result"
            elif tool_name == "view" and call is not None:
                lifecycle = ToolResultLifecycle.FRESH
                reason = "view_source_read"
                target = _view_target(part.metadata.get("data"), call.arguments)
                if target is not None:
                    targets = (target,)
            elif tool_name == "read_multi":
                lifecycle = ToolResultLifecycle.FRESH
                reason = "read_multi_source_read"
                targets = _read_multi_targets(part.metadata.get("data"))

            records[key] = ToolResultLifecycleRecord(
                message_id=message.id,
                part_id=part.id,
                lifecycle=lifecycle,
                reason=reason,
                content_fingerprint=fingerprint,
                source_targets=targets,
            )

            if targets:
                for target in targets:
                    source_reads.append((key, target))
                _supersede_covered_reads(records, source_reads, key, targets)
            elif lifecycle is ToolResultLifecycle.DERIVED:
                derived_parts.append(key)

            if tool_name in {"write", "edit", "delete", "apply_patch"} and part.metadata.get("ok") is True:
                for path in _mutation_paths(tool_name, part.metadata.get("data")):
                    _stale_matching_reads(records, source_reads, path)

    _mark_derived_duplicates(records, derived_parts)
    return records


def _index_tool_calls(messages: list[AgentMessage]) -> dict[str, _ToolCall | None]:
    calls: dict[str, _ToolCall | None] = {}
    for message in messages:
        if message.role != "assistant":
            continue
        for part in message.parts:
            if part.kind != "tool_call":
                continue
            call_id = part.metadata.get("tool_call_id")
            name = part.metadata.get("tool_name")
            arguments = part.metadata.get("arguments")
            if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
                continue
            if not isinstance(arguments, dict):
                arguments = {}
            if call_id in calls:
                calls[call_id] = None
            else:
                calls[call_id] = _ToolCall(name=name, arguments=arguments)
    return calls


def _call_for_result(part: MessagePart, calls: dict[str, _ToolCall | None]) -> _ToolCall | None:
    call_id = part.metadata.get("tool_call_id")
    if not isinstance(call_id, str) or not call_id:
        return None
    call = calls.get(call_id)
    result_name = part.metadata.get("tool_name")
    if call is None or (isinstance(result_name, str) and result_name and result_name != call.name):
        return None
    return call


def _view_target(data: object, arguments: dict[str, Any]) -> SourceReadTarget | None:
    if not isinstance(data, dict):
        return None
    path = _normalize_path(data.get("path"))
    if path is None:
        return None

    start_line = data.get("start_line")
    end_line = data.get("end_line")
    if "start_line" in data or "end_line" in data:
        if not _is_line_range(start_line, end_line):
            return None
    else:
        offset = arguments.get("offset", 0)
        limit = arguments.get("limit")
        if isinstance(offset, int) and not isinstance(offset, bool) and offset >= 0 and isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
            start_line, end_line = offset + 1, offset + limit
        else:
            return None

    total_lines = data.get("total_lines")
    is_complete_view = data.get("truncated") is False and isinstance(total_lines, int) and not isinstance(total_lines, bool) and isinstance(end_line, int) and not isinstance(end_line, bool) and total_lines >= end_line
    if is_complete_view:
        end_line = total_lines
    return SourceReadTarget(
        path=path,
        start_line=start_line,
        end_line=end_line,
        is_full_file=is_complete_view and start_line == 1 and end_line == total_lines,
    )


def _read_multi_targets(data: object) -> tuple[SourceReadTarget, ...]:
    if not isinstance(data, dict) or data.get("truncated") is not False:
        return ()
    files = data.get("files")
    if not isinstance(files, list) or not files:
        return ()
    paths: list[str] = []
    for file_data in files:
        if not isinstance(file_data, dict):
            return ()
        path = _normalize_path(file_data.get("path"))
        if path is None:
            return ()
        paths.append(path)
    return tuple(SourceReadTarget(path=path, start_line=None, end_line=None, is_full_file=True) for path in paths)


def _mutation_paths(tool_name: str, data: object) -> tuple[str, ...]:
    if not isinstance(data, dict) or data.get("dry_run") is True:
        return ()
    if tool_name != "apply_patch":
        path = _normalize_path(data.get("path"))
        return (path,) if path is not None else ()

    paths: list[str] = []
    for key in ("changed_files", "deleted_files"):
        values = data.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            path = _normalize_path(value)
            if path is not None:
                paths.append(path)
    moved_files = data.get("moved_files")
    if isinstance(moved_files, list):
        for move in moved_files:
            if not isinstance(move, dict):
                continue
            for key in ("source", "destination"):
                path = _normalize_path(move.get(key))
                if path is not None:
                    paths.append(path)
    return tuple(dict.fromkeys(paths))


def _stale_matching_reads(
    records: dict[tuple[str, str], ToolResultLifecycleRecord],
    source_reads: list[tuple[tuple[str, str], SourceReadTarget]],
    path: str,
) -> None:
    for key, target in source_reads:
        if target.path != path:
            continue
        record = records[key]
        if record.lifecycle is ToolResultLifecycle.SUPERSEDED:
            continue
        records[key] = _replace_lifecycle(record, ToolResultLifecycle.STALE, "source_mutated")


def _supersede_covered_reads(
    records: dict[tuple[str, str], ToolResultLifecycleRecord],
    source_reads: list[tuple[tuple[str, str], SourceReadTarget]],
    current_key: tuple[str, str],
    current_targets: tuple[SourceReadTarget, ...],
) -> None:
    for key, earlier in source_reads:
        if key == current_key:
            continue
        if any(_covers(later, earlier) for later in current_targets):
            records[key] = _replace_lifecycle(records[key], ToolResultLifecycle.SUPERSEDED, "source_range_reread")


def _covers(later: SourceReadTarget, earlier: SourceReadTarget) -> bool:
    if later.path != earlier.path:
        return False
    if later.is_full_file:
        return True
    if earlier.is_full_file or later.start_line is None or later.end_line is None:
        return False
    return later.start_line <= cast(int, earlier.start_line) and later.end_line >= cast(int, earlier.end_line)


def _mark_derived_duplicates(
    records: dict[tuple[str, str], ToolResultLifecycleRecord],
    derived_parts: list[tuple[str, str]],
) -> None:
    latest_by_fingerprint: dict[str, tuple[str, str]] = {}
    for key in reversed(derived_parts):
        record = records[key]
        newer = latest_by_fingerprint.get(record.content_fingerprint)
        if newer is None:
            latest_by_fingerprint[record.content_fingerprint] = key
            continue
        records[key] = ToolResultLifecycleRecord(
            message_id=record.message_id,
            part_id=record.part_id,
            lifecycle=ToolResultLifecycle.DUPLICATE,
            reason="derived_content_duplicate",
            content_fingerprint=record.content_fingerprint,
            duplicate_of_part_id=newer[1],
            source_targets=record.source_targets,
        )


def _replace_lifecycle(
    record: ToolResultLifecycleRecord,
    lifecycle: ToolResultLifecycle,
    reason: str,
) -> ToolResultLifecycleRecord:
    return ToolResultLifecycleRecord(
        message_id=record.message_id,
        part_id=record.part_id,
        lifecycle=lifecycle,
        reason=reason,
        content_fingerprint=record.content_fingerprint,
        duplicate_of_part_id=record.duplicate_of_part_id,
        source_targets=record.source_targets,
    )


def _normalize_path(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    path = value.replace("\\", "/")
    if not path or path.startswith("/") or PureWindowsPath(value).is_absolute():
        return None
    parts = [part for part in path.split("/") if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        return None
    return "/".join(parts)


def _is_line_range(start_line: object, end_line: object) -> bool:
    return isinstance(start_line, int) and not isinstance(start_line, bool) and isinstance(end_line, int) and not isinstance(end_line, bool) and start_line >= 1 and end_line >= start_line


def _content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
