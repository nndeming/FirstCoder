"""压缩后工具结果的持久化归档: sha256 内容寻址, 写盘 all-or-nothing.

归档只管磁盘上的字节和两种占位符格式; 什么时候该压是 context pipeline 的策略,
不归这里管. 归档是 append-only 的: 替换任何内容前必须先把原文归档, 归档失败就保留现状.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from firstcoder.context.models import MessagePart, utc_now_iso
from firstcoder.context.token_budget import estimate_text_tokens
from firstcoder.context.versions import ARCHIVE_SCHEMA_VERSION

_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9_-]{1,128}")


class ArchiveIntegrityError(RuntimeError):
    """归档的不可变内容与摘要对不上时抛出 (视为损坏, 绝不就地修复或覆盖)."""


@dataclass(frozen=True, slots=True)
class ArchiveRecord:
    """一份归档的公开身份与体积信息, 不含任何文件系统路径."""

    archive_id: str
    session_id: str
    content_sha256: str
    original_chars: int
    original_tokens: int
    created_at: str
    schema_version: str = ARCHIVE_SCHEMA_VERSION


@dataclass(slots=True)
class ToolResultArchive:
    """落盘保存工具输出原文, 并构造压缩后的上下文投影 (占位符)."""

    root: str | Path

    def store_original(
        self,
        session_id: str,
        part: MessagePart,
        original_content: str | None = None,
    ) -> ArchiveRecord:
        """把完整原文按内容寻址 (sha256) 存成不可变归档.

        红线是"先归档后替换": 调用方必须先拿到本方法返回的 record 才能替换视图内容,
        归档失败就跳过该 part, 不做半个替换. 重复存同样内容是 no-op; 已存在的文件
        摘要对不上视为损坏, 绝不修复或覆盖.
        """

        self._validate_part(part)
        raw = part.content if original_content is None else original_content
        digest = _sha256(raw)
        return self._store(
            session_id=session_id,
            archive_id=f"ar_{digest[:32]}",
            raw=raw,
            content_sha256=digest,
        )

    def read(self, session_id: str, archive_id: str) -> tuple[ArchiveRecord, str]:
        """返回经过完整性校验的归档元数据和完整原文.

        调用方拿不到文件系统路径; 所有完整性校验都在归档边界内完成, 字节出边界前必过.
        """

        text_path, metadata_path = self._archive_paths(session_id, archive_id)
        raw = text_path.read_text(encoding="utf-8")
        metadata = _read_metadata(metadata_path)
        actual_digest = _sha256(raw)

        expected_id = metadata.get("archive_id")
        expected_digest = metadata.get("content_sha256")
        if metadata.get("schema_version") != ARCHIVE_SCHEMA_VERSION:
            raise ArchiveIntegrityError("archive metadata uses an unsupported schema")
        if expected_id != archive_id or not isinstance(expected_digest, str):
            raise ArchiveIntegrityError(f"{ARCHIVE_SCHEMA_VERSION} archive metadata is invalid")
        if expected_digest != actual_digest:
            raise ArchiveIntegrityError("archive content does not match its SHA-256")
        if archive_id != _content_addressed_id(actual_digest):
            raise ArchiveIntegrityError(f"{ARCHIVE_SCHEMA_VERSION} archive id does not match its content SHA-256")
        if metadata.get("original_chars") != len(raw):
            raise ArchiveIntegrityError("archive character count does not match content")
        return (
            ArchiveRecord(
                archive_id=archive_id,
                session_id=session_id,
                content_sha256=actual_digest,
                original_chars=len(raw),
                original_tokens=_metadata_tokens(metadata, raw),
                created_at=_metadata_created_at(metadata),
            ),
            raw,
        )

    def make_placeholder(
        self,
        part: MessagePart,
        record: ArchiveRecord,
        lifecycle: str = "derived",
        summary: str | None = None,
        key_errors: tuple[str, ...] = (),
    ) -> MessagePart:
        """创建 v2 投影占位符: 只留生命周期说明和 archive_id, 绝不内嵌原文字节.

        模型读到占位符后可凭 archive_id 调 retrieve_archive 取回原文.
        """

        self._validate_part(part)
        # 各字段都先过 _short 截断, 防止异常长的 metadata 把占位符撑爆
        tool_name = _short(part.metadata.get("tool_name") or "tool", 64)
        status = _short(_tool_status(part), 32)
        safe_lifecycle = _short(lifecycle, 32)
        resolved_summary = _short(summary or _default_summary(part, original_tokens=record.original_tokens), 240)
        # key_errors 最多留 3 条, 它们是裁剪时最先被丢的可选信息
        errors = tuple(_short(error, 72) for error in key_errors[:3] if str(error).strip())
        # 占位符各行: 身份六行 + summary + key_errors + 取回指引, 布局见 _fit_placeholder
        lines = [
            "[Tool result archived]",
            f"archive_id={record.archive_id}",
            f"tool={tool_name}",
            f"status={status}",
            f"lifecycle={safe_lifecycle}",
            f"original_tokens={record.original_tokens}",
            f"summary={resolved_summary}",
        ]
        lines.extend(f"key_errors={error}" for error in errors)
        lines.append("Use retrieve_archive(archive_id, ...) to inspect the original.")
        # 480 字符的名片上限, 超了由 _fit_placeholder 按牺牲顺序裁剪
        content = _fit_placeholder(lines, maximum=480)

        metadata: dict[str, Any] = dict(part.metadata)
        # 调用方可能传入的是旧的投影 part; 不把它的预览字段带进 v2 占位符
        metadata.pop("preview", None)
        metadata.pop("preview_tokens", None)
        metadata.update(
            {
                # archive_id + sha256 是取回和校验原文的凭证, token 数用于记账
                "archive_id": record.archive_id,
                "original_content_sha256": record.content_sha256,
                "original_tokens": record.original_tokens,
                "compaction_state": "archived",
                "compacted_by": "l3_archive",
            }
        )
        return MessagePart(
            id=part.id,
            message_id=part.message_id,
            kind=part.kind,
            content=content,
            metadata=metadata,
        )

    def _store(
        self,
        *,
        session_id: str,
        archive_id: str,
        raw: str,
        content_sha256: str,
    ) -> ArchiveRecord:
        # 原文和元数据分两个文件存: ar_xxx.txt 是原文, ar_xxx.json 是元数据
        text_path, metadata_path = self._archive_paths(session_id, archive_id)
        text_path.parent.mkdir(parents=True, exist_ok=True)

        # 内容寻址的去重: 文件已存在就核对哈希, 一致则是重复存储 (no-op), 不一致即损坏, 绝不覆盖
        if text_path.exists():
            if _sha256(text_path.read_text(encoding="utf-8")) != content_sha256:
                raise ArchiveIntegrityError("existing archive text has a different SHA-256")
        else:
            _atomic_write(text_path, raw)

        record = ArchiveRecord(
            archive_id=archive_id,
            session_id=session_id,
            content_sha256=content_sha256,
            original_chars=len(raw),
            original_tokens=estimate_text_tokens(raw),
            created_at=utc_now_iso(),
        )
        expected_metadata = {
            "archive_id": record.archive_id,
            "content_sha256": record.content_sha256,
            "original_chars": record.original_chars,
            "original_tokens": record.original_tokens,
            "created_at": record.created_at,
            "schema_version": record.schema_version,
        }
        # 元数据同样按去重处理: 已存在则逐项核对 (含 schema 版本), 全部一致就沿用首次存入的
        # created_at / original_tokens, 保证同一归档的 record 永远不变; 对不上即损坏, 抛错
        if metadata_path.exists():
            existing = _read_metadata(metadata_path)
            if existing.get("schema_version") == ARCHIVE_SCHEMA_VERSION:
                if existing.get("archive_id") != archive_id or existing.get("content_sha256") != content_sha256 or existing.get("original_chars") != len(raw):
                    raise ArchiveIntegrityError("existing archive metadata disagrees with content")
                record = ArchiveRecord(
                    archive_id=archive_id,
                    session_id=session_id,
                    content_sha256=content_sha256,
                    original_chars=len(raw),
                    original_tokens=_metadata_tokens(existing, raw),
                    created_at=_metadata_created_at(existing),
                )
            else:
                raise ArchiveIntegrityError("existing archive metadata uses an incompatible schema")
        else:
            _atomic_write(metadata_path, json.dumps(expected_metadata, ensure_ascii=False, sort_keys=True))
        return record

    def _archive_paths(self, session_id: str, archive_id: str) -> tuple[Path, Path]:
        _validate_component(session_id, "session_id")
        _validate_component(archive_id, "archive_id")
        directory = Path(self.root) / "archives" / session_id
        return directory / f"{archive_id}.txt", directory / f"{archive_id}.json"

    @staticmethod
    def _validate_part(part: MessagePart) -> None:
        if part.kind != "tool_result":
            raise ValueError("ToolResultArchive only accepts tool_result parts")


def _validate_component(value: str, name: str) -> None:
    if not isinstance(value, str) or _SAFE_COMPONENT.fullmatch(value) is None:
        raise ValueError(f"{name} must contain only letters, digits, underscores, or hyphens")


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _content_addressed_id(content_sha256: str) -> str:
    return f"ar_{content_sha256[:32]}"


def _atomic_write(path: Path, content: str) -> None:
    """同目录临时文件 + os.replace 原子落盘: all-or-nothing, 失败时清理临时文件, 不留下半个写入."""

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArchiveIntegrityError("archive metadata cannot be read") from exc
    if not isinstance(value, dict):
        raise ArchiveIntegrityError("archive metadata must be an object")
    return value


def _metadata_tokens(metadata: dict[str, Any], raw: str) -> int:
    tokens = metadata.get("original_tokens")
    return tokens if isinstance(tokens, int) and tokens >= 0 else estimate_text_tokens(raw)


def _metadata_created_at(metadata: dict[str, Any]) -> str:
    created_at = metadata.get("created_at")
    return created_at if isinstance(created_at, str) else ""


def _short(value: object, maximum: int) -> str:
    text = str(value).replace("\n", " ").strip()
    return text[:maximum]


def _tool_status(part: MessagePart) -> str:
    if part.metadata.get("ok") is False:
        return "failed"
    status = str(part.metadata.get("status") or "").strip().lower()
    if status in {"failed", "failure", "error", "errored"} or part.metadata.get("is_error"):
        return "failed"
    return "success"


def _fit_placeholder(lines: list[str], *, maximum: int) -> str:
    """把占位符各行拼成正文, 并裁剪到 maximum 字符以内.

    输入行的固定布局 (由 make_placeholder 拼出):

        0: [Tool result archived]      ┐
        1: archive_id=ar_xxx           │
        2: tool=...                    │  身份信息 (required 前 6 行)
        3: status=...                  │
        4: lifecycle=...               │
        5: original_tokens=...         ┘
        6: summary=...                 <- optional
        7+: key_errors=...             <- 哪个桶都不在, 裁剪时最先丢
        8: Use retrieve_archive(archive_id, ...) ...   <- required 最后一行

    裁剪有明确的牺牲顺序: summary 是唯一面向模型、可伸缩的字段, 先砍它;
    key_errors 哪个桶都不在, 进入裁剪分支即被丢弃. 无论怎么裁,
    取回原文的指引行 (archive_id) 必须保住.
    """

    content = "\n".join(lines)
    if len(content) <= maximum:
        return content
    # 分桶: required 是身份和指引等必留行, optional 是可裁剪的 summary 行.
    # 拼回时 optional 插在第 6 行之后, 让超长时被末尾截断先牺牲的是它.
    required = [line for line in lines if not line.startswith(("summary=", "key_errors="))]
    optional = [line for line in lines if line.startswith("summary=")]
    candidate = "\n".join(required[:6] + optional + required[6:])
    if len(candidate) > maximum:
        # 还超就精确缩短 summary 文本, 把超出量从它身上扣掉
        excess = len(candidate) - maximum
        summary = optional[0][8:]
        optional = [f"summary={summary[: max(0, len(summary) - excess)]}"]
        candidate = "\n".join(required[:6] + optional + required[6:])
    # 最后兜底整体截断, 保证返回值严格不超 maximum
    return candidate[:maximum]


def _default_summary(part: MessagePart, *, original_tokens: int) -> str:
    tool_name = str(part.metadata.get("tool_name") or "tool")
    # Keep the automatic summary short and language-stable.  The full output
    # remains in the archive; the placeholder only needs enough context for
    # the model to decide whether retrieval is worthwhile.
    return f"Large {tool_name} result ({original_tokens} tokens)"
