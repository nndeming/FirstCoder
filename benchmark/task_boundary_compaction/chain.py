"""真实任务 A 基准上下文的私有捕获与重放基础功能。"""

from __future__ import annotations

from pathlib import Path
import shutil

from firstcoder.context.store import JsonlSessionStore


def clone_recorded_session(
    *,
    source_data_root: str | Path,
    destination_data_root: str | Path,
    session_id: str,
) -> JsonlSessionStore:
    """复制已捕获的会话存储，用于隔离的任务 B 试验。

    调用方管理两个目录，并负责在试验结束后删除它们。此函数只复制会话数据，
    不接受也不转移任务 A 的项目目录。
    """

    source = Path(source_data_root)
    destination = Path(destination_data_root)
    session_path = source / "sessions" / f"{session_id}.jsonl"
    if not session_path.is_file():
        raise FileNotFoundError(f"recorded session does not exist: {session_path}")
    if destination.exists():
        raise FileExistsError(f"recorded session destination already exists: {destination}")
    shutil.copytree(source, destination)
    return JsonlSessionStore(destination)
