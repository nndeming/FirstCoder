"""面向用户的多模态输入辅助功能，包括附件、剪贴板与粘贴解析。"""

from firstcoder.input.attachments import (
    PreparedAttachment,
    UserAttachment,
    attach_path,
    prepare_attachments_for_session,
    resolve_paste_attachments,
)
from firstcoder.input.clipboard import read_clipboard_image_bytes

__all__ = [
    "PreparedAttachment",
    "UserAttachment",
    "attach_path",
    "prepare_attachments_for_session",
    "read_clipboard_image_bytes",
    "resolve_paste_attachments",
]
