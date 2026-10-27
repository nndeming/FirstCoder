"""把 chat 执行异常渲染成用户可读的错误文本。

ProviderError 携带的是 SDK 原始错误字符串(整段响应 payload 的 dict repr), 直接
展示全是转义字符和内部字段。这里提取 HTTP 状态码、上游 message/detail, 并按
错误类别附一句可操作建议; 提取失败时回退到截断后的原始消息。
"""

from __future__ import annotations

import re

from firstcoder.providers.errors import ProviderError, ProviderErrorKind

_MAX_MESSAGE_CHARS = 200

_STATUS_CODE_RE = re.compile(r"Error code:\s*(?P<status>\d{3})")
# payload 是 dict repr; 内嵌 raw JSON 里的引号先被 JSON 转义成 \", 再被 repr
# 转义成 \\", 所以不能按"转义引号"配对解析, 改为取到引号+结构符(, 或 })为止。
_DETAIL_RE = re.compile(r'"detail":"(?P<detail>[^}]*?)"\s*[,}]')
_MESSAGE_RE = re.compile(r"'message': '(?P<message>(?:[^'\\]|\\.)*)'")
_PROVIDER_NAME_RE = re.compile(r"'provider_name': '(?P<name>[^']*)'")

_KIND_TITLES: dict[ProviderErrorKind, str] = {
    ProviderErrorKind.AUTH_ERROR: "认证失败",
    ProviderErrorKind.RATE_LIMIT: "请求被限流",
    ProviderErrorKind.PROMPT_TOO_LONG: "上下文超出模型长度限制",
    ProviderErrorKind.TIMEOUT: "请求超时",
    ProviderErrorKind.NETWORK_ERROR: "网络连接失败",
    ProviderErrorKind.SERVER_ERROR: "模型服务内部错误",
    ProviderErrorKind.API_ERROR: "API 请求被拒绝",
    ProviderErrorKind.CONFIG_ERROR: "provider 配置错误",
    ProviderErrorKind.UNSUPPORTED: "当前 provider 不支持该请求",
    ProviderErrorKind.USER_ABORT: "请求已取消",
    ProviderErrorKind.UNKNOWN: "未知错误",
}

_KIND_HINTS: dict[ProviderErrorKind, str] = {
    ProviderErrorKind.AUTH_ERROR: "检查 API key 是否配置正确(环境变量或 firstcoder.toml)。",
    ProviderErrorKind.RATE_LIMIT: "稍后重试, 或用 /model 切换到其他模型。",
    ProviderErrorKind.PROMPT_TOO_LONG: "先用 /compact 压缩上下文, 或开新会话再继续。",
    ProviderErrorKind.TIMEOUT: "检查网络/代理后重试。",
    ProviderErrorKind.NETWORK_ERROR: "检查网络/代理后重试。",
    ProviderErrorKind.SERVER_ERROR: "provider 侧故障, 稍后重试或用 /model 切换模型。",
    ProviderErrorKind.API_ERROR: "用 /model 确认模型名是否正确、当前账号是否有权限。",
}


def format_chat_error(exc: BaseException, *, prefix: str = "Chat error") -> str:
    """把 chat worker 抛出的异常格式化成多行的可读错误文本。"""

    if not isinstance(exc, ProviderError):
        return f"{prefix}: {type(exc).__name__}: {_truncate(str(exc))}"

    title = _KIND_TITLES.get(exc.kind, "未知错误")
    status = _extract_status_code(exc.message)
    lines = [f"{prefix}: {title}" + (f" (HTTP {status})" if status is not None else "")]

    detail = _extract_upstream_detail(exc.message)
    if detail is not None:
        lines.append(f"上游返回: {_truncate(detail)}")
    elif not _looks_like_sdk_payload(exc.message):
        # 内部构造的 ProviderError(如配置错误)消息本身可读, 直接附上。
        lines.append(_truncate(exc.message))

    hint = _hint_for(exc.kind, status)
    if hint:
        lines.append(f"建议: {hint}")
    return "\n".join(lines)


def _extract_status_code(message: str) -> int | None:
    match = _STATUS_CODE_RE.search(message)
    return int(match.group("status")) if match else None


def _extract_upstream_detail(message: str) -> str | None:
    """从 SDK 错误 payload 里挑最有信息量的一段: 优先 detail, 其次上游 message。"""

    detail_match = _DETAIL_RE.search(message)
    if detail_match is not None:
        return _unescape(detail_match.group("detail"))
    message_match = _MESSAGE_RE.search(message)
    if message_match is not None:
        upstream = _unescape(message_match.group("message"))
        provider_match = _PROVIDER_NAME_RE.search(message)
        if provider_match is not None:
            return f"[{provider_match.group('name')}] {upstream}"
        return upstream
    return None


def _unescape(text: str) -> str:
    """剥掉 JSON + repr 两层转义: \\\\" -> ", \\" -> ", 单引号同理。"""

    return (
        text.replace('\\\\"', '"')
        .replace('\\"', '"')
        .replace("\\\\'", "'")
        .replace("\\'", "'")
    )


def _looks_like_sdk_payload(message: str) -> bool:
    return message.startswith("Error code:")


def _hint_for(kind: ProviderErrorKind, status: int | None) -> str | None:
    if kind == ProviderErrorKind.API_ERROR and status == 404:
        return "模型不存在、已下线或当前 key 无权访问, 用 /model 换一个模型试试。"
    return _KIND_HINTS.get(kind)


def _truncate(text: str) -> str:
    if len(text) <= _MAX_MESSAGE_CHARS:
        return text
    return f"{text[:_MAX_MESSAGE_CHARS]}..."
