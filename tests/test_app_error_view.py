"""error_view: chat 错误文本的可读化渲染。"""

import json

from firstcoder.app.error_view import format_chat_error
from firstcoder.providers.errors import ProviderError, ProviderErrorKind

# 真实采集: OpenRouter 免费模型下线时, OpenAI SDK 抛出的 404 错误字符串。
# SDK 消息格式为 "Error code: <status> - <body dict 的 repr>", body 里 metadata.raw
# 是上游响应的 JSON 字符串(引号按 JSON 规则转义)。
_RAW_PAYLOAD = json.dumps(
    {
        "status": 404,
        "title": "Not Found",
        "detail": (
            'Function "ac74040f-9fc9-4c5e-ac74-279ba5161d69": '
            'Not found for account "sxpm0VR-knE2_K2__i14ZbzmXEludXpJbFGRvrvc08w"'
        ),
    },
    separators=(",", ":"),
)
_BODY = {
    "error": {
        "message": "Provider returned error",
        "code": 404,
        "metadata": {"raw": _RAW_PAYLOAD, "provider_name": "Nvidia", "is_byok": False},
    },
    "user_id": "user_3L5Gsc6iRZ97fEETOLolRJQVv51",
}
_OPENROUTER_404_MESSAGE = f"Error code: 404 - {_BODY}"


def test_format_chat_error_extracts_status_detail_and_hint_from_sdk_payload() -> None:
    error = ProviderError(ProviderErrorKind.API_ERROR, _OPENROUTER_404_MESSAGE)

    rendered = format_chat_error(error)

    lines = rendered.splitlines()
    assert lines[0] == "Chat error: API 请求被拒绝 (HTTP 404)"
    assert lines[1].startswith("上游返回: ")
    assert 'Function "ac74040f-9fc9-4c5e-ac74-279ba5161d69"' in lines[1]
    assert "Not found for account" in lines[1]
    assert lines[2].startswith("建议: ")
    assert "/model" in lines[2]
    # 内部字段(user_id / is_byok 等)不应出现在展示文本里。
    assert "user_id" not in rendered
    assert "is_byok" not in rendered
    assert "\\" not in rendered


def test_format_chat_error_falls_back_to_upstream_message_with_provider_name() -> None:
    message = (
        "Error code: 429 - {'error': {'message': 'Rate limit reached', "
        "'metadata': {'provider_name': 'OpenAI'}}}"
    )
    error = ProviderError(ProviderErrorKind.RATE_LIMIT, message)

    rendered = format_chat_error(error)

    assert rendered.splitlines()[0] == "Chat error: 请求被限流 (HTTP 429)"
    assert "[OpenAI] Rate limit reached" in rendered
    assert "建议: " in rendered


def test_format_chat_error_shows_readable_internal_message_directly() -> None:
    error = ProviderError(ProviderErrorKind.CONFIG_ERROR, "provider fake 不支持 tool calling，不能发送 tools")

    rendered = format_chat_error(error)

    assert rendered.splitlines()[0] == "Chat error: provider 配置错误"
    assert "provider fake 不支持 tool calling，不能发送 tools" in rendered


def test_format_chat_error_handles_plain_exceptions() -> None:
    rendered = format_chat_error(RuntimeError("provider down"))

    assert rendered == "Chat error: RuntimeError: provider down"


def test_format_chat_error_truncates_long_messages() -> None:
    rendered = format_chat_error(RuntimeError("x" * 500))

    assert len(rendered) < 300
    assert rendered.endswith("...")
