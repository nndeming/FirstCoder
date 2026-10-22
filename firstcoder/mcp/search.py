"""在本地查找未包含在初始工具定义集中的 MCP 工具。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import cast

from firstcoder.providers.types import ToolDefinition
from firstcoder.tools.types import Tool, ToolExecutor, ToolResult, make_error_result

MCP_TOOL_SEARCH_NAME = "mcp_tool_search"
MCP_TOOL_SEARCH_LIMIT = 8
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class McpSearchEntry:
    """可参与本地搜索、向模型提供的 MCP 工具定义。"""

    server: str
    tool: str
    definition: ToolDefinition


def search_mcp_tools(
    entries: tuple[McpSearchEntry, ...],
    query: str,
) -> tuple[McpSearchEntry, ...]:
    """按确定顺序返回本地 MCP 搜索中得分最高的匹配项。"""

    normalized_query = _normalize(query)
    query_tokens = set(_tokens(query))
    ranked: list[tuple[int, str, McpSearchEntry]] = []
    for entry in entries:
        score = _score(entry, normalized_query, query_tokens)
        if score > 0:
            ranked.append((-score, entry.definition.name, entry))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in ranked[:MCP_TOOL_SEARCH_LIMIT])


def create_mcp_tool_search(entries: tuple[McpSearchEntry, ...]) -> Tool:
    """创建只读工具，用于搜索一份 MCP 目录快照。"""

    def execute(*, query: str) -> ToolResult:
        if not isinstance(query, str) or not query.strip():
            return make_error_result(
                MCP_TOOL_SEARCH_NAME,
                "MCP tool search query must not be blank.",
                mcp_tool_search={"activated_tools": []},
            )
        matches = search_mcp_tools(entries, query)
        activated = [entry.definition.name for entry in matches]
        return ToolResult(
            name=MCP_TOOL_SEARCH_NAME,
            ok=True,
            content=_render_matches(matches),
            data={"mcp_tool_search": {"activated_tools": activated}},
        )

    return Tool(
        definition=ToolDefinition(
            name=MCP_TOOL_SEARCH_NAME,
            description=(
                "Search connected MCP tools by capability. Matching tool schemas "
                "become available for the remainder of the current user turn."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Describe the external capability or MCP operation needed.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        executor=cast(ToolExecutor, execute),
    )


def _score(
    entry: McpSearchEntry,
    normalized_query: str,
    query_tokens: set[str],
) -> int:
    name = _normalize(entry.tool)
    server = _normalize(entry.server)
    description = _normalize(entry.definition.description)
    visible_name = _normalize(entry.definition.name)
    score = 10_000 if normalized_query in {name, visible_name} else 0
    score += 100 * len(query_tokens.intersection(_tokens(name)))
    score += 20 * len(query_tokens.intersection(_tokens(server)))
    score += len(query_tokens.intersection(_tokens(description)))
    return score


def _normalize(value: str | None) -> str:
    return " ".join(_tokens(value or ""))


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN.findall(value.casefold().replace("_", " ").replace("-", " ")))


def _render_matches(matches: tuple[McpSearchEntry, ...]) -> str:
    if not matches:
        return "No matching MCP tools found. Try a more specific capability description."
    lines = [
        f"- {entry.definition.name}: {entry.server}/{entry.tool} — "
        f"{' '.join(entry.definition.description.split()) or 'No description provided.'}"
        for entry in matches
    ]
    return "Matching MCP tools activated for this user turn:\n" + "\n".join(lines)
