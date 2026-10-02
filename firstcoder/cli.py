"""Command-line entry point for single-turn FirstCoder runs."""

from __future__ import annotations
from firstcoder.app.ports import ChatRunnerLike

import argparse
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

from firstcoder.agent.loop_limits import AgentLoopLimits
from firstcoder.agent.runtime_capabilities import AgentRuntimeCapabilities
from firstcoder.app.factory import create_firstcoder_app
from firstcoder.config import load_config
from firstcoder.config.settings import default_global_config_path, project_config_path, render_default_config
from firstcoder.input.attachments import UserAttachment, attach_path
from firstcoder.mcp.config_store import McpConfigStore, McpConfigStoreError
from firstcoder.permissions.types import PermissionMode


@dataclass(frozen=True, slots=True)
class CliConfig:
    project_root: Path
    data_root: Path | None
    session_id: str | None
    message: str
    model_spec: str | None = None
    max_tool_rounds: int | None = None
    max_turn_seconds: float | None = None
    reasoning_effort: str | None = None
    benchmark: bool = False
    resume_session: bool = False
    attachments: tuple[Path, ...] = ()


CliRunner = Callable[[CliConfig], str]


def read_message(message: str | None, *, stdin_text: str | None = None) -> str:
    """Return a user message from an argument or stdin."""

    if message is not None:
        return message.strip()
    text = sys.stdin.read() if stdin_text is None else stdin_text
    return text.strip()


def build_parser() -> argparse.ArgumentParser:
    """
    创建命令行参数解析器
    """
    parser = argparse.ArgumentParser(description="Run a single FirstCoder user turn.")
    subparsers = parser.add_subparsers(dest="command")

    # config 相关子命令
    config_parser = subparsers.add_parser("config", help="Inspect or initialize FirstCoder configuration.")
    config_subparsers = config_parser.add_subparsers(dest="config_command")
    config_subparsers.add_parser("path", help="Show global and project config paths.")
    config_subparsers.add_parser("show", help="Show effective provider configuration without secrets.")
    init_parser = config_subparsers.add_parser("init", help="Create a starter global config file.")
    init_parser.add_argument("--force", action="store_true", help="Overwrite the existing global config.")

    # mcp 相关子命令
    mcp_parser = subparsers.add_parser("mcp", help="Add, list, or remove MCP server configuration.")
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command")
    add_parser = mcp_subparsers.add_parser("add", help="Add a local command or remote URL MCP server.")
    add_parser.add_argument("name")
    add_parser.add_argument("--url", help="Remote MCP URL. Omit for a local stdio command.")
    add_parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    add_parser.add_argument("--header", action="append", default=[], metavar="KEY=VALUE")
    add_parser.add_argument("--bearer-token-env-var", help="Environment variable containing a remote bearer token.")
    add_parser.add_argument("server_command", nargs="*", metavar="COMMAND")
    mcp_subparsers.add_parser("list", help="List configured MCP servers without secrets.")
    remove_parser = mcp_subparsers.add_parser("remove", help="Remove one configured MCP server.")
    remove_parser.add_argument("name")

    # 正常命令行参数
    parser.add_argument("--project", default=".", help="Project root for tools and AGENTS.md.")
    parser.add_argument("--data-root", default=None, help="Directory for FirstCoder session data.")
    parser.add_argument("--session-id", default=None, help="Session id to create or reuse.")
    parser.add_argument(
        "--resume-session",
        action="store_true", # 默认False, 携带则为true
        help="Resume an existing session id instead of creating a new session.",
    )
    parser.add_argument("--model", default=None, help="Model reference, for example provider/model.")
    parser.add_argument("--message", default=None, help="Single user message. Reads stdin when omitted.")
    parser.add_argument(
        "--attachment",
        action="append",
        default=[],
        metavar="PATH",
        help="Attach a local file to the single user message. Repeat for multiple files.",
    )
    parser.add_argument("--interactive", action="store_true", help="Run a line-oriented interactive session.")
    parser.add_argument("--tui", action="store_true", help="Run the Textual TUI.")
    parser.add_argument("--auto-approve", action="store_true", help="Automatically answer permission confirmations with allow_once.")
    parser.add_argument("--max-tool-rounds", type=_positive_int, default=None, help="Override per-turn tool round limit.")
    parser.add_argument(
        "--max-turn-seconds",
        type=_positive_float,
        default=None,
        help="Override the wall-clock limit for one user turn.",
    )
    parser.add_argument("--reasoning-effort", default=None, help="Provider-specific reasoning effort passed in the model request.")
    parser.add_argument(
        "--benchmark",
        action="store_true", # 默认False, 携带则为true
        help="Run the message with the non-interactive benchmark adapter using bypass permissions.",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    runner: CliRunner | None = None,
    stdin_text: str | None = None,
) -> int:
    parser = build_parser()
    # 解析命令行参数, args为已知参数, extras为未知参数
    args, extras = parser.parse_known_args(argv)
    if extras:
        # 仅 mcp add 本地模式宽容处理未知参数: 启动命令由用户自定义, 其中的选项
        # (如 npx -y 的 -y)无法预先登记, 会被 parse_known_args 放进 extras,
        # 这里归队到 server_command 保证命令完整; 其他命令的参数是封闭集合,
        # 出现未知参数即用户输错, 照常报错
        if args.command == "mcp" and args.mcp_command == "add" and not args.url:
            # extras 和 server_command 都是 list
            # list.extend(另一个列表) = 把另一个列表的元素逐个追加到尾部 
            args.server_command.extend(extras)
        else:
            parser.error(f"unrecognized arguments: {' '.join(extras)}")
    if args.command == "config":
        # config 相关命令
        return run_config_command(args)
    if args.command == "mcp":
        # mcp 相关命令
        return run_mcp_command(args)

    if args.attachment and (args.tui or args.interactive):
        print("error: --attachment is only supported for single-message runs", file=sys.stderr)
        return 2

    if args.tui or (args.message is None and stdin_text is None and sys.stdin.isatty() and not args.interactive):
        # 显示要求tui || 不携带--message, 未通过管道/程序注入输入文本, 开了真实终端, 不携带--interactive
        # TUI模式
        config = CliConfig(
            project_root=Path(args.project),
            data_root=Path(args.data_root) if args.data_root is not None else None, # 会话数据存放目录
            session_id=args.session_id,
            message="",
            model_spec=args.model,  # 模型引用
            max_tool_rounds=args.max_tool_rounds,
            max_turn_seconds=args.max_turn_seconds,
            reasoning_effort=args.reasoning_effort, # 推理强度
            benchmark=args.benchmark,
            resume_session=args.resume_session,
            attachments=tuple(Path(value) for value in args.attachment),
        )
        try:
            app = create_cli_app(config)
            # 走TUI运行, 此处使用的是Textual库
            # 具体使用方法: 继承Textual库的基类, 按约定实现特定函数即可, 剩下的交给框架
            app.run()
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.interactive:
        # REPL模式, CLI交互
        config = CliConfig(
            project_root=Path(args.project),
            data_root=Path(args.data_root) if args.data_root is not None else None,
            session_id=args.session_id,
            message="",
            model_spec=args.model,
            max_tool_rounds=args.max_tool_rounds,
            max_turn_seconds=args.max_turn_seconds,
            reasoning_effort=args.reasoning_effort,
            benchmark=args.benchmark,
            resume_session=args.resume_session,
            attachments=tuple(Path(value) for value in args.attachment),
        )
        try:
            app = create_cli_app(config)
            assert app.chat_runner is not None
            # 仅在部分test下lines不为None
            lines = stdin_text.splitlines() if stdin_text is not None else None
            # 开始和模型交互
            run_repl(app.chat_runner, lines, auto_approve=args.auto_approve)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    # 单轮对话模式
    # 从 --message 参数或标准输入(stdin 管道)获取用户消息，--message 优先
    message = read_message(args.message, stdin_text=stdin_text)
    if not message:
        print("error: message is required via --message or stdin", file=sys.stderr)
        return 2

    config = CliConfig(
        project_root=Path(args.project),
        data_root=Path(args.data_root) if args.data_root is not None else None,
        session_id=args.session_id,
        message=message,    # 单轮对话会在此处直接添加用户prompt
        model_spec=args.model,
        max_tool_rounds=args.max_tool_rounds,
        max_turn_seconds=args.max_turn_seconds,
        reasoning_effort=args.reasoning_effort,
        benchmark=args.benchmark,
        resume_session=args.resume_session,
        attachments=tuple(Path(value) for value in args.attachment),
    )
    run = runner or run_single_turn # 正常运行下, runner为None, runner参数的设计主要是为了测试使用
    try:
        output = run(config)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if output:
        print(output)
    return 0


def run_single_turn(config: CliConfig) -> str:
    if config.benchmark:
        # benchmark特殊处理
        return run_benchmark_turn(config)
    app = create_cli_app(config)
    assert app.chat_runner is not None
    response = app.chat_runner.run_user_turn(
        config.message,
        attachments=_prepare_cli_attachments(config.attachments),
    )
    return response.content


def run_benchmark_turn(config: CliConfig) -> str:
    """Run Harbor's non-interactive turn with benchmark-safe session settings."""

    app = create_cli_app(config)
    assert app.current_session is not None
    assert app.chat_runner is not None
    app.current_session.set_permission_mode(PermissionMode.BYPASS)
    app.current_session.session.require_prewrite_review = False
    app.current_session.session.set_benchmark_task(config.message)
    app.chat_runner.limits = _benchmark_limits(
        config.max_tool_rounds,
        max_turn_seconds=config.max_turn_seconds,
    )
    response = app.chat_runner.run_user_turn(
        config.message,
        attachments=_prepare_cli_attachments(config.attachments),
    )
    return response.content


def create_cli_app(config: CliConfig):
    # 根据是否为benchmark模式, 开放不同的能力
    capabilities = (
        AgentRuntimeCapabilities.benchmark(config.message)
        if config.benchmark
        else AgentRuntimeCapabilities.interactive()
    )
    # 使用配置创建app
    app = create_firstcoder_app(
        project_root=config.project_root,
        data_root=config.data_root,
        session_id=config.session_id,
        model_spec=config.model_spec,
        resume_session=config.resume_session,
        allow_user_input=capabilities.allow_user_input,
        runtime_capabilities=capabilities,
    )
    if config.max_tool_rounds is not None or config.max_turn_seconds is not None:
        assert app.chat_runner is not None
        # 默认兜底配置
        limits = AgentLoopLimits.default()
        if config.max_tool_rounds is not None:
            limits = limits.with_max_tool_rounds(config.max_tool_rounds)
        if config.max_turn_seconds is not None:
            limits = replace(limits, max_turn_seconds=config.max_turn_seconds)
        app.chat_runner.limits = limits
    if config.reasoning_effort is not None:
        # 推理强度
        assert app.chat_runner is not None
        effort = config.reasoning_effort.strip()
        if not effort:
            raise ValueError("reasoning_effort must be a non-blank string")
        options = app.chat_runner.request_options
        extra_body = dict(options.extra_body)
        extra_body["reasoning_effort"] = effort
        app.chat_runner.request_options = replace(options, extra_body=extra_body)
    return app


def _prepare_cli_attachments(paths: tuple[Path, ...]) -> list[UserAttachment]:
    return [attach_path(path, source="path") for path in paths]


def run_config_command(args: argparse.Namespace) -> int:
    # 获取用户输入的"config"下的相关命令信息
    command = args.config_command or "show"
    # 项目根目录(默认.)
    project_root = Path(args.project)
    if command == "path":
        print(f"global: {default_global_config_path()}")
        print(f"project: {project_config_path(project_root)}")
        return 0
    if command == "init":
        path = default_global_config_path()
        if path.exists() and not args.force:
            print(f"config already exists: {path}", file=sys.stderr)
            print("use --force to overwrite", file=sys.stderr)
            return 1
        path.parent.mkdir(parents=True, exist_ok=True)
        # 在 default_global_config_path 写入 default_config
        path.write_text(render_default_config(), encoding="utf-8")
        print(f"created: {path}")
        return 0
    if command == "show":
        config = load_config(project_root=project_root)
        catalog = config.model_catalog()
        # 供应商: moonshot
        print(f"provider: {_effective_provider(config)}")
        # 模型: kimi/k2
        print(f"model: {_effective_model(config)}")
        if catalog.profiles:
            print(f"default_model: {catalog.default_ref or '<first configured model>'}")
            print("models:")
            for profile in catalog.list():
                print(f"  - {profile.ref} ({profile.label})")
        print(f"base_url: {_effective_base_url(config)}")
        # 是否允许模型在一次回复中并行发起多个工具调用
        print(f"parallel_tool_calls: {_effective_parallel_tool_calls(config)}")
        print("config_files:")
        for path in config.loaded_config_paths:
            print(f"  - {path}")
        if not config.loaded_config_paths:
            print("  - <none>")
        return 0
    print(f"error: unknown config command: {command}", file=sys.stderr)
    return 2


def run_mcp_command(args: argparse.Namespace) -> int:
    """编辑全局 MCP 配置；运行期连接仍由 app factory 管理。"""

    if args.mcp_command is None:
        print("error: choose mcp add, list, or remove", file=sys.stderr)
        return 2
    store = McpConfigStore(default_global_config_path())
    try:
        if args.mcp_command == "list":
            servers = store.list_servers()
            if not servers:
                print("No MCP servers configured.")
                return 0
            for server in servers:
                status = "enabled" if server["enabled"] else "disabled"
                print(f'{server["name"]} {server["type"]} {server["endpoint"]} {status}')
            return 0
        if args.mcp_command == "remove":
            if not store.remove(args.name):
                print(f"MCP server not found: {args.name}", file=sys.stderr)
                return 1
            print(f"Removed MCP server: {args.name}")
            return 0
        # 以下是add命令
        env = _key_values(args.env, "--env")
        headers = _key_values(args.header, "--header")
        if args.url:
            if env:
                # 走 env 环境变量的是本地mcp
                print("error: --env is only supported for local MCP servers", file=sys.stderr)
                return 2
            if args.server_command:
                # 带 --url 的是远程mcp
                print("error: local command cannot be used with --url", file=sys.stderr)
                return 2
            # 添加远程mcp
            store.add_remote(
                args.name,
                args.url,
                headers=headers,
                bearer_token_env_var=args.bearer_token_env_var,
            )
            print(f"Added remote MCP server: {args.name}")
            return 0
        if headers:
            print("error: --header is only supported for remote MCP servers", file=sys.stderr)
            return 2
        if args.bearer_token_env_var:
            print("error: --bearer-token-env-var is only supported for remote MCP servers", file=sys.stderr)
            return 2
        # 添加本地mcp
        store.add_local(args.name, args.server_command, env=env)
        print(f"Added local MCP server: {args.name}")
        return 0
    except McpConfigStoreError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _key_values(values: list[str], option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, content = value.partition("=")
        if not separator or not key or not content:
            raise McpConfigStoreError(f"{option} 必须使用 KEY=VALUE 格式")
        result[key] = content
    return result


def _effective_model(config) -> str:
    catalog = config.model_catalog()
    if catalog.default_ref:
        return catalog.default_ref
    profiles = catalog.list()
    return profiles[0].ref if profiles else "<not configured>"


def _effective_provider(config) -> str:
    profile = _effective_profile(config)
    return profile.provider.id if profile is not None else "<not configured>"


def _effective_profile(config):
    catalog = config.model_catalog()
    profile = catalog.get(catalog.default_ref) if catalog.default_ref else None
    if profile is None and catalog.profiles:
        profile = catalog.profiles[0]
    return profile


def _effective_base_url(config) -> str:
    profile = _effective_profile(config)
    return profile.provider.base_url if profile and profile.provider.base_url else "<provider default>"


def _effective_parallel_tool_calls(config) -> str:
    profile = _effective_profile(config)
    if profile is None:
        return "false"
    enabled = config.get_provider_bool(
        "parallel_tool_calls",
        env="FIRSTCODER_PARALLEL_TOOL_CALLS",
        default=False,
        provider_name=profile.provider.id,
    )
    return "true" if enabled else "false"


def _benchmark_limits(
    max_tool_rounds: int | None,
    *,
    max_turn_seconds: float | None = None,
) -> AgentLoopLimits:
    base = AgentLoopLimits.swe_lite()
    if max_tool_rounds is not None:
        base = base.with_max_tool_rounds(max_tool_rounds, provider_call_reserve=40)
    if max_turn_seconds is not None:
        base = replace(base, max_turn_seconds=max_turn_seconds)
    return base


def run_repl(
    chat_runner: ChatRunnerLike,
    lines: Iterable[str] | None = None,
    *,
    auto_approve: bool = False,
) -> None:
    # 注意if部分实际上只在部分test场景下被使用(涉及到直接往main函数传参)
    # 正常使用的话走的是else分支的 _stdin_lines()
    # _stdin_lines()是带yield的函数, source此处只是获得了generator
    source = iter(lines) if lines is not None else _stdin_lines()
    pending = None
    for raw_line in source:
        # line是每次的用户输入
        line = raw_line.strip()
        if not line:
            continue
        if line in {"/exit", "/quit"}:
            break

        # 整个 REPL 是一个大循环, 上一轮可能没跑完就被挂起了(pending 不为 None):
        # 要么卡在权限确认, 要么某个工具在反问用户。这时不能开新一轮,
        # 要把用户这行的输入当作"回答", 走 resume_with_user_input 从断点继续。
        if pending is not None:
            if _pending_kind(pending) == "permission_confirmation":
                # 获取用户选择的权限
                choice = _permission_choice_for_text(line, pending)
                if choice is None:
                    print(f"Unknown permission choice: {line}")
                    print(_permission_choice_help_text(pending))
                    print(_permission_options_text(pending))
                    continue
                line = choice
            # 调用大模型获取响应
            response = chat_runner.resume_with_user_input(_pending_id(pending), line)
        else:
            # 具体实现在firstcoder\app\runtime.py
            response = chat_runner.run_user_turn(line)

        print(f"FirstCoder> {response.content}")
        pending = getattr(chat_runner, "last_pending_input", None)
        # 根据"启动参数有没有开 --auto-approve"决定要不要代替用户回答; 每次代答固定用 allow_once(只放行本次)
        # 只要模型连续不断地抛出权限确认, 就一直代答下去, 直到它干完活或碰到非权限类的提问为止
        while pending is not None and auto_approve and _pending_kind(pending) == "permission_confirmation":
            print("Auto-approve> allow_once")
            response = chat_runner.resume_with_user_input(_pending_id(pending), "allow_once")
            print(f"FirstCoder> {response.content}")
            pending = getattr(chat_runner, "last_pending_input", None)

        if pending is not None:
            if _pending_kind(pending) == "permission_confirmation":
                print(_permission_options_text(pending))
            else:
                print(f"Permission> {_pending_question(pending)}")


def _stdin_lines():
    # 本函数是生成器: yield 产出一行后挂起(执行位置与局部状态冻结),
    # 外层 for 循环下次取数时原地苏醒、从挂起点继续执行。
    # return 与 yield 的区别:
    #   return - 函数彻底结束, 局部变量销毁, 再次调用从头执行
    #   yield  - 函数暂停而不结束, 产出一个值后冻结, 下次被迭代时原地"复活"
    # 示例:
    #   def f():
    #       yield 1  # 产出 1, 挂起, 状态保留
    #       yield 2  # 下次被叫醒, 从这继续, 产出 2, 再挂起
    #       yield 3  # 再产出 3
    # 逐个 yield 正是为了把"无限的、随时间到达的输入流"
    # 写成惰性序列: 用户敲一行、REPL 处理一轮、再回来取下一行。
    # 创建一个PromptSession(增强版输入器, 支持行编辑、历史记录)
    prompt = _create_prompt_session()
    if prompt is not None:
        # 创建成功, 则使用这个增强版输入器
        while True:
            try:
                # 产出一行输入, 然后挂起; REPL 下一轮 for 再来取时苏醒
                yield prompt.prompt("You> ")
            except (EOFError, KeyboardInterrupt):
                # Ctrl+D(EOF) 或 Ctrl+C: 结束迭代, REPL 会话收尾
                break
        return

    # 创建失败, 则降级为普通输入器
    while True:
        try:
            # 同上: yield 出一行后挂起, 等待下次被迭代
            yield input("You> ")
        except EOFError:
            break


def _create_prompt_session():
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import InMemoryHistory
    except ImportError:
        return None
    return PromptSession(history=InMemoryHistory())


def _pending_id(pending: object) -> str:
    return str(getattr(pending, "id"))


def _pending_question(pending: object) -> str:
    return str(getattr(pending, "question", "需要用户输入。"))


def _pending_kind(pending: object) -> str:
    return str(getattr(pending, "kind", ""))


def _permission_choice_for_text(text: str, pending: object) -> str | None:
    """权限确认翻译\n\n
    将用户随手输入的(1, y, yes, allow)转为标准权限选项
    """
    normalized = text.strip().lower().replace(" ", "_")
    raw = text.strip()
    if raw.lower().startswith(("reject:", "reject_with_feedback:")):
        return f"reject_with_feedback: {raw.split(':', 1)[1].strip()}"
    aliases = {
        "1": "deny",                                            # 拒绝
        "n": "deny",
        "no": "deny",
        "deny": "deny",
        "reject": "reject_with_feedback",                       # 拒绝并反馈给模型
        "reject_with_feedback": "reject_with_feedback",
        "2": "allow_once",                                      # 只允许一次
        "y": "allow_once",
        "yes": "allow_once",
        "allow": "allow_once",
        "once": "allow_once",
        "allow_once": "allow_once",
        "3": "allow_always_same_scope",                         # 同范围内永久允许
        "always": "allow_always_same_scope",
        "allow_always": "allow_always_same_scope",
        "allow_always_same_scope": "allow_always_same_scope",
    }
    if normalized in aliases:
        return aliases[normalized]

    # 兜底措施, 从当前pending对象取出index和option选项
    for index, option in enumerate(_permission_options(pending), start=1):
        option_id = _option_id(option)
        label = _option_label(option)
        values = {
            str(index).lower(),     # 选项编号(1, 2, 3)
            option_id.lower(),      # 选项的规范ID
            label.strip().lower().replace(" ", "_"), # 显示名归一化
        }
        if normalized in values:
            return option_id
    return None


def _permission_options_text(pending: object) -> str:
    """进一步具体展示每个选项对应的权限
    """
    question = _pending_question(pending)
    options = _permission_options(pending)
    option_lines = [f"  {index}. {_option_label(option)}" + (f" ({_option_id(option)})" if _option_id(option) != _option_label(option) else "") for index, option in enumerate(options, start=1)]
    if not option_lines:
        option_lines = [
            "  1. Deny",
            "  2. Allow once",
            "  3. Allow always for same scope",
        ]
    return "\n".join(
        [
            f"Permission> {question}",
            "Choose:",
            *option_lines,
        ]
    )


def _permission_choice_help_text(pending: object) -> str:
    """生成"请输入 1、2、3…"这样的编号提示文本
    """
    count = len(_permission_options(pending)) or 3
    choices = ", ".join(str(index) for index in range(1, count + 1))
    return f"Please choose {choices}."


def _permission_options(pending: object) -> list[object]:
    return list(getattr(pending, "options", []) or [])


def _option_id(option: object) -> str:
    if isinstance(option, dict):
        return str(option.get("id") or option.get("label") or "")
    return str(getattr(option, "id", getattr(option, "label", "")))


def _option_label(option: object) -> str:
    if isinstance(option, dict):
        return str(option.get("label") or option.get("id") or "")
    return str(getattr(option, "label", getattr(option, "id", "")))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed
