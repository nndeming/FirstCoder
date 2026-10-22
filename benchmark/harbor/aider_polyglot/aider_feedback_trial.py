"""为单步 Harbor 编码任务提供 Aider 风格的测试反馈重试。

执行器仅用于基准测试。只有任务真实的共享验证器给出 ``reward=0`` 后，
才允许代理进行一次修复。首轮不暴露验证器文件，
也不重试未产生奖励值的基础设施故障。
"""

from __future__ import annotations

from typing import Any, override

from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.models.task.verifier_mode import VerifierEnvironmentMode, resolve_task_verifier_mode
from harbor.trial.errors import AgentTimeoutError
from harbor.trial.single_step import SingleStepTrial
from harbor.verifier.verifier import RewardFileNotFoundError


_MAX_FEEDBACK_CHARS = 60_000


def should_request_feedback_round(rewards: dict[str, float | int] | None) -> bool:
    """仅当验证器明确报告奖励为零时返回真值。"""

    return rewards is not None and rewards.get("reward") == 0


def should_request_feedback_after_missing_reward(test_output: str) -> bool:
    """识别 C++ 验证器正常的编译失败流程。

    当 CMake 无法编译提交内容时，本地 C++ 验证器会在写入 ``reward.txt`` 前退出。
    完整编译诊断仍可作为 Aider 风格的有效反馈。安装、Docker 与超时故障
    不含该标记，仍视为试验错误。
    """

    return "CMake build failed" in test_output and "error:" in test_output.lower()


def build_aider_feedback(test_output: str) -> str:
    """格式化修复轮次消息，避免暗示代理可以修改测试。"""

    output = test_output[-_MAX_FEEDBACK_CHARS:]
    return (
        "The tests are correct. Do not modify the tests.\n"
        "Fix the code in the current workspace to resolve the testing errors below, "
        "then run the relevant tests if possible.\n\n"
        "Testing errors:\n"
        f"{output}"
    )


class AiderFeedbackTrial(SingleStepTrial):
    """共享验证器失败后，在同一会话中额外执行一轮修复。"""

    @override
    async def _run(self) -> None:
        # 独立验证器会在测试前销毁代理环境，
        # 因此无法原地执行修复轮次。此类任务保留 Harbor 的
        # 标准行为，避免静默改变执行方式。
        if resolve_task_verifier_mode(self.task.config) != VerifierEnvironmentMode.SHARED:
            await super()._run()
            return

        await self._run_agent()
        await self._upload_agent_logs()
        await self._collect_artifacts()
        missing_reward_compile_failure = False
        try:
            await self._run_verifier()
        except RewardFileNotFoundError:
            missing_reward_compile_failure = should_request_feedback_after_missing_reward(
                self._read_verifier_output()
            )
            if not missing_reward_compile_failure:
                raise

        failed_reward = should_request_feedback_round(
            self.result.verifier_result.rewards if self.result.verifier_result else None
        )
        if self.result.exception_info is None and (failed_reward or missing_reward_compile_failure):
            feedback = build_aider_feedback(self._read_verifier_output())
            await self._run_agent_with_instruction(feedback)
            await self._upload_agent_logs()
            await self._run_verifier()

        await self._stop_agent_environment()

    async def _run_agent_with_instruction(self, instruction: str) -> None:
        """复用任务环境与会话来处理反馈消息。"""

        try:
            await self._run_agent_phase(
                target=self.result,
                instruction=instruction,
                timeout_sec=self._agent_timeout_sec,
                user=self.task.config.agent.user,
            )
        except (AgentTimeoutError, NonZeroAgentExitCodeError) as exc:
            self._record_exception(exc)
        finally:
            await self._sync_agent_output(self.result)

    def _read_verifier_output(self) -> str:
        output = self.paths.verifier_dir / "test-stdout.txt"
        try:
            return output.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "The verifier reported reward=0 but did not provide test output."


async def create_aider_feedback_trial(
    trial_class: Any,
    config: Any,
) -> AiderFeedbackTrial | None:
    """为单步任务创建反馈执行器，其他情况返回 None。"""

    task, download_result = await trial_class._load_task(config)
    if task.has_steps:
        return None
    return AiderFeedbackTrial(
        config,
        _task=task,
        _task_download_result=download_result,
    )
