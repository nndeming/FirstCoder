"""仅供基准使用的 AgentLoop 变体，对应三种因果实验条件。"""

from __future__ import annotations

from firstcoder.agent.loop import AgentLoop
from firstcoder.context.manager import ContextWindowTrigger

from benchmark.task_boundary_compaction.models import Arm


class BenchmarkAgentLoop(AgentLoop):
    """保留生产环境控制流程，并按需关闭某个实验条件下的单项行为。"""

    def __init__(self, *args, arm: Arm, **kwargs) -> None:
        self.arm = arm
        super().__init__(*args, **kwargs)

    def _classify_task_boundary(self, basis_message_id: str) -> None:
        if self.arm is not Arm.AUTO_ONLY:
            super()._classify_task_boundary(basis_message_id)

    async def _classify_task_boundary_async(self, basis_message_id: str) -> None:
        if self.arm is not Arm.AUTO_ONLY:
            await super()._classify_task_boundary_async(basis_message_id)

    def _compact_if_needed(
        self,
        *,
        trigger: ContextWindowTrigger,
        runtime_instruction: str | None = None,
    ):
        if self.arm is Arm.CLASSIFIER_ONLY and trigger is ContextWindowTrigger.TASK_HASH_CHANGED:
            return None
        return super()._compact_if_needed(
            trigger=trigger,
            runtime_instruction=runtime_instruction,
        )
