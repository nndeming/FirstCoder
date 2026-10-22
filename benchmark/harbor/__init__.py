"""FirstCoder 的 Harbor 基准测试适配器。

Harbor 是可选的外部基准运行环境。将适配器放在 ``benchmark.harbor``，
避免其依赖与容器生命周期进入 FirstCoder 核心运行时。
数据集专用集成位于 ``aider_polyglot`` 和 ``terminal_bench``，
可复用的运行辅助功能位于 ``shared``。
"""
