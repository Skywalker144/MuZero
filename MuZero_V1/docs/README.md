# 文档索引

- 使用、配置入口与算法边界：[MuZero README](../README.md)。
- 修改温度、FPU、LCB 或搜索预算：[配置](../muzero/config.py)、[共享搜索与推理](../muzero/mcts.py)。
- 修改串行／并行自我对弈：[训练入口](../muzero/trainer.py)、[并行调度](../muzero/muzero_parallel.py)、[对局记录与落子](../muzero/selfplay.py)。
- 修改网络、WDL 或策略辅助目标：[网络](../muzero/network.py)、[目标类型与变换](../muzero/targets.py)、[展开样本](../muzero/replay_buffer.py)、[损失与梯度缩放](../muzero/trainer.py)。
- 修改 shaped Dirichlet 噪声或对称增强：[utils.py](../muzero/utils.py)。
- 修改真实规则或观测编码：[井字棋](../envs/tictactoe.py)、[五子棋](../envs/gomoku.py)。
- 验证搜索与训练边界：[功能测试](../tests/test_search_training_features.py)、[噪声与 LCB 测试](../tests/test_noise_lcb.py)、[搜索测试](../tests/test_mcts.py)、[训练测试](../tests/test_trainer.py)、[并行测试](../tests/test_parallel.py)、[网络测试](../tests/test_network.py)。
