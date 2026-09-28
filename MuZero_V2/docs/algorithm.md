# 算法边界

本项目使用无 reward 分支的棋类 MuZero，真实棋规由每局采样的规则决定，规则事实源见 [game.h](../cpp/include/muzero/game.h) 与 [rules.cpp](../cpp/src/rules.cpp)。Renju 禁手允许落子并立即判黑负；恰好五子优先于同手其他方向的禁手。根节点由真实观测编码，树内仅通过动作条件化的 dynamics 推进隐状态；树内不执行真实棋规，不使用真实终局检测，也不跨实际落子复用隐状态子树。占用位置的合法动作掩码仅作用于根节点；实际棋盘以外的画布位置在所有深度均不属于动作空间。

网络输入包含规则、执棋颜色、有效棋盘与 Renju 禁手特征，通道编号由 [protocol.json](../protocol.json) 定义。有效棋盘 mask 随隐状态原样传递，归一化、池化和策略输出排除画布外位置。树内不重新计算禁手或占用状态。

随机冷启动使用 [RandomEvaluator](../cpp/src/random_evaluator.cpp) 替代神经网络评估；树内状态只用于可复现的随机输出，不执行真实棋规。搜索和训练目标沿用同一流程，随机输出参与搜索和采样权重，价值监督仍取真实终局。首次训练提交后的切换、初始化训练来源及 iteration 恢复由 [run.py](../python/muzero/run.py) 管理，配置入口见 [selfplay.cfg](../configs/baseline/selfplay.cfg)。

网络自对弈的开局初始化采用 SkyZero V8.1 默认 KataGomo 路径，源码入口见 [opening.cpp](../cpp/src/opening.cpp)。初始化使用真实局面的 initial 推理，包含换行棋方的反事实观测，不使用 dynamics 或 MCTS。开局动作保存为轨迹前缀，不进入策略或价值监督；初始化终局保留对局记录但不产生训练样本。随机冷启动跳过网络开局初始化。记录与回放入口见 [文档索引](README.md)。

增强基线的 WDL 按当前行棋方监督终局结果，搜索使用胜概率减负概率，回传逐层取反。网络同时学习 main、soft、opponent、soft-opponent 四个策略头；搜索仅消费 main。训练行权重与 surprise 重分配见 [training_targets.cpp](../cpp/src/training_targets.cpp)，完整轨迹保留 cheap 搜索供 dynamics 展开与 opponent 监督。回放按行权重采样起点，展开步按当前行权重相对回放平均权重缩放全部损失；opponent 掩码只取决于下一步目标是否存在。soft 目标的范围是有效棋盘。终局之后继续交替 WDL 目标，关闭策略监督。

full 自我对弈搜索施加根温度与 Dirichlet 噪声；根节点强制探索和事后访问权重回调见 [search.cpp](../cpp/src/search.cpp)。自对弈落子使用回调后的分布和落子温度，训练策略目标另行应用 LCB。NN policy temperature 在 [TorchBackend](../cpp/src/torch_backend.cpp) 统一处理，覆盖开局、根节点及 dynamics 推理，不改变训练 logits。LCB 是选择启发式，隐状态误差及相关回传样本使它不构成严格的统计置信保证。没有引入 AlphaZero 的树内规则搜索、跨手树复用。

[exp_muzero 预设](../configs/exp_muzero) 使用单策略头、tanh 标量价值和 MSE 价值损失。关闭 cheap 搜索、surprise 加权、根强制探索、策略目标剪枝、FPU、LCB、shaped 噪声、NN 与根策略温度调整、辅助策略头、WDL、D4 增强、动态回放窗口及自适应产量；使用固定回放窗口和固定每轮对局数。选点温度按手数阈值切换。PUCT、均匀 Dirichlet 根噪声、隐状态归一化和展开梯度缩放属于保留的 MuZero 核心。

该预设针对本项目的棋类模型，不包含通用 MuZero 的 reward 分支、Atari 标量分布支持或 reanalyse。并行批量推理、C++ 执行和 checkpoint 编排属于运行设施，各配置共用。

实现入口与参数定义统一见 [文档索引](README.md)。

[exp_consistency](../configs/exp_consistency) 采用 [EfficientZero §4.1](https://arxiv.org/abs/2111.00210) 的多步表征一致性目标：Dynamics 展开状态与相同步数的真实观测编码，经共享 projector 和预测分支的 predictor 对齐，目标分支停止梯度。损失使用 `1 - cosine`，与负余弦仅差常数。为保留棋盘位置并支持混合尺寸，投影采用逐位置的 1×1 卷积 MLP 和现有 MaskedNorm，余弦在有效位置的完整特征上计算；这是本项目的棋盘适配，不是 Atari 网络结构的逐层复刻。

真实轨迹、动作和策略共用同一 D4 变换；一致性损失沿用展开行权重和梯度缩放。分片未保存最终落子后的观测，因此仅监督存在真实后续观测的展开位置，不重建终局特征，也不监督终局后的虚构状态。辅助头只参与训练，TorchScript 导出移除辅助头。实现入口见 [replay.py](../python/muzero/replay.py)、[network.py](../python/muzero/network.py) 与 [train.py](../python/muzero/train.py)。
