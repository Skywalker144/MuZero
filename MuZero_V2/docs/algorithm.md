# 算法边界

本项目使用无 reward 分支的棋类 MuZero，真实棋规由每局采样的规则决定，规则事实源见 [game.h](../cpp/include/muzero/game.h) 与 [rules.cpp](../cpp/src/rules.cpp)。Renju 禁手允许落子并立即判黑负；恰好五子优先于同手其他方向的禁手。根节点由真实观测编码，树内仅通过动作条件化的 dynamics 推进隐状态；树内不执行真实棋规，不使用真实终局检测，也不跨实际落子复用隐状态子树。占用位置的合法动作掩码仅作用于根节点；实际棋盘以外的画布位置在所有深度均不属于动作空间。

网络输入包含规则、执棋颜色、有效棋盘与 Renju 禁手特征，通道编号由 [protocol.json](../protocol.json) 定义。有效棋盘 mask 随隐状态原样传递，归一化、池化和策略输出排除画布外位置。树内不重新计算禁手或占用状态。

随机冷启动使用 [RandomEvaluator](../cpp/src/random_evaluator.cpp) 替代神经网络评估；树内状态只用于可复现的随机输出，不执行真实棋规。搜索和训练目标沿用同一流程，随机价值只参与搜索回传，价值监督仍取真实终局。首次训练提交后的切换、初始化训练来源及 iteration 恢复由 [run.py](../python/muzero/run.py) 管理，配置入口见 [selfplay.cfg](../configs/baseline/selfplay.cfg)。

网络自对弈的开局初始化采用 SkyZero V8.1 默认 KataGomo 路径，源码入口见 [opening.cpp](../cpp/src/opening.cpp)。初始化使用真实局面的 initial 推理，包含换行棋方的反事实观测，不使用 dynamics 或 MCTS。开局动作保存为轨迹前缀，不进入策略或价值监督；初始化终局保留对局记录但不产生训练样本。随机冷启动跳过网络开局初始化。记录与回放入口见 [文档索引](README.md)。

增强基线的 WDL 按当前行棋方监督终局结果，搜索使用胜概率减负概率，回传逐层取反。网络同时学习 main、soft、opponent、soft-opponent 四个策略头；搜索仅消费 main。cheap 搜索仍提供价值目标，策略监督权重关闭；opponent 目标的监督权重取下一步的搜索模式。终局之后继续交替 WDL 目标，关闭策略监督。

full 自我对弈搜索施加根温度与 Dirichlet 噪声；落子从原始访问分布经过落子温度后采样，训练策略目标可应用 LCB。LCB 是选择启发式，隐状态误差及相关回传样本使它不构成严格的统计置信保证。没有引入 AlphaZero 的树内规则搜索、跨手树复用、强制探索或策略目标剪枝。

[muzero 预设](../configs/muzero) 使用单策略头、tanh 标量价值和 MSE 价值损失。关闭 cheap 搜索、FPU、LCB、shaped 噪声、根策略温度调整、辅助策略头、WDL、D4 增强、动态回放窗口及自适应产量；使用固定回放窗口和固定每轮对局数。选点温度按手数阈值切换。PUCT、均匀 Dirichlet 根噪声、隐状态归一化和展开梯度缩放属于保留的 MuZero 核心。

该预设针对本项目的棋类模型，不包含通用 MuZero 的 reward 分支、Atari 标量分布支持或 reanalyse。并行批量推理、C++ 执行和 checkpoint 编排属于运行设施，各配置共用。

实现入口与参数定义统一见 [文档索引](README.md)。
