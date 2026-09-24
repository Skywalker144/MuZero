# MuZero

MuZero 的棋类实现，支持井字棋和自由规则五子棋。`h` 将真实观测编码为隐状态，`g` 根据动作推进隐状态，`f` 输出四组策略 logits 和胜／平／负 logits。默认使用跨对局批量推理自我对弈。

文档与模块入口见 [文档索引](docs/README.md)。

## 运行

在 `MuZero_V1/` 目录执行：

```bash
python -m pip install -r requirements.txt
python -m tictactoe.train
python -m gomoku.train
python -m tictactoe.play
python -m gomoku.play -n 100
python -m tictactoe.play -n 0
```

本机可使用 `conda activate pytorch`。`-n 0` 表示纯网络对弈；对弈使用 main policy 与 WDL 期望值。设备选择见 [auto_device](muzero/utils.py)。

## 配置

游戏入口为 [井字棋](tictactoe/train.py) 和 [五子棋](gomoku/train.py)，通过 `train_args` 覆盖默认参数。

- 搜索预算随机化、落子温度、root 温度、FPU、LCB、PUCT 和 shaped 根噪声：默认值、校验及调度统一定义在 [SearchConfig](muzero/config.py)。`num_simulations` 是 full 搜索预算，`cheap_search_visits` 是 cheap 搜索预算上限，均不含根评估。`cheap_search_prob=0` 关闭预算随机化。
- WDL 与辅助策略损失权重：[LossConfig](muzero/config.py)。四个平面的顺序与目标定义见 [targets.py](muzero/targets.py)。
- 网络规模、K 步展开、优化器、自我对弈后端、训练循环与 checkpoint：[trainer.py](muzero/trainer.py)。`parallel=False` 使用串行后端；`num_parallel_games` 控制并行对局数。
- 动态回放窗口与采样：[ReplayBuffer](muzero/replay_buffer.py)。

训练产物写入游戏入口指定的 `data_dir`，包括 `models/`、`checkpoints/`、训练图片和 CSV。`learn()` 自动加载该目录下的 checkpoint。模型权重和回放样本必须匹配当前网络及目标结构。

## 算法边界

- 只有搜索根节点使用真实棋盘的合法动作；树内按完整动作空间展开，不调用规则引擎判断终局。每手从真实观测重新编码，不跨手复用隐状态子树。
- 仅 full 训练搜索施加根温度与 Dirichlet 噪声。实际落子另用原始访问分布的温度调度；full 步的回放策略使用 LCB 修正后的分布。
- shaped Dirichlet 将噪声浓度的一半均匀分配，另一半按根先验形状分配；先验封顶按棋盘尺寸缩放。`shaped_dirichlet_noise=False` 使用均匀浓度。
- LCB 根据节点回传值的均值与平方均值修正根策略，使用最小访问比例门槛与方差正则。训练时只修正 full 步目标，对弈时用于选点；cheap 搜索与纯网络对弈不作 LCB 修正。`use_lcb_for_selection=False` 关闭。它不改变 PUCT，也不包含强制探索或目标剪枝；隐状态预测误差与相关回传样本使其不具备严格的统计置信保证。
- cheap 步仍记录完整动作、观测和终局 WDL；其 main／soft policy 损失权重为零。opponent 两个输出的权重取决于下一步的搜索模式。
- opponent policy 预测实际轨迹下一时刻的搜索策略。其输出不接收待选动作，也不代替动作条件化的 `g → f` 搜索。
- 四个策略输出参与训练，只有 main policy 用于搜索。soft 目标仅在目标时刻的合法动作内平滑。终局后的所有 policy 损失关闭；WDL 目标按行棋方交替交换胜负，和棋不变。
- 省略 reward 分支，value 直接监督真实终局胜／平／负。WDL 在搜索中转为胜概率减负概率，节点统计保持标量。
- 串行和并行共用搜索、推理、落子与对局记录逻辑；单并行对局的等价性由测试验证。

## 验证

安装 `pytest` 后，可执行与搜索及训练相关的测试：

```bash
python -m pytest tests/test_search_training_features.py tests/test_noise_lcb.py tests/test_mcts.py tests/test_network.py tests/test_parallel.py tests/test_trainer.py -q
```

自我对弈统计与损失用于检查训练流程；项目没有固定对手的棋力评估入口。
