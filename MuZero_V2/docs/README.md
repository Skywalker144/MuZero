# 文档索引

- 使用与运行入口：[README](../README.md)。
- 默认参数与配置解析：[baseline](../configs/baseline)、[config.py](../python/muzero/config.py)。
- 原始棋类 MuZero 预设：[muzero](../configs/muzero)；11×11 实验基线：[exp_baseline](../configs/exp_baseline)。
- 真实棋规与观测：[game.h](../cpp/include/muzero/game.h)。
- MuZero 搜索、FPU、根噪声与 LCB：[search.cpp](../cpp/src/search.cpp)。
- 对局线程与共享推理队列：[selfplay_main.cpp](../cpp/src/selfplay_main.cpp)、[batcher.cpp](../cpp/src/batcher.cpp)、[torch_backend.cpp](../cpp/src/torch_backend.cpp)。
- 网络与 TorchScript：[network.py](../python/muzero/network.py)、[train.py](../python/muzero/train.py)。
- 完整对局协议与展开目标：[record.h](../cpp/include/muzero/record.h)、[replay.py](../python/muzero/replay.py)。
- baseline 与 muzero 对比实验：[exp_muzero_opt](../configs/exp_muzero_opt/exp.cfg)。
- iteration 恢复与实验调度：[run.py](../python/muzero/run.py)、[experiment.py](../python/muzero/experiment.py)。
- 算法边界：[algorithm.md](algorithm.md)。
- Linux 验证入口：[testing.md](testing.md)。
