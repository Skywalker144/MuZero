# MuZero V2

C++17 负责棋规、MuZero MCTS、自我对弈及共享 LibTorch 批量推理；Python 负责网络、展开目标、训练和持久化编排。支持 Freestyle、Standard、Renju，以及同一模型的多尺寸、多规则混合训练。各对局独立搜索，网络模式共享一个推理服务；随机冷启动入口见 [文档索引](docs/README.md)。每轮自我对弈完成后按数据门槛训练，再发布模型。

模块入口见 [文档索引](docs/README.md)。默认超参数的唯一来源是 [configs/baseline](configs/baseline)，实验调度参数见 [exp.cfg](configs/exp_muzero_opt/exp.cfg)。

## Linux 入口

需要 Python 3.10+、CMake 3.18+、支持 C++17 的编译器、zlib 开发库，以及与机器 CUDA 环境匹配的 PyTorch。构建脚本从当前 Python 的 PyTorch 包定位 LibTorch，不使用另一份独立 LibTorch。

在本目录执行：

```bash
python3 -m pip install -r requirements.txt
bash scripts/run.sh --dry-run
bash scripts/run.sh 10
CONFIG_DIR=configs/exp_muzero bash scripts/run.sh 10
CONFIG_DIR=configs/exp_baseline bash scripts/run.sh 10
CONFIG_DIR=configs/minimal_test bash scripts/run.sh
CONFIG_DIR=configs/exp_muzero_opt bash scripts/autoexp.sh --dry-run
ARM_GPUS=0,1 CONFIG_DIR=configs/exp_muzero_opt bash scripts/autoexp.sh
```

`PYTHON=/path/to/python` 选择环境；`BUILD_JOBS` 控制编译并发。`run.sh` 自动增量构建；也可单独执行 `bash scripts/build.sh`。显式指定 `MUZERO_BINARY` 时使用该可执行文件。

位置参数是绝对、排他的 iteration 上限；配置中 `max_iters = 0`、`max_time_seconds = 0` 表示关闭相应限制。时间限制在完整 iteration 边界检查，可能超出一轮。中断后以相同 `DATA_DIR` 重启，已发布分片内的对局和已经提交的训练不会重复消费；尚未发布的对局会重新生成。

## 配置与实验

配置采用小写参数和 INI 分组，按职责拆分：

| 文件 | 内容 |
|---|---|
| [run.cfg](configs/baseline/run.cfg) | 运行、设备、数据路径、停止条件 |
| [env.cfg](configs/baseline/env.cfg) | 棋盘尺寸、规则与每局采样权重 |
| [net.cfg](configs/baseline/net.cfg) | 网络规模、价值头和辅助策略头 |
| [train.cfg](configs/baseline/train.cfg) | 并行线程、批量推理、展开训练、batch、优化器、损失权重与回放窗口 |
| [selfplay.cfg](configs/baseline/selfplay.cfg) | 产量、搜索预算、噪声、FPU、LCB、落子与根温度 |
| [eval.cfg](configs/baseline/eval.cfg) | 独立评估搜索、树并行、落子与推理参数 |

`CONFIG_DIR` 相对本目录解析；子配置的 `run.cfg` 用 `extends = baseline` 声明继承，父路径相对 `configs/` 解析，也接受绝对路径。只需要为变化的参数添加对应文件与分组；解析器会校验分组所属文件。满搜预算使用 `[search] full_search_visits`，相应环境变量为 `FULL_SEARCH_VISITS`。

优先级为：继承父配置、当前配置、叶子目录各文件的 `.local`、同名大写环境变量。例如 `[parallel] num_game_threads = 32` 可用 `NUM_GAME_THREADS=16` 覆盖。布尔值使用 `true` / `false`；配置不执行 shell。未知分组、未知参数和同层跨文件重复参数会在启动前报错。

评估使用同一目录继承链，仅读取 `eval.cfg` 和叶子目录的 `eval.cfg.local`，环境变量使用 `EVAL_` 前缀，例如 `EVAL_NUM_SEARCH_THREADS=8`。评估参数独立于训练及自对弈参数，配置与校验入口为 [config.py](python/muzero/config.py)。

`data_dir = auto` 按配置目录生成独立数据路径。机器参数可写入不跟踪的 `run.cfg.local`、`train.cfg.local` 等文件。改变算法、网络或训练超参数应使用新的数据目录；停止条件与设备、线程和批量推理参数可在恢复时调整。

多尺寸配置见 [baseline/env.cfg](configs/baseline/env.cfg)；混合规则可在配置的 `[env]` 中设置 `rules` 与 `rule_weights`。尺寸与规则分别按权重在每局开始时独立采样；网络画布由尺寸列表最大值派生。回放按窗口内的训练行权重采样，因此训练局面比例会受对局长度与行权重影响；目标和采样规则见 [算法边界](docs/algorithm.md)。

预设入口：[baseline](configs/baseline/run.cfg) 为增强基线；[exp_muzero](configs/exp_muzero/run.cfg) 为关闭增强的棋类 MuZero；[exp_baseline](configs/exp_baseline/run.cfg) 继承 15×15 至 11×11 的混合尺寸并使用增强网络；[minimal_test](configs/minimal_test/run.cfg) 为小规模五子棋验证。`exp_muzero` 的算法边界见 [algorithm.md](docs/algorithm.md)。

实验伞目录使用独立的 `exp.cfg`，其中只有 `[experiment]` 调度分组；每个实验臂仍使用五文件配置体系。所有调度参数由实验伞的 `exp.cfg` 定义，同名大写环境变量可覆盖。`run.sh` 仅读取 `[run]` 的停止条件。

`[experiment] max_time_seconds` 是每个 arm 独立的累计运行时间上限，包含自我对弈、训练、导出和循环内 I/O；排队、编译、共享初始化和初始 checkpoint 准备不计入。耗时保存在该 arm 的恢复状态中，正常中断后续跑继续累计，在完整 iteration 边界检查停止，因此可能超出一轮。若还设置了 `max_iters`，任一条件满足即停止。`0` 表示不设该项上限。

[exp_muzero_opt/exp.cfg](configs/exp_muzero_opt/exp.cfg) 调度 [baseline 臂](configs/exp_muzero_opt/exp_baseline/run.cfg) 和 [exp_muzero 臂](configs/exp_muzero_opt/exp_muzero/run.cfg)，分别继承对应预设。相同网络结构和种子的实验臂使用同一个初始化文件；baseline 与 exp_muzero 的输出头结构不同，按各自结构初始化。未设置 `ARM_GPUS` 时顺序执行；设置后每个槽位启动一臂，剩余实验排队。单个训练任务使用 `DEVICE` 指定的一张卡。

## 模型评估

浏览器人机对弈入口见 [Web UI](../web/README.md)。

构建后，在本目录执行：

```bash
PYTHONPATH=python python3 -m muzero.evaluate \
  --model data/exp_baseline/models/latest.pt --size 11 --rule freestyle \
  --config-dir configs/exp_baseline --moves 60 61
```

`--moves` 和输出 `action` 使用实际棋盘的零起始行优先编号；网络画布从模型读取。该入口对给定局面执行一次搜索，输出 JSON；候选按 LCB 修正后的归一化选择权重排序，落子温度为零时选择首位。`root_value` 为当前行棋方的局面估值，`seconds` 包含首次推理初始化，不适合作为预热后的吞吐。接口参数见 [evaluate.py](python/muzero/evaluate.py)。

## 产物与验证

每轮完成后自动更新 `DATA_DIR/training.png`；续跑时从逐轮日志重建历史。绘图入口与指标面板见 [plots.py](python/muzero/plots.py)，也可执行 `bash scripts/plot.sh data/baseline` 手动重绘，无需启动训练。

`DATA_DIR` 下的 `selfplay/` 保存压缩的完整轨迹分片与每轮生成来源，`checkpoints/latest.pt` 是训练状态真源，`models/` 保存用于网络推理的各代 TorchScript 和 `latest.pt` 镜像，随机冷启动尚未训练时不导出模型。`replay/` 保存逐轮采样窗口快照，`logs/` 保存配置、恢复状态及逐轮 JSON 指标。回放窗口只限制训练采样范围，原始对局保留在磁盘。逐轮指标包含自对弈评估器、模型代次，以及按规则和尺寸分组的对局、回放、实际训练采样量与损失统计。数据、模型和 checkpoint 必须匹配 [协议版本](protocol.json)，不兼容的产物须使用新数据目录重新训练。checkpoint 与 `INIT_MODEL` 的训练进度字段由 [train.py](python/muzero/train.py) 定义；缺失训练来源的初始化产物不接受自动推断。

Linux 上的验证入口见 [tests](docs/testing.md)。
