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
CONFIG_DIR=configs/exp_consistency bash scripts/run.sh 10
CONFIG_DIR=configs/minimal_test bash scripts/run.sh
CONFIG_DIR=configs/exp_muzero_opt bash scripts/autoexp.sh --dry-run
ARM_GPUS=0,1 CONFIG_DIR=configs/exp_muzero_opt bash scripts/autoexp.sh
```

`PYTHON=/path/to/python` 选择环境；`BUILD_JOBS` 控制编译并发。`run.sh` 自动增量构建；也可单独执行 `bash scripts/build.sh`。显式指定 `MUZERO_BINARY` 时使用该可执行文件。

位置参数是绝对、排他的 iteration 上限；配置中 `max_iters = 0`、`max_time_seconds = 0` 表示关闭相应限制。时间限制在完整 iteration 边界检查，可能超出一轮。中断后以相同 `DATA_DIR` 重启，从最后完整提交的 iteration 继续；未提交轮的自对弈、训练、回放快照和耗时整轮作废。

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
| [match.cfg](configs/baseline/match.cfg) | 独立比赛搜索、平衡开局、棋规和并发参数 |

`CONFIG_DIR` 相对本目录解析；子配置的 `run.cfg` 用 `extends = baseline` 声明继承，父路径相对 `configs/` 解析，也接受绝对路径。只需要为变化的参数添加对应文件与分组；解析器会校验分组所属文件。满搜预算使用 `[search] full_search_visits`，相应环境变量为 `FULL_SEARCH_VISITS`。

优先级为：继承父配置、当前配置、叶子目录各文件的 `.local`、同名大写环境变量。例如 `[parallel] num_game_threads = 32` 可用 `NUM_GAME_THREADS=16` 覆盖。布尔值使用 `true` / `false`；配置不执行 shell。未知分组、未知参数和同层跨文件重复参数会在启动前报错。

评估使用同一目录继承链，仅读取 `eval.cfg` 和叶子目录的 `eval.cfg.local`，环境变量使用 `EVAL_` 前缀，例如 `EVAL_NUM_SEARCH_THREADS=8`。评估参数独立于训练及自对弈参数，配置与校验入口为 [config.py](python/muzero/config.py)。

`data_dir = auto` 按配置目录生成独立数据路径。机器参数可写入不跟踪的 `run.cfg.local`、`train.cfg.local` 等文件。改变算法、网络或训练超参数应使用新的数据目录；停止条件与设备、线程和批量推理参数可在恢复时调整。一致性损失的开关与权重也允许在原目录续训时调整，入口见下文。

多尺寸配置见 [baseline/env.cfg](configs/baseline/env.cfg)；混合规则可在配置的 `[env]` 中设置 `rules` 与 `rule_weights`。尺寸与规则分别按权重在每局开始时独立采样；网络画布由尺寸列表最大值派生。回放按窗口内的训练行权重采样，因此训练局面比例会受对局长度与行权重影响；目标和采样规则见 [算法边界](docs/algorithm.md)。

预设入口：[baseline](configs/baseline/run.cfg) 为增强基线；[exp_muzero](configs/exp_muzero/run.cfg) 为关闭增强的棋类 MuZero；[exp_baseline](configs/exp_baseline/run.cfg) 继承 15×15 至 11×11 的混合尺寸并使用增强网络；[minimal_test](configs/minimal_test/run.cfg) 为小规模五子棋验证。`exp_muzero` 的算法边界见 [algorithm.md](docs/algorithm.md)。

[exp_consistency](configs/exp_consistency/run.cfg) 直接继承 baseline，仅开启 EfficientZero 风格的自监督一致性损失。开关与损失权重见 [baseline/train.cfg](configs/baseline/train.cfg)，算法适配见 [algorithm.md](docs/algorithm.md)。沿用旧配置和 `DATA_DIR`，设置 `USE_CONSISTENCY_LOSS=true` 即可在旧 checkpoint 上开启并续训；主网络、Adam 状态、EMA、训练步数与回放均保留，仅初始化缺失的辅助头。设置为 `false` 可关闭；关闭后保存的 checkpoint 不保留辅助头，再开启会重新初始化辅助头。主网络架构与数据协议仍须匹配。

使用 `INIT_MODEL=/absolute/path/checkpoint.pt` 可在新目录导入完整 checkpoint 的权重、优化器、EMA 与训练步数；原目录的回放与 iteration 进度不会导入。共享随机初始化文件不含优化器状态时使用新优化器。

实验伞目录使用独立的 `exp.cfg`，其中只有 `[experiment]` 调度分组；每个实验臂仍使用五文件配置体系。所有调度参数由实验伞的 `exp.cfg` 定义，同名大写环境变量可覆盖。`run.sh` 仅读取 `[run]` 的停止条件。

`[experiment] max_time_seconds` 是每个 arm 独立的已提交轮次累计墙钟预算，包含本轮自我对弈、回放读取、训练、导出、指标准备和绘图；每轮在产物准备完成、最终指标与提交记录写入前取时间戳。排队、编译、启动恢复、共享初始化、暂停和作废轮次不计入。逐轮日志的 `elapsed_seconds` 对应该轮输出模型的累计时间，恢复计时使用同一个值。在完整 iteration 边界检查停止，因此可能超出一轮；等时评估应依据逐轮时间选择模型，不能假定最后模型恰好达到预算。若还设置了 `max_iters`，任一条件满足即停止。`0` 表示不设该项上限。

[exp_muzero_opt](configs/exp_muzero_opt) 包含 baseline、exp_muzero 和搜索消融配置；调度器自动发现伞目录下包含 `run.cfg` 的全部子目录。相同网络结构和种子的实验臂使用同一个初始化文件；baseline 与 exp_muzero 的输出头结构不同，按各自结构初始化。未设置 `ARM_GPUS` 时顺序执行；设置后每个槽位启动一臂，剩余实验排队。单个训练任务使用 `DEVICE` 指定的一张卡。

## 模型评估

浏览器人机对弈入口见 [Web UI](../web/README.md)。

实验间等墙钟 Elo、C++ 批量并行比赛、断点续测与绘图见 [Elo 入口](docs/elo.md)。

构建后，在本目录执行：

```bash
PYTHONPATH=python python3 -m muzero.evaluate \
  --model data/exp_baseline/models/latest.pt --size 11 --rule freestyle \
  --config-dir configs/exp_baseline --moves 60 61
```

`--moves` 和输出 `action` 使用实际棋盘的零起始行优先编号；网络画布从模型读取。该入口对给定局面执行一次搜索，输出 JSON；候选按 LCB 修正后的归一化选择权重排序，落子温度为零时选择首位。`root_value` 为当前行棋方的局面估值，`seconds` 包含首次推理初始化，不适合作为预热后的吞吐。接口参数见 [evaluate.py](python/muzero/evaluate.py)。

## 产物与验证

每轮完成后自动更新 `DATA_DIR/training.png`；续跑时从逐轮日志重建历史。绘图入口与指标面板见 [plots.py](python/muzero/plots.py)，也可执行 `bash scripts/plot.sh data/baseline` 手动重绘，无需启动训练。

`DATA_DIR` 下的 `selfplay/` 保存压缩的完整轨迹分片与每轮生成来源，`logs/state.json` 是整轮提交记录，指向对应代次的完整 checkpoint；`checkpoints/latest.pt` 是可重建的已提交 checkpoint 入口，`models/` 保存用于网络推理的各代 TorchScript 和 `latest.pt` 镜像，随机冷启动尚未训练时不导出模型。`replay/` 保存逐轮采样窗口快照，`logs/` 保存配置、恢复状态及逐轮 JSON 指标。回放窗口只限制训练采样范围，原始对局保留在磁盘。逐轮指标包含自对弈评估器、模型代次，以及按规则和尺寸分组的对局、回放、实际训练采样量与损失统计。数据、模型和 checkpoint 必须匹配 [协议版本](protocol.json)，运行目录还必须匹配 [run.py](python/muzero/run.py) 定义的提交格式；不兼容的产物须使用新数据目录重新训练。整轮提交与残留清理由 [storage.py](python/muzero/storage.py) 统一管理。checkpoint 与 `INIT_MODEL` 的训练进度字段由 [train.py](python/muzero/train.py) 定义；缺失训练来源的初始化产物不接受自动推断。

Linux 上的验证入口见 [tests](docs/testing.md)。
