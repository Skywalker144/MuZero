# Linux 验证入口

在 `MuZero_V2/` 执行。以下命令按需运行，训练数据使用独立目录。

```bash
bash scripts/build.sh
ctest --test-dir build --output-on-failure
PYTHONPATH=python python3 -m unittest discover -s tests -p 'test_config.py'
PYTHONPATH=python python3 -m unittest discover -s tests -p 'test_pipeline.py'
PYTHONPATH=python python3 -m unittest discover -s tests -p 'test_iteration_store.py'
MUZERO_TEST_BINARY="$PWD/build/muzero_selfplay" MUZERO_TEST_DEVICE=cuda:0 PYTHONPATH=python \
  python3 -m unittest discover -s tests -p 'test_iteration_recovery.py'
PYTHONPATH=python python3 -m unittest discover -s tests -p 'test_bootstrap.py'
PYTHONPATH=python python3 -m unittest discover -s tests -p 'test_learning.py'
PYTHONPATH=python conda run -n pytorch python -m unittest discover -s tests -p 'test_consistency.py'
MUZERO_TEST_BINARY="$PWD/build/muzero_selfplay" MUZERO_TEST_DEVICE=cuda:0 PYTHONPATH=python \
  conda run -n pytorch python -m unittest discover -s tests -p 'test_consistency.py'
MUZERO_TEST_BINARY="$PWD/build/muzero_selfplay" PYTHONPATH=python \
  python3 -m unittest discover -s tests -p 'test_data_pipeline.py'
PYTHONPATH=python python3 -m unittest discover -s tests -p 'test_plots.py'
PYTHONPATH=python python3 -m unittest discover -s tests -p 'test_network.py'
PYTHONPATH=python python3 -m unittest discover -s tests -p 'test_multigame.py'
PYTHONPATH=python python3 -m unittest discover -s tests -p 'test_masked_learning.py'
MUZERO_TEST_BINARY="$PWD/build/muzero_selfplay" PYTHONPATH=python \
  python3 -m unittest discover -s tests -p 'test_native_pipeline.py'
MUZERO_TEST_BINARY="$PWD/build/muzero_selfplay" MUZERO_TEST_DEVICE=cuda:0 PYTHONPATH=python \
  python3 -m unittest discover -s tests -p 'test_native_bootstrap.py'
MUZERO_TEST_BACKEND_BINARY="$PWD/build/torch_backend_test" MUZERO_TEST_DEVICE=cuda:0 PYTHONPATH=python \
  python3 -m unittest discover -s tests -p 'test_native_backend.py'
CONFIG_DIR=configs/minimal_test DATA_DIR=data/smoke bash scripts/run.sh
```

- [rules_test.cpp](../cpp/tests/rules_test.cpp)：三种规则的成五、长连、满盘、Renju 三三/四四、成五优先级和假活三。
- [test_multigame.py](../tests/test_multigame.py)：混合分布和非法配置。
- [test_masked_learning.py](../tests/test_masked_learning.py)：混合回放、mask、画布补零不变性、优化器与导出。
- [core_test.cpp](../cpp/tests/core_test.cpp)：真实落子、终局、搜索预算、根合法动作与网络直出。
- [opening_test.cpp](../cpp/tests/opening_test.cpp)、[test_opening.py](../tests/test_opening.py)：反事实观测、开局重试、policy init 终局、轨迹前缀、白方起始与零样本回放；[test_native_opening.py](../tests/test_native_opening.py) 覆盖真实 LibTorch 零样本分片及恢复。
- [batcher_test.cpp](../cpp/tests/batcher_test.cpp)：并发请求返回及推理异常传播。
- [test_native_backend.py](../tests/test_native_backend.py)、[torch_backend_test.cpp](../cpp/tests/torch_backend_test.cpp)：真实 TorchScript 数值对照、混合请求、动态 batch 及 latent 存储独立性；`MUZERO_TEST_DEVICE` 选择 CPU 或 CUDA。
- [random_evaluator_test.cpp](../cpp/tests/random_evaluator_test.cpp)：随机状态递归、可复现性、输出范围及真实搜索掩码。
- [test_config.py](../tests/test_config.py)：配置继承、覆盖、循环、非法参数。
- [test_eval_config.py](../tests/test_eval_config.py)：评估配置继承、环境隔离、往返与参数校验。
- [search_parallel_test.cpp](../cpp/tests/search_parallel_test.cpp)：真实批量队列下的共享树预算、唯一扩展、取消、推理异常、虚拟损失与 LCB 选点。
- [test_native_eval.py](../tests/test_native_eval.py)：真实模型的 CUDA 评估、画布映射、候选排名和非法棋谱。
- [eval_session_test.cpp](../cpp/tests/eval_session_test.cpp)：对弈会话、画布转换、悔棋与终局；HTTP 和浏览器验证见 [Web UI](../../web/README.md)。
- [test_pipeline.py](../tests/test_pipeline.py)：产量计划、原子状态、独占锁、实验隔离。
- [test_iteration_store.py](../tests/test_iteration_store.py)：提交后入口恢复、清理中断与重复恢复、缺失已提交 checkpoint 检查。
- [test_iteration_recovery.py](../tests/test_iteration_recovery.py)：真实自对弈、优化器更新、checkpoint、导出与提交前中断的整轮回退和累计计时；训练入口及实验调度器的真实 SIGINT。
- [test_bootstrap.py](../tests/test_bootstrap.py)：训练步数持久化、共享随机初始化和预训练来源。
- [test_native_bootstrap.py](../tests/test_native_bootstrap.py)：无网络冷启动、首次训练切换、冷启动后采集预算与中断恢复、导出失败恢复、跨线程对局恢复及预训练模型；`MUZERO_TEST_DEVICE` 指定训练和网络推理设备。
- [test_data_pipeline.py](../tests/test_data_pipeline.py)：分片压缩与损坏检测、提交后续跑、写入失败传播、采样预取一致性与退出、回放快照、EMA 恢复及导出来源。
- [test_learning.py](../tests/test_learning.py)：动态 batch 导出、recurrent 推理、展开目标、D4 动作映射与优化器更新。
- [test_consistency.py](../tests/test_consistency.py)：多步真实观测与 D4 对齐、终局掩码、停止梯度、padding 不变性、旧配置与 checkpoint 的 Adam/EMA 恢复、开关切换、辅助头导出移除；设置原生可执行文件后验证 C++ 自对弈与原目录连续切换续训。
- [test_plots.py](../tests/test_plots.py)：指标曲线、冷启动、无对局轮次与图片原子写入；真实训练与续跑绘图见 [test_native_pipeline.py](../tests/test_native_pipeline.py)。
- [test_network.py](../tests/test_network.py)：独立网络深度与宽度、零残差块、状态递归、padding 不变性、梯度及 checkpoint 架构校验。
- [test_native_pipeline.py](../tests/test_native_pipeline.py)：真实 C++→回放→训练→导出流程，以及已提交 checkpoint 和模型入口的重建；混合棋规测试覆盖独立网络宽度，`MUZERO_TEST_DEVICE` 指定该测试的训练和推理设备。

仅验证不依赖 PyTorch 的 C++ 核心时，可使用独立构建目录：

```bash
cmake -S cpp -B build/core -DMUZERO_WITH_TORCH=OFF
cmake --build build/core --parallel 2
ctest --test-dir build/core --output-on-failure
```

评估定向验证：

```bash
PYTHONPATH=python python3 -m unittest discover -s tests -p '*config.py'
ctest --test-dir build/core -R '^(core|search_parallel|random_evaluator)$' --output-on-failure
MUZERO_TEST_EVAL_BINARY="$PWD/build/muzero_eval" MUZERO_TEST_DEVICE=cuda:0 PYTHONPATH=python \
  python3 -m unittest discover -s tests -p 'test_native_eval.py'
```

开局的定向验证与 SkyZero 源码对照：

```bash
ctest --test-dir build/core -R '^opening$' --output-on-failure
PYTHONPATH=python python3 -m unittest discover -s tests -p 'test_opening.py'
MUZERO_TEST_BINARY="$PWD/build/muzero_selfplay" MUZERO_TEST_DEVICE=cuda:0 PYTHONPATH=python \
  python3 -m unittest discover -s tests -p 'test_native_opening.py'
python3 scripts/verify_opening.py --source "$HOME/RL/SkyZero/SkyZero_V8.1"
```

[verify_opening.py](../scripts/verify_opening.py) 直接读取 SkyZero 的默认平衡开局及 policy init 函数，以相同随机流、真实 MuZero 棋规和确定性评估器对照落子、逐次观测、价值、重试次数及随机流消耗。它验证初始化算法，不验证两个模型的预测一致性，也不要求不同随机数发生器的 seed 对齐。需先构建 `build/core` 并通过配置解析 fixture；`--build-dir` 可指定其他构建目录。

搜索吞吐基准使用已导出的模型，配置中的画布必须与模型匹配：

```bash
PYTHONPATH=python python3 -m muzero.benchmark \
  --model data/baseline/models/latest.pt --device cuda:0 \
  --threads 32 64 128 --batch-sizes 32 64 --wait-us 0 100 1000 \
  --searches-per-thread 3 --repeats 3 --output build/benchmark.jsonl
```

入口及参数由 [benchmark.py](../python/muzero/benchmark.py) 定义。使用真实棋规、MCTS 和 LibTorch；每个线程先完成一次完整搜索预热，再对固定空棋盘重复搜索。JSONL 保存配置、模型和可执行文件哈希、吞吐及 initial/recurrent 实际 batch 分布。`seconds` 排除模型加载和预热，`backend_seconds` 是推理服务的主机耗时，不等同于 GPU 活跃时间。该负载用于比较搜索吞吐，不代表完整训练或对局产量；对照实验需使用相同模型、搜索预算和空闲 GPU。

KataGo 对齐验证：

```bash
conda run -n pytorch python scripts/verify_alignment.py --source /home/sky/RL/SkyZero/SkyZero_V8.1
PYTHONPATH=python conda run -n pytorch python -m unittest discover -s tests -p 'test_alignment.py'
conda run -n pytorch ctest --test-dir build -R '^(alignment|core|search_parallel)$' --output-on-failure
```

[verify_alignment.py](../scripts/verify_alignment.py) 直接抽取指定源码的策略与 WDL 损失函数验证数值和梯度，并对照噪声分布、surprise 行权重和访问权重反解函数；[alignment_test.cpp](../cpp/tests/alignment_test.cpp) 覆盖根强制探索与搜索目标；[test_alignment.py](../tests/test_alignment.py) 覆盖损失系数、行采样与 soft 目标。

Elo 定向验证：

```bash
conda run -n pytorch bash scripts/build.sh --target muzero_match match_test
conda run -n pytorch ctest --test-dir build -R '^match$' --output-on-failure
PYTHONPATH=python conda run -n pytorch python -m unittest discover -s tests -p 'test_arena.py'
PYTHONPATH=python conda run -n pytorch python -m unittest discover -s tests -p 'test_elo.py'
MUZERO_TEST_MATCH_BINARY="$PWD/build/muzero_match" \
MUZERO_TEST_MODEL_A=/absolute/path/model_a.pt MUZERO_TEST_MODEL_B=/absolute/path/model_b.pt \
MUZERO_TEST_DEVICE=cuda:0 PYTHONPATH=python conda run -n pytorch \
  python -m unittest discover -s tests -p 'test_native_match.py'
```

[match_test.cpp](../cpp/tests/match_test.cpp) 验证真实棋规和搜索下的双模型路由、latent 归属、完整棋谱重放和取消；[test_native_match.py](../tests/test_native_match.py) 用真实 TorchScript/GPU 验证开局配额、换先、中断后逐局续测及幂等恢复。[test_arena.py](../tests/test_arena.py) 覆盖配置隔离、已提交模型选点、连通赛程和原子结果存储；[test_elo.py](../tests/test_elo.py) 覆盖已知棋力与先手优势恢复、断连拒绝、全胜正则及成对开局 bootstrap。
