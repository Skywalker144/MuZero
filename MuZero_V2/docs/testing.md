# Linux 验证入口

在 `MuZero_V2/` 执行。以下命令按需运行，训练数据使用独立目录。

```bash
bash scripts/build.sh
ctest --test-dir build --output-on-failure
PYTHONPATH=python python3 -m unittest discover -s tests -p 'test_config.py'
PYTHONPATH=python python3 -m unittest discover -s tests -p 'test_pipeline.py'
PYTHONPATH=python python3 -m unittest discover -s tests -p 'test_learning.py'
MUZERO_TEST_BINARY="$PWD/build/muzero_selfplay" PYTHONPATH=python \
  python3 -m unittest discover -s tests -p 'test_native_pipeline.py'
CONFIG_DIR=configs/minimal_test DATA_DIR=data/smoke bash scripts/run.sh
```

- [core_test.cpp](../cpp/tests/core_test.cpp)：真实落子、终局、搜索预算、根合法动作与网络直出。
- [batcher_test.cpp](../cpp/tests/batcher_test.cpp)：并发请求返回及推理异常传播。
- [test_config.py](../tests/test_config.py)：配置继承、覆盖、循环、非法参数。
- [test_pipeline.py](../tests/test_pipeline.py)：产量计划、原子状态、独占锁、实验隔离。
- [test_learning.py](../tests/test_learning.py)：动态 batch 导出、recurrent 推理、展开目标、D4 动作映射与优化器更新。
- [test_native_pipeline.py](../tests/test_native_pipeline.py)：真实 C++→回放→训练→导出流程，以及训练 checkpoint 提交后、iteration 状态提交前的恢复。

仅验证不依赖 PyTorch 的 C++ 核心时，可使用独立构建目录：

```bash
cmake -S cpp -B build/core -DMUZERO_WITH_TORCH=OFF
cmake --build build/core --parallel 2
ctest --test-dir build/core --output-on-failure
```
