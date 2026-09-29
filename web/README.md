# MuZero 对弈室

在仓库根目录运行：

```bash
bash web/webui.sh
```

默认使用 Conda `pytorch`，增量构建 V2 评估引擎，并在 <http://127.0.0.1:8765> 提供单用户对弈界面。棋局由服务保存，刷新页面可继续；关闭服务后不保留。

```bash
bash web/webui.sh --model MuZero_V2/data/exp_baseline/models/model_00000059.pt
EVAL_NUM_SEARCH_THREADS=8 EVAL_DEVICE=cuda:0 bash web/webui.sh --port 8765
```

启动选项以 [server.py](server.py) 为准。自动扫描数据目录中的分代 TorchScript 模型，不读取训练 checkpoint；其他模型可用 `--model` 指定。模型列表在服务启动时生成。加载后使用常驻模型，同一模型新局复用进程，模型更换在新局时生效。

评估配置与环境变量约定见 [V2 README](../MuZero_V2/README.md)，参数事实源为 [eval.cfg](../MuZero_V2/configs/baseline/eval.cfg)。界面的搜索次数在新局时生效。悔棋撤回到最近一次人类落子之前；AI 执黑的开局首手不会被单独撤回。模型忙碌时等待本步结束后再操作。

棋盘下方的策略热力图并排显示网络先验和搜索结果，可收起；搜索图可切换原始访问分布与 LCB 选择权重。两图使用同一个 AI 落子前的棋盘快照，颜色各自按最大值线性缩放，格内与悬停显示实际百分比。网络先验在根合法点上归一化，不含根温度和探索噪声；新局、悔棋或下一次人类落子会清除旧分析。

实现入口：

- [eval_session.h](../MuZero_V2/cpp/include/muzero/eval_session.h)：真实棋局、悔棋及评估会话。
- [eval_main.cpp](../MuZero_V2/cpp/src/eval_main.cpp)：单局面入口与 `--serve` 常驻命令协议。
- [engine.py](../MuZero_V2/python/muzero/engine.py)：模型进程生命周期与请求响应。
- [app.py](app.py)、[server.py](server.py)：单用户会话、版本检查和 HTTP 服务。
- [static](static)：棋盘和对弈界面。

真实模型测试，在仓库根目录运行：

```bash
PYTHONPATH=MuZero_V2/python:. \
MUZERO_WEB_TEST_MODEL="$PWD/MuZero_V2/data/exp_baseline/models/model_00000059.pt" \
MUZERO_TEST_DEVICE=cuda:0 conda run --no-capture-output -n pytorch \
  python -m unittest discover -s web/tests -p test_web.py
```

服务运行时可执行 [test_browser.py](tests/test_browser.py)，需要已安装 Playwright 和 Chromium。此测试会在指定服务上新建并完成对局，应使用测试服务：

```bash
MUZERO_WEB_TEST_URL=http://127.0.0.1:8765 conda run --no-capture-output -n pytorch \
  python -m unittest discover -s web/tests -p test_browser.py
```
