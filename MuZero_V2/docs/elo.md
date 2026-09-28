# 等墙钟 Elo

在 `MuZero_V2/` 执行：

```bash
conda run -n pytorch bash scripts/build.sh --target muzero_match
PYTHONPATH=python conda run -n pytorch python -m muzero.arena --dry-run
PYTHONPATH=python conda run --no-capture-output -n pytorch python -m muzero.arena
PYTHONPATH=python conda run -n pytorch python -m muzero.arena --fit-only
```

`--data` 指向包含各实验臂的目录，`--output` 指定独立评测目录；参数入口见 [arena.py](../python/muzero/arena.py)。中断后重复原命令续测。结果目录保存不可变的模型、配对、二进制和配置身份，逐开局、逐局原子提交；改变实验身份需另用输出目录。可增大 `--games` 补充新的成对开局，已完成棋局不会重跑；不允许缩减目标局数。对局的落子编号为实际棋盘行优先编号。

评测配置独立使用 [match.cfg](../configs/baseline/match.cfg)，沿 `run.cfg` 继承链解析，支持叶子目录的 `match.cfg.local` 与 `MATCH_` 环境变量。训练和单局分析的参数不会覆盖比赛配置。匹配双方需要相同网络画布，网络结构可以不同。C++ 比赛入口：[match_main.cpp](../cpp/src/match_main.cpp)，比赛组件：[match.h](../cpp/include/muzero/match.h)。

赛程选取已提交的历史推理模型，包含首个可用模型、按 iteration 间隔的模型和最后模型；组内近邻与跨组邻近墙钟配对共同构成连通比较图。墙钟口径见 [README 的配置与实验](../README.md#配置与实验)。图上的连线用于展示采样点，不代表测得了中间时刻棋力，也不向停止训练后的时段外推。

每个配对的局数必须为四的倍数。双方各生成一半开局，每个开局交换模型执色下两局；交换的是模型的黑白身份，不改变棋盘颜色或 Renju 规则。只接受尚未终局的成功平衡开局。没有认输或提前截断，按真实棋规结束。

`elo.json`、`elo.csv`、`elo.png`、`elo.svg` 由 [elo.py](../python/muzero/elo.py) 从全部赛果重建。评分使用联合 Bradley–Terry 得分模型，和棋记半分，拟合共同先手优势；anchor 规定相对 Elo 零点，不代表外部等级分。弱高斯正则避免全胜/全负导致无穷评分；其强度随结果记录。95% 区间按配对分层、以同开局的两局为单位 bootstrap，条件于本次选用的模型、开局生成规则和比赛协议。全胜配对的区间可能受正则与有限样本影响，结果会标记此情形。区间重叠本身不能作为组间差异检验。
