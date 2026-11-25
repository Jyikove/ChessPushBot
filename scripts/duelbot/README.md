# 双车融合策略

按实车脚本的层次组织，供 MuJoCo 双车仿真使用。实车入口仍为 `scripts/main.py`。

| 层 | 文件 | 职责 |
|---|---|---|
| 输入 | `sim/runtime/duel_worker.py` | 接收车载图像、航向和四轮编码器读数 |
| 感知 | `perception/vision.py`、`perception/piece_values.py` | 棋子、棋面、对手推板、得分区线与墙脚检测 |
| 建模 | `modeling/pose.py` | 阵营坐标转换和累计位移更新 |
| 决策 | `decision/routes.py` | 根据棋子分值、时间及车道通行情况选择目标 |
| 规划 | `planning/policy.py` | 开局收棋、送分、回收、扫门及退出的状态机 |
| 执行 | `execution/drive.py` | 把车体速度指令换算为仿真四轮 PWM |

`settings.py` 保存现有策略常量。相机和模型继续使用 `sim/resources/` 中的资产。比赛命令和观测协议见 [双车仿真](../../sim/duel.md)。

```sh
python sim/duel.py --seed 77 --swap --video
```

入口 `sim/policies/behind_pile.py` 加载融合策略。它只使用策略协议提供的车载观测。融合策略尚未通过完整实车验证。
