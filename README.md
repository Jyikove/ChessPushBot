# ChessPushBot

基于 RASPBOT V2 和树莓派 5 的视觉自主推棋项目，目标是在竞赛场地内识别、收集并推送棋子进入得分区。

当前仓库仅初始化目录与五类模型资产，尚未迁入仿真、VIO、控制或部署程序。

## 目录

```text
ChessPushBot/
├── README.md
├── AGENTS.md
├── docs/               # 项目与接口文档
├── sim/                # 仿真程序
├── assets/
│   ├── field/          # 场地模型、纹理与尺寸
│   ├── robot/          # 小车原厂 STEP 与仿真视觉网格
│   ├── pusher/         # 三段圆弧推板、建模源码与打印文件
│   ├── imu_mount/      # IMU 转接板、建模源码与打印文件
│   ├── chess/          # 带实体文字的棋子及碰撞模型
│   └── manifests/      # 文件清单与校验值
├── scripts/            # 启动、部署、采集、标定命令
└── data/               # 本地原始数据，不提交
```

空目录由 `.gitkeep` 保留；`.gitignore` 排除本地数据和配置。

## 获取与使用

```sh
git clone https://github.com/Jyikove/ChessPushBot.git
cd ChessPushBot
```

模型文件直接保存在 Git 中，无需额外下载或 Git LFS。STEP 可用 CAD 软件打开，STL 可导入切片软件，GLB 可用于三维预览。运行与部署入口将在相关代码迁入后补充。

| 资产 | 入口 | 单位及说明 |
| --- | --- | --- |
| 小车 | `assets/robot/raspbot.step` | 原厂 STEP，毫米；存在原始几何导入问题，仿真网格为外形近似 |
| 小车仿真网格 | `assets/robot/simulation.json` | OBJ 为米；配置记录装配坐标变换、推板碰撞分块及简化假设 |
| 场地 | `assets/field/field.obj`、`field.png`、`field.json` | 3×2 米；纹理与 OBJ 的 UV 配合使用，渲染器需显式绑定纹理 |
| 推板 | `assets/pusher/pusher.step`、`pusher.stl` | 毫米；宽 195、壁厚 3、安装片 1、离地 3；OBJ 是已转换到仿真车体坐标的米制网格 |
| IMU 转接板 | `assets/imu_mount/imu_mount.step`、`imu_mount.stl` | 毫米；板厚 2、固定孔距 49×58、支座外径 6；搭配 14 mm 铜柱 |
| 棋子 | `assets/chess/chess.json` | 14 种棋面、按数量组成 32 枚；直径 47 mm、高 15 mm；STEP/STL 毫米、GLB 场景变换后为米 |

棋子双面文字已建模，黑方炮使用火字旁“炮”。碰撞模型简化为圆柱；质量与摩擦未实测。场地绘图区域与既有实验评分区存在差异，具体坐标保留在 `field.json` 中，不能直接视为统一后的比赛判分配置。

原厂车体来自 Yahboom RASPBOT V2 资料；场地外观依据项目提供的尺寸图；棋子按所购产品尺寸近似建模，字形和装饰未逐项实测。推板和转接板为本项目设计。未对厂商模型或字体单独授予开源许可。

## 重新建模

推板与转接板目录分别提供 `build_pusher.py` 和 `build_imu_mount.py`，依赖 `numpy`、`cadquery-ocp`、`trimesh`，在模型目录运行对应脚本。已有 STEP/STL 可直接使用，无需先执行脚本。建模脚本从现有设计整理，迁移后已做语法检查，尚未在新环境重新生成 CAD。

校验清单见 `assets/manifests/models.json`。模型采用稳定名称，修改历史由 Git 保存。
