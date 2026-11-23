# 树莓派实车推棋

`scripts/` 是脚本根目录。实车程序以一台 RASPBOT V2 的相机、串口 IMU 和 I²C 电机板为输入输出接口。相对开局航向和车体坐标由实车传感器提供。

| 模块 | 作用 |
| --- | --- |
| `chessbot/input/` | 相机与 IMU 独立采集，使用主机单调时钟记录接收时间 |
| `chessbot/perception/` | NCNN YOLO 棋子检测、跟踪、得分区、出发区与围栏识别 |
| `chessbot/modeling/` | 航向估计及跨层观测结构 |
| `chessbot/decision/` | 选择去程与回程的可见棋子 |
| `chessbot/planning/` | 推棋状态机、车体速度与航向闭环 |
| `chessbot/execution/` | 四轮补偿、PWM 限幅、I²C 下发与指令超时 |
| `chessbot/runtime.py` | 检查观测有效期并执行一周期 |
| `main.py`、`bin/chess` | 树莓派程序及终端入口 |

## 部署

在树莓派准备含 OpenCV、NumPy、NCNN、Pygame、pyserial、smbus 的 Python 环境。另备实车 416×320 NCNN 模型目录（`model.ncnn.param`、`model.ncnn.bin`）和实车四轮标定文件。运动指令需要有权限访问 I²C 总线，相机和串口同样需要设备权限。避免和其它脚本同时占用硬件。

从仓库根目录执行：

```sh
cp -n scripts/config/pi.example.json scripts/config/pi.local.json
mkdir -p "$HOME/.local/bin"
ln -s "$PWD/scripts/bin/chess" "$HOME/.local/bin/chess"
```

将 `$HOME/.local/bin` 加入 PATH。编辑 `scripts/config/pi.local.json`，把 `python`、`model_dir`、`wheel_profile` 指向实际文件。模板里的绝对路径可以用 `~`，相对路径以 `scripts/` 为基准。默认设备是 `/dev/video0` 和项目所用 USB 串口，按实际设置更新。相机内外参由 `camera_profile` 指定，附带初值对应 pan 77、tilt 0，现场需要复核。机器配置、标定结果、运行日志与录像留在本机。

```sh
chess test
chess run
chess record-run
```

- `chess test` 显示实时检测、状态和航向。按 X 可跳过当前状态，此模式禁用脱困和电机输出。
- `chess run` 按状态机驱动车辆，通过运行日志查看状态。
- `chess record-run` 运行相同控制，同时保存带检测叠图的车载 MP4。

三个模式都可在启动终端按 Ctrl+C 退出。测试窗口也支持 Ctrl+C。运动模式会按配置在完成一轮、超时或故障时停止。默认日志目录为 `~/.local/state/chessbot/runs/`。初次启动先用 test 检查图像、航向和场地识别，车体在 IMU 初始化时需保持静止。

## 坐标与时间

相机原始图像为 640×480，YOLO 输入为 416×320。观测框归一化为图像宽高的比例，目标与入口距离为车体坐标系的米。航向以开局为 0，逆时针为正，使用弧度。传感器消息时间是单调时钟接收秒数。

运动规划输出依次为前向速度、左向速度、逆时针角速度，单位为 m/s、m/s、rad/s。执行层按左前、左后、右前、右后输出有符号整数 PWM，适用四轮标定文件的上限。PWM 对速度的比例与转向臂保留为配置初值，实车连续速度映射需验证。
