# 高速公路实时车辆检测

面向单路固定机位的交通视频分析。支持本地录像和 RTSP，提供实时画面叠加、车辆轨迹和分车道统计。日常运行用 `work.py`，复跑点位 1 的固定两分钟测试用 `test_point1.py`。

## 项目功能

| 功能 | 实现和结果 |
| --- | --- |
| 车辆识别与跟踪 | YOLOv8s 检测小汽车、摩托车、客车和货车，ByteTrack 在连续视频帧中关联车辆并分配轨迹 ID。 |
| 车道归属 | 基于固定机位标注的车道多边形判断车辆所在车道，连续 3 帧确认跨车道变化。 |
| 车速估算 | 将车辆接地点投影到米制路面坐标，结合视频时间戳估算速度，单位为 km/h。轨迹长度、时间间隔和有效标定范围都会影响读数。 |
| 距离估算 | 可选的 Depth Anything V2 Small 室外米制模型。默认报告估算的光轴深度，配置相机内参后可换算为到车身采样点的直线距离。 |
| 分车道密度 | 用车道检测区内的车辆数除以该区间的长度，单位为辆/km/车道。 |
| 过线计数与流量 | 按各车道的有向虚拟线统计通过车辆，记录穿线时间、累计通过量、最近 60 秒通过量和折算流率。 |
| 预览与导出 | 实时预览车辆框、ID、车道和统计数据。常规运行保存一份 CSV，点位 1 测试另外录制一段标注 MP4。 |

## 核心文件

| 文件 | 用途 |
| --- | --- |
| [work.py](work.py) | 常规入口。调度读取、检测、跟踪、测量、统计、预览和导出。 |
| [test_point1.py](test_point1.py) | 点位 1 两分钟测试入口。内置该机位的近似标定、复跑参数、视频录制和 ID 连续性核对。 |
| [camera.example.json](camera.example.json) | 机位配置模板，包含车道多边形、计数线、检测区长度和路面标定点。 |
| [requirements.txt](requirements.txt) | Python 依赖清单。 |
| [video/__init__.py](video/__init__.py) | PyAV 解码、视频时间戳、RTSP 重连和最新帧缓冲。 [PyAV 时间戳文档](https://pyav.org/docs/stable/api/time.html) |
| [vehicle_detection/__init__.py](vehicle_detection/__init__.py) | YOLOv8s 推理和 COCO 车辆类别筛选。 [YOLOv8 文档](https://docs.ultralytics.com/models/yolov8/) |
| [tracking/__init__.py](tracking/__init__.py) | ByteTrack 关联及同一轨迹的车型加权投票。 [ByteTrack，ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf) |
| [lanes/__init__.py](lanes/__init__.py)、[calibration/__init__.py](calibration/__init__.py) | 车道归属、路面单应性和可选去畸变。 [路侧相机标定，BMVC 2014](https://bmva-archive.org.uk/bmvc/2014/files/paper013.pdf) |
| [speed/__init__.py](speed/__init__.py) | 对短窗地面轨迹拟合 Theil–Sen 斜率，估算车速。 [Theil–Sen 方法](https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/ts_fit.htm) |
| [distance/__init__.py](distance/__init__.py) | 异步推理室外米制深度模型，车辆共享整帧深度图。 [Depth Anything V2，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/26cfdcd8fe6fd75cc53e92963a656c58-Paper-Conference.pdf) · [官方权重](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf) |
| [density/__init__.py](density/__init__.py)、[flow/__init__.py](flow/__init__.py) | 区间密度、穿线时间插值、过线去重和流量窗口。 [FHWA 高速公路手册](https://ops.fhwa.dot.gov/freewaymgmt/publications/frwy_mgmt_handbook/chapter1_02.htm) · [AI City Challenge，CVPR 2021 Workshop](https://openaccess.thecvf.com/content/CVPR2021W/AICity/html/Naphade_The_5th_AI_City_Challenge_CVPRW_2021_paper.html) |
| [output/__init__.py](output/__init__.py) | 统一 CSV 导出和共用的画面叠加渲染。 |

## 安装与常规运行

建议使用 Python 3.10–3.12。以下命令在 Windows PowerShell 中执行。

```powershell
Set-Location D:\env\detection_workspace
python -m venv .venv
$python = ".\.venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
```

使用 RTX 50 系列 GPU 时，可先安装 CUDA 12.8 的 PyTorch。 [PyTorch 安装文档](https://pytorch.org/get-started/locally/)

```powershell
& $python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

安装项目依赖。CPU 环境可以直接从这一步开始，并在运行时指定 `--device cpu`。

```powershell
& $python -m pip install -r requirements.txt
```

首次使用会下载所需的官方权重。离线运行可用 `--weights` 指定本地 `yolov8s.pt`。启用测距时，可用 `--depth-model` 指定包含 `config.json`、`preprocessor_config.json` 和 `model.safetensors` 的模型目录。

先将示例视频路径替换为实际路径。第一条命令可验证车辆检测和跟踪，配置机位后再查看车道、车速、密度和流量。

```powershell
& $python work.py --source "D:\videos\highway.mp4" --no-depth
& $python work.py --source "D:\videos\highway.mp4" --camera camera.json
& $python work.py --source "rtsp://camera.example/stream" --camera camera.json --no-preview
```

- `--device` 默认自动选择 CUDA 或 CPU。CUDA 下检测和测距使用 FP16。
- `--imgsz` 默认 640。测距默认最高 GPU 5 Hz、CPU 2 Hz，可用 `--depth-hz` 调整，或用 `--no-depth` 关闭。
- 预览窗口按 **Q** 或 **Esc** 退出。`--no-preview` 用于命令行运行，CSV 仍随运行写入。画面延迟从解码到渲染前计算。
- RTSP 使用最新解码帧，断流后自动重连。重连或时间戳跳变时重置跟踪和流量窗口，保留运行累计通过量。RTSP 缺少源时间戳时，车速留空。

### 配置其他机位

保存首帧，复制模板，再按实际道路标注 `camera.json`。模板的坐标仅用于说明格式。

```powershell
& $python work.py --source "D:\videos\highway.mp4" --snapshot reference.png
Copy-Item camera.example.json camera.json
```

| 配置项 | 填写方法 |
| --- | --- |
| `image_size` | 标注截图的宽和高，原点在左上角。等比缩放时程序会换算坐标。裁剪、变焦或移动机位后须重做标定。 |
| `lanes[].id`、`lanes[].polygon` | 车道编号与按边界顺序排列的像素坐标。 |
| `lanes[].length_m` | 多边形覆盖区间沿车道的长度，单位为米，用于密度。 |
| `lanes[].gate.line`、`lanes[].gate.downstream` | 横跨车道的线段和位于车辆行驶方向一侧的参考点，用于有向过线计数。 |
| `road_calibration.points` | 在同一近似平坦的道路上，标注至少 4 个非共线的像素点 `image: [u, v]`，以及对应的实际路面坐标 `road_m: [x, y]`。可用独立测量点填写 `check_points`，程序会报告路面位置 RMSE。 |
| `intrinsics` | 可选。 `fx_fy_cx_cy` 按 `[fx, fy, cx, cy]` 填写，可附上 OpenCV 畸变系数 `distortion`。 |

测速和密度的绝对精度依赖现场尺度。可用实测位置核对距离、用雷达记录核对速度、用人工逐车核对通过量。 [BrnoCompSpeed 测速验证参考](https://arxiv.org/abs/1702.06441)

## 如何测试

在项目根目录用同一个 Python 环境运行点位 1 测试。脚本默认读取 `点位1 8-9点（中间大雾）.mp4`，路径定义在 `test_point1.py` 的 `DEFAULT_SOURCE`。原录像移到其他位置时，加上 `--source`。

```powershell
& $python test_point1.py
& $python test_point1.py --source "D:\videos\point1.mp4" --device cuda
```

- 测试只读原录像的前 **120 秒**，检查原始 **1920×1080、20 FPS** 的画幅和时间序列。该机位的四车道标定已经内置，近似检测区长度为 **90.219 米/车道**。
- 测试使用 YOLOv8s 的 960 输入尺寸及跨类别去重，复用常规跟踪和统计逻辑。该测试关闭模型测距，速度与密度按内置近似标定估算。
- 脚本校验 MP4 为 **2,400 帧、120 秒**，CSV 覆盖整个区间，并在终端报告分车道穿线数、已分配 ID 数、逐帧输出 ID 数、采样 ID 数，以及过线前后各 **0.5 秒**的 ID 连续性。
- 用本项目当前配置验证这段录像时，共记录 **10 次过线**，10 次均通过临线 ID 连续性核对。

## 输出什么

| 运行方式 | 默认输出 |
| --- | --- |
| `work.py --source ...` | 当前运行目录下的 `runs/traffic_results_时间戳.csv`，每次检测运行一份。 |
| `test_point1.py` | 项目下的 `runs/point1_test_时间戳/` 内一份 CSV 和一段同名的 **1280×720、20 FPS** 标注 MP4。视频叠加车辆框、ID、估算车速、车道、密度及累计通过量。 |
| `work.py --source ... --snapshot reference.png` | 保存首帧图片后退出，用于机位标注。 |

`--output` 可改变常规检测或点位 1 测试的结果目录。

### 如何读 CSV

| `record_type` | 每行含义 |
| --- | --- |
| `vehicle` | 每秒采样的单车记录。车辆 ID、车速、车道和当时所属车道的密度在同一行。 |
| `lane` | 某个采样时刻无车的车道记录。含车辆数及已有的密度、流量统计，车辆字段留空。 |
| `crossing` | 一次车辆穿线事件，含穿线时间、ID、车型和车道。统计区间内通过量时，按这一类型计数。 |

| 字段 | 含义 |
| --- | --- |
| `segment`、`video_s` | 连续视频段编号和视频时间（秒）。在同一次运行内，用 `segment + track_id` 标识轨迹。 |
| `track_id`、`vehicle_class`、`confidence`、`x1`、`y1`、`x2`、`y2` | 轨迹编号、车型、置信度和输入画面中的检测框坐标。候选轨迹也会占用编号，短轨迹有时落在两次采样之间，因此 CSV 里的 ID 可以跳号。 |
| `lane`、`speed_kmh` | 车道编号和估计车速（km/h）。 |
| `distance_m`、`distance_type` | 模型估计距离（米）及类型，`optical_axis_depth` 表示光轴深度，`slant_range` 表示直线距离。 |
| `vehicles_in_zone`、`density_veh_per_km_per_lane` | 该时刻车道检测区内的车辆数和密度（辆/km/车道）。 |
| `crossings_total`、`crossings_in_window`、`flow_veh_per_hour`、`flow_window_s` | 该车道累计通过量、窗口内通过量、折算流率（辆/h）和窗口长度。流率在得到完整 60 秒窗口后出值。 |

按车道汇总密度或流量时，在 `vehicle` 和 `lane` 行中按 `segment + video_s + lane` 取一组统计。同一时刻同车道的多辆车共享这些数值。缺少对应标定、轨迹样本不足或功能关闭时，相关字段留空。
