# 动车组受电弓网视频故障监测系统 (Gongwang Monitor)

本项目是动车组受电弓网视频故障实时监测子系统，基于统一插件化分层架构（Layered Plugin Architecture），融合 **YOLOv11**、**FastAD (开集反常度量)** 与 **HRNet (关键点姿态回归)** 三大算法引擎，实现对弓网 6 大类典型故障的高精度、低时延实时监测。

---

## 1. 监测故障分类与算法映射

| 故障类别 | 算法引擎 | 调度模式 | 算法机理说明 |
| :--- | :---: | :---: | :--- |
| **1. 大火花 (`spark`)** | YOLOv11 | 逐帧 (30 FPS) | 滑板受流接触区强光与火花目标检测，时空尺度积分 |
| **2. 画面脏污 (`dirt`)** | FastAD | 周期 (1.0s) | 图像空域网格化时序 EMA 衰减累积，识别镜头附着物 |
| **3. 悬挂异物 (`foreign_object`)** | FastAD | 周期 (1.0s) | 接触网开集反常特征度量与连通域聚类过滤 |
| **4. 结构异常 (`structural`)** | FastAD + YOLO | 周期 (1.0s) | 受电弓几何 Mask 内裂纹、变形与掉块特征度量 |
| **5. 受电弓倾斜 (`tilt`)** | HRNet | 周期 (1.0s) | 4 关键点下横梁浮动坐标系几何姿态角评估 |
| **6. 异常飘弓 (`float`)** | YOLOv11 | 逐帧 (30 FPS) | 受电弓垂向中心时序轨迹跟踪与突跳速度判决 |

---

## 2. 硬件与运行环境

- **测试与开发环境**: NVIDIA TITAN RTX (24GB) / x86_64 / Ubuntu 20.04+
- **边缘部署环境**: NVIDIA Jetson AGX Orin (64GB) / aarch64
- **推荐 Python 环境**: Python 3.8 ~ 3.10, PyTorch >= 2.0.0, CUDA >= 11.8

---

## 3. 目录结构说明

```text
gongwang_clean/
├── AppConfig.xlsx              # 车载通信点表与PHM配置
├── config_prod.yaml            # 生产环境服务主配置 (RTSP、相机、车号、UDP、PHM)
├── start.sh                    # 服务后台启动脚本
├── close.sh                    # 服务安全停止脚本
├── close.py                    # 进程终止控制程序
├── main.py                     # 统一主入口
├── monitor_main.py             # 多路 RTSP 拉流与检测线程管理服务
├── ai_detector.py              # 分层多算法故障检测引擎 (实现 6 层架构)
├── phm_service.py              # PHM 心跳上报与 0xFF 0x20 标准二进制报文生成
├── platform_utils.py           # 平台架构检测与适配工具
├── ftp_uploader.py             # 故障抓拍与前后录像 FTP 自动上传服务
├── usrcode.py                  # 车载寄存器与状态码表
├── requirements.txt            # Python 核心运行依赖清单
├── README.md                   # 本文档
├── AGENTS.md                   # 业务规则与模型规范
├── .gitignore                  # Git 忽略配置
│
├── common/                     # 通用基础设施 (配置读取、全局日志)
│   ├── config_manager.py
│   └── logger.py
│
├── config/                     # 系统与检测器配置
│   ├── config_dev.yaml         # 开发/本地测试配置
│   ├── config_prod.yaml        # 车载生产配置
│   └── detectors.yaml          # 核心检测插件注册、调度与阈值参数配置
│
├── core/                       # 分层插件化框架核心实现
│   ├── data_types.py           # FramePacket, DetectionResult, FusionResult 等核心数据类
│   ├── detectors/              # 6 大类故障检测器插件 (BaseDetector 实现)
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── spark_detector.py
│   │   ├── dirt_detector.py
│   │   ├── tilt_detector.py
│   │   ├── float_detector.py
│   │   ├── struct_detector.py
│   │   └── foreign_detector.py
│   ├── fusion/                 # 跨模块互斥抑制与综合裁决引擎
│   │   └── fusion_engine.py
│   ├── ingestion/              # RTSP 重连拉流与前后 60s 录像环形缓冲区
│   │   ├── ring_buffer.py
│   │   └── stream_reader.py
│   ├── output/                 # UDP 报文组装、工业 HUD 可视化渲染与存储管理
│   │   ├── alarm_sender.py
│   │   ├── storage_manager.py
│   │   └── visualizer.py
│   └── scheduler/              # 分频安全分发调度器 (沙箱保护与时序对齐)
│       └── scheduler.py
│
├── fastad_core/                # FastAD 底层算子与马氏距离度量核心库
├── fastad_detector/            # FastAD 开集模型加载与分类器适配层
├── keypoint_tilt/              # HRNet 受电弓 4 关键点检测与倾斜评估库
│
├── models/                     # 生产模型权重库 (自包含轻量级模型，~212MB)
│   ├── best.pt                 # YOLOv11 受电弓定位与火花检测权重 (5.3MB)
│   ├── best_hrnet_w32_panto.pth# HRNet 关键点回归模型权重 (110MB)
│   ├── fastad_layer2_256_reg1e4_nq3_fmq1.pt # FastAD 异常度量主模型 (96.5MB)
│   └── fastad_384.pt           # FastAD 备用权重
│
├── deploy/                     # 容器化与边缘设备部署配置
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── jetson_deployment_manual.md
│
├── tests/                      # 离线验证与性能基准评测套件
│   └── test_benchmark_suite.py
│
├── events/                     # 告警事件持久化空目录 (.gitkeep)
└── local_cache/                # 故障图片抓拍与录像缓存空目录 (.gitkeep)
```

---

## 4. 快速使用与运行验证

### 4.1 安装依赖
```bash
pip install -r requirements.txt
```

### 4.2 离线快速自检 (单图基准评测)
```bash
python tests/test_benchmark_suite.py --image /home/ibd/jyb/pantograph_broken.jpeg
```

### 4.3 视频离线全场景评测
```bash
python tests/test_benchmark_suite.py --video /path/to/test_video.mp4
```

### 4.4 生产环境服务启动 / 停止
```bash
# 后台启动车载监测服务 (拉取 RTSP 流、推送 UDP 8080 与 PHM 39008)
./start.sh

# 安全终止服务
./close.sh
```
