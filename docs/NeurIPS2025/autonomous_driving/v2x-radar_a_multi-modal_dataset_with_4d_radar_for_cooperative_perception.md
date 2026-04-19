---
title: >-
  [论文解读] V2X-Radar: A Multi-Modal Dataset with 4D Radar for Cooperative Perception
description: >-
  [NeurIPS 2025][自动驾驶][cooperative perception] 提出 V2X-Radar，首个大规模真实世界多模态车路协同感知数据集，包含 4D 雷达、LiDAR 和多视角相机数据，覆盖多种天气和光照条件，提供 20K LiDAR 帧、40K 相机图像、20K 4D 雷达数据和 350K 标注框，并建立三个子数据集的全面基准。
tags:
  - NeurIPS 2025
  - 自动驾驶
  - cooperative perception
  - 4D radar
  - V2X
  - 多模态
  - 目标检测
---

# V2X-Radar: A Multi-Modal Dataset with 4D Radar for Cooperative Perception

**会议**: NeurIPS 2025  
**arXiv**: [2411.10962](https://arxiv.org/abs/2411.10962)  
**代码**: [GitHub](https://github.com/yanglei18/V2X-Radar)  
**领域**: Autonomous Driving  
**关键词**: cooperative perception, 4D radar, V2X, multi-modal dataset, 3D object detection

## 一句话总结

提出 V2X-Radar，首个大规模真实世界多模态车路协同感知数据集，包含 4D 雷达、LiDAR 和多视角相机数据，覆盖多种天气和光照条件，提供 20K LiDAR 帧、40K 相机图像、20K 4D 雷达数据和 350K 标注框，并建立三个子数据集的全面基准。

## 研究背景与动机

单车感知面临遮挡和感知范围受限的安全挑战。车路协同 (V2X) 感知通过共享信息扩展感知范围，但目前存在一个关键空白：

**现有 V2X 数据集缺少 4D 雷达**：OPV2V、DAIR-V2X、V2V4Real 等数据集仅包含相机和 LiDAR，忽视了 4D 雷达在恶劣天气下的鲁棒性优势

**单车数据集已有 4D 雷达**：K-Radar、Dual-Radar 等证明了 4D 雷达对雨雪雾等极端天气的强适应性，但协同感知领域尚无此类数据集

**场景多样性不足**：现有真实 V2X 数据集多为特定时段/天气采集，缺乏昼夜、晴雨雪雾的全面覆盖

## 方法详解

### 整体框架

V2X-Radar 数据集包含三个子集：
- **V2X-Radar-C**：车路协同感知（40 个序列）
- **V2X-Radar-I**：路侧感知（10 个序列）
- **V2X-Radar-V**：单车感知（10 个序列）

### 关键设计

1. **传感器平台配置**：

    - **车端**：RoboSense RS-Ruby-80 LiDAR（80 线）+ Basler 相机（1920×1080）+ Arbe Phoenix 4D 雷达（77GHz，水平 ±50°）+ GPS/IMU + C-V2X 通信单元
    - **路侧**：RoboSense RS-Ruby-80 LiDAR + 3 个 Basler 多视角相机（1536×864）+ OCULI EAGLE 4D 雷达（79GHz，水平 ±56°）+ GPS/IMU + C-V2X 通信单元
    - 4D 雷达提供稀疏点云 + 多普勒信息，对恶劣天气具有天然鲁棒性

2. **数据同步与标定**：

    - **时间同步**：所有计算机时钟先与 GPS 时间对齐，再通过 PTP + PPS 信号实现硬件触发同步，车端与路侧传感器时间差控制在 20ms 以内
    - **空间标定**：相机内参用棋盘格标定；LiDAR-相机外参用 100 个 2D-3D 点对最小化重投影误差；LiDAR-4D 雷达用角反射器上的高强度点对标定
    - **车路配准**：初始配准基于 RTK 定位，再用 CBM 算法和人工调整精细化

3. **数据标注与质量控制**：

    - **自动标注 + 人工精修**流程，多轮质量检查
    - 五个类别：行人、骑行者、小轿车、公交车、卡车
    - 协同标注：车端和路侧标注分别在各自 LiDAR 坐标系生成，统一到路侧坐标系后用 IoU 匹配去重
    - **隐私保护**：模型检测 + 人工逐帧检查，匿名化所有路名、定位数据、车牌、人脸

### 损失函数 / 训练策略

作为数据集论文，V2X-Radar 不涉及新的训练策略，但提供了三种融合策略的基准实验：
- **Early Fusion**：合并原始点云后统一检测
- **Late Fusion**：各方独立检测后 NMS 合并
- **Intermediate Fusion**：各方提取中间特征后传输融合（F-Cooper, V2X-ViT, CoAlign, HEAL）

## 实验关键数据

### 主实验

路侧 3D 目标检测 (V2X-Radar-I)：

| 方法 | 模态 | Vehicle AP@0.7 (Easy) | Pedestrian AP@0.5 (Easy) | Cyclist AP@0.5 (Easy) |
|------|------|----------------------|-------------------------|----------------------|
| PV-RCNN | LiDAR | 88.83 | 77.13 | 91.82 |
| CenterPoint | LiDAR | 86.44 | 67.90 | 90.26 |
| BEVHeight++ | Camera | 48.48 | 33.05 | 45.19 |
| RPFA-Net | 4D Radar | 64.79 | 51.64 | 45.86 |

单车 3D 目标检测 (V2X-Radar-V)：

| 方法 | 模态 | Vehicle AP@0.7 (Easy) | Pedestrian AP@0.5 (Easy) | Cyclist AP@0.5 (Easy) |
|------|------|----------------------|-------------------------|----------------------|
| PV-RCNN | LiDAR | 88.27 | 67.04 | 89.48 |
| BEVHeight++ | Camera | 17.47 | 10.43 | 12.99 |
| RPFA-Net | 4D Radar | 42.77 | 11.51 | 17.03 |

### 消融实验

协同感知 vs 单车感知（车辆类别，AP@0.5）：

| 配置 | Overall | 0-30m | 30-50m | 50-100m | 说明 |
|------|---------|-------|--------|---------|------|
| No Fusion (Camera) | 6.76 | 9.41 | 5.16 | 2.27 | 单相机极弱 |
| Late Fusion (Camera) | 32.88 | 40.58 | 30.37 | 20.00 | 协同大幅提升 |
| V2X-ViT (LiDAR, Sync) | ~85+ | ~90+ | ~80+ | ~60+ | 中间融合最优 |
| V2X-ViT (LiDAR, Async) | 降低 | - | - | - | 通信延迟显著影响性能 |

### 关键发现

- **LiDAR 最强，4D Radar 优于相机**：在路侧检测中，4D Radar (RPFA-Net, 64.79%) 虽远低于 LiDAR (PV-RCNN, 88.83%)，但显著优于相机 (BEVHeight++, 48.48%)
- **车端相机性能急剧下降**：单视角车端相机 (17.47%) 远低于多视角路侧相机 (48.48%)，凸显视角覆盖的重要性
- **协同感知大幅提升**：相比单车感知，所有协同方法均显著提升性能
- **通信延迟是关键挑战**：异步设置 (100ms 延迟) 下所有方法性能明显下降，尤其在严格 IoU=0.7 阈值下
- **每样本标注数最高可达 90**：远超 KITTI/nuScenes 等单车数据集（约 10-30），体现协同感知的环境理解提升

## 亮点与洞察

- 填补了 V2X 领域 4D 雷达数据集的空白，这对恶劣天气下的安全驾驶至关重要
- 数据采集跨越 9 个月，覆盖晴/雨/雾/雪、昼/黄昏/夜多种条件，场景多样性优于同类数据集
- 严格的时间同步（<20ms）和空间标定流程保证了数据质量
- 三个子数据集设计支撑多种研究方向，增强了数据集的通用性

## 局限与展望

- 数据规模相对有限（20K 帧），与 nuScenes (40K) 或 Waymo (200K+) 仍有差距
- 仅包含一辆车和一个路侧单元的 V2I 场景，未涵盖 V2V 或多路侧协同
- 4D 雷达点云极其稀疏（车辆平均仅几十个点），现有方法直接应用效果有限
- 标注仅包含 3D 框，未提供语义分割、跟踪 ID 等更丰富的标注
- 车端仅一个前视相机，限制了环视感知算法的适用性

## 相关工作与启发

- 与 DAIR-V2X 的关系：DAIR-V2X 是真实 V2I 数据集但无 4D 雷达；V2X-Radar 增加了 4D 雷达模态
- 与 K-Radar/Dual-Radar 的关系：这些是单车 4D 雷达数据集；V2X-Radar 将 4D 雷达引入协同感知
- 启发：4D 雷达在协同感知中的潜力尚未充分发掘，特别在多模态融合（雷达+LiDAR+相机）和恶劣天气鲁棒性方面有广阔研究空间

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个 V2X 4D 雷达数据集，填补重要空白
- 实验充分度: ⭐⭐⭐⭐ 三个子集 × 多种检测方法 × 同步/异步设置
- 写作质量: ⭐⭐⭐⭐ 数据集论文标准写法，细节翔实
- 价值: ⭐⭐⭐⭐ 对推动 4D 雷达在协同感知中的研究有重要意义

<!-- RELATED:START -->

## 相关论文

- [UrbanIng-V2X: A Large-Scale Multi-Vehicle Multi-Infrastructure Dataset Across Multiple Intersections for Cooperative Perception](urbaning-v2x_a_large-scale_multi-vehicle_multi-infrastructure_dataset_across_mul.md)
- [RadarMP: Motion Perception for 4D mmWave Radar in Autonomous Driving](../../AAAI2026/autonomous_driving/radarmp_motion_perception_for_4d_mmwave_radar_in_autonomous_driving.md)
- [H-V2X: A Large Scale Highway Dataset for BEV Perception](../../ECCV2024/autonomous_driving/h-v2x_a_large_scale_highway_dataset_for_bev_perception.md)
- [CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection](../../ICCV2025/autonomous_driving/cvfusion_cross-view_fusion_of_4d_radar_and_camera_for_3d_object_detection.md)
- [AdaRadar: Rate Adaptive Spectral Compression for Radar-based Perception](../../CVPR2026/autonomous_driving/adaradar_rate_adaptive_spectral_compression_for_radar-based_perception.md)

<!-- RELATED:END -->
