---
title: >-
  [论文解读] InFlux: A Benchmark for Self-Calibration of Dynamic Intrinsics of Video Cameras
description: >-
  [NeurIPS 2025][视频理解][camera intrinsics] 提出首个包含逐帧动态相机内参真值的真实视频基准 InFlux（386 视频、143K+ 标注帧），通过镜头元数据到内参的查找表（LUT）实现精确标注，并揭示现有内参预测方法在动态内参场景下表现不佳。
tags:
  - "NeurIPS 2025"
  - "视频理解"
  - "camera intrinsics"
  - "dynamic calibration"
  - "benchmark"
  - "lookup table"
  - "video 3D understanding"
---

# InFlux: A Benchmark for Self-Calibration of Dynamic Intrinsics of Video Cameras

**会议**: NeurIPS 2025  
**arXiv**: [2510.23589](https://arxiv.org/abs/2510.23589)  
**代码**: [项目主页](https://influx.cs.princeton.edu/)  
**领域**: 视频理解  
**关键词**: camera intrinsics, dynamic calibration, benchmark, lookup table, video 3D understanding

## 一句话总结

提出首个包含逐帧动态相机内参真值的真实视频基准 InFlux（386 视频、143K+ 标注帧），通过镜头元数据到内参的查找表（LUT）实现精确标注，并揭示现有内参预测方法在动态内参场景下表现不佳。

## 研究背景与动机

- **3D 算法的恒定内参假设不成立**：NeRF、3DGS、SLAM 等主流 3D 方法假设视频内相机内参保持不变，但 DSLR 变焦镜头和智能手机自动对焦都会导致内参逐帧变化，严重限制了方法在野外视频上的鲁棒性。
- **缺少动态内参基准**：现有数据集（KITTI、EuRoC、ETH3D 等）均在固定镜头设置下采集，仅需一次标定；唯一涉及变内参的 [Liao et al. 2025] 只提供了标定板视频（缺乏场景多样性）和 300 张网图的镜头焦距标注（不等于真实 CFL）。
- **合成数据无法替代真实基准**：合成数据集存在视觉 sim-to-real gap，且缺乏内参标注或场景多样性。
- **逐帧真值标注极难获取**：对每帧进行完整标定代价高昂且会破坏视频连续性（需逐帧暂停拍摄），因此此前无人实现。

## 方法详解

### 核心思路：LFL-FD 查找表（LUT）

关键观察：变焦镜头的光学状态由两个参数唯一确定——镜头焦距 (LFL) 和对焦距离 (FD)。使用支持 /i Technology 元数据记录的专业电影镜头（Canon CINE-SERVO 17-120mm、Fujinon Premista 80-250mm），可以为每帧记录 LFL 和 FD 值。因此只需预先构建一张 LUT 将 (LFL, FD) 映射到完整内参，即可将逐帧标定转化为一次性查表。

### 标定实验设计

针对不同的 FOV 空间足迹（FSF）大小，使用不同尺度的标定目标：

- **小/中 FSF — 标定板标定**：使用4种尺寸的 AprilGrid 标定板（$100\times75$mm 到 $800\times600$mm），选择能完全在 FOV 内的最大尺寸。通过抖动拍摄辅助关键帧提取，使用 ANMS 基于检测数量选帧。
- **大 FSF — 无人机标定**：当 FSF 大到无法制造足够大的平面标定板时，使用搭载 RTK 定位芯片（Septentrio Mosaic X5，cm 级精度）和红色 LED 的 Holybro X500 V2 无人机作为标定目标。夜间拍摄红色 LED 获取精确 2D 检测，RTK 提供 3D 位置，通过时序同步建立 2D-3D 对应关系。

### 改进版 Kalibr

原始 Kalibr 在 LM 优化中存在收敛问题和主点漂移：

- **CFL 初始化改进**：用薄透镜近似公式替代原始的消失点方法，利用已知 LFL 和 FD 信息
- **固定点初始化**：畸变初始化阶段周期性将主点重置回图像中心，防止异常漂移
- **多次运行取中位数**：对最终优化的随机排序进行多次 rollout，选择中位数结果减少方差

### LUT 插值方案

- **网格区域（标定板实验）**：LFL-FD 空间近似规则网格，使用梯形双线性插值
- **非网格区域（含无人机实验）**：使用 Delaunay 三角化 + 重心插值

$$\mathbf{K}(l, d) = \text{Interpolate}\big(\{(\text{LFL}_i, \text{FD}_i, \mathbf{K}_i)\}\big)$$

其中 $\mathbf{K}$ 包含 $f_x, f_y, c_x, c_y$ 和 Brown-Conrady 畸变参数。

## 实验

### 数据集统计

| 属性 | 数量 |
|:---|:---:|
| 视频总数 | 386 |
| 标注帧数 | 143K+ |
| 室内视频 | 126 |
| 室外视频 | 260 |
| 镜头类型 | 2（Canon 17-120mm, Fujinon 80-250mm）|
| 内参变化类型 | 单调变焦/对焦、周期性变化、非单调波动、电影式推拉 |

### 表1：内参预测基线方法评估

| 方法 | %$f_x$ Error↓ | %$f_y$ Error↓ | %$c_x$ Error↓ | %$c_y$ Error↓ | %EPE<300px↑ |
|:---|:---:|:---:|:---:|:---:|:---:|
| GeoCalib | **56.5** | **56.5** | **0.099** | **0.204** | **52.9** |
| WildCamera | 45.6 | 46.9 | 5.04 | 6.39 | 47.2 |
| UniDepthV2 | 50.6 | 51.1 | 1.61 | 2.58 | 46.1 |
| DroidCalib | 68.1 | 70.0 | 10.1 | 15.7 | 28.0 |
| Perspective Fields | 64.6 | 64.6 | 18.6 | 19.7 | 17.8 |
| COLMAP | 1270 | 1280 | 0.112 | 0.299 | 7.85 |

**关键发现**：
- **所有方法表现不佳**：即便最好的 GeoCalib，在 $3424\times2202$ 分辨率下也仅有 52.9% 的帧点对 EPE <300px
- **COLMAP 近乎完全失效**：92% 的帧无法产生预测，CFL 误差高达 1270%
- **DroidCalib 依赖光流**：在少运动视频上 15% 的帧无法预测
- **逐帧方法缺乏时序平滑性**：GeoCalib/WildCamera 等单帧预测方法产生的内参序列不平滑

### 改进 Kalibr 的合成实验验证

在 Blender 渲染的合成标定场景上，改进版 Kalibr 对比原版：
- 消除了原版偶发的大误差尖峰
- 所有实验均成功收敛（原版部分失败）
- CFL 和主点的预测方差显著降低

## 亮点

- **填补关键空白**：首个提供逐帧动态内参真值的真实视频基准，使得研究社区首次可以系统评估动态内参预测方法
- **标注方案巧妙**：LUT + 镜头元数据的方案将逐帧标定难题转化为一次性查表，兼顾精度和拍摄自然性
- **无人机标定创新**：RTK + LED 的设计优雅地解决了大 FOV 场景下标定板无法覆盖的问题
- **评估揭示痛点**：定量结果清晰展示现有方法在动态内参上的脆弱性，为后续研究指明方向

## 局限性

- **硬件依赖高**：需要专业电影级相机和镜头（ARRI Alexa Mini + 电影变焦镜头），难以推广到消费级设备
- **镜头覆盖有限**：仅包含2种镜头，相机类型单一
- **无训练/测试分割**：作为 benchmark 未提供标准的训练集，限制了数据驱动方法的直接使用
- **仅适用于支持元数据的镜头**：不记录 LFL/FD 的镜头无法使用此方案获取真值
- **插值精度有限**：线性/重心插值可能无法完美建模复杂的真实镜头系统

## 相关工作

- **真实相机内参数据集**（KITTI、EuRoC、ETH3D）：均为固定内参，不支持动态变化
- **合成内参数据集**（[Ray+ 2024]、[Liao+ 2025]）：存在 sim-to-real gap 或覆盖不足
- **标定方法**（Kalibr [Maye+ 2013]、OpenCV [Bradski 2000]）：InFlux 的改进版 Kalibr 显著提升精度
- **内参估计方法**（COLMAP、GeoCalib [Jin+ 2023]、UniDepthV2 [Piccinelli+ 2025]）：在 InFlux 上均暴露出动态场景下的不足

## 评分

- 新颖性: ⭐⭐⭐⭐ — 填补了动态内参基准的空白，LUT 标注方案设计新颖
- 实验充分度: ⭐⭐⭐⭐ — 6 种基线方法 + 合成验证 + 丰富的数据多样性分析
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，技术细节完整，图表丰富
- 价值: ⭐⭐⭐⭐ — 为 3D 视觉社区提供了急需的动态内参评测基础设施

## 实验关键数据

## 亮点

## 局限与展望

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Attention-Seeker: Dynamic Self-Attention Scoring for Unsupervised Key-Frame Extraction](../../ACL2025/video_understanding/attention-seeker_dynamic_self-attention_scoring_for_unsupervised_key-frame_extra.md)
- [\[NeurIPS 2025\] QiMeng-NeuComBack: Self-Evolving Translation from IR to Assembly Code](qimeng-neucomback_self-evolving_translation_from_ir_to_assembly_code.md)
- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)
- [\[NeurIPS 2025\] EgoGazeVQA: Egocentric Gaze-Guided Video Question Answering Benchmark](egogazevqa_egocentric_gaze_guided_video_question_answering.md)
- [\[NeurIPS 2025\] MUVR: A Multi-Modal Untrimmed Video Retrieval Benchmark with Multi-Level Visual Correspondence](muvr_a_multi-modal_untrimmed_video_retrieval_benchmark_with_multi-level_visual_c.md)

</div>

<!-- RELATED:END -->
