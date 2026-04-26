---
title: >-
  [论文解读] AllTracker: Efficient Dense Point Tracking at High Resolution
description: >-
  [ICCV 2025][视频理解][密集点跟踪] 提出AllTracker，将点跟踪重新表述为多帧长程光流问题，在低分辨率网格上通过2D卷积+像素对齐时序注意力迭代优化对应估计再上采样，仅16M参数即实现SOTA准确率和高分辨率（768×1024）全像素密集跟踪，跟踪速度接近光流方法。
tags:
  - ICCV 2025
  - 视频理解
  - 密集点跟踪
  - 光流
  - 长程对应
  - 高分辨率跟踪
  - 循环网络
---

# AllTracker: Efficient Dense Point Tracking at High Resolution

**会议**: ICCV 2025  
**arXiv**: [2506.07310](https://arxiv.org/abs/2506.07310)  
**代码**: https://alltracker.github.io  
**领域**: 视频理解 / 密集点跟踪  
**关键词**: 密集点跟踪, 光流, 长程对应, 高分辨率跟踪, 循环网络

## 一句话总结
提出AllTracker，将点跟踪重新表述为多帧长程光流问题，在低分辨率网格上通过2D卷积+像素对齐时序注意力迭代优化对应估计再上采样，仅16M参数即实现SOTA准确率和高分辨率（768×1024）全像素密集跟踪，跟踪速度接近光流方法。

## 研究背景与动机

估计2D图像序列中任意点的长程轨迹是计算机视觉的基础问题。光流方法（如RAFT、SEA-RAFT）能估计连续两帧间的逐像素运动，但将瞬时光流"链接"为长程轨迹会累积漂移，且需在遮挡处小心处理。直接计算参考帧与远帧间的"长程光流"虽能避免漂移，但随时间间隔增大，视角、光照、场景几何的变化使估计愈发困难。

近年来，一系列专用点跟踪器（PIPs、TAP、CoTracker等）通过学习多帧时序先验来减少漂移并跨遮挡跟踪，取得了显著进展。然而，这些方法以牺牲空间感知为代价增加时序感知，只能跟踪稀疏点集。近期尝试"密集"点跟踪的方法（DTF、DELTA）准确率不如最新稀疏跟踪器，且难以处理高分辨率输入。

本文的核心洞察是：可学习的多帧时序先验与高分辨率空间感知可以共同构建——只需将点跟踪重新表述为多帧长程光流问题。这样既能像光流方法那样生产高分辨率密集对应场，又能像点跟踪器那样跨遮挡跟踪。

## 方法详解

### 整体框架
给定视频和查询帧索引，AllTracker输出 $T \times H \times W \times 4$ 张量：每个像素在每一帧的光流偏移(2通道)+可见性+置信度。采用滑动窗口策略（窗口长度 $S=16$，步长 $S/2$），每个窗口内：(1) CNN编码器提取低分辨率特征 → (2) 构建多尺度4D相关体 → (3) 初始化跟踪估计 → (4) 循环精炼模块迭代更新 → (5) 上采样到全分辨率。

### 关键设计

1. **ConvNeXt-Tiny编码器**:

    - 功能：将视频帧压缩为低分辨率特征图 ($H/8 \times W/8$)
    - 核心思路：使用预训练ConvNeXt-Tiny的前3个block（12.72M参数），将第3个block的stride-2改为stride-1（通过双三次插值扩展2×2核到3×3核），输出256维特征
    - 设计动机：低分辨率特征允许快速2D卷积消息传递，最终通过上采样恢复空间精度。使用stride 8而非CoTracker3的stride 4显著节省内存

2. **多尺度4D相关体**:

    - 功能：捕获基于外观的跟踪线索
    - 核心思路：将每帧特征图通过平均池化构建特征金字塔（5个尺度：{1,2,4,8,16}），然后查询帧特征图与各时间步金字塔做点积交叉相关。每个被跟踪像素在每个尺度、每个时间步获得一个热力图。迭代精炼时在当前估计位置附近提取 $(2R+1)^2$ 的小块（$R=4$），展平为向量 $\mathbf{q}$（$L \cdot (2R+1)^2 = 5 \times 81 = 405$维）
    - 设计动机：多尺度相关提供强归纳偏置用于特征匹配，加速训练

3. **交替空间-时序循环精炼模块（核心）**:

    - 功能：迭代更新所有像素在所有时间步的运动、可见性和置信度估计
    - 核心思路：每像素每时间步的输入包括特征向量 $\mathbf{f}$（256D）、可见性/置信度 $\mathbf{v,c}$（2D）、运动估计 $\mathbf{m}$（2D）、相关向量 $\mathbf{q}$（405D），共665通道。交替执行：
        - **空间块**：2D ConvNeXt block，在空间维度传播信息
        - **时序块**：像素对齐的Transformer注意力，仅沿时间轴（$S=16$帧）做注意力，对所有像素并行执行
      注意所有张量与查询帧对齐，因此像素对齐注意力自然是对应像素间的注意力
    - 设计动机：空间消息传递用简单2D卷积而非全局注意力，高效且有效。时序注意力学习运动先验和遮挡处理。迭代精炼通过残差更新 $\mathbf{x}_{\text{new}} = \mathbf{x}_{\text{old}} + \delta\mathbf{x}$ 逐步改善

4. **Pixel-Shuffle上采样**:

    - 功能：从1/8分辨率恢复到全分辨率
    - 核心思路：循环模块额外解码pixel-shuffle权重，应用于可见性、置信度和运动图
    - 设计动机：这一技术在RAFT中已知但在点跟踪文献中被低估，是实现高分辨率密集跟踪的关键

### 损失函数 / 训练策略
- **跟踪损失**：$L_{\text{track}} = \alpha \sum_k^K \gamma^{K-k} (\mathbb{1}_{\text{occ}}/5 + \mathbb{1}_{\text{vis}}) \|P_k - \hat{P}\|_1$，$\gamma=0.8$使后期精炼步权重更大，$\alpha=0.05$
- **可见性损失**：BCE，匹配GT二值标签
- **置信度损失**：BCE，目标为估计位置是否在GT 12像素内
- 两阶段训练：先在Kubric上200K步（lr=5e-4），再在混合数据集上400K步（lr=1e-5）
- 混合数据集联合光流数据（FlyingChairs, FlyingThings3D, AutoFlow, Spring, VIPER等）和点跟踪数据（Kubric, PointOdyssey等），均匀采样

## 实验关键数据

### 主实验：9个benchmark平均 $\delta_{avg}$（384×512分辨率）

| 方法 | 参数量 | BADJA | Davis | Kinetics | RGB-Stack | RoboTAP | 平均 |
|------|--------|-------|-------|----------|-----------|---------|------|
| RAFT | 5.3M | 23.7 | 48.5 | 64.3 | 82.8 | 72.2 | 48.3 |
| SEA-RAFT | 19.7M | 23.9 | 48.7 | 64.3 | 85.7 | 67.6 | 48.7 |
| CoTracker3-Kub | 25.4M | 47.5 | 77.4 | 70.6 | 83.4 | 77.2 | 64.5 |
| CoTracker3 | 25.4M | 48.3 | 77.1 | 71.8 | 84.2 | 81.6 | 65.0 |
| **AllTracker** | **16.5M** | **51.5** | 76.3 | **72.3** | **90.0** | **83.4** | **66.1** |

### 高分辨率实验（768×1024）

| 方法 | 参数量 | BADJA | Davis | RGB-Stack | 平均 |
|------|--------|-------|-------|-----------|------|
| CoTracker3 | 25.4M | 49.8 | 79.6 | 77.9 | 66.9 |
| AllTracker-Tiny | 6.3M | 51.6 | 79.1 | 87.4 | 67.5 |
| **AllTracker** | **16.5M** | **53.6** | **80.6** | **90.6** | **69.5** |

### 关键发现
- AllTracker以16.5M参数（CoTracker3的65%）在9个benchmark平均上超越CoTracker3 1.1个点
- 高分辨率下优势扩大：768×1024时平均69.5 vs CoTracker3的66.9（+2.6），AllTracker性能随分辨率稳步提升而CoTracker3在448×768后趋于饱和
- AllTracker-Tiny（仅6.3M参数）在768×1024下已超过完整CoTracker3（67.5 vs 66.9）
- RGB-Stacking上优势最大（90.6 vs 77.9），说明模型能有效利用宽区域空间上下文
- 联合光流+点跟踪数据训练至关重要：仅Kubric训练平均64.8，加入光流数据后66.1（+1.3）
- 实时模式512×512可达57.9 FPS，$\delta_{avg}$仍有62.6
- Average Jaccard: AllTracker 68.9 vs CoTracker3 63.1；遮挡准确率: 91.5 vs 89.3

## 亮点与洞察
- 核心贡献是将问题重新表述为多帧长程光流而非稀疏点跟踪，使得(1)密集跟踪成为自然输出(2)可联合使用光流和点跟踪数据训练
- 低分辨率精炼+pixel-shuffle上采样的策略是实现高分辨率密集跟踪的关键——这一在RAFT中已知的技术在点跟踪文献中被低估
- 像素对齐时序注意力（cost仅与窗口长度S=16成正比）比全局注意力高效得多
- 光流数据混合训练的效果验证了跨任务数据利用的价值

## 局限与展望
- 在CroHD（监控）和DriveTrack（驾驶）数据集上额外训练数据未能可靠提升性能，暗示数据平衡或模型容量仍有改进空间
- 滑动窗口策略带来实时模式约3.5点精度损失（66.1→62.6）
- 未与CoTracker3的伪标签自举方案结合，该方案可能进一步提升性能

## 相关工作与启发
- **vs CoTracker3**: CoTracker3是之前SOTA的稀疏点跟踪器，使用虚拟点的空间传播。AllTracker用2D卷积替代，更简单高效，且原生支持全像素密集跟踪
- **vs DELTA**: 同期的密集跟踪工作，使用全局空间注意力近似（稀疏anchor token），内存消耗大且准确率不如AllTracker
- **vs SEA-RAFT**: AllTracker继承了SEA-RAFT的低分辨率估计+上采样策略，但从2帧扩展到16帧窗口，增加时序注意力实现遮挡跟踪

## 评分
- 新颖性: ⭐⭐⭐⭐ 问题重新表述简洁优雅，组合已知技术产生新能力
- 实验充分度: ⭐⭐⭐⭐⭐ 9个benchmark+3种分辨率+速度对比+AJ/遮挡指标，极为全面
- 写作质量: ⭐⭐⭐⭐⭐ 设计选择阐述清晰，与prior work的关系梳理得当
- 价值: ⭐⭐⭐⭐⭐ 使密集长程点跟踪达到实用水平，统一了光流和点跟踪两个方向

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Online Dense Point Tracking with Streaming Memory](online_dense_point_tracking_with_streaming_memory.md)
- [\[ICCV 2025\] MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation](memfof_high-resolution_training_for_memory-efficient_multi-frame_optical_flow_es.md)
- [\[ICCV 2025\] ResidualViT for Efficient Temporally Dense Video Encoding](residualvit_for_efficient_temporally_dense_video_encoding.md)
- [\[NeurIPS 2025\] Cloud4D: Estimating Cloud Properties at a High Spatial and Temporal Resolution](../../NeurIPS2025/video_understanding/cloud4d_estimating_cloud_properties_at_a_high_spatial_and_temporal_resolution.md)
- [\[NeurIPS 2025\] Fixed-Point RNNs: Interpolating from Diagonal to Dense](../../NeurIPS2025/video_understanding/fixed-point_rnns_interpolating_from_diagonal_to_dense.md)

<!-- RELATED:END -->
