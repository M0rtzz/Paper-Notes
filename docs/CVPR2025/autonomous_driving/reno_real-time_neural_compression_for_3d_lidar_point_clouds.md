---
title: >-
  [论文解读] RENO: Real-Time Neural Compression for 3D LiDAR Point Clouds
description: >-
  [CVPR 2025][自动驾驶][点云压缩] RENO提出稀疏占用码（Sparse Occupancy Codes）和一次性推理策略，首次实现了3D LiDAR点云的实时神经压缩（10fps@14-bit），以仅1MB的模型大小超越G-PCC标准12.25%码率节省。
tags:
  - CVPR 2025
  - 自动驾驶
  - 点云压缩
  - 实时编解码
  - 稀疏占用码
  - LiDAR
  - 神经编码器
---

# RENO: Real-Time Neural Compression for 3D LiDAR Point Clouds

**会议**: CVPR 2025  
**arXiv**: [2503.12382](https://arxiv.org/abs/2503.12382)  
**代码**: [github.com/NJUVISION/RENO](https://github.com/NJUVISION/RENO)  
**领域**: Autonomous Driving / 3D Vision  
**关键词**: 点云压缩, 实时编解码, 稀疏占用码, LiDAR, 神经编码器

## 一句话总结

RENO提出稀疏占用码（Sparse Occupancy Codes）和一次性推理策略，首次实现了3D LiDAR点云的实时神经压缩（10fps@14-bit），以仅1MB的模型大小超越G-PCC标准12.25%码率节省。

## 研究背景与动机

LiDAR点云在自动驾驶、机器人和3D建图中广泛使用，实时压缩（10Hz匹配LiDAR采集频率）是工业应用的刚需。现有方法面临两难：

- 传统方法如G-PCC率失真性能好但不够实时（编码一帧约1秒）；Draco速度快但压缩效率低
- 学习方法（如Unicorn）展现出色的率失真性能，但在RTX 3090上编码一帧仍需约2秒

现有神经编码器的瓶颈在于两个阶段：(1) 预处理阶段需要构建八叉树结构，耗时显著；(2) 神经推理阶段采用多阶段处理，需要对上采样后的 $8 \times N_d$ 个体素进行推理，计算量巨大。

核心问题：**如何设计一个既实时又高效的神经LiDAR压缩器？**

## 方法详解

### 整体框架

RENO基于多尺度稀疏张量表示，跳过耗时的八叉树构建。将点云几何压缩形式化为逐尺度压缩稀疏占用码序列 $\mathcal{O} = (O^1, O^2, \ldots, O^{D-1})$。使用快速占用生成器（FOG）和快速坐标生成器（FCG）实现编解码映射，目标占用预测器（TOP）建模跨尺度上下文进行熵编码。

### 关键设计

**1. 稀疏占用码（Sparse Occupancy Codes）**

- **功能**：将点云几何编码为离散占用码序列，使压缩问题转化为序列压缩
- **核心思路**：使用固定权重稀疏卷积（核大小2，步长2，权重 $[1,2,4,8,16,32,64,128]$）直接在稀疏空间中生成占用码 $o_i^{d-1} \in [1, 255]$，无需构建八叉树。整个点云可由初始状态 $(C^0, O^0)$ 和占用码序列 $\mathcal{O}$ 无损重建
- **设计动机**：八叉树占用符号和稀疏张量中的占用码携带相同的分类值（1-255），但稀疏占用码是无序的而非树序的，可通过并行稀疏卷积高效生成，避免了树结构构建的复杂性

**2. 目标占用预测器（TOP）+ 目标嵌入**

- **功能**：利用低尺度先验信息估计当前尺度占用码的概率分布，用于熵编码
- **核心思路**：$P_\theta(O^d) = \text{TOP}(C^{d-1}, O^{d-1}, C^d)$。先通过Embedding+ResNet提取低尺度特征 $F^{d-1}$，再通过目标嵌入将特征从 $C^{d-1}$ 复制到 $C^d$ 的位置（Feature Replication + Octant Position Infusion），最后用MLP+SoftMax预测255维概率
- **设计动机**：跨尺度相关性是点云压缩的关键先验。直接复制特征并注入相对位置信息（octant），实现了从低尺度到高尺度的一次性推理，避免了多阶段上采样的计算瓶颈

**3. 分位两阶段概率预测（Bitwise Two-stage）**

- **功能**：将8-bit占用码分为两个4-bit子码分步预测，同时提升压缩性能和计算效率
- **核心思路**：$P_\theta(O^d) = P_\theta(S_2^d | S_1^d) P_\theta(S_1^d)$，先预测高4位再条件预测低4位
- **设计动机**：预测4-bit符号（16类）比8-bit（255类）容易得多；更重要的是，GPU→CPU传输概率表的带宽减少约8倍（$2 \times N \times 16$ vs $N \times 255$），大幅降低熵编码延迟

### 损失函数

交叉熵损失：$\mathcal{L} = \sum_{d=1}^{D-1} \mathbb{E}_{O^d \sim P(O^d)} [-\log P_\theta(O^d)]$，直接优化占用码序列的无损压缩效率。

## 实验关键数据

### 主实验：BD-BR增益和速度对比（KITTI数据集）

| 方法 | BD-BR D1(%) | 14-bit编码时间(s) | 14-bit解码时间(s) |
|------|------------|------------------|------------------|
| Draco | baseline(+48.34) | 0.075 | 0.032 |
| G-PCCv23 | baseline(+12.26) | 0.973 | 0.343 |
| RENO | **-12.26 vs G-PCC** | **0.095** | **0.090** |
| Unicorn | SOTA压缩 | ~2.0 | ~2.0 |

RENO以10fps实时速率运行，编解码时间均约0.1秒，比G-PCC快10倍。

### 消融实验：各组件贡献

| 组件变体 | BD-BR变化 |
|---------|----------|
| 无跨尺度上下文 | +8.5% |
| 无目标嵌入（输入仅$C^d$） | +4.2% |
| 8-bit直接预测 vs 4+4-bit分阶段 | 分阶段更优且更快 |

### 关键发现

- RENO是**首个实时**神经LiDAR点云压缩器，同时超越G-PCC标准
- 模型仅**1MB**大小，极具实用部署价值
- 在Ford数据集上同样展现12.5% BD-BR节省，泛化性良好
- 在3D目标检测下游任务中，RENO压缩的点云保持与原始数据相近的检测精度

## 亮点与洞察

1. **"跳过八叉树"的核心洞察**：八叉树占用符号和稀疏张量占用码本质上携带相同信息，但后者通过固定权重卷积即可并行生成，彻底消除了树结构构建的瓶颈
2. **一次性推理替代多阶段处理**：通过目标嵌入将低尺度特征直接映射到高尺度目标位置，避免了对 $8 \times N$ 个上采样体素的逐步推理
3. 分位两阶段方案同时优化了压缩效率和GPU-CPU通信瓶颈，体现了对系统级优化的深入理解

## 局限与展望

- 当前仅关注几何压缩（点位置），未处理属性压缩（如颜色、强度）
- 模型在不同LiDAR传感器间的泛化能力可进一步验证
- 未考虑时序冗余（帧间压缩），未来可扩展为视频点云压缩
- 固定权重卷积的设计虽高效但缺乏自适应性，可能限制某些场景的压缩上限

## 相关工作与启发

- **与Unicorn的关系**：同基于多尺度稀疏张量，但Unicorn需多阶段推理，RENO通过占用码+一次性推理实现10倍加速
- **与G-PCC/Draco的关系**：RENO首次在神经方法中同时超越了这两个传统标准的速度和/或压缩效率
- **启发**：在设计实时系统时，瓶颈不仅在网络推理，还在预处理和数据传输——需要系统级全局优化

## 评分

⭐⭐⭐⭐

首次实现LiDAR点云实时神经压缩是重要工程里程碑。1MB模型大小和10fps速度使其具有实际部署价值。核心创新（稀疏占用码+一次性推理）简洁有效。稍显不足的是压缩增益相比最优学习方法仍有差距，且应用场景限于几何压缩。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] PSA-SSL: Pose and Size-aware Self-Supervised Learning on LiDAR Point Clouds](psa-ssl_pose_and_size-aware_self-supervised_learning_on_lidar_point_clouds.md)
- [\[CVPR 2025\] WeatherGen: A Unified Diverse Weather Generator for LiDAR Point Clouds via Spider Mamba Diffusion](weathergen_a_unified_diverse_weather_generator_for_lidar_point_clouds_via_spider.md)
- [\[CVPR 2025\] Neural Inverse Rendering from Propagating Light](neural_inverse_rendering_from_propagating_light.md)
- [\[CVPR 2025\] Unlocking Generalization Power in LiDAR Point Cloud Registration](unlocking_generalization_power_in_lidar_point_cloud_registration.md)
- [\[ECCV 2024\] SFPNet: Sparse Focal Point Network for Semantic Segmentation on General LiDAR Point Clouds](../../ECCV2024/autonomous_driving/sfpnet_sparse_focal_point_network_for_semantic_segmentation_on_general_lidar_poi.md)

<!-- RELATED:END -->
