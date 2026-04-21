---
title: >-
  [论文解读] Spatiotemporal Decoupling for Efficient Vision-Based Occupancy Forecasting
description: >-
  [CVPR 2025][自动驾驶][占用预测] 提出 EfficientOCF，通过空间解耦（将 3D 占用分解为 2D BEV 占用 + 高度值）和时间解耦（通过光流关联实例实现逐步 OCF 而非端到端预测）解决占用预测中的空间偏置和时间偏置问题，实现 SOTA 3D 占用预测性能和 82.33ms 的快速推理。
tags:
  - CVPR 2025
  - 自动驾驶
  - 占用预测
  - 时空解耦
  - BEV表征
  - 实例感知
---

# Spatiotemporal Decoupling for Efficient Vision-Based Occupancy Forecasting

**会议**: CVPR 2025  
**arXiv**: [2411.14169](https://arxiv.org/abs/2411.14169)  
**代码**: 无  
**领域**: Autonomous Driving / Occupancy Forecasting  
**关键词**: 占用预测, 时空解耦, BEV表征, 实例感知, 自动驾驶

## 一句话总结

提出 EfficientOCF，通过空间解耦（将 3D 占用分解为 2D BEV 占用 + 高度值）和时间解耦（通过光流关联实例实现逐步 OCF 而非端到端预测）解决占用预测中的空间偏置和时间偏置问题，实现 SOTA 3D 占用预测性能和 82.33ms 的快速推理。

## 研究背景与动机

3D 占用预测 (OCF) 利用过去和当前感知数据预测未来环境的占用状态，对自动驾驶中的避障和路径规划至关重要。现有 3D OCF 方法存在两个核心偏置：

1. **空间偏置**：3D 空间中绝大部分体素是空的，但端到端方法仍需处理所有体素网格，计算浪费且使占用预测偏向"空"状态
2. **时间偏置**：只有少量可移动物体在短期内改变位置，静态物体占主导，端到端预测对动态物体的形状随时间发散

现有方法（如 OCFNet）采用密集 3D 特征编解码处理所有体素，导致高计算成本和可移动物体预测不准确。BEV 方法虽高效但缺乏 z 轴空间结构理解。

核心思路：**将 3D OCF 的空间和时间维度解耦**——空间上用 2D BEV + 高度实现 3D，时间上将实例分割与占用预测分离。

## 方法详解

### 整体框架

EfficientOCF 由四个模块组成：(1) **感知模块**：从环视图像提取 2D 特征并 Lift-Splat-Shoot 到 3D 体素特征；(2) **聚合模块**：通过自适应双池化将 3D 特征压缩为 2D BEV 特征，聚合多帧；(3) **预测模块**：三个共享 2D 编解码结构的头（分割/高度/光流）；(4) **精炼模块**：通过光流关联实例，逐步精炼 OCF 结果。

### 关键设计

**1. 空间解耦：BEV 占用 + 高度表征**

- **功能**: 将 3D 占用预测降维为 2D 预测，大幅提升效率
- **核心思路**: 不用传统的密集 3D 体素 $O_t^{3D} \in \mathbb{R}^{1 \times H \times W \times L}$，而是预测 2D BEV 占用 $O_t^{2D} \in \mathbb{R}^{1 \times H \times W}$ 和对应高度值 $O_t^{height} \in \mathbb{R}^{1 \times H \times W}$，仅为占用网格存储高度信息。最终通过高度 lifting 恢复 3D 占用
- **设计动机**: 3D 空间中空体素占绝大多数，为每个都分配计算资源是浪费的。BEV + 高度表征保留了 3D 信息但只需 2D 计算量

**2. 时间解耦：实例关联精炼**

- **功能**: 将实例分割与占用预测解耦，通过实例关联提升未来帧的预测质量
- **核心思路**: 光流头预测 2D 后向向心流 $O_t^{flow} \in \mathbb{R}^{2 \times H \times W}$（指向前一帧实例中心）。精炼模块在 $t=-1$ 时做 NMS 提取实例中心，沿时间轴用光流迭代关联实例 ID $M_t^{2D}$，生成实例 mask $\bar{M}_t^{2D}$ 来过滤初始 OCF 结果：$\bar{O}_t^{2D} = O_t^{2D} \cdot \bar{M}_t^{2D}$
- **设计动机**: 基于当前观测的实例分割比端到端未来预测更准确。通过实例传播而非重新预测，保持物体形状在时间上的一致性，减少形状发散

**3. 自适应双池化策略**

- **功能**: 将 3D 体素特征高效转换为 2D BEV 特征
- **核心思路**: 同时使用平均池化（捕获整体信息）和最大池化（隐式对应高度值的显著占用特征），通过可学习权重 $\alpha_{avg}$ 和 $\alpha_{max}$ 自适应融合：$F^{BEV} = \alpha_{avg} F^{avg} + \alpha_{max} F^{max}$
- **设计动机**: 单一池化策略信息损失大，双池化从不同角度压缩 z 轴信息

### 损失函数 / 训练策略

- **总损失**: $\mathcal{L}_{all} = \frac{1}{N_f+1}\sum_{t}(\lambda_1 \mathcal{L}_{occ} + \lambda_2 \mathcal{L}_{height} + \lambda_3 \mathcal{L}_{flow})$
    - $\mathcal{L}_{occ}$: 交叉熵损失（2D BEV 占用）
    - $\mathcal{L}_{height}$: Smooth L1 损失（高度预测）
    - $\mathcal{L}_{flow}$: Smooth L1 损失（光流预测）
- 提出新指标 **C-IoU**：在 bounding box 内减少假阳性惩罚，更合理评估标注不完整情况
- 在 nuScenes、nuScenes-Occupancy 和 Lyft-Level5 三个数据集上训练评估

## 实验关键数据

### 主实验

nuScenes 3D 占用预测对比（inflated annotations）：

| 方法 | IoU_c ↑ | IoU_f ↑ | IoU_all ↑ | 推理时间 |
|------|---------|---------|-----------|---------|
| PowerBEV | 36.15 | 34.18 | 34.58 | - |
| OccFormer | 41.68 | 28.55 | 31.00 | - |
| OCFNet | 40.25 | 30.38 | 32.33 | 173ms |
| **EfficientOCF** | **43.25** | **36.11** | **37.46** | **82.33ms** |

### 消融实验

各组件消融（nuScenes, 2D BEV IoU）：

| 配置 | IoU_c | IoU_f | IoU_all |
|------|-------|-------|---------|
| Baseline (avg pooling) | 33.62 | 30.07 | 30.77 |
| + 双池化 | 提升 | 提升 | 提升 |
| + 高度预测 | 进一步提升 | - | - |
| + 实例精炼 | **最高** | **最高** | **最高** |

### 关键发现

1. **推理速度提升 2 倍以上**：82.33ms vs OCFNet 的 173ms，得益于 2D 预测替代 3D
2. **实例精炼显著改善未来帧预测**：IoU_f 从基线提升明显，尤其在远未来时步
3. **C-IoU 指标更合理**：在标注不完整（LiDAR 稀疏导致）的 nuScenes-Occupancy 上，C-IoU 减少了对bounding box内假阳性的不公平惩罚
4. **在三个数据集上均达到 SOTA**，验证了方法的通用性

## 亮点与洞察

1. **空间解耦的 BEV+高度思路简洁高效**：利用 3D 空间大量空体素的特性，只为占用网格保存高度信息
2. **时间解耦中"实例传播优于端到端预测"的洞察**非常实用：对形状保持确实有明显帮助
3. **C-IoU 指标**填补了不完整标注下 OCF 评估的空白

## 局限与展望

1. 高度预测为每列单一高度值，无法表示垂直方向上的多层占用（如立交桥）
2. 实例精炼依赖光流质量，远距离/遮挡物体的光流可能不准确
3. C-IoU 虽更合理但仍在 bounding box 级别，未完全解决标注质量问题
4. 可探索与语义预测联合的多任务框架

## 相关工作与启发

- **OCFNet (Cam4DOcc)**: 首个视觉 3D OCF benchmark，本文在此基础上提出更高效的解耦范式
- **PowerBEV**: 2D BEV 实例预测方法，用后向向心流实现实例关联。EfficientOCF 扩展此思路到 3D
- **FIERY**: 首个环视摄像头输入的 BEV 实例预测方法
- **Lift-Splat-Shoot**: 2D 到 3D 特征提升的经典方法，被 EfficientOCF 的感知模块采用

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 时空解耦的设计思路清晰实用，BEV+高度表征对效率提升显著
- **实验充分度**: ⭐⭐⭐⭐ — 三个数据集、多种评估指标、消融实验完整
- **写作质量**: ⭐⭐⭐⭐ — 方法动机和设计逻辑清晰
- **价值**: ⭐⭐⭐⭐ — 对自动驾驶实时占用预测有实用价值，推理速度和精度的双重提升

<!-- RELATED:START -->

## 相关论文

- [Trajectory Mamba: Efficient Attention-Mamba Forecasting Model Based on Selective SSM](trajectory_mamba_efficient_attention-mamba_forecasting_model_based_on_selective_.md)
- [Occupancy Learning with Spatiotemporal Memory](../../ICCV2025/autonomous_driving/occupancy_learning_with_spatiotemporal_memory.md)
- [GaussianFormer-2: Probabilistic Gaussian Superposition for Efficient 3D Occupancy Prediction](gaussianformer-2_probabilistic_gaussian_superposition_for_efficient_3d_occupancy.md)
- [SDKD: Frequency-Aligned Knowledge Distillation for Lightweight Spatiotemporal Forecasting](../../ICCV2025/autonomous_driving/frequency-aligned_knowledge_distillation_for_lightweight_spatiotemporal_forecast.md)
- [DecoupledGaussian: Object-Scene Decoupling for Physics-Based Interaction](decoupledgaussian_object-scene_decoupling_for_physics-based_interaction.md)

<!-- RELATED:END -->
