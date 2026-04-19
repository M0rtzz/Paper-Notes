---
title: >-
  [论文解读] GaussianFormer-2: Probabilistic Gaussian Superposition for Efficient 3D Occupancy Prediction
description: >-
  [CVPR 2025][自动驾驶][3D占用预测] 本文提出GaussianFormer-2，通过概率高斯叠加模型从概率视角重新诠释3D语义高斯——用概率乘法定理聚合几何、高斯混合模型归一化语义，配合基于分布的初始化模块，在nuScenes和KITTI-360上以仅8.9%的高斯数量达到SOTA性能。
tags:
  - CVPR 2025
  - 自动驾驶
  - 3D占用预测
  - 高斯表示
  - 概率模型
  - 稀疏表示
  - nuScenes
---

# GaussianFormer-2: Probabilistic Gaussian Superposition for Efficient 3D Occupancy Prediction

**会议**: CVPR 2025  
**arXiv**: [2412.04384](https://arxiv.org/abs/2412.04384)  
**代码**: [huang-yh/GaussianFormer](https://github.com/huang-yh/GaussianFormer)  
**领域**: 自动驾驶  
**关键词**: 3D占用预测, 高斯表示, 概率建模, 稀疏场景表示, 高斯混合模型

## 一句话总结

本文提出GaussianFormer-2，从概率视角重新诠释3D语义高斯：每个高斯表示其邻域被占用的概率分布，通过概率乘法聚合几何预测、高斯混合模型归一化语义预测，彻底消除了高斯描述空区域和相互冗余重叠的问题，以仅8.9%的高斯数量达到SOTA。

## 研究背景与动机

### 领域现状
3D语义占用预测（Occupancy Prediction）是视觉自动驾驶的重要任务，可以精细描述周围环境的几何和语义信息。主流方法包括：(1) 基于密集体素的方法（计算量大）；(2) 基于平面（BEV/TPV）的方法（压缩损失信息）；(3) 基于3D语义高斯的稀疏方法（GaussianFormer，以对象为中心但仍有冗余）。

### 现有痛点
1. **高斯仍然描述空区域**：在GaussianFormer中，高斯通过语义属性 $\mathbf{c}$ 统一建模占用和非占用区域，导致在户外场景（大部分空间为空）中，大量高斯被分类为"空"，高斯利用率极低
2. **聚合过程导致冗余重叠**：GaussianFormer通过简单加法聚合各高斯的贡献（Eq.1），导致语义logits无界。模型为拟合这种无界输出会学习分配更多高斯到同一区域，加剧重叠
3. **初始化不够精准**：原有的learnable initialization策略在训练开始时随机初始化高斯位置，但高斯的局部感受野限制了它们在后续refinement中"移动"到正确位置的能力

### 核心矛盾
如何让3D语义高斯真正做到"以对象为中心"——只描述被占用的区域，不浪费计算在空区域上，同时避免高斯之间的冗余重叠？

### 切入角度
从概率的角度重新诠释高斯：将每个高斯理解为其邻域被占用的**概率分布**，中心处概率为100%并随距离指数衰减。这自然地限制了高斯只能描述非空区域。

### 核心idea
提出概率高斯叠加模型：(1) 几何预测——用概率乘法定理聚合各高斯的占用概率，避免加法聚合的无界问题；(2) 语义预测——用高斯混合模型（GMM）产生归一化的语义预测，防止不必要的重叠。同时设计基于分布的初始化模块，学习逐像素的占用分布而非表面深度，更精准地初始化高斯。

## 方法详解

### 整体框架
GaussianFormer-2保持了GaussianFormer的整体架构：2D图像backbone提取特征 → 初始化高斯 → 多块注意力Refinement（自编码、图像交叉注意力、属性优化）→ 聚合输出占用预测。核心改变在于几何/语义的聚合方式和高斯初始化策略。

### 关键设计

#### 1. 概率高斯叠加——几何预测

- **功能**：限制每个高斯只描述被占用区域，并通过概率乘法避免冗余重叠
- **核心思路**：
    - 将每个高斯中心处的占用概率设为100%，随距离按协方差矩阵指数衰减：$\alpha(\mathbf{x}; \mathbf{G}) = \exp(-\frac{1}{2}(\mathbf{x}-\mathbf{m})^T \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\mathbf{m}))$
    - 假设各高斯的占用概率独立，通过概率乘法定理聚合总占用概率：$\alpha(\mathbf{x}) = 1 - \prod_{i=1}^{P}(1 - \alpha(\mathbf{x}; \mathbf{G}_i))$
- **设计动机**：
    - 高斯中心概率=100%意味着每个高斯**必须**描述被占用区域，无法"浪费"在空区域上
    - 概率乘法确保 $\alpha(\mathbf{x}) \geq \alpha(\mathbf{x}; \mathbf{G}_i)$：只要一个点靠近任一高斯就被预测为占用，无需多个高斯重叠来增大概率
    - 相比加法聚合（GaussianFormer），概率乘法天然有界（输出在[0,1]内），消除了驱动重叠的梯度信号

#### 2. 高斯混合模型——语义预测

- **功能**：产生归一化的语义类别概率，避免无界logits导致的高斯重叠
- **核心思路**：
    - 去除语义属性中的"空"类别（已由几何预测处理）
    - 将所有高斯视为GMM：用opacity $a_i$ 作为先验概率（$l^1$归一化），高斯分布作为条件概率，softmax归一化的 $\tilde{\mathbf{c}}_i$ 作为语义
    - 通过贝叶斯后验聚合：$\mathbf{e}(\mathbf{x}) = \sum_{i=1}^P p(\mathbf{G}_i|\mathbf{x}) \tilde{\mathbf{c}}_i$
    - 最终占用预测：$\hat{\mathbf{o}}(\mathbf{x}) = [1-\alpha(\mathbf{x}); \alpha(\mathbf{x}) \cdot \mathbf{e}(\mathbf{x})]$
- **设计动机**：GMM天然产生归一化的输出，每个点的语义被最近的高斯主导而非所有高斯的无差别叠加。这鼓励每个高斯专注于空间中独立的区域，减少冗余

#### 3. 基于分布的初始化模块

- **功能**：为概率高斯提供更精准的样本级初始化，将高斯直接初始化到被占用区域附近
- **核心思路**：
    - 对每个像素沿射线等间隔采样R个参考点，查询GT占用标注获取二值标签 $\mathbf{l} = \{l_i\}_{i=1}^R$
    - 用图像backbone B和分布预测器M预测逐像素的占用分布 $\hat{\mathbf{l}}$
    - 用BCE损失训练：$loss_{init} = BCE(\hat{\mathbf{l}}, \mathbf{l}) = BCE(M(B(\mathcal{I})), \mathbf{l})$
- **设计动机**：
    - 不同于深度预测（只能捕捉可见表面），分布预测学习完整的占用分布（包括被遮挡的区域）
    - 不需要额外的LiDAR监督，仅用占用标注即可
    - 概率高斯要求初始化在非空区域附近（因为中心概率=100%），learnable initialization无法满足这一需求

## 实验关键数据

### 主实验 — nuScenes（环视3D语义占用预测）

| 方法 | IoU | mIoU | 高斯/体素数 |
|------|-----|------|-----------|
| MonoScene | 23.96 | 7.31 | 200×200×16 |
| TPVFormer | 30.86 | 17.10 | - |
| SurroundOcc | 31.49 | 20.30 | 200×200×16 |
| GaussianFormer | 29.83 | 19.10 | - |
| **Ours (Ch.128)** | 30.56 | 20.02 | - |
| **Ours (Ch.192)** | **31.74** | **20.82** | - |

GaussianFormer-2 (Ch.192) 的mIoU达**20.82**，超越SurroundOcc和GaussianFormer。

### KITTI-360（单目占用预测）

相比GaussianFormer，GaussianFormer-2在SSCBench-KITTI-360上取得了明显提升（论文报告超越GaussianFormer clear margin）。

### 效率对比

GaussianFormer-2仅需GaussianFormer中**8.9%**的高斯数量即可达到SOTA性能，大幅提升了高斯利用率。

### 关键发现

1. **概率乘法 vs 加法聚合**：概率乘法消除了GaussianFormer中约90%的冗余高斯（描述空区域的高斯），使有效高斯比例大幅提升
2. **GMM语义归一化关键**：移除"空"类别并用softmax归一化语义后，高斯重叠率显著降低
3. **分布初始化 vs 深度初始化**：分布初始化可以感知被遮挡的占用区域（如障碍物背后），深度初始化只能感知可见表面
4. 在barrier、bicycle、bus、car等各个类别上均有提升，尤其是barrier (+1.87)和bicycle (+2.18)提升显著

## 亮点与洞察

1. **概率视角的重新诠释优雅且深刻**：将3D语义高斯从"分布值"重新诠释为"占用概率"，这一小改变带来了整个聚合框架的根本性改善——从加法变为概率乘法，从无界变为有界，从鼓励重叠变为抑制重叠
2. **理论与实践的统一**：概率乘法定理和高斯混合模型都有坚实的数学基础，不是临时的工程trick，而是从第一性原理导出的解决方案
3. **效率提升的根本原因清晰**：高斯数量减少到8.9%不是靠剪枝或压缩，而是因为每个高斯的利用率从~10%提升到接近100%
4. **分布初始化的思路通用**：学习逐像素的3D占用分布（而非表面深度）是一个可推广到其他3D感知任务的有价值思路

## 局限性

1. 概率乘法的独立性假设在密集场景中可能不完全成立（相邻物体的占用概率可能相关）
2. GMM中的opacity先验用$l^1$归一化可能不是最优选择，更复杂的先验建模可能进一步提升性能
3. 分布初始化模块增加了额外的训练开销（需要per-pixel的占用分布预测）
4. 实验主要在nuScenes和KITTI-360上验证，更大规模数据集（如Waymo）上的验证缺失

## 相关工作与启发

- **与GaussianFormer (v1)的关系**：GaussianFormer-2是对v1的核心建模范式升级。v1提出了3D语义高斯的概念，但加法聚合和空区域建模限制了效率。v2通过概率重新诠释彻底解决了这些问题
- **与3DGS渲染中alpha compositing的类比**：概率乘法定理的形式与3DGS渲染中的alpha blending公式有异曲同工之妙，但这里用于占用预测而非渲染
- **稀疏表示的研究方向**：从体素→BEV/TPV→语义高斯→概率高斯的演进，展示了3D场景表示从密集到稀疏再到真正对象中心化的发展轨迹
- **对occupancy forecasting的启发**：概率高斯天然适合建模时序变化——高斯的运动对应物体运动，概率变化对应出现/消失

## 评分

⭐⭐⭐⭐⭐ (5/5)

数学上优雅、工程上高效、实验上全面。从概率视角重新诠释高斯表示是一个深刻且有影响力的贡献，解决了GaussianFormer v1的根本局限。以8.9%的高斯数量达到SOTA，展示了稀疏表示的真正潜力。

<!-- RELATED:START -->

## 相关论文

- [GaussianWorld: Gaussian World Model for Streaming 3D Occupancy Prediction](gaussianworld_gaussian_world_model_for_streaming_3d_occupancy_prediction.md)
- [GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction](../../ECCV2024/autonomous_driving/gaussianformer_scene_as_gaussians_for_vision-based_3d_semantic_occupancy_predict.md)
- [EVolSplat: Efficient Volume-based Gaussian Splatting for Urban View Synthesis](evolsplat_efficient_volume-based_gaussian_splatting_for_urban_view_synthesis.md)
- [ProtoOcc: 3D Occupancy Prediction with Low-Resolution Queries via Prototype-aware View Transformation](3d_occupancy_prediction_with_low-resolution_queries_via_prototype-aware_view_tra.md)
- [M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs](m2-occ_resilient_3d_semantic_occupancy_prediction_for_autonomous_driving_with_in.md)

<!-- RELATED:END -->
