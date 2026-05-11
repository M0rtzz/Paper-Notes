---
title: >-
  [论文解读] TopoTTA: Topology-Enhanced Test-Time Adaptation for Tubular Structure Segmentation
description: >-
  [ICCV 2025][图像分割][管状结构分割] 首个针对管状结构分割（TSS）的测试时适应（TTA）框架，通过拓扑元差分卷积（TopoMDCs）适应跨域拓扑结构差异，并通过拓扑硬样本生成（TopoHG）策略修复拓扑连续性断裂，在10个数据集上平均clDice提升31.81%。
tags:
  - ICCV 2025
  - 图像分割
  - 管状结构分割
  - 测试时适应
  - 拓扑连续性
  - 差分卷积
  - 域偏移
---

# TopoTTA: Topology-Enhanced Test-Time Adaptation for Tubular Structure Segmentation

**会议**: ICCV 2025  
**arXiv**: [2508.00442](https://arxiv.org/abs/2508.00442)  
**代码**: 无  
**领域**: 图像分割  
**关键词**: 管状结构分割, 测试时适应, 拓扑连续性, 差分卷积, 域偏移

## 一句话总结
首个针对管状结构分割（TSS）的测试时适应（TTA）框架，通过拓扑元差分卷积（TopoMDCs）适应跨域拓扑结构差异，并通过拓扑硬样本生成（TopoHG）策略修复拓扑连续性断裂，在10个数据集上平均clDice提升31.81%。

## 研究背景与动机

管状结构分割（TSS）在血流动力学分析和路线规划等应用中具有重要价值。然而与语义分割或肿瘤分割不同，TSS 对域偏移更加敏感——不同域之间不仅存在对比度、噪声等常规变化，还存在轨迹、曲率、分支模式和粗细等**拓扑结构特征**的差异。

现有 TTA 方法主要分为两类：(1) 基于归一化的方法（如 TENT）通过更新 BN 层参数来适应新域，但在严重像素级类别不平衡的分割任务中效果有限；(2) 基于教师-学生的方案（如 CoTTA）可更新全部参数，但对管状结构缺乏针对性处理。

**核心矛盾**在于：现有 TTA 方法是"通用型"的，无法感知管状结构独有的两大挑战——
- **挑战1：不一致的拓扑结构**。源域和目标域的管状结构在粗细、曲率、分支模式上可能存在巨大差异，通用的模型适应策略难以捕获这些跨域拓扑变化。
- **挑战2：脆弱的拓扑连续性**。跨域场景下区分前景/背景的局部特征（颜色、纹理、对比度）会显著变化，导致前景像素被错误分类，从而破坏管状结构的连续性。

**核心idea**：设计一个两阶段的拓扑增强TTA框架——Stage 1 通过8个方向性的拓扑元差分卷积增强模型对不同拓扑结构的适应能力（不修改预训练参数），Stage 2 通过生成伪断裂硬样本并对齐预测来提升拓扑连续性。

## 方法详解

### 整体框架
TopoTTA 包含两个阶段：Stage 1 进行拓扑结构适应（Topological Structure Adaptation），Stage 2 进行拓扑连续性精炼（Topological Continuity Refinement）。两个阶段各执行3次迭代，共6次适应迭代。模型采用教师-学生范式，教师参数通过EMA更新。

### 关键设计

1. **拓扑元差分卷积（TopoMDCs）**:

    - 功能：在不引入额外参数的前提下，用8个方向性差分卷积替换编码器中的 $3 \times 3$ 卷积，增强模型对各种拓扑模式的感知能力。
    - 核心思路：受中心差分卷积（CDC）启发，但CDC仅计算中心像素与邻居之间的差异，缺乏对管状结构方向性和连续性的感知。TopoMDCs 将差分扩展到8个方向（4个正交 + 4个对角），模拟各种基本拓扑模式。以左上方向 $\mathcal{C}_1$ 为例：
    $\mathcal{C}_1(r_x,r_y) = \mathcal{C}_c(r_x,r_y) - \sum_{(\Delta r_x,\Delta r_y)\in\mathcal{R}_1} w(\Delta r_x,\Delta r_y)\cdot x_{\text{in}}(r_x,r_y) + \sum w \cdot x_{\text{in}}(r_x+1,r_y+1)$
    - 自适应组合：将输入图像分成 $n \times n$ 个不重叠patch，每个patch学习一组可学习路由参数 $\boldsymbol{\delta}_j$（仅8个标量），控制不同方向TopoMDC的权重。最终输出为：
    $\hat{x}_{\text{out}} = \mathcal{C}_0(x_{\text{in}}) - \sum_{j=1}^{n \times n} \boldsymbol{\delta}_j \sum_{i=1}^{8} \mathcal{C}_i(x_{\text{in}}^j)$
    - 设计动机：管状结构在局部区域的拓扑模式相似但远距离区域差异大，因此按patch分配不同路由参数可自适应增强不同区域的拓扑感知。

2. **拓扑硬样本生成（TopoHG）**:

    - 功能：在测试图像上构造伪断裂（pseudo-breaks），生成挑战性样本，迫使模型学习目标域的前景-背景判别特征。
    - 核心思路分三步：
        - **Step 1 关键点选择**：从教师模型预测中选取置信度 $> \tau = 0.95$ 的前景点，随机选 $N_p = k \cdot |\mathcal{P}|$ 个关键点。
        - **Step 2 滑动搜索**：对每个关键点，以 $s \times s$ 窗口为前景窗口，在相邻区域滑动搜索伪标签置信度最低的背景窗口。
        - **Step 3 频域伪断裂生成**：对前景窗口和背景窗口进行FFT变换，交换低频分量（保留高频前景特征），生成看起来"断裂"但保留关键前景特征的硬样本：
       $$x_p^{\text{swap}} = \text{iFFT}(f_p^{\text{fg}} \cdot (1 - m_{\text{low}}) + f_p^{\text{bg}} \cdot m_{\text{low}})$$
    - 设计动机：通过模拟拓扑断裂，引导模型关注区分前景/背景的局部特征，增强跨域场景下的连续性。

3. **路由参数更新与模型更新**:

    - Stage 1仅更新路由参数 $\boldsymbol{\delta}$，不修改预训练网络参数，使用熵最小化损失：$\mathcal{L}_{\text{EM}} = -\sum_{j=1}^{2} \hat{y}_j \log(\hat{y}_j)$。每个新样本到来时 $\boldsymbol{\delta}$ 重置为零。
    - Stage 2使用交叉熵一致性损失更新全部参数，伪断裂区域权重乘10倍：$\mathcal{W}(u,v) = 10$（断裂区域），$1$（其他区域）。

### 损失函数 / 训练策略
- Stage 1：熵最小化损失更新路由参数 $\boldsymbol{\delta}$
- Stage 2：加权交叉熵一致性损失 $\mathcal{L}_{\text{CE}}$ 更新学生网络全部参数
- 教师网络通过EMA更新
- 每张图像独立适应6次（每stage 3次）
- TopoTTA 是即插即用方案，兼容CNN基础的TSS模型

## 实验关键数据

### 主实验
在4个场景（视网膜血管、道路提取、微观神经元、视网膜OCT血管造影）共10个数据集上进行跨域测试：

| 数据集迁移 | 指标 | TopoTTA | CoTTA (次优) | 提升 |
|-----------|------|---------|-------------|------|
| DRIVE→CHASE | clDice | **77.05%** | 71.53% | +5.52 |
| DRIVE→STARE | clDice | **62.74%** | 59.86% | +2.88 |
| CHASE→DRIVE | clDice | **70.26%** | 64.80% | +5.46 |
| Neub1→Neub2 | Dice | **66.88%** | 52.21% | +14.67 |
| OCTA500→ROSE | clDice | **78.24%** | 75.65% | +2.59 |

### 消融实验
在视网膜血管分割场景上的平均结果：

| 配置 | Dice(%) | clDice(%) | β↓ | 说明 |
|------|---------|-----------|-----|------|
| Baseline | 65.37 | 61.69 | 82.24 | 仅更新BN统计 |
| + TopoMDCs (Stage1) | 68.70 | 65.14 | 76.28 | 拓扑结构适应 |
| Baseline★ + TopoHG (Stage2) | 68.82 | 66.61 | 73.63 | 拓扑连续性精炼 |
| TopoTTA (完整) | **69.87** | **67.81** | **73.27** | 两阶段联合 |

TopoMDC类型消融：正交(𝒞₁₋₄) + 对角(𝒞₅₋₈) 组合效果最优，8方向全部使用达到最佳性能。

### 关键发现
- TopoTTA在所有10个跨域实验中均显著提升clDice,平均提升31.81%
- TopoMDCs仅引入极少量路由参数（每patch 8个标量），不增加网络参数
- TopoHG的频域低频交换设计优于高斯模糊、随机噪声和空域图像交换
- 在UNet和CS2Net两种基线上均有一致的提升（即插即用）

## 亮点与洞察
- **首个TSS专用TTA框架**，填补了管状结构分割在TTA领域的空白
- TopoMDCs设计精妙：用8个方向差分卷积模拟管状结构的基本拓扑模式，既增强了拓扑感知又不增加网络参数
- TopoHG策略新颖：在频域生成伪断裂，模拟真实的连续性破坏场景，引导模型"修复"断裂
- 两阶段解耦设计合理：Stage 1专注拓扑结构适应，Stage 2专注拓扑连续性精炼
- 路由参数每样本重置为零，保证对每个样本独立适应

## 局限与展望
- 目前仅适用于CNN基础的模型，对Transformer架构的支持尚未验证
- 每张图像需要6次迭代适应，对实时性要求高的场景可能不够高效
- TopoMDCs仅替换3×3卷积层，其他卷积核尺寸未考虑
- patch数量 $n \times n$ 是固定的超参数，未能自适应于不同分辨率和结构密度
- 频域伪断裂生成依赖于预设的低频掩码，缺乏自适应的频率选择机制

## 相关工作与启发
- 中心差分卷积（CDC）的方向性扩展思路可推广到其他细粒度视觉任务
- TTA领域从通用走向任务特异化是重要趋势，其他具有独特结构特征的分割任务（如骨骼、裂缝）也可借鉴类似思路
- 频域数据增强用于测试时适应是有前景的方向

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个TSS专用TTA框架，TopoMDCs和TopoHG设计新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 4个场景10个数据集，2个基线网络，充分的消融实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，问题动机阐述充分，但公式较多略显密集
- 价值: ⭐⭐⭐⭐ 填补TSS+TTA的空白，实际应用价值明确

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Hybrid-TTA: Continual Test-time Adaptation via Dynamic Domain Shift Detection](hybrid-tta_continual_test-time_adaptation_via_dynamic_domain_shift_detection.md)
- [\[ICCV 2025\] Correspondence as Video: Test-Time Adaption on SAM2 for Reference Segmentation in the Wild](correspondence_as_video_test-time_adaption_on_sam2_for_reference_segmentation_in.md)
- [\[ICML 2025\] IT³: Idempotent Test-Time Training](../../ICML2025/segmentation/it3_idempotent_test-time_training.md)
- [\[CVPR 2026\] The Golden Subspace: Where Efficiency Meets Generalization in Continual Test-Time Adaptation](../../CVPR2026/segmentation/the_golden_subspace_where_efficiency_meets_generalization_in_continual_test-time.md)
- [\[ICCV 2025\] DeRIS: Decoupling Perception and Cognition for Enhanced Referring Image Segmentation through Loopback Synergy](deris_decoupling_perception_and_cognition_for_enhanced_referring_image_segmentat.md)

</div>

<!-- RELATED:END -->
