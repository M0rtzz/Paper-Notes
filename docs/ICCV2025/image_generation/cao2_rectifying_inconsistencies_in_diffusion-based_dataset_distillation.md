---
title: >-
  [论文解读] CaO2: Rectifying Inconsistencies in Diffusion-Based Dataset Distillation
description: >-
  [ICCV 2025][图像生成][数据集蒸馏] 揭示了基于扩散模型的数据集蒸馏中存在的"目标不一致"和"条件不一致"两个关键问题，提出两阶段框架CaO2：第一阶段通过分类器引导的样本选择缓解目标不一致，第二阶段通过隐空间优化最大化条件似然缓解条件不一致，在ImageNet上平均提升2.3%。
tags:
  - ICCV 2025
  - 图像生成
  - 数据集蒸馏
  - 扩散模型
  - 条件一致性
  - 目标一致性
  - 隐空间优化
---

# CaO2: Rectifying Inconsistencies in Diffusion-Based Dataset Distillation

**会议**: ICCV 2025  
**arXiv**: [2506.22637](https://arxiv.org/abs/2506.22637)  
**代码**: [GitHub](https://github.com/hatchetProject/CaO2)  
**领域**: 图像生成 / 数据集蒸馏  
**关键词**: 数据集蒸馏, 扩散模型, 条件一致性, 目标一致性, 隐空间优化

## 一句话总结

揭示了基于扩散模型的数据集蒸馏中存在的"目标不一致"和"条件不一致"两个关键问题，提出两阶段框架CaO2：第一阶段通过分类器引导的样本选择缓解目标不一致，第二阶段通过隐空间优化最大化条件似然缓解条件不一致，在ImageNet上平均提升2.3%。

## 研究背景与动机

数据集蒸馏旨在构建一个紧凑的代理数据集，使在其上训练的模型能达到接近全量数据的性能。传统方法基于匹配优化（梯度匹配、特征匹配、轨迹匹配），但难以扩展到大规模高分辨率数据集。

近年来，基于扩散模型的蒸馏方法（如Minimax Diffusion、D4M）利用预训练扩散模型作为强大的分布学习器，生成代表性样本替代原始数据。然而作者发现这些方法**忽略了评估过程**，导致两个严重的不一致：

**目标不一致 (Objective Inconsistency, OI)**: 蒸馏阶段优化的是生成模型的目标（图像逼真度），而评估阶段使用的是分类目标。扩散模型生成的图像虽逼真，但不一定具有判别性。

**条件不一致 (Condition Inconsistency, CI)**: 扩散模型在实际中训练不完美，给定类别条件 $c_i$ 生成的图像 $\mathbf{x}_0^i$ 在其他类别条件下也有非零似然，即 $p_\theta(\mathbf{x}_0^i | c_j) > 0, j \neq i$，导致图像-标签配对质量不佳。

## 方法详解

### 整体框架

CaO2是一个两阶段框架：第一阶段生成图像池并通过轻量分类器筛选判别性强的样本（缓解OI），第二阶段在隐空间中优化选中样本的latent表示以最大化条件似然（缓解CI）。整个流程不需要训练扩散模型主干网络，仅需单张A6000 GPU。

### 关键设计

1. **目标引导的样本选择 (Objective-guided Sample Selection)**:
   对每个类别，先用预训练扩散模型生成 $m \times \text{IPC}$ 张图像（$m$ 为扩展因子，通常2-4即可），再用轻量预训练分类器（如ResNet-18）获取每张图像的预测概率。筛选流程：
    - 仅保留被正确分类为条件类别的样本
    - 根据任务难度选择样本：低IPC设定下选高置信度样本（简单样本），高IPC设定下选低置信度样本（困难样本）
    - 不足部分从剩余图像中随机补充

2. **条件感知的隐空间优化 (Condition-aware Latent Optimization)**:
   固定扩散模型参数，对选中的图像隐变量进行梯度优化，使其向条件似然更高的区域移动。优化目标为：
    $\min_{\mathbf{x}} \mathbb{E}_{t,\varepsilon}\left[\|\epsilon_\theta(\mathbf{x}_t, \hat{\mathbf{c}}, t) - \varepsilon\|_2^2 + \lambda\|\epsilon_\theta(\mathbf{x}_t, \hat{\mathbf{c}}, t) - \varepsilon\|_\infty\right]$
   其中 $\lambda=10$ 控制正则化强度，$\|\cdot\|_\infty$ 项防止局部区域过度偏离。时间步 $t$ 从 $[1, \hat{T}]$ 采样（$\hat{T} \ll T$），确保隐变量只受到适度扰动。每张图像优化100次迭代。

3. **任务导向的变体 (Task-oriented Variation)**:

    - 根据分类任务的验证精度判断难度，选择不同的优化条件 $\hat{\mathbf{c}}$
    - 简单任务（易区分类别）：使用真实类别标签 $c$ 作为条件
    - 困难任务（难区分类别）：使用无条件标签 $\phi$ （分类器自由引导），让条件信息嵌入图像隐变量本身
    $\hat{\mathbf{c}} = c \cdot \mathbb{1}(c \in \mathbb{C}_e) + \phi \cdot \mathbb{1}(c \in \mathbb{C}_h)$

4. **扩展到MAR (Masked Autoregressive Model)**:
   方法不限于扩散模型，也可应用于MAR等自回归生成模型。主要差异：用随机mask替代加噪操作，设计零标签嵌入替代分类器自由引导嵌入。

### 损失函数 / 训练策略

- 使用预训练DiT（256×256分辨率）作为骨干，50步去噪采样
- Adam优化器，学习率0.0006
- 每张图像优化100次迭代，固定输入噪声
- 单张RTX A6000即可完成全部实验
- 样本选择阶段扩展因子 $m=2$ 或 $m=4$ 即足够

## 实验关键数据

### 主实验

| 数据集 | IPC | SRe2L | Minimax | RDED | CaO2 (本文) | 提升 |
|--------|-----|-------|---------|------|------------|------|
| ImageWoof (ResNet-18) | 10 | 20.2 | 40.1 | 38.5 | **45.6** | +5.5 |
| ImageWoof (ResNet-18) | 50 | 23.3 | 67.0 | 68.5 | **68.9** | +0.4 |
| ImageNette (ResNet-18) | 10 | 29.4 | 61.4 | 61.4 | **65.0** | +3.6 |
| ImageNette (ResNet-50) | 50 | 71.2 | 77.1 | 78.0 | **82.7** | +4.7 |
| ImageNet-100 (ResNet-18) | 50 | 27.0 | 63.9 | 61.6 | **68.0** | +4.1 |
| ImageNet-1K (ResNet-18) | 10 | 21.3 | 44.3 | 42.0 | **46.1** | +1.8 |
| ImageNet-1K (ResNet-50) | 10 | 28.4 | 49.7 | 43.6 | **53.0** | +3.3 |

### 消融实验

| OSS | CLO | ImageWoof IPC=10 | ImageWoof IPC=50 | ImageNette IPC=10 | ImageNette IPC=50 |
|-----|-----|-----------------|-----------------|-------------------|-------------------|
| ✗ | ✗ | 38.7 | 66.1 | 61.4 | — |
| ✓ | ✗ | 42.3 | 67.5 | 63.2 | — |
| ✗ | ✓ | 41.8 | 67.2 | 62.8 | — |
| ✓ | ✓ | **45.6** | **68.9** | **65.0** | — |

### 关键发现

- 两个组件（OSS和CLO）各自独立贡献相当，组合使用效果更优
- 在低IPC设定下提升更加显著（每张图对数据集质量影响更大）
- ImageNette的平均提升（4.3%）大于ImageWoof（1.6%），说明多样性维护更容易见效
- 方法独立于评估模型——同一蒸馏数据集可跨ResNet-18/50/101使用
- 扩展到MAR后同样有效，验证了框架的通用性

## 亮点与洞察

- **问题定义精准**：将扩散蒸馏的问题归结为两种可量化的"不一致"，定义清晰、形式化严谨
- **极其高效**：不需要训练/微调任何生成模型，仅对隐变量做轻量优化，单GPU即可完成1000类ImageNet的蒸馏
- **即插即用**：可作为现有扩散蒸馏方法的后处理模块，也可直接用于DiT随机采样的图像

## 局限与展望

- 样本选择依赖预训练分类器的质量，分类器本身的偏差可能传递到蒸馏集
- 隐空间优化可能导致视觉质量轻微下降（虽然判别性提高）
- 在超大IPC设定下优势减小
- 条件优化策略（简单/困难类别区分）需要额外的验证精度信息

## 相关工作与启发

- 与Minimax Diffusion的区别：后者通过微调扩散模型提升生成代表性，而CaO2不修改模型参数
- 与D4M的区别：D4M用原型学习聚类隐变量，仍然缺乏分类目标指导
- 核心启发：**生成模型的"好图"不等于分类的"好样本"**——这个洞察对所有基于生成模型的数据增强策略都有启示意义

## 评分

- **新颖性**: ⭐⭐⭐⭐ 问题分析深刻，两个不一致的定义令人信服
- **实验充分度**: ⭐⭐⭐⭐ 多数据集、多模型、多IPC的系统实验
- **写作质量**: ⭐⭐⭐⭐⭐ 形式化定义清晰，问题-方案对应关系明确
- **价值**: ⭐⭐⭐⭐ 对扩散蒸馏领域有方法论指导意义

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Learnability-Guided Diffusion for Dataset Distillation](../../CVPR2026/image_generation/learnability-guided_diffusion_for_dataset_distillation.md)
- [\[ICML 2025\] Taming Diffusion for Dataset Distillation with High Representativeness (D³HR)](../../ICML2025/image_generation/taming_diffusion_for_dataset_distillation_with_high_representativeness.md)
- [\[NeurIPS 2025\] Entropy Rectifying Guidance for Diffusion and Flow Models](../../NeurIPS2025/image_generation/entropy_rectifying_guidance_for_diffusion_and_flow_models.md)
- [\[ICCV 2025\] SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation](sanasprint_onestep_diffusion_with_continuoustime_consistency.md)
- [\[ICCV 2025\] SuperEdit: Rectifying and Facilitating Supervision for Instruction-Based Image Editing](superedit_rectifying_and_facilitating_supervision_for_instruction-based_image_ed.md)

<!-- RELATED:END -->
