---
title: >-
  [论文解读] CD-Buffer: Complementary Dual-Buffer Framework for Test-Time Adaptation in Adverse Weather Object Detection
description: >-
  [CVPR 2026][目标检测][测试时适应] 提出 CD-Buffer 框架，通过统一的域差异度量驱动减性缓冲（通道抑制）和加性缓冲（轻量适配器补偿）的互补协作，实现跨不同严重程度恶劣天气条件下的鲁棒测试时目标检测适应。
tags:
  - CVPR 2026
  - 目标检测
  - 测试时适应
  - 恶劣天气
  - 通道自适应
  - 加性-减性互补
---

# CD-Buffer: Complementary Dual-Buffer Framework for Test-Time Adaptation in Adverse Weather Object Detection

**会议**: CVPR 2026  
**arXiv**: [2603.26092](https://arxiv.org/abs/2603.26092)  
**代码**: [网站](https://wkfsksdl99.github.io/cd_buffer/) (有)  
**领域**: Object Detection  
**关键词**: 测试时适应, 恶劣天气, 目标检测, 通道自适应, 加性-减性互补

## 一句话总结

提出 CD-Buffer 框架，通过统一的域差异度量驱动减性缓冲（通道抑制）和加性缓冲（轻量适配器补偿）的互补协作，实现跨不同严重程度恶劣天气条件下的鲁棒测试时目标检测适应。

## 研究背景与动机

**领域现状**：测试时适应（TTA）通过在线更新源域预训练模型来应对域迁移，无需离线重训或目标标签。现有 TTA 方法分为加性方法（引入轻量模块学习目标特定调整）和减性方法（移除对域敏感的通道）。

**现有痛点**：加性方法（如 BufferTTA）在中等域迁移下效果好，但严重退化时难以修复；减性方法（如 PruningTTA）在严重域迁移下表现好，但在中等迁移时会过度删除可修复的有用信息。两种范式各自只在有限范围内有效。

**核心矛盾**：真实场景中同一图像的不同特征通道可能面临不同程度的域迁移——某些通道严重退化需要抑制，其他通道只需微调。现有方法对所有通道统一处理，无法适应这种异质性。

**本文目标**：设计自适应机制，根据每个通道的域迁移程度自动平衡"移除"和"补偿"策略。

**切入角度**：测量通道级域差异，用统一度量同时驱动两种互补操作。

**核心 idea**：差异驱动的双缓冲耦合——严重迁移的通道被抑制并获得强补偿，中等迁移的通道被精细调整，稳定通道基本不干预。

## 方法详解

### 整体框架

基于 Faster R-CNN + ResNet-50 骨干网络。在每个 BN 层设置减性缓冲（可学习掩码分数），在残差路径设置加性缓冲（轻量 1×1 + 3×3 卷积适配器），两者通过统一的通道差异度量 $D_c$ 耦合。

### 关键设计

1. **特征级域差异分数 (Feature-Level Domain Discrepancy)**：结合图像级和实例级两个层面的特征差异：
   $$D^I = \frac{\sum_{n=1}^{N}\|X_t^c - \bar{X}_s^c\|_1}{NHW}, \quad D^O = \frac{\sum_{m=1}^{M}\|x_t^c - \bar{x}_s^c\|_1}{Mhw}$$
   $$D = D^I + D^O$$
   其中 $X_s$ 为源域预计算均值特征，$x_s$ 为实例级（通过 RoI Align）均值特征。设计动机：目标检测需要同时考虑全局场景和局部实例的域迁移。高 $D$ 表示通道严重偏移需干预，低 $D$ 表示仅需微调。

2. **减性缓冲 (Subtractive Buffer)**：引入可学习掩码分数 $s \in \mathbb{R}^C$（从 BN 权重初始化 $s_c = |\gamma_c|$），通过差异加权正则化驱动通道抑制：
   $$\mathcal{L}_{mask} = \frac{1}{C}\sum_c \|D_c \cdot s_c\|_1$$
   高差异通道的 $D_c$ 大，产生更强梯度压低 $s_c$，促使该通道被阈值化掩码抑制。通过动态百分位阈值 $\tau = \text{Percentile}(\{|s_c^{(l)}|\}, \rho_{target})$ 控制整体抑制率（5%），并用 straight-through estimator 保证梯度流通。另有随机重激活防止永久删除有用通道。

3. **加性缓冲 (Additive Buffer)**：轻量适配器 $F_{add} = \frac{\text{Conv}_{1\times1}(F) + \text{Conv}_{3\times3}(F)}{2} \odot \boldsymbol{\alpha}$，使用可学习通道缩放因子 $\boldsymbol{\alpha}$（初始化为 $10^{-2}$）。关键创新是**逆向软掩码**：
   $$\hat{m}_{soft}^{-1} = k \cdot \text{Norm}(\mathbf{1} - \hat{m}_{soft})$$
   被减性缓冲强抑制的通道（$\hat{m}_{soft}$ 接近 0）获得最强的加性补偿（$\hat{m}_{soft}^{-1}$ 最大），实现"移除多少就补偿多少"的自动平衡。设计动机：减性缓冲删除严重退化特征的同时不可避免丢失信息，加性缓冲的逆向调制自动为这些通道提供最强补偿。

### 损失函数 / 训练策略

$$\mathcal{L} = \mathcal{L}_{align} + \lambda_{reg} \cdot \mathcal{L}_{mask}$$
- $\mathcal{L}_{align}$：源域和目标域特征均值/方差的 L1 对齐
- 层级梯度缩放：根据层差异 $D^l$ 放大加性缓冲梯度
- 联合优化：加性缓冲参数、BN 仿射参数、掩码分数

## 实验关键数据

### 主实验

| 方法 | KITTI Fog 50m | Fog 75m | Fog 150m | Rain 200mm | Rain 100mm |
|------|-------------|---------|----------|-----------|------------|
| Direct Test | 21.27 | 30.84 | 50.45 | 47.11 | 65.32 |
| BufferTTA (加性) | 23.21 | 33.12 | 52.50 | 50.96 | 69.28 |
| PruningTTA (减性) | 33.97 | 42.83 | 58.58 | 50.94 | 65.42 |
| ActMAD | 39.65 | 49.95 | 60.18 | 56.37 | 62.94 |
| **CD-Buffer** | **44.80** | **56.06** | **68.42** | **63.22** | 71.40 |

| 方法 | ACDC Fog | ACDC Snow | ACDC Rain | ACDC Night |
|------|---------|-----------|-----------|------------|
| Direct Test | 16.50 | 11.04 | 7.82 | 4.83 |
| BufferTTA | 24.16 | 17.18 | 11.70 | 6.98 |
| **CD-Buffer** | **24.45** | 15.41 | **13.71** | **8.92** |

### 消融实验

| 配置 | KITTI Fog 50m | 说明 |
|------|-------------|------|
| 仅加性缓冲 | ~23.2 | 严重域迁移下效果有限 |
| 仅减性缓冲 | ~34.0 | 移除退化特征但信息丢失 |
| 双缓冲无耦合 | 显著弱于完整方法 | 独立操作缺乏协调 |
| 完整 CD-Buffer | 44.80 | 差异驱动耦合最优 |

### 关键发现

1. **互补模式验证**：BufferTTA 在中等迁移（Rain 75mm）表现好但严重迁移（Fog 50m）差；PruningTTA 呈现相反模式。CD-Buffer 在所有严重程度下一致优秀。
2. **连续 TTA 稳定性**：在 KITTI Fog 50m→75m→150m 连续适应实验中，CD-Buffer 适应最快且表现最稳定。ActMAD 初始提升快但收敛不稳定。
3. 逆向软掩码的耦合机制是性能提升的关键——将两种范式从独立策略统一为协调系统。

## 亮点与洞察

- 首次系统揭示加性和减性 TTA 范式的互补特性，并提供直觉性的解释框架。
- 差异驱动耦合设计优雅简洁：一个度量 $D_c$ 同时驱动两种相反操作，自动实现通道级差异化处理。
- 逆向软掩码是亮点设计：不需要额外网络来决定补偿强度，直接从减性缓冲的掩码分数推导。
- 批量大小不依赖：不像 BN 统计方法受小批量影响，CD-Buffer 通过结构修改实现适应。

## 局限与展望

- 仅在 Faster R-CNN + ResNet-50 上实验，未验证 DETR 类端到端检测器。
- 通道抑制率固定为 5%，可考虑自适应确定。
- 源域特征统计需预计算并存储，增加部署复杂度。
- 未探索与自监督目标（如对比学习）的结合。

## 相关工作与启发

- 与 ActMAD 的多层特征对齐不同，CD-Buffer 通过通道级差异化处理提供更精细的适应。
- 逆向掩码思路可推广到其他需要平衡保留和修改的场景（如模型压缩中的知识蒸馏）。
- 为 TTA 领域提供了"加性 vs 减性"的分类框架，有助于理解未来工作。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 加性/减性互补的洞察新颖，差异驱动耦合机制设计优雅
- 实验充分度: ⭐⭐⭐⭐ 多数据集多严重度评估充分，连续 TTA 实验有意义
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述完整
- 价值: ⭐⭐⭐⭐ 为 TTA 提供了新的范式整合思路

<!-- RELATED:START -->

## 相关论文

- [Test-Time Adaptive Object Detection with Foundation Model](../../NeurIPS2025/object_detection/test-time_adaptive_object_detection_with_foundation_model.md)
- [Beyond Prompt Degradation: Prototype-Guided Dual-Pool Prompting for Incremental Object Detection](beyond_prompt_degradation_prototype-guided_dual-pool_prompting_for_incremental_o.md)
- [Efficient Test-Time Adaptive Object Detection via Sensitivity-Guided Pruning](../../CVPR2025/object_detection/efficient_test-time_adaptive_object_detection_via_sensitivity-guided_pruning.md)
- [Robust Long-term Test-Time Adaptation for 3D Human Pose Estimation through Motion Discretization](../../AAAI2026/object_detection/robust_long-term_test-time_adaptation_for_3d_human_pose_estimation_through_motio.md)
- [Is Less More? Exploring Token Condensation as Training-free Test-time Adaptation](../../ICCV2025/object_detection/is_less_more_exploring_token_condensation_as_training-free_test-time_adaptation.md)

<!-- RELATED:END -->
