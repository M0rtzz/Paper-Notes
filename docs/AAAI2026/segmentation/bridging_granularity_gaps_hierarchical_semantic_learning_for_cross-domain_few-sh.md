---
title: >-
  [论文解读] Bridging Granularity Gaps: Hierarchical Semantic Learning for Cross-Domain Few-Shot Segmentation
description: >-
  [AAAI 2026][图像分割][跨域少样本分割] 提出 HSL 框架，通过双重风格随机化 (DSR)、层次语义挖掘 (HSM) 和原型置信度调制阈值 (PCMT) 三个模块，解决跨域少样本分割中源域和目标域之间的**分割粒度差异**问题，在四个目标域数据集上达到 SOTA。
tags:
  - AAAI 2026
  - 图像分割
  - 跨域少样本分割
  - 层次语义学习
  - 风格随机化
  - 超像素
  - 原型置信度
---

# Bridging Granularity Gaps: Hierarchical Semantic Learning for Cross-Domain Few-Shot Segmentation

**会议**: AAAI 2026  
**arXiv**: [2511.12200](https://arxiv.org/abs/2511.12200)  
**代码**: [有](https://github.com/Sparkling-Water/HSLNet)  
**领域**: 分割  
**关键词**: 跨域少样本分割, 层次语义学习, 风格随机化, 超像素, 原型置信度

## 一句话总结

提出 HSL 框架，通过双重风格随机化 (DSR)、层次语义挖掘 (HSM) 和原型置信度调制阈值 (PCMT) 三个模块，解决跨域少样本分割中源域和目标域之间的**分割粒度差异**问题，在四个目标域数据集上达到 SOTA。

## 研究背景与动机

跨域少样本分割 (CD-FSS) 在仅依赖少量标注样本的条件下，对来自训练中未见过的目标域新类别进行分割。现有方法主要关注源域和目标域之间的**风格差异**（如颜色、纹理等），但忽视了一个核心问题：**分割粒度差异**。

具体来说，源域中前景（如鸟）和背景（其他类别）的差异通常是粗粒度的（很明显），但目标域中前景和背景可能非常相似（如皮肤病变区域与正常皮肤），这种细粒度的差异类似于源域中前景内部的细微区别（如鸟不同颜色羽毛之间的差异）。由于模型仅在源域上训练，倾向于将前景整体视为单一实体，无法有效捕获目标域中更细粒度的前景-背景语义划分。

核心思路：提取**层次化语义特征**，使模型在不同粒度上都具备类内一致性和类间区分性。

## 方法详解

### 整体框架

HSL 框架由三个核心模块组成：

1. **DSR (Dual Style Randomization)**: 对训练数据进行前景风格和全局风格的双重随机化
2. **HSM (Hierarchical Semantic Mining)**: 利用多尺度超像素掩码挖掘层次语义特征
3. **PCMT (Prototype Confidence-modulated Thresholding)**: 在测试时缓解前景-背景过于相似时的分割歧义

流程：先用超像素分割模型获取多尺度超像素掩码 → DSR 增强 → 图像编码器特征提取 → HSM 特征增强 → SSP 模块计算原型 → PCMT 生成最终预测。

### 关键设计

#### DSR: 双重风格随机化

**前景风格随机化 (Foreground Style Randomization)**:
- 从最粗粒度超像素掩码中随机选取一个局部区域图像 $\mathbf{I}^{local}$
- 对前景图像 $\mathbf{I}^{fg}$ 和局部区域图像进行 FFT 分解为幅度谱和相位谱
- 将两者幅度谱加权融合：$\mathbf{A}^{fusion} = \omega \mathbf{A}^{local} + (1-\omega) \mathbf{A}^{fg}$，权重 $\omega \sim N(0, \sigma_f^2)$
- 保持前景相位谱不变，用融合的幅度谱通过 IFFT 重建，模拟不同程度的前景-背景差异

**全局风格随机化 (Global Style Randomization)**:
- 通过随机卷积 (RC) 层扰动前景已随机化图像的局部纹理
- 同样基于 FFT：保留原图相位谱，使用随机卷积输出的幅度谱重建
- 避免随机卷积直接破坏内容细节

#### HSM: 层次语义挖掘

核心思想：多尺度超像素掩码天然地将图像划分为不同粒度的局部区域，可近似不同粒度的语义区域。

具体步骤：
1. 对每个尺度的超像素掩码，生成各区域的二值掩码
2. 提取浅层低级特征 $\mathbf{F}^l$ 和深层高级特征 $\mathbf{F}^h$
3. 对低级特征降采样后，用 MAP (Masked Average Pooling) 计算各区域的低级和高级原型
4. 低级原型通过两层多头自注意力 (MSA) 增强后，与高级原型加权融合：$\mathbf{p}_{ij} = \alpha \tilde{\mathbf{p}}_{ij}^l + (1-\alpha) \mathbf{p}_{ij}^h$
5. 用 RMAP 将区域原型恢复为特征图，所有尺度特征图叠加到高级特征上

这使得每个像素受多尺度区域原型影响，增强不同粒度下的类内一致性和类间区分性。

#### PCMT: 原型置信度调制阈值

针对测试时前景-背景过于相似导致分割歧义的问题：

1. 计算前景置信度图：$\mathbf{M}_q^{conf} = \mathbf{M}_q^{fg} - \mathbf{M}_q^{bg}$
2. 用 OTSU 计算自适应阈值 $t$
3. 引入**原型置信度** $C$ 衡量分割歧义的概率（基于跨视图原型相似度）
4. 最终阈值：$\frac{1}{1+e^{\beta(C+\gamma)}} t$
    - 原型置信度高 → 阈值趋近 0（等效传统相似度比较）
    - 原型置信度低 → 使用自适应阈值 $t$（缓解歧义）

### 损失函数 / 训练策略

- 使用 BCE loss 训练
- DSR 仅在训练时使用，PCMT 仅在测试时使用
- 采用 meta-learning 范式，每个 episode 包含 support set 和 query set
- 超像素掩码使用 4 个尺度：$\{5^2, 10^2, 15^2, 20^2\}$
- 使用 SGD 优化 5 个 epoch，学习率 1e-3

## 实验关键数据

### 主实验

在 PASCAL VOC 2012 + SBD 上训练，在 4 个目标域数据集上测试：

| 方法 | Backbone | Deepglobe 1/5-shot | ISIC 1/5-shot | Chest X-ray 1/5-shot | FSS-1000 1/5-shot | 平均 1/5-shot |
|------|----------|-------------------|---------------|---------------------|-------------------|--------------|
| DRA | Res-50 | 41.29/50.12 | 40.77/48.87 | 82.35/82.31 | 79.05/80.40 | 60.86/65.42 |
| LoEC | ViT-base | 42.12/51.48 | 52.91/62.43 | 83.94/84.12 | 81.05/83.69 | 65.01/70.43 |
| **HSL (Ours)** | **Res-50** | **46.13/53.80** | **48.01/55.56** | **84.57/85.34** | 78.22/80.36 | **64.23/68.77** |
| **HSL (Ours)** | **ViT-base** | **45.77/54.56** | **59.36/64.62** | **85.95/86.25** | **81.89/83.84** | **68.24/72.32** |

ViT-base 下 1-shot/5-shot 分别超越 SOTA LoEC **+3.23%** / **+1.89%**。

### 消融实验

| DSR | HSM | PCMT | Res-50 mIoU | ViT-base mIoU |
|-----|-----|------|-------------|---------------|
| ✗ | ✗ | ✗ | 57.82 | 62.24 |
| ✓ | ✗ | ✗ | 60.44 (+2.62) | 64.55 (+2.31) |
| ✗ | ✓ | ✗ | 60.92 (+3.10) | 65.29 (+3.05) |
| ✓ | ✓ | ✗ | 62.97 | 67.05 |
| ✓ | ✓ | ✓ | **64.23** | **68.24** |

多尺度超像素掩码消融：去除任一尺度均导致性能下降，4 尺度最优 (67.05 → 去除 5×5 后 66.34)。

阈值策略消融（ViT-base 1-shot）：

| 策略 | Deepglobe | ISIC | Chest | FSS | 平均 |
|------|-----------|------|-------|-----|------|
| 固定阈值 0 | 44.54 | 55.79 | 85.93 | 81.93 | 67.05 |
| OTSU | 45.57 | 59.98 | 85.81 | 80.43 | 67.95 |
| PCMT (Ours) | 45.77 | 59.36 | 85.95 | 81.89 | **68.24** |

### 关键发现

- **HSM 贡献最大**：单独引入 HSM 比 DSR 带来更大提升（3.10% vs 2.62%），表明层次语义挖掘是解决粒度差异的关键
- **PCMT 灵活平衡**：OTSU 在易歧义的 ISIC 上效果好但在接近源域的 FSS 上下降，PCMT 根据样本自适应调整阈值
- **多尺度缺一不可**：每去掉一个尺度均有损失，细粒度和粗粒度信息互补

## 亮点与洞察

1. **首次聚焦分割粒度差异**：不同于其他 CD-FSS 方法关注风格差异，本文揭示了粒度差异这一被忽视的核心问题
2. **FFT 频域操作的巧妙使用**：在前景风格随机化中，通过幅度谱融合改变前景风格而保持内容不变
3. **超像素作为层次语义先验**：多尺度超像素自然地提供不同粒度的语义分区，简洁有效
4. **PCMT 的自适应机制**：避免了"一刀切"的阈值策略，根据原型置信度在传统方法和自适应阈值间平滑过渡

## 局限与展望

- 超像素分割模型的质量直接影响 HSM 效果，对于结构简单的目标域（如医学图像）可能不是最优先验
- 4 个超像素尺度和各种超参数（$\sigma_f$, $\sigma_g$, $K$, $\alpha$, $\beta$, $\gamma$）需要调优
- PCMT 依赖 OTSU 算法，对于多模态分布的置信度图可能不鲁棒
- 仅在 PASCAL VOC → 4 个目标域的设置下验证，未讨论对更大规模源域的泛化

## 相关工作与启发

- **PATNet** (ECCV 2022)：首个引入 CD-FSS 的工作，提出将特征转换到域无关空间
- **DRA** (CVPR 2024)：通过域随机化增强泛化能力
- **LoEC** (CVPR 2025)：在特征层面进行风格扰动
- **频域方法的广泛应用**：FFT 在域适应/泛化中越来越重要，amplitude swap 思路值得在其他任务中推广

## 评分

- 新颖性: ⭐⭐⭐⭐ — 粒度差异视角新颖，DSR+HSM+PCMT 三模块协同设计合理
- 技术深度: ⭐⭐⭐⭐ — FFT 频域增强、多尺度超像素挖掘、自适应阈值调制技术扎实
- 实验充分度: ⭐⭐⭐⭐ — 消融全面，覆盖两种 backbone 和四个目标域
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，图示直观

<!-- RELATED:START -->

## 相关论文

- [Adapter Naturally Serves as Decoupler for Cross-Domain Few-Shot Semantic Segmentation](../../ICML2025/segmentation/adapter_naturally_serves_as_decoupler_for_cross-domain_few-shot_semantic_segment.md)
- [Self-Disentanglement and Re-Composition for Cross-Domain Few-Shot Segmentation](../../ICML2025/segmentation/self-disentanglement_and_re-composition_for_cross-domain_few-shot_segmentation.md)
- [Dual-Agent Optimization framework for Cross-Domain Few-Shot Segmentation](../../CVPR2025/segmentation/dual-agent_optimization_framework_for_cross-domain_few-shot_segmentation.md)
- [The Devil is in Low-Level Features for Cross-Domain Few-Shot Segmentation](../../CVPR2025/segmentation/the_devil_is_in_low-level_features_for_cross-domain_few-shot_segmentation.md)
- [InfoCLIP: Bridging Vision-Language Pretraining and Open-Vocabulary Semantic Segmentation via Information-Theoretic Alignment Transfer](infoclip_bridging_vision-language_pretraining_and_open-vocab.md)

<!-- RELATED:END -->
