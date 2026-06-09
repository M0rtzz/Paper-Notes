---
title: >-
  [论文解读] Enhancing Reward Models for High-quality Image Generation: Beyond Text-Image Alignment
description: >-
  [ICCV 2025][图像生成][奖励模型] 本文揭示了基于 CLIP/BLIP 的奖励模型在评估高质量图像时的「评分悖论」——细节丰富的高质量图像反而得低分，并提出 ICT Score（Image-Contained-Text…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "奖励模型"
  - "ICT Score"
  - "HP Score"
  - "文图对齐"
  - "人类偏好"
  - "扩散模型优化"
---

# Enhancing Reward Models for High-quality Image Generation: Beyond Text-Image Alignment

**会议**: ICCV 2025  
**arXiv**: [2507.19002](https://arxiv.org/abs/2507.19002)  
**代码**: [GitHub](https://github.com/BarretBa/ICTHP)  
**领域**: 图像生成评估/奖励模型  
**关键词**: 奖励模型, ICT Score, HP Score, 文图对齐, 人类偏好, 扩散模型优化

## 一句话总结
本文揭示了基于 CLIP/BLIP 的奖励模型在评估高质量图像时的「评分悖论」——细节丰富的高质量图像反而得低分，并提出 ICT Score（Image-Contained-Text，评估图像包含文本信息的程度）和 HP Score（纯图像模态的人类偏好评分）两个新指标，在 Pick-High 数据集上训练后，偏好预测准确率提升超过 10%，并成功优化 SD3.5-Turbo 生成更高质量的图像。

## 研究背景与动机

### 核心问题

现代扩散模型（SD3.5、FLUX）已能生成高保真、高美学质量的图像，远超基本的文图对齐需求。然而，现有评估框架（CLIP Score、PickScore、ImageReward）未能与之同步进化。

### 奖励模型的评分悖论

基于 CLIP/BLIP 微调的人类偏好奖励模型存在根本缺陷：**对细节丰富、美学价值高的图像反而打低分**，与真实人类偏好严重偏离。

**信息论解释**：根据信息分解原理，图像信息量 $I(v) = I(v;t) + I(v|t)$，其中 $I(v;t)$ 是文图互信息（对齐分量），$I(v|t)$ 是图像特有信息（美学、纹理、氛围等）。

CLIP 评分机制基于余弦相似度：

$$\text{CLIP}(v,t) \approx \frac{I(v;t)}{\sqrt{I(t) \cdot (I(v;t) + I(v|t))}}$$

当高质量模型生成细节丰富的图像时，虽然 $I(v;t)$ 增加，但 $I(v|t)$ 增长更快，导致分母增幅大于分子，**整体分数反而下降**。

### 实际后果

这意味着用 CLIP Score / PickScore / ImageReward 作为奖励函数优化 SD3.5 等先进模型时，反而会引导模型生成视觉稀疏、美学匮乏的图像——这正是目前领域的核心困境。

## 方法详解

### 整体框架

本文提出三个核心贡献：
1. **Pick-High 数据集**：大规模高质量图像偏好数据集
2. **ICT Score**：超越文图对齐的新评估目标
3. **HP Score**：纯图像模态的人类偏好评分

### 关键设计 1：Pick-High 数据集

- 从 PickAPic_v2 中精选 360K 文本 prompt
- 利用 LLM 的 Chain-of-Thought 能力精心设计 **refined prompt**（更符合人类审美偏好）
- 用 SOTA 图像生成模型基于 refined prompt 生成 360K 高质量图像
- 构建**三元组偏好排序**：$I_1$（不偏好）< $I_2$（偏好）< $I_3$（高质量精修 prompt 生成）

### 关键设计 2：ICT Score（Image-Contained-Text）

核心思想：**评估图像包含文本信息的程度**，而非双向对齐。这避免了惩罚包含超出文本描述的丰富细节的高质量图像。

**阈值化机制**（避免 CLIP 对高质量图像的偏见）：

$$\mathcal{C}(I, P) = \min\left(\frac{\text{CLIP}(I, P)}{\theta}, 1\right)$$

**基础 prompt ICT 分数**：
- $E_3 = 1$（高质量图像满分）
- $E_2 = \mathcal{C}(I_2, P_{\text{easy}})$
- $E_1 = \min(\mathcal{C}(I_1, P_{\text{easy}}), E_2)$（保证排序一致性）

**精修 prompt ICT 分数**：结合文本间相似度
- $R_3 = 1$
- $R_2 = E_2 \times \text{CLIP}(P_{\text{easy}}, P_{\text{ref}})$
- $R_1 = E_1 \times \text{CLIP}(P_{\text{easy}}, P_{\text{ref}})$

**ICT 模型训练**：在 CLIP 基础上微调，使用 MSE 损失对齐预测分数与 ICT 标签：

$$\mathcal{L}_{\text{ICT}} = \sum_{i=1}^3 (E_i - y_{i,e})^2 + \sum_{i=1}^3 (R_i - y_{i,r})^2$$

**负样本挖掘**：引入 sigmoid 加权策略处理潜在假负样本：

$$w(y) = \frac{1}{1 + e^{\alpha(|y| - \beta)}}$$

总损失：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ICT}} + \lambda \mathcal{L}_{\text{neg}}$

### 关键设计 3：HP Score（High-Preference）

当 ICT 分数达到上限（图像完全表达了文本语义）后，需要从**纯图像模态**进一步评估质量。

对三元组 $\{I_1, I_2, I_3\}$ 使用 margin ranking loss 微调 CLIP image encoder + MLP：

$$\mathcal{L}_{\text{margin}} = \sum\left[\max(0, -\Delta(I_2, I_1) + m) + \max(0, -\Delta(I_3, I_2) + m)\right]$$

**组合使用**：ICT × HP = ICT-HP Score，综合评估文本表达和美学质量。

### 扩散模型优化

使用 DRaFT-K 方法对 SD3.5-Large-Turbo 进行微调，直接最大化可微的 ICT/HP/ICT-HP 奖励函数。

## 实验

### 偏好预测准确率

在 Pick-High + PickAPic_v2 测试集上：

| 模型 | 平均准确率 ↑ | $I_2 > I_1$ ↑ | $I_3 > I_2$ ↑ | $I_3 > I_1$ ↑ |
|------|-------------|---------------|---------------|---------------|
| Random | 50.00 | 50.00 | 50.00 | 50.00 |
| CLIP | 60.30 | 64.29 | 52.80 | 63.79 |
| ImageReward | 63.81 | 64.58 | 58.02 | 68.84 |
| PickScore | 79.04 | 74.80 | 75.37 | 86.94 |
| **ICT** | **87.58** | **64.65** | **100** | **100** |
| **HP** | **88.47** | **64.97** | **100** | **100** |
| **ICT-HP** | **88.84** | **66.42** | **100** | **100** |

关键发现：
- ICT-HP 相比 PickScore 平均准确率提升约 **10%**（88.84 vs 79.04）
- 在高质量图像对比（$I_3 > I_2$）上，ICT/HP/ICT-HP 达到 **100%** 准确率（PickScore 仅 75.37%）
- CLIP/BLIP 的基础多模态模型仅略好于随机，说明文图对齐目标本身不适合评估生成质量

### 扩散模型优化效果

GenEval 基准上的定量评估：

| 模型 | Mean ↑ | Single ↑ | Counting ↑ | Colors ↑ | Position ↑ |
|------|--------|----------|------------|----------|------------|
| SD3.5-Turbo | 0.69 | 0.99 | 0.69 | 0.80 | 0.25 |
| + PickScore | 0.66 ↓ | 0.99 | 0.67 | 0.74 ↓ | 0.24 |
| + ImageReward | 0.70 | 0.99 | 0.68 | 0.80 | 0.28 |
| **+ ICT** | **0.71** | 0.98 | **0.70** | 0.81 | **0.31** |
| **+ ICT-HP** | **0.70** | 0.99 | 0.68 | 0.79 | 0.28 |
| + CLIP (crash) | 0.13 | 0.38 | 0.06 | 0.26 | 0.01 |

关键发现：
- PickScore 优化后 Colors 等指标反而下降，验证了现有奖励模型的缺陷
- CLIP 直接用作奖励函数导致训练崩溃（Mean 0.13），完全不可用
- ICT 在 Mean、Counting、Position 上均实现最优，说明 ICT 目标有效避免了对高质量图像的惩罚

### JPEG 压缩率与美学评分

| 模型 | JPEG 压缩率 ↑ | 美学分数 ↑ |
|------|--------------|-----------|
| SD3.5-Large | 374.80 | 6.307 |
| FLUX.1-dev | 270.58 | 6.436 |
| SD3.5-Turbo | 313.10 | 6.293 |
| + HP (Ours) | **334.86** | **6.448** |
| + ICT-HP (Ours) | 330.23 | 6.300 |

HP 优化后的 SD3.5-Turbo 在美学分数上甚至超越了更大的 SD3.5-Large 和 FLUX.1-dev。

## 亮点与洞察

1. **问题定义精准**：从信息论角度严格论证了 CLIP 对齐范式在评估高质量图像时的根本缺陷，不是经验观察而是理论推导
2. **ICT Score 的巧妙设计**：将「双向对齐」转为「图像包含文本」的单向评估，消除了对超出文本描述的丰富视觉信息的惩罚
3. **双阶段评估体系**：ICT 确保文本信息被充分表达 → HP 在此基础上进一步评估美学质量，二者互补
4. **实际影响力**：现有 RLHF 方式优化扩散模型时使用的奖励函数可能实际损害高端模型质量，本文提供了修正方案
5. **ICT 文本编码器可迁移**：直接迁移到 SD2.1 显著提升图像质量

## 局限性

1. Pick-High 数据集依赖 LLM 生成 refined prompt，可能引入偏见
2. ICT Score 的阈值参数 $\theta$ 需要调节，对不同模型可能需要不同设置
3. 实验主要在 SD3.5-Turbo 上验证，对其他架构（如 DiT、FLUX）的泛化性有待验证
4. HP Score 仅基于图像模态，可能无法区分「高质量但偏题」的图像

## 相关工作

- **奖励模型**：CLIP Score → ImageReward（人类偏好微调）→ PickScore（大规模偏好数据）→ HPSv2 → 本文（超越对齐的 ICT+HP）
- **偏好数据集**：PickAPic_v2、ImageRewardDB、HPDv2 → 本文 Pick-High（高质量三元组排序）
- **扩散模型优化**：DRaFT-K、ReFL 等基于奖励的微调方法

## 评分

- 创新性: ⭐⭐⭐⭐⭐ — 从信息论角度揭示评分悖论并提出超越对齐的新目标，洞察深刻
- 技术深度: ⭐⭐⭐⭐ — ICT/HP Score 设计巧妙但整体方法不算复杂
- 实验充分度: ⭐⭐⭐⭐⭐ — 偏好预测+GenEval+美学+人工评估+迁移实验全覆盖
- 实用价值: ⭐⭐⭐⭐⭐ — 直接解决 RLHF 优化高端扩散模型时的核心痛点

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Cycle Consistency as Reward: Learning Image-Text Alignment without Human Preferences](cycle_consistency_as_reward_learning_imagetext_alignment_wit.md)
- [\[ICCV 2025\] CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models](compass_enhancing_spatial_understanding_in_text-to-image_diffusion_models.md)
- [\[ICCV 2025\] Semantic Watermarking Reinvented: Enhancing Robustness and Generation Quality with Fourier Integrity](semantic_watermarking_reinvented_enhancing_robustness_and_generation_quality_wit.md)
- [\[CVPR 2026\] Enhancing Spatial Understanding in Image Generation via Reward Modeling](../../CVPR2026/image_generation/enhancing_spatial_understanding_in_image_generation_via_reward_modeling.md)
- [\[NeurIPS 2025\] UltraHR-100K: Enhancing UHR Image Synthesis with A Large-Scale High-Quality Dataset](../../NeurIPS2025/image_generation/ultrahr-100k_enhancing_uhr_image_synthesis_with_a_large-scale_high-quality_datas.md)

</div>

<!-- RELATED:END -->
