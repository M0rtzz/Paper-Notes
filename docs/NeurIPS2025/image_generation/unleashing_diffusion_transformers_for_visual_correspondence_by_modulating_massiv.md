---
title: >-
  [论文解读] Unleashing Diffusion Transformers for Visual Correspondence by Modulating Massive Activations
description: >-
  [NeurIPS 2025][图像生成][Transformer] 发现 Diffusion Transformers (DiTs) 中存在 massive activations 现象导致特征不可区分，揭示其与 AdaLN 的内在联系，提出无需训练的 DiTF 框架来提取语义判别性特征，在视觉对应任务上超越 DINO 和 SD 模型。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "Transformer"
  - "Massive Activations"
  - "AdaLN"
  - "视觉对应"
  - "特征提取"
---

# Unleashing Diffusion Transformers for Visual Correspondence by Modulating Massive Activations

**会议**: NeurIPS 2025  
**arXiv**: [2505.18584](https://arxiv.org/abs/2505.18584)  
**代码**: [GitHub](https://github.com/ganchaofan0000/DiTF)  
**领域**: 视觉对应, 扩散模型, 特征提取  
**关键词**: 扩散 Transformer, Massive Activations, AdaLN, 视觉对应, 特征提取

## 一句话总结
发现 Diffusion Transformers (DiTs) 中存在 massive activations 现象导致特征不可区分，揭示其与 AdaLN 的内在联系，提出无需训练的 DiTF 框架来提取语义判别性特征，在视觉对应任务上超越 DINO 和 SD 模型。

## 研究背景与动机

### 领域现状

**领域现状**：预训练 Stable Diffusion (SD) 模型已被证明可以作为有效的视觉对应特征提取器

### 核心矛盾

**核心矛盾**：Diffusion Transformers (DiTs) 在可扩展性和生成质量上优于 SD，但直接提取特征用于感知任务效果很差

### 现有痛点

**现有痛点**：分析发现 DiTs 存在 massive activations 现象：极少数特征维度的激活值比其他维度高 100 倍以上

### 解决思路

**解决思路**：这些 massive activations 使得所有空间 token 的特征向量方向高度相似，导致余弦相似度无法区分不同位置

## 方法详解

### 整体框架
- DiTF 是一个无需训练的框架，利用 DiTs 内置的 AdaLN 层来提取语义判别性特征
- 核心流程：原始特征提取 → AdaLN 通道调制 → 通道丢弃 → 最终特征

### 关键设计
1. **Massive Activations 分析**：

    - 空间分布：出现在所有图像 patch token 中（不同于 LLM 中集中在特殊 token）
    - 维度分布：集中在极少数固定维度（SD3-5 模型中仅维度 676）
    - 信息量低：massive activation 维度方差显著低于非 massive 维度，携带极少局部信息

2. **AdaLN 与 Massive Activations 的联系**：

    - 残差缩放因子 $\alpha_k$ 的高值维度与 massive activations 精确对应
    - AdaLN 可以自适应定位 massive activations 并通过通道调制抑制它们
    - Post-AdaLN 特征在语义一致性和空间判别性上显著优于 pre-AdaLN 特征

3. **通道丢弃策略**：

    - Post-AdaLN 特征中仍存在少量弱 massive activations
    - 将这些维度置零以进一步消除其负面影响
    - 简单有效，无需任何训练

### 损失函数 / 训练策略
- 完全无需训练（training-free）
- 直接利用 DiT 预训练模型中的 AdaLN 参数进行特征调制
- 特征提取公式：$\hat{z}_t^k = (1+\gamma_k) \text{LayerNorm}(z_t^k) + \beta_k$

## 实验关键数据

### 主实验（SPair-71k 语义对应 PCK@0.10）

| 方法 | 类型 | 平均 PCK |
|------|------|---------|
| ASIC | 无监督 | 36.9% |
| DINOv2+NN | 无监督 | 55.6% |
| DIFT | 无监督 | 57.7% |
| DiTF_sd3-5 | 无监督 | 64.6% |
| DiTF_flux | 无监督 | **67.1%** |
| SD+DINO (有监督) | 有监督 | 74.6% |

### 消融实验

| 模型 | Massive Activations 维度 | 方差与均值比 |
|------|----------------------|-------------|
| SD3-5 (1st 维度) | -44.51±0.50 | 极低方差 |
| SD3-5 (10th 维度) | -0.18±2.36 | 正常方差 |
| Flux (1st 维度) | 40.66±3.99 | 低方差 |
| Flux (10th 维度) | -0.41±3.16 | 正常方差 |

### 关键发现
- DiTF_flux 在无监督设置下在 SPair-71k 上达到 67.1% PCK，超越所有无监督方法
- 相比直接使用 DiT 原始特征（效果极差），AdaLN 调制带来了质的飞跃
- Massive activations 是 DiTs 特有现象，SD2.1 中不存在，这解释了为何 SD 可直接提取特征而 DiTs 不行
- 通道丢弃策略进一步提升约 1-2% 的性能
- 在 SPair-71k 上比 DIFT（SD 基线）高出 +9.4%

## 亮点与洞察
- 对 DiTs 中 massive activations 的系统性分析极具价值，揭示了 DiTs 与 SD 在特征提取上的本质差异
- 发现 AdaLN 同时是 massive activations 的"源头"（通过残差缩放因子）和"解药"（通过通道调制）
- 完全无需训练的特征提取方法，实用性强
- 研究思路可推广到分析其他 Transformer 变体中的激活值分布异常

## 局限与展望
- 仅测试了 SD3-5 和 Flux 两种 DiT 模型，更广泛的 DiT 变体有待验证
- 通道丢弃策略需要预先确定 massive activation 维度，自适应方案可进一步探索
- 目前仅在视觉对应任务上验证，语义分割、深度估计等任务有待测试
- 未与同时使用 DiT+DINO 的融合方案进行比较
- 对不同时间步和不同 DiT 层的特征选择策略可进一步优化
- 可探索将 DiTF 与 DINOv2 特征融合以获得更强性能

## 相关工作与启发
- LLM 中的 massive activations 研究（Sun et al.）为理解 DiTs 中的类似现象提供了基础
- ViT 中的 attention artifacts（注册 token 方案）与 DiTs 中的 massive activations 本质不同
- 利用模型内置归一化层进行特征调制的思路简洁而有效
- 跨领域的 massive activations 对比分析（LLM vs ViT vs DiT）具有学术价值
- 量化加速研究也发现 DiT 中的 outlier 导致不稳定，与本文发现互相印证
- 为后续 DiT 模型的感知任务应用奠定了特征提取基础

## 评分
- 新颖性：⭐⭐⭐⭐⭐ （问题发现和分析具有开创性）
- 技术贡献：⭐⭐⭐⭐ （无训练方法实用性强）
- 实验充分度：⭐⭐⭐⭐ （多基准比较，消融分析详细）
- 写作质量：⭐⭐⭐⭐⭐ （逻辑清晰，可视化出色）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Mind-the-Glitch: Visual Correspondence for Detecting Inconsistencies in Subject-Driven Generation](mind-the-glitch_visual_correspondence_for_detecting_inconsistencies_in_subject-d.md)
- [\[NeurIPS 2025\] Scaling Diffusion Transformers Efficiently via μP](scaling_diffusion_transformers_efficiently_via_μp.md)
- [\[NeurIPS 2025\] OmniSync: Towards Universal Lip Synchronization via Diffusion Transformers](omnisync_towards_universal_lip_synchronization_via_diffusion.md)
- [\[NeurIPS 2025\] Linear Differential Vision Transformer: Learning Visual Contrasts via Pairwise Differentials](linear_differential_vision_transformer_learning_visual_contrasts_via_pairwise_di.md)
- [\[NeurIPS 2025\] Seg4Diff: Unveiling Open-Vocabulary Segmentation in Text-to-Image Diffusion Transformers](seg4diff_unveiling_open-vocabulary_segmentation_in_text-to-image_diffusion_trans.md)

</div>

<!-- RELATED:END -->
