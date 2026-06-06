---
title: >-
  [论文解读] DEFT: Decompositional Efficient Fine-Tuning for Text-to-Image Models
description: >-
  [NeurIPS 2025][图像生成][高效微调] 提出DEFT（Decompositional Efficient Fine-Tuning），通过将权重更新分解为子空间投影和低秩调整两个组件来高效微调T2I模型，在个性化生成和通用图像生成任务上超越LoRA和PaRa。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "高效微调"
  - "扩散模型"
  - "低秩分解"
  - "个性化生成"
  - "通用图像生成"
---

# DEFT: Decompositional Efficient Fine-Tuning for Text-to-Image Models

**会议**: NeurIPS 2025  
**arXiv**: [2509.22793](https://arxiv.org/abs/2509.22793)  
**代码**: [DEFT](https://github.com/DEFT)  
**领域**: 图像生成  
**关键词**: 高效微调, 扩散模型, 低秩分解, 个性化生成, 通用图像生成

## 一句话总结

提出DEFT（Decompositional Efficient Fine-Tuning），通过将权重更新分解为子空间投影和低秩调整两个组件来高效微调T2I模型，在个性化生成和通用图像生成任务上超越LoRA和PaRa。

## 研究背景与动机

1. **领域现状**: 文本到图像（T2I）模型微调面临计算资源和过拟合的挑战。LoRA通过低秩更新实现参数高效微调，PaRa通过正交子空间投影减少秩。

2. **现有痛点**: LoRA的低秩更新缺乏约束，容易过拟合且对姿态、空间位置控制有限。PaRa仅减少预训练权重的秩但不添加新方向。多概念组合生成中概念间的干扰和混合仍是难题。

3. **核心矛盾**: 高效微调需要在三方面取得平衡——学习目标分布、保持指令遵循能力、维持编辑性（多样化提示或上下文生成），但现有方法难以兼顾。

4. **本文目标**: 设计更灵活的权重更新方式，既能高效适配新概念/任务，又能保持预训练模型的泛化能力。

5. **切入角度**: 将权重更新分解为两个互补组件——子空间投影（去除部分方向）和低秩调整（注入新方向），通过两个可训练矩阵联合实现。

6. **核心 idea**: $W_{total} = (I - PP^T)W_0 + PR$，其中 $P$ 定义子空间投影，$R$ 实现该子空间内的灵活调整，扩展了权重的列空间。

## 方法详解

### 整体框架

DEFT在预训练权重 $W_0$ 上应用分解式更新：先通过 $PP^T$ 投影去除部分子空间，再通过 $PR$ 注入新方向。适用于Stable Diffusion的UNet层和统一模型（如OmniGen）的Transformer线性层。

### 关键设计

**1. 分解式权重更新**

- **功能**: 在去除旧方向的同时注入新的任务特定方向，平衡适配和保留
- **核心思路**: $W_{total} = (I - PP^T)W_0 + PR$。$(I-PP^T)W_0$ 去除 $P$ 列空间方向的分量，$PR$ 在该子空间中注入新信息。总列空间为 $\text{col}(W_{reduce}) + \text{col}(QR) \subseteq \text{col}(W_0) + \text{col}(Q)$
- **设计动机**: 低秩更新在正交于 $W_0$ 主奇异向量的方向上最有效。PaRa仅减秩不加新方向，LoRA加方向但不去除旧方向，DEFT两者兼顾

**2. 多种分解策略**

- **功能**: 提供不同结构偏置的即插即用选择
- **核心思路**: 支持QR分解、截断SVD、低秩矩阵分解(LRMF)、非负矩阵分解(NMF)、特征分解等。NMF的非负约束产生更稀疏和结构化的更新
- **设计动机**: 不同数据体制和下游任务可能受益于不同的结构偏置

**3. 学习率差异化设计**

- **功能**: 稳定优化，平衡投影矩阵和调整矩阵的更新速率
- **核心思路**: 对 $R$ 使用较高学习率，对 $P$ 使用较低学习率，类似LoRA中 $A$/$B$ 的学习率设计
- **设计动机**: $P$ 定义子空间结构应缓慢稳定变化，$R$ 负责具体内容适配应快速调整

### 损失函数 / 训练策略

使用标准扩散去噪损失。DreamBench Plus使用rank=4，VisualCloze通用生成使用rank=32。Dreambooth风格的个性化训练。

## 实验关键数据

### 主实验

DreamBench Plus上的图文对齐评分（CLIP-T，150个主题×8个提示）：

| 方法 | T2I模型 | CLIP-T |
|------|---------|--------|
| Textual Inversion | SD v1.5 | 0.302 |
| DreamBooth | SD v1.5 | 0.323 |
| DreamBooth LoRA | SDXL v1.0 | 0.341 |
| PaRa | SDXL v1.0 | 0.354 |
| **DEFT (本文)** | SDXL v1.0 | **0.361** |

VisualCloze上的通用图像生成性能：

| 条件 | 方法 | CLIP-Score | DINO-v1 | DINO-v2 |
|------|------|-----------|---------|---------|
| Canny | OmniGen | 95.45 | 87.13 | 87.60 |
| Canny | **DEFT** | **95.78** | **90.37** | **90.65** |
| Depth | OmniGen | 92.02 | 85.16 | 77.39 |
| Depth | **DEFT** | **93.18** | **88.98** | **85.75** |

### 消融实验

| 分解方法 | CLIP-I | CLIP-T | 说明 |
|---------|--------|--------|------|
| 无分解(默认) | 基线 | 基线 | 最简单，作为默认 |
| QR | 竞争力 | 竞争力 | 正交约束 |
| NMF | 高 | 最高 | 非负约束提供更好的提示控制 |
| TSVD | 竞争力 | 竞争力 | SVD基础分解 |
| Relaxing P | 高 | 高 | 可学习投影矩阵 |

### 关键发现

- DEFT在指令遵循（CLIP-T）上超越LoRA 2个百分点，源于低秩注入扩展了微调子空间
- 风格迁移中DEFT的Image Score达到0.69，远超InstantStyle(0.60)和OmniGen(0.52)
- 多概念组合中DEFT不需要为每个概念单独的LoRA，支持联合微调
- 可控性指标（Controllability F1）与OmniGen持平，质量指标（SSIM）显著提升

## 亮点与洞察

- **理论优雅**: "去除+注入"的分解思路清晰，有线性代数理论支撑（列空间扩展证明）
- **灵活性高**: 支持多种分解方法，即插即用
- **涌现特性**: 在少量图像上微调后模型展现出未在训练集中看到的泛化组合能力
- **统一框架**: 同一方法适用于个性化、风格迁移和条件生成等多种任务

## 局限与展望

- 默认未使用分解（为简洁），最佳分解策略的选择缺乏自动化指导
- 仅在SDXL和OmniGen上验证，更大模型（如FLUX）的效果待验证
- 多概念组合的概念数量上限未明确测试
- 可探索与SVDiff等方法在理论框架上的更深入对比

## 相关工作与启发

- LoRA添加方向但不去除，PaRa去除但不添加，DEFT统一了两者
- Custom Diffusion微调子集参数，DEFT提供更灵活的参数适配空间
- 启发: 高效微调不仅关乎参数数量，更关乎更新的结构（子空间选择）

## 评分

- 新颖性: ⭐⭐⭐⭐ 分解式更新的思路有理论支撑且区别于现有方法
- 实验充分度: ⭐⭐⭐⭐ 跨多个数据集和任务类型验证
- 写作质量: ⭐⭐⭐ 内容丰富但组织可更紧凑
- 价值: ⭐⭐⭐⭐ 为高效微调提供了新的设计空间

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Towards Resilient Safety-Driven Unlearning for Diffusion Models Against Downstream Fine-tuning](towards_resilient_safety-driven_unlearning_for_diffusion_models_against_downstre.md)
- [\[ICML 2025\] Zero-Shot Adaptation of Parameter-Efficient Fine-Tuning in Diffusion Models](../../ICML2025/image_generation/zero-shot_adaptation_of_parameter-efficient_fine-tuning_in_diffusion_models.md)
- [\[CVPR 2025\] Focus-N-Fix: Region-Aware Fine-Tuning for Text-to-Image Generation](../../CVPR2025/image_generation/focus-n-fix_region-aware_fine-tuning_for_text-to-image_generation.md)
- [\[CVPR 2025\] Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models](../../CVPR2025/image_generation/efficient_fine-tuning_and_concept_suppression_for_pruned_diffusion_models.md)
- [\[NeurIPS 2025\] GraLoRA: Granular Low-Rank Adaptation for Parameter-Efficient Fine-Tuning](gralora_granular_low-rank_adaptation_for_parameter-efficient_fine-tuning.md)

</div>

<!-- RELATED:END -->
