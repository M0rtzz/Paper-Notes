---
title: >-
  [论文解读] Weight Decay may matter more than μP for Learning Rate Transfer in Practice
description: >-
  [ICLR 2026][LLM/NLP][μP] 本文通过大规模实证分析表明，μP 的核心对齐假设仅在训练初期短暂成立，实际训练中是独立权重衰减（independent weight decay）而非 μP 在正确稳定跨宽度的特征学习动态，μP 的实际益处可被解释为一种隐式学习率预热。
tags:
  - "ICLR 2026"
  - "LLM/NLP"
  - "μP"
  - "学习率迁移"
  - "权重衰减"
  - "AdamW"
  - "特征学习"
---

# Weight Decay may matter more than μP for Learning Rate Transfer in Practice

**会议**: ICLR 2026  
**arXiv**: [2510.19093](https://arxiv.org/abs/2510.19093)  
**代码**: 无  
**领域**: LLM 训练优化  
**关键词**: [μP, 学习率迁移, 权重衰减, AdamW, 特征学习]

## 一句话总结

本文通过大规模实证分析表明，μP 的核心对齐假设仅在训练初期短暂成立，实际训练中是独立权重衰减（independent weight decay）而非 μP 在正确稳定跨宽度的特征学习动态，μP 的实际益处可被解释为一种隐式学习率预热。

## 研究背景与动机

Maximal Update Parameterization（μP）是大规模 LLM 训练中实现学习率迁移的核心技术，被众多开源大模型（Falcon、Cohere 等）和商业模型采用。μP 的核心思想是通过学习率缩放保持不同宽度网络中内部表征更新大小的一致性，从而将最优学习率从小模型迁移到大模型，避免昂贵的超参搜索。

然而，多项实证研究发现 μP 只有与独立权重衰减（independent WD）结合时才能实现良好的学习率迁移，而标准权重衰减则效果不佳。这一现象背后的原因一直未被深入理解。本文提出的核心问题是：μP 的对齐假设在实践中是否真的成立？学习率迁移的真正驱动力是什么？

## 方法详解

### 整体框架

作者建立了一个基于**相对更新**（relative updates）的统一框架，将 μP 和权重衰减纳入同一分析视角。核心公式将表征变化率分解为对齐比（alignment ratio）和相对权重更新的乘积：

$$\frac{\|\Delta \mathbf{Y}\|}{\|\mathbf{Y}\|} = \frac{\alpha_{\Delta W}}{\alpha_W} \cdot \frac{\|\Delta \mathbf{W}\|}{\|\mathbf{W}\|}$$

其中 $\alpha_{\Delta W}$ 是更新对齐度，$\alpha_W$ 是权重对齐度。μP 假设对齐比 $\alpha_{\Delta W}/\alpha_W = \Theta(\sqrt{C})$ 随宽度 $C$ 增长，因此通过缩放学习率 $\eta \propto 1/m$ 来抵消对齐比的变化。

### 关键设计

1. **独立权重衰减覆盖 μP 缩放机制**：做什么——分析标准 vs 独立权重衰减在 μP 缩放下的行为差异。核心思路——在 AdamW 平衡态下，权重范数 $\|\mathbf{W}\| \propto \sqrt{KC \cdot \eta/\lambda}$，相对更新 $\|\Delta \mathbf{W}\|/\|\mathbf{W}\| \propto \sqrt{\eta\lambda}$。独立缩放 $(\eta, \lambda) \mapsto (\eta/m, m\lambda)$ 保持乘积 $\eta\lambda$ 不变，因此平衡态的相对更新不随宽度变化，从而**抵消并覆盖 μP 的缩放**。设计动机——μP 的对齐假设在训练中迅速失效，对齐比趋于 1 而非随宽度增长，因此需要相同大小的相对更新来维持跨宽度的表征变化一致性。

2. **μP 对齐假设的失效机制分析**：做什么——理论与实验分析 μP 的更新对齐和权重对齐假设为何在实践中不成立。核心思路——当批大小 $B$ 远大于输入维度 $C$ 时（LLM 训练中 token 总数 1M >> 宽度 6K），更新梯度中来自其他样本的干扰项主导输出变化，使更新对齐度变为宽度相关的 $\alpha_{\Delta W} \sim \Theta(1/\sqrt{C})$，导致对齐比 $\alpha_{\Delta W}/\alpha_W \approx 1$。设计动机——μP 源自无穷宽极限下的分析，其 IID 假设在有限宽度的实际训练中仅在初始化附近短暂成立。

3. **μP 等效学习率预热效果**：做什么——将 μP + 独立权重衰减的实际效果重新解读为隐式学习率预热。核心思路——高权重衰减配置 $(\eta/m, m\lambda)$ 使训练初期的相对更新比 $(\eta, \lambda)$ 小 $1/m$ 倍，随训练进行逐渐趋近 1，形如指数预热 $s_t = (1 + (m^2-1) \cdot a^{2t})^{-1/2}$，其中 $a = 1 - \eta\lambda$。设计动机——这解释了为什么更强的显式预热调度可以替代 μP 的学习率缩放。

### 损失函数 / 训练策略

实验基于 LLaMA 架构在 DCLM 数据集上进行下一个 token 预测训练，使用 AdamW 优化器。采用 10% 线性预热 + 线性衰减的标准调度。验证实验覆盖不同宽度比 $m \in \{1, 2, 4, 8, 16\}$。

## 实验关键数据

### 主实验（表格）

| 配置 | 学习率迁移质量 | 说明 |
|:---|:---|:---|
| μP + 标准 WD | ❌ 差（长训练后偏差大） | 标准缩放在后期无法维持 RRC 一致性 |
| μP + 独立 WD | ✅ 好 | 独立 WD 覆盖 μP 缩放，稳定 RRC |
| 无 μP + 10% 线性预热 | ❌ 差 | 线性预热不足以替代 |
| 无 μP + 50% 线性预热 | ❌ 中等 | 仍不如独立 WD |
| 无 μP + 指数预热 | ✅ 近似好 | 附加宽度缩放因子的指数预热接近 μP+独立 WD |

### 消融实验（表格）

| 实验维度 | 观察 |
|:---|:---|
| 对齐比随训练变化 | 初始 $\alpha_{\Delta W}/\alpha_W \sim \sqrt{C}$，快速衰减至 ≈1 |
| 无 WD 情况 | 相对更新也趋于宽度无关（权重范数持续增长） |
| ResNet 验证 | 结论大体一致，但额外预热不如 LLM 关键 |
| 矩阵级优化器（Muon/Scion） | 可绕过对齐问题，自然实现低且稳定的更新对齐 |

### 关键发现

- μP 的核心对齐假设（$\alpha_{\Delta W} = \Theta(1)$）仅在训练初期几步内成立，随后迅速失效
- 独立权重衰减在训练主体阶段替代 μP 发挥稳定特征学习的作用
- μP 的实际效果等价于一种隐式学习率预热，可通过显式指数预热部分替代
- 对于 LLM 训练，批大小远大于模型宽度是导致 μP 假设失效的关键因素

## 亮点与洞察

- 统一了 μP 和权重衰减的分析框架，用"相对更新"视角揭示两者的本质联系
- 重新解读 μP 为"隐式预热"打破了社区对 μP 理论基础的固有认知
- 指出矩阵级优化器（如 Muon）可能从根本上绕过对齐问题，解释了其对预热需求低的特性
- 为实践者提供明确指导：使用 μP 时必须搭配独立权重衰减

## 局限与展望

- 仅针对 AdamW 优化器进行深入分析，SGD 和其他优化器的结论需进一步验证
- 实验规模虽涵盖多个宽度比，但未达到真正的超大规模（>10B 参数）验证
- 简化分析模型（如权重衰减平衡态公式）未能精确预测实际训练中预热形状的时间尺度
- 未探索不同初始化方案对对齐假设失效速度的影响

## 相关工作与启发

- Everett et al. (2024) 的逐层学习率缩放方法表明 μP 的特殊处理（输出层、注意力归一化）对迁移不是必需的
- Wang & Aitchison (2025) 从 EMA 角度分析了独立 WD 的必要性，本文从特征变化率角度提供了更直接的解释
- Kosson et al. (2024b) 的权重衰减框架为本文的相对更新分析奠定了基础
- 矩阵级优化器（Muon, Scion）可能代表了超越 μP 的更优特征学习控制方式

## 评分

⭐⭐⭐⭐ — 深刻挑战了 μP 的理论基础，实验扎实且对实践有直接指导意义，但仅限 AdamW 且缺乏超大规模验证稍显不足。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Learning Spatial Decay for Vision Transformers](../../AAAI2026/llm_nlp/learning_spatial_decay_for_vision_transformers.md)
- [\[ICLR 2026\] Fine-Grained Activation Steering: Steering Less, Achieving More](fine-grained_activation_steering_steering_less_achieving_more.md)
- [\[ICLR 2026\] First is Not Really Better Than Last: Evaluating Layer Choice and Aggregation Strategies in Language Model Data Influence Estimation](first_is_not_really_better_than_last_evaluating_layer_choice_and_aggregation_str.md)
- [\[ACL 2025\] Multilingual Encoder Knows more than You Realize: Shared Weights Pretraining for Extremely Low-Resource Languages](../../ACL2025/llm_nlp/multilingual_encoder_knows_more_than_you_realize_shared_weights_pretraining_for_.md)
- [\[ACL 2025\] Language Models, Graph Searching, and Supervision Adulteration: When More Supervision is Less and How to Make More More](../../ACL2025/llm_nlp/lm_graph_search_supervision.md)

</div>

<!-- RELATED:END -->
