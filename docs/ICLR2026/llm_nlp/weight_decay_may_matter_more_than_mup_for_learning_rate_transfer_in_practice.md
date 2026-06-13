---
title: >-
  [论文解读] Weight Decay may matter more than μP for Learning Rate Transfer in Practice
description: >-
  [ICLR 2026][LLM 其他][μP] 本文通过大规模实证分析表明，μP 的核心对齐假设仅在训练初期短暂成立，实际训练中是独立权重衰减（independent weight decay）而非 μP 在正确稳定跨宽度的特征学习动态，μP 的实际益处可被解释为一种隐式学习率预热。 Maximal Update Param…
tags:
  - "ICLR 2026"
  - "LLM 其他"
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

本文不提出新方法，而是建立一个基于**相对更新**（relative updates）的统一分析框架，把 μP 与权重衰减放进同一个透镜下审视。它的出发点是：跨宽度训练要稳定，本质上要让每一层的**相对表征变化**（relative representation change，RRC）$\|\Delta\mathbf{Y}\|/\|\mathbf{Y}\|$ 不随宽度漂移。作者把这个变化率拆成对齐比与相对权重更新的乘积，

$$\frac{\|\Delta \mathbf{Y}\|}{\|\mathbf{Y}\|} = \frac{\alpha_{\Delta W}}{\alpha_W} \cdot \frac{\|\Delta \mathbf{W}\|}{\|\mathbf{W}\|}$$

其中 $\alpha_{\Delta W}=\|\Delta\mathbf{Y}\|/(\|\Delta\mathbf{W}\|\,\|\mathbf{X}\|)$ 是**更新对齐度**（梯度更新方向和输入有多对齐，对齐越高、同样大小的权重更新带来的表征变化越大）、$\alpha_W=\|\mathbf{Y}\|/(\|\mathbf{W}\|\,\|\mathbf{X}\|)$ 是**权重对齐度**（权重范数和表征大小的换算系数）。μP 的整套缩放正建立在"对齐比 $\alpha_{\Delta W}/\alpha_W = \Theta(\sqrt{C})$ 随宽度 $C$ 增长"这一假设之上，于是用学习率缩放 $\eta \propto 1/m$ 去抵消它、把 RRC 钉住在与宽度无关的水平上。

本文的论证就沿这条分解公式展开成三步、对应下面三个关键设计：训练主体阶段真正把 RRC 钉住的其实是独立权重衰减、而非 μP（设计 1）；μP 赖以成立的对齐假设在实际训练里很快就垮（设计 2）；而 μP 早期残留的那点缩放，本质只是一段隐式的学习率预热（设计 3）。

> 这是一篇纯机制分析论文，论证围绕范数与对齐比的标度关系展开、没有可画成 pipeline 的多阶段数据流，故不配框架图；三个关键设计已按论文论证顺序排列，与上面整体框架的三步逻辑一一对应。

### 关键设计

**1. 独立权重衰减覆盖 μP 缩放：谁在真正稳定相对更新**

μP 通过缩放学习率来调节表征变化，但作者指出，在 AdamW 进入平衡态后，真正决定相对更新大小的是学习率与权重衰减的乘积，而非学习率本身。平衡态下权重范数满足 $\|\mathbf{W}\| \propto \sqrt{KC \cdot \eta/\lambda}$，于是相对更新 $\|\Delta \mathbf{W}\|/\|\mathbf{W}\| \propto \sqrt{\eta\lambda}$。这意味着 μP 常用的独立缩放 $(\eta, \lambda) \mapsto (\eta/m, m\lambda)$ 恰好保持乘积 $\eta\lambda$ 不变——它把 μP 对学习率的那一刀又补了回来，让平衡态的相对更新与宽度无关。换句话说，是独立权重衰减、而不是 μP 本身，在训练主体阶段维持着跨宽度一致的表征变化；μP 的缩放在平衡态里被悄悄覆盖掉了。这也直接解释了为何 μP 只有搭配独立 WD 才迁移得好。

**2. μP 对齐假设的失效机制：批大小远大于宽度时假设就垮了**

为什么 μP 的缩放会变得多余？因为它赖以成立的对齐假设在实际训练里只是昙花一现。μP 源自无穷宽极限下的分析，默认更新对齐度 $\alpha_{\Delta W}$ 与宽度无关。但 LLM 训练中批内 token 总数 $B$ 远大于模型宽度 $C$（例如 1M token 对 6K 宽度），梯度更新里来自其他样本的干扰项会主导单个输出的变化，使更新对齐度退化为宽度相关的 $\alpha_{\Delta W} \sim \Theta(1/\sqrt{C})$。代入分解式后对齐比 $\alpha_{\Delta W}/\alpha_W \approx 1$，而非 μP 假设的 $\sqrt{C}$。也就是说，μP 的 IID 假设只在初始化附近短暂成立，训练几步后对齐比就塌到 1，缩放因此失去了它要抵消的对象。

**3. μP 等效于隐式学习率预热：把缩放重新解读成调度**

既然平衡态由独立 WD 主导，那 μP 在训练早期到底带来了什么？作者发现高权重衰减配置 $(\eta/m, m\lambda)$ 在训练初期的相对更新比基线 $(\eta, \lambda)$ 小约 $1/m$ 倍，随训练推进才逐渐回升到 1，整条曲线形如一个指数预热

$$s_t = \left(1 + (m^2-1)\,a^{2t}\right)^{-1/2}, \quad a = 1 - \eta\lambda$$

这把 μP 的学习率缩放重新诠释为一种隐式的预热调度，也解释了为什么足够强的显式指数预热能在很大程度上替代 μP——两者在做同一件事，只是一个藏在权重衰减的瞬态里、一个写在调度表上。

### 损失函数 / 训练策略

实验基于 LLaMA 架构在 DCLM 数据集上做下一个 token 预测训练，优化器为 AdamW，采用 10% 线性预热加线性衰减的标准调度，验证覆盖宽度比 $m \in \{1, 2, 4, 8, 16\}$。

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
- [\[ICML 2026\] Why Are Linear RNNs More Parallelizable?](../../ICML2026/llm_nlp/why_are_linear_rnns_more_parallelizable.md)
- [\[ACL 2025\] Language Models, Graph Searching, and Supervision Adulteration: When More Supervision is Less and How to Make More More](../../ACL2025/llm_nlp/lm_graph_search_supervision.md)

</div>

<!-- RELATED:END -->
