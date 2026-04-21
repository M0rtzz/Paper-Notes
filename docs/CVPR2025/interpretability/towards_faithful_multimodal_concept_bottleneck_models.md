---
title: >-
  [论文解读] Towards Faithful Multimodal Concept Bottleneck Models
description: >-
  [CVPR 2025][concept bottleneck model] 提出 f-CBM，一个基于 CLIP 的忠实多模态 Concept Bottleneck Model 框架，通过可微分的 leakage 损失和 Kolmogorov-Arnold Network 预测头联合解决概念检测准确性和信息泄漏问题，在任务精度、概念检测和 leakage 三者间达到最优权衡。
tags:
  - CVPR 2025
  - concept bottleneck model
  - interpretability
  - leakage
  - KAN
  - 多模态
---

# Towards Faithful Multimodal Concept Bottleneck Models

**会议**: CVPR 2025  
**arXiv**: [2603.13163](https://arxiv.org/abs/2603.13163)  
**代码**: 待确认  
**领域**: multimodal_vlm  
**关键词**: concept bottleneck model, interpretability, leakage, KAN, multimodal XAI

## 一句话总结

提出 f-CBM，一个基于 CLIP 的忠实多模态 Concept Bottleneck Model 框架，通过可微分的 leakage 损失和 Kolmogorov-Arnold Network 预测头联合解决概念检测准确性和信息泄漏问题，在任务精度、概念检测和 leakage 三者间达到最优权衡。

## 研究背景与动机

**领域现状**: Concept Bottleneck Models (CBMs) 通过将预测路由到人类可解释的概念层来实现可解释性，在视觉领域已有广泛研究（Label-free CBM、CT-CBM 等），但在多模态场景下几乎未被探索。

**现有痛点**: 标准 CBM 面临两大忠实性问题：(1) 概念检测不准确——CBL 可能未正确检测概念；(2) Leakage——概念表示中编码了非预期的额外信息（task leakage：概念编码了超出其语义的任务相关信号；inter-concept leakage：概念间编码了超过自然相关性的互信息）。

**核心矛盾**: 现有方法将概念检测和 leakage 缓解作为独立问题处理，改善一个往往牺牲另一个或任务精度（如 Independent-CBM 降低 leakage 但削弱任务精度；CT-CBM 用残差连接吸收 leakage 但降低可解释性）。

**本文目标**: 同时实现准确的概念检测、低 leakage 和高任务精度的多模态 CBM。

**切入角度**: 通过初步分析发现概念检测准确度与 task leakage 负相关、task leakage 与 inter-concept leakage 正相关，据此设计联合优化策略。

**核心 idea**: 用可微分的 leakage 损失显式减少泄漏 + KAN 预测头提升表达力改善概念检测，形成正反馈闭环。

## 方法详解

### 整体框架

1. CLIP 的视觉和文本编码器分别提取图像/文本嵌入，拼接为 $z = [f^v(x^v) \| f^t(x^t)] \in \mathbb{R}^{2d}$
2. Concept Bottleneck Layer (CBL) $\Phi^C: \mathbb{R}^{2d} \to \mathbb{R}^{|C|}$ 将多模态表示映射到概念激活分数
3. KAN 预测层 $\Phi^{\text{kan}}: \mathbb{R}^{|C|} \to \mathcal{Y}$ 替代传统线性层产出最终预测
4. 训练目标: $\mathcal{L} = \mathcal{L}_{\text{cls}} + \tilde{\lambda} \mathcal{L}_C + \tilde{\lambda}_{\text{leak}} \alpha \mathcal{L}_{\text{leak}}$

### 关键设计

**1. 初步分析：忠实性因素的交互关系**
- **功能**: 在 N24News 数据集上训练 baseline mCBM，分析概念检测精度与 leakage 的关系。
- **核心思路**: 发现(1) 检测精度高的概念 task leakage 显著更低（p < 1% t-test）；(2) task leakage 与 inter-concept leakage 强正相关（Pearson/Spearman 均显著）。
- **设计动机**: 据此提出假设——同时优化概念检测质量和减少 task leakage 可附带减少 inter-concept leakage，即只需显式优化两个目标即可间接改善三方面。

**2. 可微分 Leakage 损失**
- **功能**: 基于 Kernel Density Estimation (KDE) 的可微分互信息估计器，显式最小化 Concept-Task Leakage。
- **核心思路**: 
  $$\mathcal{L}_{\text{leak}} = \left[\frac{\hat{I}(\hat{c}_i; y) - \hat{I}(c_i; y)}{H(y)}\right]^2$$
  用 Gaussian kernel 估计 $\hat{I}(x; y) = N^{-1} \sum_i \log[\hat{p}(x_i|y_i) / \hat{p}(x_i)]$，bandwidth 由 Scott's rule 自动确定 $\sigma = 1.06 \cdot \text{std}(x) \cdot N^{-1/5}$。使用平方而非 clamp-at-zero，保留双向梯度。
- **设计动机**: 不同于 binning 方法，KDE 保持可微性可直接嵌入训练损失；显式减少 task leakage 会附带减少 inter-concept leakage。

**3. KAN 预测层**
- **功能**: 用 Kolmogorov-Arnold Network 层替代传统线性层，每条边使用可学习单变量函数。
- **核心思路**:
  $$\Phi_o^{\text{kan}}(x) = s_o \times \sum_{i=1}^{N} \phi_{i,o}(x), \quad \phi_{i,o}(x) = \sum_{m=1}^{M} c_{i,o,m} \cdot B_m(x)$$
  其中 $B_m$ 为 degree-1 三角形基函数，$s_o$ 为可学习缩放因子。使用单层 KAN 保持可解释性。
- **设计动机**: 线性层表达力不足可能迫使概念表示编码额外信息（leakage 的一个来源）；KAN 提供足够表达力使概念层无需"作弊"，同时每条边的可学习函数可可视化为响应曲线保持解释性。

### 损失函数 / 训练策略

$$\mathcal{L} = \mathcal{L}_{\text{cls}} + \tilde{\lambda} \mathcal{L}_C + \tilde{\lambda}_{\text{leak}} \alpha \mathcal{L}_{\text{leak}}$$

- $\mathcal{L}_{\text{cls}}$: cross-entropy 分类损失
- $\mathcal{L}_C$: MSE 概念预测损失
- $\mathcal{L}_{\text{leak}}$: KDE-based leakage 损失
- 各辅助损失通过 running mean 动态缩放到与分类损失可比的尺度
- $\alpha$ 按 cosine annealing 从 0 到 1 递增，避免 leakage 损失干扰早期概念学习
- CLIP backbone 以 $10^{-5}$ 固定学习率微调，线性层使用 cosine annealing（$10^{-1}$ 或 $10^{-2}$）

## 实验关键数据

### 主实验

N24News（CLIP-base / CLIP-large）:

| 方法 | %ACC↑ | c-RMSE↓ | CTL↓ | ICL↓ |
|---|---|---|---|---|
| Black-box | 98.5 / 98.5 | - | - | - |
| Indep.-CBM | 97.3 / 97.9 | **0.045** / **0.044** | 0.027 / 0.025 | 0.004 / 0.025 |
| Label-free | 98.1 / 98.3 | 1.806 / 1.723 | 0.388 / 0.271 | 0.130 / 0.061 |
| CT-CBM | 98.3 / 98.5 | 0.296 / 0.125 | 0.377 / 0.281 | 0.136 / 0.085 |
| **f-CBM** | 97.7 / 98.2 | 0.079 / 0.057 | **0.005** / **0.004** | **0.005** / **0.003** |

CUB-200（CLIP-base / CLIP-large）:

| 方法 | %ACC↑ | c-RMSE↓ | CTL↓ | ICL↓ |
|---|---|---|---|---|
| Black-box | 91.3 / 95.8 | - | - | - |
| **f-CBM** | **79.3** / **85.3** | 0.200 / 0.273 | **0.026** / 0.045 | - / - |

### 消融实验

f-CBM 在 Pareto 前沿上：在概念检测精度 vs 聚合 leakage 的权衡中，f-CBM 位于其他方法构成的 Pareto 前沿上（Figure 1），实现了最优的忠实性-性能权衡。

关键组件消融：
- **仅 KAN（无 leakage 损失）**: 改善概念检测但 leakage 仅部分降低
- **仅 leakage 损失（无 KAN）**: leakage 显著降低但概念检测不如有 KAN 时好
- **两者结合（f-CBM）**: 在所有指标上取得最优或接近最优

### 关键发现

1. **f-CBM 将 leakage 降低 1-2 个数量级**: CTL 从 ~0.3-0.4 降至 ~0.003-0.005，ICL 从 ~0.06-0.13 降至 ~0.002-0.005。
2. **任务精度几乎不受损**: f-CBM 在 N24News 上达到 97.7-98.2%，接近 black-box 的 98.5%。
3. **假设得到验证**: 显式减少 task leakage 确实附带减少了 inter-concept leakage。
4. **多模态通用性**: f-CBM 在文本-图像数据集（N24News、CUB）和纯文本数据集（AGNews、DBpedia）上均有效。
5. **概念介入效果**: 低 leakage 使概念介入（inference-time concept correction）更可靠，不会因依赖泄漏信息而产生反效果。

## 亮点与洞察

- 首个系统性地在多模态设置下研究 CBM 忠实性的工作
- 初步分析揭示了概念检测-task leakage-inter-concept leakage 的三角关系，为方法设计提供了理论指导
- KDE-based 可微 leakage 损失是一个精巧的设计，使原本不可微的互信息度量可直接嵌入训练
- KAN 层在保持可解释性（可视化响应曲线）的同时提升表达力，解决了线性层"误导"概念层的问题
- 验证了一个重要直觉：提供足够表达力的预测头可减少概念层编码额外信息的需求

## 局限与展望

- CUB 数据集上任务精度与 black-box 差距较大（79.3% vs 91.3%），概念瓶颈的固有限制在细粒度任务上更明显
- 概念标注依赖 CLIP + sentence transformer 的自动标注，引入标注噪声
- KDE 互信息估计的计算复杂度随 batch size 增大而上升
- 仅在分类任务上验证，检索、VQA 等其他多模态任务待探索
- 概念集的选择仍依赖 LLM 生成，自动化程度和质量有改善空间

## 相关工作与启发

- Label-free CBM (Oikarinen et al.) 用 CLIP 实现无监督概念检测，但概念检测误差大（c-RMSE ~1.7）
- CT-CBM 通过概念选择和残差连接缓解 leakage，但残差通道降低了可解释性
- Mahinpei et al. 提出了 leakage 的精确度量（互信息增益），本文将其发展为可微训练目标
- 启发：CBM 的忠实性问题（检测准确性 + leakage）可能存在于所有 concept-based 解释方法中

## 评分

- **新颖性**: ⭐⭐⭐⭐ KDE leakage 损失和 KAN 预测头的结合是创新的，初步分析的三角关系发现有洞察力
- **实验充分度**: ⭐⭐⭐⭐ 4 个数据集、2 种 backbone、多种 baseline 对比，消融验证了各组件贡献
- **写作质量**: ⭐⭐⭐⭐ 从初步分析到方法设计的逻辑链清晰，忠实性度量定义严谨
- **价值**: ⭐⭐⭐⭐ 解决了 CBM 可解释性的核心信任问题，对可解释 AI 领域有重要推动

<!-- RELATED:START -->

## 相关论文

- [Improving Intervention Efficacy via Concept Realignment in Concept Bottleneck Models](../../ECCV2024/interpretability/improving_intervention_efficacy_via_concept_realignment_in_concept_bottleneck_mo.md)
- [Towards Human-Understandable Multi-Dimensional Concept Discovery](towards_human-understandable_multi-dimensional_concept_discovery.md)
- [There Was Never a Bottleneck in Concept Bottleneck Models](../../ICLR2026/interpretability/there_was_never_a_bottleneck_in_concept_bottleneck_models.md)
- [Sample- and Parameter-Efficient Auto-Regressive Image Models](sample-_and_parameter-efficient_auto-regressive_image_models.md)
- [An Analysis of Concept Bottleneck Models: Measuring, Understanding, and Mitigating the Impact of Noisy Annotations](../../NeurIPS2025/interpretability/an_analysis_of_concept_bottleneck_models_measuring_understanding_and_mitigating_.md)

<!-- RELATED:END -->
