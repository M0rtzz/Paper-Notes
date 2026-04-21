---
title: >-
  [论文解读] Coeff-Tuning: A Graph Filter Subspace View for Tuning Attention-Based Large Models
description: >-
  [CVPR 2025][图学习][PEFT] 将多头注意力重新解释为图卷积滤波器子空间，通过学习一组极小的子空间组合系数（$H \times H$ 矩阵）来线性组合预训练的注意力图，突破 softmax 造成的凸包约束从而扩展特征空间，以几乎零参数量的代价即插即用地提升各种 PEFT 方法的性能。
tags:
  - CVPR 2025
  - 图学习
  - PEFT
  - 图卷积滤波器
  - 注意力子空间
  - 子空间系数
  - 即插即用
---

# Coeff-Tuning: A Graph Filter Subspace View for Tuning Attention-Based Large Models

**会议**: CVPR 2025  
**arXiv**: [2503.18337](https://arxiv.org/abs/2503.18337)  
**代码**: https://github.com/ZichenMiao/Coeff_Tuning  
**领域**: 参数高效微调 / 图学习  
**关键词**: PEFT, 图卷积滤波器, 注意力子空间, 子空间系数, 即插即用

## 一句话总结
将多头注意力重新解释为图卷积滤波器子空间，通过学习一组极小的子空间组合系数（$H \times H$ 矩阵）来线性组合预训练的注意力图，突破 softmax 造成的凸包约束从而扩展特征空间，以几乎零参数量的代价即插即用地提升各种 PEFT 方法的性能。

## 研究背景与动机

**领域现状**：参数高效微调 (PEFT) 是适配大型预训练模型到下游任务的核心技术。主流方法如 LoRA、Adapter、VPT 主要从张量分解角度出发，通过低秩矩阵或可学习提示来高效调整线性变换层的权重。

**现有痛点**：现有 PEFT 方法都聚焦于优化线性变换权重矩阵，缺乏对 Transformer 核心操作——多头注意力层——的整体视角。softmax 注意力使得每行的注意力分数都在 (0,1) 范围内且行和为 1，这意味着每个输出 token 嵌入都被限制在 value 嵌入的凸包内，限制了注意力层的表达能力。

**核心矛盾**：微调时需要增强注意力层的表达能力以适应下游任务，但直接修改注意力图的参数量（$W_q, W_k$）很大，且 softmax 归一化始终约束输出在凸包内。

**本文目标** 如何用极少的参数突破 softmax 注意力的凸包限制，增强多头注意力的表达能力。

**切入角度**：将多头注意力视为图卷积操作，每个注意力头的注意力图是一个图卷积滤波器，H 个注意力头构成一个滤波器子空间。通过学习不受约束的线性组合系数，可以用已有的滤波器构造出值域超出 (0,1) 的新滤波器。

**核心 idea**：用 $H^2$ 个可学习系数线性组合 H 个注意力图，打破 softmax 的凸包约束来扩展注意力特征空间。

## 方法详解

### 整体框架
对每个注意力层，保持原始的 Q/K/V 投影不变，在计算出 H 个注意力图 $\{F^1, ..., F^H\}$ 后，用一个可学习的 $H \times H$ 系数矩阵 $\alpha$ 将它们线性组合，得到新的 H 个注意力图 $\hat{F}^h = \sum_{i=1}^H \alpha[h,i] F^i$。由于 $\alpha$ 不受约束，组合后的注意力图元素可以超出 (0,1) 范围甚至为负值，打破凸包限制。

### 关键设计

1. **子空间系数 (Subspace Coefficient $\alpha$)**:

    - 功能：通过线性组合已有注意力图构造更具表达力的滤波器子空间
    - 核心思路：每个注意力层增加一个 $H \times H$ 的可学习矩阵，其中 $\hat{F}^h = \sum_i \alpha[h,i] F^i$。由于 $\alpha$ 无约束，组合后的新注意力图可以包含负值元素，使输出 token 可以位于 value 嵌入凸包之外。对于 ViT-B/16（12 层、12 头），总共只需 $12 \times 12 \times 12 = 1728$ 个参数
    - 设计动机：toy 实验严格验证了当只调 $W_q, W_k$ 时输出被限制在输入范围内，加入 $\alpha$ 后可成功映射到目标范围外的位置

2. **残差参数化 (Residual Parameterization)**:

    - 功能：稳定微调过程，保留预训练知识
    - 核心思路：将系数矩阵参数化为 $\alpha' = \alpha + I$，其中 $\alpha$ 初始化为零。这样初始状态等价于原始预训练模型，微调过程中 $\alpha$ 学习的是相对于身份映射的残差修正。消融实验表明随机初始化 $\alpha$ 仅得 44.28%，身份初始化（非残差）得 61.57%，残差参数化得 73.49%
    - 设计动机：避免微调初期对预训练特征造成破坏性修改

3. **系数 Dropout 正则化**:

    - 功能：防止过拟合，增强泛化能力
    - 核心思路：训练时对 $\alpha$ 直接施加 dropout，以概率 $p$ 随机将部分系数置零。最优 dropout 率为 $p=0.2$，该设置下在 VL-BART 上得分 77.7（$p=0$ 得 75.1，$p=0.4$ 得 72.3）
    - 设计动机：$\alpha$ 虽然参数少但对注意力图影响大，容易在少样本场景下过拟合

### 损失函数 / 训练策略
与基础 PEFT 方法保持一致的训练配置。Coeff-Tuning 作为即插即用模块，与 LoRA、DoRA、SSF 等结合时不改变原有训练流程，仅在注意力计算中额外引入系数组合步骤。

## 实验关键数据

### 主实验

**VTAB-1k 少样本分类 (ViT-B/16)**:

| 方法 | 额外参数 (M) | 平均准确率 |
|------|-------------|-----------|
| Linear Probing | 0 | 52.94% |
| Adapter | 0.234 | 55.82% |
| VPT | 0.06 | 64.85% |
| LoRA | 0.305 | 72.91% |
| **Coeff α only** | **0.002** | **69.78%** |
| FACT-TT≤16 | 0.040 | 73.04% |
| **FACT + Coeff-Tuning** | **0.042** | **73.64%** |
| SNELL-8 | 0.268 | 74.17% |
| **SNELL-8 + Coeff-Tuning** | **0.270** | **74.70%** |

**VL-BART 多模态理解**:

| 方法 | 参数量% | 平均分 |
|------|--------|--------|
| Full Model | 100% | 77.3 |
| LoRA | 5.94% | 76.5 |
| **Coeff + LoRA** | **5.94%** | **77.1** |
| DoRA | 5.96% | 77.4 |
| **Coeff + DoRA** | **5.96%** | **77.7** |

### 消融实验

| 配置 | 平均准确率 |
|------|-----------|
| $\alpha$ 随机初始化 | 44.28% |
| $\alpha$ 单位矩阵初始化 | 61.57% |
| $I + \alpha$, 零初始化（残差参数化） | **73.49%** |

| Dropout rate $p$ | VL-BART Avg |
|-----------------|-------------|
| 0.0 | 75.1 |
| 0.1 | 77.3 |
| 0.2 | **77.7** |
| 0.4 | 72.3 |

### 关键发现
- 仅用 1728 个参数（α only），就能从 52.94% 提升到 69.78%，接近 LoRA（72.91%）的水平
- 与任何 PEFT 方法组合都能带来稳定提升（+0.5~1.0%），且额外参数可忽略
- 残差参数化至关重要：随机初始化直接崩溃（44.28%），残差参数化确保从预训练状态平滑出发
- 在 SDXL 概念定制任务上也有效，概念对齐度从 0.73/0.74 提升到 0.75

## 亮点与洞察
- **图卷积视角重新解释注意力**：将多头注意力理解为图卷积滤波器子空间是一个非常有洞察力的理论贡献。这个视角揭示了 softmax 注意力的凸包限制，为突破该限制提供了理论依据
- **极致的参数效率**：1728 个参数就能获得显著提升，这是目前最参数高效的 PEFT 方法之一。核心在于它操作的是注意力图的组合方式而非权重矩阵本身
- **即插即用的通用性**：可以无缝叠加到 LoRA、DoRA、SSF 等任何 PEFT 方法上，这意味着它是一个正交的改进维度

## 局限与展望
- 核心假设是注意力头之间可以互相线性组合，但当注意力头数量少时（如 H=4），可组合的多样性有限
- 理论分析基于单层注意力，多层叠加的效果缺乏理论保证
- 仅在 ViT、VL-BART、SDXL 上验证，未测试更大规模 LLM（如 LLaMA）
- dropout 率需要调参，不同任务的最优值不同

## 相关工作与启发
- **vs LoRA**: LoRA 修改权重矩阵的低秩增量，Coeff-Tuning 修改注意力图的组合系数，两者正交互补
- **vs OFT**: OFT 保持神经元关系的超球能量，Coeff-Tuning 扩展注意力特征空间，思路完全不同
- **vs DeepViT**: DeepViT 重新生成多样注意力图防止训练坍缩，Coeff-Tuning 在微调阶段用预训练注意力图的新组合实现表达力扩展
- 对需要极致参数效率的边缘设备微调场景有重要参考价值

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 图卷积滤波器子空间视角非常新颖且有理论深度
- 实验充分度: ⭐⭐⭐⭐ 覆盖视觉/多模态/生成任务，但缺少大 LLM 验证
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，toy example 直观
- 价值: ⭐⭐⭐⭐ 极致参数效率 + 即插即用的通用性有重要工程价值

<!-- RELATED:START -->

## 相关论文

- [S'MoRE: Structural Mixture of Residual Experts for Parameter-Efficient LLM Fine-tuning](../../NeurIPS2025/graph_learning/smore_structural_mixture_of_residual_experts_for_parameter-efficient_llm_fine-tu.md)
- [Magnitude-Modulated Equivariant Adapter for Parameter-Efficient Fine-Tuning of Equivariant Graph Neural Networks](../../AAAI2026/graph_learning/magnitude-modulated_equivariant_adapter_for_parameter-efficient_fine-tuning_of_e.md)
- [Graph Attention is Not Always Beneficial: A Theoretical Analysis of Graph Attention Mechanisms via Contextual Stochastic Block Models](../../ICML2025/graph_learning/graph_attention_is_not_always_beneficial_a_theoretical_analysis_of_graph_attenti.md)
- [Table as a Modality for Large Language Models](../../NeurIPS2025/graph_learning/table_as_a_modality_for_large_language_models.md)
- [Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models](../../ICML2025/graph_learning/graph-constrained_reasoning_faithful_reasoning_on_knowledge_graphs_with_large_la.md)

<!-- RELATED:END -->
