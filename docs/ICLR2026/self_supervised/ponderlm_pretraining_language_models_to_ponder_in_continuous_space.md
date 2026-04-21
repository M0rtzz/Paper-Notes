---
title: >-
  [论文解读] PonderLM: Pretraining Language Models to Ponder in Continuous Space
description: >-
  [ICLR2026][自监督学习][pondering] 提出 PonderLM，在预训练阶段引入"沉思"机制——将预测概率分布加权求和为连续嵌入后反复前向传播，无需标注数据或强化学习，使 2.8B 模型在 9 个下游任务上超越 6.9B 模型。
tags:
  - ICLR2026
  - 自监督学习
  - pondering
  - language model
  - continuous space
  - test-time compute
  - pretraining
---

# PonderLM: Pretraining Language Models to Ponder in Continuous Space

**会议**: ICLR2026  
**arXiv**: [2505.20674](https://arxiv.org/abs/2505.20674)  
**代码**: 待确认  
**领域**: self_supervised  
**关键词**: pondering, language model, continuous space, test-time compute, pretraining

## 一句话总结
提出 PonderLM，在预训练阶段引入"沉思"机制——将预测概率分布加权求和为连续嵌入后反复前向传播，无需标注数据或强化学习，使 2.8B 模型在 9 个下游任务上超越 6.9B 模型。

## 研究背景与动机

**领域现状**：提升模型能力的主流方法是扩大参数和数据规模，但面临数据耗尽、缩放饱和、通信开销等瓶颈。推理时缩放（CoT）也有限制：需要标注数据、强化学习，小模型难以受益。

**现有痛点**：CoT 在离散语言空间操作，受限于固定词表，且性能上界受基础预训练模型约束。

**核心矛盾**：需要更多计算来提升性能，但简单增加参数成本太高。

**本文目标** 在不增加参数的情况下，通过在单个 token 生成步内多次前向传播来提升性能。

**切入角度**：类比人类面对复杂问题会反复沉思，让模型在连续空间中"思考"。

**核心 idea**：将预测概率与词嵌入做加权和形成"沉思嵌入"，残差加到输入后再次前向传播，重复 $s$ 步。

## 方法详解

### 整体框架
标准 LM 生成概率 $\mathbf{P}$ → 加权求和所有词嵌入得到沉思嵌入 $\mathbf{T} = \mathbf{P}\mathbf{V}$ → 残差连接 $\mathbf{E}^1 = \mathbf{E}^0 + \mathbf{T}$ → 再次前向传播 → 重复 $s$ 步。

### 关键设计

1. **沉思机制**: $\mathbf{t} = \sum_i p_i \mathbf{e}_i$，连续嵌入保留了所有候选 token 的信息，实现可微端到端训练
2. **效率优化**: 只用 top-K（K=100）token 的概率计算沉思嵌入，复杂度从 $\mathcal{O}(n|V|d)$ 降至 $\mathcal{O}(nKd)$
3. **纯自监督**: 不需要标注数据或强化学习，通过标准语言建模预训练即可学会沉思

### 训练策略
使用标准 NTP 损失在大规模语料上预训练，$s=3$ 步沉思。

## 实验关键数据

### 主实验

| 模型 | 参数量 | 训练数据 | 9任务平均 |
|------|--------|---------|---------|
| Pythia-6.9B | 6.9B | 300B tokens | 基线 |
| **PonderPythia-2.8B** | **2.8B** | **300B tokens** | **超越 6.9B** |
| TinyLlama-1.1B | 1.1B | 3T tokens | 基线 |
| **PonderPythia-1B** | **1B** | **300B tokens** | **匹配 TinyLlama** |

### 关键发现
- 2.55B 模型匹配 Pythia-6.9B 的 loss（63% 参数减少）
- 增加沉思步数持续提升性能
- 在 GPT-2、Pythia、LLaMA 三种架构上都有效

## 消融实验与深入分析

| 消融/分析 | 发现 |
|-----------|------|
| 沉思步数 $s$ | $s=1→2→3$ 持续提升性能，加步数有稳定收益 |
| Top-K 近似 | $K=100$ 足够好，进一步增大 K 无显著提升，显著降低计算复杂度 |
| 架构通用性 | GPT-2、Pythia、LLaMA 三种架构上均有效 |
| 缩放行为 | 405M→1.4B 范围内，沉思模型始终优于同参数量的基线 |
| 推理时步数调整 | 推理时可增加沉思步数（如训练 $s=3$，推理 $s=5$），有额外增益但需验证 |
| FLOPs 控制比较 | 在相同 FLOPs 下，PonderPythia-70M 持续优于 vanilla Pythia-70M |

### 缩放曲线核心发现
- **参数效率**：2.55B 参数的 PonderPythia 匹配 6.9B 参数 Pythia 的 validation loss（63% 参数减少）
- **数据效率**：PonderPythia 用 59% 更少的 training tokens 达到 Pythia 基线的同等性能
- **FLOPs 效率**：相同计算预算下 PonderPythia 始终更优——说明额外前向传播的计算开销被性能提升所补偿

### 下游任务细项

| 模型 | LAMBADA↑ | ARC-E↑ | WinoGrande↑ | PIQA↑ | SciQ↑ | 平均↑ |
|------|----------|--------|-------------|-------|-------|-------|
| Pythia-1B (300B) | 48.3 | 58.6 | 52.8 | 71.3 | 91.6 | 50.4 |
| PonderPythia-410M (300B) | 48.9 | 58.7 | 54.0 | 70.5 | 91.0 | **51.4** (+3.8) |
| Pythia-6.9B (300B) | 基线 | 基线 | 基线 | 基线 | 基线 | 基线 |
| **PonderPythia-2.8B** (300B) | **超越** | **超越** | **超越** | **超越** | **超越** | **超越 6.9B** |

## 亮点与洞察
- **第三条缩放轴**：传统缩放只有参数缩放和推理缩放（CoT），PonderLM 开辟了"沉思缩放"——相同参数通过多次前向传播提升
- **连续空间中的思考**：CoT 在离散 token 空间操作，受词表限制；沉思嵌入是所有 token 的概率加权连续向量，信息密度更高
- **可解释性窗口**：中间沉思步的概率分布变化提供了推理过程的可视化——可以看到模型如何从初始猜测逐步修正到正确答案
- **纯自监督**：不需要标注数据或 RL，通过标准 NTP 即可学会有效沉思——这使得方法的适用性极广
- **与 CoT 正交**：沉思发生在单个 token 生成步内，CoT 发生在 token 序列层面——两者可以叠加使用

## 局限与展望
- 推理开销随沉思步数线性增长（$s$ 步需要 $s+1$ 次完整前向传播），对延迟敏感的应用不友好
- 与 CoT 的组合效果未探索——沉思模型在 RL/CoT 训练后是否有额外增益？
- 沉思步数 $s$ 在训练和推理时固定——自适应步数（根据问题难度动态调整）可能更高效
- 训练时沉思增加了每步的计算量，整体训练速度变慢——虽然 FLOPs 效率更高但 wall-clock time 未详细讨论
- 目前仅在 Pile 数据集上验证，更多数据分布和模态上的验证有价值

## 相关工作与启发
- **vs CoT/o1/R1**：CoT 在离散空间生成推理链，PonderLM 在连续空间迭代精化——前者需要标注数据或 RL，后者纯自监督
- **vs Universal Transformer (Dehghani et al.)**：UT 允许变长计算（每个 token 不同层数），PonderLM 允许同一层的多次迭代——思路相近但机制不同
- **vs PonderNet (Banino et al.)**：PonderNet 学习何时停止计算（动态 halting），PonderLM 固定步数但通过连续嵌入保留更多信息
- **启发**：沉思机制可以扩展到多模态——视觉 token 和文本 token 的混合沉思可能实现跨模态的隐式推理

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 连续空间沉思机制是全新思路，开辟第三条缩放轴
- 实验充分度: ⭐⭐⭐⭐ 三种架构+9 个下游任务，缩放曲线严谨
- 写作质量: ⭐⭐⭐⭐ 直觉解释好，伪代码清晰
- 价值: ⭐⭐⭐⭐⭐ 提出了新的计算缩放范式，与现有方向正交可叠加

<!-- RELATED:START -->

## 相关论文

- [L2D: Large Language Models to Diffusion Finetuning](../../ICML2025/self_supervised/large_language_models_to_diffusion_finetuning.md)
- [Continuous Subspace Optimization for Continual Learning (CoSO)](../../NeurIPS2025/self_supervised/continuous_subspace_optimization_for_continual_learning.md)
- [Towards LLM-Empowered Knowledge Tracing via LLM-Student Hierarchical Behavior Alignment in Hyperbolic Space](../../AAAI2026/self_supervised/towards_llm-empowered_knowledge_tracing_via_llm-student_hierarchical_behavior_al.md)
- [Scaling Language-Free Visual Representation Learning](../../ICCV2025/self_supervised/scaling_languagefree_visual_representation_learning.md)
- [A Bayesian Model Selection Criterion for Selecting Pretraining Checkpoints](../../ICML2025/self_supervised/a_bayesian_model_selection_criterion_for_selecting_pretraining_checkpoints.md)

<!-- RELATED:END -->
