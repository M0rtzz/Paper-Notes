---
title: >-
  [论文解读] CHEEM: Continual Learning by Reuse, New, Adapt and Skip -- A Hierarchical Exploration-Exploitation Approach
description: >-
  [CVPR 2026][LLM效率][持续学习] 提出 CHEEM 框架，通过分层探索-利用采样的 NAS 自动学习任务感知的动态 ViT 骨干——在每一层选择 Reuse/New/Adapt/Skip 四种操作——在 MTIL 和 VDD 两个挑战性持续学习基准上显著超越提示类方法，接近全量微调上界。
tags:
  - "CVPR 2026"
  - "LLM效率"
  - "持续学习"
  - "无样本类增量学习"
  - "神经架构搜索"
  - "层间操作"
  - "Transformer"
---

# CHEEM: Continual Learning by Reuse, New, Adapt and Skip -- A Hierarchical Exploration-Exploitation Approach

**会议**: CVPR 2026  
**arXiv**: [2303.08250](https://arxiv.org/abs/2303.08250)  
**代码**: [GitHub](https://github.com/savadikarc/cheem)  
**领域**: LLM效率  
**关键词**: 持续学习, 无样本类增量学习, 神经架构搜索, 层间操作, Vision Transformer

## 一句话总结

提出 CHEEM 框架，通过分层探索-利用采样的 NAS 自动学习任务感知的动态 ViT 骨干——在每一层选择 Reuse/New/Adapt/Skip 四种操作——在 MTIL 和 VDD 两个挑战性持续学习基准上显著超越提示类方法，接近全量微调上界。

## 研究背景与动机

**领域现状**：基于 ViT 的持续学习取得进展，主要分为提示类方法（冻结骨干+学习提示）和参数调整类方法（适配器/掩码）。

**现有痛点**：(a) 提示类方法保持稳定性但缺乏可塑性——冻结骨干无法适应与预训练显著不同的任务；(b) 参数调整类方法每层统一操作，忽略了任务难度/相似度的差异——简单任务不需要所有层，困难任务可能需要全新层。

**核心矛盾**：现有方法将计算固定或递增，无论任务难度如何——没有 Skip（简化模型处理简单任务）和 New（引入全新层处理极不相似任务）操作。

**本文目标**：学习任务感知的动态模型结构，自动在层间分配 Reuse/Adapt/New/Skip 操作。

**切入角度**：将持续学习视为持续记忆学习问题，由内部参数记忆（NAS 驱动的动态骨干）和外部质心记忆（任务 ID 推断）组成。

**核心 idea**：分层探索-利用（HEE）采样的 NAS，基于任务相似性自动选择层操作。

## 方法详解

### 整体框架

以预训练 ViT 为起点，选取 MLPDown 层和 MHSA 投影层作为可塑组件。对每个新任务：(1) 构建超网（搜索空间）；(2) HEE-NAS 训练超网；(3) 计算感知演化搜索选出目标网络；(4) 重训练目标网络。

### 关键设计

1. **四种层操作**：

    - **Reuse**：直接复用旧任务的层参数，实现知识迁移
    - **Adapt**：在旧参数上添加 LoRA 低秩适配，$\theta_l^{(3,)} = \theta_l^{(2,)} + B_l \cdot A_l$
    - **New**：引入全新层（随机初始化），处理与旧任务极不同的情况
    - **Skip**：跳过整个 FFN/MHSA 块，为简单任务减少计算
   
   设计动机：Reuse（知识迁移）和 Adapt（微调）是现有方法的常规选项，但 New（应对大域差异）和 Skip（应对简单任务）是被忽视的关键操作。例如从 ImageNet 到 MNIST，可 Skip 大量层。

2. **分层探索-利用（HEE）采样**：

    - **利用（Exploitation）**：基于任务相似度采样——用归一化余弦相似度 $S_l^{i,t} = \text{NormCosine}(\mu_l^i, \mu_l^{i \to t})$ 计算旧专家与新任务的相似度，形成分类分布
    - 两级层次采样：第一级从旧任务和 Aux（New/Skip 联合）中采样；第二级用 Bernoulli 决定 Reuse（概率 $\rho_i$）vs Adapt（概率 $1-\rho_i$）
    - **探索（Exploration）**：均匀随机采样，以概率 $\epsilon_1=0.3$ 使用
    - 在超网训练中偏向利用（$\epsilon_1=0.3$），在演化搜索中偏向探索（$\epsilon_2=0.5$）
   
   设计动机：纯均匀采样忽略任务相似性信息，纯利用会陷入局部最优。HEE 通过 epoch-wise 调度平衡两者。

3. **计算感知目标网络选择**：预定义性能容忍阈值 $\tau=2\%$，在同组候选中按计算量升序排列，通过交叉和变异演化搜索。选出后从头重训可学习参数。
   设计动机：仅追求最高精度会忽略计算效率，分组排序保证在同等性能下选择最轻量结构。

### 评估指标

提出 Figure of Merit (FoM)：
$$\text{FoM}(m,n) = \frac{A\mathbb{A}^{UB} - A\mathbb{A}^n}{A\mathbb{A}^{UB} - A\mathbb{A}^m} \cdot \frac{\text{FLOPs}^n}{\text{FLOPs}^m}$$
综合考虑精度差距和计算成本，FoM > 1 表示方法 m 优于 n。

## 实验关键数据

### 主实验 — CHEEM vs 上界

| 方法 | MTIL (ViT-B) | MTIL (DEiT-T) | VDD (ViT-B) | VDD (DEiT-T) |
|------|-------------|---------------|-------------|---------------|
| Upper Bound (Full-FT) | 88.1 | 75.3 | 88.7 | 76.21 |
| CHEEM | 85.9 | 74.5 | 86.7 | 76.18 |
| 差距 | -2.2 | -0.8 | -2.0 | **-0.03** |

### 消融实验 — FoM 对比

| 方法 | MTIL ViT-B | MTIL DEiT-T | VDD ViT-B | VDD DEiT-T |
|------|-----------|-------------|-----------|------------|
| CODA-P | 24.2 | 84.6 | 35.9 | 2811.3 |
| S-Prompts | 3.2 | 10.5 | 5.5 | 338.8 |
| LoRA-CL | 1.7 | 5.7 | 1.5 | 73.6 |
| Tuna | 50.6 | 242.4 | 58.6 | 4538.8 |

所有 FoM > 1 表明 CHEEM 在精度-计算综合指标上优于所有基线。

### 关键发现

1. **任务感知结构语义合理**：Caltech101（简单任务）Skip 5 个 MLP 块；FGVC Aircraft（细粒度困难任务）使用 New + Adapt + Skip 组合。
2. **DEiT-Tiny 上 VDD 几乎无损**：76.18 vs 76.21（Full-FT），差距仅 0.03%，说明 CHEEM 在小模型上也能充分挖掘容量。
3. **LoRA-CL（CHEEM 的特例，所有层 Adapt）**在 ViT-Base 上接近 CHEEM（FoM 1.7），但 DEiT-Tiny 上差距大（FoM 73.6），说明小模型需要 New/Skip 操作。
4. 提示类方法在 DEiT-Tiny 上几乎失效（CODA-P 仅 5.6% Acc），因为其受限于骨干容量。

## 亮点与洞察

- 首次在持续学习中完整实现 Reuse/New/Adapt/Skip 四操作，形成从"学习复用"到"学习成长"到"学习剪枝"的完整谱系。
- 学到的任务结构非常可解释——简单任务轻量、困难任务复杂，与人类直觉一致。
- HEE 采样巧妙利用 CLS token 相似度作为任务相似性代理。
- FoM 指标为持续学习社区提供了综合评价工具。

## 局限与展望

- NAS 搜索增加训练时间，虽然单次任务开销可控但多任务累积不可忽视。
- 外部任务质心记忆（任务 ID 推断）沿用前人工作，与内部记忆的联合优化未深入。
- 仅在分类任务上验证，未推广到检测/分割等更复杂视觉任务。
- New 操作引入全新参数，在任务极多时可能导致参数增长过快。

## 相关工作与启发

- 与 L2G 方法（DARTS 驱动的 ConvNet 持续学习）思路相似，但 CHEEM 加入了 Skip 操作和 HEE 采样，针对 ViT 设计。
- 为"持续学习中自适应计算分配"这一开放问题提供了可行方案。
- HEE 采样可推广到其他需要平衡探索-利用的 NAS 场景。

## 评分

- 新颖性: ⭐⭐⭐⭐ 四操作+HEE采样的组合新颖，但各技术组件（NAS/LoRA/Skip）均为已有
- 实验充分度: ⭐⭐⭐⭐⭐ 两基准×两骨干×多基线+可解释结构分析
- 写作质量: ⭐⭐⭐⭐ 框架图清晰，但文章较长
- 价值: ⭐⭐⭐⭐ 在挑战性基准上显著超越提示类方法，接近全量微调

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Language Guided Concept Bottleneck Models for Interpretable Continual Learning](../../CVPR2025/llm_efficiency/language_guided_concept_bottleneck_models_for_interpretable_continual_learning.md)
- [\[ICML 2026\] Scaling Continual Learning to 300+ Tasks with Bi-Level Routing Mixture-of-Experts](../../ICML2026/llm_efficiency/scaling_continual_learning_to_300_tasks_with_bi-level_routing_mixture-of-experts.md)
- [\[ICLR 2026\] One-Prompt Strikes Back: Sparse Mixture of Experts for Prompt-based Continual Learning](../../ICLR2026/llm_efficiency/one-prompt_strikes_back_sparse_mixture_of_experts_for_prompt-based_continual_lea.md)
- [\[ACL 2025\] CLaSp: In-Context Layer Skip for Self-Speculative Decoding](../../ACL2025/llm_efficiency/clasp_self_speculative_decoding.md)
- [\[ICLR 2026\] Understanding and Improving Length Generalization in Hierarchical Sparse Attention Models](../../ICLR2026/llm_efficiency/understanding_and_improving_length_generalization_in_hierarchical_sparse_attenti.md)

</div>

<!-- RELATED:END -->
