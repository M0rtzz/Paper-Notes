---
title: >-
  [论文解读] Imbalance in Balance: Online Concept Balancing in Generation Models
description: >-
  [ICCV 2025][LLM评测][概念组合] 通过精心设计的因果实验揭示了数据分布（而非模型规模或数据量）是扩散模型概念组合能力的决定性因素，并提出 IMBA Loss——一种在线的、概念级别的均衡损失函数，通过条件与无条件分布差异（IMBA 距离）自适应调整 token 级损失权重…
tags:
  - "ICCV 2025"
  - "LLM评测"
  - "概念组合"
  - "数据不平衡"
  - "IMBA Loss"
  - "扩散模型训练"
  - "长尾分布"
---

# Imbalance in Balance: Online Concept Balancing in Generation Models

**会议**: ICCV 2025  
**arXiv**: [2507.13345](https://arxiv.org/abs/2507.13345)  
**代码**: [https://github.com/KwaiVGI/IMBA-Loss](https://github.com/KwaiVGI/IMBA-Loss)  
**领域**: LLM评测  
**关键词**: 概念组合, 数据不平衡, IMBA Loss, 扩散模型训练, 长尾分布

## 一句话总结

通过精心设计的因果实验揭示了数据分布（而非模型规模或数据量）是扩散模型概念组合能力的决定性因素，并提出 IMBA Loss——一种在线的、概念级别的均衡损失函数，通过条件与无条件分布差异（IMBA 距离）自适应调整 token 级损失权重，只需几行代码修改即可显著提升模型的多概念生成能力。

## 研究背景与动机

当前最先进的 T2I 模型（DALL-E 3、SD3、Midjourney、Flux）在处理多概念组合时仍存在严重问题：**概念缺失**（生成图像中缺少某个概念）、**属性泄露**（属性错误匹配到其他物体）和**概念纠缠**（出现不应存在的多余概念）。

现有工作主要从推理端（training-free）着手，通过优化注意力图来增强概念响应，但受限于基础模型能力。更根本的问题是：**什么因素决定了模型的概念组合能力？** 此前的研究大多基于简单的合成数据（基本形状+颜色），与真实 T2I 任务差距巨大。

作者提出并逐一验证三个假设：(1) 数据规模够大就自然覆盖所有概念？(2) 模型够大就能学好概念组合？(3) 数据分布才是关键？结论是 **只有第三个假设成立**，且现有数据集虽然表面上规模巨大，实际上呈现严重的长尾分布，导致尾部概念的学习不充分。

## 方法详解

### 整体框架

IMBA Loss 的核心思路是：在标准扩散训练损失上乘以一个自适应的概念级权重，使尾部概念（低频概念）获得更高的损失权重。该权重由 **IMBA 距离**（条件分布与无条件分布之间的差异）实时估计，不需要离线统计数据分布，完全在线计算。

### 关键设计

1. **IMBA 距离定义与数据分布近似**: 从理论推导发现，概念 $c_j$ 的条件分布与无条件分布之差 $D_j = \|\epsilon_\theta(a_t, c_j, t) - \epsilon_\theta(a_t, \phi, t)\| \propto \frac{1}{\varphi(c_j)}$，其中 $\varphi(c_j)$ 是概念频率。直觉上，高频概念的无条件分布已偏向其方向，差异自然小；低频概念则差异大。训练中用 ground truth 噪声替代条件预测以提升稳定性：$D = \|\epsilon - \epsilon_\theta(x_t, \phi, t)\|_{sg}^\gamma$，其中 $sg$ 表示停止梯度。实质上 IMBA 距离等价于**无条件损失的 $\gamma$ 次方**。

2. **Token 级重加权（而非样本级）**: 传统类别不均衡方法为每个样本分配一个权重，但在 T2I 任务中每个图像包含多种概念、不同区域对应不同概念。IMBA Loss 在空间维度（token 级别）应用不同权重，对图像中不同概念区域赋予不同的均衡力度。这比样本级重加权更精细准确。IMBA 距离的形状为 $(B, N, C)$，在通道维度取平均以提升稳定性。

3. **在线自适应（无需离线统计）**: 传统数据均衡需要先遍历整个数据集统计频率，随着数据集增大到数百万乃至上亿级别，这变得极其昂贵。IMBA 距离在每个训练步自然产生，随模型训练动态变化，始终反映模型当前对数据分布的理解，且只需在标准训练流程中增加一次无条件前向传播（本就是 classifier-free guidance 训练的一部分）。

### 损失函数 / 训练策略

最终 IMBA Loss 形式：$L^* = \mathbb{E}_{t,x_0,\epsilon} D \|\epsilon - \epsilon_\theta(x_t, y, t)\|^2$，其中 $D = \|\epsilon - \epsilon_\theta(x_t, \phi, t)\|_{sg}^\gamma$。

训练算法（Algorithm 1）：
1. 采样数据对 $(x_0, y)$，噪声 $\epsilon$，时间步 $t$
2. 加噪得 $x_t$
3. 计算 IMBA 距离 $D$（无条件预测 vs ground truth，停止梯度）
4. 条件损失 $L^* = D \|\epsilon - \epsilon_\theta(x_t, y, t)\|^2$
5. 无条件损失 $L_u = \|\epsilon - \epsilon_\theta(x_t, \phi, t)\|^2$
6. 总损失 $L = \lambda L^* + (1-\lambda) L_u$，其中 $\lambda = 0.9$

关键超参数：$\gamma = 0.8$（避免颜色偏移），$\lambda = 0.9$（与原始条件掩码比例一致）。

## 实验关键数据

### 主实验 (表格)

**三个 Benchmark 定量对比**

| 模型 | LC-Mis CLIP↑ | LC-Mis VQA↑ | T2I-Comp Color↑ | T2I-Comp Shape↑ | T2I-Comp Texture↑ | Inert-Comp CLIP↑ | Inert-Comp VQA↑ |
|------|-------------|-------------|-----------------|-----------------|-------------------|-----------------|-----------------|
| Baseline | 0.3045 | 46.21% | 0.5812 | 0.4307 | 0.6188 | 0.3194 | 44% |
| A&E (training-free) | 0.3198 | 48.42% | 0.6141 | 0.4378 | 0.6329 | 0.3303 | 44.5% |
| Finetune (diffusion loss) | 0.3073 | 51.82% | 0.6668 | 0.4919 | 0.6575 | 0.3172 | 46% |
| **IMBA Loss (ours)** | 0.3121 | **62.89%** | **0.7067** | **0.5151** | **0.6861** | 0.3229 | **57%** |

IMBA Loss 在 VQA 成功率上从 46.21% 提升到 **62.89%**（+16.7pp），Color/Shape/Texture 属性准确率全面领先。

### 消融实验 (表格)

**数据分布 vs 数据规模因果实验**

| 概念对 | Head 样本 | Tail 样本 | 比例 | 成功率 | CLIP Score |
|--------|----------|----------|------|--------|-----------|
| piano-submarine (不平衡大) | 3K | 0.03K | 100:1 | 16% | 0.3076 |
| piano-submarine (不平衡更大) | 15K | 0.15K | 100:1 | 20% | 0.3110 |
| piano-submarine (**平衡**) | **0.15K** | **0.15K** | **1:1** | **56%** | **0.3226** |
| volcano-twins (不平衡大) | 1K | 0.1K | 10:1 | 28% | 0.2986 |
| volcano-twins (不平衡更大) | 5K | 0.5K | 10:1 | 20% | 0.2948 |
| volcano-twins (**平衡**) | **0.5K** | **0.5K** | **1:1** | **64%** | **0.3137** |

**关键发现**: 数据规模扩大 5 倍（保持相同不平衡分布）对概念组合成功率几乎无提升甚至下降；而平衡化数据（即使数据量更少）大幅提升成功率（16%→56%，28%→64%）。

**损失权重粒度对比**

| 权重粒度 | Baseline | Sample-wise | Token-wise (ours) |
|---------|----------|-------------|-------------------|
| 成功率 | 32% | 64% | **72%** |
| CLIP Score | 0.2924 | 0.3022 | **0.3106** |

Token-wise 重加权优于 sample-wise，因为它能对图像中不同概念区域施加不同权重。

**均衡方法对比**

| 方法 | Baseline | Frequency-based | IMBA Loss (ours) |
|------|----------|----------------|------------------|
| 成功率 | 33.3% | 49.3% | **65.7%** |
| CLIP Score | 0.3113 | 0.3101 | **0.3218** |

### 关键发现

- **模型规模不是瓶颈**: 100M→1B 参数，超过 200M 后概念组合能力不再随模型增大提升。
- **数据规模也不是**: 保持相同不平衡分布下增加数据量，概念组合不改善。
- **数据分布是根因**: 平衡化的小数据集胜过不平衡的大数据集。
- **IMBA 距离跨架构稳定**: 不同模型大小、架构和噪声下，IMBA 距离一致反映数据分布。
- **惰性概念（Inert concepts）**: 低频概念更难参与组合，成功率随频率下降几乎线性降低。
- **$\gamma$ 的影响**: $\gamma \to 0$ 退化为标准损失，$\gamma \to 2$ 引起严重颜色偏移，$\gamma = 0.8$ 为最佳平衡。

## 亮点与洞察

- **因果分析严谨**: 不同于大多数工作直接提方法，本文先用控制变量实验验证因果假设，再设计解决方案，研究范式值得学习。
- **理论推导自洽**: 从理想均衡分布出发推导损失权重形式，发现 IMBA 距离（无条件损失的 $\gamma$ 次方）是频率倒数的自然近似，理论与实践统一。
- **极简实现**: 只需在标准训练代码中修改几行——计算无条件预测（本来就有），用它对条件损失加权。
- **2D 合成实验精妙**: 用二维空间直观展示了数据不平衡导致无条件分布偏向高频概念、进而降低低频概念响应强度的机理。
- **新 Benchmark Inert-CompBench**: 专门针对难以组合的惰性概念设计，填补了现有 benchmark（T2I-CompBench、LC-Mis）对尾部概念覆盖不足的空白。

## 局限与展望

- 基于 1B 参数的 DiT 模型训练，未在更大模型（如 SDXL 级别）或商用模型上验证。
- IMBA 距离在 $t=1000$（全噪声）时才有跨概念可比性，中间时间步的比较需要更多研究。
- $\gamma = 0.8$ 和 $\lambda = 0.9$ 的超参数选择来自经验调优，不同任务/数据集可能需要重新调整。
- 方法假设无条件分布能反映概念频率，但在训练早期模型欠拟合时这一假设可能不完全成立。
- fine-tuning 场景下的改进有限（Inert-CompBench 上只从 44%→46%），惰性概念可能需要更长训练。
- 仅关注名词概念的组合，对动词、副词等其他类型概念的平衡未探讨。

## 相关工作与启发

- 与 Attend-and-Excite 等 training-free 方法正交，可组合使用进一步提升。
- 数据不平衡问题在分类任务（Focal Loss、类别频率加权）中已有大量研究，但 T2I 的概念级平衡面临独特挑战：每张图含多概念、概念空间开放、频率统计成本高。IMBA Loss 优雅绕过了这些问题。
- 对视频生成（Kling 团队出品）的启示：视频中概念组合更复杂（时间维度的概念交互），IMBA Loss 可直接扩展。
- **核心启发**: "看似平衡的大数据本质上是不平衡的"——这一洞见对所有使用大规模互联网数据训练的生成模型都适用。

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 因果分析+理论推导+在线方法，从问题发现到解决方案完整且优雅
- **实验充分度**: ⭐⭐⭐⭐ 控制变量实验+三个 benchmark+多种消融，但基础模型规模较小
- **写作质量**: ⭐⭐⭐⭐ 叙事逻辑清晰，但数学符号较多需要仔细阅读
- **价值**: ⭐⭐⭐⭐⭐ 对扩散模型训练有普遍指导意义，几行代码即可应用于任何扩散模型

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] BATCLIP: Bimodal Online Test-Time Adaptation for CLIP](batclip_bimodal_online_test-time_adaptation_for_clip.md)
- [\[ACL 2025\] Ad-hoc Concept Forming in the Game Codenames as a Means for Evaluating Large Language Models](../../ACL2025/llm_evaluation/ad-hoc_concept_forming_in_the_game_codenames_as_a_means_for_evaluating_large_lan.md)
- [\[NeurIPS 2025\] Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training](../../NeurIPS2025/llm_evaluation/exploiting_vocabulary_frequency_imbalance_in_language_model_pre-training.md)
- [\[ICCV 2025\] Lay2Story: Extending Diffusion Transformers for Layout-Togglable Story Generation](lay2story_extending_diffusion_transformers_for_layout-togglable_story_generation.md)
- [\[ICML 2025\] Addressing Imbalanced Domain-Incremental Learning through Dual-Balance Collaborative Experts (DCE)](../../ICML2025/llm_evaluation/addressing_imbalanced_domain-incremental_learning_through_dual-balance_collabora.md)

</div>

<!-- RELATED:END -->
