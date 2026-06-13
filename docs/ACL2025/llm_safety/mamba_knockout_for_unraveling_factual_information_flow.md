---
title: >-
  [论文解读] Mamba Knockout for Unraveling Factual Information Flow
description: >-
  [ACL 2025][LLM安全][Mamba] 将 Transformer 上的 Attention Knockout 可解释性方法迁移至 Mamba-1 和 Mamba-2，揭示了 SSM 模型中事实信息的流动模式——发现 Mamba 与 Transformer 共享"主语 token 在中后层向最后 token 传递关键信息"的普遍模式，但在首 token 偏置和关系 token 依赖等方面存在架构特异性差异。
tags:
  - "ACL 2025"
  - "LLM安全"
  - "Mamba"
  - "SSM"
  - "注意力敲除"
  - "事实信息流"
  - "可解释性"
---

# Mamba Knockout for Unraveling Factual Information Flow

**会议**: ACL 2025  
**arXiv**: [2505.24244](https://arxiv.org/abs/2505.24244)  
**代码**: [有](https://github.com/nirendy/mamba-knockout)  
**领域**: LLM安全  
**关键词**: Mamba, SSM, 注意力敲除, 事实信息流, 可解释性

## 一句话总结

将 Transformer 上的 Attention Knockout 可解释性方法迁移至 Mamba-1 和 Mamba-2，揭示了 SSM 模型中事实信息的流动模式——发现 Mamba 与 Transformer 共享"主语 token 在中后层向最后 token 传递关键信息"的普遍模式，但在首 token 偏置和关系 token 依赖等方面存在架构特异性差异。

## 研究背景与动机

Transformer 模型的内部事实信息流已被广泛研究（如 Geva et al. 2023 的 Attention Knockout），但基于状态空间模型（SSM）的 Mamba 架构在事实知识如何存储和传递方面仍是未知领域。

关键理论联系：近期研究（Ali et al., 2024; Dao & Gu, 2024）证明了选择性 SSM 可以通过"隐含注意力"视角来理解——Mamba-2 被直接证明等价于一类线性注意力 Transformer。这使得将 Transformer 上的可解释性工具迁移到 Mamba 成为可能。

核心问题：
1. Mamba 中的事实信息流模式是否与 Transformer 一致？哪些是架构通用的，哪些是架构特异的？
2. SSM 独特的结构（上下文相关 vs 上下文无关特征）各自扮演什么角色？

## 方法详解

### 整体框架

本文的方法论包含两个层次：

1. **注意力敲除（Attention Knockout）**：从 Transformer 迁移到 Mamba，用于分析 token 间信息流
2. **特征敲除（Feature Knockout）**：利用 SSM 的独特结构，分析不同类型特征的作用

### 关键设计

#### 1. Mamba-1 的隐含注意力敲除

**功能**：在 Mamba-1 的核矩阵（等价于注意力矩阵）中，将两个 token 之间的连接置零，观察对预测的影响。

**核心思路**：利用 Ali et al. (2024) 提出的隐含注意力视角，Mamba-1 的选择性 SSM 可以表示为核矩阵：

$$\mathbf{M}_{i,j} = Q_i \cdot H_{i,j} \cdot K_j$$

其中 $Q_i = C(i)$，$K_j = B(j)$，$H_{i,j} = \prod_{t=i}^{j} A(t)$。要敲除 token i 到 j 的信息流，直接设 $\mathbf{M}_{i,j} = 0$。

**设计动机**：虽然 Sharma et al. (2024) 质疑精细粒度 阻断的可行性（因卷积和 softmax 层），但实验证明直接实现就能有效复现 Transformer 中观察到的现象。

#### 2. Mamba-2 的隐式线性注意力敲除

**功能**：在 Mamba-2 中，SSM 层直接表达为掩码线性注意力 $\mathbf{L} \circ (\mathbf{X}\mathbf{M}\mathbf{X}^\top)\mathbf{X}$，其中注意力矩阵中的 (i,j) 元素量化了 token i 对 token j 的关注程度，直接置零即可实施敲除。

**设计动机**：Mamba-2 的 SSD 框架明确建立了与线性注意力的等价关系，使得敲除操作有更直接的语义解释。

#### 3. 特征敲除（Feature Knockout）— 独创贡献

**功能**：利用 SSM 中每个特征由独立 SSM 建模的性质，按特征类型进行选择性敲除。

**核心思路**：根据状态转移矩阵 A̅ 的衰减特性将特征分为两类：
- **上下文相关特征**（$\|\bar{A}\|_1$ 最大的 1/3）：A̅ ≈ 1，保留长程历史信息，负责 token 间信息传递
- **上下文无关特征**（$\|\bar{A}\|_1$ 最小的 1/3）：A̅ ≈ 0，快速遗忘，只关注局部信息，负责单 token 内表示丰富

**设计动机**：这是 SSM 独有的分析维度——Transformer 中没有类似的"衰减特征"概念。通过对比两类特征的敲除效果，揭示它们在事实推理中的不同角色。

### 损失函数 / 训练策略

本文是可解释性分析工作，不涉及模型训练。所有实验基于预训练模型的推理时干预。

## 实验关键数据

### 主实验（表格）

**主语 token 敲除对正确预测概率的影响（相对变化）**：

| 模型 | 中后层敲除效果 | 主语依赖 | 首 token 偏置 | 关系 token 依赖 |
|------|----------------|----------|---------------|-----------------|
| GPT-2 (355M-1.5B) | 显著概率下降 | ✓ 强 | ✓ 强 | 早期层依赖，后期减弱 |
| Mamba-1 (130M-2.8B) | 显著概率下降 | ✓ 强 | ✗ 弱 | 后期层依赖，先降后升 |
| Mamba-2 (130M-2.7B) | 显著概率下降 | ✓ 强 | ✗ 弱 | 后期层依赖 |
| Falcon-Mamba | 显著概率下降 | ✓ 强 | ✗ 弱 | 后期层先降后升 |

### 消融实验（表格）

**特征敲除实验（主语→最后 token，主要在 Mamba-1/2 上）**：

| 敲除范围 | 效果 |
|----------|------|
| 全部特征 | 显著概率下降（基线） |
| 仅上下文相关特征 | 效果几乎等同全部敲除 |
| 仅上下文无关特征 | 几乎无影响 |

这证明了**上下文相关特征**是事实信息 token 间传递的关键载体。

**窗口大小消融（Mamba-1 130M/1.4B/2.8B）**：

| 窗口大小 | 1 | 3 | 5 | 9 | 12 | 15 |
|----------|---|---|---|---|----|----|
| 效果 | 微弱 | 中等 | 明显 | 明显 | 强 | 最强 |
| 分辨率 | 最高 | 高 | 中 | 中 | 低 | 最低 |

关键权衡：窗口越大效果越强但分辨率越低；小模型适合用小窗口（大窗口会阻断过大比例的层）。

### 关键发现

1. **通用模式**：所有模型（Transformer 和 Mamba）在中后层都依赖主语 token 向最后 token 的信息流——这可能是 LLM 的普遍特性
2. **Mamba 无首 token 偏置**：与 GPT-2 的强首 token attention sink 不同，Mamba-1/2 均不依赖首 token
3. **Mamba-1 独特的最后 token 依赖**：阻断最后 token 到自身的后层注意力反而**显著提升**正确概率（接近 1.0），这一现象在 Mamba-2 和 GPT-2 中不存在
4. **关系 token 依赖的架构差异**：GPT-2 在早期层依赖关系 token，Mamba 在后期层依赖，且表现出先降后升的特殊模式
5. **上下文相关特征主导 token 间通信**：敲除上下文相关特征几乎等同于敲除全部特征，上下文无关特征对事实传递几乎无贡献

## 亮点与洞察

- **跨架构可解释性迁移**：首次成功将 Transformer 的 Attention Knockout 完整迁移到 Mamba，证明了可解释性方法的通用性
- **特征敲除是原创贡献**：利用 SSM 独有的衰减参数结构，提供了 Transformer 中不存在的分析维度
- **"共性 vs 特性"的分类框架**：清晰地将信息流模式分为架构通用（主语→最后 token in 中后层）和架构特异（首 token 偏置、关系 token 时序等），对理解 LLM 原理有启发性
- **Mamba-1 的"自注意力阻断提升"现象**令人意外，暗示 Mamba-1 后层可能存在某种"信息过载"机制

## 局限与展望

1. 注意力敲除只能在连续层窗口中操作，无法捕捉更分散的信息流模式
2. 方法识别了关键连接，但未解析连接中实际传递的**信息内容**
3. 敲除效果不是生态的（即干预本身改变了网络行为），变化可能部分归因于干预本身
4. 仅关注 token 间信息流，无法解释 token 无关操作（如门控、卷积）的作用
5. 未解释为何不同架构会收敛到相似的信息流模式——归纳偏置的根源未探讨
6. COUNTERFACT 数据集仅 672 条，规模有限

## 相关工作与启发

- Geva et al. (2023) 奠定了 Transformer Attention Knockout 的基础框架，本文忠实继承并扩展
- Sharma et al. (2024) 先一步研究了 Mamba-1 的事实关联，但只做粗粒度阻断（单 token→所有 token），本文做细粒度（单→单）
- Meng et al. (2022) 的因果追踪（Causal Tracing）定位关键 MLP 层，与本文的 token 间信息流分析互补
- SSM-注意力等价理论（Ali et al., Dao & Gu）是本文方法论的理论基础

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将不可解释性方法跨架构迁移 + 原创的特征敲除机制，视角新颖
- **实验充分度**: ⭐⭐⭐⭐ — 覆盖 Mamba-1/2 多规模 + GPT-2/Llama/Mistral，窗口大小消融完整，大量可视化
- **写作质量**: ⭐⭐⭐⭐ — 图表丰富（10+ 张信息流热图），方法阐述清晰，共性与特性的对比框架组织得当
- **价值**: ⭐⭐⭐⭐ — 对理解新兴 SSM 架构内部机制有重要贡献，特征敲除为未来 SSM 剪枝和微调提供了方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Towards Effective Extraction and Evaluation of Factual Claims](towards_effective_extraction_and_evaluation_of_factual_claims.md)
- [\[ACL 2025\] Core: Robust Factual Precision with Informative Sub-Claim Identification](core_robust_factual_precision_with_informative_sub-claim_identification.md)
- [\[ACL 2025\] REVS: Unlearning Sensitive Information in Language Models via Rank Editing in the Vocabulary Space](revs_unlearning_sensitive_information_in_language_models_via_rank_editing_in_the.md)
- [\[ICML 2025\] Revealing Weaknesses in Text Watermarking Through Self-Information Rewrite Attacks](../../ICML2025/llm_safety/revealing_weaknesses_in_text_watermarking_through_self-information_rewrite_attac.md)
- [\[NeurIPS 2025\] Bits Leaked per Query: Information-Theoretic Bounds on Adversarial Attacks Against LLMs](../../NeurIPS2025/llm_safety/bits_leaked_per_query_information-theoretic_bounds_on_adversarial_attacks_agains.md)

</div>

<!-- RELATED:END -->
