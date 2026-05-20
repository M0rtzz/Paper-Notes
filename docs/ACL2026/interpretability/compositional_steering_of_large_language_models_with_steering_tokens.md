---
title: >-
  [论文解读] Compositional Steering of Large Language Models with Steering Tokens
description: >-
  [ACL 2026][可解释性][组合引导] 本文提出组合引导 token，通过自蒸馏将行为指令压缩为输入空间的嵌入向量，并训练专用组合 token <and> 来捕获"组合"的通用概念，在未见过的行为组合、未见过的行为以及未见过的组合数量上均展现强泛化能力。
tags:
  - "ACL 2026"
  - "可解释性"
  - "组合引导"
  - "引导token"
  - "自蒸馏"
  - "多行为控制"
  - "零样本组合"
---

# Compositional Steering of Large Language Models with Steering Tokens

**会议**: ACL 2026  
**arXiv**: [2601.05062](https://arxiv.org/abs/2601.05062)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 组合引导, 引导token, 自蒸馏, 多行为控制, 零样本组合

## 一句话总结

本文提出组合引导 token，通过自蒸馏将行为指令压缩为输入空间的嵌入向量，并训练专用组合 token <and> 来捕获"组合"的通用概念，在未见过的行为组合、未见过的行为以及未见过的组合数量上均展现强泛化能力。

## 研究背景与动机

**领域现状**：LLM 部署需要同时满足多个行为约束（如语言、长度、格式）。微调计算成本高且可能破坏通用能力，$N$ 种行为的任意组合意味着 $2^N$ 种微调。指令引导灵活但脆弱——语义等价的提示产生不一致的行为。

**现有痛点**：(1) 激活空间引导方法（如 CAA）通过向量加法组合行为，但直接组合独立训练的模块具有破坏性；(2) Gist token 仅处理单行为压缩，未解决组合问题；(3) 现有组合引导缺乏严格评估——大多仅提供轶事证据或缺少基线比较。

**核心矛盾**：独立训练的行为表示在组合时产生干扰，但如果为每种组合单独训练则组合爆炸。需要一种表示方式能学习"组合"本身的概念，而非每种特定组合。

**本文目标**：学习一个通用的组合 token <and>，使其在未见过的行为组合上泛化，包括未见过的行为和未见过的组合数量。

**切入角度**：行为表示放在输入空间（而非激活空间），支持更好的零样本组合；训练组合 token 时冻结行为 token，迫使 <and> 学习行为无关的组合函数。

**核心 idea**：组合 = 一个可学习的通用操作符，而非针对每种行为对的特定调整。

## 方法详解

### 整体框架

两阶段训练：(1) 通过自蒸馏独立训练每个行为的引导 token <b>——教师接收指令文本，学生接收引导 token，最小化 KL 散度；(2) 冻结 LLM 和行为 token，在行为对组合上训练组合 token <and>——教师接收两条指令，学生接收 <bi><and><bj>。推理时组合为 $[\mathbf{E}_x, \mathbf{e}_{b_i}, \mathbf{e}_{\text{<and>}}, \mathbf{e}_{b_j}]$。

### 关键设计

1. **输入空间引导 token**:

    - 功能：将行为指令压缩为单个输入嵌入向量
    - 核心思路：引导 token $\mathbf{e}_b \in \mathbb{R}^d$ 存在于模型输入嵌入空间（而非中间激活空间），通过自蒸馏训练：最小化 $\text{KL}(P_{\text{teacher}}(y|x, I_b) \| P_{\text{student}}(y|x, \texttt{<b>}))$。使用 10 种指令改写防止过拟合
    - 设计动机：输入空间的行为表示比激活空间更适合组合——激活空间的直接加法容易产生干扰

2. **通用组合 token <and>**:

    - 功能：学习行为无关的"组合"概念
    - 核心思路：在行为对上训练 $\mathbf{e}_{\text{<and>}}$，关键在于冻结行为 token——确保 <and> 学习的是组合函数本身而非修改个别行为。零初始化（不偏向任何行为）+ 正交正则化（防止坍缩到行为 token 空间）
    - 设计动机：如果 <and> 在训练中能修改行为 token，则只会学到特定组合的调整而非通用组合能力

3. **正交正则化**:

    - 功能：防止组合 token 与行为 token 表示坍缩
    - 核心思路：$\mathcal{L}_{\text{orth}} = \sum_{b \in \mathcal{B}_{\text{seen}}} (\frac{\mathbf{e}_{\text{<and>}} \cdot \mathbf{e}_b}{\|\mathbf{e}_{\text{<and>}}\| \cdot \|\mathbf{e}_b\|})^2$，最终损失 $\mathcal{L} = \mathcal{L}_{\text{dist}} + \lambda \cdot \mathcal{L}_{\text{orth}}$
    - 设计动机：消融实验证明正交正则化对零样本组合泛化至关重要

### 损失函数 / 训练策略

自蒸馏损失 + 正交正则化。LLM 完全冻结，仅学习 $|\mathcal{B}|+1$ 个 $d$ 维向量。温度 $T=10.0$ 鼓励匹配完整概率分布。行为 token 语义初始化（指令 token 嵌入的均值），组合 token 零初始化。

## 实验关键数据

### 主实验

**Qwen3-8B 上的 2-行为组合准确率（%）**

| 方法 | 已见组合 | 未见组合 | 顺序方差↓ |
|------|---------|---------|----------|
| CAA（激活引导） | 1.6 | 0.5 | - |
| LM-Steer | 18.1 | 13.4 | - |
| LoRA DARE | 81.5 | 44.8 | - |
| 指令引导 | 86.2 | 67.3 | 12.3 |
| **引导 token** | **90.5** | **75.5** | **4.8** |
| **引导 token + 指令** | **92.0** | **80.3** | **3.5** |

### 消融实验

| 配置 | 已见组合 | 未见组合 | 说明 |
|------|---------|---------|------|
| 无 <and> token（直接拼接） | 下降 | 大幅下降 | 组合 token 关键 |
| 无正交正则化 | 轻微下降 | 明显下降 | 正交性对泛化重要 |
| 随机初始化 <and> | 轻微下降 | 下降 | 零初始化更优 |
| 仅 2-行为训练 | - | 泛化到 3-行为 | 组合概念可泛化 |

### 关键发现

- 引导 token 在已见和未见组合上均大幅超越激活引导方法（CAA: 1.6% vs 引导 token: 90.5%）
- 组合 token 成功泛化到：未见过的行为组合、包含未见行为的组合、3-行为组合（仅训练了 2-行为）
- 引导 token + 指令的混合方法在所有设置下最优，两者具有互补性
- 组合准确率和鲁棒性随模型规模增长而提升（4B → 8B → 14B）
- 引导 token 的顺序方差远低于指令引导，说明行为表示更稳定

## 亮点与洞察

- "学习组合操作符而非每种组合"的思路简洁有力——类似于学习函数 vs 记忆表格
- 冻结行为 token 训练 <and> 是关键设计决策，确保了泛化而非过拟合
- 引导 token 与指令的互补性令人惊喜——压缩表示和自然语言提供不同类型的控制信号

## 局限与展望

- 仅在可自动验证的约束上评估（长度、格式、语言），主观行为（如风格、语气）未覆盖
- 每个行为需要独立训练引导 token，行为数量增长时训练成本线性增加
- 组合 token 仅训练了 2-行为组合，更多行为的组合效果可能下降
- 依赖自蒸馏质量——如果教师（指令引导）本身不遵循指令，学生也无法学好

## 相关工作与启发

- **vs CAA/Rimsky et al.**: 激活空间引导在组合时干扰严重（1.6%），输入空间引导 token 完胜
- **vs Gist token**: Gist token 仅压缩单指令，未解决组合问题
- **vs LoRA merging**: LoRA DARE 在已见组合上有竞争力（81.5%）但在未见组合上泛化差（44.8%）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 通用组合操作符的概念和冻结训练设计独创性强
- 实验充分度: ⭐⭐⭐⭐⭐ 七个模型、15 种行为、多种组合设置、100万+评估，极其全面
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，方法优雅，实验设计严谨
- 价值: ⭐⭐⭐⭐⭐ 为多行为可控生成提供了简洁有效的新范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] FineSteer: A Unified Framework for Fine-Grained Inference-Time Steering in Large Language Models](finesteer_a_unified_framework_for_fine-grained_inference-time_steering_in_large_.md)
- [\[ACL 2026\] From Weights to Activations: Is Steering the Next Frontier of Adaptation?](from_weights_to_activations_is_steering_the_next_frontier_of_adaptation.md)
- [\[CVPR 2026\] Language Models Can Explain Visual Features via Steering](../../CVPR2026/interpretability/language_models_can_explain_visual_features_via_steering.md)
- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](knowledge_vector_of_logical_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](tracing_relational_knowledge_recall_in_large_language_models.md)

</div>

<!-- RELATED:END -->
