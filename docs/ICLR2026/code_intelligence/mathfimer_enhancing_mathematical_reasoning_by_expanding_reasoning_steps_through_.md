---
title: >-
  [论文解读] MathFimer: Enhancing Mathematical Reasoning by Expanding Reasoning Steps through Fill-in-the-Middle Task
description: >-
  [ICLR 2026][数学推理] 借鉴代码补全中的 Fill-in-the-Middle (FIM) 范式，训练一个专门的步骤扩展模型 MathFimer-7B，在已有数学解题链中插入更细粒度的中间推理步骤，从而系统性提升下游模型的数学推理能力。
tags:
  - ICLR 2026
  - 数学推理
  - Fill-in-the-Middle
  - 推理步骤扩展
  - Chain-of-Thought
  - 数据增强
---

# MathFimer: Enhancing Mathematical Reasoning by Expanding Reasoning Steps through Fill-in-the-Middle Task

**会议**: ICLR 2026  
**arXiv**: [2502.11684](https://arxiv.org/abs/2502.11684)  
**代码**: 无  
**领域**: LLM Reasoning  
**关键词**: 数学推理, Fill-in-the-Middle, 推理步骤扩展, Chain-of-Thought, 数据增强

## 一句话总结

借鉴代码补全中的 Fill-in-the-Middle (FIM) 范式，训练一个专门的步骤扩展模型 MathFimer-7B，在已有数学解题链中插入更细粒度的中间推理步骤，从而系统性提升下游模型的数学推理能力。

## 研究背景与动机

Chain-of-Thought (CoT) 已是 LLM 数学推理的主流范式，但训练数据中推理步骤的质量和粒度直接制约模型性能。已有研究 (Jin et al., 2024) 表明更详细的中间步骤可显著提升推理准确率。然而，现有的步骤扩展方法存在三大问题：

**依赖更强模型**：需要用更大的外部模型来生成更好的步骤，形成"越大越好"的循环依赖

**计算成本高昂**：像 MCTS 这样的搜索策略在探索推理路径时需要大量算力

**可靠性不足**：这些方法通常生成全新的推理链，而非在已有的人工验证步骤基础上进行增强，可能引入新的错误

核心研究问题：能否开发一种更高效、更可靠的方法来扩展推理步骤，同时保留已有人工生成解答的有效性？

## 方法详解

### 整体框架

MathFimer 框架分为两个阶段：(1) 训练 FIM 模型；(2) 使用训练好的模型扩展已有数据集的推理步骤。灵感来源于代码补全中的 Fill-in-the-Middle 任务——给定代码的前缀和后缀，模型补全中间缺失部分。

### 关键设计

1. **NuminaMath-FIM 数据集构建**：

    - 基于 NuminaMath-CoT（853K 数学问答对）进行步骤分解
    - 对每个样本的解答 $Y = \{y_1, y_2, ..., y_n\}$，随机选取步骤 $y_i$，将 $y_1...y_{i-1}$ 作为前缀 P，$y_{i+1}...y_n$ 作为后缀 S，$y_i$ 作为待预测的中间步骤 M
    - 每个样本进行 3 轮随机采样，最终得到 **2.5M** 训练样本
    - 采用 PSM（Prefix-Suffix-Middle）序列格式，使用 `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>` 三个特殊 token

2. **MathFimer-7B 训练**：

    - 基座模型：Qwen2.5-Math-7B（数学专用模型）
    - 训练方式：SFT，仅对 `<|fim_middle|>` 之后的 token 计算 loss
    - 核心公式：$\text{FIM}(Q, P, S) \Rightarrow M$

3. **步骤扩展策略**：

    - 对原始解答中每对连续步骤之间执行 FIM 推理：$\hat{y}_i = \text{FIM}(Q, y_1...y_{i-1}, y_i...y_n)$
    - **相似度过滤**：计算生成步骤 $\hat{y}_i$ 与原始下一步 $y_i$ 的序列相似度，设置阈值 $\eta = 0.8$，相似度大于阈值的标记为无效（说明原始步骤已足够详细）
    - 仅插入相似度低于阈值的新步骤

### 损失函数 / 训练策略

- 标准的交叉熵损失，仅作用于 middle 部分的 token
- 使用 Megatron-LM 框架进行 SFT
- max_length = 8192，全局 batch size = 128
- 学习率 1e-5
- 所有训练样本进行 packing 以加速训练
- 在 64 块 Ascend H910B-64G 上训练

## 实验关键数据

### 主实验

在 4 个基座模型 × 5 个数据集上进行了全面实验：

| 基座模型 | 数据集 | GSM8K | MATH | Odyssey | OB-EN | 平均提升 |
|---------|--------|-------|------|---------|-------|---------|
| Llama-3.1-8B | MathInstruct-CoT | 67.78→75.21 | 18.74→22.90 | 22.11→24.42 | 2.37→3.56 | **+3.77** |
| Llama-3.1-70B | MathInstruct-CoT | 89.31→90.98 | 41.96→44.72 | 36.50→39.33 | 9.19→12.15 | **+2.56** |
| Qwen2.5-Math-7B | MetaMathQA | 93.18→93.10 | 70.22→79.08 | 49.10→52.70 | 34.81→41.04 | **+4.65** |
| Qwen2.5-Math-72B | MetaMathQA | 90.22→92.95 | 57.68→63.40 | 42.93→47.30 | 20.00→24.89 | **+4.43** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 蒸馏数据 vs 原始数据 | G+M 原始 +5.61 vs 蒸馏 +1.13 (GSM8K) | 部分提升来自模型蒸馏，但 FIM 步骤扩展本身也提供额外增益 |
| 迭代扩展 (1→3轮) | MI-CoT: 67.78→75.21→80.21→83.32 (GSM8K) | 迭代扩展持续提升性能，3轮提升 +15.54% |
| 模型规模 (7B vs 72B) | 7B: +7.43 vs 72B: +6.14 (GSM8K, MI-CoT) | 7B 和 72B 模型性能差距很小，说明步骤扩展不依赖大模型 |

### 关键发现

1. **普适性强**：MathFimer 在通用模型 (Llama) 和数学专用模型 (Qwen-Math) 上都有效，在 4 个基座模型和 5 个数据集上均观察到一致的提升
2. **迭代可扩展**：多轮步骤扩展可以持续累积改进效果，GSM8K 上 3 轮迭代可达 +15.54%
3. **小模型即可**：MathFimer-7B 和 72B 的扩展效果非常接近，表明步骤扩展任务不需要大模型容量
4. **数学专用模型受益更大**：Qwen2.5-Math 系列在 MetaMathQA 和 ScaleQuest 上获得了最大提升（MATH +8.86%）

## 亮点与洞察

- **巧妙的类比迁移**：将代码补全中成熟的 FIM 范式迁移到数学推理步骤扩展，思路新颖且有效
- **保留原始结构**：不像其他方法重新生成整条推理链，MathFimer 在保留原始步骤的基础上进行插入，更可靠
- **相似度过滤机制**：自动判断某位置是否需要扩展，避免冗余插入
- **实用性强**：无需外部更强模型，一个 7B 模型即可完成扩展
- **数据资产**：开源了 NuminaMath-FIM 数据集（2.5M 样本）和 MathFimer-7B 模型

## 局限与展望

1. **领域泛化性未知**：目前仅验证了数学推理，是否适用于代码推理、逻辑推演、常识推理等领域尚不清楚
2. **生成可靠性**：缺乏对插入步骤的逻辑一致性和数学正确性的自动验证机制，多轮迭代可能导致错误累积
3. **方法论局限**：主要扩展已有解题模式，对极其复杂或非常规问题的新解法探索有限
4. **未探索自适应扩展**：当前对所有步骤对统一执行扩展，缺乏对"哪些步骤真正需要扩展"的智能判断

## 相关工作与启发

- CoT Prompting (Wei et al., 2023) 和 Self-Consistency (Wang et al., 2023) 是推理增强的基础
- Jin et al. (2024) 证明了推理步骤长度对 LLM 的重要影响
- MCTS 方法 (Chen et al., 2024; Guan et al., 2025) 走的是搜索最优路径的方向
- 与 OpenMathInstruct (Toshniwal et al., 2024) 不同，MathFimer 不依赖强模型生成新数据
- **启发**：FIM 范式可能可以推广到任何需要"补全中间步骤"的序列任务

## 评分
- 新颖性: ⭐⭐⭐⭐ （FIM 迁移到数学推理是新颖的，但本质上是数据增强）
- 实验充分度: ⭐⭐⭐⭐⭐ （4 个基座模型 × 5 个数据集 × 4 个 benchmark，非常全面）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，实验详实）
- 价值: ⭐⭐⭐⭐ （方法简单实用，可直接应用于现有数据集的增强）

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] FractalBench: Diagnosing Visual-Mathematical Reasoning Through Recursive Program Synthesis](../../NeurIPS2025/code_intelligence/fractalbench_diagnosing_visual-mathematical_reasoning_through_recursive_program_.md)
- [\[ICLR 2026\] The Limits of Long-Context Reasoning in Automated Bug Fixing](the_limits_of_long-context_reasoning_in_automated_bug_fixing.md)
- [\[ICLR 2026\] Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning](supervised_reinforcement_learning_from_expert_trajectories_to_step-wise_reasonin.md)
- [\[ICML 2025\] AdaptiveStep: Automatically Dividing Reasoning Step through Model Confidence](../../ICML2025/code_intelligence/adaptivestep_automatically_dividing_reasoning_step_through_model_confidence.md)
- [\[ICML 2025\] Reasoning Through Execution: Unifying Process and Outcome Rewards for Code Generation](../../ICML2025/code_intelligence/reasoning_through_execution_unifying_process_and_outcome_rewards_for_code_genera.md)

<!-- RELATED:END -->
