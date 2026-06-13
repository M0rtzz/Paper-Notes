---
title: >-
  [论文解读] Theorem Prover as a Judge for Synthetic Data Generation
description: >-
  [ACL 2025][LLM 其他][定理证明器] 提出 TP-as-a-Judge 框架，利用 Lean 定理证明器验证 LLM 生成的中间推理步骤，结合迭代自动形式化和基于定理证明器反馈的强化学习（RLTPF），仅用 3,508 个样本就在多个数学推理基准上取得了显著提升。 合成数据在增强 LLM 数学推理能力方面潜力巨…
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "定理证明器"
  - "合成数据"
  - "自动形式化"
  - "强化学习"
  - "数学推理"
---

# Theorem Prover as a Judge for Synthetic Data Generation

**会议**: ACL 2025  
**arXiv**: [2502.13137](https://arxiv.org/abs/2502.13137)  
**代码**: [GitHub](https://github.com/joshuaongg21/RLTPF)  
**领域**: 其他  
**关键词**: 定理证明器, 合成数据, 自动形式化, 强化学习, 数学推理

## 一句话总结

提出 TP-as-a-Judge 框架，利用 Lean 定理证明器验证 LLM 生成的中间推理步骤，结合迭代自动形式化和基于定理证明器反馈的强化学习（RLTPF），仅用 3,508 个样本就在多个数学推理基准上取得了显著提升。

## 研究背景与动机

合成数据在增强 LLM 数学推理能力方面潜力巨大，但确保中间推理步骤（而非仅最终答案）的正确性仍然是一个关键挑战。现有方法存在以下问题：

1. **LLM-as-a-Judge 的偏差**：用 LLM 自身评判推理质量容易引入偏差，且无法严格验证逻辑正确性
2. **人工标注成本高**：为逐步推理标注训练数据需要大量人力，自动标注方法因噪声奖励信号效果有限
3. **自动形式化错误率高**：虽然定理证明器能有效验证推理，但将自然语言证明转化为形式化表示（如 Lean）的过程容易出错，初始执行率仅约 60%
4. **MCTS 等方法计算开销大**：蒙特卡洛树搜索虽能提升推理质量，但大量 rollout 限制了扩展性

本文的核心动机是：通过定理证明器替代人类标注者，为合成数据提供严格的逻辑验证，同时通过迭代形式化解决执行失败问题。

## 方法详解

### 整体框架

TP-as-a-Judge 包含三个关键阶段：(1) LLM 数据生成，通过 Reverse Question-Answering 方法生成数学问题和答案；(2) 定理证明器验证，通过 Lean 证明器验证问题和答案的形式化表示；(3) RLTPF，利用验证结果进行 SFT 和 DPO 训练。

### 关键设计

1. **问题形式化（四阶段验证）**：将原始问题 s₀ 经过四个阶段处理——(s₁) 用 CoMAT 转为符号表示，(s₂) 自动形式化并由定理证明器验证，(s₃) Auto-Informalisation 将形式化结果翻译回自然语言，(s₄) Alignment Check 通过 gpt-4o 检查原始问题与翻译回的自然语言是否一致。设计动机是解决问题自动形式化缺乏正确性评估指标的难题。

2. **答案形式化与迭代自动形式化**：答案的每个推理步骤都被形式化并由定理证明器验证，产生三种结果——verified、false、error。对于 error 结果，提出迭代自动形式化方法，将错误信息反馈给模型进行修正，最多迭代 5 次。这种方法将 Lean 证明器的执行率从 60% 提升到 87%。核心思路是模拟人类专家多次尝试形式化复杂问题的过程。

3. **RLTPF（基于定理证明器反馈的强化学习）**：用定理证明器替代人类标注者，根据两个 LLM 的验证结果分配数据——双方验证通过用于 SFT，一方正确一方错误用于 DPO，双方都错误则丢弃。设计动机是让模型同时从正确推理模式和需要改进的案例中学习。

### 损失函数 / 训练策略

- SFT 阶段使用标准负对数似然损失训练验证通过的数据
- DPO 阶段使用 Direct Preference Optimization，以验证通过的解答为正例、验证失败的为负例
- 训练使用 LoRA 微调以提高计算效率
- 先进行 SFT 再进行 DPO 的顺序训练

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 (RLTPF) | CoT 基线 | 提升 |
|--------|------|-------------|---------|------|
| MultiArith | Acc (Llama-3.1-8B) | 97.78% | 97.78% | +0.00% |
| SVAMP | Acc (Llama-2-7B) | 44.00% | 38.00% | +6.00% |
| GSM8K | Acc (Llama-3.1-8B) | 83.75% | 82.38% | +1.37% |
| GSM-Symbolic | Acc (Llama-3.2-3B) | 59.60% | 55.80% | +3.80% |
| AQUA | Acc (Llama-3.1-8B) | 57.09% | 53.54% | +3.55% |
| MultiArith | Acc (Mistral-7B) | 67.78% | 62.22% | +5.56% |
| AIME 2024 | Acc (Llama-3.2-3B) | 13.33% | 10.00% | +3.33% |

### 消融实验

| 配置 | GSM8K Acc | 说明 |
|------|-----------|------|
| SFT (All Instances) | 82.23% | 使用所有实例 |
| SFT (Only Rejected) | 80.87% | 只用拒绝的实例 |
| SFT (Only Verified) | 81.55% | 只用验证通过的实例 |
| SFT (All) + RLTPF | 79.60% | 全实例 SFT + DPO 冲突 |
| SFT (Verified) + RLTPF | **83.75%** | 最优配置 |

### 关键发现

- TP-as-a-Judge 的 F1 达到 0.87，Recall 0.91，而 o1-mini 作为 LLM-as-a-Judge 的 F1 仅 0.72，False Positive 是前者的 2 倍
- 迭代自动形式化中约 40% 的样本需要迭代修正，大多数在第 3 次迭代内成功
- 迭代次数与 token 序列长度正相关，更复杂的推理需要更多迭代
- 仅用 3,508 个样本，在 GSM8K 和 AIME 上与使用数万至数百万样本的模型（如 OpenMath-2）表现接近
- gpt-4o 在自生成的合成问题上准确率仅 51.85%，说明自生成问题的挑战性

## 亮点与洞察

- **数据效率极高**：3,508 个样本达到了大规模数据集微调的竞争性结果，证明了中间推理步骤质量的重要性远超数据量
- **形式化验证的优势**：定理证明器提供的是逻辑级别的严格验证，比 LLM 自判断更可靠，False Positive 减半
- **迭代修正的自然性**：模拟人类专家多次尝试形式化的过程，90% 以上的修正在 3 次迭代内完成
- **RLTPF 的数据分配策略**：SFT + DPO 的组合利用方式巧妙，验证通过和失败的数据各尽其用

## 局限与展望

- 目前仅适用于数学推理领域，扩展到其他领域（如代码验证、逻辑推理）是开放问题
- 数学覆盖范围限于代数、计数概率和问题求解，缺少几何和微积分
- 数据集复杂度受限于 LLM 能可靠解决的问题难度
- 迭代形式化的计算开销较大，多轮修正增加了成本
- 仅在 ≤8B 参数的模型上实验，更大模型的扩展性未知

## 相关工作与启发

- 与 NuminaMath（860k 样本）和 OpenMath-2（14M 样本）相比，本文用 3,508 个样本就达到了竞争性结果，说明数据质量 >> 数据量
- 与 rStar-Math（MCTS）相比，TP-as-a-Judge 避免了大量 rollout 的计算开销
- 迭代自动形式化思路可启发代码生成中的编译器反馈迭代修正

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 4 | 定理证明器 + 合成数据 + RLHF 的新颖结合 |
| 实用性 | 3 | 受限于数学领域和 Lean 证明器生态 |
| 实验充分度 | 4 | 多模型、多基准、详细消融和分析 |
| 写作质量 | 4 | 结构清晰，方法描述详细 |
| 总分 | 4 | 有意义的工作，证明了形式化验证在合成数据质量控制中的价值 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] TARGA: Targeted Synthetic Data Generation for Practical Reasoning over Structured Data](targa_targeted_synthetic_data_generation_for_practical_reasoning_over_structured.md)
- [\[ACL 2025\] PersonaBench: Evaluating AI Models on Understanding Personal Information through Accessing (Synthetic) Private User Data](personabench_evaluating_ai_models_on_understanding_personal_information_through_.md)
- [\[ACL 2025\] Generating Synthetic Relational Tabular Data via Structural Causal Models](generating_synthetic_relational_tabular_data_via_structural_causal_models.md)
- [\[ACL 2025\] KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding](kodcode_a_diverse_challenging_and_verifiable_synthetic_dataset_for_coding.md)
- [\[ACL 2025\] Literature Meets Data: A Synergistic Approach to Hypothesis Generation](literature_meets_data_hypothesis.md)

</div>

<!-- RELATED:END -->
