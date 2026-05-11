---
title: >-
  [论文解读] Large Language and Reasoning Models are Shallow Disjunctive Reasoners
description: >-
  [ACL 2025][LLM推理][系统性推理] 本文利用合成的空间与时间推理基准（STaR）评估LLM和LRM在需要组合多条推理路径的析取规则推理任务上的系统性泛化能力，发现即使是o3-mini等推理模型也只能处理单路径推理，在多路径析取推理场景中性能急剧下降。
tags:
  - ACL 2025
  - LLM推理
  - 系统性推理
  - 析取规则
  - 空间时间推理
  - 大推理模型
  - OOD泛化
---

# Large Language and Reasoning Models are Shallow Disjunctive Reasoners

**会议**: ACL 2025  
**arXiv**: [2503.23487](https://arxiv.org/abs/2503.23487)  
**代码**: 无  
**领域**: LLM推理  
**关键词**: 系统性推理, 析取规则, 空间时间推理, 大推理模型, OOD泛化

## 一句话总结

本文利用合成的空间与时间推理基准（STaR）评估LLM和LRM在需要组合多条推理路径的析取规则推理任务上的系统性泛化能力，发现即使是o3-mini等推理模型也只能处理单路径推理，在多路径析取推理场景中性能急剧下降。

## 研究背景与动机

**领域现状**：近年来LLM在数学和编程任务上展现了强大能力，基于强化学习后训练的"大推理模型"（LRM）如DeepSeek-R1、o3-mini等通过链式思维进一步提升了分析推理能力。学术界对这些模型的推理能力寄予厚望。

**现有痛点**：然而现有评测主要集中在数学和编程领域，这些领域容易受到数据集污染和记忆训练数据的影响。更重要的是，这些任务往往只需要沿单条推理链（Horn规则）推导即可得到答案，无法检验模型在需要组合多条推理路径的复杂推理场景下的真实能力。GSM8k、MMLU等静态基准也存在泄露到训练语料的风险。

**核心矛盾**：链式思维（CoT）天然适配单推理链的Horn规则问题，但现实世界许多推理任务涉及**析取规则**——即从已知事实只能推出一组候选结论的析取（或），需要组合多条路径的信息才能锁定唯一答案。现有评测无法区分模型是真正在推理，还是在利用浅层模式匹配。

**本文目标**：在精确可控的合成基准上系统评估LLM和LRM的析取推理能力，特别量化其在分布外（OOD）泛化场景下的表现。

**切入角度**：作者选择定性空间推理（RCC-8）和时间推理（Interval Algebra）作为测试域。这两个域的组合规则可以用析取规则精确描述，问题难度通过路径长度$k$和路径数$b$严格控制，且多项式时间可解——理论上在LRM的能力范围内。

**核心 idea**：利用STaR基准的可控难度参数，揭示LLM/LRM只是"浅层析取推理者"——能处理单推理路径但无法有效组合多条路径。

## 方法详解

### 整体框架

本文不提出新方法，而是设计了系统性的评测框架。输入是一个有向标记图$\mathcal{G}$，顶点代表实体，边标注RCC-8或IA关系。目标是推断指定头尾实体之间的唯一关系。模型需要：(1) 理解组合表中每对关系组合的候选结果；(2) 沿每条路径执行链式推理获得候选关系集；(3) 取所有路径结果的交集得到唯一答案。

### 关键设计

1. **STaR基准的难度控制机制**:

    - 功能：通过两个参数精确控制问题难度
    - 核心思路：$k$控制每条简单路径的长度（需要$k-1$次规则组合），$b$控制头尾实体之间独立路径的数量（需要$b$条路径的析取结果取交集）。训练集为$k \in \{2,3,4\}, b \in \{1,2,3\}$，测试集扩展到$k \in \{2,...,10\}, b \in \{1,2,3,4\}$
    - 设计动机：确保OOD测试集在结构上比训练集更复杂，同时保证解题所需的所有原子知识在训练集中都已出现

2. **三种评测设置（Zero-shot / Few-shot / Fine-tuned）**:

    - 功能：分别评估模型的指令跟随能力、上下文学习能力和学习后的泛化能力
    - 核心思路：Zero-shot和Few-shot直接通过prompt提供组合表（整数编码），Fine-tuned使用完整训练集（RCC-8约57600条，IA约93400条）进行微调
    - 设计动机：区分模型是否能从prompt中的规则进行推理（zero/few-shot），以及能否从数据中学习并泛化（fine-tuned）

3. **LRM专项分析（o3-mini vs 蒸馏R1模型）**:

    - 功能：评估推理模型在析取推理上的特殊表现模式
    - 核心思路：对o3-mini和Qwen R1蒸馏模型在相同配置下进行zero-shot评测，并分析输出token数量与问题难度的关系
    - 设计动机：验证LRM的CoT机制在多路径析取推理中是否有效，以及思考时间分配是否合理

### 损失函数 / 训练策略

微调使用QLoRA（4-bit量化），AdamW优化器（学习率$2 \times 10^{-4}$），gradient accumulation步数4，仅训练1个epoch。LoRA适配器应用于Q/K/V/O/Gate/Up/Down投影。

## 实验关键数据

### 主实验

| 模型 | 设置 | RCC-8 (k=9,b=1) Acc | RCC-8 (k=9,b=2) Acc | RCC-8 (k=9,b=3) Acc |
|------|------|---------------------|---------------------|---------------------|
| o3-mini | Zero-shot | 0.90 | 0.48 | 0.30 |
| Qwen2.5-7B (R1) | Zero-shot | 0.08 | 0.06 | 0.12 |
| Qwen2.5-14B (R1) | Zero-shot | 0.07 | 0.02 | 0.07 |
| Qwen2.5-14B | Fine-tuned | ~0.40 | ~0.35 | ~0.35 |
| Qwen2.5-72B | Zero-shot | ~0.15 | ~random | ~random |

### 消融实验（Fine-tuned模型按关系类别分析，k=9, b=2）

| 关系 | Qwen2.5-14B F1 | o3-mini F1 | 说明 |
|------|----------------|------------|------|
| EQ | 1.00 | 0.60 | 微调模型靠"EQ链"捷径完美预测，o3-mini反而不完美 |
| NTPPI | 0.81 | 0.33 | 微调模型利用可学习规则 |
| NTPP | 0.00 | 0.33 | 微调模型完全失败，o3-mini有基本能力 |
| PO | 0.16 | 0.33 | 需要真正析取推理的难关系 |

### 关键发现

- o3-mini在$b=1$（单路径）时准确率达0.90，但$b \geq 2$时急剧下降至0.48/0.30，说明CoT机制只适配单路径推理
- 微调LLM的表现质量上不同于o3-mini：它们通过学到简单规则（如EQ链传递）在部分关系上达到完美准确率，但在需要真正析取推理的关系上彻底失败
- R1蒸馏模型在所有$b \geq 2$的设置中低于随机猜测水平，表现极差
- Qwen系列模型在微调后一致优于Llama系列，即使是7B的Qwen也优于70B的Llama-3.3

## 亮点与洞察

- **合成基准的精准控制**：$k$和$b$两个参数即可精确控制推理链长度和析取复杂度，让模型的推理能力边界清晰可测。这种设计思路可迁移到任何需要评测组合泛化的场景。
- **LRM vs Fine-tuned LLM的"正交"失败模式**：o3-mini靠系统性规则应用做推理但容易犯错（均匀地在各关系上有非零表现），微调模型靠捷径在简单关系上完美但在复杂关系上完全失败。这一洞察对理解不同训练范式下模型的推理机制极具价值。
- **CoT token分配的反直觉发现**：R1模型在$b$增大时反而生成更少思考token，似乎在多路径场景下"放弃"了推理——这暗示当前CoT机制缺乏对问题结构的深层理解。

## 局限与展望

- 仅使用RCC-8和IA两个定性推理域，泛化到其他析取推理任务未知
- 受算力限制，推理模型仅在少量$(k,b)$配置上评测
- 较大模型（如o3、Claude等）可能表现更好，但未纳入评测
- 未探索针对析取推理的专门训练策略，例如在训练时引入多路径交集的显式监督

## 相关工作与启发

- **vs CLUTRR（家族关系推理）**: CLUTRR只涉及Horn规则的链式推理，STaR增加了析取维度，更具挑战性
- **vs Path-of-Thoughts**: 后者通过图结构提取帮助LLM处理关系推理，但未涉及析取规则组合
- 本文的评测框架可作为未来评测推理模型析取能力的标准基准

## 评分

- 新颖性: ⭐⭐⭐⭐ 析取推理视角是对LRM评测的重要补充，但实验设计是对已有STaR基准的直接应用
- 实验充分度: ⭐⭐⭐⭐ 覆盖11个模型、3种设置、细粒度分析，但缺少更多最新模型
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，从问题定义到实验分析逻辑链完整
- 价值: ⭐⭐⭐⭐ 对理解LRM推理局限性有重要启发，但缺乏改进方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models](chain_of_reasoning_unified_math.md)
- [\[ACL 2025\] Ranked Voting based Self-Consistency of Large Language Models](ranked_voting_based_self-consistency_of_large_language_models.md)
- [\[ACL 2025\] Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models via a Multi-Paradigm Perspective](chain_of_reasoning_unified_math.md)
- [\[ACL 2025\] Can Large Language Models Detect Errors in Long Chain-of-Thought Reasoning?](can_large_language_models_detect_errors_in_long_chain-of-thought_reasoning.md)
- [\[ACL 2026\] Large Reasoning Models Are (Not Yet) Multilingual Latent Reasoners](../../ACL2026/llm_reasoning/large_reasoning_models_are_not_yet_multilingual_latent_reasoners.md)

</div>

<!-- RELATED:END -->
