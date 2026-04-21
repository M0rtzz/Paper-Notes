---
title: >-
  [论文解读] WarriorCoder: Learning from Expert Battles to Augment Code Large Language Models
description: >-
  [ACL 2025][LLM/NLP][code LLM] 提出 WarriorCoder，通过构建多个专家代码 LLM 之间的竞技场（arena），让攻击者用自身擅长的领域挑战防御者，由裁判评估后用胜者回答训练目标模型，从而无需依赖专有模型或预存数据集即可从零生成高质量、高多样性的代码训练数据，实现 SOTA 性能。
tags:
  - ACL 2025
  - LLM/NLP
  - code LLM
  - data flywheel
  - expert battles
  - Elo rating
  - instruction mining
---

# WarriorCoder: Learning from Expert Battles to Augment Code Large Language Models

**会议**: ACL 2025  
**arXiv**: [2412.17395](https://arxiv.org/abs/2412.17395)  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: code LLM, data flywheel, expert battles, Elo rating, instruction mining

## 一句话总结
提出 WarriorCoder，通过构建多个专家代码 LLM 之间的竞技场（arena），让攻击者用自身擅长的领域挑战防御者，由裁判评估后用胜者回答训练目标模型，从而无需依赖专有模型或预存数据集即可从零生成高质量、高多样性的代码训练数据，实现 SOTA 性能。

## 研究背景与动机

**领域现状**：代码 LLM 的性能强烈依赖高质量微调数据，但数据收集和标注成本高昂。

**现有痛点**：现有数据飞轮方法（如 WizardCoder、Magicoder、WaveCoder）依赖现成数据集和有限的专有 LLM（GPT-4 等）进行数据增强，导致数据多样性不足且引入系统性偏差。

**核心矛盾**：开源代码专家 LLM 虽然能力强，但训练数据不公开，无法直接利用；而使用少量专有 LLM 扩展数据又限制了数据源的多样性。

**本文目标** 设计一种不依赖专有 LLM、不依赖预存数据集的数据生成范式，能自动从多个专家 LLM 中提取知识并整合其优势。

**切入角度**：借鉴对战竞技场的思想（如 LMSYS Chatbot Arena），让专家 LLM 互相挑战，目标模型从胜者中学习。

**核心 idea**：构建代码专家的竞技场，让模型在对战中暴露各自长处，目标模型通过学习每场比赛的胜者回答来融合所有专家优势。

## 方法详解

### 整体框架
1. 选定 5 个顶级开源代码专家（Athene-V2-Chat、DeepSeek-Coder-V2-Lite-Instruct、Llama-3.3-70B-Instruct、Qwen2.5-72B-Instruct、QwQ-32B-Preview）；
2. 每轮竞技中，一个模型作为攻击者（attacker），另一个作为防御者（defender），其余作为裁判；
3. 攻击者利用 completion-based 方法挖掘自身擅长的指令，用来挑战防御者；
4. 双方各自回答，裁判投票评判；
5. 结合本地投票比例和全局 Elo Rating 选择胜者回答加入训练集；
6. 目标模型（DeepSeekCoder-Base-6.7B）在收集到的数据上做 SFT。

### 关键设计

1. **Completion-based 指令挖掘**

    - 受 Magpie 启发，将聊天模板的前缀（system prompt + user 标签）输入专家 LLM，让其自动补全生成用户指令
    - 这些指令直接从模型分布中采样，避免了模式过拟合和输出分布偏移
    - 使用 9 种不同的生成配置（temperature × top-p 组合）增加多样性

2. **指令质量控制**

    - 去重：消除重复指令
    - 难度过滤：裁判将指令分为4个等级（Excellent 9-10, Good 6-8, Average 3-5, Poor 1-2），仅保留 Good 和 Excellent 级别
    - Embedding 压缩：使用 KCenterGreedy 算法基于 all-roberta-large-v1 embedding 选择最终指令，确保多样性和代表性

3. **Win-Loss 决策机制**

    - **本地评分**：基于裁判投票比例 $x_{A>B}^i = t_A/(t_A+t_B)$
    - **全局评分**：引入 Elo Rating 系统，动态追踪每个模型的全局相对实力
    - **最终得分**：$e_A^i = \sum_{B \in Com \setminus A} \alpha X_{A>B}^{Elo} + (1-\alpha) x_{A>B}^i$，其中 $\alpha=0.7$ 平衡局部偶然性和全局一致性
    - Elo Rating 可防止弱模型因偶然因素在某些指令上意外胜出

### 损失函数 / 训练策略

- 选择最终得分最高的回答作为 gold output，使用标准 SFT 训练
- 训练硬件：8 × NVIDIA A800 80G GPU
- 全局 batch size 512，总训练步数 448
- 学习率 1e-5，权重衰减 3e-7
- WarmupLR 调度器，warmup ratio 0.2
- 竞技轮数设为 70,000 轮，Elo 系统 K=40

## 实验关键数据

### 主实验

| 基准 | WarriorCoder (6.7B) | 最佳基线 | 提升 |
|------|-------------------|---------|------|
| HumanEval | 80.5% | MagicoderS-DS 76.8% | +3.7 |
| HumanEval+ | 75.6% | MagicoderS-DS 70.7% | +4.9 |
| MBPP | 76.2% | Magicoder-DS 75.4% | +0.8 |
| MBPP+ | 64.8% | MagicoderS-DS 64.4% | +0.4 |

- 相比同 backbone（DeepSeekCoder-Base-6.7B）的 HumanEval 从 47.6% 提升到 80.5%（+32.9）
- 在 CRUXEval 代码推理上 pass@5 超越 GPT-3.5-Turbo（66.5% vs 63.2%）
- DS-1000 库使用基准上在 SciPy、Sklearn、TensorFlow 上超越所有基线
- **最关键**：不依赖任何专有 LLM，完全基于开源模型

### 消融实验

| 专家数量 | HumanEval | HumanEval+ | MBPP | MBPP+ |
|---------|-----------|------------|------|-------|
| 1 | 75.4 | 72.6 | 73.3 | 62.4 |
| 2 | 77.2 | 73.3 | 74.5 | 62.9 |
| 5 | 80.5 | 75.6 | 76.2 | 64.8 |

- 专家数量越多，性能越好，说明多专家知识融合有效

### 关键发现

1. **数据独立性**：挖掘的指令与现有训练数据集（WizardCoder、Magicoder 等）的 ROUGE 重叠率绝大部分低于 0.3，无一超过 0.6，说明数据是从模型内部分布新采样的
2. **任务多样性**：训练数据覆盖代码生成（51.4%）、调试（12.2%）、理论解释（22.2%）、优化（3.8%）等 7 类任务
3. **专家胜率矩阵**：没有任何单一专家在所有任务上胜出，说明多专家对战有效利用了各模型的不同优势
4. **难度分布**：大部分指令在 Good 级别，Excellent 级别较少，反映了代码专家的内部知识分布

## 亮点与洞察

1. **范式创新**：从"扩展已有数据集"转向"从专家对战中从零生成数据"，彻底摆脱了对专有 LLM 和种子数据集的依赖
2. **Completion-based 指令挖掘**：巧妙利用 LLM 自身的补全能力来挖掘其已掌握的知识，比传统提示生成更自然
3. **全局+局部评分**：Elo Rating 与投票结合的机制有效避免了单纯投票的偶然性和偏差
4. **可扩展性**：框架可以轻松加入更多专家 LLM，专家越多数据质量越高

## 局限与展望

1. 当专家数量多时对战过程耗时，需探索更高效的竞赛模式
2. 目前仅在代码任务上验证，未扩展到其他复杂任务（如数学推理）
3. 裁判模型（剩余的专家 LLM）仍可能引入 LLM-as-a-judge 的固有偏差（位置偏差、冗余偏差等）
4. 未比较使用更大目标模型（如 33B 或 70B）时的效果

## 相关工作与启发

- **LMSYS Chatbot Arena**：本文的竞技场思想直接受到此工作启发，但将人类在线评估替换为 LLM 自动评估
- **Magpie**：completion-based 指令挖掘方法的灵感来源
- **Self-Instruct / Evol-Instruct**：传统数据飞轮方法，本文指出其依赖专有 LLM 的局限
- **启发**：这种多专家对战的框架可推广到其他领域（数学、推理等），只要有多个能力互补的专家模型

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术深度 | 4 |
| 实验充分性 | 4 |
| 写作质量 | 4 |
| **总评** | **4.0** |

<!-- RELATED:START -->

## 相关论文

- [ToolCoder: A Systematic Code-Empowered Tool Learning Framework for Large Language Models](toolcoder_code_empowered_tool_learning.md)
- [Interactive and Expressive Code-Augmented Planning with Large Language Models](interactive_and_expressive_code-augmented_planning_with_large_language_models.md)
- [OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models](opencoder_the_open_cookbook_for_top-tier_code_large_language_models.md)
- [Transforming Podcast Preview Generation: From Expert Models to LLM-Based Systems](transforming_podcast_preview_generation_from_expert_models_to_llm-based_systems.md)
- [INTERACT: Enabling Interactive, Question-Driven Learning in Large Language Models](interact_enabling_interactive_question-driven_learning_in_large_language_models.md)

<!-- RELATED:END -->
