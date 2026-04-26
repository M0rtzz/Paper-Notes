---
title: >-
  [论文解读] System Prompt Optimization with Meta-Learning
description: >-
  [NeurIPS 2025][LLM/NLP][系统提示优化] 提出双层系统提示优化问题并设计 MetaSPO 元学习框架，通过外循环优化跨任务泛化的系统提示、内循环优化任务特定的用户提示，使优化后的系统提示在 14 个未见任务上显著超越基线。
tags:
  - NeurIPS 2025
  - LLM/NLP
  - 系统提示优化
  - 元学习
  - 双层优化
  - 提示工程
  - 跨任务泛化
---

# System Prompt Optimization with Meta-Learning

**会议**: NeurIPS 2025  
**arXiv**: [2505.09666](https://arxiv.org/abs/2505.09666)  
**代码**: [GitHub](https://github.com/Dozi01/MetaSPO)  
**领域**: llm_nlp  
**关键词**: 系统提示优化, 元学习, 双层优化, 提示工程, 跨任务泛化  

## 一句话总结

提出双层系统提示优化问题并设计 MetaSPO 元学习框架，通过外循环优化跨任务泛化的系统提示、内循环优化任务特定的用户提示，使优化后的系统提示在 14 个未见任务上显著超越基线。

## 研究背景与动机

LLM 的输入提示由两部分组成：**系统提示**（task-agnostic，定义 LLM 的行为框架）和**用户提示**（task-specific，针对特定查询）。然而现有提示优化研究几乎完全聚焦于用户提示，忽视了系统提示的优化。

系统提示优化的独特价值：
- **一次优化，多处使用**：一个好的系统提示可泛化到多个任务和域
- **与用户提示协同**：优化的系统提示为 LLM 建立鲁棒的行为框架，与用户提示形成协同
- **降低适配成本**：好的系统提示使新任务的用户提示优化更快（需更少迭代和数据）

现有工作的不足：
- APE (2022)、OPRO (2023)、TextGrad (2023) 等仅优化用户提示
- SPRIG (2024) 虽优化系统提示，但不考虑与用户提示的交互，也缺乏元学习框架
- 缺少对系统提示跨任务泛化能力的系统研究

## 方法详解

### 整体框架

将系统提示优化形式化为**双层优化**问题：

**外层目标**（系统提示）：$\mathbf{s}^* = \arg\max_{\mathbf{s}} \mathbb{E}_{T_i \sim \mathcal{T}}[\mathbb{E}_{(\mathbf{q},\mathbf{a}) \sim T_i}[f(\text{LLM}(\mathbf{s}, \mathbf{u}_i^*, \mathbf{q}), \mathbf{a})]]$

**内层目标**（用户提示）：$\mathbf{u}_i^* = \arg\max_{\mathbf{u}} \mathbb{E}_{(\mathbf{q},\mathbf{a}) \sim T_i}[f(\text{LLM}(\mathbf{s}, \mathbf{u}, \mathbf{q}), \mathbf{a})]$

### 关键设计：MetaSPO（Meta-level System Prompt Optimizer）

**内循环（用户提示优化）**：
1. 用当前用户提示在目标任务上评估，识别错误预测样本
2. 将当前用户提示与错误样本一起送入 LLM 进行**失败分析**
3. 基于分析生成多个候选用户提示
4. 评估所有候选提示，选择 top-$k$ 个最优提示

**外循环（系统提示优化）**：
1. 用当前系统提示在**所有**源任务上评估，汇总各任务的错误样本
2. 对系统提示进行跨任务的失败分析
3. 生成多个候选系统提示
4. 在所有任务上（配合各自的优化用户提示）评估候选系统提示
5. 选择跨任务表现最优的 top-$k$ 个系统提示

两个循环交替迭代 3 次。

### 实现细节

- **基座模型**：Llama 3.2 (3B) 用于生成回答
- **优化器模型**：GPT-4o mini 用于生成候选提示
- **内循环**：每轮生成 3 个候选用户提示，保留 3 个；使用 3 个错误样本做分析
- **外循环**：每轮生成 9 个候选系统提示，保留 1 个；使用每任务 2 个错误样本
- **温度设置**：基座模型 0（确保一致性），优化器模型 1（鼓励多样性）

### 损失函数 / 训练策略

纯推理时优化（gradient-free），不修改模型参数。依赖任务度量函数 $f$（accuracy、F1 等）作为评估信号。优化过程完全通过 LLM 的文本生成能力实现——提示分析和候选生成都通过精心设计的元提示完成。

## 实验关键数据

### 主实验：未见任务泛化

在 5 个域（医疗、评论分析、推理、安全、事实核查）共 14 个未见目标任务上的平均得分：

| 方法 | Medical | Review | Reasoning | Safety | Grounding | Avg |
|------|---------|--------|-----------|--------|-----------|-----|
| Default ("You are a helpful...") | 33.3 | 37.4 | 42.8 | 25.0 | 13.4 | 32.2 |
| CoT ("Let's think step by step") | — | — | — | — | — | 33.2 |
| Service (商业系统提示) | — | — | — | — | — | 34.2 |
| SPRIG (遗传算法) | 37.0 | 56.8 | 38.7 | 28.1 | 14.1 | 35.0 |
| **MetaSPO (Domain)** | **48.9** | **62.7** | **52.2** | **36.5** | **16.4** | **44.5** |

MetaSPO 比默认系统提示**平均提升 12.3 分**，比最强基线 SPRIG 提升 7-9 分。

### 测试时适配实验

固定系统提示，额外优化用户提示：

| 方法 | Medical | Review | Reasoning | Safety | Grounding | Avg |
|------|---------|--------|-----------|--------|-----------|-----|
| Default | 45.1 | 68.9 | 64.0 | 59.9 | 17.5 | 51.1 |
| SPRIG | 45.4 | 69.3 | 65.3 | 64.7 | 17.7 | 52.5 |
| **MetaSPO** | **45.6** | **71.4** | **67.3** | **67.2** | **19.9** | **54.3** |

### 消融实验

| 实验 | 关键发现 |
|------|---------|
| 仅外循环（无内循环） | 性能 38.2 vs. MetaSPO 42.2，说明用户提示协同优化至关重要 |
| MetaSPO w/ APE | 38.9，不同提示优化策略均可插入框架 |
| MetaSPO w/ EVO | 40.2，进化算法变体也有效 |
| 源任务数量 | 1→6 个源任务：Review 域提升 17.1%，Reasoning 域提升 8.26% |
| 跨模型迁移 | Llama 3.2 优化的系统提示在 Llama 3.1, Qwen 2.5, GPT-4o mini 上也有效 |
| 分离 vs. 统一输入 | 将系统/用户提示分别放入对应角色（分离输入）优于合并放入用户角色 |

### 效率分析

- MetaSPO 达到 Default 的最终性能所需：**80% 更少的优化迭代**和 **75% 更少的数据**
- 85% 的用户提示在 MetaSPO 系统提示下表现提升

### 关键发现

1. 系统提示优化是一个被严重低估的方向——简单的 MetaSPO 就能带来显著提升
2. 元学习框架的内外循环协同是成功的关键——仅做外循环效果大打折扣
3. 优化后的系统提示具有跨模型迁移能力——这说明好的系统提示编码了通用的任务解决知识
4. LLM 的注意力分数验证：MetaSPO 的系统提示确实获得了更多注意力

## 亮点与洞察

1. **新问题定义**：双层系统提示优化是一个自然但此前未被提出的问题，具有很强的实际价值
2. **元学习的自然适配**：系统提示 = 跨任务共享知识，用户提示 = 任务特定适配——完美匹配元学习范式
3. **即插即用设计**：MetaSPO 框架不依赖特定的提示优化方法，可灵活更换内外循环的优化策略
4. **跨模型泛化**：为一个模型优化的系统提示在其他模型上也有效，暗示好的系统提示是通用的
5. **实际直觉**：优化后的系统提示通常赋予 LLM 更具体的角色描述和行为指南

## 局限与展望

1. **优化器模型依赖**：效果受限于优化器 LLM 的能力（这里用 GPT-4o mini），弱模型可能不够
2. **评估任务有限**：14 个目标任务覆盖不够广泛，缺少代码生成、数学推理等任务
3. **基座模型较小**：主要在 Llama 3.2 (3B) 上实验，更大模型的效果未系统验证
4. **计算开销**：需要多轮 LLM 调用来生成和评估候选提示，成本不可忽略
5. **安全风险**：优化后的系统提示可能被滥用来引导 LLM 产生有害输出
6. **缺少与 fine-tuning 的对比**：系统提示优化是否能替代部分微调工作？

## 相关工作与启发

- **与 SPRIG (Wen et al., 2024) 的区别**：SPRIG 用遗传算法优化系统提示但不考虑用户提示交互，无元学习
- **与 MAML (Finn et al., 2017) 的关系**：借鉴了元学习"学习如何学习"的思想，但在文本空间操作
- **与 TextGrad (Pryzant et al., 2023) 的关系**：内循环使用类似的文本梯度思想
- **启发**：双层优化框架可能适用于其他 LLM 配置优化（如 RAG 检索策略、工具使用规则等）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 问题定义新颖，双层框架合理，但底层优化技术较标准
- **理论深度**: ⭐⭐ — 主要是工程和实验贡献，缺乏理论分析
- **实验充分度**: ⭐⭐⭐⭐⭐ — 5 域 14 任务、多模型、多消融、注意力分析，非常全面
- **写作质量**: ⭐⭐⭐⭐⭐ — 图文并茂，实验设计精心，分析深入
- **实用价值**: ⭐⭐⭐⭐⭐ — 直接可用于改善 LLM 应用，代码已开源
- **综合**: ⭐⭐⭐⭐ (8/10) — 实用价值极高的工作，填补了系统提示优化的空白

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] GAPO: Learning Preferential Prompt through Generative Adversarial Policy Optimization](../../ACL2025/llm_nlp/gapo_preferential_prompt.md)
- [\[NeurIPS 2025\] C²Prompt: Class-aware Client Knowledge Interaction for Federated Continual Learning](c2prompt_class-aware_client_knowledge_interaction_for_federated_continual_learni.md)
- [\[ACL 2025\] RiOT: Efficient Prompt Refinement with Residual Optimization Tree](../../ACL2025/llm_nlp/riot_efficient_prompt_refinement_with_residual_optimization_tree.md)
- [\[ICML 2025\] Beyond Induction Heads: In-Context Meta Learning Induces Multi-Phase Circuit Emergence](../../ICML2025/llm_nlp/beyond_induction_heads_in-context_meta_learning_induces_multi-phase_circuit_emer.md)
- [\[ACL 2025\] Efficient and Accurate Prompt Optimization: the Benefit of Memory in Exemplar-Guided Reflection](../../ACL2025/llm_nlp/erm_prompt_optimization_memory.md)

<!-- RELATED:END -->
