---
title: >-
  [论文解读] Is In-Context Learning Learning?
description: >-
  [ICLR2026][LLM推理][in-context learning] 通过大规模控制变量实验系统分析 ICL 是否构成"学习"，发现数学上 ICL 满足学习定义，但实证表明其泛化能力有限——模型主要依赖 prompt 中的结构规律进行模式推演（deduction），而非从示例中真正习得新能力。
tags:
  - ICLR2026
  - LLM推理
  - in-context learning
  - ICL
  - memorisation
  - distributional shift
  - generalization
  - autoregressive models
---

# Is In-Context Learning Learning?

**会议**: ICLR2026  
**arXiv**: [2509.10414](https://arxiv.org/abs/2509.10414)  
**代码**: 未开源  
**领域**: llm_reasoning  
**关键词**: in-context learning, ICL, memorisation, distributional shift, generalization, autoregressive models  

## 一句话总结

通过大规模控制变量实验系统分析 ICL 是否构成"学习"，发现数学上 ICL 满足学习定义，但实证表明其泛化能力有限——模型主要依赖 prompt 中的结构规律进行模式推演（deduction），而非从示例中真正习得新能力。

## 背景与动机

**领域现状**：In-context learning (ICL) 使自回归语言模型能够通过 next-token prediction 在不更新参数的情况下解决下游任务，只需在 prompt 中提供少量示例（exemplars）。这种能力引发了大量关于 LLM 是否能从少量示例中"学习"未见任务的讨论。

**现有痛点**：

- 关于 ICL 的研究混淆了"推演"（deduction）与"学习"（learning）的概念边界——推演不等于学习
- ICL 并不显式编码给定的观测数据，而是依赖模型的先验知识和 prompt 中的示例
- 现有研究缺乏对记忆效应（memorisation）、预训练数据泄露、分布偏移等混淆因素的系统控制
- 难以判断 ICL 的优异表现究竟来自"真正从示例中学习"还是"先验知识的检索 + 模式匹配"

**核心矛盾**：ICL 在形式上类似于学习（从 prompt 中的示例推断任务规则），但其自回归编码机制是否提供了足够的归纳偏置来支撑稳健的泛化和真正的知识获取？这一问题的答案直接关系到如何正确看待和使用 LLM。

**本文方案**：从理论和实证两个层面回答"ICL 是否是学习"——先从数学上论证 ICL 满足学习的形式定义，再通过大规模控制变量实验（ablation study）评估 ICL 的实际学习能力边界，系统消除记忆效应、预训练泄露、分布偏移和提示风格等混淆因素。

## 方法详解

### 整体框架

本文采用理论分析 + 大规模实证分析的双轨策略。理论部分论证 ICL 在数学定义层面满足学习的标准（类似 PAC 学习框架）；实证部分通过系统消除（ablate out）或控制多个混淆因素，揭示 ICL 的真实学习能力边界。实验覆盖多种模型架构、多种提示风格、多种任务类型和多种数据分布设置，构成目前 ICL 行为研究中规模最大的控制变量实验之一。

### 关键设计 1：记忆与预训练效应的解耦

核心目标是将 ICL 表现中来自预训练记忆的部分与来自 prompt 示例的真正学习部分分离。具体做法包括：

- **Benchmark 污染检测**：使用多种方法评估预训练数据是否已包含测试任务信息，量化记忆效应对 ICL 表现的贡献
- **Zero-shot vs. Few-shot 差异分析**：zero-shot 表现反映纯先验知识水平，few-shot 与 zero-shot 的增量才可能归因于"从示例中学到的"
- **反事实标签实验**：使用随机标签或反转标签来检测模型是否真正利用了示例中的输入-输出映射关系

数学上，设 $f$ 为目标函数，$\hat{f}_{\text{ICL}}$ 为 ICL 的预测函数，则 ICL 的学习增益可定义为：

$$\Delta_{\text{learn}} = \mathbb{E}[\mathcal{L}(\hat{f}_{\text{zero-shot}})] - \mathbb{E}[\mathcal{L}(\hat{f}_{\text{few-shot}})]$$

其中 $\mathcal{L}$ 为任务损失函数。本文发现在充分控制记忆效应后，$\Delta_{\text{learn}}$ 显著减小，说明 ICL 表现中大量来自先验知识而非从示例中学习。

### 关键设计 2：分布偏移与示例缩放行为分析

核心实验方法是系统改变以下变量，观察 ICL 准确率的变化模式（scaling behavior）：

- **示例数量 $k$**：从 $k=0$（zero-shot）到 many-shot，观察准确率如何随示例增加而变化
- **示例分布**：改变类别平衡、样本难度分布、样本选择策略和呈现顺序
- **提示风格**：标准 few-shot、chain-of-thought (CoT)、不同模板格式和措辞
- **模型选择**：多种规模和架构的自回归模型

关键发现：当示例数量 $k$ 增大时，准确率趋向一个与具体配置无关的极限值。形式上：

$$\lim_{k \to \infty} \text{Acc}(k; \mathcal{D}, \mathcal{M}, \mathcal{S}) \approx C_{\text{task}}$$

其中 $\mathcal{D}$ 为示例分布、$\mathcal{M}$ 为模型、$\mathcal{S}$ 为提示风格。这一发现与真正的学习预期形成对比——真正的学习应当随着更多/更好的数据而持续改进。

### 关键设计 3：Chain-of-Thought 的分布敏感性

本文特别分析了 CoT 提示风格下 ICL 的行为特征。发现 CoT 虽然在某些任务上显著提升准确率，但对 prompt 的分布特征和格式表现出更高的敏感性（distributional sensitivity）。这说明 CoT 的改进并非来自更深层的任务学习，而是利用了推理链的结构规律性来更高效地进行模式推演。具体表现为：

- 标准 few-shot 的表现相对稳定但提升天花板低
- CoT 的表现波动更大，高度依赖推理链的格式和结构
- 在形式相似但语义不同的任务上，CoT 的准确率差异显著
- ICL 实质上是从 prompt 的统计规律性中提取模式，而非编码新知识

## 实验结果

### 主实验：ICL 学习能力的系统评估

| 控制变量 | 核心结论 | 关键证据 |
|:---------|:---------|:---------|
| 记忆效应 | 预训练记忆显著贡献 ICL 表现 | 使用 contamination-free benchmark 后性能显著下降 |
| 示例数量 | 准确率快速饱和 | $k > 16$ 后增益可忽略，呈对数饱和曲线 |
| 示例分布 | 极限下分布不敏感 | 不同类别分布/难度分布下收敛值相近 |
| 模型选择 | 不同模型在极限下差异缩小 | 多种架构和规模的模型对比 |
| 提示风格 | 标准 few-shot 不敏感，CoT 敏感 | 格式变化实验显示 CoT 波动更大 |
| 输入语言特征 | 表面特征对极限表现影响有限 | 同义改写/格式变化对准确率影响微弱 |

### 消融实验：ICL 信息利用机制分析

| 实验条件 | 准确率表现 | 核心含义 |
|:---------|:----------|:---------|
| 正确标签 | 基线水平（最高） | 标准 ICL 表现 |
| 随机标签 | 微小下降 | 模型不完全依赖示例中的映射关系 |
| 反事实标签 | 中等下降 | 模型部分利用标签信息但并非核心依据 |
| 无标签（仅输入格式） | 接近 few-shot | prompt 的结构性特征比标签内容更重要 |
| 打乱示例顺序 | 几乎不变 | 模型对示例呈现顺序不敏感 |

## 评价

**评分**: ⭐⭐⭐⭐

**优点**：

- 提出了关于 ICL 本质的重要问题，兼具理论深度和实证广度
- 实验设计极为严谨：系统消除记忆、分布偏移、风格等多个混淆因素
- 核心发现（示例增多后准确率饱和且对超参不敏感）具有重要的理论和实践意义
- 对 CoT 的分布敏感性分析提供了理解 CoT 机制的新视角

**不足**：

- 主要得出否定性结论（ICL 的学习能力有限），缺少建设性的改进方向和解决方案
- 部分实验可能受限于特定 benchmark 任务集的选择偏差，对更复杂的推理和生成任务的推广性需验证
- "学习"的定义本身存在多种理解方式（PAC 学习 vs. 贝叶斯学习 vs. 泛化学习），不同定义下结论的稳健性未充分讨论
- 缺少对 ICL 在不同模型规模下行为差异的系统 scaling law 分析

**与相关工作的关键区别**：

- 不同于从 Bayesian inference 角度解释 ICL 的工作（Xie et al., 2021），本文从经验学习论角度系统质疑了 ICL 作为稳健学习机制的假说
- 不同于单因素分析的工作，本文同时控制记忆、分布、模型、风格等多维变量，得出更全面和可靠的否定性结论
- 不同于纯理论分析的工作，本文将数学论证与大规模实验结合，增强了结论的说服力

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Unlabeled Data Can Provably Enhance In-Context Learning of Transformers](../../NeurIPS2025/llm_reasoning/unlabeled_data_can_provably_enhance_in-context_learning_of_transformers.md)
- [\[ICLR 2026\] Segment-Level Attribution for Selective Learning of Long Reasoning Traces](segment-level_attribution_for_selective_learning_of_long_reasoning_traces.md)
- [\[ICLR 2026\] Adaptive Social Learning via Mode Policy Optimization for Language Agents](adaptive_social_learning_via_mode_policy_optimization_for_language_agents.md)
- [\[ACL 2025\] CoT-ICL Lab: A Synthetic Framework for Studying Chain-of-Thought Learning from In-Context Demonstrations](../../ACL2025/llm_reasoning/cot-icl_lab_a_synthetic_framework_for_studying_chain-of-thought_learning_from_in.md)
- [\[NeurIPS 2025\] Curriculum Abductive Learning](../../NeurIPS2025/llm_reasoning/curriculum_abductive_learning.md)

<!-- RELATED:END -->
