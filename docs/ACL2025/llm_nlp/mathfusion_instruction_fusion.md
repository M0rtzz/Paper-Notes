---
title: >-
  [论文解读] MathFusion: Enhancing Mathematical Problem-solving of LLM through Instruction Fusion
description: >-
  [ACL 2025][LLM/NLP][数学推理] 提出 MathFusion 框架，通过三种问题融合策略（顺序/并行/条件融合）将数学问题两两合成新问题，仅用 45K 额外合成数据就在多个基准上实现平均 18 个百分点的数学推理提升。
tags:
  - ACL 2025
  - LLM/NLP
  - 数学推理
  - 数据增强
  - 指令融合
  - SFT
  - 数学问题合成
---

# MathFusion: Enhancing Mathematical Problem-solving of LLM through Instruction Fusion

**会议**: ACL 2025  
**arXiv**: [2503.16212](https://arxiv.org/abs/2503.16212)  
**代码**: [QizhiPei/MathFusion](https://github.com/QizhiPei/MathFusion)  
**领域**: LLM/NLP  
**关键词**: 数学推理, 数据增强, 指令融合, SFT, 数学问题合成  

## 一句话总结

提出 MathFusion 框架，通过三种问题融合策略（顺序/并行/条件融合）将数学问题两两合成新问题，仅用 45K 额外合成数据就在多个基准上实现平均 18 个百分点的数学推理提升。

## 研究背景与动机

**核心问题：** LLM 的数学推理能力依赖高质量训练数据，现有数据增强方法主要在单个问题层面进行修改（如改写、变换难度），无法捕获数学知识间的关系结构。

**现有方案局限：**
- MetaMath、WizardMath 等方法聚焦于单题实例级增强（改写/难度变换/反向推理），忽略了问题之间的内在关联
- 真实数学问题往往由相互依赖的子问题组成，形成复杂的依赖图，现有方法无法建模这种结构
- 组合增强方法（如 Mosaic-IT、KPMath）存在但未针对数学问题的逻辑一致性进行优化

**核心动机：** 人类通过系统接触相互关联的概念来发展数学能力。战略性地融合互补的数学指令，可以激活更深层的推理能力，实现跨问题的知识整合。

## 方法详解

### 整体框架

MathFusion 从原始数学数据集中选取问题对 $(P_A, P_B)$，通过三种融合策略合成新问题 $P_F$，再由 GPT-4o-mini 生成解答，最终构成 MathFusionQA 数据集（60K 样本）用于 SFT 微调。

### 关键设计

1. **问题对构建：** 对每个问题 $P_A$，使用 OpenAI embedding（text-embedding-3-large）计算语义相似度，选择最相似的 $P_B$ 组成问题对，确保类型和上下文相近

2. **顺序融合（Sequential Fusion）：** $P_F^{seq} = P_B(P_A)$，将 $P_A$ 的答案作为 $P_B$ 的输入条件，建立求解依赖链。例如：船运载人数的答案 → 成为公交车乘客数量的输入

3. **并行融合（Parallel Fusion）：** $P_F^{para} = \Phi(P_A', P_B')$，将两个类比问题整合为一个新问题，封装其共享的数学本质，可能修改原题的输入条件

### 条件融合

$P_F^{cond} = \Gamma(P_A, P_B)$，将两个问题整合到一个现实场景中，最终答案通过比较或选择 $P_A$ 和 $P_B$ 的结果得出，增强条件推理能力。

### 损失函数

标准的 SFT 自回归交叉熵损失：$\mathcal{L} = -\sum_t \log P(y_t | y_{<t}, x)$。

## 实验

### 主实验：不同基座模型的数学推理性能

| 模型 | #样本 | MATH | GSM8K | College | DM | Olympiad | Theorem | AVG |
|------|------|------|------|------|------|------|------|------|
| DSMath-7B-Standard | 15K | 30.6 | 66.3 | 22.7 | 28.6 | 5.6 | 11.0 | 27.5 |
| DSMath-7B-DART-Math† | 60K | 51.4 | 82.9 | 39.1 | 62.8 | 21.0 | 27.4 | 47.4 |
| **MathFusion-DSMath-7B** | **60K** | **53.4** | 77.9 | 39.8 | **65.8** | **23.3** | 24.6 | **47.5** |
| Llama3-8B-Standard | 15K | 17.5 | 65.4 | 12.9 | 21.6 | 4.7 | 10.9 | 22.2 |
| Llama3-8B-DART-Math† | 60K | 34.1 | 77.2 | 23.4 | 36.0 | 8.7 | 18.2 | 32.9 |
| **MathFusion-Llama3-8B** | **60K** | **41.6** | **79.8** | **24.3** | **39.2** | **13.6** | 18.1 | **36.1** |
| Mistral-7B-Standard | 15K | 12.4 | 60.3 | 8.4 | 17.0 | 2.2 | 7.6 | 18.0 |
| Mistral-7B-DART-Math† | 60K | 34.1 | 77.2 | 23.4 | 36.0 | 8.7 | 18.2 | 32.9 |
| **MathFusion-Mistral-7B** | **60K** | **41.6** | **79.8** | **24.3** | **39.2** | **13.6** | 18.1 | **36.1** |

### 消融实验：三种融合策略对比（Llama3-8B）

| 融合策略 | #样本 | MATH | GSM8K | AVG |
|------|------|------|------|------|
| Standard (baseline) | 15K | 17.5 | 65.4 | 22.2 |
| Sequential 融合 | 30K | **38.8** | **77.9** | **35.6** |
| Parallel 融合 | 30K | 38.1 | 75.4 | 35.3 |
| Conditional 融合 | 30K | 34.7 | 76.9 | 31.3 |
| **三策略联合** | **60K** | **41.6** | **79.8** | **36.1** |

### 关键发现

1. MathFusion 仅用 60K 数据就超越了使用 590K 数据的 DART-Math（在 Llama3-8B 上 AVG 36.1 vs 不到 1/10 的数据量）
2. Sequential 融合在三种策略中效果最好，建模了问题间的求解依赖，符合数学推理的链式特性
3. 三种融合策略联合使用效果优于单一策略，说明不同融合视角提供了互补的推理能力
4. 与 DART-Math 结合后进一步提升，证明 MathFusion 具有互补性
5. 扩大融合规模（top-1→top-4 邻居，195K 样本）可在 DSMath-7B 上达到 AVG 49.9

## 亮点

- **创新的跨问题数据增强范式**：首次系统地将数学问题间的关系结构引入数据增强，超越传统单题修改
- **极高的数据效率**：仅 45K 额外合成数据，以不到 DART-Math 1/10 的数据量达到相当或更好的效果
- **三种融合策略设计精巧**：Sequential 建模依赖链、Parallel 建模概念共性、Conditional 建模条件推理，覆盖不同推理模式
- **方法通用性好**：在 DeepSeekMath-7B、Mistral-7B、Llama3-8B 三种基座模型上均实现显著提升

## 局限性

- 融合可能产生不完整或错误的问题（论文附录承认了这一点）
- GPT-4o-mini 生成的解答质量不可控，可能引入错误解答
- 问题对构建依赖 OpenAI embedding API，增加了成本和外部依赖
- 目前仅在 7-8B 规模模型上验证，更大模型的效果未知
- 融合策略的设计较为启发式，缺乏理论分析为何不同融合方式能增强推理

## 相关工作

- **单题数据增强**：MetaMath（改写+反向推理）、WizardMath（难度变换）、DART-Math（拒绝采样难题）、RefAug（反思增强）
- **组合数据增强**：Mixup（线性插值）、Mosaic-IT（指令拼接）、Instruct-SkillMix（技能组合）、KPMath-Plus（关键点组合）
- **数学 LLM**：DeepSeekMath（继续预训练）、Mistral、Llama3

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | 8/10 |
| 有效性 | 8/10 |
| 实验充分度 | 9/10 |
| 写作质量 | 8/10 |
| 总分 | 8/10 |

<!-- RELATED:START -->

## 相关论文

- [ArithmAttack: Evaluating Robustness of LLMs to Noisy Context in Math Problem Solving](arithmattack_evaluating_robustness_of_llms_to_noisy_context_in_math_problem_solv.md)
- [CoT-based Synthesizer: Enhancing LLM Performance through Answer Synthesis](cot-based_synthesizer_enhancing_llm_performance_through_answer_synthesis.md)
- [Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning](problem-solving_logic_guided_curriculum_in-context_learning_for_llms_complex_rea.md)
- [Veracity Bias and Beyond: Uncovering LLMs' Hidden Beliefs in Problem-Solving Reasoning](veracity_bias_llm_hidden_beliefs.md)
- [Just a Scratch: Enhancing LLM Capabilities for Self-Harm Detection through Intent Refinement](just_a_scratch_enhancing_llm_capabilities_for_self-harm_detection_through_intent.md)

<!-- RELATED:END -->
