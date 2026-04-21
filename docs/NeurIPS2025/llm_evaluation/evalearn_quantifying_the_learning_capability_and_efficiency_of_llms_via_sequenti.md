---
title: >-
  [论文解读] EvaLearn: Quantifying the Learning Capability and Efficiency of LLMs via Sequential Problem Solving
description: >-
  [NeurIPS 2025][LLM evaluation] 提出 EvaLearn 基准，通过**序列化问题求解**范式评估 LLM 的学习能力和学习效率，揭示静态能力强的模型不一定具备更强的学习潜力。
tags:
  - NeurIPS 2025
  - LLM evaluation
  - sequential learning
  - learning capability
  - benchmark
  - dynamic evaluation
---

# EvaLearn: Quantifying the Learning Capability and Efficiency of LLMs via Sequential Problem Solving

**会议**: NeurIPS 2025  
**arXiv**: [2506.02672](https://arxiv.org/abs/2506.02672)  
**代码**: [github.com/ByteDance-Seed/EvaLearn](https://github.com/ByteDance-Seed/EvaLearn)  
**领域**: llm_nlp  
**关键词**: LLM evaluation, sequential learning, learning capability, benchmark, dynamic evaluation

## 一句话总结

提出 EvaLearn 基准，通过**序列化问题求解**范式评估 LLM 的学习能力和学习效率，揭示静态能力强的模型不一定具备更强的学习潜力。

## 研究背景与动机

现有 LLM 评估基准几乎全部采用**并行评估范式**：模型在独立同分布的样本上逐一作答，汇总得到准确率等静态指标。这种方式只能衡量模型的"静态能力"，而忽视了一个同样重要的维度——**模型在特定任务中通过经验进行学习和适应的能力**（即学习能力），以及学习发生的速度（即学习效率）。

学习能力和学习效率是衡量人类智能的核心指标，但在 LLM 评估中几乎未被系统探索。并行评估范式在结构上就无法捕捉这种动态学习行为。因此需要一种全新的评估框架来弥补这一空白。

## 方法详解

### 整体框架

EvaLearn 采用**序列化评估范式**：
1. 构建 648 个具有挑战性的问题，组织为 182 个序列
2. 每个序列包含 7 个同一任务类型的问题
3. 模型需要**按顺序**逐一解决每个序列中的问题
4. 允许模型利用前面问题的解题经验来改善后续表现

涵盖六类任务：
- **摘要 (Sum)**：评估模型是否能通过经验提升摘要的准确性和覆盖度
- **分类 (Cla)**：评估模型是否能从一系列分类问题中增强分类能力
- **抽取 (Ex)**：评估模型是否能逐步提高关键信息抽取的完整性
- **逻辑推理 (LR)**：评估模型能否从错误中学习以提升逻辑推理
- **数学推理 (MR)**：评估模型能否通过反馈快速掌握解题策略
- **序列推理 (SR)**：评估模型能否通过历史经验增强序列推理能力

### 关键设计

**自动评估框架**：由于大部分挑战性问题无法用规则验证，采用**实例级 rubric + LLM-as-a-judge** 方法。每个问题附带人工编写的评分标准，使用 GPT-4o 作为评判模型。验证实验表明所有任务的评估准确率超过 95%。

**两种序列学习范式**：
- **Demonstration Learning**：为当前问题提供序列中所有前序问题及其标准答案（类似 ICL）
- **Feedback Learning**：除前序问题外，还提供模型自己先前的解答以及评判模型基于 rubric 给出的详细反馈

### 评估指标体系

设 $N=182$ 为序列数，$M=7$ 为每序列问题数，$y_{n,m} \in \{0,1\}$ 表示第 $n$ 序列第 $m$ 问是否正确。五个指标：

1. **整体准确率 (Acc)**：$\text{Acc} = \frac{1}{NM}\sum_{n=1}^{N}\sum_{m=1}^{M} y_{n,m}$

2. **拟合准确率曲线斜率 (k)**：对位置-准确率曲线 $\text{Acc}_m$ 做最小二乘线性拟合，斜率 $k$ 反映学习速度

3. **首次正确解的平均位置 ($P_{\text{first}}$)**：$P_{\text{first}} = \frac{1}{N}\sum_{n=1}^{N} p_n$，越低越好

4. **首次学习正确解的平均偏移 ($P_{\text{offset}}$)**：排除零样本已能解决的问题后，衡量模型开始学习的速度

5. **连续正确解的平均数量 ($N_{\text{consec}}$)**：$N_{\text{consec}} = \frac{1}{N}\sum_{n=1}^{N} \max_{1 \le a \le b \le M}\{b-a+1 : y_{n,a}=\cdots=y_{n,b}=1\}$

6. **预热后准确率 ($\text{Acc}_{\text{pw}}\text{-}K$)**：排除前 $K$ 个问题后计算准确率

## 实验关键数据

### 主实验

在 9 个前沿模型上测试（含 thinking 和 non-thinking 类型）：

| 模型 | Zero-shot | Feedback Learning | 变化 |
|------|-----------|-------------------|------|
| OpenAI-o3-mini | 54.3% | 64.8% | **+10.5%** |
| Claude-3.7-Sonnet | 28.4% | 35.6% | **+7.2%** |
| Claude-3.7-Sonnet-Thinking | 31.2% | 37.4% | **+6.2%** |
| Gemini-2.5-Pro | 68.5% | 67.2% | -1.3% |
| DeepSeek-R1 | 55.7% | 46.4% | **-9.3%** |

**任务维度发现**：
- 数学推理：GPT-4o 提升 +18.0%，Claude-3.7-Sonnet 提升 +15.6%
- 分类：Claude-3.7-Sonnet-Thinking 提升 +13.5%
- 摘要：9 个模型中 7 个性能下降，说明该任务更依赖预训练知识

### 消融实验

**四种范式对比**（Zero-shot vs Few-shot vs Demonstration Learning vs Feedback Learning）：
- Demonstration Learning 通常优于 Few-shot 并行求解
- Feedback Learning 在大多数模型上优于 Demonstration Learning
- 反馈学习更能促进模型在序列中更早获得正确解

### 关键发现

1. **Thinking 模型更受益于序列学习**：o3-mini 平均最长连续正确解为 3.42，GPT-4o 仅 2.58
2. **学习能力与静态能力不相关**：DeepSeek-R1 静态能力优于 Claude-3.7-Sonnet，但序列学习中反而下降 9%
3. **学习效率差异显著**：Claude-3.7-Sonnet 斜率 $k=2.08$ 最高；non-thinking 模型初始基线低，因此斜率更陡
4. **任务特异性**：每个模型在某些任务上表现出强学习能力，但没有模型在所有任务上都稳定提升

## 亮点与洞察

- **评估范式创新**：首次系统量化 LLM 的动态学习能力，突破了并行评估的局限
- **指标设计全面**：五个指标从多个角度刻画学习行为——速度、稳定性、初始学习位置等
- **指标与方法解耦**：评估指标不依赖特定学习方法，支持未来扩展
- **揭示重要现象**：证明了"静态能力强 ≠ 学习能力强"，为模型开发提供新视角
- 反馈学习中，评判模型的 rubric 反馈比直接提供标准答案更有效

## 局限与展望

- 数据规模有限（648 问题 / 182 序列），每序列仅 7 个问题，可能不足以充分展示学习曲线
- 任务覆盖面可进一步扩展（如代码、多模态等）
- 仅评估了 9 个闭源/大参数模型，未涉及开源中小模型
- 评判模型（GPT-4o）本身可能存在偏差
- 序列长度固定为 7，未探索不同长度对学习行为的影响

## 相关工作与启发

- 与 ARC（Chollet, 2019）类似在衡量模型"智能潜力"，但 EvaLearn 更聚焦于从经验中学习的能力
- 与 ICL 评测不同之处：EvaLearn 关注的是长序列中的**累积学习**而非少样本泛化
- 可结合 meta-learning / curriculum learning 设计更强的序列学习策略
- 为模型选择提供新标准：静态 benchmark 上差不多的模型，学习能力可能截然不同

## 评分

- **创新性**：⭐⭐⭐⭐⭐ — 全新的评估维度和范式
- **实用性**：⭐⭐⭐⭐ — 对模型选择和开发有实际指导意义
- **实验充分度**：⭐⭐⭐⭐ — 9 个前沿模型、6 类任务、多种学习范式
- **写作质量**：⭐⭐⭐⭐ — 结构清晰，指标定义严谨
- **综合评价**：8.5/10

## 与相关工作的对比

<!-- RELATED:START -->

## 相关论文

- [Creativity or Brute Force? Using Brainteasers as a Window into the Problem-Solving Abilities of Large Language Models](creativity_or_brute_force_using_brainteasers_as_a_window_into_the_problem-solvin.md)
- [RankLLM: Weighted Ranking of LLMs by Quantifying Question Difficulty](../../ICLR2026/llm_evaluation/rankllm_weighted_ranking_of_llms_by_quantifying_question_difficulty.md)
- [Put CASH on Bandits: A Max K-Armed Problem for Automated Machine Learning](put_cash_on_bandits_a_max_k-armed_problem_for_automated_machine_learning.md)
- [CLIMB: Class-Imbalanced Learning Benchmark on Tabular Data](climb_class-imbalanced_learning_benchmark_on_tabular_data.md)
- [LCDB 1.1: A Database Illustrating Learning Curves Are More Ill-Behaved Than Previously Thought](lcdb_11_a_database_illustrating_learning_curves_are_more_ill-behaved_than_previo.md)

<!-- RELATED:END -->
