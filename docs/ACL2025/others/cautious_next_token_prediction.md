---
title: >-
  [论文解读] Cautious Next Token Prediction
description: >-
  [ACL 2025][解码策略] 提出 Cautious Next Token Prediction (CNTP)，一种无需训练的自适应解码策略：在模型预测熵较高（不确定）时采样多条候选路径至标点处，选择困惑度最低的路径作为最终续写，从而在不牺牲多样性的前提下显著提升准确性。
tags:
  - ACL 2025
  - 解码策略
  - 熵自适应采样
  - 困惑度排序
  - training-free
  - LLM推理
---

# Cautious Next Token Prediction

**会议**: ACL 2025  
**arXiv**: [2507.03038](https://arxiv.org/abs/2507.03038)  
**代码**: [有](https://github.com/wyzjack/CNTP)  
**领域**: NLP / LLM推理  
**关键词**: 解码策略, 熵自适应采样, 困惑度排序, training-free, LLM推理

## 一句话总结

提出 Cautious Next Token Prediction (CNTP)，一种无需训练的自适应解码策略：在模型预测熵较高（不确定）时采样多条候选路径至标点处，选择困惑度最低的路径作为最终续写，从而在不牺牲多样性的前提下显著提升准确性。

## 研究背景与动机

- 当前主流 LLM 解码策略（top-p/nucleus sampling + 温度缩放）是"工业默认"，但在模型不确定时容易生成不连贯或错误的内容
- **贪心解码**：确定性强但缺乏多样性，容易陷入局部最优
- **随机采样**：有多样性但不保证连贯性，高不确定步骤容易出错
- **Beam Search**：计算开销大（O(L × B)），且不自适应——对确定和不确定步骤一视同仁
- **Self-Consistency**：需要完整重跑 N_sc 次完整解码，成本 O(N_sc × L)，无法在中间步骤适应
- 核心洞察来自人类行为：人在解题时遇到不确定步骤会"想更多"、探索多条路径，最终选最有信心的；模型也可以模仿这种"谨慎"策略
- 已有 CoT、Self-Refinement 等推理时方法多依赖外部反馈或大量采样，缺少一种**轻量、自适应、无需训练**的解码方案

## 方法详解

### 整体框架

CNTP 的核心循环：每生成一个 token 前计算当前位置的预测熵 → 熵低则正常单 token 采样 → 熵高则独立采样 N 条候选路径至标点 → 计算各路径困惑度 → 选困惑度最低路径拼接到序列 → 继续生成。

### 关键设计

#### 1. 基于熵的不确定性检测

给定当前序列 s，计算下一个 token 的概率分布熵：H(s) = -Σ_w p(w|s) log p(w|s)

设定两个阈值 H_min 和 H_max，以及最大试验次数 N_max，将熵线性映射为试验数：

N = max(1, min(N_max, ⌊(H - H_min) / (H_max - H_min) × N_max⌋))

- H(s) < H_min：模型很确定，N=1，退化为普通单步采样
- H(s) > H_max：模型非常不确定，N=N_max，全力探索

#### 2. 多路径采样与标点停止

当 N > 1 时，独立采样 N 条候选路径，每条路径从当前位置开始采样直到遇到**标点符号**（. ? ! : ; ) ] \n）或满足停止条件。标点停止的设计使 CNTP 在**句子级别**进行局部最优选择，而非 token 级别或全答案级别。

#### 3. 困惑度排序选择

对每条候选路径 s_i，计算句子级困惑度：PPL(s_i) = exp(ℒ(s_i) / |s_i|)，其中 ℒ(s_i) = -Σ_t log p(w_t | s_{<t})

选择困惑度最低的路径作为最佳续写。该策略利用模型自身的似然函数作为"裁判"，无需外部反馈。

#### 4. 与 Self-Consistency 的结合

CNTP 可以作为 Self-Consistency 的"内层"优化：在 SC 的每个独立推理链中使用 CNTP 提升单链质量，然后再进行多数投票。因此 CNTP 与 SC 是正交互补的。

### 复杂度分析

| 方法 | 复杂度 | 自适应性 |
|------|--------|---------|
| 贪心解码 | O(L) | 无 |
| Beam Search (B) | O(L × B) | 无 |
| Self-Consistency (N_sc) | O(N_sc × L) | 无 |
| CNTP | O(L × (1 + p(N_max-1))) | 有 |

其中 p 是高熵步骤比例，实际中 p ≪ 1，因此 CNTP 的计算开销远低于 Beam Search 和 SC。

### 理论保证

**定理 1**：在两个温和假设（正确路径有最低困惑度；高熵意味着正确 token 概率低）下：
1. CNTP 生成正确完整序列的概率 ≥ 单样本解码
2. 平均计算成本严格低于 L × N_max

## 实验关键数据

### 主实验：Llama-3.1-8B-Instruct

| 方法 | GSM8K | MATH | StrategyQA |
|------|-------|------|-----------|
| Greedy Decoding | 79.8 | 41.5 | 72.9 |
| Stochastic Decoding | 79.4±0.8 | 41.5±1.2 | 72.0±0.7 |
| **CNTP (Ours)** | **81.6±0.6** | **47.1±1.7** | **73.2±0.2** |
| Beam Search (beam=5) | 82.3 | 48.0 | 72.9 |
| SC (40 paths) | 84.8 | 56.0 | 76.2 |
| **CNTP + SC (40 paths)** | **85.2** | **57.5** | **76.3** |

### DeepSeek-R1-Distill-Qwen-1.5B

| 方法 | GSM8K | MATH | StrategyQA |
|------|-------|------|-----------|
| Greedy Decoding | 64.6 | 32.5 | 53.6 |
| Stochastic Decoding | 61.6±1.1 | 27.9±3.7 | 51.7±1.2 |
| **CNTP (Ours)** | **65.7±0.7** | **37.7±1.7** | **53.0±1.3** |
| SC (40 paths) | 78.3 | 29.5 | 47.7 |
| **CNTP + SC (40 paths)** | **71.7** | **41.0** | **54.1** |

### TruthfulQA (Llama-2-7B-Chat)

| 方法 | Info Acc. | Truth Acc. | Truth-info Acc. |
|------|----------|-----------|----------------|
| Stochastic Decoding | 88.0±0.6 | 78.0±0.5 | 66.0±0.3 |
| Greedy Decoding | 78.5 | 79.1 | 57.6 |
| **CNTP (Ours)** | **89.2±1.2** | **84.8±0.5** | **74.0±1.1** |

### 多模态实验 (MMVet / MathVista)

| 方法 | Llama-3.2-11B MMVet | Llama-3.2-11B MathVista |
|------|-------------------|----------------------|
| Greedy | 48.0 | 53.5 |
| Stochastic | 47.7 | 53.0 |
| **CNTP** | **53.5 (+5.5)** | **58.5 (+5.0)** |

### 消融实验

**不确定性度量方式比较**：

| 度量方式 | GSM8K | StrategyQA | MATH | TruthfulQA |
|---------|-------|-----------|------|-----------|
| Max token prob | 次优 | 次优 | 次优 | 次优 |
| Max-2nd prob | 次优 | 次优 | 次优 | 次优 |
| **Entropy (Ours)** | **最优** | **最优** | **最优** | **最优** |

**试验数与熵的关系策略**：

| 策略 | GSM8K | StrategyQA | TruthfulQA |
|------|-------|-----------|-----------|
| 固定试验数 (N=6) | 81.1 | 72.7 | 3.80 |
| 负相关（高熵少试验） | 81.2 | 72.7 | 3.80 |
| **正相关（高熵多试验）** | **81.6** | **73.2** | **74.0** |

**Best-of-N vs CNTP (GSM8K)**：

| N | 2 | 5 | 10 | 20 | 40 |
|---|---|---|----|----|-----|
| Best-of-N (全答案PPL) | 79.2 | 79.5 | 78.2 | 77.3 | 76.1 |
| **CNTP** | — | — | **81.6** | — | — |

全答案级困惑度选择不如CNTP的句子级局部选择。

### 关键发现

1. **CNTP 在单链设置下全面超越贪心和随机解码**：MATH 上比贪心+5.6%，TruthfulQA 真实性+5.7%
2. **与 SC 正交互补**：CNTP+SC 在多数任务上优于纯 SC
3. **熵是最佳不确定性度量**：考虑整个词表分布信息，优于基于 top-1/top-2 概率的启发式方法
4. **句子级而非全答案级PPL选择至关重要**：Best-of-N 使用全答案 PPL 反而效果退化
5. **多模态同样有效**：LLaVA-CoT 和 Llama-3.2-Vision 上均有提升
6. **存在最优 N_max 范围**：大约 [10, 30]，过大会出现探索-利用失衡

## 亮点与洞察

- 人类类比非常直觉且有效：不确定时多想几条路，选最有把握的——这是 CNTP 的核心哲学
- **句子级局部最优**是关键创新：在标点处截断比在 token 级或全答案级都更有效
- 作为 training-free 方法，CNTP 可即插即用于任何自回归模型，部署成本极低
- 与 Entropix（并行工作）理念相似但有关键差异：CNTP 的标点停止+正相关采样策略

## 局限与展望

- 引入额外 token 计算，虽然远低于 Beam Search/SC，但仍增加推理延迟
- 超参数 H_min=0.01、H_max=1.5、N_max=10 在所有实验中固定，未针对不同任务/模型调优
- 仅在中等规模模型上验证（≤11B），未在 70B+ 大模型上测试
- 标点集合的选择可能依赖语言/任务，跨语言泛化性待验证
- 可结合 speculative decoding 或 vLLM 进一步加速

## 相关工作与启发

- **与 Self-Consistency 互补**：SC 在全链层面投票，CNTP 在子句层面优化，二者组合效果最佳
- **与 Entropix 并行**：都利用熵指导采样策略，但 CNTP 独创标点停止机制和正相关采样
- **与 CoT/ToT 的区隔**：不依赖 prompt engineering 或搜索树结构，完全在解码层面操作
- 启发：未来可将 CNTP 扩展到自回归图像生成等非文本领域

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ⭐⭐⭐⭐ 句子级熵自适应采样思路新颖 |
| 实验充分度 | ⭐⭐⭐⭐ 6个数据集+多模型+消融 |
| 实用价值 | ⭐⭐⭐⭐ training-free即插即用 |
| 写作质量 | ⭐⭐⭐⭐ 动机清晰、理论+实验结合 |
| 总体推荐 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [On Support Samples of Next Word Prediction](on_support_samples_of_next_word_prediction.md)
- [LaTIM: Measuring Latent Token-to-Token Interactions in Mamba Models](latim_measuring_latent_token-to-token_interactions_in_mamba_models.md)
- [MEXMA: Token-level Objectives Improve Sentence Representations](mexma_token-level_objectives_improve_sentence_representations.md)
- [Advancing Sequential Numerical Prediction in Autoregressive Models](advancing_sequential_numerical_prediction_in_autoregressive_models.md)
- [Generating Plausible Distractors for Multiple-Choice Questions via Student Choice Prediction](distractor_gen_multiple_choice.md)

<!-- RELATED:END -->
