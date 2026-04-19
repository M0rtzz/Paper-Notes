---
title: >-
  [论文解读] SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs
description: >-
   利用稀疏自编码器（SAE）发现 LLM 中意外语言切换与目标语言特征异常高预激活值相关，提出 SASFT 方法在 SFT 训练中约束语言特征预激活值，将意外代码切换降低 50% 以上。
tags:

---

# SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs

## 基本信息

- **会议**: ICLR 2026
- **arXiv**: [2507.14894](https://arxiv.org/abs/2507.14894)
- **代码**: [GitHub](https://github.com/Aatrox103/SASFT)
- **领域**: 自然语言处理 / 大语言模型
- **关键词**: Code-Switching, Sparse Autoencoder, Multilingual LLMs, SFT, Language Features

## 一句话总结

利用稀疏自编码器（SAE）发现 LLM 中意外语言切换与目标语言特征异常高预激活值相关，提出 SASFT 方法在 SFT 训练中约束语言特征预激活值，将意外代码切换降低 50% 以上。

## 研究背景与动机

### 问题背景
多语言 LLM（如 Qwen-3、Llama-4、Gemma-3）在生成回复时经常出现意外的语言切换（code-switching），例如英文问题的回答中突然插入中文或韩文，严重影响用户体验。

### 现有方法的局限

**唯一已知尝试**：Guo et al. (2025) 使用 GRPO + 语言一致性奖励处理 DeepSeek-R1 的代码切换问题，但缺乏机理分析，效果有限；

**缺乏根本理解**：现有工作未深入分析代码切换的内部机制。

### 核心发现
通过 SAE 分析发现：
1. LLM 中存在**语言特异性特征**（language-specific features）——残差流中仅在处理特定语言 token 时有大投影值的方向；
2. 意外代码切换发生时，目标语言特征的**预激活值异常升高**；
3. 消融实验证实降低这些特征可减少代码切换。

## 方法详解

### 整体框架

SASFT 分两步：(1) 识别 LLM 中的语言特异性特征；(2) 在 SFT 训练中引入辅助损失约束这些特征。

### 1. 稀疏自编码器（SAE）回顾

给定残差流 $\mathbf{x} \in \mathbb{R}^N$，SAE 计算特征激活 $\mathbf{a} \in \mathbb{R}^M$（$M \gg N$）：

$$
\mathbf{f(x)} = \mathbf{W}_{\text{enc}} \mathbf{x} + \mathbf{b}_{\text{enc}}
$$

$$
\mathbf{a(x)} = \text{ReLU}(\mathbf{f(x)})
$$

本文关注**预激活值** $\mathbf{f(x)}$ 而非激活值 $\mathbf{a(x)}$，因为后者忽略了有意义的负预激活值。

### 2. 语言特异性特征识别

使用 Deng et al. (2025) 的方法度量特征的单语性。对特征 $s$ 和语言 $L$，计算：

$$
\nu_s^L = \mu_s^L - \gamma_s^L
$$

其中 $\mu_s^L$ 是该特征在语言 $L$ 上的平均激活，$\gamma_s^L$ 是在其他语言上的平均激活。$\nu$ 值最高的特征被视为语言特异性特征。

### 3. 代码切换的机理分析

**关键发现 1**：代码切换前，目标语言特征的预激活值逐渐升高（图 3）。

**关键发现 2**：通过方向消融（directional ablation）减去语言特征可降低代码切换率（图 4）：

$$
\mathbf{x}' \leftarrow \mathbf{x} - \lambda \mathbf{d}
$$

但推理时消融有缺点：(1) 需要大幅降低预激活值，可能损害其他能力；(2) 需要外部干预，增加推理开销。

### 4. SASFT 训练目标

在 SFT 期间引入辅助损失，教 LLM 自主保持适当的语言特征预激活值：

$$
L_{\text{reduce}} = \mathbb{E}_{\mathcal{D}_j \sim \mathcal{D} \setminus \{\mathcal{D}_L\}}\left[\mathbb{E}_{\mathbf{x} \sim \mathcal{D}_j}\left[\sum_{s \in \mathcal{S}_L} \text{ReLU}(\mathbf{f}_s(\mathbf{x}) - \alpha_j)\right]\right]
$$

其中 $\mathcal{S}_L$ 是语言 $L$ 的特异性特征集，$\alpha_j$ 是预估的平均预激活值阈值。

最终训练损失：

$$
L_{\text{training}} = L_{\text{cross-entropy}} + \lambda L_{\text{reduce}}
$$

### 设计要点

- 排除目标语言数据 $\mathcal{D}_L$（同语言生成不算代码切换）
- 阈值 $\alpha_j$ 不设为零，因为预激活平均值可能为负
- 可跨多层应用以获得更稳定效果

## 实验

### 主实验：代码切换率对比

| 模型 | 方法 | CS→中文 | CS→俄语 | CS→韩语 |
|------|------|---------|---------|---------|
| Gemma-2-2B | SFT (基线) | 0.74% | 0.57% | 3.45% |
| | SFT+GRPO | 0.74 (0%) | 0.49 (-14%) | 3.44 (0%) |
| | SFT+Penalty | 0.67 (-10%) | 0.41 (-27%) | 1.18 (-66%) |
| | **SASFT** | **0.42 (-43%)** | **0.22 (-61%)** | **0.73 (-79%)** |
| Gemma-2-9B | SFT (基线) | 0.78% | 0.12% | 0.81% |
| | **SASFT** | **0.41 (-47%)** | **0.01 (-94%)** | **0.13 (-84%)** |
| Llama-3.1-8B | SFT (基线) | 1.16% | 0.67% | 0.57% |
| | **SASFT** | — | — | — |

### 消融实验：不同组件的影响

| 配置 | CS→中文 (↓) | MMLU (↑) | HumanEval (↑) |
|------|------------|---------|--------------|
| SFT 基线 | 0.78% | 69.2 | 42.1 |
| SASFT (单层) | 0.52% | 69.5 | 42.8 |
| SASFT (多层) | **0.41%** | **69.8** | **43.2** |
| 推理时消融 | 0.45% | 67.3 | 40.5 |

### 关键发现

1. **SASFT 在大多数情况下将代码切换降低 50% 以上**，韩语场景甚至实现 100% 消除；
2. **显著优于 GRPO**：GRPO 在多数设置下几乎无效（0% 改善），而 SASFT 持续有效；
3. **不损害多语言能力**：在 MMLU、HumanEval、Flores-200 等 6 个基准上性能保持甚至提升；
4. **多层应用效果更稳定**：跨层 SASFT 比单层更鲁棒；
5. **降低比增强更有效**：降低非目标语言特征优于增强源语言特征；
6. **训练式方法优于推理干预**：SASFT 改变模型内部行为，无推理额外开销。

## 亮点

- 首次深入分析 LLM 意外代码切换的内部机制，揭示与语言特征预激活值的因果关系
- 从推理干预到训练时约束的巧妙转换，解决了推理干预的两大缺陷
- 通用性强：在 Gemma-2、Llama-3.1、Qwen-3 三个系列五个模型上验证
- 辅助损失设计优雅，利用 ReLU 门控仅惩罚超过阈值的预激活值

## 局限性

- 需要对应模型的 SAE（Qwen-3 需自训练 SAE，额外开销未量化）
- 语言特异性特征识别依赖多语言校准数据
- 论文主要关注中文、韩文、俄文三种语言，更多语言的泛化性待验证
- 对代码切换的定义基于脚本检测，可能遗漏细粒度的词汇混用

## 相关工作

- **LLM 代码切换**: Guo et al. (2025) 发现并尝试用 GRPO 解决 DeepSeek-R1 的代码切换
- **SAE 分析**: Deng et al. (2025) 发现 LLM 中的语言特异性特征
- **多语言 LLM**: Qwen-3 (Yang et al., 2025), Llama-4 (Meta, 2025), Gemma-3 (Team et al., 2025)
- **机械可解释性**: 稀疏自编码器用于理解 LLM 内部表征

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 首次将 SAE 可解释性与代码切换问题结合，从机理到解法一气呵成
- 技术深度：⭐⭐⭐⭐ — 完整的分析-发现-解决链路，预激活值约束设计巧妙
- 实验充分度：⭐⭐⭐⭐⭐ — 5 个模型 × 3 种语言 × 6 个基准，全面覆盖
- 实用价值：⭐⭐⭐⭐⭐ — 直接解决多语言 LLM 部署中的痛点问题

<!-- RELATED:START -->

## 相关论文

- [ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Finetuning, and Decoding the Curse of Multilinguality](atlas_adaptive_transfer_scaling_laws_for_multilingual_pretraining_finetuning_and.md)
- [KnowCoder-X: Boosting Multilingual Information Extraction via Code](../../ACL2025/multilingual_mt/knowcoder-x_boosting_multilingual_information_extraction_via_code.md)
- [CruxEval-X: A Benchmark for Multilingual Code Reasoning, Understanding and Execution](../../ACL2025/multilingual_mt/cruxeval-x_a_benchmark_for_multilingual_code_reasoning_understanding_and_executi.md)
- [MMTIT-Bench: A Multilingual and Multi-Scenario Benchmark with Cognition-Perception-Reasoning Guided Text-Image Machine Translation](../../CVPR2026/multilingual_mt/mmtit-bench_a_multilingual_and_multi-scenario_benchmark_with_cognition-perceptio.md)
- [THOR-MoE: Hierarchical Task-Guided and Context-Responsive Routing for Neural Machine Translation](../../ACL2025/multilingual_mt/thor-moe_hierarchical_task-guided_and_context-responsive_routing_for_neural_mach.md)

<!-- RELATED:END -->
