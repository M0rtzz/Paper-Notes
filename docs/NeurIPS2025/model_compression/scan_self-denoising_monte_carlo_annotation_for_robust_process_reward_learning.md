---
title: >-
  [论文解读] SCAN: Self-Denoising Monte Carlo Annotation for Robust Process Reward Learning
description: >-
  [NeurIPS 2025][模型压缩][过程奖励模型] 提出 SCAN 框架，通过分析 Monte Carlo 注释中的噪声分布，设计自去噪采样策略和鲁棒学习损失，仅用 1.5B 模型生成的 101K 样本训练的 PRM 即超越人工标注数据集 PRM800K 的效果。
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "过程奖励模型"
  - "Monte Carlo估计"
  - "噪声标签"
  - "自去噪"
  - "数学推理"
---

# SCAN: Self-Denoising Monte Carlo Annotation for Robust Process Reward Learning

**会议**: NeurIPS 2025  
**arXiv**: [2509.16548](https://arxiv.org/abs/2509.16548)  
**代码**: [有](https://scan-prm.github.io)  
**领域**: 图像复原  
**关键词**: 过程奖励模型, Monte Carlo估计, 噪声标签, 自去噪, 数学推理

## 一句话总结

提出 SCAN 框架，通过分析 Monte Carlo 注释中的噪声分布，设计自去噪采样策略和鲁棒学习损失，仅用 1.5B 模型生成的 101K 样本训练的 PRM 即超越人工标注数据集 PRM800K 的效果。

## 研究背景与动机

过程奖励模型（PRM）通过步骤级评估来引导 LLM 的推理过程，在数学推理等复杂任务中表现出色。然而 PRM 面临数据标注困境：

**人工标注成本极高**：PRM800K 等数据集虽然质量好但标注代价大，难以规模化。

**Monte Carlo（MC）估计噪声大**：使用模型进行多次 rollout 来估计步骤正确性是有前景的替代方案，但噪声比例高，模型容易过拟合。

**现有去噪方法依赖强模型蒸馏**：如使用 72B 的 critic 模型来过滤数据，本质上是将大模型能力蒸馏到小模型。

本文的问题是：**能否不依赖外部强监督，仅通过挖掘 MC 估计自身的去噪潜力和设计鲁棒学习策略来训练高质量 PRM？**

作者首先对 MC 注释中的噪声分布进行了系统研究。定义了 self-confidence 指标 $SC_\theta(q)$ 来量化 completer 模型对问题的信心，发现噪声主要来自两类：

- **低估（Under-Estimation, $t_{pred} < t_{true}$）**：模型能力不足，即使正确前缀也无法生成正确 rollout，导致过早判定错误。集中在低 self-confidence 区域。
- **高估（Over-Estimation, $t_{pred} > t_{true}$）**：模型具有纠错能力，在错误步骤后仍能生成正确 rollout，导致错误位置延迟检测。

## 方法详解

### 整体框架

SCAN 包含两个核心模块：(1) 高效数据合成框架——通过选择性采样减少推理成本；(2) 鲁棒学习策略——通过噪声容忍标签和置信度重加权抵抗噪声。

### 关键设计

1. **选择性 MC 注释（Efficient Data Synthesis）**: 

    - **只注释负样本**：生成响应后，直接答案正确的（正样本）直接用于训练，不进行逐步 MC 注释。因为高 self-confidence 区域的正样本噪声极低（Observation 4），节省了 80 次 rollout/样本的成本。
    - **仅对高置信度负样本做逐步注释**：筛选 $SC_\pi(q_i) > \epsilon$ 的负样本进行步骤级 MC 估计。这确保 100% 的 MC 注释样本都被纳入训练集。

2. **噪声容忍标签（Noise-tolerant Labeling）**: 
   针对高估问题（$t_{pred} > t_{true}$，Observation 5 表明误差通常在真实错误位置附近），对预测错误位置前 $d$ 步的标签使用 soft label $\hat{y}_t = \min(c_t / SC_\pi(q), 1)$，而非硬标签。这允许模型从噪声位置学习而不过拟合。

3. **置信度重加权（Confidence-wise Reweighting）**: 
   MC 标注的正确性概率 $c_t$ 受 completer 模型能力影响，与真实正确性 $c_t^*$ 存在偏差。通过 self-confidence 校正：$\hat{c}_i^* = \min(c_i / SC_\pi(q), 1)$。核心思想是：强模型和弱模型标注同一样本时，校正后的分数应一致——用 self-confidence 归一化消除模型能力偏差。

### 损失函数 / 训练策略

改进的 BCE 损失：

$$\mathcal{L}_{\text{SCAN}}(\theta) = -\mathbb{E}_{(x_{\leq t}, y_t) \sim D_{\text{final}}} [y_t \log P_\theta(y_t|q, \mathbf{x}_{\leq t}) + (1-y_t) \log(1 - P_\theta(y_t|q, \mathbf{x}_{\leq t}))]$$

其中标签 $\hat{y}_t$ 在错误位置附近使用 soft label，并通过置信度重加权。

## 实验关键数据

### 主实验（Best-of-8, Policy: Qwen2.5-Math-7B-Instruct）

| 模型 | 训练样本 | 标注方式 | GSM8K | MATH | College Math | Olympiad | Avg |
|------|---------|---------|-------|------|-------------|----------|-----|
| Majority Vote@8 | — | — | 96.9 | 87.3 | 47.4 | 43.0 | 68.7 |
| RLHFlow-PRM-8B | 253K | MC | 96.8 | 87.3 | 47.9 | 43.9 | 69.0 |
| Qwen2.5-Math-PRM-7B | 1500K | MC+KD | 96.8 | 88.1 | 47.7 | 47.6 | 70.1 |
| PRM800K | 264K | 人工 | 97.0 | 87.6 | 47.7 | 45.0 | 69.3 |
| **Scan-Base** | **101K** | **MC** | **97.1** | **86.9** | **47.8** | **44.4** | **69.1** |
| **Scan-Pro** | **197K** | **MC** | **97.1** | **87.3** | **48.1** | **47.7** | **70.1** |

### 消融实验（ProcessBench F1）

| 配置 | GSM8K F1 | MATH F1 | Olympiad F1 | Avg F1 | 说明 |
|------|----------|---------|-------------|--------|------|
| Baseline（无去噪） | — | — | — | ~35 | 快速过拟合 |
| + Selective Sampling | — | — | — | ~45 | 减少正样本噪声 |
| + Tolerance Labeling | — | — | — | ~52 | 抗高估噪声 |
| + Confidence Reweight | — | — | — | **59.1** | 消除模型能力偏差 |
| Qwen2.5-7B-Ins (critic) | 26.8 | 25.7 | 14.2 | 19.9 | 原始模型 |
| **Scan-Pro** | **80.9** | **65.3** | **45.9** | **59.1** | 自训练后 |

### 关键发现

- **仅用 1.5B 模型即可生成高质量数据**：Scan-Base 用 Qwen2.5-Math-1.5B 生成 101K 样本，PRM 性能接近 264K 人工标注的 PRM800K
- **自我提升显著**：Qwen2.5-7B-Ins 的 ProcessBench F1 从 19.9 提升到 59.1（+39.2），超越 70B 级 critic 模型
- **容忍距离 $d=2$ 最优**：$d=0$（硬标签）导致严重过拟合，$d=n$（全软标签）引入过多噪声
- **无去噪策略的 baseline 很快过拟合**：验证了 MC 噪声对 PRM 训练的严重影响
- **数据来源多样性有帮助**：Scan-Pro 融合三个模型的数据比单一来源更好

## 亮点与洞察

- **噪声分布的系统分析是核心贡献**：首次从 self-confidence 视角揭示 MC 注释中低估和高估噪声的来源和分布规律
- 自去噪策略极其高效——不需要外部强模型，仅利用 completer 自身的 self-confidence
- 置信度重加权巧妙解决了多模型混合标注的一致性问题
- 仅 101K 样本 + 1.5B 模型 = 媲美人工标注，验证了"小模型 + 好策略"的可行性

## 局限与展望

- 容忍距离 $d$ 需手动选择，可探索自适应设定
- self-confidence 度量依赖足够的采样（16次），采样不足时估计不准
- 目前仅在数学推理验证，代码推理/通用推理的噪声分布可能不同
- 正样本直接跳过 MC 注释可能遗漏少量隐蔽错误

## 相关工作与启发

- Math-Shepherd 开创了 MC 方法用于 PRM 数据合成，但未深入研究噪声问题
- PRM800K 是人工标注的标杆，本文证明了合成数据在正确策略下可以匹敌
- 噪声标签学习的思路可借鉴更多该领域的技术（如 MixUp、label smoothing 等）
- 该框架对其他需要过程监督的场景（如代码生成、多步推理）具有参考价值

## 评分

- 新颖性: ⭐⭐⭐⭐ — 从噪声分布视角切入 PRM 数据合成是新角度
- 实验充分度: ⭐⭐⭐⭐⭐ — BoN + ProcessBench 双评估、完整消融、多模型扩展
- 写作质量: ⭐⭐⭐⭐⭐ — 预备研究→动机→方法→实验的逻辑链非常流畅
- 价值: ⭐⭐⭐⭐⭐ — 低成本PRM训练方案，对推理增强有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Beyond Communication Overhead: A Multilevel Monte Carlo Approach for Mitigating Compression Bias in Distributed Learning](../../ICML2025/model_compression/beyond_communication_overhead_a_multilevel_monte_carlo_approach_for_mitigating_c.md)
- [\[NeurIPS 2025\] Learning to Better Search with Language Models via Guided Reinforced Self-Training](learning_to_better_search_with_language_models_via_guided_reinforced_self-traini.md)
- [\[NeurIPS 2025\] How to Build a Consistency Model: Learning Flow Maps via Self-Distillation](how_to_build_a_consistency_model_learning_flow_maps_via_self-distillation.md)
- [\[NeurIPS 2025\] AutoJudge: Judge Decoding Without Manual Annotation](autojudge_judge_decoding_without_manual_annotation.md)
- [\[NeurIPS 2025\] Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA](robust_federated_finetuning_of_llms_via_alternating_optimization_of_lora.md)

</div>

<!-- RELATED:END -->
