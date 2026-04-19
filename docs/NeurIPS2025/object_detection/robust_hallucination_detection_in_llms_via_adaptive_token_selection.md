---
title: >-
  [论文解读] Robust Hallucination Detection in LLMs via Adaptive Token Selection
description: >-
  [NeurIPS 2025][目标检测][hallucination detection] HaMI 将幻觉检测建模为多示例学习（MIL）问题，将生成序列视为 token 实例的"bag"，通过联合优化 token 选择和幻觉检测来自适应地定位最具指示性的 token，在四个 QA 基准上以 AUROC 大幅超越所有现有方法（最高提升 11.9%）。
tags:
  - NeurIPS 2025
  - 目标检测
  - hallucination detection
  - multiple instance learning
  - adaptive token selection
  - internal representation
  - predictive uncertainty
---

# Robust Hallucination Detection in LLMs via Adaptive Token Selection

**会议**: NeurIPS 2025  
**arXiv**: [2504.07863](https://arxiv.org/abs/2504.07863)  
**代码**: https://github.com/mala-lab/HaMI (有)  
**领域**: LLM安全 / 幻觉检测  
**关键词**: hallucination detection, multiple instance learning, adaptive token selection, internal representation, predictive uncertainty

## 一句话总结

HaMI 将幻觉检测建模为多示例学习（MIL）问题，将生成序列视为 token 实例的"bag"，通过联合优化 token 选择和幻觉检测来自适应地定位最具指示性的 token，在四个 QA 基准上以 AUROC 大幅超越所有现有方法（最高提升 11.9%）。

## 研究背景与动机

LLM 的幻觉（hallucination）问题是其安全部署的核心障碍——模型可能生成看似合理但实际不忠实或错误的内容。现有检测方法主要分两条路线：

**基于不确定性的方法**（如 Semantic Entropy、Perplexity、MARS）：依赖预测概率或多次采样的语义一致性，但性能受限于辅助 LLM 的能力

**基于内部表示的方法**（如 SAPLMA、HaloScope、CED）：利用 LLM 隐藏层表示训练二分类器，但严重依赖预定义的 token 位置

**核心矛盾**：大多数内部表示方法使用预定义 token（如第一个、最后一个或倒数第二个 token）的表示来训练检测器。但如图 1 所示，包含幻觉信息最丰富的 token 位置会因回复长度和幻觉实体分布的不同而显著变化。预定义位置会遗漏幻觉信息集中的关键 token。

**切入角度**：幻觉通常仅出现在回复中少数 token 的位置（如错误的实体名词），这与 MIL 的假设天然吻合——正样本 bag 中仅有少数正实例，负样本 bag 中所有实例均为负。将幻觉检测转化为 MIL 问题，可以自适应地在训练中学习哪些 token 最具幻觉指示性。

## 方法详解

### 整体框架

HaMI 框架包含两大模块：

1. **MIL 驱动的自适应 Token 选择（ATS）**：将每个生成序列视为一个 bag，bag 级标签为幻觉/可信。通过 MIL 损失联合优化 token 选择和幻觉检测。
2. **预测不确定性增强模块**：将多层次不确定性信息融入 token 内部表示，增强判别能力。

### 关键设计

1. **MIL 建模**：

    - 正样本 bag $\mathcal{B}^+$（含幻觉的回复）：仅少数 token 是真正的幻觉 token（正实例）
    - 负样本 bag $\mathcal{B}^-$（正确回复）：所有 token 均为负实例
    - 幻觉检测器 $f_\theta$ 为每个 token 分配幻觉分数
    - 选择每个 bag 中幻觉分数最高的 top-k 个 token 作为显著 token（$k = \lfloor 0.1 \times l \rfloor + 1$，l 为序列长度）

2. **MIL 损失函数**：

    - 最大化正 bag 显著 token 与负 bag 最难负 token 之间的判别边际
    - $\mathcal{L}_{MIL} = 1 - \|\frac{1}{k}\sum_{i^+ \in \mathcal{I}_{top-k}^+} f_\theta(h_{i^+})\|_2 + \|\frac{1}{k}\sum_{i^- \in \mathcal{I}_{top-k}^-} f_\theta(h_{i^-})\|_2$
    - 正 bag 的 top-k token 得分应高，负 bag 的 top-k token 得分应低

3. **平滑性约束**：

    - 利用 token 生成的序列特性，相邻 token 的幻觉分数应平滑变化
    - $\mathcal{L}_{Smooth} = (f_\theta(h_i) - f_\theta(h_{i-1}))^2$
    - 总损失：$\mathcal{L}_{ATS} = \mathcal{L}_{MIL} + \mathcal{L}_{Smooth}$

4. **预测不确定性增强**：

    - 三个层次的不确定性度量：
        - Token 级：预测概率 $P^t$
        - 句子级：困惑度（perplexity）$P^s$
        - 语义一致性级：基于多次采样的语义熵 $P^c$
    - 增强公式：$h' = (1 + \lambda \cdot P_{\text{uncertainty}}) \cdot h$
    - 默认使用语义一致性 $P^c$，$\lambda = 1.0$

### 损失函数 / 训练策略

- 检测器为两层 MLP，隐藏维度 256
- 四个基准数据集：TriviaQA、SQuAD、Natural Questions、BioASQ
- 每个数据集 2000 对 QA 训练、800 对测试
- 多次采样：每个问题提示 LLM 6 次
- 正确性标签由 GPT-4.1 判定
- 评估指标：AUROC
- 模型：LLaMA-3.1-8B、Mistral-Nemo-Instruct (12B)、LLaMA-3.3-Instruct-70B

## 实验关键数据

### 主实验（AUROC）

**LLaMA-3.1-8B 上的对比**：

| 方法 | TriviaQA | SQuAD | NQ | BioASQ | 备注 |
|------|----------|-------|-----|--------|------|
| Perplexity | 0.732 | 0.649 | 0.659 | 0.709 | 基线 |
| Semantic Entropy | 0.828 | 0.787 | 0.773 | 0.757 | 强基线 |
| SAPLMA | 0.835 | 0.769 | 0.781 | 0.821 | 内部表示 |
| MARS-SE | 0.824 | 0.780 | 0.777 | 0.744 | 增强版 |
| HaMI* (Ours) | 0.854 | 0.783 | 0.788 | 0.823 | 无不确定性增强 |
| HaMI (Ours) | **0.897** | **0.826** | **0.820** | **0.836** | 完整版 |

**LLaMA-3.3-Instruct-70B 上的对比**：

| 方法 | TriviaQA | SQuAD | NQ | BioASQ |
|------|----------|-------|-----|--------|
| Semantic Entropy | 0.819 | 0.643 | 0.769 | 0.772 |
| SAPLMA | 0.842 | 0.672 | 0.817 | 0.748 |
| HaMI (Ours) | **0.891** | **0.774** | **0.846** | **0.825** |

### 消融实验

**自适应 Token 选择 vs 预定义 Token**：

| Token 位置策略 | TriviaQA | SQuAD | 平均提升 |
|---------------|----------|-------|---------|
| First token | 0.849 | 0.774 | - |
| Before Last | 0.878 | 0.778 | +1.6% |
| Last token | 0.890 | 0.804 | +3.3% |
| ATS (Ours) | **0.897** | **0.826** | **+5.0%** |

**不确定性增强效果**：

| 不确定性类型 | TriviaQA | SQuAD | 说明 |
|-------------|----------|-------|------|
| Original（无增强） | 0.854 | 0.783 | HaMI* |
| Token 级 $P^t$ | 0.856 | 0.782 | 几乎无提升 |
| 句子级 $P^s$ | 0.871 | 0.787 | 适度提升 |
| 语义一致性 $P^c$ | **0.897** | **0.826** | 最显著提升(+6.7%) |

### 关键发现

- HaMI 在三个 LLM 上全面超越所有基线，70B 模型上优势更加显著（超 SE 平均 11.5%）
- 跨数据集泛化实验：HaMI 在未见数据集上的性能下降不超过 4.5%，远优于其他方法
- 人工评估显示 ATS 模块对幻觉 token 的召回率达 0.84，且没有 token 级监督
- Last token 是最好的预定义位置选择，但 ATS 在所有情况下都更好且更鲁棒
- 语义一致性增强（$P^c$）效果最佳，但句子级困惑度（$P^s$）也优于所有需要多次采样的外部方法（如 SE），且仅需一次生成

## 亮点与洞察

- **MIL 建模的优雅适配**：幻觉检测的本质——"正样本序列中仅少数 token 含幻觉"——与 MIL 假设完美匹配，这一建模思路非常自然且理论合理
- **端到端联合优化**：首次实现了 token 选择和幻觉检测的联合优化，避免了两阶段方法中 token 选择不匹配的问题
- **平滑性约束**：利用自回归生成的序列性质，相邻 token 的幻觉分数应连续变化，这一归纳偏置简单但有效
- **实用价值高**：HaMI* 版本（无需多次采样）已经可以超越大多数需要外部 LLM 辅助的方法，部署成本低

## 局限与展望

- 语义一致性增强 $P^c$ 需要多次采样和外部 NLI 模型判断语义等价，增加推理成本
- 训练依赖 GPT-4.1 提供的正确性标签，标签质量受 GPT-4.1 能力限制
- 仅在 QA 任务上评估，对摘要生成、对话等其他幻觉场景的适用性未验证
- k 的设定（top 10% + 1）比较启发式，不同任务可能需要不同的 k
- 检测器使用固定层的内部表示，最优层的选择需要验证集调参

## 相关工作与启发

- 与 SAPLMA（MLP 探针 + 正确性标签）直接对比：HaMI 用 MIL 替代固定 token 位置，性能提升 5-8%
- 与 Semantic Entropy（多次采样语义一致性）互补：HaMI 可以用其 $P^c$ 度量来增强内部表示
- 与 HaloScope（成员估计分数转标签）的区别：HaMI 聚焦 token 级自适应选择而非新的训练信号
- MIL 在医学图像（whole slide image 分类）中已有广泛应用，本文首次将其引入 NLP 幻觉检测
- 启发：token 级的"bag of instances"视角可能适用于其他序列级任务中需要定位关键位置的场景（如事实核查、归因分析等）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Test-Time Adaptive Object Detection with Foundation Model](test-time_adaptive_object_detection_with_foundation_model.md)
- [Thinking in Latents: Adaptive Anchor Refinement for Implicit Reasoning in LLMs](../../ICLR2026/object_detection/thinking_in_latents_adaptive_anchor_refinement_for_implicit_reasoning_in_llms.md)
- [Sample Complexity of Distributionally Robust Average-Reward Reinforcement Learning](sample_complexity_of_distributionally_robust_average-reward_reinforcement_learni.md)
- [Efficient Test-Time Adaptive Object Detection via Sensitivity-Guided Pruning](../../CVPR2025/object_detection/efficient_test-time_adaptive_object_detection_via_sensitivity-guided_pruning.md)
- [ReHARK: Refined Hybrid Adaptive RBF Kernels for Robust One-Shot Vision-Language Adaptation](../../CVPR2026/object_detection/rehark_refined_hybrid_adaptive_rbf_kernels_for_robust_one-shot_vision-language_a.md)

<!-- RELATED:END -->
