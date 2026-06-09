---
title: >-
  [论文解读] Evaluating Neuron Explanations: A Unified Framework with Sanity Checks
description: >-
  [ICML2025][可解释性][神经元解释] 提出 NeuronEval 统一框架，将 19 种现有神经元解释评估方法形式化为同一数学范式，并设计 Missing Labels / Extra Labels 两项合理性检验…
tags:
  - "ICML2025"
  - "可解释性"
  - "神经元解释"
  - "评估指标"
  - "统一框架"
  - "合理性检验"
  - "机制可解释性"
---

# Evaluating Neuron Explanations: A Unified Framework with Sanity Checks

**会议**: ICML2025  
**arXiv**: [2506.05774](https://arxiv.org/abs/2506.05774)  
**代码**: [GitHub](https://github.com/Trustworthy-ML-Lab/Neuron_Eval)  
**领域**: 可解释性 / 机制可解释性  
**关键词**: 神经元解释, 评估指标, 统一框架, 合理性检验, 机制可解释性

## 一句话总结
提出 NeuronEval 统一框架，将 19 种现有神经元解释评估方法形式化为同一数学范式，并设计 Missing Labels / Extra Labels 两项合理性检验，揭示大多数常用指标（如 Recall、AUC、top-and-random 采样下的 Correlation）不可靠，仅 Correlation(Pearson)、Cosine、AUPRC、F1 和 IoU 通过测试。

## 研究背景与动机
- **机制可解释性**近年兴起，目标是理解 DNN 内部机制，其中为单个神经元/通道/SAE 特征生成自然语言解释是重要组成部分
- 然而，现有工作在评估解释质量时使用**差异极大的指标**（Recall、IoU、Correlation、MAD 等），缺乏理论依据和标准化对比
- Huang et al. (2023) 已指出部分指标存在问题，但社区仍未形成共识
- **核心问题**：哪些评估指标能如实反映解释的好坏？是否存在指标本身对"过于泛化"或"过于特化"的解释产生系统性偏差？

## 方法详解

### NeuronEval 统一框架
将评估抽象为：给定探测数据集 $\mathcal{D}$、神经网络 $f$ 和文本描述 $t$，评估函数 $\mathcal{E}$ 输出标量分数。

**核心向量定义**：

- **神经元激活向量** $a_k \in \mathbb{R}^{|\mathcal{D}|}$：每个元素 $[a_k]_i = f_k^{0:l}(x_i)$，记录神经元 $k$ 在输入 $x_i$ 上的激活值
- **概念激活向量** $c_t \in \mathbb{R}^{|\mathcal{D}|}$：每个元素 $[c_t]_i = \mathbb{P}(t|x_i)$，表示输入中概念 $t$ 存在的概率
- **二值化函数** $B$：将激活向量转为 $\{0,1\}^n$（如 top-$\alpha$ 百分位为 1，其余为 0）

评估分数统一写为：$s_M(a_k, c_t)$，其中 $M$ 为指标名。

**统一覆盖的 18 种指标**包括：

| 类别 | 指标 |
|------|------|
| 二值指标 | Recall, Precision, F1-score, IoU, Accuracy, Balanced Accuracy |
| 连续指标 | Pearson Correlation, Spearman Correlation, Cosine Similarity |
| 面积指标 | AUC, AUPRC 及其逆版本 |
| 其他 | MAD (Mean Activation Difference), WPMI |

### 两项合理性检验（Sanity Checks）

受 Adebayo et al. (2018) 对 saliency map 合理性检验的启发，提出两项必要条件测试：

**（I）Missing Labels Test（缺失标签检验）**：

生成 $c_t^-$——将正确概念 $c_t$ 中一半标签随机置零（模拟过于特化的解释）：

$$[c_t^-]_i = \begin{cases} [c_t]_i & \text{w.p. } 0.5 \\ 0 & \text{w.p. } 0.5 \end{cases}$$

若指标可靠，$s_M(a_k, c_t^-)$ 应低于 $s_M(a_k, c_t)$。

**（II）Extra Labels Test（额外标签检验）**：

生成 $c_t^+$——在 $c_t$ 基础上随机增加等量正标签（模拟过于泛化的解释），使 $\mathbb{E}[\|c_t^+\|_1] = 2\|c_t\|_1$。

若指标可靠，$s_M(a_k, c_t^+)$ 同样应低于 $s_M(a_k, c_t)$。

**Decrease Acc** 定义：

$$\text{Decrease Acc} = \frac{1}{|K|} \sum_{k \in K} \mathbb{1}[\Delta s(k) < -\epsilon]$$

其中 $\epsilon = 0.001$。通过标准：两项检验的 Decrease Acc 均 > 90%。

### Meta-Evaluation #2：已知概念神经元上的性能
在分类层等已知 ground truth 的神经元上，计算所有 (neuron, explanation) 对的 meta-AUPRC 来直接衡量指标的区分能力。

## 实验关键数据

### Sanity Check 结果（Table 3，跨 8 个设置平均）

| 指标 | Missing Labels (实验) | Extra Labels (实验) | 通过？ |
|------|:---:|:---:|:---:|
| Recall | 98.66% | **0.00%** | ❌ |
| Precision | **45.73%** | 99.81% | ❌ |
| **F1-score** | 93.68% | 99.82% | ✅ |
| **IoU** | 93.62% | 99.81% | ✅ |
| Accuracy | 23.79% | 70.37% | ❌ |
| AUC | 94.96% | 59.18% | ❌ |
| **Correlation (Pearson)** | 99.41% | 99.92% | ✅ |
| Correlation (T&R) | 87.83% | 60.26% | ❌ |
| **Cosine** | 99.45% | 99.26% | ✅ |
| MAD | 59.81% | 99.34% | ❌ |
| **AUPRC** | 95.61% | 99.46% | ✅ |

### Meta-AUPRC 排名（Table 4，跨 10 个设置平均）

| 指标 | Avg. AUPRC | Avg. Rank |
|------|:---:|:---:|
| **Correlation** | **0.8765** | **1.60** |
| Cosine | 0.8666 | 2.30 |
| AUPRC | 0.8406 | 3.90 |
| F1/IoU | 0.8140 | 6.70 |
| Recall | 0.6722 | 11.30 |
| Spearman | 0.0853 | 16.20 |

**关键发现**：通过 Sanity Check 的 5 个指标在 Meta-AUPRC 上也名列前茅；连续指标优于二值指标（二值化损失信息）。

## 亮点与洞察
1. **统一框架价值大**：将 19 篇不同领域论文的评估方法写入同一数学形式 $s_M(a_k, c_t)$，覆盖视觉/语言/SAE/CBM/线性探针
2. **合理性检验简洁有力**：仅需随机扰动标签即可揭示指标缺陷，思路源自 saliency map sanity check，可推广到其他评估场景
3. **概念不平衡是根因**：指标失败的主要原因是无法处理类不平衡（神经元激活频率低时 AUC/Accuracy 退化），与统计学经典认知一致
4. **Top-and-random 采样有害**：被 Bills et al. (2023) 等广泛使用的 T&R 采样使 Correlation 降级为类似 Recall 的行为，无法检测过于泛化的解释
5. **实际失败模式的直接关联**：Extra Labels → 解释过泛化（描述"动物"但神经元只响应"狗"）；Missing Labels → 解释过特化（描述"黑猫"但神经元响应所有猫）

## 局限与展望
1. 合理性检验是**必要而非充分条件**——通过测试不等于评估完美，可能存在其他未捕获的失败模式
2. 仅关注 **input-based 解释**，未覆盖 output-based 解释（如对模型输出的影响）
3. 仅评估**标量激活**单元，不含注意力头等更大组件的解释
4. Cosine 指标虽通过测试，但**对平均激活值敏感**，需谨慎使用
5. 实验主要基于分类层和线性探针的 ground truth，对**隐藏层多义性神经元**的评估尚需更多验证

## 评分
- 新颖性: ⭐⭐⭐⭐ (统一框架 + sanity check 思路在机制可解释性领域具有方法论创新)
- 实验充分度: ⭐⭐⭐⭐⭐ (18 种指标 × 8/10 个设置，理论+实验双重验证)
- 写作质量: ⭐⭐⭐⭐⭐ (形式化清晰，toy example 直观，表格丰富)
- 价值: ⭐⭐⭐⭐ (为可解释性社区提供明确的评估指标选择指南，实用价值高)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] FineSteer: A Unified Framework for Fine-Grained Inference-Time Steering in Large Language Models](../../ACL2026/interpretability/finesteer_a_unified_framework_for_fine-grained_inference-time_steering_in_large_.md)
- [\[NeurIPS 2025\] Evaluating LLMs in Open-Source Games](../../NeurIPS2025/interpretability/evaluating_llms_in_open-source_games.md)
- [\[ICCV 2025\] Minerva: Evaluating Complex Video Reasoning](../../ICCV2025/interpretability/minerva_evaluating_complex_video_reasoning.md)
- [\[ACL 2025\] Establishing Trustworthy LLM Evaluation via Shortcut Neuron Analysis](../../ACL2025/interpretability/shortcut_neuron_eval.md)
- [\[ICML 2025\] Explaining, Fast and Slow: Abstraction and Refinement of Provable Explanations](explaining_fast_and_slow_abstraction_and_refinement_of_provable_explanations.md)

</div>

<!-- RELATED:END -->
