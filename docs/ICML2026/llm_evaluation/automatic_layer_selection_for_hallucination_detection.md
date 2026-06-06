---
title: >-
  [论文解读] Automatic Layer Selection for Hallucination Detection
description: >-
  [ICML 2026][LLM评测][幻觉检测] 提出 FEPoID（内在维度的首个有效峰值）作为无需训练的自动层选择准则，并结合首句截断策略（FST），在多种 QA 和摘要幻觉检测基准上持续选出接近最优的中间层，显著超越已有基线方法。
tags:
  - "ICML 2026"
  - "LLM评测"
  - "幻觉检测"
  - "中间层选择"
  - "内在维度"
  - "隐藏状态探测"
  - "大语言模型"
---

# Automatic Layer Selection for Hallucination Detection

**会议**: ICML 2026  
**arXiv**: [2605.26366](https://arxiv.org/abs/2605.26366)  
**代码**: https://github.com/DesoloYw/Automatic-Layer-Selection-for-Hallucination-Detection  
**领域**: LLM评估  
**关键词**: 幻觉检测, 中间层选择, 内在维度, 隐藏状态探测, 大语言模型

## 一句话总结
提出 FEPoID（内在维度的首个有效峰值）作为无需训练的自动层选择准则，并结合首句截断策略（FST），在多种 QA 和摘要幻觉检测基准上持续选出接近最优的中间层，显著超越已有基线方法。

## 研究背景与动机

**领域现状**：大语言模型（LLM）在实际部署中常产生流畅但事实错误的输出（幻觉），检测这些幻觉而不修改模型本身是一个关键的实用问题。已有研究表明，LLM 中间层的隐藏状态比最终层更强地编码了与幻觉相关的信号，基于此出现了隐藏状态探测（hidden-state probing）的检测范式。

**现有痛点**：虽然中间层包含更丰富的幻觉信号，但最优层的位置在不同模型架构、不同数据集之间差异很大。现有方法要么使用固定的中间层（如中间层），要么对所有候选层逐一评估，前者不可靠，后者计算代价过高。缺少一个高效且有原则的自动层选择方法。

**核心矛盾**：最优层的位置依赖于模型和数据，不存在通用的固定选择规则；同时已有的用于衡量层质量的指标（如 RankMe、曲率、梯度范数等）虽然在其他场景有用，但在幻觉检测的层选择上表现不稳定。

**本文目标**：(1) 系统评估各类层选择准则在幻觉检测中的有效性；(2) 提出一个无需训练、计算高效、跨模型/数据集鲁棒的自动层选择方法；(3) 解决表征提取时的 token 位置选择问题。

**切入角度**：作者观察到内在维度（ID）随层演变的曲线呈现稳定的多峰模式——中间层出现第一个峰值，靠近输出层出现第二个更高的峰值。作者假设第一个峰值捕捉了抽象语义信息（与幻觉检测相关），而第二个峰值主要反映表面词汇复杂度（对检测无益）。

**核心 idea**：选择内在维度曲线上的"首个有效峰值"（FEPoID）作为层选择准则，同时用首句截断（FST）去除生成末尾的噪声，两者联合实现无监督、高效的幻觉检测。

## 方法详解

### 整体框架
在隐藏状态探测框架下，预训练 LLM 保持冻结，从选定层提取最后一个 token 的表征，训练一个轻量 MLP 分类器进行幻觉检测（二分类）。输入为 prompt 和生成答案的拼接。关键问题在于如何自动选择最优层以及最优 token 位置。

### 关键设计

1. **FEPoID（内在维度首个有效峰值）**:

    - 功能：自动选择幻觉检测中最优的中间层，无需任何标注数据或训练
    - 核心思路：使用 TwoNN 估计器计算每一层表征矩阵 $\mathbf{Z}^{(\ell)} \in \mathbb{R}^{N \times d}$ 的内在维度 $d_{\text{ID}}^{(\ell)}$。对 ID 曲线找所有局部极大值，从浅到深扫描。引入前向窗口 $w$（默认 7）过滤虚假峰值：若候选峰值层 $\ell$ 满足 $d_{\text{ID}}^{(\ell)} < d_{\text{ID}}^{(\min(\ell+w, L))}$ 且窗口内 ID 单调递增，则丢弃该峰值。选择存活的最早峰值对应的层
    - 设计动机：直接取最大 ID 的层往往选到末端层（表面复杂度高但语义信息少），而首个有效峰值恰好处于抽象语义信息最丰富的位置，实验证实该层与 oracle 最优层高度一致

2. **首句截断策略（FST）**:

    - 功能：解决表征提取时 token 位置选择问题，去除生成末尾引入的噪声
    - 核心思路：用基于规则的句子边界检测器定位第一个生成句子的末尾 token，提取该位置而非整个序列最后一个 token 的隐藏状态。无需真实答案标注，也不依赖辅助 LLM
    - 设计动机：LLM（尤其是 LLaMA）生成时经常在第一句给出答案后继续生成，出现三种退化行为——不一致续写（后文与首句答案矛盾）、语义漂移（偏离问题主题）、退化重复（反复重述相同信息）。这些噪声污染了末尾 token 的表征，而首句截断有效规避了这些问题

3. **系统化层选择准则评估**:

    - 功能：全面对比 6 种现有层选择准则，建立幻觉检测场景下的基准
    - 核心思路：基于四个假设（丰富语义、任务对齐、信息压缩、高效信息容量），评估了 RankMe（信息论）、验证损失/RGN/SNR（梯度）、曲率和 ID（几何）六个准则。在多模型、多数据集上逐层训练 MLP 并记录 AUROC
    - 设计动机：这些准则在各自原始场景中表现良好，但从未在幻觉检测的自动层选择中被系统比较，实验表明没有一个能稳定胜出，从而激发了 FEPoID 的提出

## 实验关键数据

### 主实验（QA 任务）

在 5 个 QA 数据集和 2 个指令微调模型上的 AUROC 对比（提取最后生成 token 表征，$w=7$）：

| 方法 | CoQA | SQuAD | HotpotQA | TriviaQA | PsiLoQA | 平均 |
|------|------|-------|----------|----------|---------|------|
| Pred. Entropy | 0.583 | 0.570 | 0.710 | 0.686 | 0.360 | 0.582 |
| Semantic Entropy | 0.500 | 0.552 | 0.445 | 0.551 | 0.608 | 0.531 |
| Lexical Similarity | 0.678 | 0.599 | 0.729 | 0.684 | 0.408 | 0.620 |
| EigenScore | 0.525 | 0.530 | 0.599 | 0.588 | 0.508 | 0.550 |
| Probing + Val Loss | 0.671 | 0.616 | 0.768 | **0.786** | 0.784 | 0.725 |
| Probing + Curvature | 0.632 | 0.618 | 0.741 | 0.737 | 0.757 | 0.697 |
| Probing + ID | 0.671 | 0.613 | 0.693 | 0.707 | 0.737 | 0.684 |
| **Probing + FEPoID** | **0.671** | **0.638** | **0.781** | 0.752 | **0.786** | **0.725** |

*以上为 LLaMA-3.1-8B-Instruct 结果。FEPoID 在平均 AUROC 上达到最优，且在 Mistral-7B 上平均 AUROC 达 0.853，同样排名第一。*

### 摘要任务与计算效率

| 方法 | HaluEval | CNN/DM | 平均 | 计算时间(秒) |
|------|----------|--------|------|-------------|
| RankMe | 0.608 | 0.577 | 0.592 | 27.3 |
| Curvature | 0.549 | 0.592 | 0.571 | 45.2 |
| Val Loss | 0.596 | 0.586 | 0.591 | 29.6 |
| RGN | 0.571 | 0.582 | 0.577 | 58.2 |
| SNR | 0.553 | 0.547 | 0.550 | 57.9 |
| **FEPoID** | **0.617** | **0.600** | **0.608** | **10.1** |

*LLaMA-3.1-8B-Instruct 上结果。FEPoID 不仅检测性能最优，计算时间仅为其他方法的 1/3 到 1/6。*

### 关键发现
- FEPoID 在 QA 和摘要两类任务、5 种模型规模（1B-8B）、base 和 instruct 两种调优策略上均稳定表现最优或接近最优，展现了极强的泛化能力
- FST 对所有基线方法均带来一致的 AUROC 提升（方法无关的增益），在 LLaMA 上提升尤为显著（因为 LLaMA 生成更容易出现末尾噪声），Fisher 分离度和轮廓系数均大幅改善
- 直接选最大 ID 层的策略在 HotpotQA、TriviaQA 等数据集上会选到过深的层，导致性能下降；而 FEPoID 通过前向窗口机制稳定地避免了这个陷阱
- 超参数 $w$ 的敏感性分析表明 FEPoID 对 $w$ 的选择非常鲁棒，性能在较大范围内保持稳定

## 亮点与洞察
- FEPoID 的设计极其优雅——仅靠 TwoNN 内在维度估计加一个前向窗口即可实现无训练、无标注的自动层选择，计算开销可以忽略不计（全部 32 层仅需约 10 秒），这使其在实际部署中极具吸引力
- FST 的"方法无关"特性非常实用：它不仅改善了隐藏状态探测，还提升了不确定性方法和词汇相似度等完全不同范式的基线，说明"末尾噪声"是一个普遍且被低估的问题
- "ID 曲线双峰假设"提供了理解 Transformer 层级表征的新视角：中间峰值 = 抽象语义，末端峰值 = 表面复杂度，这一洞察可迁移到其他需要选择中间层表征的下游任务

## 局限与展望
- 实验仅覆盖 1B-8B 规模的模型，更大模型（70B+）的层选择行为可能不同，FEPoID 的双峰假设是否仍成立有待验证
- FST 依赖规则式句子边界检测，对非英语语言或非自然句子结构的生成（如代码、数学推导）可能不适用
- 当前仅在 QA 和摘要任务上验证，开放式生成（如对话、创意写作）中幻觉的定义和分布不同，泛化性有待测试
- 可探索将 FEPoID 的层选择动态化——针对不同输入样本选择不同层，或组合多层表征以进一步提升检测性能

## 相关工作与启发
- **INSIDE**（Chen et al., 2024）：利用 LLM 内部状态进行幻觉检测，固定选择中间层，FEPoID 提供了更优的自动化替代
- **Semantic Entropy**（Farquhar et al., 2024）：从语义层面估计不确定性，但需要多次采样，本文的隐藏状态探测方法仅需单次前向传播
- **EigenScore**（Chen et al., 2024）：基于隐藏状态协方差谱性质评估表征质量，但其层选择策略次优
- **ID 与层选择的关系**：Cheng et al.（2025）发现最大 ID 附近的层最先迁移到下游任务，本文进一步细化为"首个有效峰值才是最优选择"

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Robust Hallucination Detection in LLMs via Adaptive Token Selection](../../NeurIPS2025/llm_evaluation/robust_hallucination_detection_in_llms_via_adaptive_token_selection.md)
- [\[ICML 2026\] Building Reliable Long-Form Generation via Hallucination Rejection Sampling](building_reliable_long-form_generation_via_hallucination_rejection_sampling.md)
- [\[ICML 2026\] When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets (CAIA)](when_hallucination_costs_millions_benchmarking_ai_agents_in_high-stakes_adversar.md)
- [\[ACL 2026\] Zero-shot Large Language Models for Automatic Readability Assessment](../../ACL2026/llm_evaluation/zero-shot_large_language_models_for_automatic_readability_assessment.md)
- [\[ACL 2026\] Comprehensiveness Metrics for Automatic Evaluation of Factual Recall in Text Generation](../../ACL2026/llm_evaluation/comprehensiveness_metrics_for_automatic_evaluation_of_factual_recall_in_text_gen.md)

</div>

<!-- RELATED:END -->
