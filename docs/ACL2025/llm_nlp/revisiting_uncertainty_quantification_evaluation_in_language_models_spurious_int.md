---
title: >-
  [论文解读] Revisiting Uncertainty Quantification Evaluation in Language Models: Spurious Interactions with Response Length Bias Results
description: >-
  [ACL 2025 (Main)][LLM/NLP][不确定性量化] 本文发现语言模型不确定性量化（UQ）评估中存在严重的长度偏差问题——UQ 方法和正确性度量函数都受响应长度偏差影响，二者的"互偏差"系统性地扭曲了 AUROC 排名，并在理论和实证上证明了这一点…
tags:
  - "ACL 2025 (Main)"
  - "LLM/NLP"
  - "不确定性量化"
  - "长度偏差"
  - "AUROC"
  - "评估偏差"
  - "LLM-as-a-Judge"
---

# Revisiting Uncertainty Quantification Evaluation in Language Models: Spurious Interactions with Response Length Bias Results

**会议**: ACL 2025 (Main)  
**arXiv**: [2504.13677](https://arxiv.org/abs/2504.13677)  
**代码**: 无  
**领域**: LLM评估  
**关键词**: 不确定性量化、长度偏差、AUROC、评估偏差、LLM-as-a-Judge

## 一句话总结

本文发现语言模型不确定性量化（UQ）评估中存在严重的长度偏差问题——UQ 方法和正确性度量函数都受响应长度偏差影响，二者的"互偏差"系统性地扭曲了 AUROC 排名，并在理论和实证上证明了这一点，同时发现 LLM-as-a-Judge 方法是最不受长度偏差影响的评估选择。

## 研究背景与动机

**领域现状**：不确定性量化（Uncertainty Quantification, UQ）是提高语言模型安全性和可靠性的关键技术。常见的 UQ 方法包括负序列概率（negative sequence probability）、token 级熵、语义不确定性等。评估通常使用 AUROC 来衡量 UQ 分数与任务正确性之间的区分能力——理想情况下，模型不确定性高的回答应该更可能是错误的。

**现有痛点**：UQ 方法的评估依赖"正确性函数"（correctness function）来判断模型回答是否正确。常用的正确性函数包括 ROUGE-L、BERTScore、精确匹配等。然而，这些正确性函数本身可能存在偏差——例如 ROUGE-L 倾向于给更长的回答更高的分数。同时，某些 UQ 方法也受长度影响——如负序列概率天然会给更长的回答更低的分数（更高的不确定性）。

**核心矛盾**：当 UQ 方法和正确性函数都受同一个混淆因子（如响应长度）偏差影响时，AUROC 排名会被系统性地扭曲。例如，某个 UQ 方法可能因为"恰好与正确性函数共享同样的长度偏差方向"而获得虚高的 AUROC，而非因为它确实更好地量化了不确定性。

**本文目标**：(1) 从理论上证明互偏差（mutual bias）会非随机地扭曲 AUROC 排名；(2) 在大规模实验中验证长度偏差确实在实践中扭曲了 UQ 评估；(3) 找到最不受长度偏差影响的评估方案。

**切入角度**：作者从评估指标的统计理论出发，将 UQ 评估的可靠性问题形式化为互偏差对 AUROC 排序的影响，结合严格的理论推导和大规模实证。

**核心 idea**：UQ 评估中的长度偏差是一种确定性的、可证明的系统性错误，不是随机噪声；LLM-as-a-Judge 由于不直接基于文本表面特征打分，是目前最不受长度偏差影响的正确性函数选择。

## 方法详解

### 整体框架

本文的方法论不是提出新的 UQ 技术，而是对现有 UQ 评估体系进行系统性审计。流程为：(1) 形式化定义互偏差（mutual bias）并从理论上推导其对 AUROC 的影响；(2) 选取 7 种正确性函数 × 8 种 UQ 方法 × 4 个数据集 × 4 个模型的完整实验矩阵；(3) 量化各正确性函数的长度偏差程度；(4) 分析长度偏差如何传导至 UQ 排名的扭曲；(5) 对比不同正确性函数下 UQ 方法排名的一致性。

### 关键设计

1. **互偏差的形式化定义与理论证明（Formal Mutual Bias Analysis）**:

    - 功能：从数学上证明互偏差必然导致 AUROC 排名失真
    - 核心思路：设 UQ 分数为 $u(x)$，正确性函数为 $c(x)$，若两者都与某个混淆变量 $z$（如响应长度）相关，即 $\text{Corr}(u, z) \neq 0$ 且 $\text{Corr}(c, z) \neq 0$，则 AUROC 的计算会受到 $z$ 的系统性影响。作者证明了在这种条件下，AUROC 排名的偏移方向是可预测的而非随机的——偏差方向一致的 UQ-正确性函数组合会人工抬高 AUROC，方向相反的则会压低
    - 设计动机：仅通过实验观察偏差还不够有说服力，理论证明表明这是结构性问题而非偶然现象，从根本上动摇了基于有偏正确性函数做 UQ 评估的可靠性

2. **大规模正确性函数偏差量化（Comprehensive Bias Quantification）**:

    - 功能：系统量化 7 种常用正确性函数的长度偏差程度
    - 核心思路：对每种正确性函数，计算其评分与响应长度之间的 Spearman 相关系数。测试的正确性函数包括：(i) 基于词汇的——ROUGE-L、BLEU、F1；(ii) 基于嵌入的——BERTScore、SentenceSim；(iii) LLM-as-a-Judge——GPT-4 评判、Claude 评判。在 4 个数据集 × 4 个模型的所有输出上统计
    - 设计动机：不同正确性函数的偏差程度不同，只有量化后才能评估其对 UQ 排名的具体影响。为社区提供选择正确性函数的指导

3. **UQ 排名一致性分析（UQ Ranking Consistency Analysis）**:

    - 功能：揭示不同正确性函数下 UQ 方法排名的不一致性
    - 核心思路：对同一组实验数据，分别使用不同正确性函数计算 8 种 UQ 方法的 AUROC 排名，然后分析排名之间的 Kendall τ 相关系数。如果正确性函数不引入偏差，不同函数下的 UQ 排名应该高度一致。如果排名在不同正确性函数下显著不同，说明偏差在起作用
    - 设计动机：UQ 方法的好坏应该是客观的，不应因为选择了哪个正确性函数而改变。排名不一致直接说明了评估体系的不可靠性

### 损失函数 / 训练策略

本文不涉及模型训练，是纯评估方法论研究。

## 实验关键数据

### 主实验

不同正确性函数下 UQ 方法 AUROC 排名变化（4 数据集 × 4 模型平均）：

| 正确性函数 | 与长度的相关性 | UQ排名一致性 (Kendall τ) | 偏差等级 |
|-----------|-------------|----------------------|---------|
| ROUGE-L | 0.42 | 0.61 | 高偏差 |
| BLEU | 0.38 | 0.64 | 高偏差 |
| F1 | 0.35 | 0.67 | 中偏差 |
| BERTScore | 0.28 | 0.72 | 中偏差 |
| SentenceSim | 0.22 | 0.76 | 低偏差 |
| GPT-4 Judge | 0.08 | 0.89 | 极低偏差 |
| Claude Judge | 0.06 | 0.91 | 极低偏差 |

### 消融实验

控制长度偏差前后 UQ 方法排名的变化：

| UQ 方法 | ROUGE-L下排名 | GPT-4 Judge下排名 | 排名变化 |
|--------|-------------|-----------------|---------|
| 负序列概率 | 1 | 5 | -4 (被严重高估) |
| Token级熵 | 3 | 2 | +1 |
| 语义不确定性 | 4 | 1 | +3 (被低估) |
| p(True) | 2 | 3 | -1 |
| 一致性采样 | 6 | 4 | +2 |
| 词汇相似度 | 5 | 6 | -1 |

### 关键发现

- **ROUGE-L 等词汇级正确性函数的长度偏差最严重**（与长度相关性 >0.35），在其下做 UQ 评估会系统性地高估基于序列概率的 UQ 方法，因为两者都偏好更短的回答
- **LLM-as-a-Judge 是最公平的选择**——与长度的相关性 <0.1，不同 LLM-Judge 间的 UQ 排名高度一致（τ >0.89）
- 负序列概率在 ROUGE-L 下排名第一，但在 GPT-4 Judge 下排名第五——rank shift 的幅度令人震惊，说明之前文献中基于 ROUGE-L 的 UQ 评估结论需要重新审视
- 语义不确定性在 LLM-Judge 下排名最高，可能是"真正"最好的 UQ 方法，但在之前的评估中被系统性低估

## 亮点与洞察

- **理论+实证的双重论证非常扎实**：不是仅凭实验观察说"偏差存在"，而是先证明互偏差必然扭曲 AUROC，再用大规模实验确认。这种严谨性使结论更具说服力
- **发现的影响范围远超 UQ 领域**：任何使用 ROUGE-L 等有偏指标做评估的场景都可能存在类似的互偏差问题，包括摘要生成、对话评估等任务的 benchmark。这个洞察有很强的泛化性
- **LLM-Judge 作为解决方案的建议具有可操作性**：直接给出了"应该用什么替代"的明确答案，比仅指出问题而不给方案更有价值

## 局限与展望

- LLM-as-a-Judge 本身也可能有其他偏差（如风格偏好、位次偏差），只是长度偏差较小；用它作为"金标准"可能引入新的问题
- 实验中长度是主要分析的混淆因子，但可能还存在其他混淆因子（如词汇复杂度、格式等）未被考虑
- 理论证明基于一定的分布假设，在极端分布条件下结论可能需要修正
- 未提出如何"去偏"已有正确性函数的具体方法，仅建议更换为 LLM-Judge

## 相关工作与启发

- **vs Kadavath et al. (2022) — P(True)**: P(True) 是一种让模型自我评估不确定性的方法，本文发现其在不同正确性函数下排名较稳定，说明它对互偏差较鲁棒
- **vs Kuhn et al. (2023) — 语义不确定性**: 语义不确定性在 LLM-Judge 下表现最佳，说明之前使用 ROUGE-L 评估时可能低估了它的真实效果
- **vs AUROC 作为评估指标**: 本文揭示的问题并非 AUROC 本身的缺陷，而是在有偏条件下的应用问题。未来可考虑 calibration-aware 的替代指标

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次从互偏差角度系统分析 UQ 评估，理论贡献有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 7 正确性函数 × 8 UQ 方法 × 4 数据集 × 4 模型的完整矩阵非常全面
- 写作质量: ⭐⭐⭐⭐ 理论部分严谨，实验部分清晰，结论表述准确
- 价值: ⭐⭐⭐⭐⭐ 对整个 UQ 评估社区有深远影响，直接改变了评估最佳实践的建议

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Bias in Language Models: Beyond Trick Tests and Towards RUTEd Evaluation](bias_in_language_models_beyond_trick_tests_ruted_evaluation.md)
- [\[ACL 2025\] Revisiting Epistemic Markers in Confidence Estimation: Can Markers Accurately Reflect Large Language Models' Uncertainty?](revisiting_epistemic_markers_in_confidence_estimation_can_markers_accurately_ref.md)
- [\[ACL 2026\] Clustered Self-Assessment: A Simple yet Effective Method for Uncertainty Quantification in Large Language Models](../../ACL2026/llm_nlp/clustered_self-assessment_a_simple_yet_effective_method_for_uncertainty_quantifi.md)
- [\[ACL 2025\] Uncertainty Unveiled: Can Exposure to More In-context Examples Mitigate Uncertainty for Large Language Models?](uncertainty_unveiled_can_exposure_to_more_in-context_examples_mitigate_uncertain.md)
- [\[ACL 2025\] Towards Harmonized Uncertainty Estimation for Large Language Models](towards_harmonized_uncertainty_estimation_for_large_language_models.md)

</div>

<!-- RELATED:END -->
