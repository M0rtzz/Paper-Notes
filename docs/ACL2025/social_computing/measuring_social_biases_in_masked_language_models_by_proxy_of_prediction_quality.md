---
title: >-
  [论文解读] Measuring Social Biases in Masked Language Models by Proxy of Prediction Quality
description: >-
  [ACL 2025][社会偏见] 提出了注意力加权的预测质量代理度量 Δpa 和 CRRA，在迭代掩码实验（IME）下评估 MLM 的社会偏见，并引入模型比较函数 BSRT 来估计重训练引入的偏见，发现所提方法比 CSPS、AUL、AULA 等现有方法更准确、更敏感。
tags:
  - ACL 2025
  - 社会偏见
  - 掩码语言模型
  - 注意力加权
  - 偏见评估
  - 迭代掩码实验
---

# Measuring Social Biases in Masked Language Models by Proxy of Prediction Quality

**会议**: ACL 2025  
**arXiv**: [2402.13954](https://arxiv.org/abs/2402.13954)  
**代码**: 无（提供了开源 Python 包用于偏见计算）  
**领域**: LLM/NLP  
**关键词**: 社会偏见, 掩码语言模型, 注意力加权, 偏见评估, 迭代掩码实验

## 一句话总结

提出了注意力加权的预测质量代理度量 Δpa 和 CRRA，在迭代掩码实验（IME）下评估 MLM 的社会偏见，并引入模型比较函数 BSRT 来估计重训练引入的偏见，发现所提方法比 CSPS、AUL、AULA 等现有方法更准确、更敏感。

## 研究背景与动机

BERT、RoBERTa 等掩码语言模型（MLM）在各种 NLP 任务上取得了 SOTA 表现，但也被发现编码了对弱势群体的不良社会偏见。评估这些偏见至关重要，但现有方法存在以下问题：

**伪似然方法的局限**：CrowS-Pairs Score (CSPS) 和 StereoSet Score (SSS) 等基于伪似然的方法假设被掩码的 token 是统计独立的，且存在对高频词的选择偏差。

**AUL/AULA 的理念分歧**：Kaneko & Bollegala (2022) 的 AUL 和 AULA 通过同时预测所有 token（不使用掩码）来消除掩码相关偏差，但作者认为在评估 MLM 的核心预训练目标（掩码语言模型）时，掩码相关的偏差信息恰恰是有价值的。

**重训练偏见评估的空白**：缺乏有效的方法来评估 MLM 在掩码语言建模目标下重训练后引入的偏见变化。现有方法在重训练场景下可能低估偏见。

**注意力权重的忽视**：现有方法未充分利用 MLM 的注意力机制来衡量不同 token 对预测的贡献权重。

## 方法详解

### 整体框架

基于迭代掩码实验（IME）：逐一掩码句子中的每个 token，直到所有 token 都被掩码过，获取模型对每个 token 的预测质量。通过比较模型对偏向弱势群体（Sdis）和优势群体（Sadv）的配对句子的预测偏好来评估偏见。

### 关键设计

1. **修改版概率差 Δp（Eq. 3）**：
   $\Delta P(t|s_{\setminus t_m};\theta) = \log P(t_p|s_{\setminus t_m};\theta) - \log P(t_m|s_{\setminus t_m};\theta)$
   
   使用对数变换代替原始概率差，减少高概率区域的敏感性，增强低概率区域的区分能力，这对 MLM 通常的长尾概率分布更合适。

2. **注意力加权互补倒数排名 CRRA（Eq. 4）**：
   $\text{CRRA}(t|s_{\setminus t_m};\theta) = a_m(1 - \log \rho(t_m|s_{\setminus t_m};\theta)^{-1})$
   
   其中 $a_m$ 是与ground truth token相关的多头注意力平均值，$\rho$ 是被掩码token的排名。通过注意力加权，考虑不同 token 对 MLM 预测的重要性。

3. **注意力加权概率差 Δpa（Eq. 5）**：
   $\Delta\text{pa}(t|s_{\setminus t_m};\theta) = a_m(\log P(t_p|s_{\setminus t_m};\theta) - \log P(t_m|s_{\setminus t_m};\theta))$
   
   将注意力权重与对数概率差结合。

4. **预训练偏见得分 BSPT（Eq. 9）**：
   $\text{BSPT}(f) = \frac{100}{N}\sum_{i=1}^{N}\mathbb{1}(\Delta f_T(i) > 0)$
   
   表示对弱势群体偏见更高的句子比例。高于 50 表示模型对弱势群体有更大偏见。

5. **重训练偏见得分 BSRT（Eq. 10）**：
   $\text{BSRT}(f) = \frac{100}{N}\sum_{i=1}^{N}\mathbb{1}(\Delta f_{T_1}(i) > \Delta f_{T_2}(i))$
   
   比较重训练模型 T1 与预训练模型 T2 的相对偏见变化。这是本文的核心创新，用于评估重训练引入的偏见。

### 损失函数 / 训练策略

重训练实验设置：
- 在 CPS 数据集的 Sdis（偏向弱势群体的句子）或 Sadv（偏向优势群体的句子）上重训练 MLM
- 使用 PyTorch，P100/T4 GPU，训练 30 个 epoch
- 4 个 MLM：BERT-base-uncased、RoBERTa-base、distilBERT-base-uncased、distilRoBERTa-base

## 实验关键数据

### 主实验

**预训练模型偏见得分 BSPT（CPS数据集）**：

| 度量 | RoBERTa | BERT_unc | D-RoBERTa | D-BERT_unc |
|------|---------|----------|-----------|------------|
| CSPS | 59.35 | 60.48 | 59.35 | 56.83 |
| AUL | 58.75 | 48.34 | 53.32 | 51.59 |
| AULA | 58.09 | 48.21 | 51.86 | 52.65 |
| CRR | 58.89 | 61.07 | 57.76 | 56.23 |
| **CRRA** | **60.68** | 58.89 | **61.94** | **60.08** |
| Δp | 59.88 | 60.08 | 59.75 | 57.49 |
| **Δpa** | 60.15 | **60.81** | 59.81 | 58.02 |

所有模型在所有度量下均显示偏见得分 >50，表明所有 MLM 都编码了对弱势群体的社会偏见。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 重训练偏见对齐错误率（Sdis 重训练） | Δp, Δpa: 0% 错误 | 100%正确检测偏见方向 |
| 重训练偏见对齐错误率（Sadv 重训练） | Δp, Δpa: 0% 错误 | CSPS: 5.6%，AUL: 8.3% 错误 |
| CRR 错误率 | 0% | 与Δp、Δpa同样完美 |
| CRRA 错误率 | 2.8%（仅1例） | 接近完美 |
| CSPS 错误率 | 5.6% | 显著高于所提方法 |
| AULA 错误率 | 12.5% | 表现最差 |

**McNemar 检验结果**：
- Δp、Δpa：所有 MLM 和所有偏见类别均显著（p<0.05）
- CSPS：11 个不显著结果（Sdis 重训练）
- AULA：12 个不显著结果（Sadv 重训练）

### 关键发现

1. **所有 MLM 都编码了社会偏见**：所有四个 transformer 模型在 CPS 和 StereoSet 上均表现出对弱势群体的偏见
2. **AUL/AULA 对 BERT 的低估**：AUL 和 AULA 在 CPS 上对 BERT 的偏见估计低于 50（48.34、48.21），与其他度量矛盾
3. **重训练偏见检测**：所提度量对重训练引入的偏见最为敏感，CSPS、AUL、AULA 在重训练偏见评估中产生令人担忧的低估
4. **宗教偏见普遍较高**：在 CPS 所有 MLM 中观察到较高的宗教偏见分数
5. **性别偏见在 CPS 上较低**：CPS 中性别偏见低于 StereoSet，可能与数据集构造差异有关
6. **人类标注对齐**：CRRA 和 Δpa 在与人类标注偏见判断对齐方面优于 AUL 和 AULA

## 亮点与洞察

- **聚焦核心目标**：回归 MLM 的核心预训练目标（掩码语言建模）来评估偏见，而非使用间接代理，逻辑清晰
- **注意力权重的合理利用**：通过平均多头注意力来衡量 token 重要性，提供了比均匀加权更合理的句子级偏见估计
- **BSRT 的实用价值**：模型比较函数可以直接评估重训练对偏见的影响，对偏见缓解研究非常实用
- **严谨的统计验证**：使用 Shapiro-Wilk 正态性检验、McNemar 检验等统计方法确认结果的可靠性

## 局限与展望

1. **仅限英文**：CPS 和 StereoSet 都是英文数据集，其他语言的偏见分析需要新的基准
2. **仅限 MLM 架构**：只评估了掩码语言模型（BERT/RoBERTa），不适用于自回归模型（GPT 系列）
3. **二值偏见分类**：将偏见简化为弱势/优势群体二分法，无法捕捉更微妙的偏见层次
4. **模型规模有限**：仅测试了 base 级别模型，未涉及 large 或更大规模模型
5. **重训练设置单一**：固定 30 个 epoch，未探索不同训练量对偏见变化的影响

## 相关工作与启发

- **CrowS-Pairs (Nangia et al., 2020)**：包含 9 类社会偏见的配对句子基准，本文的核心评估数据集
- **StereoSet (Nadeem et al., 2021)**：4 类偏见的句内/句间配对数据集
- **AUL/AULA (Kaneko & Bollegala, 2022)**：不使用掩码的偏见评估方法，本文的主要对比方法
- **Salutari et al. (2023)**：提出 CRR 和 Δp 的原始版本，本文在此基础上引入注意力加权
- 启发：评估方法的选择对偏见估计结果影响巨大，特别是在重训练场景中，不同度量可能给出完全相反的结论

## 评分

- 新颖性: ⭐⭐⭐ 注意力加权是合理但渐进式的改进，BSRT函数是主要创新点
- 实验充分度: ⭐⭐⭐⭐ 两个基准、四个模型、七个度量的系统比较，统计检验严谨
- 写作质量: ⭐⭐⭐ 公式密集，符号系统复杂，阅读门槛较高
- 价值: ⭐⭐⭐⭐ 对MLM偏见评估方法论有重要贡献，BSRT对偏见缓解研究很实用

<!-- RELATED:START -->

## 相关论文

- [BanStereoSet: A Dataset to Measure Stereotypical Social Biases in LLMs for Bangla](banstereoset_a_dataset_to_measure_stereotypical_social_biases_in_llms_for_bangla.md)
- [Explicit vs. Implicit: Investigating Social Bias in Large Language Models through Self-Reflection](explicit_vs_implicit_investigating_social_bias_in_large_language_models_through_.md)
- [A Survey on Proactive Defense Strategies Against Misinformation in Large Language Models](a_survey_on_proactive_defense_strategies_against_misinformation_in_large_languag.md)
- [BiasGuard: A Reasoning-Enhanced Bias Detection Tool for Large Language Models](biasguard_a_reasoning-enhanced_bias_detection_tool_for_large_language_models.md)
- [OS-Harm: A Benchmark for Measuring Safety of Computer Use Agents](../../NeurIPS2025/social_computing/os-harm_a_benchmark_for_measuring_safety_of_computer_use_agents.md)

<!-- RELATED:END -->
