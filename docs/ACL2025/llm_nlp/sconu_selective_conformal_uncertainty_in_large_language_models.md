---
title: >-
  [论文解读] SConU: Selective Conformal Uncertainty in Large Language Models
description: >-
  [LLM/NLP] SConU 首次在 LLM 的保形不确定性框架中引入显著性检验，通过构建两种保形 p-value 来识别并过滤违反可交换性假设的不确定性数据异常点，从而在单域和跨域 QA 场景中实现对错误覆盖率（miscoverage rate）的严格管理。
tags:
  - LLM/NLP
---

# SConU: Selective Conformal Uncertainty in Large Language Models

| 项目 | 内容 |
|------|------|
| 会议 | ACL 2025 |
| arXiv | [2504.14154](https://arxiv.org/abs/2504.14154) |
| 代码 | [Zhiyuan-GG/SConU](https://github.com/Zhiyuan-GG/SConU) |
| 领域 | LLM 可靠性、不确定性量化 |
| 关键词 | conformal prediction、可交换性、p-value、错误覆盖率、QA |

## 一句话总结

SConU 首次在 LLM 的保形不确定性框架中引入显著性检验，通过构建两种保形 p-value 来识别并过滤违反可交换性假设的不确定性数据异常点，从而在单域和跨域 QA 场景中实现对错误覆盖率（miscoverage rate）的严格管理。

## 研究背景与动机

- **保形预测在 LLM 中的应用**：Split Conformal Prediction (SCP) 提供分布无关、模型无关的覆盖保证，近期研究将其应用于 LLM 的 QA 任务，通过关联非一致性分数（NS）与不确定性状态来实现用户指定风险水平下的可靠响应覆盖。
- **可交换性假设的脆弱性**：现有保形不确定性框架假设校准集和测试样本的 NS 序列满足可交换性（exchangeability），但在实际 NLG 任务中这一条件难以验证和保证。作者观察到即使在单一领域内，多个 LLM 在 MMLU-Pro 数据集上也出现显著的覆盖异常（empirical miscoverage rate 超过理论上限）。
- **跨域问题尤为严重**：当校准集和测试集来自不同学科时（如用健康学科校准数学学科），错误覆盖率严重偏离目标值。这是因为不同模型在不同领域的不确定性分布差异巨大。
- **已有方法的缺陷**：现有框架手动移除校准集中采样空间不包含正确答案的样本，限制了可处理的测试样本数量且损害了统计严谨性。

## 方法详解

### 整体框架

SConU 的工作流程：(1) 部署 LLM 和校准集后，先计算最小可管理风险水平 $\alpha_l$；(2) 对每个测试样本进行显著性检验，判断其不确定性是否与校准数据分布一致；(3) 过低的保形 p-value 表示违反可交换性，拒绝回答；(4) 对通过检验的测试样本执行保形程序，提供有限样本覆盖保证。

### 关键设计

1. **基本保形 p-value（SConU）**：对测试样本 $x_{N+1}$，构建 $p_{N+1} = \frac{1 + \sum_{i=1}^{N} \mathbf{1}\{u_i \geq u_{N+1}\}}{N+1}$，其中 $u_i$ 是用预测熵（Predictive Entropy）度量的不确定性。该 p-value 衡量测试样本的不确定性在校准集中的相对位置。
2. **增强保形 p-value（SConU-Pro）**：在计数标准中加入预测状态条件 $p'_{N+1} = \frac{1 + \sum_{i=1}^{N} \mathbf{1}\{u_i \geq u_{N+1}, y_i^* \in E(x_i, \mathcal{D}_{cal}, \alpha)\}}{N+1}$，过滤掉校准集中自身在风险水平 $\alpha$ 下无法覆盖正确答案的样本的干扰。
3. **最小风险水平推导**：不移除任何校准样本，而是推导出校准集能管理的最小风险水平 $\alpha_l = N L_N(1) / (N+1)$，即候选集中不包含正确答案的比例决定了可控风险的下界。

### 损失函数

本文非训练方法，不涉及传统损失函数。核心是统计检验：在显著性水平 $\delta$ 下，若保形 p-value 低于 $\delta$ 则拒绝零假设（即认为测试样本是不确定性异常点），拒绝回答。

## 实验

### 主实验：MMLU-Pro 单域/跨域覆盖管理

| 学科 | 指标 | 无 OD (基本 ConU) | SConU | SConU-Pro |
|------|------|-----------|-------|-----------|
| Health | EMR @ α=0.1 | 0.12±0.04 (违规) | 0.09±0.02 | 0.08±0.01 |
| Economics | EMR @ α=0.1 | 0.15±0.06 (违规) | 0.09±0.03 | 0.09±0.02 |
| 跨域(Health→Math) | EMR @ α=0.28 | 0.45 (严重违规) | 0.26 | 0.24 |

实验覆盖 8 个 LLM（LLaMA-3.1-8B、Qwen2.5-14B 等），100 次随机试验的均值和标准差。

### 消融实验：采样大小校准

| 数据集 | LLM | β=0.1 | β=0.2 | β=0.3 |
|--------|-----|-------|-------|-------|
| TriviaQA | LLaMA-3.2-3B | 0.088±0.015 | 0.177±0.011 | 0.273±0.019 |
| MedMCQA | LLaMA-3.1-8B | 0.087±0.006 | 0.177±0.038 | 0.197±0.009 |
| TriviaQA | Qwen2.5-14B | 0.084±0.020 | 0.173±0.008 | 0.173±0.008 |

采样大小校准验证了保持校准集完整性的必要性：通过 Eq.(4) 可保证 $\geq 1-\beta$ 的概率在采样中覆盖正确答案。

### 关键发现

- 即使在单一领域内，基本 ConU 框架也频繁出现 EMR 超过风险水平的违规情况；SConU 通过过滤异常点有效将 EMR 控制在目标以下
- SConU-Pro 通过考虑校准数据自身的预测状态，进一步提升了异常检测的精度
- 保持校准集完整性（不手动移除无正确答案的样本）使得校准集能覆盖更广泛的领域分布
- 不确定性度量的选择（PE vs SE vs LN+SC）对条件覆盖性能有显著影响
- 预测集中存在大量语义冗余，对人机交互 QA 应用提出了去冗余需求

## 论文亮点

- 首次在 LLM 保形不确定性框架中实现显著性检验来检测可交换性违反
- 提出两种保形 p-value 的正式统计验证，具有坚实的理论基础
- 保持校准集完整性的设计理念新颖且实用，推导出最小可管理风险水平
- 在 8 个 LLM、多个 QA 数据集上的广泛实验验证了方法的通用性

## 局限性

- 保形 p-value 检验本质上是保守的，可能过度拒绝可回答的测试样本，降低回答覆盖率
- 当校准集规模较小时 p-value 的分辨力有限
- 目前主要验证在 MCQA 和开放式 QA 上，对更复杂的 NLG 任务（如摘要、翻译）的适用性未验证
- 条件覆盖在大多数 NLG 场景下仍然不可实现，只能逼近

## 相关工作

- **保形预测在 LLM 中的应用**：ConU (Wang et al., 2024c)、CONU-MCQA (Quach et al., 2024) 等提出了各种非一致性分数设计
- **分布偏移下的保形预测**：Tibshirani et al. (2019)、Barber et al. (2023) 讨论了协变量偏移下的保形推断
- **LLM 不确定性估计**：SE (Kuhn et al., 2023)、PE (Kadavath et al., 2022) 等不确定性度量方法

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 总体推荐 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [Classifying Unreliable Narrators with Large Language Models](classifying_unreliable_narrators.md)
- [Large Language Models for Predictive Analysis: How Far Are They?](large_language_models_for_predictive_analysis_how_far_are_they.md)
- [Leveraging Large Language Models to Measure Gender Representation Bias in Gendered Language Corpora](leveraging_large_language_models_to_measure_gender_representation_bias_in_gender.md)
- [RoCoFT: Efficient Finetuning of Large Language Models with Row-Column Updates](rocoft_efficient_finetuning_of_large_language_models_with_row-column_updates.md)
- [Pitfalls of Scale: Investigating the Inverse Task of Redefinition in Large Language Models](pitfalls_of_scale_investigating_the_inverse_task_of_redefinition_in_large_langua.md)

<!-- RELATED:END -->
