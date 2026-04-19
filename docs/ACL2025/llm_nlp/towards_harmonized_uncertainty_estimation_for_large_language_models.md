---
title: >-
  [论文解读] Towards Harmonized Uncertainty Estimation for Large Language Models
description: >-
  [ACL2025][LLM/NLP][不确定性估计] 提出 CUE 框架，通过训练一个与目标 LLM 性能对齐的轻量级分类器（Corrector）来校正现有不确定性估计方法的分数，在指示性、精确-召回平衡和校准三个维度上实现协调一致的改进，最高提升达 60%。
tags:
  - ACL2025
  - LLM/NLP
  - 不确定性估计
  - LLM可信部署
  - 校准
  - 轻量级校正器
---

# Towards Harmonized Uncertainty Estimation for Large Language Models

**会议**: ACL2025  
**arXiv**: [2505.19073](https://arxiv.org/abs/2505.19073)  
**代码**: [O-L1RU1/Corrector4UE](https://github.com/O-L1RU1/Corrector4UE)  
**领域**: llm_nlp  
**关键词**: 不确定性估计, LLM可信部署, 校准, 轻量级校正器

## 一句话总结

提出 CUE 框架，通过训练一个与目标 LLM 性能对齐的轻量级分类器（Corrector）来校正现有不确定性估计方法的分数，在指示性、精确-召回平衡和校准三个维度上实现协调一致的改进，最高提升达 60%。

## 背景与动机

1. **LLM 幻觉风险**：大语言模型在回答问题时常常"自信地犯错"，生成幻觉和事实性错误，用户难以判断输出是否可靠，因此需要不确定性估计来量化输出可信度。
2. **现有方法三项指标难以兼顾**：作者从分类视角（AUROC 衡量指示性、F1 衡量精确-召回平衡）和校准视角（ECE）系统评估了现有方法，发现它们在三个维度上表现极不均衡——擅长一项则另两项拉胯。
3. **基础方法指示性差**：Lexical Similarity、Verbal Confidence、P(True)、Predictive Entropy 等基础方法的 AUROC 接近随机猜测（0.5），难以有效区分可靠与不可靠回答。
4. **增强 logit 方法 F1 极低**：SAR、SE 等改进方法虽然提升了 AUROC，但 F1 分数极低，无法在精确率和召回率之间取得平衡，导致实际使用中误报或漏报严重。
5. **校准被普遍忽视**：绝大多数现有方法的 ECE 表现较差，不确定性分数与真实概率之间存在严重偏差，不符合人类对概率的直觉理解。
6. **方法间互补性差**：作者尝试将不同方法的不确定性分数加权组合，发现几乎没有提升甚至会下降，说明现有方法具有同质性，缺乏互补信息来源。

## 方法详解

CUE（Corrector for Uncertainty Estimation）框架包含三个步骤：

### 1. 数据构造（Dataset Crafting）
- 从已有数据集中抽取 QA 对 $\mathcal{D}=\{(q_i,a_i)\}$，让目标模型 $M$ 生成回答 $r_i$
- 使用混合评判策略判断回答正确性：**规则方法**（ROUGE-L > 0.7 则判正确）和 **LLM 方法**（GPT-3.5 直接判断语义等价），二者取 OR 逻辑
- 为每个样本分配二值标签 $c_i$，并将其反转为不确定性方向：$\mathcal{D}^*_{\text{cor}}=\{(q_i, 1-c_i)\}$

### 2. 校正器训练（Corrector Training）
- 使用轻量级编码器（如 RoBERTa 或 DeBERTa）+ 全连接层构建二分类器
- 输入为问题文本的 [CLS] 表示，输出经 sigmoid 映射为 $[0,1]$ 概率值
- 最小化二元交叉熵损失进行训练，使模型学会预测"目标 LLM 在该问题上是否会出错"

### 3. 不确定性校正（Uncertainty Correcting）
- 将原始不确定性分数 Min-Max 归一化到 $[0,1]$：$U_{\text{norm}}(x)=\frac{U(x)-\min(U)}{\max(U)-\min(U)}$
- 与校正器输出的校正分数 $C(x)$ 加权融合：$U_{\text{cor}}(x)=w^*\cdot U_{\text{norm}}(x)+(1-w^*)\cdot C(x)$
- 最优权重 $w^*$ 通过网格搜索在开发集上确定

**核心洞察**：校正器提供了一种与现有 UE 方法正交的全局对齐信息——它从问题文本直接预测目标模型的出错概率，而非依赖 LLM 的内部逻辑或语言特征。

## 实验关键数据

**实验设置**：目标模型为 OPT-6.7B 和 LLaMA-3-8B-Instruct；数据集为 TriviaQA（95K QA 对）和 SciQA（2565 QA 对）；基线覆盖 logit、verbalized、consistency、internal state 四大类共 9 种方法。

### 表1：AUROC 与 ECE 改进（LLaMA-3-8B-Instruct，部分）

| 方法 | TriviaQA AUROC（原→+Corrector） | TriviaQA ECE（原→+Corrector） | SciQA AUROC（原→+Corrector） |
|------|------|------|------|
| LS | 19.57→69.82 (+50.25) | 70.25→7.41 (-62.84) | 53.67→65.38 (+11.71) |
| VC | 62.34→74.89 (+12.55) | 23.41→16.78 (-6.63) | 68.22→72.15 (+3.93) |
| SE | 80.92→82.12 (+1.20) | 13.07→12.76 (-0.31) | 71.59→72.93 (+1.34) |
| SAR | 80.92→81.90 (+0.98) | 16.17→13.76 (-2.41) | 73.88→75.19 (+1.31) |

### 表2：消融实验（LLaMA-3-8B-Instruct + TriviaQA）

| 方法 | AUROC (↑) | ECE (↓) |
|------|-----------|---------|
| 纯 Corrector | 69.87 | 6.73 |
| 原始最优方法 | 80.92 | 11.53 |
| +Corrector（概率值） | **82.12** | **10.46** |
| +Corrector（标签值） | 80.92 | 11.53 |
| +GPT-4o 评分 | 80.92 | 11.53 |

**关键发现**：
- AUROC 平均提升 0.27（TriviaQA）和 0.09（SciQA）
- F1 分数平均提升 38.97%
- 对弱基线（如 LS）提升最大（+50.25 AUROC），对强基线（如 SE）也有稳定提升
- 概率值校正远优于离散标签值和 GPT-4o 评分

## 亮点

- **思路简洁且通用**：Corrector 与现有所有 UE 方法正交，即插即用，不需要访问目标模型内部状态
- **三维度协调提升**：同时改善指示性、精确-召回平衡和校准，而非仅优化单一指标
- **轻量高效**：使用 RoBERTa/DeBERTa 等小模型作为 Corrector，训练和推理成本极低
- **实证分析扎实**：先通过系统分析暴露现有方法的不足，再有针对性地提出解决方案

## 局限与展望

- **依赖标注数据**：需要目标模型在特定领域的 QA 对及正确性标签来训练 Corrector，冷启动成本不低
- **跨领域泛化存疑**：在一个领域训练的 Corrector 迁移到其他知识领域时效果可能下降
- **仅验证白盒模型**：实验仅在开源模型上进行，未在闭源 API 模型（如 GPT-4）上验证
- **归一化依赖全局统计**：Min-Max 归一化需要整个测试集的最大最小值，在线场景下不太方便

## 与相关工作的对比

### vs Semantic Entropy (SE)
SE 通过语义聚类计算熵来估计不确定性，在 AUROC 上表现较好但 F1 和 ECE 较差。CUE 作为后处理校正框架，可以在 SE 基础上进一步提升所有三项指标，且两者互补——SE 利用模型内部语义信息，CUE 利用外部对齐信息。

### vs SAR (Shifting Attention to Relevance)
SAR 通过关注相关 token 的贡献来改进预测熵，是现有 SOTA 方法之一。但 SAR 同样存在 F1 极低和校准差的问题。CUE 在 SAR 基础上仍能带来 0.98-3.35 的 AUROC 提升和显著的 ECE 降低，证明两种方法提供的信息确实正交。

### vs LARS (Learnable Response Scoring)
LARS 同样使用监督信号来学习 token 概率依赖关系，但它改进的是评分函数本身。CUE 则完全从问题端出发预测目标模型的可靠性，不依赖于 LLM 的输出 logits，适用范围更广。

## 评分
- 新颖性: ⭐⭐⭐ — 核心思路（用外部分类器校正不确定性分数）简单直接，创新性中等但实用性强
- 实验充分度: ⭐⭐⭐⭐ — 2 个模型 × 2 个数据集 × 9 个基线，消融实验覆盖格式/模型/获取方式
- 写作质量: ⭐⭐⭐⭐ — 问题分析清晰，三维度框架有说服力，图表丰富
- 价值: ⭐⭐⭐⭐ — 作为通用后处理模块，对 LLM 可信部署有直接实用价值

<!-- RELATED:START -->

## 相关论文

- [Revisiting Epistemic Markers in Confidence Estimation: Can Markers Accurately Reflect Large Language Models' Uncertainty?](revisiting_epistemic_markers_in_confidence_estimation_can_markers_accurately_ref.md)
- [Uncertainty Unveiled: Can Exposure to More In-context Examples Mitigate Uncertainty for Large Language Models?](uncertainty_unveiled_can_exposure_to_more_in-context_examples_mitigate_uncertain.md)
- [SConU: Selective Conformal Uncertainty in Large Language Models](sconu_selective_conformal_uncertainty_in_large_language_models.md)
- [Knowledge Boundary of Large Language Models: A Survey](knowledge_boundary_survey.md)
- [Neural Topic Modeling with Large Language Models in the Loop](neural_topic_modeling_with_large_language_models_in_the_loop.md)

<!-- RELATED:END -->
