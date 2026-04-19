---
title: >-
  [论文解读] The Anatomy of Evidence: An Investigation Into Explainable ICD Coding
description: >-
  [ACL 2025][ICD编码] 本文对 MDACE 数据集和当前可解释 ICD 编码系统进行了深入的应用导向分析，揭示了人工标注证据与代码描述的重叠规律、证据在文档中的分布特征，并提出了新的匹配度量来评估模型解释的实用性。
tags:
  - ACL 2025
  - ICD编码
  - 可解释性
  - 证据抽取
  - MDACE数据集
  - 特征归因
---

# The Anatomy of Evidence: An Investigation Into Explainable ICD Coding

**会议**: ACL 2025  
**arXiv**: [2507.01802](https://arxiv.org/abs/2507.01802)  
**代码**: 有 (https://github.com/lamarr-xai-group/anatomy-of-evidence)  
**领域**: NLP / 医疗信息学  
**关键词**: ICD编码, 可解释性, 证据抽取, MDACE数据集, 特征归因

## 一句话总结

本文对 MDACE 数据集和当前可解释 ICD 编码系统进行了深入的应用导向分析，揭示了人工标注证据与代码描述的重叠规律、证据在文档中的分布特征，并提出了新的匹配度量来评估模型解释的实用性。

## 研究背景与动机

自动医疗编码是缓解文档和计费流程的关键技术。ICD（国际疾病分类）编码是医院场景中最重要的编码系统，直接决定报销水平。随着深度学习的应用，自动编码系统已达到实用水平，但这些参数量达数十亿的模型**缺乏透明性**——降低了编码人员的接受度，并在监管评估中构成障碍。

可解释性研究面临的核心挑战：

**数据稀缺**：医学领域的词级标注成本极高，需要领域专家

**评估受限**：现有工作大多限于短文本和二元分类的社交媒体/产品评论

**缺乏应用视角**：先前研究以技术为中心，缺乏对数据理解和临床采用的深入分析

Cheng et al. (2023) 发布的 MDACE 数据集首次提供了临床记录中 ICD 代码的文本证据标注，为可解释 ICD 编码研究开辟了新方向。但该数据集的使用指南和现有方法的实用性评估仍然欠缺。

## 方法详解

### 整体框架

本文围绕**数据分析**和**模型解释评估**两条线展开：

**数据分析（RQ1-RQ3）：**
- RQ1：证据在文档中的**位置分布**是怎样的？
- RQ2：证据与 ICD 代码描述的**重叠程度**如何？
- RQ3：充分标注（Inpatient）是否是完全标注（Profee）的**子集**？

**模型解释评估（RQ4-RQ6）：**
- RQ4：解释长度与分类性能的**关系**如何？
- RQ5：模型解释与人工标注的**匹配程度**如何？
- RQ6：不同模型方法之间的证据**一致性**如何？

### 关键设计

1. **证据位置分析（RQ1）**

    - 分析出院摘要和医生笔记中证据的相对位置分布
    - 结果：Inpatient 证据主要出现在文档开头和结尾；Profee 在出院摘要中分布更均匀
    - 重要发现：**充分证据并非更多出现在文档开头**，推翻了"从上到下扫描"的直觉假设

2. **代码描述重叠分析（RQ2）**

    - 对证据和代码描述进行词形还原，去除停用词和标点
    - 计算每个代码的所有证据-描述对的交集/描述词集比率
    - 结果：三类分布——几乎无重叠、部分重叠、强重叠
    - 实例：R06.83 "snoring" 和 I31.3 "pericardial effusion" 有极高重叠
    - 启发：**高重叠代码可用规则系统高效处理**，释放神经模型容量给困难代码

3. **新匹配度量（RQ5 核心贡献）**

    - **Empty**：因归因分数低于阈值，未生成任何证据
    - **Exact match**：模型证据与标注证据完全一致
    - **Proximate match**：所有标注序列至少有一个 token 匹配，未匹配 token 在 k=10 的上下文窗口内
    - **Partial match**：至少一个标注序列无匹配或有 token 超出上下文窗口
    - **No match**：token ID 完全无交集
    - 动机：传统 F1/IOU 不够直观；实际场景中"引导编码人员关注正确上下文窗口"即可有效

4. **模型实验设置**

    - 50 个模型（5 种训练策略 × 10 seeds）：有监督、无监督、梯度正则化、投影梯度下降、token 掩码
    - 解释方法：AttInGrad（Attention × Input×Grad，L2 范数）
    - 选择最佳监督和最佳无监督模型进行深入分析

## 实验关键数据

### 数据集分析结果

| 分析项 | 结论 |
|--------|------|
| 证据平均长度 | Inpatient 2.18 tokens, Profee 1.96 tokens |
| 平均标签数/文档 | Inpatient 11.3, Profee 31.4 |
| Inpatient/Profee 共同子集 | 470 个唯一 note ID 中仅 118 个共同 |
| 共同代码中的子集比例 | 331 个相同代码中仅 55 个严格子集 |

### 模型解释评估（Figure 6 摘要）

| 度量 | 监督模型 | 无监督模型(IGR) |
|------|---------|----------------|
| Exact match | ~49 | ~49 |
| Proximate match | 较高 | 更高 |
| Partial match | 较高 | 较低 |
| No match | 较少 | 较多 |
| Empty | 极少 | 较少 |

### 关键发现

1. **模型解释与人工标注的重叠度很高**：最佳监督模型约 80% 的测试用例至少识别出一个正确 token
2. **"No match" 中约 46-53% 实际是语义匹配**：如 'obesity' vs 'obese'，说明纯 token ID 匹配低估了模型表现
3. **解释长度与分类性能正相关**：当模型提取的证据词数更多时，ICD 代码的召回率更高
4. **监督和无监督模型 74% 的用例具有相同匹配类型**，证据一致性较好
5. **低概率预测几乎不产生证据**（empty），实际推理中可忽略
6. **Inpatient 和 Profee 的对齐远小于预期**：不同编码规则导致代码集差异显著

### 证据多样性分析

| ICD 代码 | 出现次/唯一数 | 证据示例 |
|----------|-------------|---------|
| I10 (高血压) | 133/8 | 'hypertension', 'HTN', 'hypertensive' |
| Z87.891 (吸烟史) | 20/19 | 'smoking history', 'former smoker', 'the distant past' |

→ 部分代码证据高度固定（I10），另一些则极为多样（Z87.891），后者可能更难学习。

## 亮点与洞察

1. **应用导向的深度分析**：不同于纯技术论文，本文从临床实际需求出发评估可解释性
2. **新匹配度量实用**：Proximate match 的概念（"上下文窗口内"即可）符合编码人员的实际工作方式
3. **数据驱动的策略建议**：高重叠代码可用规则系统处理、证据多样性分析可指导训练策略
4. **跨标注方案对比**：首次深入分析 Inpatient（充分）vs Profee（完全）标注的关系

## 局限与展望

- 实验限于英语和 MIMIC-III 数据集，不同国家的编码规则和语言可能导致不同模式
- 匹配度量中的上下文窗口 k=10 为经验设置，未进行敏感性分析
- 解释方法仅评估了 AttInGrad，未比较更多类型（如 SHAP、LIME）
- 数据集规模较小（302 admissions），限制了统计分析的细粒度

## 相关工作与启发

- Edin et al. (2024) 的先驱工作提供了训练模型和 faithfulness/plausibility 评估
- Cheng et al. (2023) 创建 MDACE 数据集是本领域的基础性贡献
- PLM-ICD (Huang et al., 2022) 的 label-wise attention 架构是建模基础
- Jacovi and Goldberg (2020) 将 faithfulness 和 plausibility 作为解释性评估的两个维度

## 评分

- **新颖性**: ⭐⭐⭐ — 主要贡献在分析和洞察，方法新颖性有限
- **实验充分度**: ⭐⭐⭐⭐ — 数据分析+模型评估+定性分析多角度覆盖
- **写作质量**: ⭐⭐⭐⭐⭐ — 研究问题清晰，结构优雅，发现有实际指导价值
- **价值**: ⭐⭐⭐⭐ — 对可解释医疗编码系统的开发和部署提供了重要的实践建议

<!-- RELATED:START -->

## 相关论文

- [Retrieve to Explain: Evidence-driven Predictions for Explainable Drug Target Identification](retrieve_to_explain_drug_target_identification.md)
- [EXPERT: An Explainable Image Captioning Evaluation Metric with Structured Explanations](expert_an_explainable_image_captioning_evaluation_metric_with_structured_explana.md)
- [Towards Explainable Temporal Reasoning in Large Language Models: A Structure-Aware Generative Framework](towards_explainable_temporal_reasoning_in_large_language_models_a_structure-awar.md)
- [Normalized AOPC: Fixing Misleading Faithfulness Metrics for Feature Attribution Explainability](normalized_aopc_faithfulness_metrics.md)
- [SpEx: A Spectral Approach to Explainable Clustering](../../NeurIPS2025/interpretability/spex_a_spectral_approach_to_explainable_clustering.md)

<!-- RELATED:END -->
