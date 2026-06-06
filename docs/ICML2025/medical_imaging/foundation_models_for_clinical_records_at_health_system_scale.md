---
title: >-
  [论文解读] Foundation Models for Clinical Records at Health System Scale
description: >-
  [ICML2025][医学图像][电子健康记录] 提出 GPT-EHR，一种基于下一次就诊事件预测的生成式预训练框架，在 NYU Langone 129 万患者的纵向 EHR 数据上训练 decoder-only Transformer，零样本即可预测痴呆和膝骨关节炎发病，性能媲美全量微调的 BERT 基线…
tags:
  - "ICML2025"
  - "医学图像"
  - "电子健康记录"
  - "基础模型"
  - "生成式预训练"
  - "零样本预测"
  - "疾病预测"
  - "Transformer"
---

# Foundation Models for Clinical Records at Health System Scale

**会议**: ICML2025  
**arXiv**: [2507.00574](https://arxiv.org/abs/2507.00574)  
**作者**: Haresh Rengaraj Rajamohan, Xiang Gao, Weicheng Zhu, Shih-Lun Huang, Long Chen, Kyunghyun Cho, Cem M. Deniz, Narges Razavian (NYU Langone Health / NYU)  
**代码**: 未公开  
**领域**: 医学图像  
**关键词**: 电子健康记录, 基础模型, 生成式预训练, 零样本预测, 疾病预测, Transformer  

## 一句话总结

提出 GPT-EHR，一种基于下一次就诊事件预测的生成式预训练框架，在 NYU Langone 129 万患者的纵向 EHR 数据上训练 decoder-only Transformer，零样本即可预测痴呆和膝骨关节炎发病，性能媲美全量微调的 BERT 基线，同时揭示并解决了重复事件 token 造成评估指标虚高的关键陷阱。

## 研究背景与动机

### 问题背景
慢性疾病（痴呆、骨关节炎、癌症等）的早期检测和病程预测对改善医疗结局至关重要。电子健康记录（EHR）蕴含海量纵向数据，但其序列化、高维、不规则、异构的特性对建模提出严峻挑战。传统方法为每个特定疾病和时间窗口单独训练模型，资源消耗大、灵活性差，且无法利用不同临床事件之间的依赖关系。

### 已有工作的不足
- **掩码预训练方法**（Li et al. 2020; Yang et al. 2023）需要额外微调阶段和任务专属数据集，成本高
- **基于 next-token 的生成式方法**（McDermott et al. 2023; Renc et al. 2024）多在 ICU 记录等特定场景训练，未能应对纵向门诊记录中同一次就诊内事件无序、慢性病反复出现等独特挑战
- **重复事件陷阱**：慢性病诊断码在多次就诊中反复出现，模型只需记住已有诊断即可获得高预测分数，掩盖了对新发事件的真实检测能力（Kraljevic et al. 2022 首次观察到该现象，但缺乏系统分析与解决方案）

### 核心动机
构建一个能够从 EHR 数据中学习综合表示、生成患者轨迹的生成式基础模型：通过下一次就诊事件预测（next-visit event prediction）替代 next-token 预测，自然处理就诊内事件的无序性；同时通过正则化机制抑制重复事件的影响，提升对新发疾病的检测能力。

## 方法详解

### 整体框架：GPT-EHR 系统

GPT-EHR 采用 decoder-only Transformer 架构（基于 GPT-2），在大规模纵向 EHR 数据上进行生成式预训练，核心范式为**下一次就诊多标签预测**（next-visit multi-label prediction）。整体流程分为两阶段：

1. **预训练阶段**：给定患者历史就诊序列 $H_{t_i}$，模型预测下一次就诊 $i+1$ 中将出现的所有临床事件 token
2. **零样本推理阶段**：无需微调，直接利用预训练模型预测特定疾病（如痴呆、膝骨关节炎）在未来 2 年或 5 年内的发病概率

### 关键设计 1：EHR Token 化与输入表示

| 设计要素 | 具体方案 |
|---------|---------|
| 数据来源 | NYU Langone Health 2013-2023 年纵向 EHR，129 万患者 |
| Token 类型 | 人口学特征、就诊年龄、药物、诊断、实验室检查结果 |
| 连续值处理 | 基于分位数离散化为区间（quantile-based binning） |
| 词表大小 | 42,337 个唯一 token |
| 平均就诊 token 数 | 11.16 tokens/visit |
| 平均患者序列长度 | 474.21 tokens/patient |
| 就诊分隔 | 使用 sep 特殊 token 标记就诊边界 |
| 位置编码 | RoPE（旋转位置编码），同一次就诊内 token 共享相同位置嵌入 |
| 时间间隔编码 | sep token 位置编码设为下一次就诊的时间戳，显式编码就诊间时间间隔 |

这种设计有两个关键优势：（1）同一就诊内共享位置编码，天然处理了就诊内事件无序的问题；（2）利用 sep token 的位置差异编码时间间隔，让模型感知就诊间的时间距离。

### 关键设计 2：修改的因果注意力机制

标准因果注意力仅允许 token 关注之前的 token，但 EHR 数据中同一次就诊内的事件是同时发生的、无序的。GPT-EHR 修改注意力掩码：

- 就诊 $v$ 中的 token 可关注**所有 $v' \leq v$ 就诊中的所有 token**
- 同一就诊内的 token **可相互关注**（双向），保留就诊间的因果性

此外，为提升训练效率，采用**序列打包**（sequence packing）策略：将多个患者的序列拼接到同一训练序列中（若可容纳），允许注意力跨越拼接的患者序列，既提升 GPU 吞吐量，又可能提供更广泛的上下文学习能力。

### 关键设计 3：下一次就诊多标签预测目标

不同于 NLP 中逐 token 的自回归生成，GPT-EHR 将预测目标设定为**下一次就诊中所有事件的联合预测**：

- **输入**：患者历史 $H_{t_i}$ + 下次就诊时间 $t_{i+1}$（通过 sep token 位置编码）
- **输出**：基于 sep token 的输出表示，经线性层 + Sigmoid 激活，预测词表中每个 token 在下次就诊中出现的概率 $\hat{P}_i$
- **损失**：多标签二元交叉熵损失

### 关键设计 4：重复事件正则化

这是本文的一个核心贡献——揭示并解决 EHR 基础模型评估中的**重复事件陷阱**：

**问题**：慢性病（如糖尿病、高血压）的诊断码在患者多次就诊中反复出现。模型只需学会复制已有诊断即可在标准评估指标上表现优异，但实际上未能学习**新发疾病**的预测能力。

**解决方案**：对重复出现的事件 token 引入正则化，降低其在损失函数中的权重，鼓励模型将更多学习容量分配给**首次出现的新事件**。这使得模型在预测新发疾病方面的能力显著提升。

### 训练策略

| 训练参数 | 设定 |
|---------|------|
| 数据规模 | 1,288,242 名患者 |
| 数据划分 | 训练 70% / 验证 15% / 测试 15%（患者级别） |
| 时间跨度 | 2013 年 1 月 - 2023 年 1 月（10 年） |
| 患者就诊统计 | 中位数 21 次（均值 37.76，范围 2-2123） |
| 架构 | Decoder-only Transformer（GPT-2 风格） |
| 位置编码 | RoPE（旋转位置编码） |
| 训练技巧 | 序列打包（sequence packing）提升 GPU 利用率 |
| 推理方式 | 零样本推理，无需微调 |

## 实验关键数据

### 零样本疾病预测性能

模型以零样本方式预测痴呆和膝骨关节炎（KOA）在未来 2 年和 5 年内的新发概率，对比全量微调的 BERT 基线：

| 任务 | 预测窗口 | GPT-EHR（零样本） | BERT 基线（微调） | 对比 |
|------|---------|------------------|------------------|------|
| 痴呆发病预测 | 2 年 | 可比较性能 | 全量微调 | 零样本约等于微调 |
| 痴呆发病预测 | 5 年 | 可比较性能 | 全量微调 | 零样本约等于微调 |
| 膝骨关节炎发病预测 | 2 年 | 可比较性能 | 全量微调 | 零样本约等于微调 |
| 膝骨关节炎发病预测 | 5 年 | 可比较性能 | 全量微调 | 零样本约等于微调 |

关键发现：**零样本 GPT-EHR 可以媲美全量微调的 BERT 基线**，这表明生成式预训练范式在 EHR 上可以避免昂贵的任务专属微调。

### 重复事件对评估指标的影响

| 评估方式 | 包含重复事件 | 仅新发事件 | 差异 |
|---------|------------|----------|------|
| 标准评估 | 指标虚高 | 真实能力 | 显著差距 |
| 加入正则化后 | 改善 | 显著提升 | 差距缩小 |

重复事件正则化的效果：
- 不区分重复事件时，模型评估指标被严重高估
- 正则化后，新发事件预测性能显著改善
- 该发现对所有基于 EHR 的基础模型评估具有普适意义

## 亮点与洞察

- **范式创新**：以下一次就诊事件预测替代 next-token 预测，优雅解决就诊内事件无序问题，同时保持就诊间的因果时序
- **评估陷阱揭示**：系统揭示了 EHR 基础模型中重复事件导致评估虚高的问题，这对整个 EHR-ML 社区具有警示意义
- **零样本能力**：无需微调即可达到微调 BERT 的水平，极大降低了临床预测模型的部署成本
- **数据规模**：在 129 万患者的真实医疗系统数据上训练和验证，远超大多数 EHR 研究使用的 MIMIC 等公开数据集
- **时间编码设计**：利用 RoPE 位置编码巧妙嵌入就诊间时间间隔信息，使 sep token 成为时间信息的载体

## 局限与展望

- **单一医疗系统数据**：仅在 NYU Langone Health 数据上训练和验证，泛化性到其他医疗系统有待验证
- **评估任务有限**：仅在痴呆和膝骨关节炎两种疾病上评估，未涵盖更多临床预测任务
- **缺乏与最新生成式 EHR 模型对比**：如 MOTOR（McDermott et al. 2023）、CEHR-GPT（Renc et al. 2024）等
- **序列打包可能引入噪声**：允许注意力跨患者序列，虽提升效率，但可能导致模型学到虚假特征
- **仅限结构化数据**：未整合临床文本（Clinical Notes）、影像等非结构化 EHR 数据
- **模型规模未充分探索**：基于 GPT-2 架构，未探索更大规模模型的 scaling 效果
- **缺乏临床部署验证**：作为回顾性研究，未评估模型在前瞻性临床决策中的实际表现
- **重复事件正则化参数敏感性**：未详细分析正则化强度对不同类型事件的影响

## 相关工作与启发

- **EHR 掩码预训练**：Med-BERT（Li et al. 2020）、BEHRT（Yang et al. 2023）等采用 BERT 式掩码语言模型预训练 EHR 表示，但需要额外微调适配下游任务
- **EHR 生成式预训练**：MOTOR（McDermott et al. 2023）、CEHR-GPT（Renc et al. 2024）探索了基于 next-token 的生成式 EHR 建模，但主要在 ICU 数据上验证
- **Next-visit 预测范式**：Steinberg et al. (2021) 提出 next-visit 多标签预测，但仅用于表示学习，未做零样本推理
- **重复事件问题**：Kraljevic et al. (2022) 首次发现模型在预测新概念时性能下降，本文在此基础上提供了更系统的分析和解决方案
- **临床预测模型**：传统方法针对特定疾病和时间窗口单独建模（Dubois et al. 2015; Zhu et al. 2024），本文的基础模型方法提供了统一的替代方案
- **对 EHR-ML 评估的启发**：重复事件陷阱提醒社区在评估 EHR 预测模型时必须区分新发事件和重复事件，否则评估结论不可靠

## 评分

- 新颖性: 4/5 — 下一次就诊预测范式和重复事件正则化有创新性，但 decoder-only Transformer 用于 EHR 并非全新
- 实验充分度: 3/5 — 在大规模真实数据上验证，但仅两种疾病、缺乏与同类生成式方法的直接对比
- 写作质量: 4/5 — 问题动机清晰，方法描述完整，重复事件陷阱的阐述具有警示价值
- 价值: 4/5 — 零样本预测媲美微调的发现实用价值高，对 EHR-ML 评估陷阱的揭示对社区有广泛意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] CliCARE: Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records](../../AAAI2026/medical_imaging/clicare_grounding_large_language_models_in_clinical_guidelines_for_decision_supp.md)
- [\[AAAI 2026\] Personalization of Large Foundation Models for Health Interventions](../../AAAI2026/medical_imaging/personalization_of_large_foundation_models_for_health_interventions.md)
- [\[ICML 2025\] Raptor: Scalable Train-Free Embeddings for 3D Medical Volumes Leveraging Pretrained 2D Foundation Models](raptor_scalable_train-free_embeddings_for_3d_medical_volumes_leveraging_pretrain.md)
- [\[AAAI 2026\] G2L: From Giga-Scale to Cancer-Specific Large-Scale Pathology Foundation Models via Efficient Fine-Tuning](../../AAAI2026/medical_imaging/g2lfrom_giga-scale_to_cancer-specific_large-scale_pathology_foundation_models_vi.md)
- [\[NeurIPS 2025\] QoQ-Med: Building Multimodal Clinical Foundation Models with Domain-Aware GRPO Training](../../NeurIPS2025/medical_imaging/qoq-med_building_multimodal_clinical_foundation_models_with_domain-aware_grpo_tr.md)

</div>

<!-- RELATED:END -->
