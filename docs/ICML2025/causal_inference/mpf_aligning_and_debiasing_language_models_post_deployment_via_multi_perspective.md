---
title: >-
  [论文解读] MPF: Aligning and Debiasing Language Models post Deployment via Multi Perspective Fusion
description: >-
  [ICML 2025 (AIW Workshop)][偏见缓解] 提出 Multiperspective Fusion (MPF)，一种无需微调的后部署对齐框架，通过将基线情感分布分解为可解释的视角成分，引导 LLM 生成与人类基线对齐的响应，有效缓解模型偏见。
tags:
  - ICML 2025 (AIW Workshop)
  - 偏见缓解
  - 后训练对齐
  - 多视角融合
  - 情感分布
  - KL散度
  - 校准误差
---

# MPF: Aligning and Debiasing Language Models post Deployment via Multi Perspective Fusion

**会议**: ICML 2025 (AIW Workshop)

**arXiv**: [2507.02595](https://arxiv.org/abs/2507.02595)

**作者**: Xin Guan, PeiHsin Lin, Zekun Wu, Ze Wang, Ruibo Zhang, Emre Kazim, Adriano Koshiyama

**领域**: 因果推理 / LLM对齐与去偏

**关键词**: 偏见缓解, 后训练对齐, 多视角融合, 情感分布, KL散度, 校准误差

## 一句话总结

提出 Multiperspective Fusion (MPF)，一种无需微调的后部署对齐框架，通过将基线情感分布分解为可解释的视角成分，引导 LLM 生成与人类基线对齐的响应，有效缓解模型偏见。

## 研究背景与动机

大语言模型（LLMs）在部署后常表现出系统性偏见，例如在招聘场景中对特定大学背景的候选人表现出不同的情感倾向。现有的去偏方法通常需要：

**微调**：需要大量计算资源和训练数据，且可能引入新的偏见

**复杂的提示工程**：需要人工设计提示，难以泛化

**重新训练**：对已部署的模型不可行

MPF 的核心动机是：**能否在不修改模型参数的前提下，通过后处理方式实现与人类期望基线的对齐？** 该框架基于 SAGED 管道构建，SAGED 是一个自动化系统，用于构建偏见基准测试并提取可解释的基线分布。

## 方法详解

### 整体框架

MPF 框架包含三个核心模块：

1. **基线分布提取**：通过 SAGED 管道从人类专家（如 HR 专业人员）中获取基线情感分布
2. **视角分解**：将基线分布分解为多个可解释的视角成分
3. **多视角融合生成**：通过加权采样和平衡来引导模型生成

### 关键设计：视角分解

给定基线分布 $P_{\text{baseline}}(s)$（其中 $s$ 代表情感），MPF 将其分解为 $K$ 个视角成分：

$$P_{\text{baseline}}(s) = \sum_{k=1}^{K} \pi_k \cdot P_k(s)$$

其中 $\pi_k$ 是第 $k$ 个视角的权重，满足 $\sum_{k=1}^{K} \pi_k = 1$，$P_k(s)$ 是第 $k$ 个视角下的情感分布。

### 多视角融合生成

在生成阶段，MPF 通过以下步骤引导输出：

1. 从每个视角 $k$ 独立采样生成响应 $r_k$
2. 根据分解得到的权重 $\pi_k$ 进行加权采样
3. 最终输出分布为：

$$P_{\text{MPF}}(r) = \sum_{k=1}^{K} \pi_k \cdot P_{\text{LLM}}(r | \text{perspective}_k)$$

### 损失函数与优化目标

MPF 的优化目标是最小化输出分布与目标基线之间的 KL 散度：

$$\mathcal{L}_{\text{align}} = D_{\text{KL}}(P_{\text{MPF}} \| P_{\text{baseline}})$$

同时引入校准误差约束：

$$\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} |{\text{acc}(b) - \text{conf}(b)}|$$

## 实验关键数据

### 主实验：与不同基线的对齐效果

实验在招聘场景下评估，比较 LLM 对不同大学背景候选人的情感分布。

| 方法 | KL 散度 (↓) | 校准误差 ECE (↓) | 泛化到未见问题 |
|:---|:---:|:---:|:---:|
| 原始 LLM（无干预） | 0.85+ | 高 | N/A |
| 简单提示去偏 | 0.40~0.60 | 中等 | 差 |
| MPF + 反事实基线 | **~0.05** | **低** | **是** |
| MPF + HR 基线 | **~0.08** | **低** | **是** |

### 不同基线类型的对齐结果

| 基线类型 | 目标分布特征 | MPF 对齐后 KL | 情感偏移修正 |
|:---|:---|:---:|:---:|
| 反事实基线（绝对平等） | 各大学背景情感均匀分布 | ≤ 0.05 | 消除偏好 |
| HR 基线（Top University 偏向） | 保留 HR 专家的真实偏好 | ≤ 0.10 | 对齐专家判断 |
| 随机基线 | 随机分布 | ≤ 0.15 | 趋近随机 |

### 关键发现

1. **双基线验证**：MPF 能同时对齐反事实基线（绝对公平）和 HR 基线（现实偏好），表明框架的灵活性
2. **未见问题泛化**：在训练问题之外的新问题上，MPF 的对齐效果依然保持，KL 散度增幅小于 0.03
3. **无需微调**：整个过程不修改模型参数，仅通过推理时的多视角采样实现

## 亮点与洞察

- **即插即用**：与已部署的 LLM 兼容，无需访问模型权重或训练数据，真正做到后部署对齐
- **可解释性**：视角分解提供了偏见的可解释分析——每个视角成分对应一种可识别的观点倾向
- **双目标设计**：既支持"绝对公平"（反事实基线），也支持"人类期望公平"（HR 基线），应用场景灵活
- **SAGED 集成**：利用自动化偏见基准构建管道，减少人工干预

## 局限性

1. **领域受限**：实验仅在招聘场景（大学偏见）中验证，其他偏见类型（种族、性别）的泛化性有待探索
2. **Workshop 论文**：作为 AIW Workshop 论文，实验规模和深度相对有限
3. **基线获取成本**：需要人类专家提供基线分布，在实际应用中获取成本可能较高
4. **计算开销**：多视角生成需要多次采样，推理延迟会增加
5. **分解质量**：视角分解的质量依赖于基线分布的特征，对于复杂的多维偏见场景可能效果不佳

## 相关工作与启发

- **SAGED Pipeline**：MPF 的前置工作，自动构建偏见基准和提取基线分布
- **DPO / RLHF**：需要微调的对齐方法，MPF 提供了无参数修改的替代方案
- **Constitutional AI**：通过规则约束生成，MPF 通过分布对齐实现类似目标
- **推理时干预**：与 Inference-time Intervention (ITI) 等方法类似，但 MPF 不需要内部表示访问

**启发**：将偏见缓解建模为分布对齐问题，而非分类问题，可能为因果公平性研究提供新的技术路径。

## 评分

| 维度 | 分数 (1-10) | 评价 |
|:---|:---:|:---|
| 创新性 | 6 | 多视角分解思路有新意，但整体框架较为简单 |
| 技术深度 | 5 | Workshop 论文，理论分析不够深入 |
| 实验充分性 | 5 | 仅单一场景验证，缺少大规模对比实验 |
| 写作质量 | 7 | 表达清晰，框架描述直观 |
| 实用价值 | 7 | 无需微调的后部署方案有较高实用性 |
| 总体推荐 | 6 | 有价值的工作方向，但需更多验证 |

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Social Debiasing for Fair Multi-modal LLMs](../../ICCV2025/causal_inference/social_debiasing_for_fair_multi-modal_llms.md)
- [\[ACL 2025\] On the Reliability of Large Language Models for Causal Discovery](../../ACL2025/causal_inference/llm_causal_discovery_reliability.md)
- [\[NeurIPS 2025\] Revealing Multimodal Causality with Large Language Models](../../NeurIPS2025/causal_inference/revealing_multimodal_causality_with_large_language_models.md)
- [\[ACL 2025\] Counterfactual-Consistency Prompting for Relative Temporal Understanding in Large Language Models](../../ACL2025/causal_inference/counterfactual-consistency_prompting_for_relative_temporal_understanding_in_larg.md)
- [\[NeurIPS 2025\] Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models](../../NeurIPS2025/causal_inference/counterfactual_reasoning_for_steerable_pluralistic_value_alignment_of_large_lang.md)

<!-- RELATED:END -->
