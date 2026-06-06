---
title: >-
  [论文解读] IRT-Router: Effective and Interpretable Multi-LLM Routing via Item Response Theory
description: >-
  [ACL 2025][可解释性][LLM routing] IRT-Router 借鉴心理测量学的项目反应理论（IRT），将 LLM 视为"考生"、query 视为"考题"，学习多维能力向量和难度/区分度参数实现可解释的多 LLM 路由…
tags:
  - "ACL 2025"
  - "可解释性"
  - "LLM routing"
  - "item response theory"
  - "multi-model selection"
  - "interpretability"
  - "cost optimization"
---

# IRT-Router: Effective and Interpretable Multi-LLM Routing via Item Response Theory

**会议**: ACL 2025  
**arXiv**: [2506.01048](https://arxiv.org/abs/2506.01048)  
**代码**: [https://github.com/Mercidaiha/IRT-Router](https://github.com/Mercidaiha/IRT-Router)  
**领域**: 可解释性  
**关键词**: LLM routing, item response theory, multi-model selection, interpretability, cost optimization

## 一句话总结
IRT-Router 借鉴心理测量学的项目反应理论（IRT），将 LLM 视为"考生"、query 视为"考题"，学习多维能力向量和难度/区分度参数实现可解释的多 LLM 路由，在 OOD 场景下达 87%+ 准确率且成本仅为 GPT-4o 的 1/30。

## 研究背景与动机

### 领域现状

**领域现状**：领域现状**：使用多个 LLM 时需要根据 query 特点自动选择最合适的模型，平衡性能和成本

**现有痛点**：现有路由方法（RouteLLM、RouterBench）用简单启发式或黑箱分类器，缺乏可解释性，无法说明"为什么路由到这个模型"

**核心矛盾**：需要同时解决可解释性、cold-start（新 query 如何路由）、性能-成本权衡三个问题

**核心 idea**：IRT 天然建模"能力-难度"关系，将其迁移到 LLM 路由可同时获得可解释性和效果

## 方法详解

### 整体框架
两个实现版本：(1) **MIRT-Router**（多维IRT）：$\hat{P}(q_i, M_j) = 1/(1 + \exp(-a_i^T \theta_{M_j} + b_i))$，$\theta_{M_j}$ 为 LLM 能力向量，$a_i$ 为区分度，$b_i$ 为难度；(2) **NIRT-Router**（神经IRT）：引入 relevance vector 和神经网络交互函数。

### 关键设计

1. **IRT 建模**：每个 LLM 有多维能力向量 $\theta_{M_j}$，每个 query 有难度 $b_i$ 和区分度 $a_i$，参数通过 embedding + 线性变换学习
2. **Cold-start Warm-up**：对未见 query，用邻近已知 query 的嵌入插值：$e_{q_i}' = (1-\lambda) e_{q_i} + \lambda \cdot \text{mean(neighbors)}$，$\lambda=0.3\text{-}0.4$ 最优
3. **评分函数**：$S(q_i, M_j) = \alpha \hat{P}(q_i, M_j) - \beta C(M_j)$，$\alpha+\beta=1$ 平衡性能和成本

## 实验关键数据

### 主实验

| 方法 | 准确率 | 成本 | Reward |
|------|-------|------|--------|
| MIRT-Router | 80.67% | $0.42 | 63.89 |
| RouterBench | 80.01% | $1.15 | 62.23 |
| RouteLLM | 77.25% | $12.80 | 42.00 |
| GPT-4o only | 77.53% | $12.93 | 42.02 |

### OOD 场景（20 个候选 LLM，12 个数据集）

| 方法 | 准确率 | 成本 |
|------|-------|------|
| MIRT-Router | 87.12% | $0.14 |
| NIRT-Router | 87.37% | $0.15 |

### Top-k 路由准确率（MIRT-Router）

| 场景 | Top-1 | Top-2 | Top-3 | Top-5 |
|------|-------|-------|-------|-------|
| ID | 2.72% | 9.88% | 32.51% | 47.85% |
| OOD | 2.15% | 7.50% | 27.29% | 39.47% |

Top-1 较低因为多个 LLM 能力相近（如 DeepSeek-Chat/Coder 均 81%），但 Top-3 已达 32.5%。路由分析显示：高难度 query 80% 路由到 DeepSeek-Chat，低难度 query 99% 路由到最便宜的 Qwen2.5-32B-GPTQ（$0.2/M）。

### 关键发现
- **性能-成本最优**：准确率比 GPT-4o 高 3%，成本仅 1/30
- **可解释性**：能力向量和难度分数有明确语义（DeepSeek-Chat 能力最强=81%，GPT-4o=78%）
- **Cold-start 有效**：warm-up 机制显著提升 OOD 表现，对 NIRT-Router 影响更大

## 亮点与洞察
- **IRT→LLM 路由的跨领域迁移**很优雅：心理测量学的成熟理论直接适用于 LLM 能力评估
- **可解释性是核心卖点**：不仅路由效果好，还能解释每个 LLM 擅长什么、每个 query 难在哪里

## 局限与展望
- Top-1 路由准确率较低（2-3%），因为多个模型能力相似，能力向量在高维空间中距离较近
- 对全新 LLM（训练时未见过的模型）泛化有限，需要少量校准数据才能初始化新模型的能力向量
- 路由器对成本参数变化不够敏感，$\alpha/\beta$ 比值的微调对路由结果影响有限
- Cold-start warm-up 的邻近度量基于 embedding 空间的欧氏距离，可能无法准确反映 query 的真实难度相似性
- IRT 模型假设能力和难度是静态的，但 LLM 能力会随 API 版本更新而变化，需要定期重新校准
- NIRT 版本虽引入神经网络提升灵活性，但牺牲了 MIRT 版本的完全可解释性

## 评分
- 新颖性: ⭐⭐⭐⭐ IRT 用于 LLM 路由是巧妙的跨领域迁移
- 实验充分度: ⭐⭐⭐⭐⭐ 20 个 LLM × 12 个数据集 × ID+OOD 场景
- 写作质量: ⭐⭐⭐⭐ IRT 理论介绍清晰，可解释性分析充分
- 价值: ⭐⭐⭐⭐⭐ 对多 LLM 部署场景有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Diagnosing the Reliability of LLM-as-a-Judge via Item Response Theory](../../ICML2026/interpretability/diagnosing_the_reliability_of_llm-as-a-judge_via_item_response_theory.md)
- [\[ACL 2025\] Safety is Not Only About Refusal: Reasoning-Enhanced Fine-tuning for Interpretable LLM Safety](safety_is_not_only_about_refusal_reasoning-enhanced_fine-tuning_for_interpretabl.md)
- [\[ACL 2025\] Establishing Trustworthy LLM Evaluation via Shortcut Neuron Analysis](shortcut_neuron_eval.md)
- [\[ACL 2025\] CLEME2.0: Towards Interpretable Evaluation by Disentangling Edits for Grammatical Error Correction](cleme2_gec_evaluation.md)
- [\[ICML 2025\] What Makes an Ensemble (Un)interpretable?](../../ICML2025/interpretability/what_makes_an_ensemble_un_interpretable.md)

</div>

<!-- RELATED:END -->
