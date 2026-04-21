---
title: >-
  [论文解读] Principled Fine-tuning of LLMs from User-Edits: A Medley of Preference, Supervision, and Reward
description: >-
  [NeurIPS 2025][用户编辑] 系统研究如何利用用户编辑数据微调 LLM，将偏好、监督标签和代价三种反馈类型统一起来，并提出一种简单的集成方法，在不同用户分布下实现鲁棒适应。
tags:
  - NeurIPS 2025
  - 用户编辑
  - LLM微调
  - 偏好学习
  - 监督学习
  - 集成学习
---

# Principled Fine-tuning of LLMs from User-Edits: A Medley of Preference, Supervision, and Reward

**会议**: NeurIPS 2025

**arXiv**: [2601.19055](https://arxiv.org/abs/2601.19055)

**代码**: 无

**领域**: LLM/NLP

**关键词**: 用户编辑, LLM微调, 偏好学习, 监督学习, 集成学习

## 一句话总结

系统研究如何利用用户编辑数据微调 LLM，将偏好、监督标签和代价三种反馈类型统一起来，并提出一种简单的集成方法，在不同用户分布下实现鲁棒适应。

## 研究背景与动机

在基于 LLM 的写作助手和代码代理等应用中，用户会自然地对模型输出进行编辑。这些编辑数据（context + agent response + user edit）是个性化 LLM 的宝贵信号来源，但如何有效利用这类数据尚缺乏理论基础。

核心挑战：

**多种反馈类型的统一**: 用户编辑数据同时蕴含偏好信号（原始 vs 编辑后）、监督标签（编辑结果）和代价信号（编辑幅度）

**各类反馈的权衡**: 不同反馈类型在不同用户和数据分布下有不同的优缺点

**分布鲁棒性**: 测试时用户的编辑风格可能与训练时不同，需要鲁棒的学习策略

## 方法详解

### 整体框架

给定用户编辑数据集 $\{(c_i, y_i^{\text{agent}}, y_i^{\text{edit}})\}$，研究三种学习范式：
- **偏好学习**: 将 $(y^{\text{agent}}, y^{\text{edit}})$ 视为偏好对，用 DPO/RLHF 优化
- **监督学习**: 直接以 $y^{\text{edit}}$ 为目标进行 SFT
- **代价学习**: 以编辑距离为代价信号，最小化期望代价

### 关键设计

**1. 三种反馈类型的理论分析**

对每种学习方法推导误差界：
- 偏好学习: 适合用户编辑充分且多样的场景，但对编辑质量敏感
- 监督学习: 当编辑结果接近最优时表现好，但可能过拟合特定编辑风格
- 代价学习: 利用编辑幅度信息更精细，但需要准确的代价函数

**2. 集成方法 (Ensembling Procedure)**

- 在多个验证任务上评估三种方法的性能
- 通过凸组合权重 $\alpha$ 最优地混合三种策略: $\pi_{\text{ensemble}} = \alpha_1 \pi_{\text{pref}} + \alpha_2 \pi_{\text{sup}} + \alpha_3 \pi_{\text{cost}}$
- 权重通过在验证集上最小化集成损失自动确定
- 简单但有效，能适应不同用户-数据分布场景

**3. 理论权衡分析**

- 证明三种方法在不同假设下各有优势，不存在一种方法在所有场景下都最优
- 集成方法的性能界适应性地逼近三种方法中最优的那个

### 损失函数 / 训练策略

- 偏好学习: $\mathcal{L}_{\text{pref}} = -\mathbb{E}[\log \sigma(r(y^{\text{edit}}) - r(y^{\text{agent}}))]$
- 监督学习: $\mathcal{L}_{\text{sup}} = -\mathbb{E}[\log \pi(y^{\text{edit}} | c)]$
- 代价学习: $\mathcal{L}_{\text{cost}} = \mathbb{E}[d(y, y^{\text{edit}}) \cdot \nabla \log \pi(y | c)]$

## 实验关键数据

### 主实验

在两个领域的评估结果 (基于 Gao et al. 2024 的设置):

写作助手任务 (Win Rate vs Reference):

| 方法 | 轻度编辑用户 | 中度编辑用户 | 重度编辑用户 | 整体 |
|------|------------|------------|------------|------|
| SFT-only | 55.2% | 48.3% | 42.1% | 48.5% |
| DPO | 52.8% | 51.7% | 50.3% | 51.6% |
| Cost-based | 50.1% | 52.4% | **53.8%** | 52.1% |
| Ensemble (Ours) | **56.3%** | **54.1%** | 53.2% | **54.5%** |

代码代理任务 (Pass@1):

| 方法 | 简单修改 | 重构编辑 | 混合场景 |
|------|---------|---------|---------|
| SFT-only | **72.1%** | 45.3% | 56.8% |
| DPO | 68.5% | 51.2% | 58.3% |
| Cost-based | 65.3% | 49.8% | 56.1% |
| Ensemble (Ours) | 71.8% | **53.6%** | **61.2%** |

### 消融实验

集成权重随用户类型变化的分析:

| 用户类型 | $\alpha_{\text{pref}}$ | $\alpha_{\text{sup}}$ | $\alpha_{\text{cost}}$ |
|---------|-----|-----|-----|
| 轻度编辑 | 0.15 | 0.72 | 0.13 |
| 中度编辑 | 0.38 | 0.35 | 0.27 |
| 重度编辑 | 0.45 | 0.12 | 0.43 |

### 关键发现

1. 没有单一反馈类型在所有场景下最优，验证了理论分析
2. SFT 在轻度编辑用户上表现最好（编辑即最优解），DPO 在重度编辑用户上更鲁棒
3. 集成方法在所有场景下接近或达到最优，展现出良好的适应性
4. 代价学习在编辑幅度差异大的场景中特别有价值

## 亮点与洞察

- **统一视角**: 首次从理论上统一分析用户编辑数据中的三种反馈类型
- **实用导向**: 用户编辑是真实应用中自然产生的数据，方法可直接应用
- **简单有效**: 集成方法无需复杂架构，通过权重调整即可适应不同场景

## 局限与展望

1. 集成方法需要验证集来调整权重，在数据稀缺时可能不稳定
2. 当前假设用户编辑总是改善结果，未考虑错误编辑的情况
3. 理论分析基于线性或低复杂度假设，与实际深度模型有差距
4. 仅在两个领域验证，通用性有待扩展

## 相关工作与启发

- **RLHF/DPO**: 标准的 LLM 偏好学习方法
- **Gao et al. 2024**: 用户编辑作为学习信号的先驱工作
- **集成学习**: 简单的模型组合在实践中常出奇制胜

## 评分

- ⭐ 创新性: 8/10 — 首次理论化用户编辑学习,统一三种反馈
- ⭐ 实用性: 8/10 — 对写作助手、代码代理等应用直接相关
- ⭐ 写作质量: 8/10 — 理论与实验结合得当

<!-- RELATED:START -->

## 相关论文

- [GiFT: Gibbs Fine-Tuning for Code Generation](../../ACL2025/code_intelligence/gift_gibbs_fine_tuning_code_gen.md)
- [SparseLoRA: Accelerating LLM Fine-Tuning with Contextual Sparsity](../../ICML2025/code_intelligence/sparselora_accelerating_llm_fine-tuning_with_contextual_sparsity.md)
- [EffiCoder: Enhancing Code Generation in Large Language Models through Efficiency-Aware Fine-tuning](../../ICML2025/code_intelligence/efficoder_enhancing_code_generation_in_large_language_models_through_efficiency-.md)
- [CoRe: Benchmarking LLMs' Code Reasoning Capabilities through Static Analysis Tasks](core_benchmarking_llms_code_reasoning_capabilities_through_static_analysis_tasks.md)
- [IMSE: Intrinsic Mixture of Spectral Experts Fine-tuning for Test-Time Adaptation](../../ICLR2026/code_intelligence/imse_intrinsic_mixture_of_spectral_experts_fine-tuning_for_test-time_adaptation.md)

<!-- RELATED:END -->
