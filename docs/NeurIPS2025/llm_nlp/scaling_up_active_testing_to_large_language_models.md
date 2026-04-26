---
title: >-
  [论文解读] Scaling Up Active Testing to Large Language Models
description: >-
  [NeurIPS 2025][LLM/NLP][active testing] 通过三项关键简化——用 in-context learning 构建固定代理模型、使用小代理模型评估大目标模型、无需目标模型预测进行数据采集——将 active testing 扩展到 LLM，风险估计误差比随机采样降低 25%-80%。
tags:
  - NeurIPS 2025
  - LLM/NLP
  - active testing
  - LLM evaluation
  - risk estimation
  - surrogate model
  - label efficiency
---

# Scaling Up Active Testing to Large Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2508.09093](https://arxiv.org/abs/2508.09093)  
**代码**: [GitHub](https://github.com/gabrielleberrada/scaling-up-active-testing)  
**领域**: llm_nlp  
**关键词**: active testing, LLM evaluation, risk estimation, surrogate model, label efficiency

## 一句话总结
通过三项关键简化——用 in-context learning 构建固定代理模型、使用小代理模型评估大目标模型、无需目标模型预测进行数据采集——将 active testing 扩展到 LLM，风险估计误差比随机采样降低 25%-80%。

## 研究背景与动机

**LLM 评估日益昂贵**：前沿模型越来越复杂，标注成本高且评估数据可能泄露到训练集，需要持续动态地收集新评估数据

**Active testing 的瓶颈**：现有方法需要反复梯度训练代理模型、对所有池化数据做代理/目标模型推理，计算成本限制了其在 LLM 上的应用

**核心问题**：如何在保持 active testing 有效性的同时大幅降低计算成本，使其可扩展到 70B 级 LLM？

## 方法详解

### 整体框架

Active testing 通过智能选择标注哪些测试输入来更准确地估计模型风险 $R = \mathbb{E}[\ell(f(x), y)]$。本文解决三个计算瓶颈：

### 关键设计 1：固定代理模型（解决训练瓶颈）

传统方法在每次获取新标签后重新训练代理模型。本文改为：
- 用少量初始标注数据通过 **in-context learning** 一次性构建代理模型
- 之后保持代理模型**固定不变**
- 完全避免了循环内的梯度训练开销

### 关键设计 2：小代理模型（解决推理成本瓶颈）

代理模型可以显著小于目标模型：
- 用 7B 模型作为代理评估 70B 目标模型
- 甚至用 Gemma3-4B 或 Phi-2 评估 Llama-2-70B
- 旧模型（Llama 2）可有效作为新模型（Gemma 3）的代理

### 关键设计 3：无需目标模型预测（解决 N 次推理瓶颈）

用代理模型近似目标模型的预测：
- 采集函数从交叉熵 $H[\pi_m(y|x) \| p_f(y|x)]$ 简化为代理模型预测熵 $H[\pi_m(y|x)]$
- 目标模型预测从 N 次减少到 M 次（M << N）

### 风险估计

使用 LURE（Levelled Unbiased Risk Estimator）进行无偏风险估计：

$$\hat{R}_{\text{LURE}} = \frac{1}{M}\sum_{m=1}^M v_m \ell(f(x_{i_m}), y_{i_m})$$

其中 $v_m$ 是纠正非均匀采样的重要性权重。

### Bootstrap 误差估计

提出 bootstrap 方法估计单次 active testing 运行中的风险估计误差，为实际部署提供置信区间。

## 实验关键数据

### 主实验：风险估计误差降低

| 数据集 | 目标模型 | 代理模型 | 相对误差降低 |
|--------|----------|----------|-------------|
| SST-2 | 70B-few | 7B-few | ~50% |
| FPB | 70B-few | 7B-few | ~40% |
| HS | 70B-few | 7B-few | ~60% |
| Subj | 70B-few | 7B-few | ~30% |
| 平均 | - | - | **25%-80%** |

### 跨模型代理

| 代理模型 | 目标模型 | 有效性 |
|----------|----------|--------|
| Llama-2-7B | Llama-2-70B | 有效 |
| Gemma3-4B | Llama-2-70B | 有效 |
| Phi-2 | Llama-2-70B | 有效 |
| Llama-2-7B | Gemma3-4B | 有效 |

### 采样 vs 插值方法

| 方法 | 鲁棒性 | 说明 |
|------|--------|------|
| LURE（采样） | 高 | 对代理模型质量不敏感 |
| ASE（插值） | 低 | 对代理模型质量敏感，固定代理时退化 |

证实了固定代理时采样方法优于插值方法的预测。

### Bootstrap 误差估计

95% 置信区间包含真实误差的覆盖率为 **88%**（K≥100 时达 ~94%）。

### 关键发现

1. 不更新代理模型几乎不损失性能，但节省巨大计算成本
2. 代理模型可比目标模型小 10 倍甚至更多
3. 无需目标模型预测的采集函数（用预测熵替代交叉熵）效果出奇地好
4. 标签错误会严重影响 active testing——强代理模型反而可能更差

## 亮点与洞察

1. **三个瓶颈的优雅解决**：每个简化都有理论动机和实验验证
2. **逆直觉发现**：小代理模型评估大目标模型，效果甚至更好
3. **Bootstrap 诊断工具**：为实际部署提供了判断 active testing 是否有效的方法
4. **数据集筛选的副产品**：active testing 可用于 dataset curation，选择子集评估模型

## 局限与展望

1. 仅实验了文本分类任务，生成任务更复杂
2. 标签错误情况下 active testing 可能失效（SST-2 案例）
3. 对更具挑战性的 MMLU 数据集改进幅度较小
4. Bootstrap 估计缺乏理论收敛保证

## 相关工作与启发

- **Kossen et al. (2021, 2022)**：提出 active testing 的采样和插值方法，本文将其扩展到 LLM
- **TinyBenchmarks/DELE**：数据集精简方法，关注跨模型通用性；本文针对特定模型进行采集
- **启发**：active testing 可与评估基准的持续更新结合，应对数据泄露问题

## 评分

- 新颖性: ⭐⭐⭐⭐ 将已有方法扩展到 LLM，每个简化都合理但非根本创新
- 实验充分度: ⭐⭐⭐⭐ 多模型、多数据集、多设置，包含失败案例分析
- 写作质量: ⭐⭐⭐⭐⭐ 三个瓶颈的分析框架清晰，实验设计系统
- 价值: ⭐⭐⭐⭐ 对 LLM 评估效率提升有实用价值，尤其在标注昂贵的场景

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Genetic Instruct: Scaling up Synthetic Generation of Coding Instructions for Large Language Models](../../ACL2025/llm_nlp/genetic_instruct_scaling_up_synthetic_generation_of_coding_instructions_for_larg.md)
- [\[ACL 2025\] LESA: Learnable LLM Layer Scaling-Up](../../ACL2025/llm_nlp/lesa_learnable_llm_layer_scaling-up.md)
- [\[NeurIPS 2025\] PluralisticBehaviorSuite: Stress-Testing Multi-Turn Adherence to Custom Behavioral Policies](pluralistic_behavior_suite_stress-testing_multi-turn_adherence_to_custom_behavio.md)
- [\[NeurIPS 2025\] Solving Inequality Proofs with Large Language Models](solving_inequality_proofs_with_large_language_models.md)
- [\[ACL 2025\] Reversal of Thought: Enhancing Large Language Models with Preference-Guided Reverse Reasoning Warm-up](../../ACL2025/llm_nlp/reversal_of_thought_enhancing_large_language.md)

<!-- RELATED:END -->
