---
title: >-
  [论文解读] JoPA: Explaining Large Language Model's Generation via Joint Prompt Attribution
description: >-
  [ACL 2025][LLM/NLP][提示学习] 提出 JoPA（Joint Prompt Attribution）框架，将 LLM 生成任务的 prompt 归因建模为组合优化问题，用概率搜索算法高效寻找对输出有因果影响的输入 token 组合，解决了现有方法忽略 token 间协同效应的问题。
tags:
  - ACL 2025
  - LLM/NLP
  - 提示学习
  - interpretability
  - counterfactual
  - combinatorial optimization
  - generation explanation
---

# JoPA: Explaining Large Language Model's Generation via Joint Prompt Attribution

**会议**: ACL 2025  
**arXiv**: [2405.20404](https://arxiv.org/abs/2405.20404)  
**代码**: https://github.com/yuruic/JoPA  
**领域**: LLM/NLP  
**关键词**: prompt attribution, interpretability, counterfactual, combinatorial optimization, generation explanation

## 一句话总结
提出 JoPA（Joint Prompt Attribution）框架，将 LLM 生成任务的 prompt 归因建模为组合优化问题，用概率搜索算法高效寻找对输出有因果影响的输入 token 组合，解决了现有方法忽略 token 间协同效应的问题。

## 研究背景与动机

**领域现状**：LLM 生成的可解释性研究主要集中在分类任务和下一词预测，很少有工作解释"输入 prompt 如何影响整个生成序列"。

**现有痛点**：Captum 等方法逐个删除 token 测量影响，忽略了 token 间的语义交互——如"doctor"和"patient"单独删除影响小，但组合删除影响大。

**核心矛盾**：穷举所有 token 组合在长输入下不可行（指数级搜索空间），如何高效找到关键组合？

**本文目标** 设计高效算法在离散空间中搜索对生成有最大因果影响的 prompt token 组合。

**切入角度**：反事实解释——"如果去掉这些 token，生成会怎样变化？"变化最大的组合就是最重要的。

**核心 idea**：将 prompt 归因转化为 mask 的组合优化问题，用梯度引导+概率更新的搜索算法在离散空间高效求解。

## 方法详解

### 整体框架
对输入 token 序列学习一个二值 mask -> 优化目标：mask 掉的 token 应使生成概率变化最大 -> 用概率搜索算法迭代优化 mask -> 输出最重要的 token 组合作为解释。

### 关键设计

1. **反事实目标函数**

    - 最大化：mask 掉部分 token 后生成概率的变化量
    - 约束：mask 掉的 token 数量尽可能少（稀疏性）
    - 设计动机：找到最小且最有影响力的 token 子集

2. **概率搜索算法**

    - 每个 token 位置维护一个被 mask 的概率 $p_i$
    - 梯度信息指导概率更新方向
    - 采样+评估+更新的迭代过程
    - 设计动机：利用梯度提供搜索方向，概率机制在离散空间中平衡探索和利用

3. **生成变化度量**

    - 综合考虑：生成概率变化、词频变化、语义相似度变化
    - 设计动机：多维度衡量"生成有多大不同"

### 评估指标

| 指标 | 衡量内容 | 说明 |
|------|---------|------|
| 概率忠实度 | mask 后生成概率变化 | 越大越好 |
| 词频忠实度 | mask 后输出词频变化 | 越大越好 |
| 语义忠实度 | mask 后语义相似度下降 | 越大越好 |
| 稀疏性 | 被 mask 的 token 比例 | 越小越好 |

## 实验关键数据

### 主实验 — 三个任务的忠实度对比
| 方法 | 摘要任务 概率忠实度 | QA 任务 概率忠实度 | 通用指令 概率忠实度 | 平均稀疏性 |
|------|-------------------|-----------------|-------------------|-----------|
| Random | 低 | 低 | 低 | 10% |
| Captum (逐token) | 中 | 中 | 中 | 10% |
| Gradient saliency | 中 | 中偏高 | 中 | 10% |
| **JoPA** | **高** | **高** | **高** | **8%** |

### 消融实验
| 配置 | 概率忠实度 | 说明 |
|------|-----------|------|
| JoPA (完整) | 最高 | 梯度+概率搜索 |
| 无梯度引导 | -15% | 纯概率搜索效率低 |
| 无概率更新 | -10% | 纯梯度贪心易陷局部最优 |
| 逐token (Captum) | -25% | 忽略组合效应 |

### 关键发现
- **JoPA 在所有三个任务上一致超越基线**，且只需 mask 约 8% 的 token
- **组合效应确实存在**：逐 token 方法遗漏了 20-30% 的重要信息
- **梯度引导和概率搜索的结合是关键**：任一缺失都显著降低性能
- **解释可用于安全分析**：识别出导致有害生成的 prompt 片段
- **解释也可用于提升效率**：去除不重要 token 保持生成质量

## 亮点与洞察
- **将生成解释转化为组合优化**是一个优雅的形式化——将可解释性从"计算重要性分数"提升到"寻找因果子集"
- **概率搜索+梯度引导**的混合算法在离散空间优化上是一个通用工具，可迁移到其他组合优化场景
- **"doctor和patient"的例子**直观展示了为什么组合归因比独立归因更准确

## 局限与展望
- 搜索算法仍有一定计算成本
- 对长输入（>2K tokens）的扩展性待验证
- 改进方向：分层搜索（先粗后细）、与 attention 分析对比

## 相关工作与启发
- **vs Captum**：Captum 逐 token 归因，JoPA 考虑组合效应
- **vs LIME/SHAP**：这些是分类任务的归因，JoPA 扩展到生成任务
- **vs CoT 自解释**：CoT 可能不忠实，JoPA 通过反事实保证因果忠实

## 评分
- 新颖性: ⭐⭐⭐⭐ 组合优化形式化+生成任务归因的结合新颖
- 实验充分度: ⭐⭐⭐⭐ 三任务+多指标+消融
- 写作质量: ⭐⭐⭐⭐ 形式化清晰
- 价值: ⭐⭐⭐⭐ 对 LLM 可解释性和安全性有实用价值

<!-- RELATED:START -->

## 相关论文

- [SongComposer: A Large Language Model for Lyric and Melody Generation in Song Composition](songcomposer_llm_lyric_melody_generation.md)
- [OPTS: Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers](bandit-based_prompt_design_strategy_selection_improves_prompt_optimizers.md)
- [What Makes a Good Natural Language Prompt?](good_natural_language_prompt.md)
- [Representation Bending for Large Language Model Safety](repbend_representation_bending_safety.md)
- [Binary Classifier Optimization for Large Language Model Alignment](bco_binary_classifier_alignment.md)

<!-- RELATED:END -->
