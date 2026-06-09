---
title: >-
  [论文解读] Evolving Prompts In-Context: An Open-ended, Self-replicating Perspective
description: >-
  [ICML 2025][可解释性][剪枝] 提出 PromptQuine 框架，通过进化搜索对 ICL prompt 进行 token 级剪枝，发现将清晰示例剪成看似"乱码"的子序列反而能提升 LLM 性能，且匹配或超越 SOTA prompt 优化方法。
tags:
  - "ICML 2025"
  - "可解释性"
  - "剪枝"
  - "evolutionary search"
  - "in-context learning"
  - "提示学习"
  - "open-endedness"
---

# Evolving Prompts In-Context: An Open-ended, Self-replicating Perspective

**会议**: ICML 2025  
**arXiv**: [2506.17930](https://arxiv.org/abs/2506.17930)  
**代码**: -  
**领域**: 可解释性  
**关键词**: prompt pruning, evolutionary search, in-context learning, PromptQuine, open-endedness  

## 一句话总结

提出 PromptQuine 框架，通过进化搜索对 ICL prompt 进行 token 级剪枝，发现将清晰示例剪成看似"乱码"的子序列反而能提升 LLM 性能，且匹配或超越 SOTA prompt 优化方法。

## 研究背景与动机

传统观点认为，精心设计的指令和示例是 ICL 性能的关键。然而，近期研究发现"非自然语言"(unnatural language) prompt 有时反而效果更好。本文系统挑战这一传统认知：

- **核心假设 (Partial Context Hypothesis)**：给定自然语言 ICL prompt $\mathbf{x}=(x_1,...,x_n)$，通过剪枝部分 token 得到子序列 $\mathbf{z}=(z_1,...,z_m), m \leq n$，可以显著提升任务性能
- **动机**：LLM 可能只经历了"表面对齐"(superficial alignment)，其内部假设优先于人类语言的显式结构；部分输入特征对任务预测是冗余的
- **关键发现**：现有归因方法和 prompt 压缩算法在指导剪枝方面均不可靠

## 方法详解

### 1. 问题形式化：压缩即引导搜索

将传统 prompt 压缩问题重新定义为引导式搜索问题：

$$\mathbf{z}^* = \arg\max_{\mathbf{z} \subseteq \mathbf{x}} f(\mathbf{z}; \mathbf{x}, \mathcal{D})$$

其中 $f$ 为不可微的任务目标函数，搜索空间为原始 prompt 的所有固定顺序子序列。

### 2. 基线：Hill-climbing 搜索 (TAPruning)

采用 Threshold Accepting 算法，从左到右逐 token 尝试剪枝：
- 若删除某 token 后验证集性能不降（或降幅在阈值内，如 $\geq 96\%$ 当前最优），则接受
- 多轮迭代直到无法继续剪枝
- 已经能匹配 SOTA 方法如 Promptbreeder

### 3. 搜索景观分析

- 通过随机化剪枝顺序验证景观的**多峰性** (multimodal)
- 随机搜索 (RS) 效率极低，而进化搜索 (ES) 在获取高质量 prompt 方面显著更优
- RS 相对 ES 的成功率随任务难度增加趋近于零

### 4. PromptQuine：进化搜索框架

基于遗传算法 (GA)，核心设计：
- **编码**：二进制 token mask 作为基因型，剪枝后的 ICL prompt 作为表型
- **变异**：bit-flip (1→0) 操作，随机选择翻转 $\{1,2,3,4\}$ 个 bit
- **选择**：锦标赛选择 + 降低选择压力避免局部最优
- **正则化进化**：仅新后代竞争种群位置，有效解决 ICL 景观中的过早收敛
- **校准后重排**：使用完整验证集精度对精英 prompt 重新排序

## 实验结果

### 主实验：分类任务 (Llama-3-8B-Instruct)

| 方法 | SST-2 | Subj | AG's News | Yelp-5 | SNLI | Yahoo | 平均 |
|------|-------|------|-----------|--------|------|-------|------|
| ICL (1-shot) | 95.9 | 66.7 | 83.7 | 52.2 | 61.9 | 57.1 | 69.6 |
| Promptbreeder | 96.0 | 83.6 | 88.6 | 59.3 | 64.2 | 62.9 | 75.8 |
| TAPruning (1-shot) | 95.0 | 74.5 | 88.6 | 60.2 | 68.6 | 61.7 | 74.8 |
| **PromptQuine (1-shot)** | **96.2** | **86.5** | **89.2** | 59.7 | 69.2 | 64.2 | **77.5** |
| **PromptQuine (4-shot)** | 96.4 | **93.1** | 89.4 | **64.3** | **78.6** | **66.2** | **81.3** |

- PromptQuine 在 1-shot 设置下匹配或超越所有 SOTA 方法
- 4-shot 时性能进一步跃升，SNLI 从 69.2% → 78.6%
- 同时实现约 53% 的 prompt 压缩率

### 消融：运行效率

TAPruning 和 PromptQuine 是**首个能在分钟级完成的 token 级搜索方法**，运行效率远超 RLPrompt 等方法。

## 亮点

- 颠覆性发现：将 ICL 示例剪成"乱码"反而提升性能，揭示 LLM 对 prompt 的偏好与人类直觉截然不同
- 将 prompt 压缩重新定义为搜索问题，统一了压缩与优化视角
- 进化搜索框架天然支持并行化，具有良好的可扩展性
- 方法简洁实用，无需访问模型梯度，适用于任何黑盒 LLM

## 相关工作对比

| 方法 | 搜索空间 | 是否需要梯度 | 搜索粒度 | 运行时间 |
|------|---------|------------|---------|---------|
| RLPrompt | 任意 token | 是 | Token 级 | 数小时 |
| EvoPrompt | 自然语言指令 | 否 | 指令级 | 数小时 |
| Promptbreeder | 自然语言 + 变异 prompt | 否 | 指令级 | 数十分钟 |
| LLMLingua | 压缩 | 否 | Token 级 | 数秒 |
| **PromptQuine** | ICL 子序列 | 否 | Token 级 | **数分钟** |

PromptQuine 是首个在分钟级完成 token 级搜索的方法，同时不需要模型梯度。

## 局限性

- 搜索空间限制为固定顺序子序列，未探索 token 重排序
- 进化搜索对超参数（种群大小、变异率等）较敏感
- 发现的"最优"剪枝模式缺乏可解释性，难以转化为通用规律
- 仅在分类和生成任务上验证，对更复杂推理任务的效果未知
- 未探索剪枝策略在不同 LLM 间的迁移性

## 评分

⭐⭐⭐⭐ — 视角新颖且实验充分，将"乱码优于精心设计"这一反直觉发现系统化，对理解 ICL 机制有重要启示。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] On the Power of Context-Enhanced Learning in LLMs](on_the_power_of_context-enhanced_learning_in_llms.md)
- [\[NeurIPS 2025\] Evaluating LLMs in Open-Source Games](../../NeurIPS2025/interpretability/evaluating_llms_in_open-source_games.md)
- [\[ICML 2025\] LANTERN: Modeling User Behavior from Adaptive Surveys with Supplemental Context](modeling_user_behavior_from_adaptive_surveys_with_supplemental_context.md)
- [\[CVPR 2025\] Open Ad-Hoc Categorization with Contextualized Feature Learning](../../CVPR2025/interpretability/open_ad-hoc_categorization_with_contextualized_feature_learning.md)
- [\[ACL 2026\] Interpreting Style Representations via Style-Eliciting Prompts](../../ACL2026/interpretability/interpreting_style_representations_via_style-eliciting_prompts.md)

</div>

<!-- RELATED:END -->
