---
title: >-
  [论文解读] Learning to Retrieve User History and Generate User Profiles for Personalized Persuasiveness Prediction
description: >-
  [ACL 2026][说服力预测] 本文提出 ReCAP 框架，通过可训练的查询生成器和用户画像生成器，从用户历史记录中检索与说服相关的信息并构建上下文感知的用户画像，显著提升个性化说服力预测的效果。
tags:
  - ACL 2026
  - 说服力预测
  - 用户画像
  - 检索增强
  - DPO训练
  - 个性化
---

# Learning to Retrieve User History and Generate User Profiles for Personalized Persuasiveness Prediction

**会议**: ACL 2026  
**arXiv**: [2601.05654](https://arxiv.org/abs/2601.05654)  
**代码**: [GitHub](https://github.com/holi-lab/ReCAP)  
**领域**: 个性化推荐与用户建模  
**关键词**: 说服力预测, 用户画像, 检索增强, DPO训练, 个性化

## 一句话总结

本文提出 ReCAP 框架，通过可训练的查询生成器和用户画像生成器，从用户历史记录中检索与说服相关的信息并构建上下文感知的用户画像，显著提升个性化说服力预测的效果。

## 研究背景与动机

**领域现状**：LLM 在决策支持应用中越来越多地被用于评估和生成有说服力的消息，如健康指导、教育辅导和定向营销。这些场景要求系统能预测特定消息对特定用户的说服效果，即**说服力预测**。说服本质上是个性化的——同一论点可能对一个用户有效但对另一个无效，取决于信念、价值观、经历和推理风格等因素。

**现有痛点**：现有方法存在三个关键限制：（1）依赖预定义的显式用户属性（如意识形态、人口统计），而这些属性通常不可获取且无法捕捉深层说服相关因素；（2）使用启发式检索方法（如最近记录或随机采样）选择历史记录，无法针对当前说服上下文动态调整；（3）通用画像技术（如人口统计提取）对说服预测效果有限甚至有害。

**核心矛盾**：说服力预测需要**上下文依赖**的用户信息——哪些历史记录对当前说服有参考价值取决于帖子的话题和立场。但缺乏系统性框架来优化利用用户历史记录。

**本文目标**：设计一个可训练的用户画像框架，学习"检索什么"和"如何总结"。**切入角度**：将用户画像构建分解为查询生成和画像总结两个可学习模块，通过下游任务性能作为监督信号。**核心 idea**：有效的用户画像是上下文依赖和预测器特定的，而非静态属性。

## 方法详解

### 整体框架

ReCAP 包含三阶段推理流程：（1）检索阶段——可训练查询生成器 $\phi^{\text{query}}$ 生成用户聚焦的检索查询，从用户历史 $R_u$ 中检索 top-k 记录；（2）画像阶段——可训练画像生成器 $\phi^{\text{prof}}$ 将检索记录总结为文本形式的用户画像 $P_i$；（3）预测阶段——预测器 $\mathcal{M}^{\text{pred}}$ 基于帖子、评论和画像预测观点改变。训练通过三步进行：画像器 DPO 训练、记录级说服效用评分、查询生成器 DPO 训练。

### 关键设计

1. **基于 DPO 的画像器训练**：

    - 功能：训练画像生成器产出对说服预测最有帮助的用户画像
    - 核心思路：由于没有 ground-truth 画像标注，采用弱监督方式——随机采样多组历史记录生成候选画像，用下游预测 F1 作为画像质量度量，构建偏好对（高 F1 画像 vs 低 F1 画像），通过 DPO 训练画像器偏好产出高质量画像
    - 设计动机：将"什么是好的用户画像"这一开放问题转化为可优化的偏好学习目标

2. **记录级说服效用评分**：

    - 功能：为每条用户历史记录估计其对说服预测的贡献度
    - 核心思路：将记录随机分组（每组5条），重复3次分组过程，对每组用训练好的画像器生成3个画像（温度0.7），聚合包含该记录的所有画像的 F1 分数作为其效用分数
    - 设计动机：无法直接获取"哪条记录对说服最有用"的标注，通过边际贡献估计间接获得监督信号

3. **说服感知的查询生成器**：

    - 功能：生成以用户属性为目标的检索查询，而非简单使用帖子文本作为查询
    - 核心思路：两阶段训练——先让模型生成用户聚焦问题（如"用户对政府干预个人选择的核心价值观是什么？"），再训练模型基于帖子和问题生成检索查询，通过 NDCG@5（基于效用分数）构建偏好对进行 DPO 训练
    - 设计动机：帖子文本缺少隐含的用户属性信息（价值观、经历等），直接用帖子作查询无法检索到真正与说服相关的记录

### 损失函数 / 训练策略

画像器和查询生成器均采用 DPO 训练，以 Llama-3.1-8B-Instruct 为骨干模型。预测器保持冻结（Llama-3.1-8B、Llama-3.3-70B、GPT-4o-mini），确保画像框架的通用性。训练无需 ground-truth 标注，完全基于下游任务性能信号。

## 实验关键数据

### 主实验

| 方法 | Llama-8B F1 | Llama-70B F1 | GPT-4o-mini F1 |
|------|------------|-------------|----------------|
| No Personalization | 0.346 | 0.328 | 0.253 |
| PAG | 0.257 | 0.314 | 0.083 |
| RecurSumm | 0.313 | 0.414 | 0.105 |
| HSumm | 0.324 | 0.406 | 0.113 |
| Retrieval-only | 0.295 | 0.418 | 0.132 |
| **ReCAP (Ours)** | **0.400** | **0.466** | **0.279** |

### 消融实验

| 配置 | Llama-70B F1 | 说明 |
|------|-------------|------|
| Our retrieval + Our profiler | **0.466** | 完整模型 |
| BGE retrieval + Our profiler | 0.445 | 换用语义检索 |
| Our retrieval + Base profiler | 0.393 | 换用未训练画像器 |
| Our retrieval + Demographic | 0.384 | 换用人口统计画像 |
| HyDE retrieval + Our profiler | 0.451 | 换用 HyDE 检索 |

### 关键发现
- 现有个性化框架（PAG、HSumm、RecurSumm）在说服预测任务上迁移效果差，通用画像甚至降低性能
- 训练后的画像器在所有预测器上一致优于人口统计画像和未训练画像器
- 有效的画像维度随帖子主题和预测器模型而变化——政治帖子依赖群体认同，社会道德帖子依赖个人价值观
- 不同预测器偏好不同的用户记录（top-5 overlap 仅 0.24-0.28），说明画像应是预测器特定的
- 推理成本仅为全历史摘要方法的 1/6 到 1/13

## 亮点与洞察
- 将"什么是好的用户画像"转化为 DPO 偏好学习，无需显式标注，方法论简洁有效
- 三步训练流程（画像器 → 效用评分 → 查询生成器）设计精巧，每步都有明确的优化目标
- 实证发现"画像应是上下文依赖和预测器特定的"具有广泛启发意义，挑战了静态用户画像的假设
- 在 OpinionQA 和 PRISM 上的泛化实验表明框架不限于 Reddit 场景

## 局限与展望
- 主要在 CMV Reddit 数据集上评估，扩展到短文本对话或实时推荐需要额外验证
- 记录级评分阶段消耗约 1.4B tokens（GPT-4o-mini），训练成本不可忽视
- 未探索用户画像随时间演变的动态建模
- 未来可将框架扩展到多模态交互历史和实时个性化场景

## 相关工作与启发
- **vs PAG (Richardson et al., 2023)**: PAG 用固定方式检索和总结，ReCAP 通过可训练查询和画像器动态适配
- **vs HSumm/RecurSumm**: 全历史摘要方法成本高且在说服任务上效果差，检索式方法更高效
- **vs Zhang et al. (2025)**: 微调预测器编码用户信息需要随新数据重训，ReCAP 保持预测器冻结更具扩展性

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个系统化的可训练用户画像框架用于说服预测，DPO 驱动无标注训练
- 实验充分度: ⭐⭐⭐⭐⭐ 多预测器 × 多检索策略 × 多画像方式 × 跨数据集泛化 × 效率分析
- 写作质量: ⭐⭐⭐⭐ 框架图清晰，分析层次丰富，实验设计严谨
- 价值: ⭐⭐⭐⭐ "上下文感知 + 预测器特定"的用户画像思想可迁移到推荐、对话等个性化场景

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] HORIZON: A Benchmark for in-the-wild User Behaviour Modeling](horizon_a_benchmark_for_in-the-wild_user_behaviour_modeling.md)
- [\[ACL 2026\] HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation](harpo_hierarchical_agentic_reasoning_for_user-aligned_conversational_recommendat.md)
- [\[ICLR 2026\] ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation](../../ICLR2026/recommender/propersim_developing_proactive_and_personalized_ai_assistants_through_user-assis.md)
- [\[NeurIPS 2025\] VisualLens: Personalization through Task-Agnostic Visual History](../../NeurIPS2025/recommender/visuallens_personalization_through_task-agnostic_visual_history.md)
- [\[ACL 2026\] Personalized Benchmarking: Evaluating LLMs by Individual Preferences](personalized_benchmarking_evaluating_llms_by_individual_preferences.md)

<!-- RELATED:END -->
