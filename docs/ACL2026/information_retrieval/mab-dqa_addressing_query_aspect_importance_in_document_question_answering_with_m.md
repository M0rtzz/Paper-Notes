---
title: >-
  [论文解读] MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits
description: >-
  [ACL 2026][信息检索/RAG][文档问答] 提出 MAB-DQA 框架，将复杂查询分解为多个方面子查询，用多臂老虎机机制（Thompson Sampling）动态评估各方面的重要性并重新分配检索预算，显著提升多模态文档问答的检索精度和回答准确率。
tags:
  - "ACL 2026"
  - "信息检索/RAG"
  - "文档问答"
  - "多臂老虎机"
  - "查询分解"
  - "多模态RAG"
  - "超图推理"
---

# MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits

**会议**: ACL 2026  
**arXiv**: [2604.08952](https://arxiv.org/abs/2604.08952)  
**代码**: [GitHub](https://github.com/ElephantOH/MAB-DQA)  
**领域**: 文档问答与信息检索  
**关键词**: 文档问答、多臂老虎机、查询分解、多模态RAG、超图推理

## 一句话总结

提出 MAB-DQA 框架，将复杂查询分解为多个方面子查询，用多臂老虎机机制（Thompson Sampling）动态评估各方面的重要性并重新分配检索预算，显著提升多模态文档问答的检索精度和回答准确率。

## 研究背景与动机

- **领域现状**：文档问答（DQA）要求 AI 根据用户查询从文档中生成答案，是文档理解的核心任务。当前先进方法（如 ColPali、MoloRAG）采用视觉-查询延迟交互（Late Interaction）范式，计算查询 token 与文档图像 patch 的最大点积再求和作为相似度分数。
- **现有痛点**：Late Interaction 的"max-pooling + summation"操作对所有查询 token 赋予等权，无法像人类一样区分查询中不同方面的重要性。这导致低重要性但高频的关键词（如公司名"Best Buy"）会在无关页面上产生高虚假相似度，而真正包含关键证据的页面反而被排在后面。
- **核心矛盾**：多模态 RAG 通常只保留少数候选页面（如 Top-4），信息含量高但视觉显著性低的内容容易被忽略。作者统计发现 MMLongBench 中 19.8%、LongDocURL 中 27.8% 的样本存在因忽略关键查询条件导致的检索错误。
- **本文目标**：显式建模查询中多个隐式方面的不同重要性，动态分配检索注意力，优先检索包含关键信息的证据页面。
- **切入角度**：将每个子查询视为多臂老虎机的一个"臂"，利用 VLM 初步推理反馈作为奖励信号，通过探索-利用策略自适应地将检索预算分配给高价值方面。
- **核心 idea**：查询分解 + Thompson Sampling 驱动的动态检索预算分配 + 超图反思推理，三阶段递进式文档问答。

## 方法详解

### 整体框架

MAB-DQA 包含三个核心阶段：(1) 查询感知页面超图构建——将查询分解为方面子查询，结合文档页面构建超图；(2) 多臂老虎机引导检索——以 Thompson Sampling 动态选择高价值方面进行页面检索；(3) 超图反思推理代理——通过多阶段验证生成最终答案。

### 关键设计

1. **查询感知页面超图（Query-Aware Page Hypergraph）**：
    - 功能：将文档页面关系和查询的多方面结构统一建模
    - 核心思路：首先构建查询无关的页面图 G（基于页面间相似度），然后用 VLM 将原始查询 q 分解为 M 个方面子查询 {q₁,...,qₘ}。对每个子查询检索 Top-θ_H 页面形成候选集 Cⱼ，再筛选出在子查询下排名优于全局查询的页面构建超边 Ê_j。最终超图 H = (V_G, {Ê_j} ∪ E_G) 同时包含页面间边和方面超边
    - 设计动机：普通图无法表达一个子查询与多个页面的群组关系，超图的超边天然适合建模"一个方面关联一组页面"

2. **多臂老虎机引导检索（MAB-Guided Retrieval）**：
    - 功能：动态评估各方面重要性，将检索预算分配给高价值方面
    - 核心思路：每个子查询 Qⱼ 作为一个臂，维护 Beta(αⱼ, βⱼ) 分布。每轮通过 Thompson Sampling 采样选择臂，检索对应超边中的页面，VLM 评估页面相关性 s_vlm∈[0,1] 作为奖励更新 Beta 参数。页面综合评分 score(pᵢ) = (1-α)·max LI + α·s_vlm + β[(1-λ)·hᵢ + λ·s̄_cb]，其中 s̄_cb 是关联子查询的 Thompson Sampling 置信度均值
    - 设计动机：不同查询方面的信息价值差异大，固定权重无法捕捉这种动态变化；MAB 的探索-利用平衡天然适合此场景

3. **超图反思推理代理（Hypergraph-based Reflective Reasoning Agent, HRRA）**：
    - 功能：从检索到的证据页面中生成并验证最终答案
    - 核心思路：采用"初始回答-验证-优化"流水线，先用检索证据生成初始答案，若检测到不一致或证据缺口，重新进入超图构建查询聚焦子图进行反思循环
    - 设计动机：单次生成可能遗漏信息或产生幻觉，反思推理机制提供多阶段验证保障

### 损失函数 / 训练策略

- 该框架为推理时方法（inference-time），不涉及模型训练
- Beta 分布参数在线更新：(αⱼ, βⱼ) ← (αⱼ + s_vlm, βⱼ + 1 - s_vlm)
- 关键超参数：α=0.8（VLM 评估权重）、β=0.1（超参比例调节）、λ=0.75（页面度数 vs 臂置信度平衡）、θ_G=0.8（页面图边阈值）、θ_H=10（超边容量）、m=20（检索迭代数）

## 实验关键数据

### 主实验

| 方法 | MMLongBench | LongDocURL | FetaTab | PaperTab | 平均 |
|---|---|---|---|---|---|
| Qwen-2.5-VL-7B (Direct) | 0.204 | 0.398 | 0.350 | 0.112 | 0.266 |
| MDocAgent | 0.315 | 0.527 | 0.598 | 0.227 | 0.417 |
| MoloRAG+ | 0.372 | 0.528 | 0.600 | 0.195 | 0.424 |
| **MAB-DQA** | **0.399** | **0.564** | **0.638** | **0.269** | **0.468** |
| 提升 | +7.25% | +5.22% | +6.33% | +18.50% | +10.38% |

检索性能（Top-3，MMLongBench）：MAB-DQA 在 Recall (69.53)、Precision (34.32)、NDCG (41.05)、MRR (72.94) 上全面超越 MoloRAG 和 MoloRAG+。

### 消融实验

| 变体 | MMLongBench | LongDocURL | FetaTab | PaperTab | 平均提升 |
|---|---|---|---|---|---|
| Colpali (Baseline) | 0.296 | 0.554 | 0.537 | 0.152 | 0.0% |
| + MABR | 0.388 | 0.543 | 0.609 | 0.226 | +22.8% |
| + HRRA | 0.395 | 0.561 | 0.624 | 0.236 | +26.5% |
| MAB-DQA (Full) | 0.399 | 0.564 | 0.638 | 0.269 | +33.1% |

### 关键发现

- **查询方面重要性差异显著**：约 20-28% 的样本存在因忽略关键条件导致的检索错误，证明均匀加权的 Late Interaction 存在系统性缺陷
- **PaperTab 提升最大（+18.5%）**：涉及文档结构和表格理解的任务最受益于方面感知检索
- **MABR 和 HRRA 互补递进**：MABR 提供自适应检索聚焦关键方面 (+22.8%)，HRRA 进一步通过反思验证纠错 (+26.5%)，联合使用达到 +33.1%
- **跨 VLM 骨干通用**：在 Qwen2.5-VL-7B、LLaVa-13B、Qwen3-30B、Qwen3-32B 上均有一致提升

## 亮点与洞察

- **问题定义精准**：清晰地刻画了 Late Interaction 中"查询方面权重均匀"这一被忽视的问题，并用可视化热力图和统计数据 (Issue 列) 充分论证
- **MAB 建模巧妙**：将查询方面重要性估计转化为多臂老虎机问题，Thompson Sampling 的探索-利用平衡与检索场景天然契合
- **推理时方法，无需训练**：整个框架在推理时运行，不需要额外训练数据或微调，即插即用
- **超图建模页面-方面关系**：比普通图更能表达"一个方面关联一组页面"的群组结构

## 局限与展望

- 强依赖底层 VLM 的能力——若 VLM 在特定领域（法律、医学等）表现差，整体性能会受限
- 超参数较多（α, β, λ, θ_G, θ_H, m），目前通过网格搜索选择，作者计划未来引入贝叶斯优化自动调参
- 仅使用 Thompson Sampling，未对比 UCB、ε-Greedy 等其他 bandit 策略
- 检索时间复杂度 O(m·T_VLM)，VLM 调用次数随迭代轮数线性增长，大规模文档下效率可能成为瓶颈

## 相关工作与启发

- **ColPali (Faysse et al., 2024)**：视觉-语言嵌入模型，本文的检索骨干
- **MoloRAG (Wu et al., 2025)**：也使用图结构和 VLM 评估进行多模态 DQA，是最强基线但固定检索预算且不区分方面重要性
- **MBA-RAG (Tang et al., 2025)**：也用 MAB 但面向单模态，用 MAB 选择检索策略而非查询方面
- **GraphRAG (Edge et al., 2025)**：图增强 RAG 的代表性工作，MAB-DQA 进一步引入超图和动态预算分配
- 本文的方面感知检索思路可推广到通用多模态 RAG 场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 MAB 引入查询方面重要性建模是新颖的视角，超图构建也有独到之处
- 实验充分度: ⭐⭐⭐⭐ 4 个基准、多种基线对比、消融、敏感性分析、跨 VLM 验证，较为全面
- 写作质量: ⭐⭐⭐⭐ 图示清晰，问题动机论证充分，可视化案例分析有说服力
- 价值: ⭐⭐⭐⭐ 推理时方法、无需训练、即插即用，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] DQA: Diagnostic Question Answering for IT Support](dqa_diagnostic_question_answering_for_it_support.md)
- [\[ACL 2026\] Context Attribution with Multi-Armed Bandit Optimization](context_attribution_with_multi-armed_bandit_optimization.md)
- [\[ACL 2026\] FinRAG-12B: A Production-Validated Recipe for Grounded Question Answering in Banking](finrag-12b_a_production-validated_recipe_for_grounded_question_answering_in_bank.md)
- [\[ACL 2026\] ChatR1: Reinforcement Learning for Conversational Reasoning and Retrieval Augmented Question Answering](chatr1_reinforcement_learning_for_conversational_reasoning_and_retrieval_augment.md)
- [\[ACL 2026\] Prune-then-Merge: Towards Efficient Multi-Vector Visual Document Retrieval](sculpting_the_vector_space_towards_efficient_multi-vector_visual_document_retrie.md)

</div>

<!-- RELATED:END -->
