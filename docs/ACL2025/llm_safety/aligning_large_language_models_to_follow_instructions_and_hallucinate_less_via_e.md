---
title: >-
  [论文解读] Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filtering
description: >-
  [ACL 2025][幻觉缓解] 提出NOVA框架，通过内部一致性探测(ICP)衡量LLM对指令的熟悉度+语义等价识别(SEI)衡量LLM对目标回复的熟悉度，筛选出知识对齐的高质量指令数据，仅用5%数据微调LLaMA-3-8B即可在BioGEN上提升8.6分、FollowRAG上提升7.2分，同时保持指令遵循能力。
tags:
  - ACL 2025
  - 幻觉缓解
  - 数据筛选
  - 指令微调
  - 知识对齐
  - 内部一致性
---

# Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filtering

**会议**: ACL 2025  
**arXiv**: [2502.07340](https://arxiv.org/abs/2502.07340)  
**代码**: [GitHub](https://github.com/S1s-Z/NOVA)  
**领域**: LLM安全 / 指令微调  
**关键词**: 幻觉缓解, 数据筛选, 指令微调, 知识对齐, 内部一致性

## 一句话总结

提出NOVA框架，通过内部一致性探测(ICP)衡量LLM对指令的熟悉度+语义等价识别(SEI)衡量LLM对目标回复的熟悉度，筛选出知识对齐的高质量指令数据，仅用5%数据微调LLaMA-3-8B即可在BioGEN上提升8.6分、FollowRAG上提升7.2分，同时保持指令遵循能力。

## 研究背景与动机

**领域现状**：指令微调是LLM对齐的关键步骤。然而研究表明，在包含不熟悉知识的数据上微调会鼓励LLM产生过度自信和幻觉。

**现有痛点**：(a) RL-based方法(如FLAME-DPO)在指令微调后用偏好学习减少幻觉，但会削弱指令遵循能力，且需额外数据和API成本；(b) 现有数据筛选方法(IFD, CaR, Nuggets)仅关注质量，选出的高质量数据往往包含更多LLM不熟悉的专家级知识，反而加剧幻觉。

**核心矛盾**：高质量指令数据往往包含更深入的专家知识(correctness↑)，但这些知识可能是LLM在预训练中未学到的(familiarity↓)，导致幻觉加剧。

**本文目标**：在指令微调阶段，同时实现"遵循指令"和"减少幻觉"——筛选出既高质量又知识对齐的指令数据。

## 方法详解

### 整体框架

NOVA = ICP(衡量指令熟悉度) + SEI(衡量回复熟悉度) + Quality RM(保证数据质量)。最终rank = (familiarity_rank + quality_rank) / 2，选top-k%数据微调。

### 关键设计

1. **内部一致性探测 (ICP)**:
    - 功能：衡量LLM对给定指令 $q$ 的理解程度
    - 核心思路：对指令 $q$ 生成K个回复，提取每个回复最后一个token的内部状态作为句子embedding $E=[e_1,...,e_K]$。假设 $E \sim \mathcal{N}(\mu, \Sigma)$，计算微分熵：$F_{ins}(q) = \frac{1}{2}\sum_{i=1}^d \lambda_i + G$，其中 $\lambda_i$ 是协方差矩阵 $\Sigma$ 的特征值。熵低→回复一致→LLM熟悉该指令
    - 设计动机：相比困惑度或Rouge-L等表面指标，内部状态的微分熵能捕获更精细的语义一致性信息

2. **语义等价识别 (SEI)**:
    - 功能：衡量LLM对目标回复 $r$ 中知识的熟悉度
    - 核心思路：(1) 用NLI模型对K个生成回复做双向蕴含检测，将语义等价的回复聚类为 $[c_1,...,c_M]$；(2) 对每个聚类用投票策略判断目标回复 $r$ 属于哪个聚类；(3) $F_{res}(r) = k_{target}/\sum k_m$——目标聚类占总回复的比例越高，说明LLM越熟悉目标回复的内容
    - 设计动机：目标回复来自人工标注或GPT-4，LLM的内部状态无法有效表示这些外部输入，因此用NLI-based语义聚类+投票替代

3. **专家对齐质量奖励模型**:
    - 功能：用3751条专家标注偏好数据训练reward model，评估数据质量
    - 核心思路：最终分数 $R_{final}^{(i)} = \frac{1}{2}(R_{familiarity}^{(i)} + R_{quality}^{(i)})$，兼顾熟悉度和质量
    - 设计动机：仅考虑熟悉度(-w/o Quality RM)时，选出的数据虽大幅减少幻觉但严重降低指令遵循能力(MT-Bench从64.6降至48.6)

### 损失函数 / 训练策略

基于LLaMA-3-8B和LLaMA-3-70B，在Alpaca(52K)和Alpaca-GPT4上实验。选取top-5%/10%/15%数据做SFT。NLI模型使用DeBERTa-v3。

## 实验关键数据

### 主实验

LLaMA-3-8B, Alpaca-GPT4, 5%数据选择：

| 方法 | BioGEN(FactScore)↑ | LongFact-Obj↑ | FollowRAG-Avg↑ | MT-Bench↑ |
|------|-------------------|---------------|----------------|-----------|
| Vanilla-100% | 41.9 | 84.7 | 38.1 | 64.3 |
| IFD-5% | 46.7 | 84.4 | 42.3 | 65.0 |
| Nuggets-5% | 47.2 | 87.0 | 41.5 | 66.2 |
| FLAME-DPO | 46.3 | 87.3 | 41.5 | 56.2 |
| **NOVA-5%** | **50.5** | **90.1** | **45.3** | 64.6 |

NOVA相对Vanilla-100%的改进：BioGEN +8.6, LongFact +5.1, FollowRAG +7.2, MT-Bench +0.3。

### 消融实验

各组件贡献(LLaMA-3-8B, Alpaca-GPT4, 5%)：

| 配置 | BioGEN↑ | MT-Bench↑ |
|------|---------|-----------|
| NOVA完整 | 50.5 | 64.6 |
| -w/o ICP | 47.6 | 64.1 |
| -w/o SEI | 48.3 | 63.8 |
| -w/o Quality RM | 55.6 | **48.6** |
| -w/o ICP & SEI | 43.7 | 65.2 |

ICP替代方案比较：

| ICP替代 | BioGEN↑ | MT-Bench↑ |
|---------|---------|-----------|
| 内部状态(NOVA) | **50.5** | **64.6** |
| 困惑度 | 48.4 | 62.2 |
| Rouge-L | 47.9 | 61.5 |
| 外部Embedding模型 | 49.8 | 63.9 |

### 关键发现

1. **仅用5%数据即可超越100%全量数据训练**：在幻觉和指令遵循两个维度上
2. **RL-based方法(FLAME-DPO, SELF-EVAL)在降低幻觉的同时严重损害指令遵循**：MT-Bench分别降8.1和11.2
3. **纯质量筛选的数据可能加剧幻觉**：IFD在LongFact上反而增加了生成的facts数量(39.2 vs 32.0)
4. **Quality RM是维持指令遵循的关键**：去掉后BioGEN更高(55.6→50.5)但MT-Bench崩溃(48.6→64.6)
5. **内部状态比外部embedding更有效**：因为内部状态包含解码阶段可能丢失的细粒度信息
6. **可扩展到70B**：NOVA-5%-70B在BioGEN上达60.9(+7.2)

## 亮点与洞察

- **解决了一个fundamental trade-off**：在不引入额外RL阶段的情况下，通过数据筛选同时优化两个可能冲突的目标
- **ICP的创新性**：使用LLM内部状态的微分熵衡量一致性，比表面指标更能捕获语义细微差异
- **SEI的NLI+投票设计**：巧妙解决了"目标回复来自外部模型，LLM内部状态无法有效表示"的问题
- **Quality RM平衡器**：发现纯熟悉度筛选虽极大减少幻觉但损害指令能力，奖励模型作为平衡器的角色至关重要

## 局限与展望

- 需要为每条指令生成K个回复，增加离线数据筛选时间（但不影响推理）
- 仅适用于单轮指令数据，多轮对话场景未探索
- Quality RM需要3751条专家偏好数据训练，引入了额外数据需求
- NLI模型可能在长文本或专业领域上的语义等价判断不够准确

## 相关工作与启发

- **Gekhman et al. (2024)**：发现微调新知识鼓励幻觉的理论基础
- **FLAME (NeurIPS 2024)**：RL-based幻觉缓解，但损害指令能力
- 启发："数据选择"可能比"额外训练阶段(RL)"更高效地解决对齐问题，因为它从源头避免了问题

## 评分

- 新颖性: ⭐⭐⭐⭐ ICP和SEI的设计新颖，从知识对齐角度解决幻觉问题的视角独特
- 实验充分度: ⭐⭐⭐⭐⭐ 3个幻觉基准+2个指令基准+详尽消融+替代方法比较+人类评估
- 写作质量: ⭐⭐⭐⭐ 方法清晰，动机论证充分，但符号较多
- 价值: ⭐⭐⭐⭐⭐ 对LLM对齐研究有重要指导意义，方法简洁有效

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] ReLearn: Unlearning via Learning for Large Language Models](relearn_unlearning_via_learning_for_large_language_models.md)
- [\[ACL 2025\] Alleviating Hallucinations from Knowledge Misalignment in Large Language Models via Selective Abstention Learning](alleviating_hallucinations_from_knowledge_misalignment_in_large_language_models_.md)
- [\[ACL 2025\] Beyond Facts: Evaluating Intent Hallucination in Large Language Models](intent_hallucination_eval.md)
- [\[ACL 2025\] Chinese SimpleQA: A Chinese Factuality Evaluation for Large Language Models](chinese_simpleqa_a_chinese_factuality_evaluation_for_large_language_models.md)
- [\[ACL 2025\] Unveiling and Addressing Pseudo Forgetting in Large Language Models](unveiling_and_addressing_pseudo_forgetting_in_large_language_models.md)

<!-- RELATED:END -->
