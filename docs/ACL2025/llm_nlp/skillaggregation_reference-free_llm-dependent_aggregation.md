---
title: >-
  [论文解读] SkillAggregation: Reference-free LLM-Dependent Aggregation
description: >-
  [ACL 2025][LLM/NLP][LLM评估] 本文提出SkillAggregation方法，通过学习上下文相关的LLM评判者技能权重并利用后验估计进行推理，在无需参考标签的情况下有效聚合多个LLM评判者的预测，在多个任务上超越了现有聚合方法。
tags:
  - ACL 2025
  - LLM/NLP
  - LLM评估
  - 多模型聚合
  - 无参考聚合
  - LLM-as-a-Judge
  - 众包标注
---

# SkillAggregation: Reference-free LLM-Dependent Aggregation

**会议**: ACL 2025  
**arXiv**: [2410.10215](https://arxiv.org/abs/2410.10215)  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: LLM评估, 多模型聚合, 无参考聚合, LLM-as-a-Judge, 众包标注

## 一句话总结

本文提出SkillAggregation方法，通过学习上下文相关的LLM评判者技能权重并利用后验估计进行推理，在无需参考标签的情况下有效聚合多个LLM评判者的预测，在多个任务上超越了现有聚合方法。

## 研究背景与动机

LLM作为评判者（LLM-as-a-judge）已成为NLP任务评估的重要替代方案，但单个LLM存在自偏好偏差、冗长偏差和提示敏感性等问题。使用多个LLM评判者可以改善性能，但关键在于如何有效地聚合多个评判者的判断。

现有聚合方法的局限：

**等权方法**（如多数投票、平均概率）忽略了评判者之间的能力差异，如GPT-4通常优于GPT-3，应被赋予更高权重

**任务特定方法**（如CrossCheckGPT）仅针对特定任务（如幻觉检测），无法泛化到其他评估场景

**约束条件过强**（如PRD）要求每个评判者评估所有其他评判者，限制了实用性

作者受众包标注聚合（crowdsourcing aggregation）的启发，将LLM评判者视为"工人"，提出了一种通用的、无需参考标签的上下文感知聚合方法。

## 方法详解

### 整体框架

SkillAggregation基于Crowdlayer方法的改进，包含三个核心组件：
1. 上下文编码器（预训练语言模型如GPT-2）将文本输入编码为向量表示
2. 瓶颈层将编码表示投影为2维的类分布估计
3. 可学习的技能估计向量捕捉每个LLM评判者的能力

### 关键设计

1. **技能估计向量（Skill-Estimate Vectors）**:

    - 每个LLM评判者k对应一对标量 p̂₀^(n,k) 和 p̂₁^(n,k)
    - p̂₀^(n,k) ≈ P(b_{n,k}=0|c_n=0, X_n)：给定真实标签为0和上下文时，评判者正确判断为0的概率
    - p̂₁^(n,k) ≈ P(b_{n,k}=1|c_n=1, X_n)：给定真实标签为1和上下文时，评判者正确判断为1的概率
    - 技能可以是任务特定的（所有样本共用一组参数）或上下文特定的（SkillAggregation-X，通过线性层+Sigmoid从上下文映射）

2. **正则化项**:

    - 分析发现预测可改写为 (p̂₀ + p̂₁ - 1)·s_{n,0} + (1 - p̂₁)，即LLM判断与真实标签之间的线性关系
    - 过度自信的LLM会导致斜率(p̂₀ + p̂₁ - 1)过大，放大其影响
    - 提出正则化 L_reg = Σ(p̂₀ + p̂₁ - 1)² 来惩罚过大的斜率
    - 总损失 L = L_CE + λ·L_reg

3. **后验估计推理（Posterior Estimation）**:

    - 相比Crowdlayer仅使用瓶颈层输出进行推理，SkillAggregation在推理时同时利用LLM判断结果
    - 假设LLM在给定真实标签和上下文时条件独立（CI假设）
    - 通过贝叶斯规则计算后验 P(c_n|X_n, b_n)，用学习到的技能估计向量和瓶颈输出近似真实技能和先验
    - 最终决策：比较后验比 r_n，当 r_n > 1 时判断为正类

### 损失函数 / 训练策略

- 训练目标：最小化预测的LLM判断与实际LLM判断之间的交叉熵损失 + 正则化项
- 上下文编码器：GPT-2 base（117M参数），使用最后一个隐藏状态作为上下文表示
- 完全无监督（reference-free）：在整个测试集上直接学习，无需参考标签
- 模型选择：用250个带标签的开发集样本选择超参数
- 训练时间：在单张NVIDIA RTX 6000 Ada上仅需20-30分钟

## 实验关键数据

### 主实验

| 方法 | HaluEval 7B(%) | TruthfulQA 7B(%) | Chatbot Arena 7B(%) |
|------|------|----------|------|
| 平均概率 | 76.28 | 68.06 | 63.24 |
| 多数投票 | 76.16 | 67.47 | 63.93 |
| DawidSkene | 76.78 | 67.84 | 64.71 |
| Train on MV | 78.78 | 67.32 | 63.77 |
| Crowdlayer | 79.27 | 67.74 | 64.06 |
| SkillAgg w/o Reg | 80.22 | 68.07 | 64.17 |
| **SkillAgg** | **80.83** | **68.74** | **64.22** |
| **SkillAgg-X** | **81.06** | **68.77** | **64.43** |

SkillAggregation-X在HaluEval上获得4.9%、TruthfulQA上1.3%、Chatbot Arena上0.5%的绝对准确率提升（相比多数投票）。

| 方法 | HaluEval ~70B(%) | TruthfulQA ~70B(%) | Chatbot Arena ~70B(%) |
|------|------|----------|------|
| 多数投票 | 80.81 | 83.63 | 70.61 |
| **SkillAgg-X** | **84.79** | **84.57** | **70.72** |

使用70B级别LLM评判者时，整体性能大幅提升，但聚合方法带来的增益减小。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 正则化效果 | 三个数据集均稳定提升 | 防止过度自信LLM主导后验估计 |
| 上下文编码器替换(RoBERTa/Gemma-2B) | 性能相似 | 方法对编码器选择不敏感 |
| 评判者子集分析 | 弱评判者时接近DawidSkene，强评判者时优势明显 | 需要足够好的评判者来学习有效的先验 |
| 数据集大小 | 1000样本时性能不稳定，5000样本时稳定 | 需要足够样本才能学好技能估计 |
| 去偏后 | 增益缩小 | 部分增益来自隐式去偏效果 |

### 关键发现

1. 差异化加权方法（DawidSkene, SkillAggregation）在所有任务上均优于等权方法（多数投票、平均概率）
2. 学习到的技能估计与LLM实际准确率高度相关（HaluEval上PCC=93.6%）
3. HaluEval上改进最大，因为上下文编码器本身已有一定的任务理解能力
4. Chatbot Arena上改进最小，因为人类偏好评估本身噪声更大
5. 正则化项在7B/8B模型上贡献显著，有效缓解了小模型的过度自信问题

## 亮点与洞察

1. **无参考学习**：仅从LLM判断本身学习聚合权重，无需人工标注数据，实际应用中极具价值
2. **后验估计推理**：相比Crowdlayer仅用先验预测，在推理时引入LLM判断进行后验更新的设计巧妙且有效
3. **正则化项的理论动机**：通过分析线性关系中斜率的含义，自然推导出正则化项的必要性
4. **从众包到LLM评估的迁移**：将成熟的众包标注理论（Dawid-Skene等）适配到LLM评估场景，方法论有启发性
5. **轻量级训练**：GPT-2 base + 20-30分钟训练即可完成，部署成本极低

## 局限与展望

1. 仅关注二分类任务，未扩展到回归或多分类评估场景
2. 条件独立假设可能不成立，多个LLM可能在相同样本上犯关联错误
3. 未考虑校准（calibration）性能，仅关注准确率
4. 开发集仍需250个标注样本，并非完全无监督
5. 未探索更强大的上下文编码器是否能进一步提升性能

## 相关工作与启发

本文将众包标注中的Worker Aggregation理论引入LLM评估领域，与PoLL（等权多评判者）、CrossCheckGPT（幻觉专用聚合）、PRD（排名聚合）等工作形成对比和互补。该方法可以扩展到任何需要聚合多个模型预测的场景，如模型集成、多智能体决策等。

## 评分

- 新颖性: ⭐⭐⭐⭐ 将Crowdlayer改造为SkillAggregation并引入后验推理和正则化，创新点清晰
- 实验充分度: ⭐⭐⭐⭐ 三个任务、多种评判者配置、多维消融分析，较为全面
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导严谨，分析深入，图表清晰
- 价值: ⭐⭐⭐⭐ 对LLM评估实践有直接指导意义，方法简单高效

<!-- RELATED:START -->

## 相关论文

- [Training-free LLM Merging for Multi-task Learning](training-free_llm_merging_for_multi-task_learning.md)
- [Probabilistic Aggregation and Targeted Embedding Optimization for Collective Moral Reasoning](probabilistic_aggregation_and_targeted_embedding_optimization_for_collective_mor.md)
- [A Training-free LLM-based Approach to General Chinese Character Error Correction](a_training-free_llm-based_approach_to_general_chinese_character_error_correction.md)
- [Cuckoo: An IE Free Rider Hatched by Massive Nutrition in LLM's Nest](cuckoo_an_ie_free_rider_hatched_by_massive_nutrition_in_llms_nest.md)
- [Leveraging Self-Attention for Input-Dependent Soft Prompting in LLMs](input_dependent_soft_prompting.md)

<!-- RELATED:END -->
