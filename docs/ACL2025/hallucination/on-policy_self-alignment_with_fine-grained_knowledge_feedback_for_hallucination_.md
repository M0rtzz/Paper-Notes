---
title: >-
  [论文解读] On-Policy Self-Alignment with Fine-grained Knowledge Feedback for Hallucination Mitigation
description: >-
  [ACL 2025][幻觉检测][幻觉缓解] 提出 RLFH（Reinforcement Learning for Hallucination），一种在策略（on-policy）自对齐方法，让 LLM 自己作为评判者，将回复分解为原子事实并进行真实性和信息量评估，生成 token 级别的密集奖励信号…
tags:
  - "ACL 2025"
  - "幻觉检测"
  - "幻觉缓解"
  - "强化学习"
  - "自对齐"
  - "细粒度反馈"
  - "在策略学习"
---

# On-Policy Self-Alignment with Fine-grained Knowledge Feedback for Hallucination Mitigation

**会议**: ACL 2025  
**arXiv**: [2406.12221](https://arxiv.org/abs/2406.12221)  
**代码**: [有 (GitHub)](https://github.com/AlignRM/RLFH)  
**领域**: 幻觉检测  
**关键词**: 幻觉缓解, 强化学习, 自对齐, 细粒度反馈, 在策略学习

## 一句话总结

提出 RLFH（Reinforcement Learning for Hallucination），一种在策略（on-policy）自对齐方法，让 LLM 自己作为评判者，将回复分解为原子事实并进行真实性和信息量评估，生成 token 级别的密集奖励信号，通过在线 PPO 优化来有效缓解幻觉问题。

## 研究背景与动机

大语言模型的幻觉问题（hallucination）是当前最关键的挑战之一。所谓幻觉，是模型生成偏离其知识边界的内容——可能是错误的事实信息、对超出知识范围的问题的鲁莽回答、或对本可回答问题的回避。

现有幻觉缓解方法面临三大困境：

**离策略采样（Off-policy sampling）**：现有学习方法使用其他模型或旧版本模型生成的数据进行训练，导致分布偏移——训练数据不反映当前模型的行为，优化效果打折扣

**粗粒度反馈**：现有方法通常对整个回复给一个评分（好/坏），但一个回复中可能同时包含正确和错误的事实，粗粒度反馈无法精确定位问题

**知识边界检测不准确**：现有方法通过显式 prompting 或内部状态探测来检测模型知识边界，但结果往往不一致

另一类方法（编辑式方法）先生成再用外部知识修正，但这只是修补输出而不改善模型内在的知识利用能力，而且外部知识源的覆盖范围有限。

RLFH 的核心思路是：**让模型自己探索自己的知识边界，通过细粒度的在策略反馈来自我矫正生成行为**。

## 方法详解

### 整体框架

RLFH 包含三个步骤的循环：

1. **生成回复**：当前策略模型 $\pi$ 对输入 prompt 生成回复
2. **自评估**：策略模型自身作为评判者，对回复进行细粒度评估
3. **在线强化学习**：将评估结果转化为 token 级密集奖励，用 PPO 更新策略

### 关键设计

#### 1. **层级化原子事实提取**

功能：将模型回复分解为可验证的最小事实单元
核心思路：两级分解——先将回复拆分为句子 $\{s_i\}_{i=1}^M$，再从每个句子中提取原子事实 $\{e_{ij}\}_{j=1}^{N_i}$
设计动机：
- 句子级拆分后再提取语句，能获得更细的粒度
- 句子-语句的层级结构便于后续将评估结果映射回原始 token 位置

#### 2. **事实验证（Truthfulness）**

策略模型自身从参考文档中检索相关上下文，对每个原子事实进行验证。分类为五个级别：
- Correct（正确，有证据支持）
- Hedged Correct（正确但有不确定性表达）
- Vague（真实性无法确定）
- Hedged Wrong（错误但有不确定性表达）
- Wrong（错误，与证据矛盾）

引入 "Vague" 类别来处理因参考文档不足而无法验证的语句。

#### 3. **信息量评估（Informativeness）**

对每个语句的信息量进行 1-5 分评估。与事实验证不同的是，信息量评估需要考虑原始问题 $x$ 和完整回复 $y$ 的上下文——因为信息量需要全局判断。

这一设计**防止模型走捷径**：如果只有真实性奖励，模型可能学会拒绝大部分问题或只给极简回答来规避错误。信息量奖励迫使模型在准确性和信息量之间寻找平衡。

### 损失函数 / 训练策略

#### Token 级密集奖励

**真实性奖励**：

$$r_{\text{truth}} = \alpha \cdot f(k_{\text{truth}}) \cdot |g(k_{\text{info}})|$$

- $f$ 把真实性标签映射为标量（正确→正，错误→负）
- $|g(k_{\text{info}})|$ 加权——更重要的语句获得更大的奖励/惩罚幅度（幻觉雪球效应：关键错误会引发连锁幻觉）

**信息量奖励**：

$$r_{\text{info}} = \beta \log(\mu + \max(\epsilon, \sum_i^N g(k_{\text{info}}^i)))$$

用对数函数使增长快速饱和但惩罚快速加大，防止模型过度追求信息量。

**映射到 token 位置**：使用最长公共子序列（LCS）算法将语句级评估映射回原始回复的 token 位置，实现 token 级密集奖励。

**PPO 优化**：使用标准 Proximal Policy Optimization 算法，以 token 级密集奖励进行在线强化学习。

## 实验关键数据

### 主实验（FactScore 评估）

| 模型 | 平均Score | HotpotQA | SQuADv2 | Biography |
|------|-----------|----------|---------|-----------|
| Llama3.1-8B（基线） | 0.639 | 0.653 | 0.777 | 0.487 |
| DOLA | 0.546 | 0.524 | 0.713 | 0.399 |
| ITI | 0.646 | 0.649 | 0.776 | 0.512 |
| FACT_DPO | 0.645 | 0.652 | 0.778 | 0.506 |
| FACT_SFT | 0.653 | 0.635 | 0.783 | 0.541 |
| **RLFH (Llama3.1-8B)** | **0.686** | **0.714** | 0.786 | **0.558** |
| Qwen2.5-7B（基线） | 0.638 | 0.634 | 0.813 | 0.467 |
| **RLFH (Qwen2.5-7B)** | **0.668** | 0.651 | **0.830** | **0.523** |

### 消融实验：奖励粒度影响

| 模型(Qwen2.5-7B) | 平均Score | HotpotQA | SQuADv2 | Biography |
|-------------------|-----------|----------|---------|-----------|
| 基线 | 0.638 | 0.634 | 0.813 | 0.467 |
| Response级 | 0.651 | 0.639 | 0.819 | 0.493 |
| Sentence级 | 0.655 | 0.637 | 0.821 | 0.506 |
| **Statement级** | **0.668** | **0.651** | **0.830** | **0.523** |

### 消融实验：Judge 模型影响

| Judge 模型 → Qwen2.5-7B | 平均Score |
|--------------------------|-----------|
| DeepSeekV2-Lite | 0.643 |
| Llama3.1-8B | 0.666 |
| Qwen2.5-7B（固定） | 0.668 |
| **On-Policy（自身）** | **0.668** |

### 关键发现

1. **RLFH 在所有数据集上取得最高 FactScore**：在 Llama3.1-8B 上平均 Score 从 0.639 提升到 0.686（+7.4%），在 Qwen2.5-7B 上从 0.638 提升到 0.668（+4.7%）
2. **跨数据集泛化**：仅在 HotpotQA 上训练，但在 SQuADv2 和 Biography 两个分布外数据集上也获得显著提升
3. **粒度越细越好**：Statement 级奖励一致优于 Sentence 级和 Response 级，验证了细粒度反馈的价值
4. **On-policy 自评优势**：让模型自己做评判者的效果不低于使用同等规模的外部模型，甚至在 Llama3.1-8B 上 on-policy 设定表现最优
5. **准确性-信息量权衡**：训练后模型回复率略有下降（更保守），但提供的信息更准确——高准确率回复比例大幅增加
6. **错误和不可验证内容显著减少**：分布分析显示 RLFH 有效压制了错误语句和模糊语句

## 亮点与洞察

- **"Policy as Judge" 范式**：让被优化的模型自身作为评判者，既消除了对外部奖励模型的依赖，又保证了评估与当前策略分布的一致性——这是一个优雅的设计
- **层级化事实分解 + LCS 映射**：将语言形式的语句级评估精确映射回 token 位置，实现了从自然语言反馈到数值奖励的无缝转换
- **幻觉雪球效应的考量**：在奖励设计中通过信息量加权真实性奖励，使关键语句的错误受到更大惩罚——这比简单的均匀奖励更贴近实际
- **信息量防止退化**：防止模型学会"不说话就不犯错"的退化策略

## 局限与展望

1. 主要针对事实性知识，对更广泛领域的幻觉（如推理幻觉）尚未验证
2. 现有评测基准范围有限，可能无法完全捕捉幻觉的复杂性
3. 自动事实验证本身可能存在错误，这些错误会影响训练信号质量
4. 模型自评可能存在"自我强化偏差"——模型可能同时生成错误并验证通过
5. 目前仅在 7-8B 模型上验证，更大规模模型上的效果有待观察

## 相关工作与启发

- FactScore (Min et al., 2023) 提供了语句级事实性评估的 pipeline
- RLHF (Ouyang et al., 2022) 是 LLM 对齐的基础框架
- DOLA (Chuang et al., 2023) 提供了通过层间对比来提升事实性的无训练方法
- ITI (Li et al., 2023) 通过推理时干预提升模型真实性
- 本文将细粒度评估和在线强化学习结合，提出了更系统的幻觉缓解方案

## 评分

- **新颖性**: ⭐⭐⭐⭐ — "自我评判 + 细粒度密集奖励 + 在线 RL" 的组合是新的，但各组件已有先例
- **实验充分度**: ⭐⭐⭐⭐⭐ — 3 个数据集、多个基线模型、粒度消融、Judge 模型消融、分布分析，非常全面
- **写作质量**: ⭐⭐⭐⭐ — 方法描述清晰，图表丰富，但公式符号较多
- **价值**: ⭐⭐⭐⭐⭐ — 幻觉缓解是当前最热门的研究方向之一，该方法的实用性和改进幅度都很可观

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Fine-grained Hallucination Detection and Mitigation in Long-form Question Answering](localizing_and_mitigating_errors_in_long-form_question_answering.md)
- [\[ACL 2025\] FIHA: Autonomous Fine-grained Hallucination Evaluation in Vision-Language Models with Davidson Scene Graphs](fiha_autonomous_hallucination_evaluation_in_vision-language_models_with_davidson.md)
- [\[ACL 2025\] Correcting Hallucinations in News Summaries: Exploration of Self-Correcting LLM Methods with External Knowledge](correcting_hallucinations_in_news_summaries_exploration_of_self-correcting_llm_m.md)
- [\[ACL 2025\] DRAG: Distilling RAG for SLMs from LLMs to Transfer Knowledge and Mitigate Hallucination](drag_distilling_rag_slm.md)
- [\[ACL 2025\] Alleviating Hallucinations from Knowledge Misalignment in Large Language Models via Selective Abstention Learning](alleviating_hallucinations_from_knowledge_misalignment_in_large_language_models_.md)

</div>

<!-- RELATED:END -->
