---
title: >-
  [论文解读] Improving Preference Extraction In LLMs By Identifying Latent Knowledge Through Classifying Probes
description: >-
  [ACL 2025][LLM/NLP][LLM-as-Judge] 本文提出使用线性分类探针（classifying probes）结合对比对（contrast pairs）来提取LLM的隐含偏好判断，在LLM-as-Judge任务中持续优于传统的生成式评估方法，且监督探针甚至超越微调评估器…
tags:
  - "ACL 2025"
  - "LLM/NLP"
  - "LLM-as-Judge"
  - "线性探针"
  - "对比对"
  - "隐含知识"
  - "偏好提取"
---

# Improving Preference Extraction In LLMs By Identifying Latent Knowledge Through Classifying Probes

**会议**: ACL 2025  
**arXiv**: [2503.17755](https://arxiv.org/abs/2503.17755)  
**代码**: 无（论文声明将公开）  
**领域**: LLM/NLP  
**关键词**: LLM-as-Judge, 线性探针, 对比对, 隐含知识, 偏好提取

## 一句话总结

本文提出使用线性分类探针（classifying probes）结合对比对（contrast pairs）来提取LLM的隐含偏好判断，在LLM-as-Judge任务中持续优于传统的生成式评估方法，且监督探针甚至超越微调评估器，同时保持类似的计算成本。

## 研究背景与动机

LLM-as-Judge是当前自动评估的主流范式，通过提示LLM输出数值评分或比较判断来替代人类评估。然而，这种基于生成的评估方式存在多种问题：
- 受限解码可能引入伪影
- 提示词可能引入无意识偏差
- 冗长的推理可能模糊核心判断
- 黑盒方法可能产生不可信或事实不正确的生成
- 预训练阶段学到的偏差可能影响判断

一个关键的经验观察是：LLM在其隐含空间中编码的"信念"或"判断"方向与线性方向相关——例如对Llama 3.1 70B在MT-Bench数据集上，对比对嵌入差异的第一主成分就能大致区分人类偏好的哪个模型更好。这意味着LLM的内部表征中包含比其生成输出更准确的判断信息。

## 方法详解

### 整体框架

方法的核心思想是：不通过让LLM"说出"判断来获取评估结果，而是直接从LLM的隐含表征（激活值）中提取判断信号。具体流程：
1. 构造对比对（contrast pairs）：对同一个评估实例，分别添加正面和负面的对比令牌
2. 获取两种条件下的嵌入向量
3. 通过嵌入差异消除共享特征，突出判断相关特征
4. 在差异向量上拟合分类器来提取偏好

### 关键设计

1. **对比对构造**: 对于每个待评估实例 $s_i$，构造一对提示 $(x_i^+, x_i^-)$。例如："Between Choice 1 and Choice 2, the more [task] [item] is Choice 1" 和 "...is Choice 2"。两个提示仅在最后的对比令牌上不同，从而控制了无关变量。

2. **嵌入差异分析**: 从LLM最后一层解码器块之后、最终归一化层之前提取对比令牌的嵌入向量 $\phi(x_i^+)$ 和 $\phi(x_i^-)$。对嵌入进行中心化去除句法差异 $\Delta_{syntax}$，使得剩余最显著的对比特征为知识/判断差异 $\Delta_{knowledge}$。

3. **无监督探针（PCA-based）**: 对中心化后的嵌入差异 $\{\tilde{\phi}(x_i^+) - \tilde{\phi}(x_i^-)\}$ 做PCA，取第一主成分作为分类方向。这种方法不需要任何标签，假设"知识差异"是最显著的对比特征。

4. **监督探针（逻辑回归）**: 给定标签后，使用逻辑回归拟合分类器：$\mathbb{P}(x^+ \text{ true}) = \sigma(\mathbf{w}^T(\tilde{\phi}(x_i^+) - \tilde{\phi}(x_i^-)))$。仅需训练一个线性层的参数。

5. **位置偏差校正的基线**: 对于生成式基线，通过交换两个选项的位置并平均预测概率来消除位置偏差，但这需要对每个问题运行两次模型。

### 损失函数 / 训练策略

- 监督探针：标准的逻辑回归损失，在5000个标注样本上训练
- 无监督探针：PCA，无需标签
- 对比微调基线：LoRA (r值未详细指定) 和全量微调
- 激活提取位置：最后解码器块之后、最终归一化层之前的最后令牌嵌入

## 实验关键数据

### 主实验

MT-Bench上的F1分数（与人类面板80%一致性的多数投票作为金标准）：

| 方法 | Llama 3.1 70B F1 | 说明 |
|------|-----------------|------|
| 成对比较提示 | ~0.65 | 生成式基线 |
| 无监督探针 | ~0.80 | 无需标签 |
| 监督探针 | ~0.80 | 需少量标签 |

跨模型家族的无监督探针 vs 提示（6个数据集的聚合结果）：

| 模型 | 无监督探针F1 | 提示F1 | 提升 |
|------|------------|--------|------|
| Gemma 2 27B | ~0.85 | ~0.70 | +15% |
| Llama 3.1 70B | ~0.88 | ~0.72 | +16% |
| Qwen 2.5 72B | ~0.87 | ~0.73 | +14% |
| Mistral Large 123B | ~0.84 | ~0.70 | +14% |

### 消融实验

| 配置 | 关键结果 | 说明 |
|------|---------|------|
| 监督探针 vs LoRA微调 | 探针更优 | 相同训练数据量下 |
| 监督探针 vs 全量微调 | 探针更优 | Gemma 2全系列模型大小 |
| 跨数据集泛化（无监督） | F1 0.70-0.99 | 在不同任务间迁移能力极强 |
| 跨数据集泛化（监督） | F1 0.58-0.99 | 监督探针泛化略弱但总体仍好 |
| LLMBar对抗鲁棒性 | 探针显著更鲁棒 | 对抗提示下性能下降幅度更小 |

### 关键发现

- 无监督探针在所有6个数据集和4个模型家族中（除一个例外：Qwen 2.5 0.5B）均优于校准后的提示方法
- 重要发现：小模型使用无监督探针的性能几乎总是优于大模型使用提示的性能，说明现有LLM-as-Judge实践中大模型的使用可能是"浪费的"
- 监督探针进一步优于无监督探针，且在相同训练数据量下优于LoRA和全量微调
- 主观任务（文本质量评估）中探针优势更大，客观任务（常识推理）中微调更具竞争力
- 无监督探针在不同数据集间的泛化能力极强（余弦相似度>0.7），表明它们捕获的是通用的"信念/判断"特征
- 在LLMBar对抗测试中，探针方法比提示方法在对抗提示下的性能下降更小，更鲁棒

## 亮点与洞察

1. **能力差距的揭示**：模型通过生成表达的能力与其隐含空间中实际编码的能力之间存在显著差距。探针方法直接从隐含空间提取判断，绕过了生成过程中的各种偏差和损失。
2. **计算效率**：每个样本仅需两次前向传播（与位置校正的提示方法相同），但准确度显著更高。监督探针的训练仅涉及逻辑回归，几乎无计算开销。
3. **尺度效率**：小模型+探针 > 大模型+提示，对资源受限的应用场景有重要实践价值。
4. **通用判断方向的发现**：无监督探针在跨任务泛化中的优异表现暗示，LLM中存在某种通用的"判断"或"信念"方向，这是一个值得深入研究的现象。

## 局限与展望

- 仅关注成对比较任务，对直接评分（Likert量表）等单标注任务的扩展需要多分类探针
- 监督探针的性能饱和点未被研究——在极大数据量下微调可能反超
- 对对抗提示（如"You are a smart professor"）仍有脆弱性，这是探索方法的根本局限
- 嵌入提取层的选择对结果有影响，默认使用最后一层但未系统优化
- 未考虑LLM的持续更新——新模型的指令跟随能力提升后，生成方法与探针的差距可能缩小
- 无监督探针的符号（sign）无法确定——PCA方向可能反转，实际使用时仍需少量标签校正

## 相关工作与启发

- Burns et al. (2023) 提出的对比一致搜索（CCS）是本文无监督探针的直接灵感来源
- 与Marks & Tegmark (2024) 关于LLM中事实知识的线性表征的工作一致，将探针从知识检测扩展到偏好判断
- 对G-Eval等基于logit加权的评估方法构成了范式上的挑战：直接从隐含空间提取信息比从输出概率中提取更有效
- 为AI安全和对齐研究提供了工具：如果模型"知道"正确答案但"不说"，探针可以用来检测这种不一致

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将表征探针方法应用于LLM-as-Judge是全新方向，洞察深刻
- 实验充分度: ⭐⭐⭐⭐⭐ 4个模型家族（0.5B-123B）、6个数据集、泛化/对抗测试极为全面
- 写作质量: ⭐⭐⭐⭐ 理论框架清晰，对比分析系统，但部分公式可更简洁
- 价值: ⭐⭐⭐⭐⭐ 揭示了LLM评估中的能力差距，方法实用且高效，对实践者有直接指导

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] GenKnowSub: Improving Modularity and Reusability of LLMs through General Knowledge Subtraction](genknowsub_improving_modularity_and_reusability_of_llms_through_general_knowledg.md)
- [\[ACL 2025\] Self-Tuning: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching](self-tuning_instructing_llms_to_effectively_acquire_new_knowledge_through_self-t.md)
- [\[ACL 2025\] Analyzing LLMs' Knowledge Boundary Cognition Across Languages Through the Lens of Internal Representations](knowledge_boundary_crosslingual.md)
- [\[ACL 2025\] Classifying Unreliable Narrators with Large Language Models](classifying_unreliable_narrators.md)
- [\[ACL 2025\] MHA2MLA: Towards Economical Inference by Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs](mha2mla_deepseek_latent_attention.md)

</div>

<!-- RELATED:END -->
