---
title: >-
  [论文解读] Semantic Reranking at Inference Time for Hard Examples in Rhetorical Role Labeling
description: >-
  [ACL 2026][NLP理解][修辞角色标注] 提出 RiSE，一种推理时语义重排序框架，通过自动识别低置信度困难样本并利用对比学习的标签语义表示重排序模型输出，在 8 个修辞角色标注数据集上困难样本平均提升 +9.15 macro-F1。
tags:
  - "ACL 2026"
  - "NLP理解"
  - "修辞角色标注"
  - "推理时重排序"
  - "标签语义"
  - "对比学习"
  - "困难样本"
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Semantic Reranking at Inference Time for Hard Examples in Rhetorical Role Labeling

**会议**: ACL 2026  
**arXiv**: [2605.18007](https://arxiv.org/abs/2605.18007)  
**代码**: [GitHub](https://github.com/AnasBelfathi/rise-framework)  
**领域**: medical_imaging  
**关键词**: 修辞角色标注, 推理时重排序, 标签语义, 对比学习, 困难样本

## 一句话总结
提出 RiSE，一种推理时语义重排序框架，通过自动识别低置信度困难样本并利用对比学习的标签语义表示重排序模型输出，在 8 个修辞角色标注数据集上困难样本平均提升 +9.15 macro-F1。

## 研究背景与动机
**领域现状**: 修辞角色标注（RRL）为文档中每个句子分配功能角色，广泛应用于法律、医学和科学领域。语言模型在平均性能上表现良好，但在困难样本上仍不可靠。

**现有痛点**: (1) 现有方法将标签视为离散标识符，忽略标签名称中编码的语义信息；(2) 困难样本（低置信度预测）的处理通常是隐式的，缺乏专门机制；(3) 语义近似的标签之间容易混淆，Top-1 vs Top-3 的 macro-F1 差距表明正确标签常在高排名但未被选中。

**核心矛盾**: 标准分类器使用独热向量表示标签，无法利用标签之间的语义关系；而纯相似度方法虽利用标签语义但丢失了分类器的判别能力。

**本文目标**: 设计一种推理时方法，在保持分类器判别行为的同时利用标签语义改善困难样本预测。

**切入角度**: 推理时介入——自动检测低置信度样本，用对比学习的文本-标签语义相似度重新加权分类器 logits。

**核心 idea**: 困惑加权对比学习 + 自适应困难样本检测 + logits 语义重排序 = 无需重训练的推理时困难样本修复。

## 方法详解

### 整体框架
RiSE 在推理时操作：首先用底层分类器对输入句子产生 logits；然后基于 logit 方差自动识别困难样本；对困难样本，利用对比学习的文本-标签表示计算语义相似度，通过逐元素乘法重排序 logits。非困难样本直接使用原始输出。

### 关键设计
1. **自适应困难样本检测**:

    - 功能：基于模型置信度自动识别预测不确定的样本
    - 核心思路：计算 logit 向量的方差作为置信度指标——低方差意味着多个标签得分接近（强标签竞争）。自适应阈值定义为验证集上误分类样本的平均 logit 方差 $\sigma^2_{\text{mis}} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \text{Var}(\mathbf{z}_i)$
    - 设计动机：固定阈值不适用于不同模型和数据集组合；使用方差而非熵，因方差直接捕捉原始决策空间的得分分散度，对校准不敏感

2. **困惑加权对比学习（标签语义学习）**:

    - 功能：学习编码分类器混淆模式的文本-标签共享嵌入空间
    - 核心思路：使用分类器在验证集上的预测行为构建标签亲和度权重 $w_{y'} = P(y, y')$（归一化的标签混淆概率）；在加权 InfoNCE 损失 $\mathcal{L}_{\text{CW}}$ 中，高混淆对的负样本获得更大权重，迫使模型更努力地区分语义相近的标签
    - 设计动机：标签间的混淆模式是领域特定的（如法律中"分析"和"论证"容易混淆），使用混淆概率加权比均匀加权更有效

3. **推理时语义重排序**:

    - 功能：将标签语义相似度融入分类器输出以修正困难样本预测
    - 核心思路：对困难样本 $x$，计算输入嵌入 $\mathbf{e}_x$ 与各标签嵌入 $\mathbf{e}_y$ 的余弦相似度向量 $\mathbf{s}_x \in \mathbb{R}^C$，通过逐元素乘法重排序：$\tilde{\mathbf{z}}_x = \mathbf{s}_x \odot \mathbf{z}_x$
    - 设计动机：仅对困难样本干预保留了分类器在简单样本上的判别能力，逐元素乘法优雅地融合了判别信号和语义信号

## 实验关键数据

### 主实验（7 个 LM × 8 个数据集，macro-F1 / weighted-F1 均值）

| 模型 | Baseline mF1 | + RiSE mF1 | Baseline wF1 | + RiSE wF1 |
|------|-------------|-----------|-------------|-----------|
| LLaMA-3-8B | 67.98 | **68.51†** | 74.88 | **75.65†** |
| Mistral-7B | 67.66 | **69.50†** | 74.61 | **75.75†** |
| Qwen3-8B | 67.59 | **68.82†** | 74.28 | **75.23†** |
| ALBERT-base | 66.45 | **66.99** | 72.98 | **73.39** |
| BERT-base | 66.17 | **67.25†** | 73.49 | **73.86** |
| DeBERTa-base | 68.02 | **68.50** | 74.44 | **74.86** |
| RoBERTa-base | 67.21 | **68.11†** | 74.24 | **74.77** |

### 困难样本提升（平均 across 模型和数据集）

| 指标 | 提升幅度 |
|------|---------|
| 困难样本 macro-F1 | **+9.15** |
| 全集 macro-F1 | +0.5 ~ +1.8 |

### 关键发现
- RiSE 在困难样本上平均提升 +9.15 macro-F1，同时保持或略微提升全集性能
- 在 7 个模型（含 encoder 和 causal 架构）和 8 个数据集（法律/医学/科学）上一致有效，泛化性强
- 人工困难度标注与模型困难度的 Cohen's κ=0.40（中等一致），说明模型困难和人类困难部分重叠但不完全相同
- causal LM（LLaMA-3/Mistral/Qwen3）的绝对提升略大于 encoder 模型，可能因 causal LM 更容易产生标签混淆

## 亮点与洞察
- 推理时零成本介入：不修改模型、不重训练、不改架构，即插即用的通用框架
- 困惑加权对比学习巧妙利用分类器自身的错误模式来指导标签语义学习——从失败中学习
- 方差作为困难度指标简洁有效，优于熵等替代方案
- 人工困难度标注的引入为理解模型行为提供了可解释性视角

## 局限与展望
- 标签语义嵌入器需要在每个数据集上用训练数据训练对比模型，虽然轻量但并非完全零开销
- 方差阈值依赖验证集上的误分类样本统计，在极端类不平衡时可能不稳定
- 逐元素乘法的融合方式较为简单，可能丢失更复杂的信号交互
- 仅在句子级分类任务上验证，是否适用于 token 级或更细粒度的标注任务待探索

## 相关工作与启发
- HiCuLR 在训练时通过课程学习处理困难样本，RiSE 在推理时处理——两者互补
- 标签语义在零样本/少样本文本分类中已被广泛使用，RiSE 将其应用于推理时困难样本修复是新角度
- 对比学习 + 混淆矩阵加权的思路可推广到其他多类分类任务的后处理改进
- 推理时干预的范式（如 self-consistency、重排序等）在 LLM 时代日益重要

## 评分
- 新颖性: ⭐⭐⭐⭐ 推理时利用混淆加权标签语义重排序是清晰的新贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 7 个模型 × 8 个数据集 × 3 个领域 + 人工标注分析
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法动机和设计逻辑连贯
- 价值: ⭐⭐⭐⭐ 即插即用的通用框架，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] It's High Time: A Survey of Temporal Question Answering](it39s_high_time_a_survey_of_temporal_question_answering.md)
- [\[ACL 2026\] Filling the Gap: Is Commonsense Knowledge Generation useful for Natural Language Inference?](filling_the_gap_is_commonsense_knowledge_generation_useful_for_natural_language_.md)
- [\[ACL 2026\] Test-Time Reasoners Are Strategic Multiple-Choice Test-Takers](test-time_reasoners_are_strategic_multiple-choice_test-takers.md)
- [\[ACL 2026\] Accurate and Efficient Statistical Testing for Word Semantic Breadth](accurate_and_efficient_statistical_testing_for_word_semantic_breadth.md)
- [\[ACL 2026\] LLM-Guided Semantic Bootstrapping for Interpretable Text Classification with Tsetlin Machines](llm-guided_semantic_bootstrapping_for_interpretable_text_classification_with_tse.md)

</div>

<!-- RELATED:END -->
