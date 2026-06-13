---
title: >-
  [论文解读] Basic Reading Distillation
description: >-
  [ACL 2025][模型压缩][知识蒸馏] 本文提出基础阅读蒸馏（BRD），通过让教师LLM在通用语料上生成基础阅读行为数据（包括NER和问答），训练小型学生模型模仿这些行为，使564M参数的小模型在不接触下游任务数据的情况下就能在多种NLP任务上达到或超过20倍大的教师模型性能。 现有的蒸馏方法主要分为两类：知识蒸馏（模…
tags:
  - "ACL 2025"
  - "模型压缩"
  - "知识蒸馏"
  - "基础阅读教育"
  - "命名实体识别"
  - "问答"
  - "小模型增强"
---

# Basic Reading Distillation

**会议**: ACL 2025  
**arXiv**: [2507.19741](https://arxiv.org/abs/2507.19741)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 知识蒸馏, 基础阅读教育, 命名实体识别, 问答, 小模型增强

## 一句话总结
本文提出基础阅读蒸馏（BRD），通过让教师LLM在通用语料上生成基础阅读行为数据（包括NER和问答），训练小型学生模型模仿这些行为，使564M参数的小模型在不接触下游任务数据的情况下就能在多种NLP任务上达到或超过20倍大的教师模型性能。

## 研究背景与动机
现有的蒸馏方法主要分为两类：知识蒸馏（模仿教师模型的隐式内部特征，如logits、hidden states、attention maps）和任务蒸馏（模仿教师模型在特定任务上的输出行为）。两类方法都忽略了一个基本问题：**学生模型缺少在通用文本上的基础阅读教育**。

本文类比人类教育：LLM应该先接受"高中/大学"级别的阅读教育，培养基本的文本理解能力，再去参加"考试"。而传统训练只是消耗token做next token prediction，然后直接参加测试。核心idea是：**受过基础阅读教育的学生模型比没有接受教育的学生模型更有效**。

BRD的两个优势：(1) 可以将任何通用文本扩展为基础阅读训练数据，突破任务蒸馏中数据规模和多样性不足的问题；(2) 避免了知识蒸馏中模仿隐式特征导致的可解释性缺乏。

## 方法详解

### 整体框架
两阶段流程：(1) 利用教师LLM（Vicuna-13B）对通用语料（CC-100）中的每个句子生成基础阅读行为数据；(2) 将生成的数据与原始语料混合，训练学生模型（XGLM-564M）。整个过程不涉及任何下游任务数据。

### 关键设计
1. **基础阅读行为定义**:

    - 两种核心能力：命名实体识别（NER）和问答（QRA）
    - NER：识别句子中的实体及其类型（人名、组织、地理实体等），教师模型还会生成额外的描述信息
    - QRA：对句子内容/结构/态度提出问题，并在原文中找到答案
    - 使用few-shot prompting让教师模型生成这些行为，prompt中包含任务描述、示例和输入句子

2. **训练数据构造**:

    - 以段落为单位组织训练：将句子按原始顺序排列，每个句子后跟随其NER或QRA注释
    - NER段落格式：$s_1$ \<sep\> NER($s_1$) \<sep\> $s_2$ \<sep\> NER($s_2$) ...
    - QRA段落格式：$s_1$ \<sep\> QRA($s_1$) \<sep\> $s_2$ \<sep\> QRA($s_2$) ...
    - 原始段落也保留用于混合训练，避免灾难性遗忘
    - 三种数据混合：$D_{ORI}$（原始段落）+ $D_{NER}$（NER段落）+ $D_{QRA}$（QRA段落）

3. **测试策略**:

    - 使用per-token平均log概率作为候选答案的评分函数
    - $\bar{P} = \frac{1}{n} \sum_{i=1}^{n} \log P_i(y_i | x_{\text{prompt}})$
    - 选择得分最高的候选答案作为最终预测
    - 这种方法适用于候选答案长度不同的任务

4. **BRD的正交性设计**:

    - BRD关注通用文本上的基础阅读能力，不涉及任何隐式特征或特定任务
    - 因此与知识蒸馏和任务蒸馏正交，可以叠加使用

### 损失函数 / 训练策略
标准自回归语言模型损失：$L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \log P(y_t | y_{<t})$

使用500万句CC-100语料，段落级训练，与原始语料混合。学生模型基于XGLM-564M初始化。

## 实验关键数据

### 主实验 - 无下游任务监督（盲测）

| 模型 | XNLI | RTE | CB | PAWS-X | BOOLQ | SST-2 | BIG-bench | Avg. |
|------|------|-----|-----|--------|-------|-------|-----------|------|
| Vicuna-13B(教师) | 59.1 | 78.3 | 71.4 | 62.9 | 84.3 | 81.5 | 35.6 | 67.6 |
| XGLM-7.5B | 36.6 | 50.9 | 60.7 | 56.8 | 57.2 | 69.5 | 34.3 | 52.3 |
| XGLM-564M | 35.5 | 46.2 | 53.6 | 51.3 | 51.2 | 63.9 | 34.0 | 48.0 |
| MiniLLM(KD SOTA) | 34.2 | 58.1 | 73.2 | 44.1 | 55.9 | 62.4 | 34.6 | 51.8 |
| **XGLM-BRD** | **36.2** | 53.8 | 58.9 | **56.7** | **61.0** | **78.1** | **34.8** | **54.2** |

### 有任务输入（松弛测试）

| 模型 | XNLI | RTE | CB | PAWS-X | BOOLQ | SST-2 | BIG-bench | Avg. |
|------|------|-----|-----|--------|-------|-------|-----------|------|
| TaskDistillation | 57.1 | 58.1 | 60.7 | 64.8 | 74.8 | 77.2 | 41.6 | 62.0 |
| **XGLM-BRD2** | **59.2** | **62.5** | **82.1** | 64.8 | **75.0** | **81.9** | **44.1** | **67.1** |
| Vicuna-13B | 59.1 | 78.3 | 71.4 | 62.9 | 84.3 | 81.5 | 35.6 | 67.6 |

### 消融实验

| 配置 | Avg. | 说明 |
|------|------|------|
| XGLM-BRD2 完整 | 70.9 | 基准(松弛测试) |
| -NER | 68.4 | 去掉NER，性能下降2.5% |
| -QRA | 67.8 | 去掉QRA，性能下降3.1%，QRA更重要 |
| 段落级训练 vs 句子级 | 57.5 vs 55.6 | 段落级训练优于句子级 |
| XGLM-564M-FURTHER | 47.1 | 仅用原始语料继续训练反而降低 |

### 正交性验证（BRD + 其他蒸馏方法）

| 方法 | 原始Avg. | +BRD后Avg. | 提升 |
|------|---------|-----------|------|
| SKD (GPT-2 120M) | 51.0 | 58.9 | +7.9 |
| MiniLLM (GPT-2 120M) | 52.6 | 57.8 | +5.2 |
| SKD (GPT-2 760M) | 48.7 | 56.7 | +8.0 |
| TaskDistillation (XGLM-564M) | 65.5 | 68.2 | +2.7 |

### 关键发现
- 564M的XGLM-BRD在盲测中接近甚至超过15倍大的XGLM-7.5B (54.2 vs 52.3)
- XGLM-BRD2在松弛测试中在某些任务上超过26倍大的Vicuna-13B教师模型
- 交叉熵分析证实BRD有效地将学生模型的概率分布推向教师模型（除PIQA外所有任务的交叉熵都降低）
- 仅使用原始语料继续训练（XGLM-564M-FURTHER）性能反而下降，说明关键在于阅读行为数据而非更多训练
- BRD与KD和TD正交：在所有组合中都带来显著提升
- QRA对阅读理解的贡献大于NER
- 性能随BRD数据量增加稳定提升，约100万段落后趋于饱和

## 亮点与洞察
- 教育学类比非常直观：模型应该先学会"阅读"再去做题，这个动机很有说服力
- 与知识蒸馏和任务蒸馏的正交性是一个很好的性质，使BRD可以作为通用增强手段
- 通过NER和QRA两种基本任务就能显著提升小模型在广泛下游任务上的性能，说明"基础阅读理解"确实是一种底层能力
- 段落级训练优于句子级训练的发现有道理——下游任务通常需要处理多句输入
- 使用CC-100这种预训练语料进行BRD，意味着数据来源充足且多样

## 局限与展望
- 仅使用Vicuna-13B作为教师（出于效率考虑），更强的教师（如GPT-4）可能效果更好但成本高
- 学生模型仅用XGLM-564M验证，缺少对其他架构（如LLaMA系列小模型）的验证
- NER和QRA是否是最优的阅读行为组合？可以探索更多行为类型（如摘要、情感分析、逻辑推理等）
- 推理时使用per-token概率评分有局限，无法用于开放式生成任务
- 研究idea：可以将BRD扩展为多轮对话式阅读（而非单轮QRA），可能学到更深层的理解能力
- 5M句子的数据规模可能对更大的模型不够，需研究BRD在不同规模模型上的缩放行为

## 相关工作与启发
- 与PICL（内在任务预训练）有关但不同：PICL需要37个下游任务来训练检索器，BRD完全不依赖下游任务
- 与自监督学习的对比：BRD可以看作一种教师指导的自监督增强方法
- 模仿人类阅读教育的思路可以扩展到其他模态，如视觉模型的"基础观察训练"

## 评分
- 新颖性: ⭐⭐⭐⭐ "基础阅读教育蒸馏"的理念新颖，但具体实现（NER+QRA）相对简单
- 实验充分度: ⭐⭐⭐⭐ 多种设置（盲测/松弛/有监督）、正交性验证、消融实验完备，但仅一个学生模型
- 写作质量: ⭐⭐⭐⭐ 教育类比使论文读起来很自然，逻辑清晰，但方法描述有些冗长
- 价值: ⭐⭐⭐⭐ 提供了一种低成本增强小模型的通用方法，正交性使其有广泛的适用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] BaSIC: BayesNet Structure Learning for Computational Scalable Neural Image Compression](../../ECCV2024/model_compression/basic_bayesnet_structure_learning_for_computational_scalable_neural_image_compre.md)
- [\[ACL 2025\] Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation](beyond_logits_aligning_feature_dynamics_for_effective_knowledge_distillation.md)
- [\[ACL 2025\] Data Laundering: Artificially Boosting Benchmark Results through Knowledge Distillation](data_laundering_artificially_boosting_benchmark_results_through_knowledge_distil.md)
- [\[ACL 2025\] Pre-training Distillation for Large Language Models: A Design Space Exploration](pre-training_distillation_for_large_language_models_a_design_space_exploration.md)
- [\[ACL 2025\] Flipping Knowledge Distillation: Leveraging Small Models' Expertise to Enhance LLMs in Text Matching](flipping_kd_small_to_large.md)

</div>

<!-- RELATED:END -->
