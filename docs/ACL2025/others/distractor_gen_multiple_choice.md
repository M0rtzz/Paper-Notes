---
title: >-
  [论文解读] Generating Plausible Distractors for Multiple-Choice Questions via Student Choice Prediction
description: >-
  [ACL 2025][干扰项生成] 本文提出了一个通过成对排序器预测学生选择倾向、再利用DPO训练干扰项生成器的三步流水线，使生成的多选题干扰项更具有迷惑性和区分度。
tags:
  - ACL 2025
  - 干扰项生成
  - 多选题
  - DPO
  - 学生选择预测
  - 教育评估
---

# Generating Plausible Distractors for Multiple-Choice Questions via Student Choice Prediction

**会议**: ACL 2025  
**arXiv**: [2501.13125](https://arxiv.org/abs/2501.13125)  
**代码**: [GitHub](https://github.com/holi-lab/distractor-generator) (有)  
**领域**: 教育NLP / 自动出题  
**关键词**: 干扰项生成, 多选题, DPO, 学生选择预测, 教育评估

## 一句话总结

本文提出了一个通过成对排序器预测学生选择倾向、再利用DPO训练干扰项生成器的三步流水线，使生成的多选题干扰项更具有迷惑性和区分度。

## 研究背景与动机

**领域现状**: 多选题（MCQ）是教育中重要的评估工具，其中干扰项（错误选项）的质量直接决定了测试的有效性。自动化干扰项生成已成为研究热点，但现有方法主要关注生成与人工编写相似的干扰项，忽视了提升迷惑性。

**现有痛点**: 先前工作生成的干扰项往往过于简单，学生一眼即可排除，无法有效评估学生的真实理解水平，降低了MCQ的教育价值和区分度。

**核心矛盾**: 生成可信的干扰项需要理解学生常见的误解和知识盲区，这种"学生心智模型"难以直接编码到生成模型中。

**本文目标**: 训练一个能生成高迷惑性干扰项的模型，使其产出的干扰项更可能被学生选择，从而提高MCQ的区分指数（DI）。

**切入角度**: 利用真实学生答题数据中的选择率信息，先训练一个能推断学生误解并排序干扰项可信度的模型，再用排序结果通过DPO训练生成器。

**核心 idea**: 用成对排序器学习学生的选择偏好模式，构造偏好数据集驱动DPO训练，使干扰项生成器能"投其所误"地生成更具迷惑性的选项。

## 方法详解

### 整体框架

三步训练流水线：Step 1 训练成对排序器（预测哪个干扰项更可能被选）→ Step 2 构建学生选择数据集（对干扰项排序）→ Step 3 用DPO训练干扰项生成器。

### 关键设计

1. **成对排序器（Pairwise Ranker）**
    - 功能：给定题目、正确答案和两个干扰项，判断哪个更可能被学生选择
    - 核心思路：模型先生成推理过程（分析学生可能的误解），再输出选择结果。用GPT-4o生成推理训练数据进行SFT，再用DPO纠正推理错误
    - 设计动机：通过结构化推理增强可解释性和准确性，推理过程揭示了学生误解的具体原因

2. **学生选择数据集构建**
    - 功能：为每道题目的所有干扰项建立可信度排序
    - 核心思路：用GPT-4o为每题生成新干扰项，与原始干扰项一起用排序器评分排序。原始干扰项间保留基于真实选择率的排序
    - 设计动机：扩充干扰项候选池并建立偏好对，为DPO提供chosen/rejected样本

3. **干扰项生成器训练（SFT + DPO）**
    - 功能：生成高迷惑性的干扰项
    - 核心思路：SFT阶段学习基本生成能力（包括先判断题目类型），DPO阶段用top-n vs bottom-n干扰项构建偏好对，引导模型偏向生成更可信的干扰项
    - 设计动机：题目类型判断（正确/错误陈述题）对生成有效性至关重要，DPO比SFT更好地优化了迷惑性

### 损失函数 / 训练策略

- 排序器和生成器均基于Mistral-7B-Instruct-v0.2，使用LoRA微调
- 排序器：SFT + DPO两阶段训练，DPO的rejected样本来自SFT模型的错误推理
- 生成器：SFT + DPO，DPO使用排名前n和后n的干扰项构建偏好对
- 为解决位置偏差，排序时交换AB顺序多次推理直至一致

## 实验关键数据

### 主实验

| 排序器 | Python准确率 | DB准确率 | MLDL准确率 | 平均准确率 |
|--------|------------|---------|-----------|----------|
| GPT-3.5 (Reasoning) | 0.633 | 0.523 | 0.606 | 0.587 |
| GPT-4o (Reasoning) | 0.686 | 0.664 | 0.570 | 0.640 |
| Ours (DPO, Comb.) | **0.712** | **0.659** | 0.655 | **0.675** |
| 人类专家 | - | - | - | 0.717 |

### 消融实验

| 消融条件 | 平均排序准确率 |
|---------|-------------|
| Ours (SFT, Sep.) | 0.587 |
| Ours (SFT, Comb.) | 0.657 |
| Ours (DPO, Comb.) | 0.675 |
| Ours (SFT w/o Reasoning) | 0.567 |

### 关键发现

- DPO排序器（67.5%）超越GPT-4o（64.0%），接近人类专家水平（71.7%），尽管训练数据来自GPT-4o的推理——这体现了"学生超越老师"的现象
- 去掉推理过程的排序准确率从67.5%大幅下降到56.7%，说明结构化推理对任务至关重要
- 人类评估中，DPO生成的干扰项区分指数（DI）最高，证实了高迷惑性干扰项确实有助于区分不同水平的学生
- 学生误解的主要因素：代码题中"对函数输出/操作的错误假设"最常见，概念题中"与相似术语的概念混淆"最常见

## 亮点与洞察

- **闭环设计精巧**：真实选择率→排序器→合成偏好数据→DPO生成器，每一步都紧密衔接且有数据依据
- **可解释性强**：排序器的推理过程揭示了学生常见误解的类型和分布，对教学有直接指导价值
- **超越GPT-4o的小模型**：7B的Mistral通过DPO后在排序准确率上超越了GPT-4o，展示了专门化训练的威力

## 局限与展望

- 仅在CS领域（Python/DB/MLDL）验证，其他学科（历史、语言等）的泛化性未知
- 依赖于真实学生选择率数据，冷启动场景（新题目无历史数据）无法直接应用
- 对"无效干扰项"（与正确答案矛盾或格式错误）的过滤主要手工处理，可自动化
- 未探讨生成的干扰项对实际教学效果（学生学习进步）的长期影响

## 相关工作与启发

- **Scarlatos et al. (2024)**: 提出了数学题的overgenerate-and-rank范式，但排序器不含推理，本文的推理式排序更通用且准确
- **DPO (Rafailov et al. 2024)**: 从RLHF简化而来的偏好优化方法，本文创新地将其应用于教育场景的干扰项生成
- 启发：这种"先学排序偏好、再用偏好优化生成"的范式可以推广到其他需要人类校准的生成任务

## 评分

- **新颖性**: ⭐⭐⭐⭐（成对排序+推理+DPO组合在教育场景的应用有创意）
- **实验充分度**: ⭐⭐⭐⭐（自动指标+人类评估+真实学生测试，评估全面）
- **写作质量**: ⭐⭐⭐⭐（流程图清晰，动机阐述充分）
- **价值**: ⭐⭐⭐⭐（实际教育场景有直接应用价值）

<!-- RELATED:START -->

## 相关论文

- [The Publication Choice Problem](../../AAAI2026/others/the_publication_choice_problem.md)
- [Knowledge Tracing in Programming Education Integrating Students' Questions](knowledge_tracing_in_programming_education_integrating_students_questions.md)
- [Optimizer Choice Matters for the Emergence of Neural Collapse](../../ICLR2026/others/optimizer_choice_matters_for_the_emergence_of_neural_collapse.md)
- [Infogen: Generating Complex Statistical Infographics from Documents](infogen_generating_complex_statistical_infographics_from_documents.md)
- [Cautious Next Token Prediction](cautious_next_token_prediction.md)

<!-- RELATED:END -->
