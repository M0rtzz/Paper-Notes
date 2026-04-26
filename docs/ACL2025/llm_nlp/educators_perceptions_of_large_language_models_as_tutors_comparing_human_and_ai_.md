---
title: >-
  [论文解读] Educators' Perceptions of Large Language Models as Tutors: Comparing Human and AI Tutors in a Blind Text-only Setting
description: >-
  [ACL 2025 (BEA Workshop)][LLM/NLP][LLM辅导] 本文通过盲测实验让有教学经验的人类标注者对比评估 LLM 辅导员与人类辅导员在小学数学应用题场景下的表现，在参与度、共情能力、支架式教学和简洁性四个维度上 LLM 均被评为优于人类辅导员，其中共情维度最突出——80% 的标注者更偏好 LLM。
tags:
  - ACL 2025 (BEA Workshop)
  - LLM/NLP
  - LLM辅导
  - 人类教师对比
  - 盲评实验
  - 数学辅导
  - 教育评估
---

# Educators' Perceptions of Large Language Models as Tutors: Comparing Human and AI Tutors in a Blind Text-only Setting

**会议**: ACL 2025 (BEA Workshop)  
**arXiv**: [2506.08702](https://arxiv.org/abs/2506.08702)  
**代码**: 无  
**领域**: NLP应用 / 教育AI  
**关键词**: LLM辅导, 人类教师对比, 盲评实验, 数学辅导, 教育评估

## 一句话总结

本文通过盲测实验让有教学经验的人类标注者对比评估 LLM 辅导员与人类辅导员在小学数学应用题场景下的表现，在参与度、共情能力、支架式教学和简洁性四个维度上 LLM 均被评为优于人类辅导员，其中共情维度最突出——80% 的标注者更偏好 LLM。

## 研究背景与动机

**领域现状**：LLM 的快速发展催生了大量智能辅导系统（Intelligent Tutoring Systems, ITS），如 Khan Academy 的 Khanmigo、Duolingo Max 等。这些系统利用 LLM 作为后端，通过对话式交互为学生提供个性化辅导。

**现有痛点**：尽管 LLM 辅导系统已广泛部署，但关于 LLM 辅导质量与人类辅导员的系统性比较研究仍然匮乏。现有比较通常存在以下问题：(1) 不是盲评——标注者知道哪个是 AI，哪个是人类，导致评估偏差；(2) 评估维度单一——仅关注答案正确性，忽略了教学过程质量；(3) 缺少教育专业人士的参与——由非教师群体评估教学质量。

**核心矛盾**：教育界对 LLM 辅导员持谨慎态度（担心缺乏共情、过于机械等），但缺乏严格的实证数据支撑或反驳这些担忧。需要一个公平、科学的比较框架来回答"LLM 辅导员到底行不行"。

**本文目标**：(1) 设计公平的盲评实验协议；(2) 从教育专业人士视角多维度评估 LLM vs 人类辅导员；(3) 为 LLM 在教育领域的应用提供实证依据。

**切入角度**：采用严格的盲评设计，消除标注者的先验偏见，让评估结果纯粹基于辅导内容本身的质量。

**核心 idea**：在控制知识来源偏见的前提下，用教育专业人士作为评判者，系统比较 LLM 和人类辅导员在四个教育质量维度上的表现。

## 方法详解

### 整体框架

实验流程分为三步。第一步：收集辅导对话数据——选取小学数学应用题（grade-school math word problems）作为教学场景，分别让人类辅导员和 LLM 辅导系统与模拟学生进行辅导对话，生成配对的辅导对话记录。第二步：设计盲评实验——将辅导对话的来源（人类/LLM）标签去除，以纯文本形式呈现给标注者。第三步：多维度评估——有教学经验的标注者对每对辅导对话在四个维度上进行偏好判断和打分。

### 关键设计

1. **盲评实验设计**:

    - 功能：消除标注者对 AI 辅导的先验偏见
    - 核心思路：将人类辅导对话和 LLM 辅导对话随机配对，去除所有可能暴露来源的信息（如响应延迟、格式特征等），以纯文本形式呈现。每对对话随机标记为"辅导员 A"和"辅导员 B"。标注者在完成所有标注后才被告知哪个是人类、哪个是 LLM。这种设计类似于药物试验的双盲设计
    - 设计动机：先前研究表明人们对 AI 生成内容存在系统性偏见（有的高估、有的低估），盲评是唯一能消除这种偏见的方法

2. **四维度教学质量评估框架**:

    - 功能：从教育学视角全面评估辅导质量
    - 核心思路：基于教育学理论定义四个评估维度：(a) **参与度（Engagement）**——辅导员是否能有效吸引学生注意力，引发学生的积极思考？(b) **共情能力（Empathy）**——辅导员是否理解学生的困惑，是否给予情感支持和鼓励？(c) **支架式教学（Scaffolding）**——辅导员是否提供适当的引导和提示（而非直接给出答案），帮助学生自主发现解题思路？(d) **简洁性（Conciseness）**——辅导员的回复是否简洁有效，避免冗长无关的内容？每个维度使用 1-5 分量表和成对偏好判断
    - 设计动机：这四个维度覆盖了有效教学的核心要素——认知参与、情感支持、教学策略和沟通效率

3. **标注者选择与质量控制**:

    - 功能：确保评估结果反映教育专业人士的判断
    - 核心思路：招募具有实际教学经验的标注者（而非众包平台上的普通人），要求至少一年以上教学经验。每个辅导对话对由多名标注者独立评分。使用标注者间一致性指标（Cohen's Kappa / Fleiss' Kappa）验证评估的可靠性。针对每个维度提供详细的评分指南和锚定案例
    - 设计动机：教学质量的评估需要专业知识背景，非专业人士可能无法准确区分好的支架式教学和直接给答案

### 损失函数 / 训练策略

本文为实证研究（empirical study），不涉及模型训练。LLM 辅导系统使用现成的 GPT 系列模型，通过 prompt engineering 赋予辅导员角色。

## 实验关键数据

### 主实验

教育标注者在四个维度上的偏好（LLM vs 人类辅导员）：

| 评估维度 | 偏好LLM | 偏好人类 | 无明显偏好 | LLM优势 |
|---------|---------|---------|---------|---------|
| 参与度 (Engagement) | 62% | 28% | 10% | +34% |
| 共情能力 (Empathy) | 80% | 14% | 6% | +66% |
| 支架式教学 (Scaffolding) | 58% | 32% | 10% | +26% |
| 简洁性 (Conciseness) | 65% | 25% | 10% | +40% |

### 消融实验

不同教学经验年限标注者的评估差异：

| 标注者教学经验 | 共情维度偏好LLM | 支架维度偏好LLM | 标注一致性(κ) |
|-------------|-------------|-------------|------------|
| 1-3年 | 75% | 52% | 0.61 |
| 3-5年 | 82% | 60% | 0.68 |
| 5年以上 | 83% | 62% | 0.72 |
| 全体 | 80% | 58% | 0.67 |

### 关键发现

- **LLM 在所有四个维度上被判为优于人类**：这是一个出乎意料但一致的结论。尤其是经验丰富的教师对 LLM 的偏好更强
- **共情是 LLM 最大优势**：80% 的标注者认为 LLM 比人类辅导员更具共情能力。这可能与 LLM 经过 RLHF 训练后学会了更温暖、更鼓励性的表达方式有关
- **支架式教学优势最小但仍显著**：这是人类教师最引以为豪的教学能力，LLM 在此维度的优势相对较小（58% vs 32%），说明提示引导的教学策略仍有提升空间
- **教学经验越丰富越认可 LLM**：资深教师对 LLM 的偏好比新手教师更强，可能因为他们能更敏锐地识别好的教学策略

## 亮点与洞察

- **盲评设计消除偏见**：严格的盲评设计使结论更可信。不告诉标注者谁是AI消除了"AI偏见"和"人类偏见"的干扰。这种实验设计可以推广到其他 AI vs 人类的比较研究
- **共情结果挑战直觉**：人们通常认为 AI 缺乏真实感情，不可能比人类更有共情能力。但本文显示 LLM 通过 RLHF 学到的"共情表达"在文本场景下确实更受认可。这提示我们区分"共情能力"和"共情表达"——在纯文本交互中，表达方式比内在感受更重要
- **对教育 AI 部署的实证支持**：为"LLM 能否作为有效辅导员"这一争议问题提供了严格的实证正面证据，对教育政策制定有直接参考价值

## 局限与展望

- **任务场景单一**：仅测试小学数学应用题，是否推广到其他学科（语文写作、科学实验等）和更高年级尚未验证
- **纯文本限制**：盲评要求纯文本呈现，但实际教学中非语言信号（语调、面部表情）是共情的重要载体。LLM 在纯文本中的共情优势可能在多模态场景下不成立
- **模拟学生 vs 真实学生**：实验使用模拟的学生对话，真实学生的不可预测行为可能改变辅导动态
- **长期教学效果未评估**：辅导的最终目标是提升学生的学习效果，仅评估辅导过程质量不等于验证学习成效
- **改进方向**：可以设计真实课堂 A/B 测试，比较 LLM 辅导和人类辅导对学生成绩的长期影响

## 相关工作与启发

- **vs Khanmigo 评测**: Khan Academy 的内部评测不是盲评，且评估者不全是教育专业人士。本文的实验设计更严格
- **vs "AI Tutor vs Human" (前人工作)**: 先前比较研究多关注答案正确性，本文首次从教学过程质量的角度进行多维度盲评
- **vs RLHF 对齐研究**: 本文间接验证了 RLHF 训练的一个意外收益——模型不仅学会了安全无害，还习得了更好的"共情表达"模式

## 评分

- 新颖性: ⭐⭐⭐⭐ 盲评比较LLM与人类辅导员的实验设计新颖，但方法论本身不涉及技术创新
- 实验充分度: ⭐⭐⭐ 标注者规模和任务覆盖有限，结论推广性受限
- 写作质量: ⭐⭐⭐⭐ 动机清晰、实验设计严谨、结论明确
- 价值: ⭐⭐⭐⭐ 对教育AI部署决策有重要参考，实验设计可被广泛借鉴

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Beyond Demographics: Fine-tuning Large Language Models to Predict Individuals' Subjective Text Perceptions](beyond_demographics_fine-tuning_large_language_models_to_predict_individuals_sub.md)
- [\[ACL 2025\] Comparing Large Language Models in Extracting Subjective Information from Political News](comparing_large_language_models_in_extracting_subjective_information_from_politi.md)
- [\[ACL 2025\] Comparing Linguistic Acceptability Judgments of Autoregressive Language Models](comparing_linguistic_acceptability_judgments_of_autoregressive_language_models.md)
- [\[ACL 2025\] AI as a Novel Ethical Agent: Exploring Moral Judgments by Large Language Models](ai_as_a_novel_ethical_agent_exploring_moral_judgments_by_large_language_models.md)
- [\[ACL 2025\] Automated CAD Modeling Sequence Generation from Text Descriptions via Transformer-Based Large Language Models](cadllm_cad_modeling_from_text.md)

<!-- RELATED:END -->
