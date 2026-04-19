---
title: >-
  [论文解读] LOTUS: A Leaderboard for Detailed Image Captioning from Quality to Societal Bias and User Preferences
description: >-
  [ACL 2025][图像描述评估] 提出 LOTUS 排行榜，从描述质量（对齐性、描述性、语言复杂度）、副作用（幻觉、有害性）和社会偏见（性别、肤色）三个维度统一评估大型视觉语言模型的详细图像描述能力，并支持基于用户偏好的定制化评估。
tags:
  - ACL 2025
  - 图像描述评估
  - 大型视觉语言模型
  - 社会偏见
  - 用户偏好
  - 评估基准
---

# LOTUS: A Leaderboard for Detailed Image Captioning from Quality to Societal Bias and User Preferences

**会议**: ACL 2025  
**arXiv**: [2507.19362](https://arxiv.org/abs/2507.19362)  
**代码**: https://huggingface.co/spaces/nvidia/lotus-vlm-bias-leaderboard (有，Leaderboard)  
**领域**: recommender  
**关键词**: 图像描述评估, 大型视觉语言模型, 社会偏见, 用户偏好, 评估基准

## 一句话总结

提出 LOTUS 排行榜，从描述质量（对齐性、描述性、语言复杂度）、副作用（幻觉、有害性）和社会偏见（性别、肤色）三个维度统一评估大型视觉语言模型的详细图像描述能力，并支持基于用户偏好的定制化评估。

## 研究背景与动机

**领域现状**：随着 LLaVA 等大型视觉语言模型（LVLM）的出现，图像描述从生成简洁标题转向生成详细描述。这一转变增强了视觉语义理解，但也对评估方法提出了新挑战。

**现有痛点**：
   - **缺乏统一评估框架**：现有研究分别针对描述性、对齐性、幻觉检测等单一维度评估，缺乏标准化的综合框架，导致跨研究比较困难
   - **缺乏偏见感知评估**：尽管研究表明 LVLM 常展现社会偏见（如性别偏见），现有评估方法基本忽略了这些偏见
   - **评估不考虑用户偏好**：详细描述的质量高度主观，有些用户看重描述丰富性，有些看重最小化幻觉风险

**核心矛盾**：如何在一个统一框架中同时评估描述质量、副作用风险和社会公平性，并支持不同用户的差异化需求？

**本文目标**：建立一个综合的、偏见感知的、用户偏好敏感的详细图像描述评估排行榜。

**切入角度**：统一多个已有评估维度（对齐性、描述性、复杂度、副作用），并创新性地加入社会偏见评估和用户偏好导向的评估机制。

**核心 idea**：用统一排行榜整合质量+风险+偏见三大维度，揭示描述越详细偏见风险越大的 trade-off，并让用户按需选择最佳模型。

## 方法详解

### 整体框架

LOTUS 评估框架包含四个主要维度：
1. **对齐性（Alignment）**：描述与图像内容的匹配程度
2. **描述性（Descriptiveness）**：描述的详细程度
3. **语言复杂度（Language Complexity）**：句子结构的复杂程度
4. **副作用（Side Effects）**：幻觉和有害内容

此外还有两个偏见维度：
5. **性别偏见**
6. **肤色偏见**

### 关键设计

#### 1. 统一综合评估

- **对齐性指标**：
    - CLIPScore：$\text{CLIPScore} = \max(0, \cos(\phi_I(I), \phi_T(y')))$，衡量图像-描述的语义相似度
    - CapScore：GPT-4 评分，包括相似度（CapScore$_S$）和对齐度（CapScore$_A$），范围 0-1

- **描述性指标**：
    - CLIP Recall@$k$：描述是否能唯一标识对应图像
    - 名词/动词覆盖率：$\text{Noun Coverage} = \frac{|N(y) \cap N(y')|}{|N(y')|}$，衡量关键对象和动作的覆盖

- **语言复杂度**：
    - 句法复杂度：依存树的最大深度
    - 语义复杂度：场景图中的节点数量

- **副作用**：
    - CHAIRs：$\text{CHAIR}_s = \frac{O_H}{O_T}$，幻觉对象占比
    - FaithScore：$\text{FaithScore} = \frac{1}{K}\sum_{k=1}^K V(f_k, I)$，基于原子事实的忠实度
    - NSFW 词汇检测

- **设计动机**：每个维度使用多个指标增强可靠性，并通过 Min-Max 归一化计算 N-avg 汇总分数

#### 2. 偏见感知评估

- **功能**：量化模型在不同人口统计群体间的性能差异
- **核心思路**：将数据集按受保护属性（如性别、肤色）分组，分别生成描述并计算各指标，偏见度定义为组间绝对性能差异
- **性别偏见**：$\mathcal{D}_g = \{(I,y,a) \in \mathcal{D} | a = g\}$，$g \in \{\text{woman}, \text{man}\}$
- **肤色偏见**：类似方式，$a \in \{\text{darker-skin}, \text{lighter-skin}\}$
- **语言差异**：比较不同提示语言（英语、日语、中文）对生成结果的影响
- **设计动机**：揭示模型在不同群体上的不平等表现

#### 3. 用户偏好导向评估

- **功能**：根据不同用户类型选择评估维度，给出定制化的模型推荐
- **三种用户类型**：
    - **细节导向用户**：选择{对齐性, 描述性}
    - **风险意识用户**：选择{对齐性, 副作用, 性别偏见, 肤色偏见}
    - **准确性导向用户**：选择{对齐性, 副作用}
- **偏好分数**：所选维度 N-avg 分数的平均值
- **设计动机**：没有普遍最优的模型，只有针对特定需求最优的模型

### 损失函数/训练策略

LOTUS 是评估框架，不涉及模型训练。评估的五个模型均使用 7B 参数变体（MiniGPT-4, InstructBLIP, LLaVA-1.5, mPLUG-Owl2, Qwen2-VL），确保公平比较。

## 实验关键数据

### 主实验

COCO Karpathy 测试集（5000张图像）上的统一评估（N-avg，归一化到 0-1）：

| 模型 | 对齐性 | 描述性 | 复杂度 | 副作用↑ |
|------|--------|--------|--------|---------|
| MiniGPT-4 | 0.19 | 0.22 | 0.38 | 0.18 |
| InstructBLIP | 0.18 | 0.40 | 0.41 | 0.66 |
| LLaVA-1.5 | 0.67 | 0.11 | 0.08 | 0.71 |
| mPLUG-Owl2 | 0.49 | 0.34 | 0.28 | 0.58 |
| **Qwen2-VL** | **0.82** | **1.00** | **1.00** | 0.46 |

偏见评估（性别偏见，N-avg↑表示偏见越小越好）：

| 模型 | 性别偏见 N-avg | 肤色偏见 N-avg |
|------|---------------|---------------|
| MiniGPT-4 | 0.51 | 0.55 |
| LLaVA-1.5 | 0.46 | **0.67** |
| Qwen2-VL | **0.63** | 0.50 |

### 消融实验

用户偏好导向评估结果（偏好分数）：

| 用户类型 | 最佳模型 | 说明 |
|---------|---------|------|
| 细节导向 | Qwen2-VL | 描述性和对齐性远超其他模型 |
| 风险意识 | LLaVA-1.5 | 幻觉风险低，肤色偏见最小 |
| 准确性导向 | LLaVA-1.5 | 副作用最少，最可靠 |

### 关键发现

1. **没有万能模型**：Qwen2-VL 在描述质量上最强（对齐0.82、描述性1.00），但副作用和肤色偏见较差
2. **描述性与偏见的trade-off**：
    - 描述性高 → 性别偏见低（相关系数 -0.92）：更详细的模型趋于公平描述两种性别
    - 描述性高 → 肤色偏见高（相关系数 0.94）：更详细的模型更多描述深肤色相关词汇
3. **性别偏见与肤色偏见负相关**（-0.55）：同时减少两种偏见是困难的
4. **语言差异显著**：Qwen2-VL 的语言差异最大（N-avg 0.28），其句法复杂度差异高达 90.8
5. **性别词使用差异**：LLaVA-1.5 对男女图像的性别词回忆差异最大（|Δ|=6.1）
6. **种族词使用**：Qwen2-VL 对深肤色图像使用种族相关词的比例（7.0%）远高于浅肤色（2.9%），|Δ|=4.1

## 亮点与洞察

1. **揭示了"描述越详细偏见越大"的trade-off**：这是一个重要的发现，提醒社区在追求描述质量时注意公平性
2. **用户偏好评估的实用价值**：不同用户需要不同的模型推荐，这对实际部署非常有意义
3. **指标设计全面且可复现**：统一了多个已有指标，N-avg 归一化方案简单有效
4. **偏见分析的深入性**：不仅量化偏见大小，还分析了偏见产生的原因（如描述性增加导致种族词使用增多）
5. **LOTUS 作为持续更新的排行榜**：在 HuggingFace 上部署，可持续评估新模型

## 局限与展望

1. 仅评估了五个模型且限于 7B 变体，更大/更新的模型（如 GPT-4V, Gemini）未涵盖
2. 社会偏见仅考虑二元性别和二元肤色，未涵盖更细粒度或其他维度（如年龄、残疾等）
3. 偏见度量基于性能差异，但差异方向的含义可能需要更深入讨论
4. COCO 数据集的代表性可能不足，特定领域图像的评估需要补充
5. 语言差异实验仅覆盖三种语言，需要更广泛的多语言评估
6. 未提供减少偏见的具体建议或方法

## 相关工作与启发

- **DCScore, DetailCaps**：专注于描述性的评估方法，LOTUS 将其纳入统一框架
- **CHAIR, FaithScore**：幻觉评估指标，在 LOTUS 中作为副作用维度
- **Zhao et al. (2021)**：COCO 的性别和肤色标注，为偏见分析提供了基础
- 启发："No one-size-fits-all"的评估理念可以推广到其他生成任务的评估中

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 统一框架+偏见评估+用户偏好的组合新颖，描述性-偏见trade-off是重要发现
- **实验充分度**: ⭐⭐⭐⭐ — 多维度评估全面，但模型数量有限
- **写作质量**: ⭐⭐⭐⭐ — 可视化优秀，条理清晰
- **价值**: ⭐⭐⭐⭐ — 作为社区基准设施价值高，但缺乏改进方法

<!-- RELATED:START -->

## 相关论文

- [Aligning LLMs by Predicting Preferences from User Writing Samples](../../ICML2025/recommender/aligning_llms_by_predicting_preferences_from_user_writing_samples.md)
- [FineVQ: Fine-Grained User Generated Content Video Quality Assessment](../../CVPR2025/recommender/finevq_fine-grained_user_generated_content_video_quality_assessment.md)
- [QuRe: Query-Relevant Retrieval through Hard Negative Sampling in Composed Image Retrieval](../../ICML2025/recommender/qure_query-relevant_retrieval_through_hard_negative_sampling_in_composed_image_r.md)
- [VisionArena: 230k Real World User-VLM Conversations with Preference Labels](../../CVPR2025/recommender/visionarena_230k_real_world_user-vlm_conversations_with_preference_labels.md)
- [AID-AppEAL: Automatic Image Dataset and Algorithm for Content Appeal Enhancement and Assessment Labeling](../../ECCV2024/recommender/aid-appeal_automatic_image_dataset_and_algorithm_for_content_appeal_enhancement_.md)

<!-- RELATED:END -->
