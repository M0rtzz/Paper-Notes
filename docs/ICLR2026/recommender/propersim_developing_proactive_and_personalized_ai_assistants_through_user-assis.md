---
title: >-
  [论文解读] ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation
description: >-
  [ICLR 2026][推荐系统][proactive agent] 提出ProPerSim模拟框架，构建基于大五人格的32种用户persona在Smallville家庭环境中的日常行为模拟，AI助手通过每2.5分钟的主动推荐决策和DPO偏好学习，在14天模拟中将用户满意度从2.2/4提升至3.3/4…
tags:
  - "ICLR 2026"
  - "推荐系统"
  - "proactive agent"
  - "personalization"
  - "user simulation"
  - "DPO"
  - "Big Five personality"
  - "generative agents"
---

# ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation

**会议**: ICLR 2026  
**arXiv**: [2509.21730](https://arxiv.org/abs/2509.21730)  
**代码**: [GitHub](https://github.com/jiho283/ProPerSim)  
**领域**: 推荐系统  
**关键词**: proactive agent, personalization, user simulation, DPO, Big Five personality, generative agents

## 一句话总结
提出ProPerSim模拟框架，构建基于大五人格的32种用户persona在Smallville家庭环境中的日常行为模拟，AI助手通过每2.5分钟的主动推荐决策和DPO偏好学习，在14天模拟中将用户满意度从2.2/4提升至3.3/4，首次验证了主动性+个性化统一的可行性。

## 研究背景与动机

**领域现状**：LLM助手正从被动应答向主动推荐和个性化两个方向分别演进。Proactive Agent（Lu et al., 2024）探索了主动推荐但不考虑个人偏好，个性化方法（RLHF等）适配用户但仍需用户发起交互。

**现有痛点**：
- 仅有主动性 → 给素食者推荐牛排馆（Figure 1的例子），推荐时机和内容与个人偏好不匹配
- 仅有个性化 → 即使推荐再精准也需要用户主动开口，错过了最佳推荐时机
- 大规模真实行为数据收集面临成本和隐私挑战，真人实验极其昂贵
- 现有proactive研究是事件驱动的（用户做了某action才触发），未探索基于时间的持续监控模式

**核心矛盾**：需要大量用户-助手交互数据来同时学习"何时推荐"和"推荐什么" → 但真实数据收集不可行。

**本文目标** 在模拟环境中统一主动性和个性化，开发能随时间适应个体用户的AI助手。

**切入角度**：用LLM-based user agent（基于大五人格的丰富persona）模拟真实用户行为，在模拟中收集偏好数据做DPO训练。

**核心 idea**：用Generative Agents模拟用户+个性化rubric评估推荐+DPO偏好学习→形成持续改进的proactive+personalized闭环。

## 方法详解

### 整体框架

系统由三部分组成：(1) 基于persona的用户agent在家庭环境中生成日常行为序列 $\{(A_i, \text{Range}_i)\}$；(2) AI助手每隔 $T=2.5$ 分钟观察用户行为决定是否推荐 $R_t = \mathcal{A}_\theta(A_t, S_t^{(a)})$；(3) 用户agent基于个性化rubric评分 $\text{Score}_t = \mathcal{E}(P, r, A_t, R_t, S_t^{(u)})$。

### 关键设计

1. **大五人格驱动的用户Persona系统**:
    - 功能：构建32种多样化用户persona，驱动行为生成和推荐评估
    - 核心思路：每个persona由5个大五人格维度（Extraversion/Agreeableness/Openness/Conscientiousness/Neuroticism的High/Low）+ 6个扩展属性（年龄、背景、兴趣、生活方式、日计划需求、长期目标）定义。GPT-4o生成属性，确保与人格特质一致。UMAP+HDBSCAN验证32个persona的分离性和多样性
    - 设计动机：大五人格是心理学中最广泛验证的个性模型，不同人格组合自然导致不同的推荐偏好——低外向性persona偏好独处活动，高尽责性persona偏好结构化推荐

2. **四维个性化评估Rubric**:
    - 功能：基于353人AMT调研筛选的4个评估维度，为每个persona生成个性化评估标准
    - 核心思路：从10个候选维度经AMT投票（排除<50%支持的Diversity和Interruption）保留：Personal Preference（内容对齐）、Frequency（推荐频率）、Timing（时机恰当性）、Communication & Safety（沟通风格+安全）。每个维度的具体标准由GPT-4o根据persona定制（如低外向性persona："I prefer receiving recommendations no more than once every two hours"）。评估用Gemini 2.0 Flash，每维度二值评分
    - 设计动机：评估标准必须同时反映任务的通用重要性（来自大规模调研）和个体差异（来自persona定制），两层设计确保既有共识基础又有个性化空间

3. **RAG+DPO偏好对齐的ProPerAssistant**:
    - 功能：构建一个能持续从用户反馈中学习的主动推荐助手
    - 核心思路：内部状态 $S_t^{(a)}$ 包含结构化日记忆（近10分钟详细+早期压缩为1h/4h摘要）+ OpenAI embedding检索的top-5相似历史交互。每个时间步生成 $n=2$ 候选推荐（含"无推荐"选项），用户评分后形成偏好对，存入replay buffer。每日结束时从buffer随机采样200条做DPO训练：$\mathcal{L}_{\text{DPO}} = -\log\sigma(\beta(\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}))$
    - 设计动机：DPO避免了RLHF的reward model训练复杂性；replay buffer借鉴RL经验回放，防止遗忘早期经验；LoRA微调的LLaMA 3.3 70B（4-bit量化）平衡性能与效率

### 损失函数 / 训练策略

基座模型：LLaMA 3.3 70B（4-bit量化），LoRA微调。DPO训练：每天结束后从累积replay buffer随机采样200条，候选数 $n=2$。模拟设置：时间步 $T=2.5$ 分钟，每个persona前后模拟14天。单persona模拟成本：约10天×1 A100 GPU + ~$30 API费用。

## 实验关键数据

### 主实验——方法对比

| 方法 | Day 1 均分 | Day 14 均分 | 特点 |
|------|-----------|------------|------|
| No Memory | ~2.1 | ~2.2 | 仅当前action |
| AR Memory (A,R) | ~2.3 | ~2.3 | 历史action+推荐 |
| ARS Memory (A,R,Score) | ~2.6 | ~2.5 | 加评分到prompt |
| **ProPerAssistant** | **~2.2** | **~3.3** | DPO偏好学习 |

### Persona维度分析

| 分析维度 | 最佳Persona | 最差Persona | 差异原因 |
|---------|-------------|-------------|---------|
| 最终得分 | 3.8/4 | 2.5/4 | 偏好复杂度差异 |
| 偏好特征 | 简单哲学/创意类 | 数据驱动/辩论类 | 后者需多维匹配 |
| 时间窗口 | 灵活 | 严格(6-9AM/21:00+) | 窄窗口更难适应 |

### 关键发现
- ProPerAssistant从Day 2开始快速上升并保持领先，日均分接近3.4/4，证明DPO偏好学习远优于in-context reward信号（ARS Memory）
- 推荐频率从初始24次/小时降至约6次/小时→学会了"不推荐"同样重要
- 成功推荐率（score≥3的推荐占比）从51.06%→71.51%
- 低外向性persona改善更多（家庭场景匹配独处偏好），低开放性persona也改善更多（偏好一致性推荐更容易学习）
- Frequency和Timing维度改善最显著，Personal Preference改善较平——因为推荐总数下降，high-quality推荐占比实际提升（0.77→0.83）
- 人类评估确认高质量：行为自然度8.25/10，persona一致性8.02/10，评估合理率90.54%

## 亮点与洞察
- **首创主动性+个性化统一框架**：填补了两个独立研究方向的空白，定义了proactive+personalized的新任务形态
- **时间驱动 vs 事件驱动的主动性**：每 $T$ 时间步决策更接近真实助手的持续监控模式，比事件驱动更自然
- **DPO >> in-context reward**：ARS Memory直接把分数放到prompt里但效果远不如DPO训练——显式偏好学习是必要的，in-context reward信号不足以驱动真正的适应
- **"不推荐"是关键能力**：助手学会抑制推荐（频率下降4×）与推荐内容质量提升同等重要

## 局限与展望
- 计算成本极高（单persona 10天A100+$30 API），32个persona的完整实验约320天GPU时
- 用户行为和评估均基于LLM模拟而非真人——模拟与真实行为的差距未被量化
- 仅限家庭场景（Smallville house），未扩展到工作、社交、户外等场景
- DPO候选数n=2受限于成本，更多候选可能提供更丰富的偏好信号
- 仅优化即时reward，未考虑延迟reward（如长期满意度、推荐多样性）

## 相关工作与启发
- **vs Proactive Agent (Lu et al., 2024)**：Lu的工作用6790训练事件训练主动agent但不考虑个人偏好差异；ProPerSim通过persona驱动的模拟实现个性化
- **vs Generative Agents (Park et al., 2023)**：Park的25个agent做社会模拟；ProPerSim将generative agent框架扩展到用户-助手交互模拟，增加了评估维度和偏好学习
- **vs 个性化RLHF**：传统个性化通过一次性对齐完成；ProPerAssistant通过日累积replay buffer实现持续适应

## 评分
- 新颖性: ⭐⭐⭐⭐ 主动+个性化统一是有意义的新方向，模拟框架设计完整
- 实验充分度: ⭐⭐⭐⭐ 32 persona、4基线、人格维度分析、人类评估——但缺乏真人验证
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，评估设计系统，persona示例丰富
- 价值: ⭐⭐⭐⭐ 为个人助手研究提供有价值的模拟平台和基线

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Learning to Retrieve User History and Generate User Profiles for Personalized Persuasiveness Prediction](../../ACL2026/recommender/learning_to_retrieve_user_history_and_generate_user_profiles_for_personalized_pe.md)
- [\[AAAI 2026\] TraveLLaMA: A Multimodal Travel Assistant with Large-Scale Dataset and Structured Reasoning](../../AAAI2026/recommender/travellama_a_multimodal_travel_assistant_with_large-scale_dataset_and_structured.md)
- [\[ACL 2026\] WildFeedback: Aligning LLMs With In-situ User Interactions And Feedback](../../ACL2026/recommender/wildfeedback_aligning_llms_with_in-situ_user_interactions_and_feedback.md)
- [\[ACL 2026\] Personalized Benchmarking: Evaluating LLMs by Individual Preferences](../../ACL2026/recommender/personalized_benchmarking_evaluating_llms_by_individual_preferences.md)
- [\[ACL 2026\] Scripts Through Time: A Survey of the Evolving Role of Transliteration in NLP](../../ACL2026/recommender/scripts_through_time_a_survey_of_the_evolving_role_of_transliteration_in_nlp.md)

</div>

<!-- RELATED:END -->
