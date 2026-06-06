---
title: >-
  [论文解读] SenseJudge: Human-Centric Preference-Driven Judgment Framework
description: >-
  [ACL 2026][推荐系统][LLM评估] 提出 SenseJudge，一种基于显式人类偏好的可定制化 LLM 判断框架，配合真实多轮对话基准 SenseBench，在个性化评判任务中平均准确率比基线高 16.08%，模型排名与真实人类排名一致。
tags:
  - "ACL 2026"
  - "推荐系统"
  - "LLM评估"
  - "个性化判断"
  - "偏好驱动"
  - "多轮对话"
  - "模型排名"
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# SenseJudge: Human-Centric Preference-Driven Judgment Framework

**会议**: ACL 2026  
**arXiv**: [2606.03189](https://arxiv.org/abs/2606.03189)  
**代码**: [GitHub](https://github.com/qiongrenpiqida/SenseJudge)  
**领域**: recommender  
**关键词**: LLM评估, 个性化判断, 偏好驱动, 多轮对话, 模型排名

## 一句话总结
提出 SenseJudge，一种基于显式人类偏好的可定制化 LLM 判断框架，配合真实多轮对话基准 SenseBench，在个性化评判任务中平均准确率比基线高 16.08%，模型排名与真实人类排名一致。

## 研究背景与动机
**领域现状**: LLM-as-a-Judge 范式日益流行，用于评估模型响应、生成偏好数据和模型排名。

**现有痛点**: (1) 现有判断方法（PandaLM、Auto-j、奖励模型）依赖固定偏好数据训练，学到的是同质化标准，忽略了用户偏好的多样性；(2) 现有基准（MT-Bench、Auto-j）以单轮或双轮对话为主，与真实多轮人机交互场景脱节；(3) 训练好的奖励模型在面对多样化真实场景时泛化能力有限。

**核心矛盾**: 用户偏好是多样化和场景依赖的（有人重视创意、有人重视格式、有人重视准确性），但现有判断器只学到了一种固定的偏好标准。

**本文目标**: 构建能适应不同用户偏好的可定制化 LLM 判断框架，以及能真实反映人机交互复杂度的评测基准。

**切入角度**: 从少量人工标注中提取显式偏好文本，用多偏好投票机制让小模型也能做出准确的个性化判断。

**核心 idea**: 偏好提取 + 偏好集选择 + 多偏好投票 = 无需训练的个性化 LLM 判断。

## 方法详解

### 整体框架
SenseBench 通过质量过滤+挑战过滤从真实用户对话中构建多轮评测基准。SenseJudge 从少量人工标注对中提取偏好文本，选择最优偏好子集，推理时通过多偏好投票产生最终判断。

### 关键设计
1. **SenseBench 基准构建**:

    - 功能：提供贴近真实人机交互的多轮、多领域评测基准（8 个类别 × 125 题）
    - 核心思路：两阶段过滤——(1) 质量过滤：用 Qwen3-14B 去噪+分类（数学/逻辑/代码/创意写作/角色扮演/翻译/QA/NLU）；(2) 挑战过滤：多模型响应对比（强模型 vs 弱模型）+ GPT-4 自动筛选 + 人工校验，确保问题具有区分度
    - 设计动机：现有基准多为单轮简单任务，无法反映真实用户场景的复杂性和多轮上下文依赖

2. **偏好提取与选择**:

    - 功能：从少量标注中蒸馏出可泛化的显式偏好集
    - 核心思路：(1) 偏好生成：用 DeepSeek-R1 从标注对 $(q, \text{chosen}, \text{rejected})$ 中生成显式偏好文本；(2) 偏好集选择：遍历所有偏好子集 $\mathcal{P}_k \subseteq P$，在标注集上多偏好投票，选择准确率最高的子集 $\mathcal{P}_k^*$；(3) 偏好应用：在测试集上用 $\mathcal{P}_k^*$ 中的每个偏好独立判断，最终多数投票
    - 设计动机：不同偏好文本捕捉用户标注决策的不同方面（如"重视逻辑严谨性"vs"重视回答全面性"），组合使用比单一偏好更鲁棒

3. **输入输出格式与投票机制**:

    - 功能：标准化判断流程，减少位置偏差
    - 核心思路：输入 $I = \{q, (r_1, r_2), p\}$，输出 judgment + analysis；不允许"平局"选项以强制模型区分；正反序都评估以检测位置偏差；最终通过多偏好投票产生稳定判断
    - 设计动机：位置偏差是 LLM-as-Judge 的已知问题（模型倾向选第一个/最后一个），双序评估和多偏好投票可有效缓解

## 实验关键数据

### 主实验（LLM-as-a-Personalized-Judge 准确率 %）

| 方法 | Math | Code | Logic | QA | Write | Role | NLU | Trans | Overall |
|------|------|------|-------|-----|-------|------|-----|-------|---------|
| GPT-4o | 66.00 | 61.60 | 65.47 | 72.93 | 60.80 | 63.20 | 65.47 | 56.40 | 63.98 |
| DeepSeek-V3 | 72.80 | 62.27 | 66.67 | 77.07 | 62.67 | 64.40 | 64.80 | 61.87 | 66.57 |
| Skywork-Reward-Gemma2-27B | 70.40 | 61.60 | 66.10 | 74.10 | 64.00 | 60.00 | 62.70 | 58.40 | 64.70 |
| Qwen2.5-14B + **SenseJudge** | **73.45** | **80.90** | **72.44** | **85.67** | **72.89** | **75.24** | **76.80** | **74.21** | **76.88** |
| Qwen2.5-72B + **SenseJudge** | **82.30** | **89.01** | **79.76** | **89.87** | **79.82** | **82.12** | **78.10** | **75.23** | **81.99** |
| Qwen3-14B + **SenseJudge** | **86.53** | **87.96** | **83.69** | **92.24** | **75.27** | **81.04** | **78.72** | **75.78** | **82.65** |

### 一致性与位置偏差

| 模型 | 原始一致性 | +SenseJudge 一致性 |
|------|-----------|------------------|
| Qwen2.5-14B-Instruct | 69.97% | **74.17%** |
| Llama3.1-8B-Instruct | 60.36% | **68.19%** |
| Qwen2.5-72B-Instruct | 78.86% | 78.79% |
| Qwen3-14B-Instruct | 81.23% | 81.30% |

### 关键发现
- SenseJudge 平均比基线提升 +16.08%，即使在 8B/14B 小模型上也超越 GPT-4o 等强模型的直接判断
- 8 个类别全面提升，其中 Code (+20.10) 和 Trans (+18.84) 提升最大
- 奖励模型（INF-ORM-70B、QRM-27B）在个性化数据集上准确率 <65%，说明固定偏好难以泛化
- SenseJudge 显著缓解位置偏差，尤其对小模型效果更好
- 在 RewardBench 上达到 90.55%，接近专门训练的 Skywork-Critic（92.2%），验证通用有效性
- 模型排名结果与 Arena 人类排名一致：DeepSeek-R1 > Claude-3-7-Sonnet > GPT-4o > Qwen2.5-72B > GPT-3.5

## 亮点与洞察
- 偏好提取 + 子集选择 + 投票的三步流程简洁优雅，不需要训练判断模型即可实现个性化
- 从失败中学习的思路——通过少量标注反向推断偏好——比直接训练奖励模型更数据高效
- SenseBench 的构建方法（强弱模型对比 + 人工校验）确保了评测基准的区分度
- 证明了小模型 + 好偏好 > 大模型 + 无偏好，为低成本部署提供了新思路

## 局限与展望
- 偏好构建依赖 DeepSeek-R1 等强模型生成，偏好质量受生成模型能力制约（消融实验证实弱模型生成的偏好效果更差）
- 仅 3 位标注者，标注规模有限（每人 1000 条），更大规模标注可能揭示更丰富的偏好模式
- 偏好子集选择需要遍历组合空间，随偏好集增大计算量指数增长
- 跨域偏好迁移效果参差不齐（数学→逻辑 78.62% vs 数学→翻译 61.83%）

## 相关工作与启发
- Auto-j / PandaLM 等训练式判断器学到固定偏好，SenseJudge 的显式偏好文本更灵活可解释
- 个性化 LLM（OPPU / 多粒度兴趣预测）关注响应个性化，SenseJudge 关注评判个性化——互补方向
- 偏好投票机制可推广到任何需要多视角聚合的评估场景（如代码审查、内容审核）

## 评分
- 新颖性: ⭐⭐⭐⭐ 显式偏好驱动的个性化判断是有意义的新方向
- 实验充分度: ⭐⭐⭐⭐ 多模型对比+一致性/位置偏差分析+消融+跨域+RewardBench 验证
- 写作质量: ⭐⭐⭐ 结构完整但部分公式表述可更简洁
- 价值: ⭐⭐⭐⭐ 实用性强，低成本个性化评判的落地价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Personalizing LLMs with Binary Feedback: A Preference-Corrected Optimization Framework](personalizing_llms_with_binary_feedback_a_preference-corrected_optimization_fram.md)
- [\[ACL 2026\] Intent-Driven Semantic ID Generation for Grounded Conversational News Recommendation](intent-driven_semantic_id_generation_for_grounded_conversational_news_recommenda.md)
- [\[AAAI 2026\] Preference is More Than Comparisons: Rethinking Dueling Bandits with Augmented Human Feedback](../../AAAI2026/recommender/preference_is_more_than_comparisons_rethinking_dueling_bandits_with_augmented_hu.md)
- [\[ACL 2026\] Quality Over Clicks: Intrinsic Quality-Driven Iterative RL for Cold-Start E-Commerce Query Suggestion](quality_over_clicks_intrinsic_quality-driven_iterative_reinforcement_learning_fo.md)
- [\[ACL 2026\] Mirroring Users: Towards Building Preference-aligned User Simulator with User Feedback in Recommendation](mirroring_users_towards_building_preference-aligned_user_simulator_with_user_fee.md)

</div>

<!-- RELATED:END -->
