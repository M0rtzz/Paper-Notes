---
title: >-
  [论文解读] SocialEval: Evaluating Social Intelligence of Large Language Models
description: >-
  [ACL 2025][LLM/NLP][Social Intelligence] 提出 SocialEval —— 一个基于叙事脚本的双语社会智能基准，通过手工构建 153 个"世界树"将社交互动建模为目标条件 MDP，整合结果导向的目标达成评估（GAE）和过程导向的人际能力评估（IAE），系统评测 LLM 在多回合社交场景中的社会智能及其与人类的差距。
tags:
  - ACL 2025
  - LLM/NLP
  - Social Intelligence
  - Benchmark
  - World Tree
  - Interpersonal Ability
  - Goal Achievement
  - BESSI
---

# SocialEval: Evaluating Social Intelligence of Large Language Models

**会议**: ACL 2025  
**arXiv**: [2506.00900](https://arxiv.org/abs/2506.00900)  
**代码**: https://github.com/thu-coai/SocialEval  
**领域**: LLM NLP / 社会智能评估  
**关键词**: Social Intelligence, Benchmark, World Tree, Interpersonal Ability, Goal Achievement, BESSI

## 一句话总结

提出 SocialEval —— 一个基于叙事脚本的双语社会智能基准，通过手工构建 153 个"世界树"将社交互动建模为目标条件 MDP，整合结果导向的目标达成评估（GAE）和过程导向的人际能力评估（IAE），系统评测 LLM 在多回合社交场景中的社会智能及其与人类的差距。

## 研究背景与动机

- **领域现状**：LLM 在社会仿真（如 Generative Agents）中展现出新兴的社会智能（SI），被广泛用于社会科学研究、人际情境训练等场景，但"LLM 的 SI 究竟有多强、与人类差距多大"这一根本问题尚未得到充分回答。
- **现有痛点**：已有评估工作（SOTOPIA、AgentSense 等）存在两大缺陷：(1) 仅关注单回合社会动态，忽略了多回合序列依赖的社交互动过程；(2) 仅评估终端目标达成，缺少对目标追求过程中人际能力运用的细粒度评测。
- **核心矛盾**：社会心理学指出，SI 本质上是一个动态过程——个体在不断演变的叙事性社交活动中（脚本理论）持续调节人际技能以达成社交目标，现有方法无法捕捉这一完整过程。
- **本文切入角度**：借鉴脚本理论（Schank & Abelson）和戏剧理论（Goffman），将社交活动建模为"世界树"——多条由人际能力驱动的情节线交织而成的树结构，同时评估 LLM 在树中的导航结果（目标达成）和导航过程（能力运用）。

## 方法详解

### 社交世界分类法

基于 interdependence theory（Kihlstrom & Cantor），用自利（self-interest）和利他（altruism）两个维度的笛卡尔积构建社交世界分类法。每个维度取值 {1, 0, -1}，产生 9 种取向，排除缺乏目标的闲聊 (0,0) 和罕见的退缩 (-1,0) 后，保留 7 种社交取向：

| 大类 | 取向 | (自利, 利他) | 典型行为 |
|------|------|-------------|---------|
| 亲社会 | 合作 Cooperation | (1, 1) | 双方互利合作 |
| 亲社会 | 谈判 Negotiation | (1, 0) | 争取自身最优结果 |
| 亲社会 | 援助 Assistance | (0, 1) | 主动帮助他人 |
| 亲社会 | 利他 Altruism | (-1, 1) | 牺牲自我支持他人 |
| 亲自我 | 竞争 Competition | (1, -1) | 损人利己 |
| 反社会 | 诱导 Induction | (0, -1) | 操纵他人使其受损 |
| 反社会 | 冲突 Conflict | (-1, -1) | 对抗性破坏行为 |

### 人际能力清单（BESSI 框架）

采用心理学 BESSI 框架，定义 5 大能力维度、32 种具体人际能力作为过程评估维度：

- **社交参与**（5 种能力）：领导力、说服力、会话、表达、精力管理
- **合作**（5 种能力）：团队协作、信任、换位思考、社交温暖、伦理
- **自我管理**（12 种能力）：任务/时间/细节管理、目标调节、决策、适应性、独立性、自我反思等
- **情绪韧性**（5 种能力）：压力调节、乐观、愤怒管理、自信、冲动控制
- **创新**（5 种能力）：抽象思维、创造力、艺术、文化能力、信息处理

### 世界树构建

每个世界树由以下组件构成：

- **角色**：一个主角 + 若干配角，每人有公开信息、私密信息和社交目标
- **场景**：世界树的根节点，参考 Bilibili/YouTube 互动视频创作
- **情节（Episodes）**：由角色间的对话互动组成，平均每棵树 6.5 个情节
- **情节过渡**：关键抉择点，主角从多个 candidate utterances（平均 2.17 个）中选择，每个选项体现不同的人际能力，导向不同后续情节线
- **情节结局**：标注是否达成社交目标，用于 GAE 评估

质量控制流程：检查员先经培训（pilot 2 棵世界树），然后经三阶段交叉检查，最终一致率达 **95%**；翻译使用 GPT-4o 并由专业译者审核，接受率达 **97%**。

### 双任务评估框架

将每棵世界树形式化为目标条件 MDP $(S, A, T, R)$，定义两个评估任务：

**任务 1 — GAE（目标达成评估）**：LLM 扮演主角，在每个情节过渡点选择 utterance 以推进情节。经多步决策后到达结局，判断是否达成社交目标。指标为目标达成率。

**任务 2 — IAE（人际能力评估）**：对每个 candidate utterance 构造选择题——包含正确 utterance 和若干看似合理但反映错误能力的干扰项，测试 LLM 是否正确识别目标能力。指标为能力选择准确率。

为消除位置偏差，每个样本将选项顺序随机打乱 3 次，取多数投票。人类基线由 20 名中英文母语研究生完成。

## 实验关键数据

### GAE 目标达成评估（目标达成率 %）

| 模型 | 亲社会 zh/en | 亲自我 zh/en | 反社会 zh/en | 总体 zh/en |
|------|-------------|-------------|-------------|-----------|
| Human (best) | 100.0/100.0 | 100.0/100.0 | 100.0/100.0 | 100.0/100.0 |
| Human (avg) | 64.9/59.9 | 55.0/40.0 | 51.3/50.0 | 61.8/55.2 |
| DeepSeek-R1 | 54.3/52.8 | 32.2/30.8 | 28.0/26.4 | 47.1/45.7 |
| o1 | 54.3/52.7 | 32.5/31.0 | 27.5/25.7 | 47.0/45.4 |
| Claude-3-opus | 54.0/52.3 | 31.5/29.8 | 29.7/27.6 | 47.0/45.2 |
| DeepSeek-V3 | 53.4/51.9 | 30.1/28.5 | 25.7/23.5 | 46.4/44.8 |
| GPT-4o | 52.8/51.6 | 27.6/25.5 | 23.2/17.6 | 44.6/42.7 |
| Qwen-2.5-72B | 47.5/44.4 | 26.6/23.4 | 20.2/16.1 | 40.3/37.1 |
| Llama-3.1-8B | 37.9/33.7 | 21.2/18.7 | 13.0/10.8 | 31.8/28.2 |

### IAE 人际能力评估（选择准确率 %）

| 模型 | 社交参与 zh/en | 合作 zh/en | 自我管理 zh/en | 情绪韧性 zh/en | 创新 zh/en | 总体 zh/en |
|------|--------------|----------|-------------|-------------|----------|----------|
| Human (best) | 84.9/85.7 | 89.6/92.9 | 86.3/80.6 | 81.8/86.7 | 81.5/85.7 | 85.7/85.3 |
| Human (avg) | 79.5/82.2 | 82.7/84.6 | 80.5/74.5 | 78.6/79.1 | 76.8/79.9 | 80.2/79.1 |
| DeepSeek-R1 | 77.3/76.3 | 83.5/81.9 | 75.2/73.4 | 78.1/76.6 | 75.2/73.2 | 77.6/75.4 |
| Claude-3-opus | 77.5/76.5 | 84.5/82.5 | 74.6/72.5 | 76.9/72.4 | 75.2/72.1 | 77.6/75.3 |
| o1 | 76.2/74.7 | 82.4/80.8 | 75.1/73.1 | 78.1/76.6 | 75.0/73.1 | 77.3/75.5 |
| DeepSeek-V3 | 75.6/74.2 | 81.4/78.5 | 74.9/72.4 | 77.6/76.3 | 74.9/72.9 | 76.5/73.8 |
| Qwen-2.5-72B | 68.9/64.6 | 75.7/73.5 | 68.3/64.6 | 71.2/65.2 | 69.0/64.4 | 70.4/66.5 |
| Mistral-7B | 59.4/55.2 | 64.2/62.1 | 58.5/52.6 | 55.4/52.5 | 49.9/47.7 | 58.8/54.7 |

## 关键发现与分析

1. **LLM 全面落后于人类**：最佳 LLM（DeepSeek-R1）在 GAE 上仍落后 Human-avg 14.7/9.5 个百分点（zh/en），差距主要来自亲自我和反社会场景。

2. **LLM 呈现强烈的亲社会偏好**：人类在亲社会与反社会场景的表现差距约 13.7 个百分点，而 DeepSeek-R1 的差距高达 26.3 个百分点。LLM 偏好选择正面积极行为，即使这会导致目标失败。人类则能灵活调整行为策略。

3. **跨语言 SI 差异显著**：Wilcoxon 检验 ($p<0.001$) 表明 LLM 在中英文 SI 评估上存在显著差异，中文普遍优于英文，这与人类的跨语言差异模式一致。

4. **模型规模正相关**：开源 LLM 的 SI 与参数量正相关。Qwen-2.5 系列从 7B 到 72B，GAE 总体从 33.9/29.9 提升至 40.3/37.1。

5. **选择题能反映生成能力**：LLM 自由生成的 utterance 与 candidate utterances 的语义相似度高达 67%（zh），且相似 utterance 被选中的比例超过 80%，验证了选择题评估的有效性。

6. **类脑功能分区**：分析 Llama-3.1 的表征空间发现，8B 模型的 5 大能力维度已初步聚类但有交叠，70B 模型则形成了清晰分离的簇。神经元激活分析显示，大模型的人际能力对应的神经元区域逐渐变得更密集且相互隔离，类似人脑的功能分区（lobes hypothesis）。

## 亮点与洞察

- **世界树建模范式**：将社交互动建模为世界树（MDP），既有结果评估又有过程评估，这种双维度设计远优于只看终端结果的已有方法。
- **亲社会偏好的 Alignment 意义**：LLM 宁可失败也不愿采取攻击性行为——这说明当前的对齐训练在社会场景下可能过度约束了模型的行为灵活性。
- **类脑分析思路新颖**：用 t-SNE 聚类和 Wanda Score 分析人际能力在 LLM 中的表征和神经元分布，首次将"脑区功能分区"的类比具体化。

## 评分

- ⭐⭐⭐⭐ 创新性：世界树 + 双任务评估范式具有较强原创性
- ⭐⭐⭐⭐ 实验充分度：19+ 个 LLM、人类基线、行为分析、表征分析、神经元分析
- ⭐⭐⭐ 可扩展性：手工构建成本高（每棵树约 12 小时/$40），反社会场景仅 20 棵树
- ⭐⭐⭐⭐ 实用价值：为理解和改进 LLM 社会行为提供了全面的评测工具和深入洞察

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] ExpliCa: Evaluating Explicit Causal Reasoning in Large Language Models](explica_evaluating_explicit_causal_reasoning_in_large_language_models.md)
- [\[ACL 2025\] MIRAGE: Exploring How Large Language Models Perform in Complex Social Interactive Environments](mirage_exploring_how_large_language_models_perform_in_complex_social_interactive.md)
- [\[ACL 2025\] Evaluating Language Models as Synthetic Data Generators](evaluating_lms_synthetic_data_gen.md)
- [\[ACL 2025\] Catching Shortcuts: A Framework for Evaluating Shortcuts in Large Language Models](catching_shortcuts_a_framework_for_evaluating_shortcuts_in_large_language_models.md)
- [\[ACL 2025\] SCoP: Evaluating the Comprehension Process of Large Language Models from a Cognitive View](scop_evaluating_the_comprehension_process_of_large_language_models_from_a_cognit.md)

<!-- RELATED:END -->
