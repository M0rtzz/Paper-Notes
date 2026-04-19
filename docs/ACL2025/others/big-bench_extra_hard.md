---
title: >-
  [论文解读] BIG-Bench Extra Hard
description: >-
  [ACL 2025][LLM评测] 为应对 BIG-Bench Hard 被前沿模型饱和的问题，Google DeepMind 推出 BIG-Bench Extra Hard (BBEH)，用 23 个更难的任务替换 BBH 中的对应任务，最强通用模型仅达 9.8%（调和平均）、最强推理模型达 44.8%，揭示了 LLM 在通用推理上的巨大差距。
tags:
  - ACL 2025
  - LLM评测
  - 通用推理
  - BIG-Bench
  - 基准饱和
  - 推理能力
---

# BIG-Bench Extra Hard

**会议**: ACL 2025  
**arXiv**: [2502.19187](https://arxiv.org/abs/2502.19187)  
**代码**: [github.com/google-deepmind/bbeh](https://github.com/google-deepmind/bbeh)  
**领域**: 其他  
**关键词**: LLM评测, 通用推理, BIG-Bench, 基准饱和, 推理能力

## 一句话总结

为应对 BIG-Bench Hard 被前沿模型饱和的问题，Google DeepMind 推出 BIG-Bench Extra Hard (BBEH)，用 23 个更难的任务替换 BBH 中的对应任务，最强通用模型仅达 9.8%（调和平均）、最强推理模型达 44.8%，揭示了 LLM 在通用推理上的巨大差距。

## 研究背景与动机

当前 LLM 推理能力的评估过度偏向数学和编程领域，而推理实际上涵盖逻辑推导、时空理解、常识推理、幽默理解等广泛的认知技能。BIG-Bench Hard (BBH) 长期作为评估通用推理的事实标准，但存在以下问题导致其逐渐失效：

**性能饱和**：最先进模型在 BBH 上已达 90%+ 准确率，无法区分模型间差异

**随机基线过高**：8/23 的任务为二分类，5/23 的任务选项不超过 5 个

**捷径可用**：某些任务可通过简单规则（如出现三个 L 命令就是三角形）绕过推理直接回答

**输入过短**：BBH 任务的宏平均输入长度仅约 700 字符

**推理步数少**：大多数问题只需少量推理步骤

**技能覆盖有限**：虽然技能种类多，但仍可大幅扩展

BBEH 的目标是在保留 BBH 多样性优势的同时，大幅提升难度并扩展所需的推理技能。

## 方法详解

### 整体框架

BBEH 用 23 个新任务一一替换 BBH 中的原始任务，每个新任务：
- 位于相同的推理领域
- 测试相似（或更多）的推理技能
- 难度显著更高
- 每个任务 200 个样本（Disambiguation QA 为 120 个）

### 关键设计

1. **需要的推理技能大幅扩展**：在 BBH 原有的 11 类技能（时间理解、空间几何、常识、幽默、因果、世界知识、逻辑推导、语言知识、计数过滤、数据结构与算法、算术运算）基础上，新增了 12 类更高阶的技能需求：

    - 多跳推理（Many-hop reasoning）
    - 超长程依赖（Very long-range dependency）
    - 违背强先验（Going against strong prior）
    - 即时学习（Learning on the fly）
    - 抗干扰（Dealing with distractors）
    - 长上下文处理（Long-context）
    - 大海捞针（Needle in a haystack）
    - 推理链纠错（Finding errors in reasoning traces）
    - 归纳推理（Inductive reasoning）
    - 约束满足（Constraint satisfaction）
    - 组合理解（Compositional understanding）
    - 知识密集推理（Knowledge-intensive reasoning）

2. **半对抗式难度校准**：选择两个参考模型——Gemini 1.5 Flash（通用）和 Gemini-2.0-Flash-Thinking-Exp（推理），迭代增加任务难度，直到两个参考模型的准确率均低于 70%。通常只将模型作为黑盒使用，但在必要时会分析模型策略（如发现模型用 Python 直接执行布尔表达式后，改用自然语言子表达式替换 True/False）。

3. **典型任务升级示例**：

    - **Boolean Expressions**：将 "True" 替换为 "The capital of Canada is Ottawa" 等文本子表达式，防止模型用代码执行
    - **Buggy Tables**：从简单表格查询升级为理解和重建大型有缺陷表格
    - **Object Counting**：从短列表简单计数升级为超长列表中带大量干扰项的特定类型计数
    - **Word Sorting**：从标准字母序排序升级为使用修改后的字母序（违背先验）+ 查找排序错误

### 数据集属性

- **输入长度**：BBEH 的宏平均上下文长度是 BBH 的约 6 倍
- **所需思考量**：以 Gemini 2.0 Flash 的输出长度为代理指标，BBEH 是 BBH 的约 7 倍
- **随机基线**：BBEH 整体随机基线为 8.4%，远低于 BBH
- **BBEH Mini**：包含 460 个样本（每任务 20 个），用于快速低成本实验

## 实验关键数据

### 主实验（BBEH 调和平均准确率）

| 模型 | 类型 | BBEH 调和平均↑ |
|------|------|-------------|
| Qwen-2.5-7B-Instruct | 通用 | 2.4% |
| Llama 3.1 8B Instruct | 通用 | 3.0% |
| Gemma2 27B IT | 通用 | 3.6% |
| Gemma3 27B | 通用 | 4.5% |
| Gemini 2.0 Flash-Lite | 通用 | 4.9% |
| Gemini 2.0 Flash | 通用 | 8.0% |
| GPT-4o | 通用 | **9.8%** |
| Distill R1 Qwen 32B | 推理 | 6.0% |
| DeepSeek R1 | 推理 | 5.2% |
| o3-mini (high) | 推理 | **44.8%** |

### 单任务结果亮点

| 任务 | GPT-4o | o3-mini | DeepSeek R1 | 说明 |
|------|--------|---------|-------------|------|
| Buggy Tables | 3.5 | 59.5 | 4.5 | 推理模型巨大优势 |
| Object Counting | 11.0 | 90.0 | 76.5 | 推理模型在计数上优势明显 |
| Object Properties | 1.5 | 56.5 | 0.0 | o3-mini 独占鳌头 |
| Temporal Sequences | 0.5 | 68.5 | 0.0 | o3-mini 独占鳌头 |
| NYCC (幽默) | 23.0 | 16.0 | 20.0 | GPT-4o 在幽默理解上领先 |
| SARC Triples (讽刺) | 38.5 | 24.0 | 28.5 | GPT-4o 领先 |
| Causal Understanding | 54.0 | 54.0 | 54.5 | 因果理解各模型接近 |

### 关键发现

1. **通用推理仍极具挑战**：最强通用模型 GPT-4o 调和平均仅 9.8%，说明即使是最先进的 LLM 在通用推理上仍有巨大提升空间
2. **推理模型优势不均衡**：o3-mini 在计数、规划、算术、数据结构等形式化问题上优势巨大，但在常识、幽默、讽刺、因果推理等"软"推理技能上甚至不如通用模型
3. **模型大小的影响类似**：更大模型在形式化推理上提升大，在软推理上提升小
4. **上下文长度和思考量的影响**：o3-mini 的优势随上下文长度和所需思考量增加而增大
5. **DeepSeek R1 的不均衡表现**：微平均准确率优于通用模型，但因部分任务极低导致调和平均反而低于两个通用模型
6. **不同模型擅长不同推理类型**：没有一个模型在所有任务上全面领先

## 亮点与洞察

1. **评价指标的选择**：使用调和平均而非算术平均或微平均，能更好地惩罚"偏科"模型，反映真正的通用推理能力——这是一个重要的方法论贡献
2. **揭示推理模型的真实边界**：当前推理模型在形式化问题上飞跃式提升（如 AIME2024 从 13.4% 到 87.3%），但在需要常识、幽默、因果等软推理的任务上进展甚微
3. **基准设计的系统方法论**：保留 BBH 优势 + 修复缺陷 + 半对抗式校准的设计流程，为未来构建新基准提供了范式
4. **真实世界推理能力的代理**：BBEH 的任务更接近现实场景（长上下文、多干扰、需要多步推理），比纯数学/编程基准更能反映模型的实际推理能力

## 局限与展望

- **参考模型偏差**：半对抗构建不可避免地偏向参考模型的弱点，非参考模型的公平比较受限
- **静态基准的时效性**：随着模型持续进步，BBEH 也终将被饱和
- **纯文本限制**：不涉及多模态推理能力的评估
- **样本量有限**：每任务仅 200 个样本，统计显著性可能不足
- **缺乏过程评估**：只看最终答案正确性，无法深入分析模型的推理过程质量
- **评估成本**：推理模型在 BBEH 上的输出很长，评估成本显著高于 BBH

## 相关工作与启发

- **与 BBH 的传承关系**：直接替换 BBH 的 23 个任务，保持领域一致性
- **与数学/编程基准的互补**：AIME、GSM8K 等测试的是数学推理，BBEH 测试的是更广泛的通用推理
- **Inverse Scaling 的启发**：某些任务专门测试模型是否能违背已有先验（如修改的字母排序），与 inverse scaling 研究一脉相承
- **对模型开发的指导**：不同类型推理能力的差异化表现，为模型训练的能力平衡提供了指引

## 评分

- 新颖性: ⭐⭐⭐⭐ 任务设计巧妙，技能覆盖全面，半对抗式构建方法有创新
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖多个主流模型系列，分析维度丰富（模型大小、类型、上下文长度等）
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、结构严谨、分析深入、可视化出色
- 价值: ⭐⭐⭐⭐⭐ 填补了通用推理评测的空白，将成为该领域的重要基准

<!-- RELATED:START -->

## 相关论文

- [Unique Hard Attention: A Tale of Two Sides](unique_hard_attention_a_tale_of_two_sides.md)
- [Hard Negative Mining for Domain-Specific Retrieval in Enterprise Systems](hard_negative_mining_for_domain-specific_retrieval_in_enterprise_systems.md)
- [I Am Big, You Are Little; I Am Right, You Are Wrong](../../ICCV2025/others/i_am_big_you_are_little_i_am_right_you_are_wrong.md)
- [How Hard is it to Explain Preferences Using Few Boolean Attributes?](../../AAAI2026/others/how_hard_is_it_to_explain_preferences_using_few_boolean_attributes.md)
- [From Easy to Hard: Progressive Active Learning Framework for Infrared Small Target Detection with Single Point Supervision](../../ICCV2025/others/from_easy_to_hard_progressive_active_learning_framework_for_infrared_small_targe.md)

<!-- RELATED:END -->
