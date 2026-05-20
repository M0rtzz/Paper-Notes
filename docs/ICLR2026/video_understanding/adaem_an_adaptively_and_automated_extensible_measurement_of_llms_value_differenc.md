---
title: >-
  [论文解读] AdAEM: An Adaptively and Automated Extensible Measurement of LLMs' Value Difference
description: >-
  [ICLR 2026 (Oral)][视频理解][LLM价值观评估] 提出 AdAEM，一个自适应、自扩展的 LLM 价值观评估框架，通过信息论优化自动生成能最大化揭示不同 LLM 价值差异的测试问题，解决现有静态基准无法区分模型价值取向的"信息量不足"问题。
tags:
  - "ICLR 2026 (Oral)"
  - "视频理解"
  - "LLM价值观评估"
  - "动态基准"
  - "信息论优化"
  - "Schwartz价值理论"
  - "文化差异"
---

# AdAEM: An Adaptively and Automated Extensible Measurement of LLMs' Value Difference

**会议**: ICLR 2026 (Oral)  
**arXiv**: [2505.13531](https://arxiv.org/abs/2505.13531)  
**代码**: [https://github.com/ValueCompass/AdAEM](https://github.com/ValueCompass/AdAEM)  
**领域**: 视频理解  
**关键词**: LLM价值观评估, 动态基准, 信息论优化, Schwartz价值理论, 文化差异

## 一句话总结
提出 AdAEM，一个自适应、自扩展的 LLM 价值观评估框架，通过信息论优化自动生成能最大化揭示不同 LLM 价值差异的测试问题，解决现有静态基准无法区分模型价值取向的"信息量不足"问题。

## 研究背景与动机

大语言模型（LLM）虽在知识和指令跟随方面取得巨大进展，但可能生成有害、偏见或非法内容。评估 LLM 内在价值取向已成为全面诊断模型错误对齐、文化适应性和偏见的重要途径。

**现有痛点**：当前价值评估基准面临"信息量不足"挑战——测试问题要么过时、被污染、要么过于通用，只能捕捉到不同 LLM 之间共享的安全价值取向（如 HHH），导致评估结果趋同且无法区分。例如，在现有基准 SVS 和 ValueBench 上，GPT-4 和 GLM-4（分别来自美国和中国）在享乐主义维度展现出几乎相同的偏好，这显然不合理。

**核心矛盾**：静态基准无法与 LLM 的发展同步演化，且无法探索文化差异导致的争议性话题。

**核心 idea**：设计一个自扩展的动态评估框架，通过探测多个来自不同文化和时间段的 LLM 内部价值边界，自动生成能激发价值差异的测试问题，从理论上最大化信息论目标。

## 方法详解

### 整体框架
输入：一组来自不同文化/时期的 LLM 集合 + 初始通用社会话题。输出：一个包含高区分度测试问题的价值评估基准。Pipeline 由两个核心组件构成：信息量优化（exploitation）和探索算法（exploration），形成类 Multi-Armed Bandit 的迭代过程。

### 关键设计

1. **信息量优化（Informativeness Optimization）**：

    - 目标函数基于广义 Jensen-Shannon 散度，最大化不同 LLM 在给定问题上展现的价值分布差异
    - 同时加入"解耦"正则项，防止 LLM 的价值评估结果被问题本身的价值倾向所主导
    - 采用类 EM 算法的迭代优化：E步（Response Generation）采样 LLM 的意见并选择得分最高者；M步（Question Refinement）固定意见优化问题使之更具信息量
    - 每步评估包含四个维度：价值一致性（value conformity）、价值差异（value difference）、语义连贯（semantic coherence）、语义差异（semantic difference）

2. **探索算法（Exploration Algorithm）**：

    - 基于 Multi-Armed Bandit（MAB）变体，自适应决定是继续优化当前话题还是探索新话题
    - 使用 UCB 策略选择最有潜力的话题进行扩展和优化
    - 利用较小、较快的 LLM 集合（P1）做低成本探索，用更强的 LLM 集合（P2）做最终评分
    - 预算 B 控制总探索次数，平衡问题质量与计算成本

3. **评估指标设计**：

    - 基于意见的价值评估：从 LLM 响应中提取多个意见，对每个意见识别 Schwartz 10维价值标签，用逻辑或合并
    - 基于相对排名的聚合：使用 TrueSkill 系统（贝叶斯技能评估）对所有 LLM 进行多方位比较排名，计算胜率作为最终价值评分，比绝对打分更可靠

### 损失函数 / 训练策略
无需训练。所有优化均在 in-context 方式下完成，通过 LLM API 调用实现。核心优化目标为：
$$x^* = \arg\max_x \sum_{i=1}^K \left\{ \alpha_i \text{KL}[p_{\theta_i}(v|x) \| p_M(v|x)] + \frac{\beta}{2} \sum_v |p̂(v|x) - p_{\theta_i}(v|x)| \right\}$$

## 实验关键数据

### 主实验
AdAEM Bench 基于 Schwartz 价值理论的 10 个维度构建，包含 12,310 个测试问题，覆盖 106 个国家。

| 基准 | 问题数 | 平均长度 | Self-BLEU | 相似度 |
|------|--------|---------|-----------|--------|
| SVS | 57 | 13.00 | 52.68 | 0.61 |
| ValueBench | 40 | 15.00 | 26.27 | 0.60 |
| ValueDCG | 4,561 | 11.21 | 13.93 | 0.36 |
| **AdAEM** | **12,310** | **15.11** | **13.42** | **0.44** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 价值启动实验（priming） | 目标价值 +31%, 对立价值 -58% | p < 0.01，验证评估有效性 |
| 同组价值变化 | +17% | 符合 Schwartz 价值结构预测 |
| 可靠性分析 | Cronbach's α = 0.8991 | "良好"可靠性 |
| 人类评估改进 | 合理性 +6.7%, 价值区分度 +31.6% | Cohen's κ = 0.93 |

### 关键发现
- 16 个 LLM 的价值基准测试揭示四个有趣发现：(1) 更先进的 LLM 更偏好安全相关维度（如普世主义）；(2) 同一系列的 LLM 价值取向相似，与模型大小无关；(3) 推理型与聊天型 LLM 价值差异显著；(4) 更大的 LLM 增强特定维度偏好
- AdAEM 在仅数次迭代后就能超越基线基准的信息量得分
- 不同话题类别（技术创新 vs 哲学信仰）下，所有 LLM 展现明显不同的价值取向
- GLM-4（中国开发）和 GPT-4-Turbo（美国开发）在文化相关话题上展现出显著的地域差异

## 亮点与洞察
- 首个将动态评估引入 LLM 价值评估领域的工作，理论驱动的自扩展机制非常优雅
- 信息论目标函数的设计很巧妙，同时兼顾区分度和解耦性
- Multi-Armed Bandit 的探索-利用策略自然且高效
- TrueSkill 评分系统的引入相比传统绝对打分更可靠
- 获得 ICLR 2026 Oral，说明审稿人高度认可
- 跨文化分析揭示了 LLM 训练数据/对齐策略中的文化偏见

## 局限与展望
- 仅基于 Schwartz 价值理论，未覆盖道德基础理论（MFT）、Kohlberg 道德发展阶段等
- 主要关注英语语境，未充分探索多语言和多文化场景
- 由于预算限制，只选取了有限的代表性 LLM
- 自动生成的争议性内容可能被恶意利用
- 价值分类器（GPT-4o）本身可能存在偏见

## 相关工作与启发
- 与 ValueBench、ValueDCG 等静态基准形成互补，引入了动态评估范式
- 受 DyVal 等动态评估工作启发，但首次应用于价值评估
- 与 PromptAgent 等黑盒优化工作相关，但目标函数面向价值区分度
- 启发：该方法可迁移至其他需要动态基准的评估场景，如安全评估、文化适应性测试

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] AnveshanaAI: A Multimodal Platform for Adaptive AI/ML Education through Automated Question Generation and Interactive Assessment](anveshanaai_a_multimodal_platform_for_adaptive_aiml_education_through_automated_.md)
- [\[CVPR 2025\] EDCFlow: Exploring Temporally Dense Difference Maps for Event-based Optical Flow Estimation](../../CVPR2025/video_understanding/edcflow_exploring_temporally_dense_difference_maps_for_event-based_optical_flow_.md)
- [\[ACL 2026\] Full-Duplex-Bench-v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner](../../ACL2026/video_understanding/full-duplex-bench-v2_a_multi-turn_evaluation_framework_for_duplex_dialogue_syste.md)
- [\[ACL 2026\] Distorted or Fabricated? A Survey on Hallucination in Video LLMs](../../ACL2026/video_understanding/distorted_or_fabricated_a_survey_on_hallucination_in_video_llms.md)
- [\[ACL 2026\] Confidence Estimation for LLMs in Multi-turn Interactions](../../ACL2026/video_understanding/confidence_estimation_for_llms_in_multi-turn_interactions.md)

</div>

<!-- RELATED:END -->
