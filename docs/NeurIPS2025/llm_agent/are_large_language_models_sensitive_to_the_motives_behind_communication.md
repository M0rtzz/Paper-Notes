---
title: >-
  [论文解读] Are Large Language Models Sensitive to the Motives Behind Communication?
description: >-
  [NeurIPS 2025][LLM Agent][motivational vigilance] 通过三个递进实验系统评估LLM是否具备"动机警觉性"——识别信息源的意图和激励并相应调整信任度的能力：在控制实验中前沿非推理LLM表现接近理性模型(Pearson's $r>0.9$)且比理性模型更像人类，但在真实YouTube赞助广告场景中警觉性大幅下降($r<0.2$)，简单的prompt steering可部分恢复($r$提升至0.31)。
tags:
  - NeurIPS 2025
  - LLM Agent
  - motivational vigilance
  - 动机警觉
  - 策略沟通
  - 信息可信度
  - 理性模型
---

# Are Large Language Models Sensitive to the Motives Behind Communication?

**会议**: NeurIPS 2025  
**arXiv**: [2510.19687](https://arxiv.org/abs/2510.19687)  
**代码**: 无  
**领域**: LLM Agent / 社会认知  
**关键词**: motivational vigilance, 动机警觉, 策略沟通, 信息可信度, 理性模型

## 一句话总结

通过三个递进实验系统评估LLM是否具备"动机警觉性"——识别信息源的意图和激励并相应调整信任度的能力：在控制实验中前沿非推理LLM表现接近理性模型(Pearson's $r>0.9$)且比理性模型更像人类，但在真实YouTube赞助广告场景中警觉性大幅下降($r<0.2$)，简单的prompt steering可部分恢复($r$提升至0.31)。

## 研究背景与动机

**领域现状**：LLM increasingly作为Agent代替用户在真实世界中行动——处理邮件、浏览网页、做购买决策。这些场景中的信息天然带有发送者的动机——销售员的推荐、赞助商的广告、竞争者的评价。人类通过"认知警觉(epistemic vigilance)"自然地识别和折扣带有偏见动机的信息。

**现有痛点**：LLM已知存在多种警觉性缺失：(1) 越狱攻击(jailbreaking)——恶意指令绕过安全机制；(2) 谄媚(sycophancy)——盲目迎合用户的错误信念；(3) 弹窗/干扰内容劫持Agent行为。这些问题的共同根源是LLM缺乏对信息源动机的深层理解。但目前缺乏系统性的评估框架。

**核心矛盾**：LLM的训练范式优先考虑遵循指令和用户满意度，而非监控信息源的激励和真实性——但后者恰是Agent在真实世界中可靠行动的关键。

**本文目标** LLM究竟在多大程度上具备动机警觉性？在控制环境vs真实环境中表现如何？能否通过简单干预改善？

**切入角度**：使用认知科学中的理性模型(Oktar et al., 2024/2025)作为规范性基准，从三个递进难度的实验范式评估LLM警觉。

**核心 idea**：LLM在简单控制场景中展现出接近理性的动机警觉，但在包含丰富上下文的真实场景中，额外信息反而分散了对动机相关线索的注意力。

## 方法详解

### 整体框架

三个递进实验：(1) 基础能力——LLM能否区分故意沟通和偶然信息？(2) 精细校准——LLM能否根据说话者的善意度和激励精确调整信任？(3) 真实泛化——LLM能否在YouTube赞助广告中保持警觉？每个实验都以Oktar et al.的理性贝叶斯模型为规范基准。

### 关键设计

1. **实验1：故意沟通 vs 偶然观察的区分**:
    - 功能：测试LLM能否因信息来源的不同（来自有动机的"建议"vs偶然"窥见"）调整自身判断
    - 核心思路：改编自Watson & Morgan (2024)的两人判断任务——Player 2收到Player 1的刻意建议或偷看到的真实答案。合作/竞争支付结构影响信息可信度。LLM作为Player 2判断是否及多大程度上更新自己的答案
    - 设计动机：这是动机警觉的最基本前提——能否区分信息是否带有strategic intent

2. **实验2：基于理性模型的精细警觉校准**:
    - 功能：测试LLM能否根据两个关键因素（善意度λ和激励$R_S$）精确调整对推荐的信任
    - 核心思路：使用Oktar et al.的理性贝叶斯模型作为规范基准：说话者选择utterance的概率 $P_S(u) \propto \exp\{\beta_S \cdot \sum R_{Joint} \pi_L(a|u)\}$，其中$R_{Joint} = \lambda R_L + (1-\lambda)R_S$。警觉听者推断产品质量 $P_L(R_L|u) \propto P_S(u|...) P(R_S) P(R_L) P(\lambda)$。在金融/房地产/医疗三领域测16种说话者-激励组合
    - 设计动机：提供定量基准——不仅看LLM是否警觉，还测其警觉程度是否理性

3. **实验3：YouTube赞助广告中的生态有效性**:
    - 功能：在300个真实YouTube赞助片段上测试LLM警觉性能否泛化
    - 核心思路：从SponsorBlock获取赞助时间戳，抓取视频元数据和字幕，遮盖品牌名避免先验知识干扰。让LLM分别估计产品质量($P(R_L|u)$)、赞助收益($R_S$)和频道可信度(λ)。对比LLM推断与理性模型预测的相关性。并测试"警觉性prompt steering"——在prompt中提醒关注说话者的激励和意图
    - 设计动机：控制实验无法代表真实部署场景——真实赞助广告包含大量干扰信息，是真正的试金石

## 实验关键数据

### 主实验

| 模型 | 控制场景 Bayesian-LLM $r$ | 控制场景 LLM-Human $r$ | 真实场景 $r$ | +Steering $r$ |
|------|------------------------|---------------------|-------------|--------------|
| GPT-4o | 0.911 | **0.943** | 0.024~0.121 | **0.137~0.312** |
| Claude 3.5 Sonnet | 0.845 | **0.941** | 0.033~0.190 | **0.200~0.283** |
| Llama 3.3-70B | 0.876 | 0.922 | -0.011~0.098 | **0.029~0.152** |
| o1 | 0.705 | 0.861 | - | - |
| DeepSeek-R1 | 0.326 | 0.643 | - | - |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 非推理vs推理模型 | Bayesian相关 | 非推理(0.8-0.9) >> 推理(0.3-0.7) |
| 直接输出vs CoT | 影响倍率 | CoT使LLM更容易被Player 1影响(>60%) |
| Agent角色vs助手角色 | DeepSeek $r$ | Agent: 0.793 → 助手: -0.141 (完全丧失警觉) |
| 小模型(3B/8B) | Bayesian $r$ | 0.29~0.61——规模决定警觉能力 |
| 默认prompt vs 警觉steering | 真实场景 $r$ | Steering平均提升0.1-0.2，多数显著(p<.05) |

### 关键发现
- 前沿非推理LLM在控制场景中比理性模型更像人类($r_{LLM-Human} > r_{Bayesian-Human}$)——暗示LLM捕获了人类警觉中超理性的启发式偏差
- 推理模型(o1/o3-mini/DeepSeek-R1)在警觉任务上反而更差——推理步骤可能干扰了直觉性的动机判断
- CoT使LLM更容易被社交信息影响(与人类偏差方向相反)——CoT增强信任而非增强怀疑
- 真实场景中警觉性崩溃的主要原因是额外上下文信息分散了对动机线索的注意力
- 简单的prompt steering(提醒关注动机)可部分恢复警觉——说明能力存在但默认未激活

## 亮点与洞察
- 首次使用认知科学理性模型系统评估LLM的动机警觉能力
- 控制实验vs真实场景的巨大落差（0.9→0.02）揭示了实验室评估的局限性
- "推理模型警觉性更差"的反直觉发现对推理LLM作为Agent的部署有重要警示
- prompt steering的有效性暗示一条低成本的改进路径

## 局限与展望
- YouTube赞助广告是单一场景，其他形式的motivated communication（假新闻、钓鱼邮件等）未测试
- 理性模型本身是否足够作为规范性基准有争议——人类也并非完全理性
- Steering prompt需要人工设计，难以适用于所有motivated communication场景
- 仅评估英语场景，跨语言/文化的动机理解可能有差异

## 相关工作与启发
- **vs 越狱研究**: 越狱测试LLM对恶意指令的抵抗；本文测试对motivated信息的理性折扣——后者更贴近Agent部署场景
- **vs Theory of Mind研究**: ToM是警觉的前提之一(需理解他人的信念/意图)；本文聚焦于警觉如何将ToM信息转化为校准的信念更新
- **vs 谄媚(sycophancy)研究**: 谄媚是缺乏警觉的症状之一——不质疑用户错误信念；本文提供了量化相关能力的框架

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将认知科学的动机警觉框架引入LLM评估
- 实验充分度: ⭐⭐⭐⭐ 三个递进实验＋多模型＋人类对比＋缓解策略
- 写作质量: ⭐⭐⭐⭐ 跨学科写作清晰，理性模型介绍充分
- 价值: ⭐⭐⭐⭐⭐ 对LLM Agent部署安全有重要启示，揭示控制实验无法代替真实场景评估

<!-- RELATED:START -->

## 相关论文

- [Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?](debate_or_vote_which_yields_better_decisions_in_multi-agent_large_language_model.md)
- [Adaptive Tool Use in Large Language Models with Meta-Cognition Trigger](../../ACL2025/llm_agent/meco_metacognition_tool_use.md)
- [Zero-Shot Large Language Model Agents for Fully Automated Radiotherapy Treatment Planning](zero-shot_large_language_model_agents_for_fully_automated_radiotherapy_treatment.md)
- [ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use](../../ACL2025/llm_agent/toolhop_multi_hop_tool_use.md)
- [AgentTTS: Large Language Model Agent for Test-time Compute-optimal Scaling Strategy in Complex Tasks](agenttts_large_language_model_agent_for_testtime_computeopti.md)

<!-- RELATED:END -->
