---
title: >-
  [论文解读] WebArbiter: A Principle-Guided Reasoning Process Reward Model for Web Agents
description: >-
  [ICLR2026][LLM Agent][Web Agent] WebArbiter 提出一种推理优先、原则引导的过程奖励模型 (WebPRM)，将奖励建模形式化为文本生成任务，通过推理蒸馏+强化学习的两阶段训练，在 WebPRMBench 上以 7B 模型超越 GPT-5 达 9.1 个百分点。
tags:
  - ICLR2026
  - LLM Agent
  - Web Agent
  - 过程奖励模型
  - 推理优先
  - 原则引导
  - 强化学习
  - 推理蒸馏
---

# WebArbiter: A Principle-Guided Reasoning Process Reward Model for Web Agents

**会议**: ICLR2026  
**arXiv**: [2601.21872](https://arxiv.org/abs/2601.21872)  
**代码**: [WebArbiter Project Page](https://webarbiter.github.io/)  
**领域**: llm_agent  
**关键词**: Web Agent, 过程奖励模型, 推理优先, 原则引导, 强化学习, 推理蒸馏

## 一句话总结
WebArbiter 提出一种推理优先、原则引导的过程奖励模型 (WebPRM)，将奖励建模形式化为文本生成任务，通过推理蒸馏+强化学习的两阶段训练，在 WebPRMBench 上以 7B 模型超越 GPT-5 达 9.1 个百分点。

## 背景与动机
- Web Agent 涉及长视域、多步决策和不可逆动作，需要过程级 (step-level) 监督
- 结果奖励模型 (ORM) 仅提供稀疏且延迟的反馈，可能误判错误轨迹为成功
- 已有 WebPRM 存在明显缺陷：
    - **标量 WebPRM**：将进度压缩为粗粒度分数，缺乏可解释性和弱接地
    - **检查列表 WebPRM**：依赖脆弱的模板匹配，在布局或语义变化下失效
    - **LLM-as-Judge**：成本高、可扩展性差、易产生幻觉
- 核心问题：如何构建一个既可解释又稳健的过程奖励模型，能够抵抗表面相关性并提供可审计的推理链？

## 方法详解

### 1. 问题形式化
Web 导航建模为 POMDP：$\mathcal{E} = (\mathcal{S}, \mathcal{A}, \mathcal{O})$

给定任务指令 $\mathcal{I}$、当前观察 $o_p$、历史动作和推理 $(a_{<p}, c_{<p})$，以及候选动作对 $(a_p^1, c_p^1)$ 和 $(a_p^2, c_p^2)$，WebArbiter 生成结构化论证 $j = (j_1, \ldots, j_L)$，最终产出偏好裁决 $\hat{y}$。

输入紧凑表示：
$$x = (\mathcal{I}, o_p, a_{<p}, c_{<p}, (a_p^1, c_p^1), (a_p^2, c_p^2))$$

自回归生成论证：
$$\pi_\theta(j | x) = \prod_{l=1}^{L} \pi_\theta(j_l | x, j_{<l})$$

### 2. 训练数据构建
基于 WebPRM Collection (Chae et al., 2025)：
- 每个实例包含指令、观察序列和专家标注轨迹
- 正向动作来自专家演示 $A^+$，负向动作来自被拒绝的轨迹 $A^-$
- 转化为成对偏好样本用于训练

### 3. 两阶段训练流程

**总体目标**：
$$\max_{\pi_\theta} \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{Train}}, \hat{y} \sim \pi_\theta(j|x)} [\mathbb{1}(\hat{y} = y)]$$

**Stage 1: 推理蒸馏**
- 使用更强的教师模型 (o3) 生成原则引导的结构化论证
- 论证流程：从指令和状态推导任务特定原则 → 将原则落实到页面 → 比较候选动作 → 输出偏好
- 蒸馏损失：

$$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{K} \sum_{i=1}^{K} \sum_{l=1}^{L_i} \log \pi_\theta(\hat{j}_l^{(i)} | x^{(i)}, \hat{j}_{<l}^{(i)})$$

- 使用 10K 样本进行蒸馏训练

**Stage 2: 强化学习**
- 用可验证奖励对齐裁决与正确性信号
- 奖励函数：$R(x, \hat{y}) \in \{-1, 1\}$（基于裁决是否匹配真值）
- 蒸馏模型作为参考策略 $\pi_{\text{ref}}$

RL 优化目标（使用 GRPO）：
$$\mathcal{L}_{\text{RL}}(\theta) = \max_{\pi_\theta} \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{RL}}, \hat{y} \sim \pi_\theta(j|x)} [R(x, \hat{y})] - \beta \mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

- 使用剩余 20K 样本进行 RL 训练

### 4. 核心创新
- **原则引导**：从用户意图和当前状态动态推导原则（而非使用固定检查列表模板）
- **推理优先**：先生成结构化推理论证，再输出裁决，使判断可审计
- 模型基于 Qwen2.5-3B/7B-Instruct，使用 LoRA 微调

## WebPRMBench 基准

### 数据分布
- 跨 4 个 Web 环境：Mind2Web、WebArena、AssistantBench、WorkArena
- 共 1,150 个步骤级偏好实例（每个包含 1 个正确+4 个被拒绝动作）

### 评估指标
**成对准确率 (Pairwise Acc)**：
$$\text{Acc}_{\text{Pairwise}} = \frac{1}{|\mathcal{D}|} \sum_{(a^+, a^-)} \mathbb{1}[\pi_\theta(a^+) \succ \pi_\theta(a^-)]$$

**Best-of-N 准确率 (BoN Acc)**：更严格，要求正确动作同时优于所有 4 个干扰项：
$$\text{Acc}_{\text{BoN}} = \frac{1}{|\mathcal{D}|} \sum_{i=1}^{|\mathcal{D}|} \prod_{q=1}^{4} \mathbb{1}[\pi_\theta(a_i^+) \succ \pi_\theta(a_i^{-_q})]$$

## 实验关键数据

### WebPRMBench 主要结果

| 模型 | Mind2Web BoN | WebArena BoN | AssistantBench BoN | WorkArena BoN | Avg BoN |
|------|-------------|-------------|-------------------|-------------|---------|
| GPT-4o | 52.62 | 66.67 | 66.67 | 55.19 | 60.29 |
| GPT-5 | 62.39 | 71.64 | 63.33 | 64.62 | 65.50 |
| Claude-3.7-Sonnet | 57.90 | 64.10 | 61.30 | 60.60 | 60.98 |
| DeepSeek-R1 | 57.37 | 60.21 | 56.18 | 63.89 | 59.41 |
| WebShepherd-8B | 73.69 | 43.88 | 30.00 | 25.53 | 43.28 |
| **WebArbiter-7B** | **89.53** | **68.66** | **70.00** | **70.19** | **74.60** |

WebArbiter-7B 以 Avg BoN Acc 超越 GPT-5 达 **9.1 个百分点**，超越前 SOTA WebShepherd-8B 达 **31.32 个百分点**。

### 训练策略消融实验

| 方法 | Mind2Web BoN | WebArena BoN | AssistantBench BoN | WorkArena BoN | Avg BoN |
|------|-------------|-------------|-------------------|-------------|---------|
| Instruct (原始) | 39.18 | 42.79 | 53.33 | 35.85 | 42.78 |
| + Cold Start RL | 86.00 | 35.80 | 33.60 | 37.90 | 48.33 |
| + Cold Start RL + Principles | 88.00 | 46.30 | 48.90 | 51.80 | 58.75 |
| + SFT (无原则) + RL | 94.34 | 41.50 | 40.20 | 44.60 | 55.16 |
| **WebArbiter (SFT+原则+RL)** | **89.53** | **68.66** | **70.00** | **70.19** | **74.60** |

### WebArena-Lite 实际搜索效果
在奖励引导轨迹搜索中，WebArbiter 超越 WebShepherd 最高达 **7.2 个百分点**。

## 核心消融发现

### 1. 冷启动 RL 不稳定
- 直接在 Instruct 模型上做 RL，Mind2Web 上升到 86.00，但其他环境反而下降
- 说明没有推理蒸馏基础的 RL 在跨环境泛化上不稳定

### 2. 原则引导至关重要
- 去除显式原则仅保留推理论证：BoN Acc 从 74.60 降至 55.16（-19.44）
- 原则引导使判断更有根据，抵抗表面相关性

### 3. SFT 是 RL 的必要前提
- 推理蒸馏为 RL 提供稳定的起点，RL 主要起放大器作用
- SFT + RL 的组合效果远超任一单独使用

## 亮点
1. **推理优先范式**：将奖励建模从分数预测转变为可审计的推理生成，极大提升可解释性
2. **原则动态引导**：从任务指令和状态推导原则，而非依赖固定模板，适应性强
3. **跨环境稳健泛化**：仅在 Mind2Web 训练，在 4 个不同环境均达最佳
4. **小模型超大模型**：7B 模型超越 GPT-5 和 DeepSeek-R1
5. **两阶段训练策略**：推理蒸馏 + RL 的组合互补性强

## 局限与展望
- 训练数据仅 30K 且来自 Mind2Web 单一环境，扩展多环境训练数据可能进一步提升
- 当前仅支持成对比较，多候选设置需要进一步验证
- 基于文本的观察表示（accessibility tree），未利用视觉信息
- 推理生成增加了推理延迟，实时部署场景需权衡
- WebPRMBench 的负样本由模型生成，可能存在分布偏差

## 与相关工作的对比
- 相比 WebShepherd（检查列表 WebPRM）：WebArbiter 在新环境上完全碾压（WorkArena BoN 70.19 vs 25.53）
- 相比标量 WebPRM（Miao et al., 2025）：提供可审计的推理链而非数值分数
- 相比 LLM-as-Judge：7B 专用模型远超通用 GPT-5
- 相比 Reasoning RM 文献（Chen et al., 2025）：首次将推理 RM 应用于 Web Agent 领域

## 启发与关联
- 原则引导的推理蒸馏范式可推广到其他过程奖励建模场景
- 两阶段 SFT → RL 流程对训练可验证奖励模型有参考价值
- WebPRMBench 提供了标准化的 WebPRM 评估框架
- 推理优先的奖励模型可与搜索/规划算法结合，实现推理时扩展

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (推理优先+原则引导的WebPRM设计全新，两阶段训练策略创新)
- 实验充分度: ⭐⭐⭐⭐⭐ (4环境benchmark、多类型baseline、详细消融、实际搜索验证)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，但符号较多，需要仔细阅读)
- 价值: ⭐⭐⭐⭐⭐ (7B超越GPT-5，开源WebPRMBench，对Web Agent领域贡献巨大)

<!-- RELATED:START -->

## 相关论文

- [Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning in Web Agents](web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_in_web_agents.md)
- [Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents](web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)
- [ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents](st-webagentbench_a_benchmark_for_evaluating_safety_and_trustworthiness_in_web_ag.md)
- [ProBench: Benchmarking GUI Agents with Accurate Process Information](../../AAAI2026/llm_agent/probench_benchmarking_gui_agents_with_accurate_process_infor.md)
- [Web-Shepherd: Advancing PRMs for Reinforcing Web Agents](../../NeurIPS2025/llm_agent/web-shepherd_advancing_prms_for_reinforcing_web_agents.md)

<!-- RELATED:END -->
