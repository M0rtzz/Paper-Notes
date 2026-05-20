---
title: >-
  [论文解读] Toward a Dynamic Stackelberg Game-Theoretic Framework for Agent-Based Conversational AI Defense Against LLM Jailbreaking
description: >-
  [ICLR 2026][强化学习][game theory] 将 LLM 越狱攻防形式化为动态 Stackelberg 扩展形式博弈，结合快速扩展随机树 (RRT) 搜索提示空间，提出 Purple Agent 防御架构实现"红队思维，蓝队行动"的预见性防御。
tags:
  - "ICLR 2026"
  - "强化学习"
  - "game theory"
  - "Stackelberg game"
  - "jailbreaking defense"
  - "Purple Agent"
  - "RRT"
  - "LLM safety"
---

# Toward a Dynamic Stackelberg Game-Theoretic Framework for Agent-Based Conversational AI Defense Against LLM Jailbreaking

**会议**: ICLR 2026  
**arXiv**: [2507.08207](https://arxiv.org/abs/2507.08207)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: game theory, Stackelberg game, jailbreaking defense, Purple Agent, RRT, LLM safety

## 一句话总结

将 LLM 越狱攻防形式化为动态 Stackelberg 扩展形式博弈，结合快速扩展随机树 (RRT) 搜索提示空间，提出 Purple Agent 防御架构实现"红队思维，蓝队行动"的预见性防御。

## 研究背景与动机

LLM 越狱（jailbreaking）指通过精心构造的提示绕过模型安全机制，诱导生成受限或有害内容。传统防御方法面临根本性挑战：

1. **反应式修补**：基于逐案修补或宽泛内容过滤，无法跟上攻击者的速度和复杂性
2. **静态过滤器**：无法捕捉多轮对话中"偷偷摸摸"的渐进式探测策略
3. **单轮视角**：越狱通常不是一次性事件，而是多轮对话中渐进的战略探测

**核心洞察**：攻防交互本质上是一个序贯博弈——防御者当前轮次的响应决定了攻击者未来的优化空间。因此需要从"启发式防御"转向"基于原理的博弈论框架"。

## 方法详解

### 整体框架

将攻防交互形式化为二人完美信息扩展形式博弈 $\Gamma = (N, A, V, E, x_0, H, o_T, u)$：

- **玩家**：攻击者（跟随者，优化越狱）和防御者（领导者，优化安全）
- **行动**：每轮防御者先承诺响应 $a_{2,t}$，攻击者观察后发出后续提示 $a_{1,t}$
- **终局**：$o_T(h_T) \in \{\text{Jailbreak}, \text{Safe}, \text{Blocked}\}$
- **效用**：越狱时攻击者 +1/防御者 -1，其余情况双方 0

**Stackelberg 范式的关键**：防御者作为领导者需预判攻击者的最优响应后再决策。近视安全（如重定向）可能反而最大化攻击者未来的可达效用。

### 关键设计 1：子博弈完美 Stackelberg 均衡 (SPSE)

通过反向归纳递归定义值函数。防御者在每个历史状态 $h_{t-1}$ 选择最优行动：

$$a_{2,t}^* \in \arg\max_{a_{2,t} \in A_{2,t}} v_{2,t}(h_{t-1} \cup \{a_{2,t}, \text{BR}_{1,t}(a_{2,t})\})$$

其中 $\text{BR}_{1,t}(a_{2,t})$ 是攻击者的最优响应。

### 关键设计 2：局部 ε-均衡

由于全局 SPSE 在无界提示空间中不可计算，引入局部ε均衡条件：

$$\bar{v}_1^{(\tau)}(h_t) \leq v_1^{(\tau)}(h_t) + \varepsilon$$

基于此定义三种体制：

| 体制 | 条件 | 含义 |
|------|------|------|
| I: 防御者错误 | $v_1^{(\tau)} = 1$ | 当前越狱成功，防御者次优 |
| II: 脆弱安全 | $v_1^{(\tau)} = 0, \bar{v}_1^{(\tau)} \leq \varepsilon_{\text{large}}$ | 当前安全但邻域漏洞密集 |
| III: 局部均衡 | $v_1^{(\tau)} = 0, \bar{v}_1^{(\tau)} \leq \varepsilon_{\text{small}}$ | 整个语义邻域被中和 |

Purple Agent 的收敛定义为从体制 I/II 向体制 III 的迭代驱动过程。

### 关键设计 3：RRT 提示空间搜索

将 RRT（快速扩展随机树）从机器人运动规划适配到自然语言提示空间：

1. **采样**：生成候选提示 $p_{\text{rand}}$（如通过角色扮演）
2. **最近邻**：找语义最近的节点 $p_{\text{near}}$
3. **扩展**：合成新提示 $p_{\text{new}}$ 插值
4. **评估**：LLM 黑箱回馈——Safe/Redirect 继续扩展，Reject 剪枝，Jailbreak 终止

RRT 将攻击者建模为进行**结构化、反馈驱动的探索**的代理，而非随机模糊测试。

### Purple Agent："红队思维，蓝队行动"

Purple Agent 是混合元推理器，具有两个互补功能：

1. **探索性推理**（Think Red）：使用 RRT 框架模拟攻击者可能生成有害提示的方式
2. **防御干预**（Act Blue）：在检测到攻击可能性时主动部署防御

核心优势：**在攻击实际发生前预测并部署预防性防御**，在高风险聚类周围创建排除区（Regime III）。

### 损失函数

本文为博弈论框架，不涉及训练损失函数。优化目标是最小化 $\varepsilon$，将系统状态从体制 I/II 驱动到体制 III。

## 实验关键数据

### 主实验 — 攻防动态

在 DeepSeek-V3 上跨不同查询预算评估：

| 方法 | 预算 | 仅攻击越狱数 | 防御后成功越狱 | 减少率 |
|------|------|-------------|--------------|--------|
| Baseline RRT | 50 | 17.6±6.8 | 4.2±3.0 | ~76% |
| Baseline RRT | 100 | 34.8±7.0 | 7.2±5.5 | ~79% |
| Baseline RRT | 200 | 54.4±12.5 | 13.3±8.8 | ~76% |
| Reward-Guided RRT | 50 | 17.0±2.8 | 5.0±1.1 | ~71% |
| Reward-Guided RRT | 100 | 46.4±9.3 | 17.7±5.9 | ~62% |
| Reward-Guided RRT | 200 | 79.0±17.4 | 39.4±10.5 | ~50% |

**关键发现**：200 轮预算下 Reward-Guided RRT 的越狱数从 79.0 降至 39.4（约 50%），且仅触发约 9.6 次模拟阻断——防御高度精准。

### 跨模型泛化

| 模型 | 方法 | 仅攻击 | 防御后 | 减少率 |
|------|------|--------|--------|--------|
| DeepSeek-V3 | RG-RRT | 46.4 | 17.7 | ~62% |
| Llama-3.1-70B | RG-RRT | 33.8 | 27.2 | ~20% |
| Qwen-Plus | RG-RRT | 31.0 | 18.0 | ~42% |
| Gemini-2.5-Flash | RG-RRT | 36.0 | 23.4 | ~35% |

### 语义结构分析（t-SNE 可视化）

| 状态 | 观察 | 含义 |
|------|------|------|
| 仅攻击 | 越狱形成密集聚类 | 脆弱安全 (Regime II)，邻域充满漏洞 |
| Purple Agent | 越狱变为稀疏孤立点 | 鲁棒局部均衡 (Regime III)，排除区有效 |

### 关键发现

1. **"脆弱安全"边界是对齐 LLM 的基本拓扑特征**：跨平台共享的弱点，攻击者可利用
2. **Purple Agent 无需模型特定微调即展现鲁棒迁移性**
3. **自主构建排除区是模型无关策略**，有效缩小对抗攻击面
4. 从密集越狱聚类到孤立点的转变作为均衡的**几何证书**

## 亮点与洞察

1. **将越狱攻防从分类问题提升为序贯决策过程**：这一视角转换具有根本性意义
2. **RRT 在提示空间的创新应用**：将机器人运动规划算法适配到自然语言空间，优雅且高效
3. **三种体制分类（防御者错误 / 脆弱安全 / 局部均衡）**提供了安全状态的精确刻画
4. **t-SNE 可视化从密集聚类到孤立点**的转变直观地证明了防御效果
5. **"近视安全"的例子**：重定向策略虽然避免了即时失败，但延长了博弈范围，允许攻击者利用先前上下文

## 局限性

1. **计算可扩展性**：RRT 搜索在每次防御中的开销未充分讨论
2. **Reward-Guided RRT 下的防御效力下降**：面对最强攻击者（200 轮 RG-RRT）只能减少 ~50% 越狱
3. **Llama-3.1-70B 上效果较弱**（仅 ~20% 减少）：对某些模型的迁移性有限
4. **未考虑真实多代理场景**：当前仅为双人博弈
5. **理论框架与实际部署间的差距**：需要进一步工程化才能应用于生产环境
6. **仅用 5 次独立运行**取平均，标准差较大

## 相关工作与启发

- **PAIR** (Chao et al., 2025)：黑箱越狱的代表性方法
- **Tree of Attacks** (Mehrotra et al., 2024)：自动化越狱攻击
- **SmoothLLM** (Robey et al., 2023)：基于扰动的防御
- **Stackelberg 博弈** (Başar & Olsder, 1998)：经典博弈论框架

核心启发：**LLM 安全不应被视为静态分类问题，而是需要在序贯博弈的框架下通过预见性推理来实现**。Purple Agent 的"红队思维,蓝队行动"范式为防御提供了一条超越反应式修补的道路。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 将 Stackelberg 博弈论与 RRT 搜索创新结合，Purple Agent 概念新颖
- 实验充分度: ⭐⭐⭐ — 跨 4 个模型测试，但每次仅 5 轮运行，缺乏与其他防御方法的直接对比
- 写作质量: ⭐⭐⭐⭐ — 数学形式化清晰，配图优秀
- 价值: ⭐⭐⭐⭐ — 为 LLM 安全提供了新的理论范式和实用防御架构

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] GraphOmni: A Comprehensive and Extensible Benchmark Framework for Large Language Models on Graph-theoretic Tasks](graphomni_a_comprehensive_and_extensible_benchmark_framework_for_large_language_.md)
- [\[AAAI 2026\] A Multi-Agent Conversational Bandit Approach to Online Evaluation and Selection of User-Aligned LLM Responses](../../AAAI2026/reinforcement_learning/a_multi-agent_conversational_bandit_approach_to_online_evaluation_and_selection_.md)
- [\[ICLR 2026\] Robust Deep Reinforcement Learning against Adversarial Behavior Manipulation](robust_deep_reinforcement_learning_against_adversarial_behavior_manipulation.md)
- [\[AAAI 2026\] Distilling Deep Reinforcement Learning into Interpretable Fuzzy Rules: An Explainable AI Framework](../../AAAI2026/reinforcement_learning/distilling_deep_reinforcement_learning_into_interpretable_fuzzy_rules_an_explain.md)
- [\[ICLR 2026\] Learning to Play Multi-Follower Bayesian Stackelberg Games](learning_to_play_multi-follower_bayesian_stackelberg_games.md)

</div>

<!-- RELATED:END -->
