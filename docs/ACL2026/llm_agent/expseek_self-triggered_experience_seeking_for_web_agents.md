---
title: >-
  [论文解读] ExpSeek: Self-Triggered Experience Seeking for Web Agents
description: >-
  [ACL 2026][LLM Agent][Web Agent] ExpSeek 提出了一种基于步级熵自触发的经验主动寻求框架，让 Web Agent 在交互过程中根据自身信号判断何时需要指导、获取什么指导，在 Qwen3-8B/32B 上分别实现 9.3% 和 7.5% 的绝对提升。
tags:
  - ACL 2026
  - LLM Agent
  - Web Agent
  - 经验干预
  - 熵触发
  - 主动寻求指导
  - 多轮交互
---

# ExpSeek: Self-Triggered Experience Seeking for Web Agents

**会议**: ACL 2026  
**arXiv**: [2601.08605](https://arxiv.org/abs/2601.08605)  
**代码**: [https://github.com/WYRipple/ExpSeek](https://github.com/WYRipple/ExpSeek)  
**领域**: LLM Agent  
**关键词**: Web Agent, 经验干预, 熵触发, 主动寻求指导, 多轮交互

## 一句话总结

ExpSeek 提出了一种基于步级熵自触发的经验主动寻求框架，让 Web Agent 在交互过程中根据自身信号判断何时需要指导、获取什么指导，在 Qwen3-8B/32B 上分别实现 9.3% 和 7.5% 的绝对提升。

## 研究背景与动机

**领域现状**：Web Agent 需要在开放网络中进行多轮交互获取信息并回答复杂查询。经验干预（experience intervention）已被证明是提升 agent 能力的有效范式，现有方法主要分为离线经验提炼和在线自演化两条路线。

**现有痛点**：现有经验注入方式是被动的——在任务开始前将经验作为全局上下文一次性注入系统提示。然而在 agent 与环境的多轮交互中，上下文观测持续变化，初始注入的静态经验难以适应动态场景，可能导致决策偏差。

**核心矛盾**：经验的有效性依赖于时机和内容的精准匹配：过于频繁的干预增加推理负担，过于稀疏则错失关键指导窗口；全局经验无法针对当前步骤的具体状态提供定制化指导。

**本文目标**：构建一种主动经验寻求框架，解决两个核心问题——(1) 何时寻求经验（when）：利用模型自身信号判断干预时机；(2) 寻求什么经验（what）：设计步级定制化经验内容。

**切入角度**：作者观察到 LLM 的步级熵（token entropy 均值）与推理质量存在统计相关性——错误步骤的熵显著高于正确步骤。这种内在信号可以作为 agent "困惑"程度的指示器，无需额外的奖励模型。

**核心 idea**：用模型自身的步级熵作为自触发信号判断干预时机，结合经验库和经验模型动态生成步级定制化指导，实现从被动全局注入到主动步级寻求的范式转变。

## 方法详解

### 整体框架

ExpSeek 包含三个阶段：(1) 经验库构建——从成功/失败轨迹对中提取结构化经验三元组并按主题组织；(2) 熵自触发机制——通过 logistic 回归和 bootstrap 重采样估计过程步和回答步的熵阈值区间；(3) 推理时引导干预——当步级熵超过阈值时，经验模型基于当前上下文检索相关经验并生成定制化指导。

### 关键设计

1. **经验库构建 (Experience Base Construction)**:

    - 功能：从训练轨迹中提炼可复用的引导经验
    - 核心思路：对训练集中每个查询采样 $k$ 条轨迹，配对成功和失败轨迹 $(\tau^+, \tau^-)$。由工具模型逐步分析失败轨迹，对每个错误步骤生成经验三元组：行为描述（Behavior）、错误分析（Mistake）、纠正方向（Guidance，不直接给答案）。最后通过迭代批处理为三元组归纳主题标签，形成按主题组织的经验库 $\mathcal{E}_p$（过程步）和 $\mathcal{E}_a$（回答步）
    - 设计动机：三元组设计模拟了人类从错误中学习的模式，主题组织使检索更高效，按步骤类型分库匹配了过程步和回答步不同的熵分布特征

2. **熵自触发机制 (Entropy as Self-Trigger)**:

    - 功能：利用模型内在信号自动判断何时需要经验干预
    - 核心思路：计算每步的平均 token 熵 $\bar{H}_t = \frac{1}{|R_t|} \sum_{x \in R_t} H(x)$。对过程步和回答步分别拟合 logistic 回归模型 $P(y_t=0|\bar{H}_t) = 1/(1+e^{-(w \cdot \bar{H}_t + b)})$，通过 1000 次 bootstrap 重采样估计 95% 置信区间 $[\theta_{lower}, \theta_{upper}]$ 作为阈值区间。推理时，低于下界不干预，高于上界必定干预，区间内按线性概率干预
    - 设计动机：KS 检验证实正确/错误步骤的熵分布在统计上可区分（过程步 KS=0.1998, 回答步 KS=0.3809, p<0.001）。概率化干预避免了硬阈值的脆弱性，bootstrap 提供了稳健的区间估计

3. **步级引导干预 (Guided Intervention at Inference)**:

    - 功能：在触发时生成与当前上下文匹配的定制化指导
    - 核心思路：当熵触发干预且上一步未干预时，经验模型 $\mathcal{M}_e$ 读取当前步的历史上下文 $h_t$，从对应经验库中选择 3 个最相关主题，基于这些主题下的经验三元组为当前场景动态生成指导 $e_t$。过程步的指导追加在环境观测后，回答步的指导则使 agent 可以继续推理或修正答案
    - 设计动机：生成式指导优于检索式（实验证实检索式效果差很多），因为生成可以根据当前具体语境调适通用经验。一步冷却期（干预后下一步不再干预）防止过度干预

### 损失函数 / 训练策略

ExpSeek 为推理时框架，不涉及训练。经验库构建使用 Qwen3-235B-A22B-Instruct 作为工具模型。Agent 使用 Qwen3-8B/32B，采样温度 1.0，top-p 0.95，最大 30 步 ReAct 交互。

## 实验关键数据

### 主实验

**四个 Web Agent 基准上的准确率（%）**

| 方法 | WebWalkerQA | GAIA | Seal | xbench | Avg. |
|------|-------------|------|------|--------|------|
| **Qwen3-8B** | | | | | |
| No Experience | 38.47 | 29.13 | 23.23 | 25.60 | 32.23 |
| Training-Free GRPO | 40.62 | 29.32 | 25.59 | 26.00 | 33.79 |
| ReasoningBank+ | 40.78 | 32.04 | 26.38 | 28.00 | 34.80 |
| **ExpSeek** | **48.25** | **36.89** | **30.16** | **37.20** | **41.50** |
| **Qwen3-32B** | | | | | |
| No Experience | 45.01 | 36.50 | 27.80 | 27.40 | 37.79 |
| ReasoningBank+ | 45.60 | 33.01 | 29.84 | 36.33 | 39.33 |
| **ExpSeek** | **51.09** | **43.88** | **32.76** | **42.00** | **45.32** |

### 消融实验

| 变体 (8B) | GAIA | xbench |
|-----------|------|--------|
| 仅过程步指导 | 33.01 (+3.9) | 28.40 (+2.8) |
| 仅回答步指导 | 30.29 (+1.2) | 34.80 (+9.2) |
| 完整 ExpSeek | **36.89** (+7.8) | **37.20** (+11.6) |

**触发与指导方式对比 (8B, GAIA)**

| 触发方式 | 指导方式 | Acc. | 平均步数 | 平均时间 |
|----------|----------|------|----------|----------|
| 规则触发 | 经验模型 | 38.81 | 9.52 | 329.71s |
| Claude-4 | 经验模型 | 39.47 | 8.55 | 370.82s |
| **熵触发** | **经验模型** | **36.89** | **5.75** | **127.57s** |
| 熵触发 | 检索嵌入 | 30.92 | 5.54 | 110.61s |

### 关键发现

- 熵触发的效率优势显著：步数仅为规则触发的 60%，时间仅为 39%，同时保持相当的准确率
- 4B 经验模型即可有效指导 32B Agent（GAIA +5.2%, xbench +9.7%），验证了弱模型引导强模型的可行性
- 经验指导使过程步熵增加（促进探索），回答步熵降低（增强收敛），形成"发散-收敛"的行为模式
- 即使每个主题仅保留 1 条经验，性能仍然稳健，说明经验模型能从少量种子经验中泛化

## 亮点与洞察

- 将经验干预从被动的全局注入转变为主动的步级寻求，是范式层面的创新
- 利用模型自身的熵信号作为触发器，无需额外奖励模型，既优雅又实用
- "发散-收敛"的熵行为模式提供了对 ExpSeek 工作机制的直觉解释
- 跨任务泛化能力强：仅用 WebWalkerQA 25% 数据构建经验库，在三个 OOD 基准上仍显著有效

## 局限与展望

- 阈值估计依赖训练集和工具模型对步骤质量的评估，更精确的策略有待探索
- 尚未验证在非 Web 领域和更多工具集上的效果
- 可探索 ExpSeek 作为 Agentic RL 的 rollout 增强技术，提升收敛速度和采样质量

## 相关工作与启发

- 与 ReasoningBank 等离线/在线经验积累方法互补，ExpSeek 关注的是经验的利用方式（时机和内容）
- 熵作为推理质量指示器的成功应用，启发了在其他 agent 场景中利用模型不确定性信号的可能性
- 弱模型指导强模型的成功案例，为实际部署中降低指导成本提供了新思路

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 从被动注入到主动寻求的范式转换，熵自触发机制设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 四个基准、两个模型规模、丰富的消融和分析（效率、缩放律、迁移性、内部机制）
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机充分，实验分析深入

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] SynthAgent: Adapting Web Agents with Synthetic Supervision](synthagent_adapting_web_agents_with_synthetic_supervision.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning in Web Agents](../../ICLR2026/llm_agent/web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents](../../ICLR2026/llm_agent/web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)
- [\[NeurIPS 2025\] Web-Shepherd: Advancing PRMs for Reinforcing Web Agents](../../NeurIPS2025/llm_agent/web-shepherd_advancing_prms_for_reinforcing_web_agents.md)
- [\[ACL 2025\] Explorer: Scaling Exploration-Driven Web Trajectory Synthesis for Multimodal Web Agents](../../ACL2025/llm_agent/explorer_scaling_exploration-driven_web_trajectory_synthesis_for_multimodal_web_.md)

</div>

<!-- RELATED:END -->
