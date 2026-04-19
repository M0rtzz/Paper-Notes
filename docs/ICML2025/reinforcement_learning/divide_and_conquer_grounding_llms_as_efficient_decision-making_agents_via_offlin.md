---
title: >-
  [论文解读] Divide and Conquer: Grounding LLMs as Efficient Decision-Making Agents via Offline Hierarchical Reinforcement Learning
description: >-
  [ICML 2025][层次化强化学习] GLIDER 引入参数高效的层次化结构——高层策略学习抽象的分步计划并指导低层控制器执行，通过离线层次化 RL 将复杂长时域决策分解为连贯的 CoT 推理子任务，在 ScienceWorld 和 ALFWorld 上取得一致的性能提升和更强的泛化能力。
tags:
  - ICML 2025
  - 层次化强化学习
  - 离线RL
  - LLM决策
  - 分治策略
  - 时间抽象
---

# Divide and Conquer: Grounding LLMs as Efficient Decision-Making Agents via Offline Hierarchical Reinforcement Learning

**会议**: ICML 2025  
**arXiv**: [2505.19761](https://arxiv.org/abs/2505.19761)  
**代码**: https://github.com/NJU-RL/GLIDER (有)  
**领域**: 强化学习 / LLM智能体  
**关键词**: 层次化强化学习, 离线RL, LLM决策, 分治策略, 时间抽象

## 一句话总结

GLIDER 引入参数高效的层次化结构——高层策略学习抽象的分步计划并指导低层控制器执行，通过离线层次化 RL 将复杂长时域决策分解为连贯的 CoT 推理子任务，在 ScienceWorld 和 ALFWorld 上取得一致的性能提升和更强的泛化能力。

## 研究背景与动机

**领域现状**：LLM 展现了强大的推理能力，但在长时域（long-horizon）决策任务中表现不佳。这类任务要求智能体执行一系列连贯的动作来完成复杂目标，如科学实验操作（ScienceWorld）或家务任务（ALFWorld）。

**现有痛点**：LLM 作为决策智能体面临两个核心挑战——（1）**探索不足**：在稀疏奖励场景下，LLM 很难通过随机尝试找到正确的动作序列；（2）**长期信用分配（credit assignment）困难**：一个长序列中哪些动作对最终成功贡献最大，LLM 无法有效判断。

**核心矛盾**：LLM 的优势在于高层推理和规划，但直接让 LLM 做底层的逐步决策效率极低——它既要理解任务全局又要处理具体执行细节。如何让 LLM 发挥擅长的高层推理，同时高效地完成底层执行？

**本文目标**：设计一个层次化框架，让 LLM 只负责高层规划（"做什么"），由低层控制器负责具体执行（"怎么做"），两者通过离线 RL 协同训练。

**切入角度**：借鉴分治（divide-and-conquer）思想——将复杂的长时域任务分解为一系列子任务（高层），每个子任务由专门的执行器完成（低层）。高层策略生成抽象计划，低层策略执行具体动作。

**核心 idea**：通过引入参数高效的层次结构到 LLM 策略中，高层策略学习并输出抽象的分步计划（类似 CoT），低层控制器被监督学习这些计划的具体执行方式，从而提供灵活的时间抽象来增强长时域任务的探索和学习。

## 方法详解

### 整体框架

GLIDER（**G**rounding **L**anguage Models as Eff**I**cient **D**ecision-Making Agents via Offline Hi**E**rarchical **R**einforcement Learning）是一个两层结构：

```
任务描述 + 环境状态
       ↓
  [高层策略 (High-Level Policy)]
       ↓ 输出：抽象子目标/分步计划
  [低层控制器 (Low-Level Controller)]
       ↓ 输出：具体动作序列
     环境交互
```

训练分三个阶段：SFT → 离线 RL (ORL) → 在线适应 (O2O)

### 关键设计

1. **高层策略（High-Level Policy）**:

    - **功能**：接收任务描述和当前环境状态，输出抽象的分步计划（step-by-step plan）
    - **核心思路**：高层策略本质上是一个 CoT 推理过程——将复杂任务分解为有序的子目标序列。例如，在"加热水"的任务中，高层可能输出："1.找到容器 → 2.装水 → 3.放到加热器上 → 4.开启加热"
    - **设计动机**：LLM 本身擅长高层推理和规划，通过让高层只做"规划"而不做"执行"，可以充分发挥 LLM 的语言理解和世界知识优势。这种分离也大幅减少了搜索空间

2. **低层控制器（Low-Level Controller）**:

    - **功能**：接收高层生成的子目标，输出对应的具体动作序列来完成该子目标
    - **核心思路**：低层控制器被训练为**与任务无关（task-agnostic）**的执行技能——它学会的是"如何找到东西""如何操作物品"等通用技能，而不是特定任务的解法
    - **设计动机**：task-agnostic 的低层技能有两大好处——（1）可复用性强，不同任务可以共享底层技能；（2）在非平稳环境中能快速适应，因为底层执行逻辑不变，只需改变高层计划

3. **离线层次化 RL 训练（三阶段）**:

    - **SFT 阶段**：用离线数据中的专家轨迹分别对高层和低层进行监督学习，建立基本的规划和执行能力
    - **ORL 阶段**（离线 RL）：使用 AWAC（Advantage Weighted Actor-Critic）等离线 RL 算法进一步优化策略。用收集的交互数据训练，使高层策略学会生成更好的子目标，低层控制器学会更高效的执行
    - **O2O 阶段**（Offline-to-Online）：将离线训练的策略部署到新环境中进行在线适应，利用低层技能的可迁移性快速适应非平稳环境
    - **设计动机**：三阶段训练逐步提升：SFT 建立基础 → ORL 优化策略 → O2O 适应新环境。离线 RL 避免了直接在环境中训练的高成本

4. **时间抽象（Temporal Abstraction）**:

    - **功能**：高层策略不需要每步都做决策，而是以较低频率输出子目标，每个子目标覆盖多个底层步骤
    - **核心思路**：高层每隔 $k$ 步输出一个新子目标（$k$ 是灵活的，取决于子任务的复杂度），低层在此期间持续执行。这种机制类似 Options Framework
    - **设计动机**：时间抽象使得高层的决策空间大幅缩小（从数百步压缩到数个子目标），从而显著改善探索效率和长期信用分配

### 损失函数 / 训练策略

- **SFT 阶段**：标准交叉熵损失，高层学习子目标生成，低层学习动作执行
- **ORL 阶段**：AWAC 风格的加权策略学习，使用 advantage function 加权数据中的"好"样本：

$$\mathcal{L}_{\text{AWAC}} = -\mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ \frac{\exp(A(s,a)/\lambda)}{Z(s)} \log \pi_\theta(a|s) \right]$$

其中 $A(s,a)$ 是 advantage，$\lambda$ 是温度参数

- **参数高效**：使用 LoRA 等参数高效微调技术，避免全参数训练的高开销

## 实验关键数据

### 主实验

**ScienceWorld**（科学实验模拟环境）：

| 方法 | 平均分 | 提升 | 说明 |
|------|--------|------|------|
| SayCan (LLM 直接决策) | 较低 | - | LLM 无层次化 |
| ReAct | 中等 | - | Prompt-based |
| Flat RL (无层次) | 中等 | - | 标准离线 RL |
| **GLIDER** | **最高** | 一致性提升 | 层次化高低层协同 |

**ALFWorld**（家务任务环境）：

| 方法 | 成功率 | 提升 | 说明 |
|------|--------|------|------|
| Flat BC (行为克隆) | 基线 | - | 无层次化 |
| Flat RL | 中等 | - | 离线 RL 无层次 |
| **GLIDER** | **最高** | 持续提升 | 高层规划 + 低层执行 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| GLIDER (完整) | 最佳 | 高层 + 低层 + ORL |
| 仅 SFT (无 ORL) | 次优 | 缺少 RL 优化 |
| 无层次化 (Flat) | 弱 | 单一策略处理全部 |
| 固定高层 (冻结) | 较弱 | 高层不更新限制了适应性 |
| 无 O2O 适应 | 泛化差 | 无法适应新环境 |
| 移除时间抽象 | 探索差 | 高层每步决策，搜索空间爆炸 |

### 关键发现

1. **层次化结构的关键性**：移除层次化结构（Flat baseline）导致性能显著下降，尤其在长时域任务上，证明分治策略对 LLM 决策至关重要
2. **离线 RL 优于纯 SFT**：ORL 阶段在 SFT 基础上进一步提升了策略质量，说明离线 RL 能从次优数据中学到更好的决策
3. **泛化能力**：GLIDER 在 O2O 阶段展现出优秀的迁移能力——低层的 task-agnostic 技能可以直接复用到新环境中，只需微调高层规划
4. **时间抽象的收益**：高层策略以较低频率输出子目标，显著减少了搜索空间和信用分配的困难

## 亮点与洞察

- **层次化设计与 LLM 的天然契合**：LLM 擅长高层推理和规划，让它只做高层决策而不管底层执行，恰好发挥了其优势。这种"分工"比让 LLM 做所有事情更高效
- **CoT 即规划**：高层策略输出的子目标序列本质上就是一种 CoT——把复杂问题分解为有序子步骤。这连接了"LLM 推理"和"RL 决策"两个领域
- **离线训练的实用性**：完全基于离线数据训练 + 轻量在线适应的范式，比在线 RL 更实际

## 局限与展望

1. 实验仅在 ScienceWorld 和 ALFWorld 两个基准上进行，环境复杂度有限
2. 高层子目标的粒度如何自动确定（而非人工设定）是一个开放问题
3. 离线数据的质量和覆盖度对 GLIDER 的性能影响未充分探讨
4. 与 online RL 方法（如 PPO）的直接对比缺失
5. 在更真实的机器人或 Web 任务环境中的验证是重要的下一步

## 相关工作与启发

- **Options Framework / HAM**：经典的层次化 RL 理论，GLIDER 将其思想与 LLM 结合
- **SayCan / Inner Monologue**：LLM 驱动的机器人决策，但不使用 RL 训练
- **ReAct**：Prompt-based 的推理-行动框架，GLIDER 通过 RL 训练进一步优化
- **AWAC**：离线 RL 算法，GLIDER 在此基础上扩展为层次化版本
- **启发**：LLM 作为决策智能体的关键可能不是让它"做一切"，而是让它在正确的抽象层次上工作——做规划而非执行

## 评分
- 新颖性: ⭐⭐⭐⭐ 将层次化 RL 与 LLM 决策有机结合，设计自然但有实质创新
- 实验充分度: ⭐⭐⭐⭐ 两个基准环境 + 详细消融，但环境数量可以更多
- 写作质量: ⭐⭐⭐⭐ 方法阐述清晰，动机论证充分
- 价值: ⭐⭐⭐⭐ 为 LLM-as-Agent 提供了一个合理且有效的层次化 RL 框架

<!-- RELATED:START -->

## 相关论文

- [Transitive RL: Value Learning via Divide and Conquer](../../ICLR2026/reinforcement_learning/transitive_rl_value_learning_via_divide_and_conquer.md)
- [Graph-Assisted Stitching for Offline Hierarchical Reinforcement Learning](graph-assisted_stitching_for_offline_hierarchical_reinforcement_learning.md)
- [Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making](counterfactual_effect_decomposition_in_multi-agent_sequential_decision_making.md)
- [Structured Reinforcement Learning for Combinatorial Decision-Making](../../NeurIPS2025/reinforcement_learning/structured_reinforcement_learning_for_combinatorial_decision-making.md)
- [Structural Information-based Hierarchical Diffusion for Offline Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/structural_information-based_hierarchical_diffusion_for_offline_reinforcement_le.md)

<!-- RELATED:END -->
