---
title: >-
  [论文解读] Robust and Diverse Multi-Agent Learning via Rational Policy Gradient
description: >-
  [NeurIPS 2025][强化学习][多智能体强化学习] 本文提出理性保持策略优化（RPO）框架和理性策略梯度（RPG）算法，通过引入操纵者智能体和对手塑造技术，在合作和一般和博弈场景中消除对抗优化导致的自毁行为，同时实现策略鲁棒化和多样化。
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "多智能体强化学习"
  - "对抗优化"
  - "自毁行为"
  - "策略梯度"
  - "合作博弈"
---

# Robust and Diverse Multi-Agent Learning via Rational Policy Gradient

**会议**: NeurIPS 2025  
**arXiv**: [2511.09535](https://arxiv.org/abs/2511.09535)  
**代码**: [GitHub](https://github.com/niklaslauffer/rational-policy-gradient)  
**领域**: 强化学习  
**关键词**: 多智能体强化学习, 对抗优化, 自毁行为, 策略梯度, 合作博弈

## 一句话总结

本文提出理性保持策略优化（RPO）框架和理性策略梯度（RPG）算法，通过引入操纵者智能体和对手塑造技术，在合作和一般和博弈场景中消除对抗优化导致的自毁行为，同时实现策略鲁棒化和多样化。

## 研究背景与动机

在多智能体强化学习（MARL）中，训练鲁棒行为是长期挑战：智能体应能适应其他智能体可能采取的各种策略。

**零和博弈中对抗优化的成功**：在零和设定中，自博弈（self-play）天然鼓励智能体不断寻找并修复对手策略中的弱点。然而在合作或一般和博弈中，自博弈会主动避免暴露队友的弱点（因为这会损害共享奖励），导致脆弱的策略。

**自毁行为问题**：如果在合作场景中直接应用对抗优化（即激励最小化他人奖励），对手可以通过拒绝合作甚至主动破坏来轻松达到目标。例如，在一个简单的合作矩阵博弈中，对手只需选择动作 $E$（无论受害者选什么都得到 -1 的奖励），这完全不合理但满足对抗目标。

**现有方法的失败**：CoMeDi 等方法试图通过混合自博弈和交叉博弈的观测分布来防止自毁，但在某些博弈结构中仍然失败——即使只有单一观测的博弈中自毁仍然发生，因为问题根源不在观测分布。

核心洞察：**自毁行为本质上是非理性行为**——智能体做出了违背自身利益的选择。因此解决方案应当直接约束智能体保持理性：其策略必须至少是某个可能队友策略的最佳回应。

## 方法详解

### 整体框架

RPG 的核心架构引入"操纵者"（manipulator）智能体：对于每个原始的"基础"智能体，创建一个对应的操纵者。基础智能体只在操纵者环境中训练最大化自身奖励（确保理性），而操纵者通过对手塑造（opponent shaping）间接引导基础智能体的学习方向，使其朝对抗目标优化。训练完成后丢弃操纵者，保留训练后的基础智能体。

### 关键设计

1. **理性保持策略优化（RPO）形式化**：
   对于每个智能体 $i$，给定对抗目标 $O_i(\pi_1, \dots, \pi_m)$，RPO 要求：
    $\max_{\pi_i} O_i(\pi_1, \dots, \pi_m) \quad \text{s.t.} \quad \exists \pi'_{-i} \in \Pi_{-i} \text{ s.t. } \pi_i \in \text{BR}(\pi'_{-i})$
   
   即：优化对抗目标的同时，策略必须至少是某个共同策略的最佳回应。这保证了智能体的选择是"合理的"——存在某个队友策略使其行为是最优的。
   
   关键性质：在零和博弈中，RPO 约束自动满足（因为最小化对手奖励 = 最大化自己奖励），所以 RPO 是对抗训练在零和设定中的严格泛化。

2. **操纵者与对手塑造机制**：
   将目标分解为两部分：
    - **基础智能体**目标：$\max_{\pi_i} U(\pi_i, \pi^M_{-i})$（对操纵者做最佳回应，保证理性）
    - **操纵者**目标：$\max_{\pi^M_{-i}} O_i(\pi_1, \dots, \pi_m)$（通过影响基础智能体的学习来优化对抗目标）
   
   操纵者的梯度更新涉及**高阶梯度**——通过基础智能体的参数更新步取梯度：
    $\theta^M_{-i} \leftarrow \theta^M_{-i} + \nabla_{\theta^M_{-i}} O_i(\theta'_1, \dots, \theta'_m)$
   其中 $\theta'_i$ 是基础智能体更新后的参数。

3. **Partner-play 正则化**：
   为防止基础智能体（仅在操纵者环境中训练）在与其他基础智能体评估时出现分布偏移，在训练数据中加入少量基础智能体间的交叉博弈轨迹，用小系数 $\epsilon$ 加权作为辅助损失。

4. **Loaded DiCE 损失**：
   传统 RL 代理损失仅支持一阶梯度，操纵者需要计算高阶梯度。使用 Loaded DiCE（基于 magic box 算子  ）定义代理损失，支持通过自动微分计算无偏高阶梯度。

5. **五类 RPG 算法**：

    - **AP-RPG**：寻找预训练策略中的理性对抗样本
    - **AT-RPG**：鲁棒化学习中的智能体
    - **PAIRED-RPG**：基于遗憾最小化的鲁棒化
    - **PAIRED-A-RPG**：最大化受害者遗憾的对抗攻击
    - **AD-RPG**：学习真正多样化的策略集合 + 生成自动课程

### 损失函数 / 训练策略

操纵者损失（Loaded DiCE）：
$$\mathcal{L}^{O_i} = \sum_{e \in E} w_e \sum_t \gamma^t \square(\{a^{t' \leq t}_{j \in \{B,M\}}\}) r^t_e$$

基础智能体损失（标准 RL 代理损失 + partner-play 正则化）：
$$\mathcal{L} = \sum_{e \in E} w_e \sum_t \gamma^t \log(\pi(a_t|s_t)) r^t_e$$

算法每轮先进行 $N$ 步前瞻更新基础智能体（对操纵者做最佳回应），然后在更新后的基础智能体间做轨迹采样，最后更新操纵者参数。

## 实验关键数据

### 主实验：多样性（AD-RPG vs. 基线，Overcooked cramped room）

| 算法 | 自博弈奖励 | 交叉博弈奖励 | 自毁行为？ |
|------|:---:|:---:|:---:|
| CoMeDi | 220 | 2 | 是（站在盘子分发器前阻挡队友） |
| AD（原始对抗多样性） | 240 | 1.25 | 是 |
| **AD-RPG** | **240** | **240** | **否** |

AD-RPG 在自博弈中达到与基线相同的高奖励，但交叉博弈奖励也保持高水平，证明 cramped room 布局实际上几乎不存在真正的多样性——低交叉博弈分数只能通过自毁实现。

### 鲁棒性评估（对抗攻击下的受害者表现，STORM 环境）

| 受害者训练算法 | 训练奖励 | AP攻击 | PAIRED-A-RPG攻击 | AP-RPG攻击 |
|--------------|:---:|:---:|:---:|:---:|
| PAIRED | 0.13 | 0.0 | 0.50 | 0.42 |
| PAIRED-RPG | 0.93 | 0.0 | 0.84 | 0.85 |
| AT（自毁失败） | 0.0 | 0.0 | 0.0 | 0.0 |
| AT-RPG | 0.65 | 0.0 | 0.72 | 0.88 |
| AD-RPG | 0.98 | 0.0 | 0.25 | 0.96 |
| Self-play | 0.98 | 0.0 | 0.16 | 0.96 |

RPG 系列算法训练的策略在面对理性对抗攻击时保持高奖励，而非 RPG 方法因训练时的自毁而完全失败。

### 消融实验 / 跨环境泛化

| 环境 | SP (低熵) | SP (高熵) | AD | AD-RPG |
|------|:---:|:---:|:---:|:---:|
| Forced Coordination | 低交叉博弈 | 低交叉博弈 | 自毁 | **高交叉博弈** |
| Counter Circuit | 低 | 低 | 自毁 | **显著更高** |
| Hanabi(3) | 中 | 较高 | 自毁 | **最高** |
| Hanabi(4) | 中 | 较高 | 自毁 | **最高** |

AD-RPG 在所有环境中都避免了自毁并保持高鲁棒性。

### 关键发现

- AD-RPG 完全消除了现有对抗多样性算法中的自毁行为——这是该领域一个悬而未决的开放问题
- RPG 找到的对抗样本是"理性"的——如 Overcooked 中发现受害者假设智能体按顺时针绕行，对手按逆时针移动（合理但不兼容）
- PAIRED-RPG 和 AT-RPG 训练的策略在对抗攻击下表现最佳
- Partner-play 正则化对防止分布偏移至关重要

## 亮点与洞察

- **理性约束的优雅形式化**：用"存在某个队友策略使我是最佳回应"来定义理性，简洁有力且避免了自毁问题的根源
- **操纵者架构的巧妙设计**：基础智能体只需关注理性（最大化自身奖励），对抗目标的复杂性完全由操纵者承担
- **零和设定的严格泛化**：在零和博弈中 RPO 自动退化为标准对抗训练，保证了方法的通用性
- **一个框架统一五种算法**：AP-RPG、AT-RPG、PAIRED-RPG、PAIRED-A-RPG、AD-RPG 均为 RPG 的实例化

## 局限与展望

- 高阶梯度计算引入额外开销（AD-RPG 比 AD 慢约 6 倍），需要较大 batch size 来稳定估计
- 尚无形式化的收敛性保证——不确定 RPG 在何种条件下一定能找到 RPO 的解
- 操纵者和基础智能体在高维问题中的梯度估计方差可能很大
- 目前仅在相对简单的环境（矩阵博弈、Overcooked、STORM、简化 Hanabi）中验证，尚需更复杂领域的测试

## 相关工作与启发

- **对手塑造（Foerster et al.）**：RPG 是首次将对手塑造应用于对抗训练的工作
- **CoMeDi（Sarkar et al.）**：通过混合观测分布防自毁，但在理论和实验上均被证明不够
- **PAIRED（Dennis et al.）**：RPG 将其从环境设计扩展到合作者设计
- 启发：或许可以结合无梯度的对手塑造方法（如 M-FOS）来降低 RPG 计算开销

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 理性约束框架完全解决了自毁问题这一核心挑战，理论贡献显著
- **实验充分度**: ⭐⭐⭐⭐ 四种环境 + 五种算法变体 + 鲁棒性/多样性/对抗样本三个维度的全面验证
- **写作质量**: ⭐⭐⭐⭐⭐ 矩阵博弈示例直观，理论压缩凝练，图形语言设计精巧
- **价值**: ⭐⭐⭐⭐⭐ 解决了多智能体对抗优化中持续多年的核心开放问题，具有深远影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] On the Global Optimality of Policy Gradient Methods in General Utility Reinforcement Learning](on_the_global_optimality_of_policy_gradient_methods_in_general_utility_reinforce.md)
- [\[NeurIPS 2025\] Sequential Multi-Agent Dynamic Algorithm Configuration](sequential_multi-agent_dynamic_algorithm_configuration.md)
- [\[NeurIPS 2025\] Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning](improving_retrieval-augmented_generation_through_multi-agent_reinforcement_learn.md)
- [\[NeurIPS 2025\] Extending NGU to Multi-Agent RL: A Preliminary Study](extending_ngu_to_multi-agent_rl_a_preliminary_study.md)
- [\[NeurIPS 2025\] Mean-Field Sampling for Cooperative Multi-Agent Reinforcement Learning](mean-field_sampling_for_cooperative_multi-agent_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
