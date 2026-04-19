---
title: >-
  [论文解读] MTL-KD: Multi-Task Learning Via Knowledge Distillation for Generalizable Neural Vehicle Routing Solver
description: >-
  [NeurIPS 2025][车辆路径问题] 提出基于知识蒸馏的多任务学习框架MTL-KD，通过将多个RL单任务教师模型的策略知识蒸馏到一个重解码器学生模型中，实现了对多种VRP变体的高效统一求解，并在大规模问题上展现出卓越的泛化能力。
tags:
  - NeurIPS 2025
  - 车辆路径问题
  - 多任务学习
  - 知识蒸馏
  - 神经组合优化
  - 规模泛化
---

# MTL-KD: Multi-Task Learning Via Knowledge Distillation for Generalizable Neural Vehicle Routing Solver

**会议**: NeurIPS 2025  
**arXiv**: [2506.02935](https://arxiv.org/abs/2506.02935)  
**代码**: [GitHub](https://github.com/CIAM-Group/MTLKD)  
**领域**: 强化学习  
**关键词**: 车辆路径问题, 多任务学习, 知识蒸馏, 神经组合优化, 规模泛化

## 一句话总结

提出基于知识蒸馏的多任务学习框架MTL-KD，通过将多个RL单任务教师模型的策略知识蒸馏到一个重解码器学生模型中，实现了对多种VRP变体的高效统一求解，并在大规模问题上展现出卓越的泛化能力。

## 研究背景与动机

车辆路径问题（VRP）是一类经典的NP-hard组合优化问题，在物流配送、交通调度等场景中有广泛应用。近年来，神经组合优化（NCO）方法通过神经网络学习求解策略，已成为有前景的VRP求解途径。

**现有方法的两难局面**：

**重编码器-轻解码器（HELD）架构**：大多数多任务统一模型采用此架构，可直接用RL训练，在小规模问题上表现出色。然而，轻解码器在面对大规模问题时，无法从高密度、复杂的节点嵌入中提取足够信息，导致泛化能力大幅下降。

**轻编码器-重解码器架构**：在单任务VRP上展示了出色的规模泛化性能，因为重解码器在迭代解码过程中能重新评估剩余节点间的关系。然而，重解码器的巨大内存和计算需求使得RL训练不可行；而监督学习（SL）又面临多种VRP变体缺乏标注数据的困难。

**核心矛盾**：要实现多任务VRP求解的规模泛化，需要重解码器架构；但重解码器的训练在多任务场景下极其困难——RL训练代价过高，SL缺乏标注数据。

**本文切入角度**：利用知识蒸馏绕过上述训练困难。先用RL分别训练轻量级的单任务教师模型，再通过KL散度蒸馏将策略知识无标签地迁移到重解码器学生模型中。

## 方法详解

### 整体框架

MTL-KD框架分为两个阶段：
- **阶段一**：对每个可见VRP任务，独立训练一个采用POMO结构（重编码器-轻解码器）的单任务教师模型。
- **阶段二**：构建一个多任务重解码器学生模型，利用所有教师模型的输出分布作为监督信号进行知识蒸馏训练。

### 关键设计

1. **教师模型训练**：每个教师模型采用POMO架构（6层编码器+1层解码器），通过策略梯度RL独立训练。损失函数为：

$$\mathcal{L}(\theta^T) = -\mathbb{E}_{\tau \sim \pi_{\theta^T}(\cdot|\mathcal{G})} \left[(r(\tau) - b^T(\mathcal{G})) \log \pi_{\theta^T}(\tau|\mathcal{G})\right]$$

其中$b^T(\mathcal{G})$是多起点轨迹的平均奖励基线，$r$是负路径长度（奖励）。教师模型在规模100的随机实例上训练4000轮。

2. **学生模型架构（多任务重解码器）**：采用1层编码器+6层解码器的架构设计。编码器通过线性层将节点特征（坐标、需求、时间窗等）映射为初始嵌入，再通过Transformer层生成节点嵌入$H^{enc}$。解码器在每一步提取动态特征（剩余载量、当前时间、剩余路程、是否开放路线），与最后访问节点和仓库节点的嵌入组合，通过$L$层Transformer网络更新节点嵌入，并使用padding mask处理不同批次中未访问节点数量不一致的问题。最终选择概率为：

$$\pi(i|s_t) = \text{softmax}(c(h_i^{(L)}, h_q) + M^{pad}_i + M^{feas}_i)$$

其中$M^{pad}$排除填充位置，$M^{feas}$排除不可行动作。

3. **知识蒸馏训练**：学生模型同时处理所有$M$种可见任务的实例。在每个解码步$t$，计算学生分布与对应教师分布间的KL散度作为蒸馏损失：

$$\mathcal{L}_{KD}^{(t)} = \sum_{m=1}^{M} \text{KL}\left(\pi_{\theta^{T_m}}(a_t|s_t, \mathcal{G}) \| \pi_{\theta^S}(a_t|s_t, \mathcal{G})\right)$$

这一设计实现了无标签训练——不需要最优解标注，只需在每个解码步模仿教师的动作分布。

4. **R3C推理策略（Random Reordering Re-Construction）**：在推理阶段，给定一个初始解，先将其分解为子路径，然后随机打乱子路径的外部顺序，再随机采样一段连续片段进行重新优化。与之前的RRC方法相比，R3C通过打乱子路径顺序而非反转子路径，避免了对时间窗约束的违反，同时允许更多随机的子路径组合，有助于跳出局部最优。

### 训练配置

- 教师模型：6层编码器，1层解码器，嵌入维度128，8头注意力，4000轮训练
- 学生模型：1层编码器，6层解码器，嵌入维度128（或96），批大小1500（250×6任务），850轮训练，300轮后每100轮学习率减半

## 实验关键数据

### 主实验：可见任务性能（6种VRP变体）

| 方法 | CVRP n=100 | CVRP n=500 | CVRP n=1k | VRPTW n=100 | VRPTW n=1k |
|------|-----------|-----------|----------|------------|-----------|
| HGS-PyVRP | 15.53* | 62.07* | 119.54* | 24.35* | 166.47* |
| MT-POMO(M+aug8) | 15.79(1.69%) | 67.99(9.54%) | 136.62(14.28%) | 25.61(5.18%) | 229.82(38.06%) |
| MVMoE(M+aug8) | 15.76(1.50%) | 73.61(18.59%) | 176.40(47.57%) | 25.51(4.78%) | 253.35(52.19%) |
| RF-MVMoE(M+aug8) | 15.84(2.01%) | 67.36(8.52%) | 134.85(12.80%) | 26.29(7.98%) | 187.87(12.86%) |
| **MTL-KD(R3C200)** | **15.76(1.48%)** | **63.63(2.51%)** | **122.06(2.10%)** | **25.31(3.93%)** | **181.85(9.24%)** |

### 未见任务零样本泛化性能（10种未见VRP变体）

| 方法 | OVRPL n=500 | OVRPL n=1k | VRPLTW n=500 | VRPLTW n=1k |
|------|-----------|----------|-------------|-----------|
| MT-POMO(M+aug8) | 41.28(18.95%) | 84.40(29.08%) | 116.62(26.95%) | 235.94(34.99%) |
| MVMoE(M+aug8) | 45.58(31.36%) | 116.71(78.48%) | 118.00(28.45%) | 260.07(48.79%) |
| **MTL-KD(R3C200)** | **37.27(7.41%)** | **71.35(9.12%)** | **97.85(6.51%)** | **189.02(8.14%)** |

### 关键发现

1. **规模泛化优势显著**：MTL-KD在大规模问题（n=500, 1000）上优势最为明显。例如在CVRP n=1k上，MTL-KD的gap仅为2.10%，而RF-MVMoE为12.80%，MVMoE高达47.57%。
2. **未见任务泛化强**：在10种从未训练过的VRP变体上，MTL-KD展现出远优于其他多任务方法的零样本泛化能力。在OVRPB n=1k上，MTL-KD甚至超越了传统启发式OR-Tools（-4.48% vs 基准）。
3. **R3C策略有效**：R3C推理策略在所有任务和规模上都带来了一致的性能提升，尤其在大规模问题上效果更佳。
4. **CaDA方法崩溃**：CaDA在大规模问题上出现严重的性能退化（gap超过300%），凸显了轻解码器架构在规模泛化上的根本性缺陷。

## 亮点与洞察

- 巧妙地结合了RL（训练教师）和知识蒸馏（训练学生），避免了直接用RL训练重解码器的巨大计算开销。
- "先分后合"的训练策略：单任务教师专精各自任务，学生通过蒸馏同时吸收多种任务的策略知识。
- R3C推理策略通过子路径重排序增加采样多样性，是一种通用且对各种VRP约束友好的改进。

## 局限与展望

- 需要预训练多个教师模型，整体训练流程较复杂。
- 目前仅在16种VRP变体上验证，更复杂的实际约束组合有待探索。
- 学生模型的编码器仅1层，可能对某些复杂约束的建模能力不足。

## 相关工作与启发

- 与RouteFinder等工作相比，MTL-KD首次将知识蒸馏引入多任务VRP的重解码器训练，开辟了新的训练范式。
- 对于其他需要规模泛化的组合优化问题，类似的"RL教师 → KD学生"框架可能同样有效。

## 评分

- 新颖性: ⭐⭐⭐⭐ 知识蒸馏训练重解码器的思路新颖，但整体框架组合性强
- 实验充分度: ⭐⭐⭐⭐⭐ 16种VRP变体、4种规模、多个基线的全面评估
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图表丰富
- 价值: ⭐⭐⭐⭐ 对多任务VRP求解有实际意义，规模泛化提升显著

<!-- RELATED:START -->

## 相关论文

- [Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning](router-r1_teaching_llms_multi-round_routing_and_aggregation_via_reinforcement_le.md)
- [Continual Knowledge Adaptation for Reinforcement Learning](continual_knowledge_adaptation_for_reinforcement_learning.md)
- [EgoBridge: Domain Adaptation for Generalizable Imitation from Egocentric Human Data](egobridge_domain_adaptation_for_generalizable_imitation_from_egocentric_human_da.md)
- [Thompson Sampling in Function Spaces via Neural Operators](thompson_sampling_in_function_spaces_via_neural_operators.md)
- [Provable Ordering and Continuity in Vision-Language Pretraining for Generalizable Embodied Agents](provable_ordering_and_continuity_in_vision-language_pretraining_for_generalizabl.md)

<!-- RELATED:END -->
