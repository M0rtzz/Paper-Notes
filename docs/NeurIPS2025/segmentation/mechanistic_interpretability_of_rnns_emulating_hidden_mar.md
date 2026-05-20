---
title: >-
  [论文解读] Mechanistic Interpretability of RNNs Emulating Hidden Markov Models
description: >-
  [NeurIPS 2025][语义分割][RNN] 训练 vanilla RNN 复现 HMM 的发射统计，然后通过反向工程揭示 RNN 实现离散随机状态转换的机制：噪声驱动的轨道动力学 + "kick 神经元"触发的快速转换，本质是自诱导随机共振（SISR），该动力学基元可组合复用以模拟更复杂的离散潜在结构…
tags:
  - "NeurIPS 2025"
  - "语义分割"
  - "RNN"
  - "HMM"
  - "mechanistic interpretability"
  - "stochastic resonance"
  - "kick neurons"
  - "orbital dynamics"
---

# Mechanistic Interpretability of RNNs Emulating Hidden Markov Models

**会议**: NeurIPS 2025  
**arXiv**: [2510.25674](https://arxiv.org/abs/2510.25674)  
**作者**: Elia Torre, Michele Viscione, Lucas Pompe, Benjamin F. Grewe, Valerio Mante (ETH Zurich / University of Zurich)
**代码**: [https://github.com/EliaTorre/hmmrnn](https://github.com/EliaTorre/hmmrnn)  
**领域**: 图像分割  
**关键词**: RNN, HMM, mechanistic interpretability, stochastic resonance, kick neurons, orbital dynamics

## 一句话总结

训练 vanilla RNN 复现 HMM 的发射统计，然后通过反向工程揭示 RNN 实现离散随机状态转换的机制：噪声驱动的轨道动力学 + "kick 神经元"触发的快速转换，本质是自诱导随机共振（SISR），该动力学基元可组合复用以模拟更复杂的离散潜在结构。

## 研究背景与动机

### 现有痛点

**现有痛点**：**领域现状**：循环神经网络（RNN）在计算神经科学中广泛用于推断神经群体的潜在动力学和生成关于行为的计算假说。然而过往研究主要聚焦于相对简单、输入驱动、大体确定性的任务——对于自然环境下更丰富的、自发的、潜在随机的行为，RNN 如何实现，知之甚少。

隐马尔可夫模型（HMM）可以将自然行为分割为离散潜在状态，状态间存在随机转换。这种离散-随机的动力学与 RNN 的连续状态空间似乎格格不入。核心问题是：**RNN 能否利用连续动力学生成离散状态间的随机转换？如果能，其机制是什么？**

已有的 RNN 反向工程方法主要依赖不动点拓扑分析和局部线性化，适用于确定性任务（如运动辨别、情感分类），但对于噪声驱动的随机行为场景并不适用。本文填补了这一空白。

## 方法详解

### 1. 训练框架：用 RNN 逼近 HMM

**网络架构**：标准 vanilla RNN，隐状态维度 $|h| \in \{50, 150, 200\}$。每个时间步接收高斯噪声输入 $x_t \sim \mathcal{N}(0, I_d)$，$d \in \{1, 10, 100, 200\}$。

$$h_t = \text{ReLU}(h_{t-1} W_{hh}^\top + x_t W_{ih}^\top), \quad y_t = h_t A^\top$$

输出 logits 通过 Gumbel-Softmax 转为类别采样，模拟 HMM 的离散发射。

**损失函数**：采用 Sinkhorn 散度（一种基于最优传输的距离度量），比较 RNN 输出序列与 HMM 目标序列的分布。Sinkhorn 散度通过归一化耦合矩阵实现可微且高效的分布比较。

**HMM 家族**：设计三类目标 HMM 以测试泛化能力：
- **线性链 HMM**（$M \in \{2,3,4,5\}$ 状态）：从最大离散到准连续的谱系
- **全连接 HMM**（3 状态）：任意状态间均可转换
- **循环 HMM**（4 状态）：双向闭环转换

**性能指标**：(i) Sinkhorn 对齐的欧氏距离；(ii) 发射转换矩阵；(iii) 边际观测频率；(iv) 观测波动率。训练后的 RNN 在所有指标上均能复现目标 HMM 的发射统计。

### 2. 全局潜在动力学：噪声维持的轨道动力学

将 RNN 隐状态投影到前两个主成分（PCA），揭示了关键发现：

- **无输入时**：从随机初始化出发，活动收敛到**单一不动点**——没有多吸引子结构
- **有噪声输入时**：轨迹转变为**轨道动力学**——沿封闭轨道单向演化

噪声将活动推离不动点，而循环连接将其拉回，共同形成稳定的闭合轨道。沿轨道，RNN 展现出慢区域（clusters），每个对应 HMM 的一个输出类别，区域间存在快速转换。

**轨道半径与噪声方差线性缩放**：二阶摄动分析表明，在无偏高斯输入下，一阶摄动平均抵消，二阶项（与方差线性相关）主导了转换后的动力学。

**训练过程中轨道动力学的涌现**：训练初期 RNN 学习单一不动点，随后不动点失稳（不稳定特征值出现），最终转变为轨道动力学。这一转变与损失曲线的双重下降现象吻合。

### 3. 局部潜在动力学：Clusters、Transitions 和 Kick-Zones

超越不动点分析，论文通过短 rollout 在状态空间中识别三种功能区域：

**Clusters（驻留时间 >> 8）**：
- 轨迹停留最久的区域
- logit 梯度符号频繁翻转（5-20 次）
- 几乎只有收缩特征值，局部稳定
- 每个 cluster 对应不同的输出概率分布

**Kick-zones（2 ≤ 驻留时间 ≤ 8）**：
- 位于 cluster 下游
- 中等 logit 梯度变化（2-4 次）
- 存在少量不稳定方向，局部拉伸流场
- 触发状态转换的关键区域

**Transitions（驻留时间 < 2）**：
- 跨过 kick-zone 后进入的短暂通道
- 几乎确定性地向下一个 cluster 运动
- logit 梯度变化极少（<1），流场稳定且有方向性

**噪声敏感性验证**：Transition 区域对噪声条件几乎不敏感——一旦跨过 kick-zone，轨迹准确定性地前进。而 Cluster 区域高度噪声敏感——不同噪声条件下轨迹分歧显著。

### 4. 单神经元计算与连接结构

**"Kick 神经元"的发现**：两组三联体神经元具有独特的空间激活特征——在 cluster 中激活前值强负，在 kick-zone 中接近零（ReLU 阈值附近），在 transition 中变正。小的输入扰动即可决定 ReLU 门的开闭，构成转换触发器。

**连接结构**：分析循环权重矩阵 $W_{hh}$ 发现：
- 同一三联体内 kick 神经元互相激励
- 两个三联体间互相抑制
- 两个较大的神经元群体（各约 70 个神经元）形成自激励-互抑制回路
- 这些"噪声整合群体"通过结构化连接调控 kick 神经元

**因果干预验证**：
- **消融（$\mu=0$）**：关闭 kick 神经元或切断噪声整合群体的输入 → 轨迹被困在当前 cluster，无法转换；关键特征值对消失，轨道动力学崩塌为不动点
- **增强（$\mu=2$）**：加倍 kick 神经元活动 → 轨迹超调越过目标 cluster；关键特征值对不变，轨道动力学保持

### 5. 自诱导随机共振

上述所有分析汇聚为一个统一的计算原理——**自诱导随机共振（SISR）**：

与经典随机共振不同（需要外部周期信号），SISR 内在地产生于具有时间尺度分离的系统中。在 RNN 中：
- **慢子系统**：噪声整合群体在 cluster 区域累积随机输入
- **快子系统**：kick 神经元在达到噪声调制阈值后触发快速转换
- 两者协作产生稳定的准周期振荡——振荡周期由噪声方差与慢整合动力学的相互作用决定

网络有效地将内部噪声转化为计算信号，利用 SISR 类动力学实现结构化的概率推理。

## 关键贡献与组合性原理

本文最重要的发现是**组合性动力学基元**：同一基本单元（慢噪声整合 + 快 kick 触发的重置）可以模块化地复用，组合生成更复杂的离散潜在结构。

对于简单的线性链 HMM，单个轨道足够；随着状态数增加，RNN 通过调整读出轴与轨道平面的对齐来捕获更精细的发射离散化。对于全连接和循环 HMM，RNN 发展出连接不同慢区域对的多条轨道——每条轨道都是同一基本 motif 的实例。

## 实验发现摘要


### 主实验

| 方面 | 核心发现 |
|------|---------|
| 无噪声动力学 | 所有架构→单一不动点，无多吸引子 |
| 有噪声动力学 | 噪声维持的轨道动力学，半径∝噪声方差 |
| 状态空间分割 | Clusters（慢/稳定）→ Kick-zones（触发）→ Transitions（快/确定性） |
| 连接结构 | 噪声整合群体 ↔ kick 神经元三联体，互激励-互抑制 |
| 因果验证 | 消融→困在cluster；增强→超调 |
| 训练动态 | 不动点→失稳→轨道涌现，伴随双重下降 |
| 跨架构泛化 | 线性链/全连接/循环 HMM 均使用同一动力学基元 |

## 局限与展望

1. **尺度限制**：仅验证了最多 5 状态的 HMM，更大规模的离散结构是否仍使用相同基元有待探索
2. **架构限制**：仅使用 vanilla RNN（ReLU），GRU/LSTM 等门控架构是否产生相同机制未知
3. **生物验证**：虽然提出了生物可能性假说，但缺乏与真实神经回路的直接对比
4. **确定性输出**：训练使用 Gumbel-Softmax 近似，与真实 HMM 的精确离散采样仍存在差距

## 个人思考

这篇工作从方法论角度非常精巧：不是直接让 RNN 做行为任务，而是用 HMM 作为"已知计算的代理"，从而在知道答案的情况下反向工程 RNN 的机制。这种"合成（synthetic）神经科学"的范式极具洞察力。

SISR 机制的发现尤为优雅——它统一解释了全局轨道动力学、局部状态空间结构、单神经元功能和连接拓扑，形成了从宏观到微观的完整因果链。组合性基元的概念则暗示了一种 RNN "编程语言"——复杂行为可以由简单模块组装而成。

对计算神经科学的启示：大脑可能不是通过多稳态吸引子实现离散状态（如传统假说），而是通过噪声驱动的连续轨道动力学在功能上"雕刻"出离散结构。这对理解自然行为的神经机制具有重要意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Attention (as Discrete-Time Markov) Chains](attention_as_discrete-time_markov_chains.md)
- [\[NeurIPS 2025\] SANSA: Unleashing the Hidden Semantics in SAM2 for Few-Shot Segmentation](sansa_unleashing_the_hidden_semantics_in_sam2_for_few-shot_segmentation.md)
- [\[NeurIPS 2025\] RoMA: Scaling up Mamba-based Foundation Models for Remote Sensing](roma_scaling_up_mamba-based_foundation_models_for_remote_sensing.md)
- [\[NeurIPS 2025\] Mars-Bench: A Benchmark for Evaluating Foundation Models for Mars Science Tasks](mars-bench_a_benchmark_for_evaluating_foundation_models_for_mars_science_tasks.md)
- [\[NeurIPS 2025\] Fast and Fluent Diffusion Language Models via Convolutional Decoding and Rejective Fine-tuning](fast_and_fluent_diffusion_language_models_via_convolutional_decoding_and_rejecti.md)

</div>

<!-- RELATED:END -->
