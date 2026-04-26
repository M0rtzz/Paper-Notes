---
title: >-
  [论文解读] Maximizing Asynchronicity in Event-based Neural Networks
description: >-
  [ICLR 2026][人体理解][事件相机] 提出EVA框架，将事件类比为语言token，用基于RWKV-6的线性注意力异步编码器实现逐事件特征更新，结合多表示预测(MRP)+下一表示预测(NRP)的自监督学习获得可泛化特征，首次在异步-同步(A2S)范式中成功完成高难度目标检测任务(Gen1数据集0.477 mAP)。
tags:
  - ICLR 2026
  - 人体理解
  - 事件相机
  - 异步处理
  - 线性注意力
  - 自监督学习
  - RWKV-6
  - A2S
---

# Maximizing Asynchronicity in Event-based Neural Networks

**会议**: ICLR 2026  
**arXiv**: [2505.11165](https://arxiv.org/abs/2505.11165)  
**代码**: [github.com/haohq19/eva](https://github.com/haohq19/eva)  
**领域**: 事件相机/高效推理  
**关键词**: 事件相机, 异步处理, 线性注意力, 自监督学习, RWKV-6, A2S

## 一句话总结
提出EVA框架，将事件类比为语言token，用基于RWKV-6的线性注意力异步编码器实现逐事件特征更新，结合多表示预测(MRP)+下一表示预测(NRP)的自监督学习获得可泛化特征，首次在异步-同步(A2S)范式中成功完成高难度目标检测任务(Gen1数据集0.477 mAP)。

## 研究背景与动机

**事件相机的特性与挑战**：事件相机以高时间分辨率（最高1μs）、低延迟、低空间冗余输出异步稀疏事件流，但标准ML算法需要tensor-like输入，事件数据的异步稀疏特性与现有方法存在根本矛盾。

**A2S范式的出现**：异步到同步(Asynchronous-to-Synchronous, A2S)框架通过设计高效异步编码器逐事件更新tensor-like特征，再按需采样给下游同步ML算法，成功桥接了异步数据和同步算法的鸿沟。

**现有A2S方法的局限**：(1) 编码器表达力不足——ALERT-Transformer使用EventNet（基于点云），没有层次学习，仅能处理简单识别任务；(2) 端到端有监督学习导致特征任务特异，缺乏跨任务泛化能力；(3) 在复杂检测任务上，A2S方法远不如密集同步方法。

**事件与语言的类比洞察**：两者共有两个关键相似性——(i) 都以序列形式组织，(ii) 都以增量方式贡献信息（事件记录增量亮度变化，词汇增量构建语义）。这启发了将NLP中的线性注意力和自监督学习技术迁移到事件处理。

**事件与语言的关键差异**：(i) 信息密度不同——单个语言token有丰富语义，单个事件仅记录像素级亮度变化，需要聚合才有意义；(ii) 空间局部性——事件具有空间属性（像素坐标），语言没有。这两个差异指导了架构设计的调整方向。

**研究目标**：设计更有表达力的异步编码器 + 自监督学习方法，使A2S框架不仅超越先前A2S方法，还能首次成功应对高难度检测任务。

## 方法详解

### 整体框架
EVA由三大组件构成：(1) 事件token化与嵌入层将原始事件转为向量表示；(2) 基于RWKV-6的异步线性注意力编码器逐事件更新特征；(3) 多任务自监督学习(MRP+NRP)训练编码器。推理时编码器逐事件更新特征，下游任务按需采样特征进行识别或检测。

### 关键设计1：事件token化与嵌入

- **功能**：将每个事件 $e_i = (t_i, x_i, y_i, p_i)$ 映射为向量 $\bm{x}_i \in \mathbb{R}^D$
- **核心思路**：空间token化使用双射映射 $\text{Tok}(x, y, p) = p \times H \times W + y \times W + x$，词汇表大小为 $2 \times H \times W$。时间嵌入采用时间差 $\Delta t_i = t_i - t_{i-1}$ 的正弦编码（而非绝对时间戳），最终嵌入为空间嵌入与时间嵌入之和
- **设计动机**：使用时间差嵌入而非绝对时间戳是为了避免长期运行时绝对时间戳持续增长导致的类似语言模型长度外推失败问题。双射映射保证每个空间位置+极性组合有唯一token

### 关键设计2：矩阵值隐藏状态(MVHS)作为输出

- **功能**：用RWKV-6线性注意力的二维矩阵隐藏状态 $\bm{S} \in \mathbb{R}^{N \times D_{\text{head}} \times D_{\text{head}}}$ 作为编码器输出特征，而非传统的一维向量输出 $\bm{y} \in \mathbb{R}^{D}$
- **核心思路**：RWKV-6的循环形式为 $\bm{S}_i = \text{diag}(\bm{w}_i) \bm{S}_{i-1} + \bm{k}_i \bm{v}_i^T$，隐藏状态自然包含聚合的全局信息。使用多头机制，每头维度 $D_{\text{head}} = D/N$，隐藏状态规模为 $N \times D_{\text{head}} \times D_{\text{head}}$，在不增加模型宽度 $D$ 的前提下扩展了特征的表达容量
- **设计动机**：(1) 单个事件信息密度低，需要聚合信息——隐藏状态正是聚合的全局信息载体；(2) MVHS可将模型规模减小约 $D_{\text{model}}/N$ 倍（相比使用1-D输出），实现轻量化实时处理；(3) 2-D结构有助于学习细粒度空间特征

### 关键设计3：Patch-wise编码(PWE)

- **功能**：将事件按空间坐标分配到不同patch中，每个patch独立编码特征
- **核心思路**：对分辨率 $(H_{\text{sensor}}, W_{\text{sensor}})$ 的事件相机，按patch大小 $P$ 将事件分成 $H_{\text{sensor}} \times W_{\text{sensor}} / P^2$ 个序列，每个patch独立运行编码器。各patch特征拼接后送入下游任务
- **设计动机**：(1) 利用事件的空间局部性（与语言的关键区别），降低序列长度和计算开销；(2) 编码器在固定大小patch上训练，天然支持不同分辨率的事件相机；(3) 模型规模减小约patch数倍，且各patch可并行计算

### 关键设计4：多任务自监督学习

- **功能**：用MRP+NRP两个自监督任务训练编码器，不依赖下游任务标签
- **核心思路(MRP)**：强制编码特征 $\mathcal{F}_i = \mathcal{M}_\theta(\{e_j\}_{j \leq i})$ 预测多种手工表示（事件计数EC、时间面TS等）：
$$\arg\max_{\theta, \Theta} \mathbb{E}_i \prod_{k=1}^{K} \textbf{Pr}(\mathcal{R}_i^k | \mathcal{F}_i; \theta_k)$$
- **核心思路(NRP)**：受NTP启发，预测未来时间窗 $\Delta T$ 内的表示：
$$\arg\max_{\theta, \Theta'} \mathbb{E}_i \prod_{k=1}^{K'} \textbf{Pr}(\mathcal{R}^k(\{e | t_i < t(e) \leq t_i + \Delta T\}) | \mathcal{F}_i; \theta_k')$$
- **设计动机**：(1) 不同手工表示捕获事件的不同信息侧面→多表示学习产生可泛化特征；(2) NRP迫使模型理解运动模式而非简单记忆历史；(3) 单个事件作为预测目标信息不足且噪声不可预测→用聚合表示作为目标更可靠

## 实验关键数据

### DVS128-Gesture动作识别

| 模型 | 编码器参数量 | 分类器参数量 | MAC/事件 | 延迟 | SA | FVA |
|------|------------|------------|---------|------|-----|-----|
| ALERT-Tr. (+RM) | 1.41M | 13.96M | 1.22M | 5.8ms | 84.6% | 94.1% |
| ALERT-Tr. (+LMM) | 0.04M | 0.57M | 0.004M | 3.9ms | 72.6% | 89.2% |
| **EVA (+ResNet-14)** | **0.62M** | **2.83M** | **0.60M** | **14.7ms** | **92.9%** | **96.9%** |

### Gen1目标检测

| 模型 | 类型 | mAP (%) |
|------|------|---------|
| NVS-S | 端到端异步(A) | 8.6 |
| AEGNN | 端到端异步(A) | 14.5 |
| DAGr-L | 端到端异步(A) | 32.1 |
| FARSE-CNN | 端到端异步(A) | 30.0 |
| ASTMNet | 同步密集(S) | 46.7 |
| RVT-B | 同步密集(S) | 47.2 |
| GET | 同步密集(S) | 47.9 |
| **EVA (+RVT-B, D=128)** | **A2S** | **47.5** |
| **EVA-L (+RVT-B, D=192)** | **A2S** | **47.7** |

### 关键消融实验

| MVHS | 时间嵌入 | FVA | SA |
|------|---------|-----|-----|
| ✓ | ✓ | **98.1%** | **94.7%** |
| ✓ | ✗ | 87.8% | 81.1% |
| ✗ | ✓ | 97.4% | 94.1% |

### 关键发现

- **A2S范式首次攻克检测任务**：EVA在Gen1上达到47.7 mAP，超越同步SOTA方法RVT-B(47.2)，这是A2S方法首次在检测任务上取得竞争力结果。此前A2S方法仅能处理简单识别任务
- **MVHS显著提升特征表达力**：移除MVHS后SA从94.7%下降到94.1%（0.6%），而移除时间嵌入的负面影响更大（SA从94.7%下降到81.1%），表明时间建模对事件处理至关重要
- **MRP多表示互相促进学习**：仅学习EC一种表示时EC损失反而更大(0.701 vs 0.366)，说明学习多种表示之间存在正向迁移效应
- **NRP贡献独立于MRP**：移除NRP后FVA从98.1%降到96.8%，SA从94.7%降到94.4%，表明预测未来表示确实帮助模型学到超越简单记忆的知识
- **小patch带来更好效果**：patch大小从16增加到128时，FVA从98.1%降到97.4%，SA从94.7%降到89.3%，尽管大patch有更小的预训练损失（因为稀疏区域多）

## 亮点与洞察

- **事件-语言类比的系统化分析**：不是简单类比，而是系统分析了相似性（序列结构、增量信息）和差异性（信息密度、空间局部性），并据此做出针对性的架构调整——MVHS应对低信息密度，PWE应对空间局部性
- **RWKV-6在事件域的首次成功应用**：线性注意力的并行训练+循环推理天然匹配A2S范式的训练+推理需求，且RWKV-6的数据依赖衰减和门控机制适合连续动态数据
- **从1-D到2-D特征的范式转变**：用矩阵隐藏状态代替向量输出的思路新颖，在不增加模型宽度的前提下扩展表达力，且2-D结构与图像任务天然匹配
- **自监督特征的跨任务迁移**：在Gen1上预训练的编码器特征可直接用于N-Cars分类任务(96.3%准确率)，验证了特征的泛化能力

## 局限性

- **实时性在高分辨率场景受限**：Gen1的事件率(0.618M/s)已超过EVA-L的吞吐量(0.541M/s)，虽然PWE策略可缓解，但对更高分辨率的Gen3(1280×720)相机仍然存在挑战
- **自监督目标依赖手工表示**：MRP和NRP的监督信号来自EC、TS等手工设计的表示，这些表示本身可能丢失某些事件信息，限制了学习上限
- **仅在事件域验证**：尽管框架理论上通用，但实验仅在事件相机数据上验证，未探索其他异步序列数据（如神经尖峰）的适用性
- **编码器延迟较大**：由于层次学习架构，EVA的单事件推理延迟(14.7ms处理8192事件)高于ALERT-Tr.，虽然总处理时间更短

## 相关工作与启发

### vs ALERT-Transformer (Martin-Turrero et al., 2024)
先前最强的A2S方法，使用EventNet做异步编码。EVA在DVS128-Gesture上FVA提升2.8%(96.9% vs 94.1%)、SA提升8.3%(92.9% vs 84.6%)。更重要的是ALERT-Tr.从未在检测任务上取得结果，而EVA达到47.7 mAP。关键差异在于EVA用RWKV-6替代EventNet实现层次学习，MVHS扩展特征表达力。

### vs RVT-B (Gehrig & Scaramuzza, 2023)
同步密集方法的SOTA，在Gen1上达到47.2 mAP。EVA-L以47.7 mAP超越之，且EVA的输入特征通道数仅为6(vs RVT-B的20)。这表明A2S范式通过更好的异步编码器可以匹配甚至超越同步方法，同时保留了异步处理的低延迟优势。

### vs DAGr (Gehrig & Scaramuzza, 2024)
端到端异步图神经网络方法，Gen1上32.1 mAP。EVA的47.7 mAP大幅超越(+15.6)，说明A2S的"编码+密集下游"范式比纯异步方法更有效，因为后者受限于图方法在时间积累上的局限。

## 评分

| 维度 | 评分 | 理由 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐ | 事件-语言类比的系统化分析及MVHS输出的设计思路新颖，但核心组件(RWKV-6、SSL)本身非新 |
| 技术深度 | ⭐⭐⭐⭐ | 架构设计有理有据，消融实验充分，从类比到架构调整的逻辑链条完整 |
| 实验充分度 | ⭐⭐⭐⭐ | 覆盖识别+检测+消融+timing分析，但缺乏更多数据集和更多下游任务的验证 |
| 工程价值 | ⭐⭐⭐⭐⭐ | A2S范式首次攻克检测任务，PWE支持任意分辨率，代码已开源，对事件相机实时应用有直接价值 |

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Time Is All It Takes: Spike-Retiming Attacks on Event-Driven Spiking Neural Networks](time_is_all_it_takes_spike-retiming_attacks_on_event-driven_spiking_neural_netwo.md)
- [\[ICLR 2026\] Rapid Training of Hamiltonian Graph Networks using Random Features](rapid_training_of_hamiltonian_graph_networks_using_random_features.md)
- [\[CVPR 2026\] Stable Spike: Dual Consistency Optimization via Bitwise AND Operations for Spiking Neural Networks](../../CVPR2026/human_understanding/stable_spike_dual_consistency_optimization_via_bitwise_and_operations_for_spikin.md)
- [\[ICLR 2026\] DGNet: Discrete Green Networks for Data-Efficient Learning of Spatiotemporal PDEs](dgnet_discrete_green_networks_for_data-efficient_learning_of_spatiotemporal_pdes.md)
- [\[ICML 2025\] Sparse Spectral Training and Inference on Euclidean and Hyperbolic Neural Networks](../../ICML2025/human_understanding/sparse_spectral_training_and_inference_on_euclidean_and_hyperbolic_neural_networ.md)

<!-- RELATED:END -->
