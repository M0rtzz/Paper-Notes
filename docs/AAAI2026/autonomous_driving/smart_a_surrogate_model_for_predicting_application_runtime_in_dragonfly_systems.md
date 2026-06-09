---
title: >-
  [论文解读] Smart: A Surrogate Model for Predicting Application Runtime in Dragonfly Systems
description: >-
  [AAAI 2026][自动驾驶][Dragonfly 网络] 提出 Smart（Surrogate Model for Predicting Application RunTime），首次将 GNN 和 LLM（Time-LLM）融合用于 Dragonfly 互连网络中的应用迭代运行时预测，在 1…
tags:
  - "AAAI 2026"
  - "自动驾驶"
  - "Dragonfly 网络"
  - "图神经网络"
  - "大语言模型"
  - "代理模型"
  - "混合仿真"
  - "运行时预测"
---

# Smart: A Surrogate Model for Predicting Application Runtime in Dragonfly Systems

**会议**: AAAI 2026  
**arXiv**: [2511.11111](https://arxiv.org/abs/2511.11111)  
**代码**: [https://github.com/SPEAR-UIC/SMART](https://github.com/SPEAR-UIC/SMART)  
**领域**: 高性能计算 / 网络模拟 / 代理模型  
**关键词**: Dragonfly 网络, 图神经网络, 大语言模型, 代理模型, 混合仿真, 运行时预测

## 一句话总结

提出 Smart（Surrogate Model for Predicting Application RunTime），首次将 GNN 和 LLM（Time-LLM）融合用于 Dragonfly 互连网络中的应用迭代运行时预测，在 1,056 节点系统上 MAPE 最低达 1.78%（LAMMPS），推理时间仅 0.515 秒，相比原始仿真实现数量级加速。

## 研究背景与动机

Dragonfly 网络是当前主流的高性能计算（HPC）互连拓扑，Top500 前十超算中有六台采用该拓扑（如 Frontier、Aurora）。其高基数、低直径设计在性价比和扩展性之间取得平衡，但存在一个关键挑战：**工作负载干扰**——多个应用在共享链路上并发执行时，网络流量动态复杂，导致应用迭代时间大幅波动。

传统分析手段是**并行离散事件仿真（PDES）**，以 CODES/ROSS 为代表，可在 flit 粒度精确建模网络行为。然而高保真 PDES 计算代价极高：一次 1,056 节点 Dragonfly 仿真耗时 36–66 小时，无法满足大规模或实时场景需求。

**混合仿真**方案将数据驱动的代理模型嵌入 PDES 流程中，在保持精度的同时大幅加速。但现有代理模型面临三大难题：

- **时空耦合复杂**：网络流量在毫秒尺度上高度动态，同时具有拓扑空间依赖和时序演变特征，传统 LSTM/ARIMA 仅建模时序，忽略空间结构
- **多应用干扰**：并发工作负载竞争共享网络资源，迭代时间随网络负载动态漂移
- **精度-效率权衡**：代理模型需足够准确以替代 PDES，又需足够轻量以实现实时推理

先前最相关的工作 DCRNN（扩散卷积循环网络）仅在 72 节点小系统上验证，扩展到大规模系统后效果显著下降。Smart 正是针对这些缺陷，首次提出 GNN+LLM 融合架构用于大规模 HPC 互连网络的运行时代理建模。

## 方法详解

### 整体框架

Smart 由三大组件构成，形成端到端的时空预测管线：

1. **GNN 编码器**：基于 GCN 捕获 Dragonfly 拓扑的空间结构依赖
2. **Temporal Transformer**：建模 GNN 嵌入序列的时间依赖
3. **LLM 预测模块**：基于 Time-LLM 机制捕获长程时间模式，融入领域知识提示

三路输出在活跃节点上逐节点拼接，经全连接层输出最终预测。

**输入表示**：将 Dragonfly 网络建模为时序图序列 $G_1, G_2, \dots, G_T$，节点为路由器端口，边定义为同路由器内端口全连接 + 全局/本地链路连接。每个节点附带端口级网络状态特征（每 250μs 采样，按迭代窗口聚合为 min/max/avg/分位数统计量）。

**问题定义**：给定进程 $p$ 的回望窗口内迭代时间序列 $\{y_{t-(L_y-1)}, \dots, y_t\}$ 和网络特征 $\{x_{t-(L_x-1)}, \dots, x_t\}$，预测下一迭代时间 $y_{t+1}$。

### 关键设计

#### 1. GNN 编码器（空间建模）

采用两层 GCN 对每个时间步的图进行空间编码：

$$H^{(l+1,t)} = \sigma\left(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l,t)}W^{(l)}\right)$$

- $\tilde{A} = A + I_N$ 为加自环的邻接矩阵
- 输入为回望窗口 $T_{inGNN}$ 内的图序列，输出每时间步的节点嵌入 $H^{(t)} \in \mathbb{R}^{|V| \times d_h}$（$d_h=128$）
- 实验表明两层 GCN 即可有效捕获局部拓扑信息，更深层数反而引入过平滑

#### 2. Temporal Transformer（时间建模）

接收 GNN 嵌入序列 $\{H^{(1)}, \dots, H^{(T_{inGNN})}\}$，通过多头自注意力捕获跨时间步依赖：

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

采用编码器-解码器结构（各 2 层，8 头注意力）。编码器处理完整时序嵌入，解码器以最近时间步嵌入为查询，融合编码器输出产生时间-空间联合嵌入 $Z^{(t)} \in \mathbb{R}^{|V| \times d_z}$（$d_z=128$）。

#### 3. LLM 预测模块（长程时间建模 + 领域知识注入）

基于 Time-LLM 机制，使用冻结的 GPT-2（32 层，$d_{llm}=768$）处理应用迭代时间序列：

- **Patch 嵌入**：将时间序列分段（patch length=2, stride=1），通过线性层映射到 LLM 兼容维度
- **Prompt-as-Prefix（PaP）**：在 patch 嵌入前注入任务提示，包含工作负载名称、回望窗口长度、时间序列统计量（min/max/median/趋势）等领域知识
- **多头注意力对齐**：将 patch 嵌入转换到 LLM 嵌入空间，送入冻结 GPT-2 生成 $E^{(t)} \in \mathbb{R}^{|V_a| \times d_{llm}}$
- **维度压缩**：线性投影为 $F^{(t)} \in \mathbb{R}^{|V_a| \times d_z}$，与 Transformer 输出维度对齐

#### 4. 融合与预测

对 Transformer 输出 $Z^{(t)}$ 按活跃节点掩码过滤得到 $Z_a^{(t)}$，与 LLM 嵌入 $F^{(t)}$ 在节点维度拼接，经全连接层输出最终预测 $\hat{y}_{t+1}$。模型端到端学习 GNN 空间特征与 LLM 时间上下文的相对权重。

### 训练与在线调优策略

- **离线训练**：使用 30% 数据训练，双 NVIDIA A100 GPU 约 3-4 小时
- **在线调优**：推理阶段每 $F_t$ 个迭代利用 PDES 反馈的真实值更新模型权重（反向传播）。$F_t=8$ 时效果最佳，使模型适应最新网络动态变化。这形成"仿真→预测→校验→微调"的自适应闭环

## 实验关键数据

### 数据集与设置

- **D1**：1,056 节点 Dragonfly，MILC（512 节点）+ LAMMPS（512 节点）+ UR 背景流量（36 节点）
- **D2**：扩展 D1，增加 NN 工作负载（3D 模板计算），MILC 384 + LAMMPS 512 + NN 160 节点
- 每个数据集测试 contiguous（连续）和 random（随机）两种作业放置策略
- 评估指标：MAPE（平均绝对百分比误差）

### 表 1：D1 数据集最佳配置下的 MAPE (%) 对比

| 模型 | MILC (Cont) | LAMMPS (Cont) | MILC (Rand) | LAMMPS (Rand) |
|------|------------|---------------|------------|---------------|
| **Smart** (最优) | **3.19** | **1.78** | **3.12** | **1.84** |
| DCRNN | 6.35 | 7.10 | 5.98 | 7.10 |
| LSTM | 6.09 | 7.25 | 6.05 | 7.28 |
| LAST | 7.86 | 8.14 | 7.83 | 8.21 |
| MEAN | 7.80 | 8.24 | 9.31 | 7.72 |

Smart 在所有条件下大幅领先，MAPE 降低约 47–75%。

### 表 2：推理时间与仿真加速比

| 模型 | 平均推理时间 (秒) | 相对 PDES 最短迭代 (8.09s) |
|------|----------------|--------------------------|
| Smart | 0.515 | 6.4% |
| LLM-only | 0.416 | 5.1% |
| GNN-only | 0.063 | 0.8% |
| DCRNN | 0.046 | 0.6% |
| LSTM | 0.040 | 0.5% |
| MEAN | 0.00001 | ~0% |

Smart 虽最慢，但仍不到最短 PDES 迭代时间的 6.4%，相比最长仿真迭代（243s）仅占 0.2%。

### 消融实验

- **GNN-only**（去除 LLM）：D1-Cont MILC MAPE 从 3.19% 升至 3.46%，LAMMPS 从 1.78% 升至 2.05%；D1-Rand LAMMPS 从 1.84% 升至 4.59%，性能严重退化，证明 LLM 对时间建模的关键性
- **LLM-only**（去除 GNN）：D1-Cont MILC MAPE 从 3.19% 升至 4.02%，证明空间特征不可或缺
- **替换 Time-LLM**：用 Autoformer 替换后 MILC 从 3.91% 升至 4.22%；DLinear 替换后 NN 从 3.07% 激增至 4.11%，Time-LLM 在非平稳模式下优势显著

### 超参数鲁棒性

最佳配置为 $T_{inLLM}=8, T_{inGNN}=2, F_t=8$：较长 LLM 窗口捕获长程模式，较短 GNN 窗口关注最近空间动态。Patch length、GCN 层数、LLM 隐藏层数的变化对 MAPE 影响有限（3.1–3.8% 范围），模型对超参数不敏感。

## 亮点与洞察

1. **首次 GNN+LLM 融合用于 HPC 网络代理建模**：创新性地将 Time-LLM 的提示机制引入网络仿真领域，通过 PaP 注入工作负载名称和时序统计量等领域知识，使冻结 LLM 无需微调即可适配 HPC 场景
2. **在线调优闭环设计**：每 8 个迭代利用 PDES 反馈微调权重，使模型持续适应网络流量漂移，形实用的混合仿真框架
3. **空间-时间互补性量化**：消融实验清晰展示 GNN 和 LLM 各自贡献——GNN 擅长建模拓扑导致的空间相关性（Random 放置下 LAMMPS MAPE 从 4.59% 降至 1.84%），LLM 擅长捕获长程时间模式
4. **实用性强**：推理 0.515 秒 vs 仿真 36-66 小时，真正实现了从"小时级"到"亚秒级"的跨越

## 局限性

1. **LLM 开销**：Smart 推理时间（0.515s）是 GNN-only（0.063s）的 8 倍，在对延迟极其敏感的场景可能成为瓶颈
2. **拓扑泛化未验证**：仅在 1D Dragonfly 上实验，论文声称可适配 Fat Tree 等其他拓扑但未提供实证
3. **单步预测**：仅预测 $y_{t+1}$，未验证多步预测能力，长时域预测的误差累积问题未讨论
4. **GPT-2 冻结使用**：LLM 参数完全冻结，仅通过 patch 嵌入和提示驱动，是否能通过微调 LLM 进一步提升精度未探索
5. **单次运行**：所有结果基于单次实验，缺乏多次重复的统计置信区间
6. **数据集规模有限**：仅 2-3 个工作负载的组合，未测试更多异构应用场景

## 相关工作

- **应用运行时预测**：传统方法（ARIMA、MLP、CNN、RNN、LSTM）主要关注时间序列，忽略 HPC 系统空间依赖。DCRNN 首次引入图结构但仅验证 72 节点小系统
- **LLM 用于时间序列**：TimesFM、Time-LLM、TimeGPT-1 等探索 LLM 在时序预测中的潜力，Smart 将其首次引入 HPC 网络场景
- **GNN 网络建模**：RouteNet、XNet 等用 GNN 估计延迟/抖动/丢包，但面向通用网络而非 HPC Dragonfly
- **PraNet**：用 GNN+Transformer+排队论建模短期流量突发，但面向互联网/元宇宙场景，使用有限仿真数据

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|-----------|------|
| 新颖性 | 7 | 首次 GNN+LLM 融合用于 HPC 代理建模，但各组件（GCN/Transformer/Time-LLM）均为现有方法组合 |
| 技术深度 | 7 | 时空建模设计合理，在线调优有工程价值，但融合方式为简单拼接 |
| 实验充分性 | 8 | 两个数据集、两种放置策略、多组超参、完整消融和时序模型对比，实验全面 |
| 写作质量 | 7 | 结构清晰，问题定义规范，但表格密集可读性一般 |
| 实用价值 | 8 | 0.515s 推理替代 36-66h 仿真，有明确落地场景 |
| **综合** | **7.5** | 面向重要 HPC 问题的实用系统，首创 GNN+LLM 融合方案效果显著，但组件组合创新有限 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] ExpertAD: Enhancing Autonomous Driving Systems with Mixture of Experts](expertad_enhancing_autonomous_driving_systems_with_mixture_of_experts.md)
- [\[AAAI 2026\] RoadSceneVQA: Benchmarking Visual Question Answering in Roadside Perception Systems for Intelligent Transportation System](roadscenevqa_benchmarking_visual_question_answering_in_roadside_perception_syste.md)
- [\[NeurIPS 2025\] Towards Predicting Any Human Trajectory in Context](../../NeurIPS2025/autonomous_driving/towards_predicting_any_human_trajectory_in_context.md)
- [\[ICLR 2026\] SMART-R1: Advancing Multi-agent Traffic Simulation via R1-Style Reinforcement Fine-Tuning](../../ICLR2026/autonomous_driving/advancing_multi-agent_traffic_simulation_via_r1-style_reinforcement_fine-tuning.md)
- [\[CVPR 2026\] Scaling-Aware Data Selection for End-to-End Autonomous Driving Systems](../../CVPR2026/autonomous_driving/scaling-aware_data_selection_for_end-to-end_autonomous_driving_systems.md)

</div>

<!-- RELATED:END -->
