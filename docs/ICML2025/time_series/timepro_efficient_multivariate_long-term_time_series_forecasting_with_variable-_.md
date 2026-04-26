---
title: >-
  [论文解读] TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state
description: >-
  [ICML 2025][时间序列][时间序列预测] 提出基于 Mamba 的 TimePro 模型，通过构建变量感知和时间感知的超级状态（hyper-state），自适应选择关键时间点来调节变量维度的隐状态，以线性复杂度实现高效的多变量长期时间序列预测。
tags:
  - ICML 2025
  - 时间序列
  - 时间序列预测
  - Mamba
  - 状态空间模型
  - 多变量建模
  - 多延迟问题
---

# TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state

**会议**: ICML 2025  
**arXiv**: [2505.20774](https://arxiv.org/abs/2505.20774)  
**代码**: [有](https://github.com/xwmaxwma/TimePro)  
**领域**: 时间序列  
**关键词**: 时间序列预测, Mamba, 状态空间模型, 多变量建模, 多延迟问题

## 一句话总结

提出基于 Mamba 的 TimePro 模型，通过构建变量感知和时间感知的超级状态（hyper-state），自适应选择关键时间点来调节变量维度的隐状态，以线性复杂度实现高效的多变量长期时间序列预测。

## 研究背景与动机

**领域现状**：长期时间序列预测（LTSF）是机器学习的重要任务。目前主流方法包括基于 Transformer（iTransformer、PatchTST）、MLP（DLinear、TimeMixer）和 Mamba（S-Mamba、TimeMachine、Bi-Mamba+）的模型。Mamba 凭借线性复杂度和高效长程依赖捕获能力，在时间序列预测中展现出显著优势。

**核心痛点——多延迟问题（multi-delay issue）**：在多变量时间序列中，不同变量对目标变量的影响存在不同的时间滞后。例如温度变化可能在几小时后影响用电量，而湿度变化可能需要更长时间才能产生影响。这种时间滞后的差异性被称为多延迟问题。

**现有方法的不足**：
- **Transformer 类**：iTransformer 关注变量关系建模，但对不同时间点执行统一的粗粒度线性投影；PatchTST 以通道独立方式捕获全局时间依赖，但对所有变量一视同仁。二者均有二次复杂度的瓶颈。
- **Mamba 类**：现有 Mamba 方法（S-Mamba、Bi-Mamba+、TimeMachine）虽然具备线性复杂度，但在扫描变量维度时仅传递普通状态（plain state），忽略了变量间不同的时间滞后特性，无法捕获变量内部复杂的时间变化。

**本文方案**：提出 TimePro，创新性地引入 **时间调谐策略（time-tune strategy）**，在 Mamba 的变量扫描过程中自适应地选择重要时间点来调节隐状态，构建同时感知变量关系和显著时间信息的**超级状态（hyper-state）**，从而在保持线性复杂度的同时有效应对多延迟问题。

## 方法详解

### 整体框架

TimePro 采用类 Transformer 的纯编码器架构，包含以下组件：

1. **可逆实例归一化（RevIN）**：对输入序列进行零均值、单位方差归一化，缓解训练/测试数据的分布偏移问题。预测后进行反归一化。
2. **时间-变量保留嵌入**：将每个单变量时间序列 $\mathbf{X}_{i,:} \in \mathbb{R}^L$ 切分为重叠的 patch，保留变量维度，得到嵌入 $\mathcal{E}_0 \in \mathbb{R}^{N \times P \times D}$，其中 $P$ 为 patch 数，$D$ 为特征维度。
3. **多层 ProBlock**：$\mathcal{E}_{i+1} = \text{ProBlock}(\mathcal{E}_i)$，每层包含 HyperMamba（变量间信息交互）和 TimeFFN（变量内时间变化捕获）。
4. **线性投影**：展平每个变量的嵌入后线性映射得到预测结果。

### 关键设计

#### 1. **HyperMamba 模块**：变量依赖的高效建模

HyperMamba 是 TimePro 的核心组件，对原始 Mamba 做了针对性修改：
- **替换选择性扫描为 Hyper-Scan**：引入时间调谐策略构建超级状态
- **去除深度可分离卷积**：变量间不存在局部性，卷积不必要
- **去除扫描后的线性投影**：由 TimeFFN 替代，避免冗余
- **扫描起始设计（Scanning Inception）**：沿通道维度分为两半，分别沿正向（$1 \to N$）和反向（$N \to 1$）变量方向扫描，增强全局变量依赖捕获

具体流程：输入 $\mathcal{E} \in \mathbb{R}^{N \times (P \times D)}$ 经两个线性投影得到 $\mathcal{E}_t$ 和 $\mathcal{E}_z$，$\mathcal{E}_t$ 经双向 Hyper-Scan 后拼接得到增强嵌入 $\hat{\mathcal{E}}_t$，最终通过门控机制输出：

$$\hat{\mathcal{E}} = \hat{\mathcal{E}}_t \cdot \text{SiLU}(\mathcal{E}_z)$$

#### 2. **Hyper-Scan（超级扫描）**：时间调谐策略的核心实现

这是本文最关键的创新。普通 Mamba 沿变量维度扫描时，隐状态仅包含变量信息，无法感知变量内部的时间变化。Hyper-Scan 通过以下步骤构建时间-变量感知的超级状态：

**Step 1 — 获取初始状态**：沿变量维度扫描嵌入 $\mathcal{E}_t \in \mathbb{R}^{N \times (P \times D)}$，得到初始隐状态 $h \in \mathbb{R}^{N \times (P \times D)}$（在 GPU SRAM 中完成以减少 HBM 读写）。

**Step 2 — 生成偏移量**：对隐状态 reshape 以恢复细粒度时间维度，通过卷积生成初始偏移 $\delta_h = \text{Conv}(h)$。

**Step 3 — 自适应采样**：将参考点与可学习偏移相加得到采样点坐标，通过可微线性插值 $\psi$ 从状态中提取关键时间点：

$$h_{samp} = h_{ref} + \delta_h$$
$$\hat{h} = h_{ref} + \psi(h; h_{samp})$$

其中 $\hat{h} \in \mathbb{R}^{N \times P \times D \times M}$，$M$ 为采样时间点数量（默认为 9）。

**Step 4 — 融合得到超级状态**：通过线性映射将采样时间点融合得到超级状态 $h_o = \text{Linear}(\hat{h})$，再与参数矩阵 $\mathbf{C}$ 相乘得到输出。

这一设计的核心思想借鉴了可变形卷积（Deformable Convolution）的思路，让模型自适应地关注对每个变量最重要的时间点，而非均匀处理所有时间步。

#### 3. **硬件感知实现（Hardware-Aware Implementation）**

充分利用 GPU 内存层次结构：
- 初始状态的获取在 GPU **SRAM** 上完成（遵循原始 Mamba 实现），避免 HBM 的频繁读写
- 其余操作（reshape、偏移生成、采样、线性映射）在 **HBM** 上执行
- 保持了与原始 Mamba 相当的计算效率

### 损失函数 / 训练策略

- 损失函数：标准的 MSE 损失
- 优化器：Adam
- 训练设置遵循 iTransformer 的通用设定
- 回望窗口 $L = 96$，预测长度 $H \in \{96, 192, 336, 720\}$
- 硬件：4 张 Tesla V100 GPU

### 复杂度分析

HyperMamba 的整体复杂度为 $O(NL)$（线性复杂度），其中 $N$ 为变量数，$L$ 为序列长度。这是因为：
- 两个线性投影：$O(NP^2D^2)$
- SSM 参数矩阵计算：$O(NPD)$（状态维度设为 1）
- 变量维度扫描：$O(NPD)$
- 时间调谐（卷积+线性映射）：$O(NPMD)$，$D$ 和 $M$ 为常数可忽略

| 模型 | 复杂度 |
|------|--------|
| **TimePro** | $O(NL)$ |
| iTransformer | $O(N^2 + NL)$ |
| PatchTST | $O(NL^2)$ |
| Transformer | $O(NL + L^2)$ |

## 实验关键数据

### 主实验

在 8 个真实数据集上评测（ETTh1/h2、ETTm1/m2、ECL、Exchange、Weather、Solar-Energy），回望窗口 $L=96$，预测长度 $H \in \{96,192,336,720\}$ 的平均结果：

| 数据集 | 指标 | TimePro | iTransformer | SOFTS | S-Mamba | PatchTST |
|--------|------|---------|-------------|-------|---------|----------|
| ECL | MSE | **0.169** | 0.178 | 0.174 | 0.170 | 0.189 |
| Weather | MSE | **0.251** | 0.258 | 0.255 | 0.251 | 0.256 |
| ETTh1 | MSE | **0.438** | 0.454 | 0.449 | 0.455 | 0.453 |
| ETTm1 | MSE | **0.391** | 0.407 | 0.393 | 0.398 | 0.396 |
| ETTm2 | MSE | **0.281** | 0.288 | 0.287 | 0.288 | 0.287 |
| Exchange | MSE | **0.352** | 0.360 | 0.361 | 0.367 | 0.367 |
| ETTh2 | MSE | 0.377 | 0.383 | **0.373** | 0.381 | 0.385 |
| Solar | MSE | 0.232 | 0.233 | **0.229** | 0.240 | 0.236 |

16 个指标中取得 **12 个第一、2 个第二**。

**效率对比**（ECL 数据集，$L=96, H=720$，V100）：
- 参数量和 FLOPs 最少，仅为 S-Mamba 的 67% 参数和 78% GFLOPs
- 推理速度是 PatchTST 的 2.7 倍、TimesNet 的 14.4 倍
- 训练/推理时间与 S-Mamba 相当

### 消融实验

**时间调谐策略消融**（Exchange / ETTh1 数据集平均）：

| 配置 | Exchange MSE | ETTh1 MSE | 说明 |
|------|-------------|-----------|------|
| Non-Adaptive（线性投影融合） | 0.360 | 0.451 | 均匀处理所有时间点 |
| **Adaptive（TimePro）** | **0.352** | **0.438** | 自适应选择关键时间点 |

**HyperMamba 结构消融**：

| 配置 | Exchange MSE | ETTh1 MSE | 说明 |
|------|-------------|-----------|------|
| Mamba + Hyper-Scan | 0.358 | 0.449 | 保留原始 Mamba 所有组件 |
| − DWConv | 0.358 | 0.447 | 去除深度可分离卷积 |
| − Linear | 0.356 | 0.447 | 再去除线性投影 |
| **HyperMamba** | **0.352** | **0.438** | 完整设计 |

### 关键发现

1. **多延迟问题的可视化验证**：通过 Pearson 相关系数可视化，HyperMamba 处理后的变量相关性矩阵明显更接近真实标签序列的相关性矩阵，验证了时间调谐策略有效缓解了多延迟问题。
2. **超参数敏感性**：特征维度 $D=48$ 时在多数数据集上达到最优；编码器层数 2-4 层为较好选择；patch 长度 16-32 为合理起点。
3. **回望窗口鲁棒性**：TimePro 在不同回望窗口长度（48/96/192/336）下都一致优于对比方法。特别当窗口较短（48）时，TimePro 相比 SOFTS 有显著优势，说明其能有效从有限数据中捕获关键时间信息。

## 亮点与洞察

1. **问题定义精准**：多延迟问题是多变量时序预测中的真实挑战，论文通过清晰的形式化定义和可视化验证，展示了问题的重要性和解决方案的有效性。
2. **巧妙的技术嫁接**：将可变形卷积中的自适应采样思想迁移到 SSM 的隐状态空间中，通过可学习偏移量实现对关键时间点的自适应选择，这一跨领域的技术迁移很有启发性。
3. **减法设计哲学**：去除 Mamba 中的深度可分离卷积和线性投影两个组件，不仅没有损失性能反而有所提升，体现了对模型结构的深入理解。
4. **效率-性能的良好平衡**：在取得最优性能的同时保持最少的参数量和 FLOPs，且维持线性复杂度，适合资源受限场景部署。

## 局限与展望

1. **数据集规模有限**：仅在 8 个较为常规的时序预测基准上评测，缺乏在大规模、高维度或更复杂实际场景中的验证。
2. **采样点数 M 固定为 9**：当前对所有数据集使用固定的采样点数，未探索自适应调整 M 的可能性。
3. **仅关注预测任务**：未探索 TimePro 在时序分类、异常检测、插补等其他时序分析任务上的泛化能力。
4. **单一预测损失**：仅使用 MSE 作为训练目标，未探索多任务学习或辅助损失对性能的影响。
5. **变量关系假设**：隐含假设变量间的时间滞后关系可以通过数据驱动的方式学习，但在某些领域中引入先验知识（如物理约束）可能会更有效。

## 相关工作与启发

- **iTransformer**（Liu et al., 2024b）：将变量作为 token、时间作为特征的倒置设计，启发了 TimePro 优先在变量维度建模的思路
- **SOFTS**（Han et al., 2024）：Series-core fusion 方法，是 TimePro 在多个数据集上的主要竞争对手
- **S-Mamba**（Wang et al., 2025）：首个将 Mamba 应用于多变量时序预测的工作，但使用序列嵌入限制了细粒度时间建模
- **Deformable ConvNets v4**（Xiong et al., 2024）：可变形卷积的自适应采样思想直接启发了 TimePro 的时间调谐策略
- **RevIN**（Kim et al., 2022）：可逆实例归一化已成为时序预测的标准组件

## 评分

- 新颖性: ⭐⭐⭐⭐ 时间调谐策略有新意，将可变形采样引入 SSM 隐状态是有价值的创新，但整体框架仍属 Mamba 变种
- 实验充分度: ⭐⭐⭐⭐ 8 个数据集、多组消融实验、效率对比和可视化分析较为全面，但缺乏不同预测长度的详细展开
- 写作质量: ⭐⭐⭐⭐ 动机清晰、方法描述系统、图示较好，但部分公式排版和符号一致性可改进
- 价值: ⭐⭐⭐⭐ 在效率和性能上取得良好平衡，线性复杂度+最优性能对实际部署有价值，未来可作为时序基础模型的基础架构

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] TQNet: Temporal Query Network for Efficient Multivariate Time Series Forecasting](temporal_query_network_for_efficient_multivariate_time_series_forecasting.md)
- [\[ICML 2025\] Causality-Aware Contrastive Learning for Robust Multivariate Time-Series Anomaly Detection](causality-aware_contrastive_learning_for_robust_multivariate_time-series_anomaly.md)
- [\[AAAI 2026\] CometNet: Contextual Motif-guided Long-term Time Series Forecasting](../../AAAI2026/time_series/cometnet_contextual_motif-guided_long-term_time_series_forecasting.md)
- [\[ICML 2025\] HyperIMTS: Hypergraph Neural Network for Irregular Multivariate Time Series Forecasting](hyperimts_hypergraph_neural_network_for_irregular_multivariate_time_series_forec.md)
- [\[ICML 2025\] WAVE: Weighted Autoregressive Varying Gate for Time Series Forecasting](wave_weighted_autoregressive_varying_gate_for_time_series_forecasting.md)

<!-- RELATED:END -->
