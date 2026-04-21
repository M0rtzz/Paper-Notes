---
title: >-
  [论文解读] Parallelization of Non-linear State-Space Models: Scaling Up Liquid-Resistance Liquid-Capacitance Networks for Efficient Sequence Modeling
description: >-
  [NeurIPS 2025][时间序列][状态空间模型] 提出 LrcSSM，通过约束液态电阻-液态电容（LRC）网络的 Jacobian 矩阵为对角形式，实现非线性 RNN 的精确高效并行化，在长序列分类任务上超越 Transformer、LRU、S5 和 Mamba 等 SOTA 方法。
tags:
  - NeurIPS 2025
  - 时间序列
  - 状态空间模型
  - 非线性RNN
  - 并行化
  - 生物启发
  - 对角Jacobian
---

# Parallelization of Non-linear State-Space Models: Scaling Up Liquid-Resistance Liquid-Capacitance Networks for Efficient Sequence Modeling

**会议**: NeurIPS 2025  
**arXiv**: [2505.21717](https://arxiv.org/abs/2505.21717)  
**代码**: [GitHub](https://github.com/MoniFarsang/LrcSSM)  
**领域**: 时间序列  
**关键词**: 状态空间模型, 非线性RNN, 并行化, 生物启发, 对角Jacobian

## 一句话总结

提出 LrcSSM，通过约束液态电阻-液态电容（LRC）网络的 Jacobian 矩阵为对角形式，实现非线性 RNN 的精确高效并行化，在长序列分类任务上超越 Transformer、LRU、S5 和 Mamba 等 SOTA 方法。

## 研究背景与动机

线性状态空间模型（如 S4、S5、Mamba）因可并行化而在序列建模中大获成功，但其线性状态转移限制了表达力。传统非线性 RNN 虽然能通过非线性状态更新更精细地捕获输入关联，却因固有的序列依赖性无法高效并行化，逐渐被边缘化。

近期 DEER 和 ELK 方法尝试通过牛顿迭代+并行扫描来并行化非线性 RNN，但存在两个问题：

**稠密 Jacobian 不可扩展**：DEER 使用完整的方阵 Jacobian，无法扩展到长序列

**近似不精确**：quasi-DEER/quasi-ELK 简单取 Jacobian 对角线，丢弃了非对角元素中可能包含的重要反馈回路信息，且数值不稳定

**核心创新思路**：与其事后丢弃 Jacobian 的非对角元素（近似），不如从模型设计层面**约束 Jacobian 就是对角的**。直觉在于：非线性 SSM 的常量突触参数矩阵本身可以对角化，复杂的神经元反馈回路可以被其复数特征值很好地概括。

## 方法详解

### 整体框架

LrcSSM 架构包含：输入编码器 → 归一化层 → 多层非线性 LRC SSM 块（2/4/6层）→ MLP + 跳跃连接 → 后归一化 → 解码器。核心是 LRC SSM 块中的并行化迭代线性化计算。

### 关键设计

1. **对角化 LRC 模型设计**：原始 LRC 方程中，每个神经元的遗忘电导 $f_i$ 和更新电导 $z_i$ 依赖于**所有**神经元的膜电位。LrcSSM 将其修改为仅依赖**自身**膜电位（状态依赖部分）和**全部外部输入**（输入依赖部分）：

   $$f_i^*(x_i, \mathbf{u}) = \underbrace{g_i^{max,x} \sigma(a_i^x x_i + b_i^x)}_{x_i \text{ 状态依赖}} + \underbrace{g_i^{max,u} \sigma(\sum_j a_{ji}^u u_j + b_j^u)}_{\mathbf{u} \text{ 输入依赖}} + g_i^{leak}$$

   同理修改 $z_i^*$ 和弹性 $\epsilon_i^*$。这使得 Jacobian 矩阵 $\mathbf{A}(\mathbf{x}, \mathbf{u})$ 天然为对角形式：

   $$\mathbf{A}(\mathbf{x}, \mathbf{u}) = \text{diag}[-\sigma(f_i^*) \sigma(\epsilon_i^*)]$$

2. **精确并行化**：由于 Jacobian 天然对角，无需 quasi 近似。直接使用 DEER/ELK 的并行扫描算法，计算复杂度为 $\mathcal{O}(TD)$，序列深度仅 $\mathcal{O}(\log T)$。关键优势：不再需要 Algorithm 1 中的 Line 8 `$J_s \leftarrow \text{Diag}(J_s)$`，因为 $J_s$ 本身就是对角的。

3. **梯度稳定性保证**：与 Liquid-S4 和 Mamba 不同，LrcSSM 的对角结构提供了形式化的梯度稳定性证明。状态矩阵中的 $-\sigma(\cdot)$ 确保所有对角元素为负，保证了系统稳定性。

### 损失函数 / 训练策略

- 分类任务采用交叉熵损失
- 使用显式 Euler 积分：$\mathbf{x}_t = \mathbf{x}_{t-1} + \Delta t \cdot \dot{\mathbf{x}}_{t-1}$
- 超参数通过验证集网格搜索选择
- 5 个不同随机种子取平均测试精度

## 实验关键数据

### 主实验（UEA 多变量时间序列分类，测试精度 %）

| 方法 | Heart (405) | SCP1 (896) | SCP2 (1152) | Ethanol (1751) | Motor (3000) | Worms (17984) | 平均 |
|------|------------|-----------|------------|---------------|-------------|-------------|------|
| LRU | 78.1 | 84.5 | 47.4 | 23.8 | 51.9 | 85.0 | 61.8 |
| S5 | 73.9 | 87.1 | 55.1 | 25.6 | 53.0 | 83.9 | 63.1 |
| Mamba | 76.2 | 80.7 | 48.2 | 27.9 | 47.7 | 70.9 | 58.6 |
| LinOSS-IM | 75.8 | 87.8 | 58.2 | 29.9 | **60.0** | **95.0** | 67.8 |
| Transformer | 70.5 | 84.3 | 49.1 | 40.5 | 50.5 | OOM | 59.0 |
| **LrcSSM** | 72.7 | 85.2 | 53.9 | **36.9** | 58.6 | 90.6 | **66.3** |

### 消融实验（不同非线性 RNN 的对角化比较，固定 64 单元 × 6 层）

| 模型 | Heart | SCP1 | SCP2 | Ethanol | Motor | Worms | 平均 |
|------|-------|------|------|---------|-------|-------|------|
| MguSSM | 74.0 | 78.3 | 49.6 | 31.1 | **56.4** | **90.0** | 63.2 |
| GruSSM | **75.7** | 80.2 | 52.5 | 34.5 | 49.6 | 86.1 | 63.1 |
| LstmSSM | 75.0 | 78.8 | 51.1 | 32.6 | 54.3 | 82.2 | 62.3 |
| **LrcSSM** | 75.0 | **84.8** | **55.4** | **36.1** | 55.7 | 85.6 | **65.4** |

### 关键发现

1. **EthanolConcentration 上优势显著**：LrcSSM 36.9% vs LRU 23.8%、Mamba 27.9%，因为该数据集含丰富的输入关联，非线性状态依赖能更好捕获
2. **长序列表现突出**：在 >1500 序列长度的三个数据集上尤为竞争力强
3. **对角化无损**：与原始稠密 Jacobian LRC 对比，强制对角化并不降低性能（见附录消融）
4. **生物启发模型胜过通用 RNN**：LrcSSM 在 6 个数据集平均优于 MguSSM、GruSSM 和 LstmSSM
5. **仅不及 LinOSS-IM**：可能因 LrcSSM 用显式 Euler 而 LinOSS-IM 用隐式积分，提示更好的积分方案可进一步提升

## 亮点与洞察

- **设计哲学**：不是"并行化后取对角近似"，而是"设计一个 Jacobian 天然对角的模型"——从源头解决问题
- **生物合理性**：液态电阻和液态电容建模了真实神经元的饱和效应和膜电容的动态特性，非线性表达力有理论根基
- **通用方法论**：论文明确展示了如何将任意非线性 RNN 转化为对角 Jacobian 形式（GRU/LSTM/MGU 验证），方法可推广

## 局限与展望

1. 并行化需多步牛顿迭代收敛，线性 SSM 无此开销，实际速度优势取决于迭代次数
2. 目前使用显式 Euler 积分，隐式积分（如 LinOSS-IM）可能进一步提升精度
3. 短序列任务（如 Heart）表现一般，说明非线性复杂度在简单任务上可能过度
4. 目前仅在分类任务上验证，未在时序预测/生成任务上测试

## 相关工作与启发

本文将生物启发的 Liquid Neural Networks（LTC→STC→LRC）与现代 SSM 并行化技术（DEER/ELK）桥接，展示了非线性 RNN 在效率和性能上同时接近线性 SSM 的可能性。与 LinOSS 的有趣对比（两者都有生物背景但建模不同现象）暗示了更丰富的生物启发设计空间。

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 从模型设计层面实现对角 Jacobian 的思路原创性强
- **实验充分度**: ⭐⭐⭐⭐ 6 个数据集 + 多模型对角化对比 + 消融，但任务类型单一
- **写作质量**: ⭐⭐⭐⭐ 生物背景和数学推导清晰，架构图直观
- **价值**: ⭐⭐⭐⭐ 为非线性 RNN 的复兴提供了切实可行的并行化路径

<!-- RELATED:START -->

## 相关论文

- [RiverMamba: A State Space Model for Global River Discharge and Flood Forecasting](rivermamba_a_state_space_model_for_global_river_discharge_and_flood_forecasting.md)
- [TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state](../../ICML2025/time_series/timepro_efficient_multivariate_long-term_time_series_forecasting_with_variable-_.md)
- [Scaling LLM Speculative Decoding: Non-Autoregressive Forecasting in Large-Batch Scenarios](../../AAAI2026/time_series/scaling_llm_speculative_decoding_non-autoregressive_forecasting_in_large-batch_s.md)
- [Physics-informed Reduced Order Modeling of Time-dependent PDEs via Differentiable Solvers](physics-informed_reduced_order_modeling_of_time-dependent_pdes_via_differentiabl.md)
- [Simple and Efficient Heterogeneous Temporal Graph Neural Network](simple_and_efficient_heterogeneous_temporal_graph_neural_network.md)

<!-- RELATED:END -->
