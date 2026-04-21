---
title: >-
  [论文解读] Wavelet Canonical Coherence for Nonstationary Signals
description: >-
  [NeurIPS 2025][时间序列][小波分析] 提出 WaveCanCoh 框架，将经典的典型相干分析（canonical coherence）扩展到小波域，基于多变量局部平稳小波（MvLSW）模型实现对非平稳多变量时间序列两组信号间时变、尺度特定的典型相干性估计。
tags:
  - NeurIPS 2025
  - 时间序列
  - 小波分析
  - 典型相干性
  - 非平稳信号
  - 多变量时间序列
  - 神经科学
---

# Wavelet Canonical Coherence for Nonstationary Signals

**会议**: NeurIPS 2025  
**arXiv**: [2505.14253](https://arxiv.org/abs/2505.14253)  
**代码**: [有](https://github.com/mhaibo/WaveCanCoh)  
**领域**: 时间序列分析 / 信号处理  
**关键词**: 小波分析, 典型相干性, 非平稳信号, 多变量时间序列, 神经科学

## 一句话总结

提出 WaveCanCoh 框架，将经典的典型相干分析（canonical coherence）扩展到小波域，基于多变量局部平稳小波（MvLSW）模型实现对非平稳多变量时间序列两组信号间时变、尺度特定的典型相干性估计。

## 研究背景与动机

在神经科学、金融等领域，理解两组多变量信号之间随时间演化的依赖关系至关重要。例如，分析大鼠海马体不同亚区电极组之间在记忆任务中的功能交互。

**现有方法的核心局限**：

**平稳性假设**：经典的典型相干分析（Brillinger 1981）基于弱平稳假设，即统计特性（期望、协方差、谱特征）需随时间恒定。但实际信号（如 EEG/LFP）几乎都是非平稳的。

**缺乏时间分辨率**：傅里叶域的典型相干分析忽略了时间动态，无法捕捉瞬态的频率特定交互。现有方法主要关注网络内的逐对（pairwise）相干性，缺少组间（between-groups）的全局相干性度量。

**无先验工作**：据作者所知，此前没有将经典典型相干扩展到小波域以度量两组多变量非平稳时间序列间时变典型相干的工作。

**小波分析的天然优势**：小波基函数在时间和频率上同时具有局部化特性，紧支撑的小波可以压缩或拉伸以适应信号的动态特征，特别适合捕捉非平稳信号的瞬态属性。

## 方法详解

### 整体框架

WaveCanCoh 建立在多变量局部平稳小波（MvLSW）过程之上，包含三个核心步骤：
1. **融合**：将两组时间序列 $\mathbf{X}_t, \mathbf{Y}_t$ 拼接为 $\mathbf{Z}_t$
2. **谱估计**：估计局部小波谱（LWS）矩阵
3. **特征分解**：通过特征值分解得到时变典型相干和典型方向向量

### 关键设计

1. **局部小波谱矩阵（LWS Matrix）**

   **功能**：在每个尺度 $j$ 和时间位置 $u$ 处，量化各通道间的局部频谱贡献。

   **核心思路**：基于 MvLSW 模型，过程 $\mathbf{X}_t$ 表示为 $\mathbf{X}_t = \sum_j \sum_k \mathbf{V}_j(k/T) \psi_{j,k}(t) \mathbf{z}_{j,k}$，LWS 矩阵定义为 $\mathbf{S}_j(u) = \mathbf{V}_j(u) \mathbf{V}_j^\top(u)$。将此扩展到跨组 LWS 矩阵：

   $$\mathbf{S}_{j;\mathbf{ZZ}}(u) = \begin{bmatrix} \mathbf{S}_{j;\mathbf{XX}}(u) & \mathbf{S}_{j;\mathbf{XY}}(u) \\ \mathbf{S}_{j;\mathbf{YX}}(u) & \mathbf{S}_{j;\mathbf{YY}}(u) \end{bmatrix}$$

   **设计动机**：小波基的时频局部化特性使得传递矩阵 $\mathbf{V}_j(k/T)$ 能捕捉过程随时间平滑变化的统计特性。

2. **小波典型相干（WaveCanCoh）定义**

   **功能**：在每个尺度 $j$ 和时间 $u$ 处定义两组信号间的最大线性依赖。

   **核心公式**：

   $$\boldsymbol{\rho}_{j;\mathbf{XY}}(u) = \max_{\mathbf{a}_j(u), \mathbf{b}_j(u)} \{\mathbf{a}_j^\top(u) \mathbf{S}_{j;\mathbf{XY}}(u) \mathbf{b}_j(u)\}^2$$

   约束条件为 $\mathbf{a}_j^\top \mathbf{S}_{j;\mathbf{XX}} \mathbf{a}_j = 1$，$\mathbf{b}_j^\top \mathbf{S}_{j;\mathbf{YY}} \mathbf{b}_j = 1$。

   解为矩阵 $\mathbf{M}_{j;\mathbf{a}} = \mathbf{S}_{j,\mathbf{XX}}^{-1} \mathbf{S}_{j,\mathbf{XY}} \mathbf{S}_{j,\mathbf{YY}}^{-1} \mathbf{S}_{j,\mathbf{YX}}$ 的最大特征值 $\Lambda_j^{(1)}(u)$。

3. **因果小波典型相干（Causal-WaveCanCoh）**

   **功能**：引入时滞关系，捕捉潜在的因果效应。

   **核心思路**：定义滞后联合过程 $\mathbf{Z}_t(h) = (\mathbf{X}_t^\top, \mathbf{Y}_{t+h}^\top)^\top$，构建滞后 LWS 矩阵并计算因果典型相干。

4. **估计过程**

   使用平滑周期图法估计 LWS 矩阵：$\hat{\mathbf{S}}_{j,k} = \sum_l A_{jl}^{-1} \tilde{\mathbf{I}}_{l,k}$，其中 $\tilde{\mathbf{I}}_{l,k}$ 为矩形窗平滑的小波周期图。在 $T, M \to \infty$ 且 $M/T \to 0$ 条件下估计一致。

## 实验关键数据

### 仿真实验

**MvLSW 仿真**（$P=6, Q=4, T=1024$）：

| 配置 | 前半段 ($u < 0.5$) | 后半段 ($u > 0.5$) | 说明 |
|------|---------------------|---------------------|------|
| 真实相干 | 弱 | 强 | 设计的时变模式 |
| WaveCanCoh 估计 | 准确跟踪弱依赖 | 准确跟踪强依赖 | 1000次重复验证 |
| 95% Wald 置信区间 | 覆盖真值 | 覆盖真值 | 估计可靠 |

**AR(2) 混合仿真**（$P=4, Q=3, T=1024$）：

| 方法 | 前半段（有 gamma 共享） | 后半段（无共享结构） | 说明 |
|------|--------------------------|--------------------------|------|
| WaveCanCoh | 检测到相干 | 准确捕捉骤降 | 胜出 |
| LSP (Fourier-based) | 检测到相干 | 未能捕捉骤降 | 失败 |

### LFP 数据分析（真实数据）

大鼠海马体 LFP 数据，22 电极分两组（T1/T2/T4/T5 vs T13-T17），尺度 $j=5$ (15.625-31.25 Hz)。

| 时间点 | 尺度 | 正确-错误差异 | p值 | 显著性 |
|--------|------|---------------|-----|--------|
| $t^*=0.5$s | $j=4$ (31.25-62.5Hz) | 0.023 | 0.001** | 显著 |
| $t^*=0.5$s | $j=5$ (15.63-31.25Hz) | 0.334 | 0.002** | 显著 |
| $t^*=0.5$s | $j=6$ (7.81-15.63Hz) | 0.039 | 0.001** | 显著 |
| $t^*=1.0$s | $j=6$ | 0.002 | 0.025** | 显著 |
| $t^*=-1.0$s | 所有尺度 | — | >0.1 | 不显著 |

相比之下，LSP（Fourier-based）方法在同一数据上的排列检验中未发现任何显著差异。

### 关键发现

- **正确试次 vs 错误试次**：在气味刺激后（$t>0$），正确记忆决策的试次在 8-62 Hz 频段表现出显著更高的区域间相干性，而刺激前无显著差异
- **通道贡献不均**：正确试次中各通道对全局相干的贡献较均衡，而错误试次中往往由少数主导通道驱动
- WaveCanCoh 相比 Fourier 方法具有更高的时间分辨敏感性

## 亮点与洞察

- **理论严谨**：完整的数学框架，从定义、估计到一致性证明，理论贡献扎实
- **解释性强**：不仅给出全局相干值，还通过典型方向向量揭示每个通道的时变贡献
- **实用价值**：在神经科学数据分析中展示了 Fourier 方法无法发现的行为相关神经协调模式
- 因果版本（Causal-WaveCanCoh）为探索脑区间定向交互提供了新工具

## 局限与展望

- 计算复杂度随通道数增加而增长（LWS 矩阵求逆）
- 平滑参数 $M$ 的选择对估计质量有显著影响，目前缺乏自适应选择方法
- 仅考虑线性依赖（典型相关的固有限制），非线性依赖需要其他框架
- 在高维通道场景下（数百个电极），可能需要正则化或降维策略

## 相关工作与启发

- Brillinger (2001) 的频域典型相关是本文的直接推广对象
- MvLSW 模型 (Nason 2000, Ombao 2014) 提供了理论基础
- 排列检验的引入使得统计推断保持了非参数特性，方法论上很优雅
- 该框架可自然扩展到金融（板块间动态相关）、语音处理等其他非平稳信号分析领域

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次将典型相干扩展到小波域的非平稳设置
- 实验充分度: ⭐⭐⭐⭐ — 仿真验证 + 真实神经数据分析 + 与Fourier方法对比
- 写作质量: ⭐⭐⭐⭐⭐ — 数学推导严谨清晰，结构组织优良
- 价值: ⭐⭐⭐⭐ — 对非平稳多变量信号分析有广泛适用性

<!-- RELATED:START -->

## 相关论文

- [Revitalizing Canonical Pre-Alignment for Irregular Multivariate Time Series Forecasting](../../AAAI2026/time_series/revitalizing_canonical_pre-alignment_for_irregular_multivariate_time_series_fore.md)
- [Channel Matters: Estimating Channel Influence for Multivariate Time Series](channel_matters_estimating_channel_influence_for_multivariate_time_series.md)
- [Structured Temporal Causality for Interpretable Multivariate Time Series Anomaly Detection](structured_temporal_causality_for_interpretable_multivariate_time_series_anomaly.md)
- [Time-O1: Time-Series Forecasting Needs Transformed Label Alignment](time-o1_time-series_forecasting_needs_transformed_label_alignment.md)
- [WaLRUS: Wavelets for Long-range Representation Using SSMs](walrus_wavelets_for_long-range_representation_using_ssms.md)

<!-- RELATED:END -->
