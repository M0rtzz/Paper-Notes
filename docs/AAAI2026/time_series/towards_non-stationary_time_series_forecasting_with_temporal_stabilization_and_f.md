---
title: >-
  [论文解读] Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing
description: >-
  [AAAI 2026][时间序列][非平稳时间序列] 提出 DTAF 双分支框架，通过时域的非平稳 MoE 滤波器提取并去除异质非平稳模式、频域的频谱差分追踪频率漂移，并通过双分支注意力融合两个域的互补信息，实现鲁棒的非平稳时间序列预测。
tags:
  - AAAI 2026
  - 时间序列
  - 非平稳时间序列
  - 混合专家
  - 频域差分
  - 双分支建模
  - 时序预测
---

# Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing

**会议**: AAAI 2026  
**arXiv**: [2511.08229](https://arxiv.org/abs/2511.08229)  
**代码**: [https://github.com/decisionintelligence/DTAF](https://github.com/decisionintelligence/DTAF)  
**领域**: 时间序列  
**关键词**: 非平稳时间序列, 混合专家, 频域差分, 双分支建模, 时序预测

## 一句话总结

提出 DTAF 双分支框架，通过时域的非平稳 MoE 滤波器提取并去除异质非平稳模式、频域的频谱差分追踪频率漂移，并通过双分支注意力融合两个域的互补信息，实现鲁棒的非平稳时间序列预测。

## 研究背景与动机

### 问题背景

时间序列预测在能源、金融、交通、云计算等领域至关重要。现实世界的时间序列通常在**时间域和频率域**都表现出非平稳性：
- **时域非平稳**：不同时间段的局部分布差异大（如均值、方差变化），影响长期依赖建模
- **频域非平稳**：频率成分随时间变化（如用电负荷中日周期和季节周期因用户行为和气候因素而变化）

### 现有方法的两大挑战

**挑战一：如何提取和分离异质非平稳模式**

时间序列固有地包含平稳和非平稳成分。直接对原始序列建模长期依赖会被非平稳动态严重干扰。虽然可以显式建模平稳成分并隔离非平稳效应，但非平稳模式高度复杂且异质，**单一架构无法全面建模**这些模式。

**挑战二：如何动态建模频率漂移**

现有频率分析方法（如 FFT）假设观测窗口内的平稳性，产生全局频率表示，**无法捕捉瞬态频率偏移和局部模式**。例如在电力负荷预测中，日周期和季节周期会因用户行为和环境因素持续变化。

### 核心思想

联合从时域和频域两个角度处理非平稳性：时域用 MoE 学习多种非平稳模式并过滤，频域用差分操作追踪频谱变化。

## 方法详解

### 整体框架

DTAF 由以下模块组成：
1. **Instance Norm**：缓解训练和推理间的分布偏移
2. **Patching & Embedding**：将长序列分割为 patch 并嵌入高维空间
3. **Temporal Stabilizing Fusion (TFS)**：时域非平稳处理 + 长期依赖建模
4. **Frequency Wave Modeling (FWM)**：频域非平稳建模
5. **Dual-branch Attention**：融合时频两域特征
6. **Predictor (FC)**：最终预测

### 关键设计

#### 1. **Temporal Stabilizing Fusion (TFS)**

TFS 分为两个子模块：Non-stationary MoE Filter 和 Temporal Fusion。

**Non-stationary MoE Filter（非平稳混合专家滤波器）**：

核心目标：学习并**移除**输入中的非平稳模式，得到近似平稳的表示。

- 由多个 Expert（每个用独立 MLP 实现）组成，每个专家专注于提取某种特定的非平稳模式
- 路由网络基于 KAN 线性层 + Softmax 为每个 patch 动态分配专家权重
- 提取的非平稳模式从原始 patch 中减去：

$$\mathbf{X}_{\mathrm{stable}}^i = \mathbf{X}_{\mathrm{patch}}^i - \mathbf{X}_{\mathrm{patterns}}^i$$

- 关键约束：用 **KL 散度损失** 确保专家确实学到了非平稳模式（即去除后各 patch 分布应趋于一致）：

$$\mathcal{L}_{\mathrm{stable}} = \alpha \sum_{i=1}^{N} \sum_{j=1}^{N} \mathrm{KL}(\mathbf{X}_{\mathrm{stable}}^i, \mathbf{X}_{\mathrm{stable}}^j) / N^2$$

**设计动机**：不同 patch 可能包含不同类型的非平稳模式（趋势变化、突变、方差漂移等），单一模型无法全面覆盖，因此用多专家分而治之。

**Temporal Fusion（时序融合）**：

在获得近平稳表示后，建模长期依赖：

1. **特征提取**：对每个 patch 进行时序分解（趋势 + 季节），分别线性变换后融合：$\mathbf{X}_h^i = \mathbf{W}_i^t \cdot \mathbf{X}_t^i + \mathbf{W}_i^s \cdot \mathbf{X}_s^i$
2. **历史权重生成**：用线性层计算当前 patch 与各历史 patch 的融合权重（因果掩码确保不泄露未来信息）
3. **加权聚合**：$\mathbf{X}_{\mathrm{history}}^i = \mathrm{MLP}(\sum_{n=1}^{i-1} \mathbf{Weight}_n^i \cdot \mathbf{X}_{\mathrm{stable}}^n)$
4. **门控机制**：动态调节当前 patch 的贡献：$\mathbf{X}_{\mathrm{current}}^i = \mathrm{Gate}(\mathbf{X}_{\mathrm{patch}}^i) \cdot \mathbf{X}_{\mathrm{patch}}^i$
5. **融合**：$\mathbf{H}_t^i = \mathbf{X}_{\mathrm{current}}^i + \mathbf{X}_{\mathrm{history}}^i$

#### 2. **Frequency Wave Modeling (FWM)**

核心创新：在频域引入**差分操作**来追踪频谱随时间的变化。

步骤：
1. 对每个 patch 的时域表示做 rFFT 得到频谱：$\mathbf{Freq}^i = \mathrm{rFFT}(\mathbf{H}_t^i)$
2. 相邻 patch 间频谱差分：$\mathbf{Wave}^i = \mathbf{Freq}^i - \mathbf{Freq}^{i-1}$
3. 选择变化最显著的 Top-K 频率分量：$\mathbf{Picks}^i = \mathrm{TopK}(\mathbf{Wave}^i)$
4. 将未被选中的频率分量置零
5. 逆 FFT 转换回时域

**设计动机**：传统 FFT 给出全局频率表示，无法区分"稳定不变的周期模式"和"正在发生变化的频率成分"。差分操作直接突出变化最大的频率，让模型聚焦于非平稳的频域动态。

#### 3. **Dual-branch Attention**

将 TFS 的时域特征和 FWM 的频域特征通过各自独立的自注意力处理后拼接：

$$\mathbf{H}_{\mathrm{fusion}} = \mathrm{Concat}(\mathbf{Atten}_t, \mathbf{Atten}_f) \in \mathbb{R}^{2N \times d}$$

最终通过 FC 层预测。

### 损失函数 / 训练策略

总损失由三部分组成：

$$\mathcal{L} = \mathcal{L}_{\mathrm{task}} + \alpha \cdot \mathcal{L}_{\mathrm{stable}} + \beta \cdot \mathcal{L}_{\mathrm{robust}}$$

- $\mathcal{L}_{\mathrm{task}}$：L1损失
- $\mathcal{L}_{\mathrm{stable}}$：KL散度约束非平稳MoE滤波器
- $\mathcal{L}_{\mathrm{robust}}$：R-Drop损失提升鲁棒性

## 实验关键数据

### 主实验

11个真实数据集上的多变量预测结果（MSE / MAE，数值越低越好）：

| 数据集 | DTAF (MSE/MAE) | Amplifier | iTransformer | PatchTST | Stationary | 提升 |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| ILI | **1.688/0.801** | 1.819/0.888 | 1.857/0.892 | 1.902/0.879 | 2.389/1.027 | vs Stationary: MSE -29.4% |
| Covid-19 | **1.351/0.040** | 5.578/0.112 | 1.488/0.049 | 1.697/0.056 | 2.658/0.078 | vs Stationary: MSE -49.2% |
| NN5 | **0.643/0.538** | 1.794/1.018 | 0.660/0.550 | 0.698/0.582 | 1.295/0.915 | vs Stationary: MSE -50.3% |
| ETTh | **0.369/0.398** | 0.385/0.416 | 0.404/0.425 | 0.384/0.415 | 0.521/0.493 | vs Stationary: MSE -29.2% |
| ETTm | **0.297/0.338** | 0.327/0.364 | 0.314/0.358 | 0.302/0.347 | 0.456/0.430 | vs Stationary: MSE -34.9% |
| Weather | **0.222/0.250** | 0.222/0.263 | 0.232/0.269 | 0.223/0.261 | 0.293/0.315 | vs Stationary: MSE -24.2% |
| Electricity | **0.160/0.248** | 0.174/0.267 | 0.163/0.258 | 0.171/0.270 | 0.194/0.295 | vs Stationary: MSE -17.5% |
| Traffic | **0.402/0.249** | 0.423/0.294 | 0.397/0.281 | 0.397/0.275 | 0.621/0.339 | vs Stationary: MSE -35.3% |

在 16 个评估指标中，DTAF 在 15 个上排名第一。在非平稳数据集（Covid-19、NN5）上表现尤其突出。

### 消融实验

在 NN5（非平稳）和 ETTh1（常规）数据集上的消融：

| 配置 | NN5-24 MSE | NN5-24 MAE | ETTh1-96 MSE | ETTh1-96 MAE | 说明 |
|:---|:---:|:---:|:---:|:---:|:---|
| **DTAF** | **0.716** | **0.559** | **0.359** | **0.384** | 完整模型 |
| w/o TFS | 0.723 | 0.567 | 0.367 | 0.391 | 去除时域模块 |
| w/o FWM | 0.729 | 0.574 | 0.363 | 0.391 | 去除频域模块 |
| w/ Cross Attention | 0.725 | 0.571 | 0.376 | 0.395 | 用交叉注意力替换双分支注意力 |

每个模块都不可或缺：TFS 影响长期依赖建模，FWM 影响周期模式追踪，双分支注意力优于交叉注意力。

### 关键发现

1. **非平稳数据集上提升最大**：在 Covid-19 上 MSE降低 49%，在 NN5 上降低 50%，验证了方法对非平稳性的处理效果
2. **专家权重分布的可解释性**：不同样本和 patch 具有不同的专家权重分配，相似模式的样本共享相似的权重分布
3. **MoE 滤波器确实稳定了分布**：经过滤波器处理后，不同 patch 的分布明显趋于一致
4. **TopK 频率选择不偏好低频**：差分后的 TopK 选择在不同 patch 间具有多样性，不会被低频高能量成分主导
5. **双分支优于单分支**：仅用时域或仅用频域都不如联合建模

## 亮点与洞察

1. **频域差分的创新性**：现有方法大多在时域做差分（如 ARIMA），在频域做差分来追踪频谱变化是一个新颖且直觉清晰的想法
2. **MoE 处理异质非平稳模式**：用混合专家让不同专家专注于不同类型的非平稳模式，比统一处理更灵活
3. **双域互补建模的系统性**：不是简单叠加时域和频域，而是各自处理完非平稳问题后再融合，逻辑清晰
4. **KL约束的精妙设计**：通过最小化去除非平稳成分后各 patch 间的 KL 散度，间接确保专家学到的确实是非平稳成分

## 局限与展望

1. **patch 长度的先验选择**：patch 划分策略需要预设长度，未探讨自适应 patch 方案
2. **MoE 的专家数量**：论文使用 4 个专家，未讨论专家数量的影响和最优选择
3. **频率差分仅用相邻 patch**：仅计算相邻时间步的频谱差分，未考虑更长跨度的频率变化
4. **未与最新的大语言模型时序方法对比**：如 TimesFM、Chronos 等基于基础模型的方法
5. **计算效率**：双分支设计 + MoE + FFT 操作的组合增加了计算开销，未进行效率分析

## 相关工作与启发

- **Non-stationary Transformer**：通过输入变换处理非平稳性，但未在频域建模
- **RevIN**：实例归一化处理分布偏移，DTAF 在此基础上进一步用 MoE 建模异质非平稳模式
- **TimesNet**：将频域信息用于多周期建模，但假设频率模式静态不变
- **PatchTST**：patch 策略的来源，DTAF 在其基础上引入了非平稳处理
- 启发：频域差分思想可推广到音频、信号处理等其他涉及频谱变化的任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 频域差分和非平稳MoE滤波器的组合是创新的设计
- **实验充分度**: ⭐⭐⭐⭐⭐ — 11个数据集、完整的消融和可视化分析
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，模块设计动机明确
- **价值**: ⭐⭐⭐⭐ — 在非平稳时序预测这一重要问题上提供了有效的解决方案

<!-- RELATED:START -->

## 相关论文

- [IdealTSF: Can Non-Ideal Data Contribute to Enhancing Time Series Forecasting?](idealtsf_can_non-ideal_data_contribute_to_enhancing_the_performance_of_time_seri.md)
- [FreqCycle: A Multi-Scale Time-Frequency Analysis Method for Time Series Forecasting](freqcycle_a_multi-scale_time-frequency_analysis_method_for_time_series_forecasti.md)
- [M2FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting](m2fmoe_multi-resolution_multi-view_frequency_mixture-of-experts_for_extreme-adap.md)
- [Neural MJD: Neural Non-Stationary Merton Jump Diffusion for Time Series Prediction](../../NeurIPS2025/time_series/neural_mjd_neural_non-stationary_merton_jump_diffusion_for_time_series_predictio.md)
- [Scaling LLM Speculative Decoding: Non-Autoregressive Forecasting in Large-Batch Scenarios](scaling_llm_speculative_decoding_non-autoregressive_forecasting_in_large-batch_s.md)

<!-- RELATED:END -->
