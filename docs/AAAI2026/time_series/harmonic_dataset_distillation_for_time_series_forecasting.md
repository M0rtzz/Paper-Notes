---
title: >-
  [论文解读] Harmonic Dataset Distillation for Time Series Forecasting
description: >-
  [AAAI 2026][时间序列][数据集蒸馏] 提出HDT（Harmonic Dataset Distillation for Time Series Forecasting），通过FFT将时间序列分解为正弦基底，在频域上通过谐波匹配（Harmonic Matching）对齐合成数据与原始数据的核心周期结构，实现强跨架构泛化和良好可扩展性的时间序列数据集蒸馏。
tags:
  - AAAI 2026
  - 时间序列
  - 数据集蒸馏
  - 时间序列预测
  - 频域优化
  - 谐波匹配
  - 跨架构泛化
---

# Harmonic Dataset Distillation for Time Series Forecasting

**会议**: AAAI 2026  
**arXiv**: [2603.03760](https://arxiv.org/abs/2603.03760)  
**代码**: 无  
**领域**: 时间序列预测 / 数据集蒸馏  
**关键词**: 数据集蒸馏, 时间序列预测, 频域优化, 谐波匹配, 跨架构泛化

## 一句话总结

提出HDT（Harmonic Dataset Distillation for Time Series Forecasting），通过FFT将时间序列分解为正弦基底，在频域上通过谐波匹配（Harmonic Matching）对齐合成数据与原始数据的核心周期结构，实现强跨架构泛化和良好可扩展性的时间序列数据集蒸馏。

## 研究背景与动机

时间序列预测面临严峻的**数据存储和计算成本**挑战：工业传感器和生物医学监测器每日产生TB级数据，加上TimesFM和Moirai等大型基础模型的出现进一步加剧了计算负担。

**数据集蒸馏**（Dataset Distillation, DD）——合成一个小而精的数据集使模型训练效果接近原始完整数据——是一个有前景的解决方案。然而，直接将图像DD方法应用于时间序列预测存在两个根本性局限：

### 窗口化方法的缺陷（图1a）

现有方法将时间序列切分为固定大小的窗口（如96步输入+96步输出），每个窗口作为独立样本进行蒸馏。这种"局部到局部"的匹配方式忽略了时间序列的全局结构：

**有限的可扩展性（L1）**：增加合成数据长度M仅延长已有局部模式，无法捕获更广泛的全局结构，导致收益递减

**架构过拟合（L2）**：局部优化完全忽视了构建整个序列的全局依赖关系，导致蒸馏数据过拟合到特定骨干模型的归纳偏好，跨架构泛化性差

### 核心洞察

时间序列的本质在于其**全局周期结构**。通过FFT将序列分解为正弦基底，每个基底函数对整个序列有全局影响。在频域进行蒸馏可以保证每次更新都修改合成序列的整体，不会破坏时间依赖关系。

## 方法详解

### 整体框架

HDT的蒸馏流程（图2）：

1. 将原始数据 $\mathcal{X}$ 和合成数据 $\mathcal{S}$ 通过FFT变换到频域
2. 选取振幅最大的top-k频率分量作为**谐波（Harmonics）**
3. 通过**谐波匹配损失** $\mathcal{L}_{\text{harm}}$ 对齐两者的谐波分布
4. 通过**梯度匹配损失** $\mathcal{L}_{\text{grad}}$ 确保训练行为一致
5. 优化频域中的谐波系数，最终通过iFFT恢复蒸馏数据

### 关键设计

#### 1. **谐波匹配（Harmonic Matching）**

为实现频率分量的精确对齐，首先从原始数据中采样与合成数据等长（M）的子序列，然后分别做FFT：

$$\mathcal{F_X} = \text{FFT}(\mathcal{X}_{\text{sub}}), \quad \mathcal{F_S} = \text{FFT}(\mathcal{S})$$

选取振幅最大的top-k频率分量作为谐波 $\mathcal{H}$：

$$\mathcal{H} = \text{arg top-}k_{i \in [0, \lfloor M/2 \rfloor]}(|\mathcal{F_X}[i]|)$$

仅保留谐波对应的频率分量，其余置零得到 $\tilde{\mathcal{F_X}}$ 和 $\tilde{\mathcal{F_S}}$。

**谐波损失**最小化两者振幅的Lp距离：

$$\mathcal{L}_{\text{harm}} = \||\tilde{\mathcal{F_X}}| - |\tilde{\mathcal{F_S}}|\|_p$$

这一损失作为正则化器，强制合成数据的周期结构与原始数据对齐。由于谐波是数据的**内在的、模型无关的属性**，这避免了对特定骨干模型的过拟合。

#### 2. **理论保证（Theorem 1）**

作者提供了严格的理论证明：**最小化谐波损失能保证合成数据保留原始数据的全局时间依赖结构**。

核心定理基于功率谱密度（PSD）与自相关函数（ACF）的关系（Wiener-Khintchine定理）：

$$\max_{|k| \leq K} |r_{\mathcal{S}}(k) - r_{\mathcal{X}}(k)| \leq C \cdot \varepsilon$$

其中 $\varepsilon$ 是频域近似误差。这意味着：频域谐波对齐得越好，合成数据的自相关结构就越接近原始数据的自相关结构。

#### 3. **全局更新机制与可扩展性**

由于每个谐波是正弦基底函数，对整个序列有全局影响，因此频域中的每次更新都修改合成序列的整体结构。

增加合成数据长度M → 可表示的周期范围更广 → 能捕获更长周期的全局结构 → 性能随M有意义地持续提升（解决L1）。

#### 4. **梯度匹配**

在频域谐波匹配的基础上，还使用梯度匹配作为蒸馏损失。首先通过iFFT从谐波重建时域信号：

$$\mathcal{X_H} = \text{iFFT}(\tilde{\mathcal{F_X}}), \quad \mathcal{S_H} = \text{iFFT}(\tilde{\mathcal{F_S}})$$

然后匹配模型在原始和合成数据上的多步梯度：

$$\mathcal{L}_{\text{grad}} = \frac{\|\mathcal{T}_j(\theta, \mathcal{S_H}) - \mathcal{T}_i(\theta, \mathcal{X_H})\|_2^2}{\|\theta - \mathcal{T}_i(\theta, \mathcal{X_H})\|_2^2}$$

### 损失函数 / 训练策略

最终优化目标：

$$\underset{\mathcal{F_S}}{\text{argmin}} \; \mathcal{L}_{\text{grad}} + \lambda \mathcal{L}_{\text{harm}}$$

- $\lambda$ 用于平衡两个损失
- 优化变量是频域系数 $\mathcal{F_S}$（而非时域数据点）
- 收敛后通过iFFT恢复最终蒸馏数据 $\mathcal{S}$
- 使用DLinear作为蒸馏骨干模型

## 实验关键数据

### 主实验（M=384，MSE）

**DLinear骨干**（L=同架构, T=iTransformer评估, C=xPatch评估）：

| 方法 | ETTh1 L/T/C | ETTh2 L/T/C | ETTm2 L/T/C | Electricity L/T/C |
|------|-------------|-------------|-------------|-------------------|
| Random | 0.945/0.757/0.664 | 1.860/0.406/0.359 | 1.504/0.256/0.234 | 0.400/0.327/0.351 |
| MTT | 0.521/0.640/0.587 | 0.661/0.387/0.346 | 0.702/0.257/0.248 | 0.342/0.412/0.489 |
| CondTSF | 0.510/0.494/0.492 | 0.392/0.336/0.325 | 0.223/0.209/0.204 | 0.231/0.241/0.238 |
| **HDT** | **0.430/0.421/0.409** | **0.359/0.331/0.311** | **0.211/0.205/0.201** | **0.208/0.239/0.232** |
| Full Data | 0.386/0.389/0.384 | 0.326/0.314/0.296 | 0.186/0.185/0.177 | 0.195/0.152/0.175 |

关键发现：先前方法在跨架构评估时性能急剧下降（甚至不如Random），而HDT在所有设置下保持稳定。

### 消融实验

| 方法 | ETTh1 | ETTh2 | ETTm1 | ETTm2 | Electricity | Traffic |
|------|-------|-------|-------|-------|-------------|---------|
| Base（窗口梯度匹配） | 0.583 | 0.465 | 0.905 | 0.402 | 0.414 | 0.934 |
| Base + 分解（频域梯度匹配） | 0.545 | 0.420 | 0.814 | 0.325 | 0.376 | 0.902 |
| HDT（完整方法） | **0.420** | **0.334** | **0.386** | **0.206** | **0.226** | **0.760** |

频域操作带来显著改善，谐波匹配进一步大幅提升。

### 效率与大规模实验

| 实验 | 结果 |
|------|------|
| 训练加速（iTransformer, Electricity） | 全数据1650s → 蒸馏数据1.98s（**834x加速**） |
| 训练加速（iTransformer, Traffic） | 全数据4266s → 蒸馏数据2.32s（**1839x加速**） |
| 大规模CA数据集（201K长度, 8600特征） | HDT: 44.25 MSE vs CondTSF: 197.95 vs Full: 46.63 |
| Moirai-Large微调（311M参数） | Few-shot+HDT: MSE 1.417, 比全微调仅差2.5%，速度快80x |

### 关键发现

1. **跨架构泛化是HDT的最大优势**：先前方法在骨干和评估模型不同时性能剧烈下降，HDT维持最小的MSE增量
2. **可扩展性**：随着M增加HDT持续改善，而其他方法在一定大小后饱和
3. **蒸馏开销极小**：FFT的 $O(M\log M)$ 复杂度相对于骨干模型的梯度计算可忽略
4. **基础模型微调**：蒸馏数据可用于大型基础模型的few-shot微调，以极小训练成本获得接近全量微调的性能

## 亮点与洞察

1. **从局部到全局的范式转移**：将蒸馏从时域局部窗口转移到频域全局谐波，是一个优雅且有理论支撑的设计
2. **理论与实践的统一**：Theorem 1从PSD-ACF关系严格证明了谐波匹配保留时间依赖的理论保证
3. **跨架构泛化的根本解决**：谐波是数据的内在属性而非特定模型的产物，这保证了模型无关性
4. **实用价值显著**：834x-1839x的训练加速、大规模数据集上的有效性、基础模型微调的应用——每一个都有即时的工业价值

## 局限与展望

1. **合成数据大小M的选择**：虽然性能随M持续改善，但最优M值需要实验确定
2. **谐波数k的超参数选择**：top-k的k值对结果有影响，但未提供自适应选择策略
3. **单变量为主**：虽然实验包含多变量数据集，但频域分解是按通道独立进行的，未利用跨通道结构
4. **蒸馏骨干的选择**：实验主要使用DLinear作为蒸馏骨干，其他骨干的影响未充分探讨
5. 可以考虑将谐波匹配扩展到其他时间序列任务（如分类、异常检测）

## 相关工作与启发

- **DC**（Zhao & Bilen, 2021）：梯度匹配的开创性工作
- **MTT**（Cazenavette et al., 2022）：轨迹匹配方法
- **CondTSF**（Ding et al., 2024）：首个专门针对TSF的DD方法，但仍是窗口化方法
- **Wiener-Khintchine定理**：PSD与ACF的对偶关系，是谐波匹配理论保证的数学基础
- 启发：当数据有强全局结构时（如时间序列的周期性），在变换域（频域）而非原始域（时域）进行操作可能更有效

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 频域蒸馏 + 谐波匹配是一个新颖且理论基础扎实的范式
- 实验充分度: ⭐⭐⭐⭐⭐ — 3种骨干 × 6个数据集 + 跨架构评估 + 大规模实验 + 基础模型微调
- 写作质量: ⭐⭐⭐⭐⭐ — 问题定义清晰，理论推导严谨，实验设计系统性强
- 价值: ⭐⭐⭐⭐⭐ — 解决了TSF数据集蒸馏的两个根本性问题，实用价值极高

<!-- RELATED:START -->

## 相关论文

- [FreqCycle: A Multi-Scale Time-Frequency Analysis Method for Time Series Forecasting](freqcycle_a_multi-scale_time-frequency_analysis_method_for_time_series_forecasti.md)
- [Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series](../../NeurIPS2025/time_series/time-imm_a_dataset_and_benchmark_for_irregular_multimodal_multivariate_time_seri.md)
- [DeepBooTS: Dual-Stream Residual Boosting for Drift-Resilient Time-Series Forecasting](deepboots_dual-stream_residual_boosting_for_drift-resilient_time-series_forecast.md)
- [HN-MVTS: HyperNetwork-based Multivariate Time Series Forecasting](hn-mvts_hypernetwork-based_multivariate_time_series_forecasting.md)
- [Finding Time Series Anomalies using Granular-ball Vector Data Description](finding_time_series_anomalies_using_granular-ball_vector_data_description.md)

<!-- RELATED:END -->
