---
title: >-
  [论文解读] Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting
description: >-
  [ICLR 2026][时间序列][不规则时间序列] 提出 ReIMTS，通过基于时间段的递归分割（而非重采样）来保留不规则多变量时间序列的原始采样模式，结合不规则感知的表示融合机制实现多尺度建模，作为插件在六种 IMTS 骨干上平均提升 27.1%。
tags:
  - ICLR 2026
  - 时间序列
  - 不规则时间序列
  - 多尺度建模
  - 递归分割
  - 采样模式保留
  - 预测
---

# Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting

**会议**: ICLR 2026  
**arXiv**: [2602.21498](https://arxiv.org/abs/2602.21498)  
**代码**: [有](https://github.com/Ladbaby/PyOmniTS)  
**领域**: 时间序列  
**关键词**: 不规则时间序列, 多尺度建模, 递归分割, 采样模式保留, 预测

## 一句话总结

提出 ReIMTS，通过基于时间段的递归分割（而非重采样）来保留不规则多变量时间序列的原始采样模式，结合不规则感知的表示融合机制实现多尺度建模，作为插件在六种 IMTS 骨干上平均提升 27.1%。

## 研究背景与动机

不规则多变量时间序列（IMTS）在医疗、气象等场景中广泛存在，其观测时间间隔不均匀，且不同变量的观测时刻可能不对齐。这种采样模式本身包含重要信息，例如 ICU 中密集监测→稀疏监测反映病情从危急到平稳的变化。

现有多尺度方法的核心问题：

- **规则时间序列方法**（Scaleformer、TimeMixer、Pathformer）假设输入均匀采样，不适用于 IMTS
- **IMTS 多尺度方法**（Warpformer、Hi-Patch、HD-TTS）依赖 **重采样** 获取粗粒度序列，这会破坏原始采样模式——如 PhysioNet'12 中 Bilirubin 从密到疏的采样模式在下采样后被打乱
- 采样模式信息（如紧急→常规监测转变）对临床决策至关重要，不应被破坏

## 方法详解

### 整体框架

ReIMTS 是一个即插即用的多尺度框架，兼容大多数编码器-解码器结构的 IMTS 模型。核心思想：在每个尺度层级，基于时间段递归分割样本为更短时间周期的子样本，保持所有观测的原始时间戳不变。

框架由三部分组成：(1) 递归分割模块，(2) 每层级的骨干编码器，(3) 不规则感知表示融合模块。

### 关键设计

**1. 基于时间段的递归分割**

在尺度层级 $n$ 上，将样本按时间段 $T^n$ 分割为 $P^n = T^1/T^n$ 个子样本。关键区别在于：

- 分割依据是 **时间段**（如12小时、24小时），而非观测数量
- 原始时间戳完全保留，零填充用于对齐
- 这避免了基于观测数量分割时不同子样本对应不同现实时间长度的问题

例如在 PhysioNet'12 中：层级1为完整48小时，层级2按24小时分割（2个子样本），层级3按12小时分割（4个子样本），形成全局到局部的视角。

**2. 多尺度表示学习**

每个尺度层级 $n$ 使用独立的骨干编码器 $\mathcal{F}^n_{\text{enc}}$：

$$\mathbf{E}^n = \mathcal{F}^n_{\text{enc}}(\mathbf{S}^n)$$

潜在表示分三类：时间表示 $\mathbf{E}^n_{\text{time}} \in \mathbb{R}^{P^n \times L^n \times D}$、变量表示 $\mathbf{E}^n_{\text{var}} \in \mathbb{R}^{P^n \times V \times D}$、观测表示 $\mathbf{E}^n_{\text{obs}} \in \mathbb{R}^{P^n \times L^n \times V \times D}$。

上层全局表示 $\mathbf{H}^n$ 通过分割（时间/观测表示）或复制（变量表示）操作，变换为与下层局部表示 $\mathbf{E}^{n+1}$ 形状匹配的表示。

**3. 不规则感知表示融合（IARF）**

在下层尺度 $n+1$，利用二值掩码 $\mathbf{M}^{n+1}$ 标记实际观测与填充值：

$$\mathbf{H}^n_{\text{IMTS}} = \begin{cases} \mathbf{H}^n \cdot \mathbf{M}^{n+1}, & \text{时间/观测表示} \\ \mathbf{H}^n, & \text{变量表示} \end{cases}$$

通过轻量级评分层计算权重 $\alpha = \text{ReLU}(\text{FF}(\mathbf{H}^n_{\text{IMTS}}))$，然后融合局部与全局表示：

$$\mathbf{G}^{n+1} = \mathbf{E}^{n+1} + \alpha \mathbf{H}^n_{\text{IMTS}}$$

变量表示的不规则信息已由 IMTS 骨干编码器编码，但时间/观测表示中仍存在填充值，因此需掩码区分。

### 损失函数 / 训练策略

在最低尺度层级 $N$，解码器将所有层级表示拼接后解码：

$$\hat{\mathbf{Z}} = \mathcal{F}_{\text{dec}}(\text{Concat}(\{\mathbf{G}^n\}_{n=1}^N))$$

训练使用 MSE 损失，仅对预测窗口内的预测查询计算：

$$\mathcal{L} = \frac{1}{Y_Q} \sum_{j=1}^{Y_Q} (\hat{z_j} - z_j)^2$$

训练最多300个 epoch，early stopping 耐心为10。

## 实验关键数据

### 主实验

在5个 IMTS 数据集（MIMIC-III/IV、PhysioNet'12、Human Activity、USHCN）和26个基线方法上评估。

| 骨干模型 | 原始 MSE(×10⁻¹) | +ReIMTS MSE(×10⁻¹) | 平均提升 |
|---------|-----------------|-------------------|---------|
| PrimeNet | 9.04/6.25/7.93/26.84/4.57 | 4.76/3.58/3.01/0.82/1.71 | **↑62.3%** |
| mTAN | 8.51/5.09/3.75/0.89/5.65 | 6.37/4.04/3.51/0.89/1.70 | ↑24.3% |
| TimeCHEAT | 4.41/2.50/3.27/0.68/1.73 | 4.40/2.02/2.90/0.52/1.62 | ↑12.1% |
| GRU-D | 4.75/5.97/3.25/1.76/2.42 | 4.67/3.91/3.25/0.51/1.89 | ↑25.8% |
| GraFITi | 4.08/2.39/2.85/0.43/1.71 | 4.07/1.79/2.83/0.42/1.66 | ↑6.3% |

与其他多尺度 IMTS 方法对比（以 GraFITi 为骨干）：

| 方法 | MIMIC-III | MIMIC-IV | PhysioNet'12 | Human Activity | USHCN |
|-----|----------|---------|-------------|---------------|-------|
| Warpformer | 4.09 | 2.42 | 2.88 | 0.54 | 1.77 |
| HD-TTS | 4.17 | 2.36 | 2.83 | 0.50 | 1.66 |
| Hi-Patch | 4.35 | 2.36 | 3.11 | 0.48 | 2.34 |
| **ReIMTS** | **4.07** | **1.79** | **2.83** | **0.42** | **1.66** |

### 消融实验

| 变体 | MIMIC-III | MIMIC-IV | PhysioNet'12 | Human Activity | USHCN |
|-----|----------|---------|-------------|---------------|-------|
| ReIMTS (完整) | **4.07** | **1.79** | **2.83** | **0.42** | **1.66** |
| rp sample (不分割) | 4.99 | 1.92 | 2.83 | 0.45 | 1.69 |
| rp split (按观测数分割) | 5.02 | 2.36 | 3.20 | 0.61 | 2.31 |
| rp IARF (融合→加法) | 4.20 | 1.84 | 2.79 | 0.47 | 1.89 |
| w/o IARF (去掉融合) | 4.77 | 2.07 | 3.06 | 0.54 | 1.69 |

### 关键发现

- 按时间段分割（ReIMTS）远优于按观测数分割（rp split），在 USHCN 上差距高达 0.65
- 老模型（mTAN、GRU-D）加上 ReIMTS 后可超越更新的模型
- 效率分析：ReIMTS 使用 GraFITi 骨干时，训练速度最快、GPU 内存最小，优于 Warpformer、HD-TTS、Hi-Patch

## 亮点与洞察

1. **保留采样模式的多尺度设计**：不用重采样，而是基于时间段递归分割，简洁且有效
2. **即插即用兼容性**：适配大多数编码器-解码器 IMTS 模型，通用性强
3. **老方法焕新生**：PrimeNet 提升 62.3%、GRU-D 提升 25.8%，说明多尺度补充是关键缺失环节
4. **效率优势**：利用轻量骨干（如 GraFITi）可同时实现最优精度和最优效率

## 局限与展望

- ODE-based 模型与 ReIMTS 结合时缺乏理论解释
- 扩散模型的噪声潜在表示与 ReIMTS 的融合机制不直接兼容
- 时间段长度的选择需人工指定（附录给出了各数据集设置），自适应选择是可探索方向
- 仅验证了预测任务，分类等其他下游任务未探索

## 相关工作与启发

- 与 tPatchGNN、PrimeNet 的关系：它们可看作 ReIMTS 的单尺度特例
- Scaleformer 等规则时序多尺度方法因重采样而破坏采样信息
- 启发：对于其他处理不规则数据的任务（如事件序列、点过程），保留原始时间信息的多尺度方法可能同样有益

## 评分

- 新颖性: ⭐⭐⭐⭐ （基于时间段分割的思路简洁有效，IARF 融合机制设计合理）
- 实验充分度: ⭐⭐⭐⭐⭐ （5个数据集、26个基线、6个骨干、完整消融和效率分析）
- 写作质量: ⭐⭐⭐⭐ （动机清晰，图示直观，与现有方法的对比详尽）
- 价值: ⭐⭐⭐⭐ （即插即用设计实用性强，已开源于 PyOmniTS）

<!-- RELATED:START -->

## 相关论文

- [Learning Time-Scale Invariant Population-Level Neural Representations](../../NeurIPS2025/time_series/learning_time-scale_invariant_population-level_neural_representations.md)
- [FreqCycle: A Multi-Scale Time-Frequency Analysis Method for Time Series Forecasting](../../AAAI2026/time_series/freqcycle_a_multi-scale_time-frequency_analysis_method_for_time_series_forecasti.md)
- [TSPulse: Tiny Pre-Trained Models with Disentangled Representations for Rapid Time Series](tspulse_tiny_pre-trained_models_with_disentangled_representations_for_rapid_time.md)
- [Revitalizing Canonical Pre-Alignment for Irregular Multivariate Time Series Forecasting](../../AAAI2026/time_series/revitalizing_canonical_pre-alignment_for_irregular_multivariate_time_series_fore.md)
- [HyperIMTS: Hypergraph Neural Network for Irregular Multivariate Time Series Forecasting](../../ICML2025/time_series/hyperimts_hypergraph_neural_network_for_irregular_multivariate_time_series_forec.md)

<!-- RELATED:END -->
