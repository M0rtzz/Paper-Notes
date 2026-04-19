---
title: >-
  [论文解读] LSCD: Lomb-Scargle Conditioned Diffusion for Time Series Imputation
description: >-
  [ICML 2025][图像生成][时间序列填补] 提出 LSCD，将可微的 Lomb-Scargle 周期图层集成到 score-based 扩散模型中用于时间序列填补，通过频域条件信息和频谱一致性损失，在高缺失率下同时提升时域填补精度和频域恢复一致性。
tags:
  - ICML 2025
  - 图像生成
  - 时间序列填补
  - Lomb-Scargle周期图
  - 频域条件生成
  - 不规则采样
  - 可微频谱层
---

# LSCD: Lomb-Scargle Conditioned Diffusion for Time Series Imputation

**会议**: ICML 2025  
**arXiv**: [2506.17039](https://arxiv.org/abs/2506.17039)  
**代码**: 无  
**领域**: 扩散模型 / 时间序列  
**关键词**: 时间序列填补, Lomb-Scargle周期图, 频域条件生成, 不规则采样, 可微频谱层

## 一句话总结
提出 LSCD，将可微的 Lomb-Scargle 周期图层集成到 score-based 扩散模型中用于时间序列填补，通过频域条件信息和频谱一致性损失，在高缺失率下同时提升时域填补精度和频域恢复一致性。

## 研究背景与动机
**领域现状**：时间序列缺失值填补方法主要在时域工作（BRITS、SAITS、CSDI），部分方法利用 FFT 提取频域特征（TimesNet），但 FFT 要求均匀采样。

**现有痛点**：数据缺失时 FFT 需先插值或零填充，高缺失率下会产生严重的频谱失真。现有扩散填补方法（CSDI）也仅在时域操作，忽略了信号的频率结构。

**核心矛盾**：不规则/缺失采样数据的频谱分析是一个根本性难题，FFT 的均匀采样假设与现实不符。

**本文目标** 如何为不规则采样数据提供可靠的频谱估计，并将其融入扩散生成过程？

**切入角度**：Lomb-Scargle 周期图天然支持不规则采样的频谱分析，且可微化后融入端到端学习。

**核心 idea**：用可微 Lomb-Scargle 层替代 FFT，为条件扩散模型提供无需插值的频域条件信息。

## 方法详解

### 整体框架
LSCD 在 CSDI 的条件扩散框架基础上：(1) 用 Lomb-Scargle 层从观测数据计算频谱，作为去噪网络的额外条件输入；(2) 通过注意力编码器将频谱编码为条件表示；(3) 训练后期引入频谱一致性损失对齐填补结果与观测频谱。

### 关键设计

1. **可微 Lomb-Scargle 层**:

    - 功能：从不规则/缺失采样数据直接计算功率谱密度
    - 核心思路：$P(\omega) = \frac{(\sum_i [x_{s_i} - \bar{x}]\cos[\omega\phi_i])^2}{\sum_i \cos^2[\omega\phi_i]} + \frac{(\sum_i [x_{s_i} - \bar{x}]\sin[\omega\phi_i])^2}{\sum_i \sin^2[\omega\phi_i]}$，其中 $\phi_i = s_i - \tau$ 保证时移不变性
    - 设计动机：无需插值，直接拟合正弦函数到观测点，适用于任意缺失模式

2. **注意力频谱编码器 $\mathcal{E}_{\text{spec}}$**:

    - 功能：将 LS 频谱编码为条件表示注入去噪过程
    - 核心思路：两层多头自注意力，捕获频率间和特征间的依赖关系，生成 $\mathbf{z}_S$ 作为每步去噪的额外条件
    - 设计动机：完整频谱包含丰富信息，需要学习哪些频率成分与填补最相关

3. **频谱一致性损失 $\mathcal{L}_{\text{SCons}}$**:

    - 功能：在训练后期保证填补结果的频谱与观测频谱一致
    - 核心思路：$\mathcal{L}_{\text{SCons}} = \|\mathcal{LS}(\mathbf{x}_0^{co}) - \mathcal{LS}(\hat{\mathbf{x}}_0^{co})\|_2^2$，比较观测部分的原始频谱和重建后的频谱
    - 设计动机：时域损失不能保证频域一致性，此损失确保频率结构被保留

### 损失函数 / 训练策略
- 主损失：标准扩散去噪 loss $\mathcal{L} = \mathbb{E}[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\|^2]$
- 后期增加频谱一致性损失 $\mathcal{L}_{\text{SCons}}$，需通过完整反向扩散得到 $\hat{\mathbf{x}}_0$
- 使用虚假告警概率(FAP)过滤不可靠的频率成分

## 实验关键数据

### 主实验
| 数据集 | 缺失率 | 指标 | CSDI | SAITS | **LSCD** |
|--------|--------|------|------|-------|----------|
| Sines (point) | 10% | MAE ↓ | 1.336 | 0.885 | **0.765** |
| Sines (point) | 50% | MAE ↓ | 1.359 | 1.041 | **0.975** |
| Sines (point) | 90% | MAE ↓ | 1.361 | 1.292 | **1.271** |
| Sines (point) | 10% | S-MAE ↓ | 0.008 | 0.043 | **0.003** |
| Sines (point) | 90% | S-MAE ↓ | 0.044 | 0.375 | **0.036** |

### 消融实验
| 组件 | 效果 | 说明 |
|------|------|------|
| 无 LS 条件（baseline CSDI）| S-MAE 最高 | 无频域信息 |
| + LS 条件 | S-MAE 显著下降 | 频域引导最关键 |
| + 频谱编码器 | MAE 进一步下降 | 学习频率权重 |
| + 频谱一致性损失 | 两指标均最优 | 频域对齐 |

### 关键发现
- LS 条件对频谱恢复（S-MAE）的贡献最为显著（CSDI 0.044 vs LSCD 0.036 在 90% 缺失）
- 在高缺失率（90%）下 LSCD 优势更明显，因为 FFT 在高缺失率下完全失效
- 序列缺失和块缺失模式下 LSCD 比 CSDI 提升更大

## 亮点与洞察
- **可微 LS 层的通用性**：不仅适用于填补，可集成到任何需要处理不规则采样频域信息的深度学习 pipeline
- **FFT vs LS 的系统对比**：清楚展示了 FFT 在缺失数据下的频谱失真问题
- **可迁移思路**：LS 条件化的思路可迁移到时间序列预测、异常检测等任务

## 相关工作与启发
- **vs CSDI**: CSDI 是纯时域条件扩散填补的代表，LSCD 在其架构上增加了频域条件和一致性损失
- **vs TimesNet**: TimesNet 用 FFT 提取周期特征，但必须先插值缺失值；LSCD 的 LS 层直接处理缺失
- **vs BRITS/SAITS**: 这些确定性方法无法提供不确定性估计；LSCD 作为概率模型可提供多次采样的分布
- 可微 LS 层可独立于 LSCD 使用，集成到任何需频域分析的深度学习 pipeline

## 局限与展望
- LS 层时间复杂度为 $O(LJ)$（L 个观测点 × J 个频率），比 FFT 的 $O(L\log L)$ 慢
- 仅在合成正弦波和两个真实数据集上验证，缺少大规模时序基准（如 ETTh、Weather）
- 频谱一致性损失需要完整反向扩散，增加了训练成本
- 非周期信号（如趋势性时序）的频谱条件效果未验证
- 多变量间频谱相关性的建模未充分利用

## 评分
- 新颖性: ⭐⭐⭐⭐ 将 Lomb-Scargle 引入深度学习是有意义的第一步
- 实验充分度: ⭐⭐⭐ 数据集偏少，缺少大规模验证
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰，背景全面
- 价值: ⭐⭐⭐⭐ 为不规则时序的频域建模提供了可行方案

<!-- RELATED:START -->

## 相关论文

- [BRIDGE: Bootstrapping Text to Control Time-Series Generation via Multi-Agent Iterative Optimization and Diffusion Modeling](bridge_bootstrapping_text_to_control_time-series_generation_via_multi-agent_iter.md)
- [A Diffusion Model for Regular Time Series Generation from Irregular Data with Completion and Masking](../../NeurIPS2025/image_generation/a_diffusion_model_for_regular_time_series_generation_from_irregular_data_with_co.md)
- [TIDMAD: Time Series Dataset for Discovering Dark Matter with AI Denoising](../../NeurIPS2025/image_generation/tidmad_time_series_dataset_for_discovering_dark_matter_with_ai_denoising.md)
- [Conditionally Whitened Generative Models for Probabilistic Time Series Forecasting](../../ICLR2026/image_generation/conditionally_whitened_generative_models_for_probabilistic_time_series_forecasti.md)
- [SimDiff: Simpler Yet Better Diffusion Model for Time Series Point Forecasting](../../AAAI2026/image_generation/simdiff_simpler_yet_better_diffusion_model_for_time_series_point_forecasting.md)

<!-- RELATED:END -->
