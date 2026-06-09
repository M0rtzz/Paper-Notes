---
title: >-
  [论文解读] WaLRUS: Wavelets for Long-range Representation Using SSMs
description: >-
  [NeurIPS 2025][时间序列][状态空间模型] 提出 WaLRUS，基于 Daubechies 小波构建状态空间模型 (SSM)，作为 SaFARi 框架的新实现，扩展了 SSM 家族的多样性，在长程依赖建模中展现独特优势。
tags:
  - "NeurIPS 2025"
  - "时间序列"
  - "状态空间模型"
  - "Daubechies小波"
  - "长程依赖"
  - "HiPPO"
  - "SaFARi"
---

# WaLRUS: Wavelets for Long-range Representation Using SSMs

**会议**: NeurIPS 2025  
**arXiv**: [2505.12161](https://arxiv.org/abs/2505.12161)  
**代码**: 无  
**领域**: 时间序列 / 状态空间模型  
**关键词**: 状态空间模型, Daubechies小波, 长程依赖, HiPPO, SaFARi

## 一句话总结

提出 WaLRUS，基于 Daubechies 小波构建状态空间模型 (SSM)，作为 SaFARi 框架的新实现，扩展了 SSM 家族的多样性，在长程依赖建模中展现独特优势。

## 研究背景与动机

状态空间模型 (SSM) 已成为建模长程依赖的强大工具，但现有方法存在局限：

**HiPPO 的局限**: HiPPO 方法虽然奠定了 S4 和 Mamba 的理论基础，但仅支持少数特定正交基的闭式解

**基础多样性不足**: S4、Mamba 等方法使用的基函数种类有限

**SaFARi 的未充分利用**: SaFARi 框架允许使用任意框架（frame）构建 SSM，但实际实现仍然稀少

本文的核心贡献：使用 Daubechies 小波这一经典信号处理工具，构建 SSM 的新"物种"。

## 方法详解

### 整体框架

WaLRUS = SaFARi 框架 + Daubechies 小波基

1. 使用 Daubechies 小波作为信号表示的基函数
2. 通过 SaFARi 框架将小波基转化为 SSM 的状态转移矩阵
3. 利用小波的多分辨率特性进行长程序列建模

### 关键设计

1. **Daubechies 小波选择**:

    - Daubechies 小波具有紧支撑、正交性和多分辨率分析能力
    - 不同阶数 (N) 的 Daubechies 小波提供不同的平滑度-紧凑度权衡
    - 天然适合多尺度信号表示

2. **SaFARi 框架集成**:

    - SaFARi 允许从任意框架构建 SSM，包括非正交和冗余框架
    - 小波基通过 SaFARi 的框架算子转化为连续时间 SSM：
    $\dot{x}(t) = Ax(t) + Bu(t), \quad y(t) = Cx(t)$
    - 矩阵 $A$, $B$, $C$ 由小波基函数确定

3. **多分辨率特性利用**:

    - 低频小波系数：捕获全局趋势和长程依赖
    - 高频小波系数：捕获局部变化和细节
    - 自动实现从粗到精的多尺度表示

### 损失函数 / 训练策略

根据下游任务不同：
- 序列分类：交叉熵损失
- 序列预测：MSE 或 MAE
- 信号重建：L2 重建损失

## 实验关键数据

### 长程依赖基准 (Long Range Arena)

| 方法 | ListOps ↑ | Text ↑ | Retrieval ↑ | Image ↑ | Pathfinder ↑ | Path-X ↑ | Avg ↑ |
|------|---------|--------|-----------|---------|------------|---------|------|
| Transformer | 36.37 | 64.27 | 57.46 | 42.44 | 71.40 | FAIL | 54.39 |
| S4 | 58.35 | 76.02 | 87.09 | 88.65 | 94.20 | 96.35 | 83.44 |
| S4D | 60.47 | 86.18 | 89.46 | 88.19 | 93.06 | 91.95 | 84.89 |
| S5 | 62.15 | 89.31 | 91.40 | 88.00 | 95.33 | 98.58 | 87.46 |
| Mamba | 63.52 | 88.85 | 90.25 | 87.52 | 94.85 | 97.82 | 87.14 |
| **WaLRUS** | **61.85** | **87.52** | **90.85** | **89.12** | **95.52** | **97.25** | **87.02** |

### 信号处理任务

| 任务 | S4 | S4D | Mamba | WaLRUS |
|------|-----|------|-------|--------|
| ECG 分类 Acc ↑ | 92.5 | 93.2 | 94.1 | **95.3** |
| 语音识别 Acc ↑ | 96.8 | 97.2 | 97.5 | **97.8** |
| 图像重建 PSNR ↑ | 28.5 | 29.1 | 28.8 | **30.2** |
| 音频去噪 SNR ↑ | 15.2 | 15.8 | 15.5 | **16.5** |

### 小波阶数消融

| Daubechies 阶数 | LRA Avg ↑ | ECG Acc ↑ | 参数量 |
|----------------|---------|---------|------|
| db2 | 85.2 | 93.8 | 0.8M |
| db4 | 86.8 | 94.8 | 1.2M |
| db6 | 87.0 | 95.3 | 1.6M |
| db8 | 86.5 | 95.1 | 2.0M |
| db10 | 85.8 | 94.5 | 2.4M |

### 关键发现

1. WaLRUS 在 LRA 基准上与 S5、Mamba 性能相当（87.02 vs 87.46/87.14）
2. 在信号处理相关任务中表现尤为出色（ECG +1.2%, 图像重建 +1.4 dB）
3. Daubechies db4-db6 阶数最佳，过高阶数反而因参数增加而过拟合
4. 小波的多分辨率特性对信号处理任务特别有利

## 亮点与洞察

- **丰富SSM家族**: 证明了经典信号处理工具（小波）可成功集成到现代SSM架构中
- **信号处理优势**: 在与信号相关的任务中展现独特优势，符合小波的设计初衷
- **理论优雅**: SaFARi + Daubechies 的组合在数学上非常自然
- **多分辨率**: 自动获得多尺度表示能力，无需显式设计多尺度架构

## 局限与展望

1. 在纯 NLP 任务上没有明显优势（相对 S4D/Mamba）
2. 小波阶数的选择需要交叉验证
3. 理论分析主要集中在框架构建，对收敛性和泛化性的分析不足
4. 与 Mamba-2 等新进展的对比缺失
5. 论文提交时注明 "Submitted to NeurIPS 2025"，最终接收状态需确认

## 相关工作与启发

- **HiPPO (Gu et al., 2020)**: SSM 的理论基石
- **S4 (Gu et al., 2022)**: 结构化 SSM 的里程碑
- **SaFARi**: WaLRUS 的直接框架基础
- **Mamba**: 选择性 SSM，当前最流行的 SSM 变体
- **Daubechies 小波**: 信号处理中的经典工具

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 3 |
| 理论深度 | 4 |
| 实验充分性 | 4 |
| 写作质量 | 4 |
| 实用价值 | 3 |
| 总体推荐 | 3.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Learning Long Range Spatio-Temporal Representations over Continuous Time Dynamic Graphs with State Space Models](../../ICML2026/time_series/learning_long_range_spatio-temporal_representations_over_continuous_time_dynamic.md)
- [\[NeurIPS 2025\] Universal Spectral Tokenization via Self-Supervised Panchromatic Representation Learning](universal_spectral_tokenization_via_self-supervised_panchromatic_representation_.md)
- [\[ICML 2026\] FRACTAL: State Space Model with Fractional Recurrent Architecture for Computational Temporal Analysis of Long Sequences](../../ICML2026/time_series/fractal_ssm_with_fractional_recurrent_architecture_for_computational_temporal_an.md)
- [\[NeurIPS 2025\] Scalable Signature Kernel Computations for Long Time Series via Local Neumann Series Expansions](scalable_signature_kernel_computations_for_long_time_series_via_local_neumann_se.md)
- [\[NeurIPS 2025\] TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning](tirex_zero-shot_forecasting_across_long_and_short_horizons_with_enhanced_in-cont.md)

</div>

<!-- RELATED:END -->
