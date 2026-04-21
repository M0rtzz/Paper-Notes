---
title: >-
  [论文解读] Rotary Masked Autoencoders are Versatile Learners
description: >-
  [NeurIPS 2025][时间序列][Masked Autoencoder] 提出 RoMAE，将旋转位置编码（RoPE）扩展到连续位置并与掩码自编码器（MAE）结合，无需任何时间序列特定的架构修改即可在不规则时间序列、图像和音频等多种模态上达到或超越专用模型的性能。
tags:
  - NeurIPS 2025
  - 时间序列
  - Masked Autoencoder
  - RoPE
  - 不规则时间序列
  - 多模态
  - 自监督预训练
---

# Rotary Masked Autoencoders are Versatile Learners

**会议**: NeurIPS 2025  
**arXiv**: [2505.20535](https://arxiv.org/abs/2505.20535)  
**代码**: [GitHub](https://chromeilion.github.io/RoMAE-Website/)  
**领域**: 时间序列  
**关键词**: Masked Autoencoder, RoPE, 不规则时间序列, 多模态, 自监督预训练

## 一句话总结

提出 RoMAE，将旋转位置编码（RoPE）扩展到连续位置并与掩码自编码器（MAE）结合，无需任何时间序列特定的架构修改即可在不规则时间序列、图像和音频等多种模态上达到或超越专用模型的性能。

## 研究背景与动机

Transformer 已在视觉和 NLP 领域取得巨大成功，但将其应用于不规则采样的时间序列时面临根本性限制：原始 Transformer 仅支持离散整数位置编码，无法处理非均匀采样数据中的连续时间戳。

现有解决方案主要有两类：

**修改 Transformer 内部架构**：如修改前馈层或使用 Neural ODE 作为位置编码，但增加了计算开销和方法复杂度

**使用 State Space Model**：如 Mamba、S5 天然支持多模态，但脱离了 Transformer 生态系统

**核心洞察**：RoPE（旋转位置编码）虽然最初为离散文本位置设计，但其旋转矩阵公式 $R(\theta, m)$ 中的位置 $m$ 天然可以取任意实数值。如果利用这一特性，就能在不修改任何 Transformer 架构的前提下处理连续位置信息，从而继承 Transformer/MAE 生态中的所有优化和发展。

## 方法详解

### 整体框架

RoMAE 遵循 MAE 的非对称编码器-解码器结构（大编码器+小解码器），并引入以下创新：(1) N维 patchification 支持任意模态输入；(2) 连续轴向 RoPE 编码位置信息；(3) p-RoPE 截断策略增强鲁棒性。

### 关键设计

1. **连续轴向 RoPE（Continuous Axial RoPE）**：将标准 RoPE 从离散整数位置扩展到连续实数位置。对于 $D$ 维输入，使用轴向 RoPE 将嵌入空间分为 $D$ 个子空间，每个子空间编码一个维度的连续位置。RoPE 的旋转公式为：

   $$\begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix} x_m^{(i)}$$

   其中 $m \in \mathbb{R}$ 可以是任意实数（时间戳），$\theta_i = 10000^{-2(i-1)/d_x}$。采用 $p$-RoPE（$p=0.75$）只保留 75% 最小的 $\theta_i$，留出部分嵌入空间作为不受 RoPE 修改的数据通道，增强模型对变长序列的鲁棒性。

2. **N维 Patchification**：定义 patch 大小 $(p_1, \ldots, p_D)$，将输入按各维度切分为不重叠的 patch。关键约束：对于任何不规则维度 $d_i$，对应 patch 大小 $p_i$ 必须为 1（因为不规则采样下 patch 内点数不固定）。所有 patch 被展平为单一序列，模型可跨所有维度联合建模。

3. **[CLS] token 与绝对位置恢复**：由于 RoPE 是相对位置编码，模型具有平移不变性。作者从理论上证明：当包含可学习 [CLS] token 时，模型可以恢复绝对位置信息（[CLS] 提供锚点）；不包含时则只有相对位置，使预训练更难但可能有益于平移不变的任务。

### 损失函数 / 训练策略

- **预训练**：掩码自编码目标，均匀随机掩码 75% 的 patch，解码器预测被掩码 patch 的原始值
- **图像预训练**：使用归一化 patch 值计算损失（follow MAE）
- **架构细节**：使用 SiLU 激活函数和 RMSNorm（follow LLaMA），比标准 LayerNorm 更高效
- **微调**：移除解码器，在编码器输出上接任务特定头

## 实验关键数据

### 主实验（不规则时间序列分类 — ELAsTiCC）

| 方法 | F-score | 特点 |
|------|---------|------|
| Transformer | 0.526 | 标准架构 |
| ATAT（专用架构） | 0.627 | 专为 ELAsTiCC 设计 |
| RoMAE-tiny-shallow | 0.711 | 与 ATAT 相当参数量 |
| **RoMAE-tiny** | **0.803** | +0.18 优势 |

### 消融实验（多模态表现汇总）

| 任务/数据集 | RoMAE | 对比方法 | 对比方法名 | 说明 |
|-----------|-------|---------|----------|-----|
| Tiny ImageNet 分类 | 0.500 (无CLS) | 0.479 (绝对PE) | MAE | RoPE≥绝对位置 |
| ESC-50 音频 (AudioSet-20k) | **84.7%** | 82.2% | SSAST | 超越同条件 SSAST |
| Pendulum 回归 MSE×10⁻³ | **3.32** | 3.41 (S5), 4.63 (ContiFormer) | S5/ContiFormer | 无需预训练即超越 |
| PhysioNet 插值 MSE | **0.467** | 0.562 | HeTVAE | 稀疏通道更均衡 |
| Spirals 插值 RMSE | **0.018** | 0.49 | ContiFormer | 数量级提升 |

### 关键发现

1. **跨模态通用性**：单一 RoMAE 架构在图像 (ImageNet)、音频 (ESC-50)、不规则时序 (ELAsTiCC) 和插值任务上均达到竞争力或最优表现
2. **无需架构特化**：不需要任何时间序列特定的架构修改，纯粹使用标准 Transformer 组件
3. **[CLS] token 打破相对性**：实验证实有 [CLS] 时位置重建 MSE 降至 0.003，无 [CLS] 时为 200.33（完全无法恢复）
4. **MAE 预训练对不规则时间序列特别有效**：在 ELAsTiCC 上比没有预训练的 ATAT 提升 18%
5. **数据高效**：在仅数百样本的 UEA 数据集上也能取得良好表现

## 亮点与洞察

- **极致的简洁性**：不发明新架构，仅利用 RoPE 的连续位置扩展 + MAE 标准框架，却获得了跨模态的强大能力
- **理论洞察深刻**：关于 [CLS] token 恢复绝对位置的理论证明和实验验证很有启发性
- **实践意义大**：证明了 Transformer 生态内的标准工具足以处理不规则时间序列，无需切换到 SSM 等新范式

## 局限与展望

1. 连续位置 RoPE 在位置每次前向传播都变化时有额外计算开销
2. 使用标准 Attention 的 $O(n^2)$ 内存复杂度限制了长序列处理
3. 在外推（extrapolation）任务上能力有限
4. N维 patchification 的 token 数随维度指数增长，目前仅用到 3 维

## 相关工作与启发

本文巧妙结合了 RoPE（RoFormer）和 MAE（He et al.）两个成功方向，证明了"标准工具的巧妙组合"有时优于"全新架构设计"。对于未来工作，将 RoPE 与线性注意力结合可突破长序列瓶颈。

## 评分

- **总体评价**: 用最少的架构改动实现了最大的模态覆盖能力，是"less is more"理念的典范
- **新颖性**: ⭐⭐⭐⭐ RoPE 连续扩展虽简单但洞察深刻，理论分析扎实
- **实验充分度**: ⭐⭐⭐⭐⭐ 覆盖 5 种任务/模态，与大量基线比较
- **写作质量**: ⭐⭐⭐⭐⭐ 理论与实验交织，表述清晰
- **价值**: ⭐⭐⭐⭐⭐ 为不规则时间序列学习提供了通用且优雅的解决方案

<!-- RELATED:START -->

## 相关论文

- [IMTS is Worth Time × Channel Patches: Visual Masked Autoencoders for Irregular Multivariate Time Series Prediction](../../ICML2025/time_series/imts_is_worth_time_times_channel_patches_visual_masked_autoencoders_for_irregula.md)
- [Universal Spectral Tokenization via Self-Supervised Panchromatic Representation Learning](universal_spectral_tokenization_via_self-supervised_panchromatic_representation_.md)
- [Time-O1: Time-Series Forecasting Needs Transformed Label Alignment](time-o1_time-series_forecasting_needs_transformed_label_alignment.md)
- [Probability Calibration for Precipitation Nowcasting](probability_calibration_for_precipitation_nowcasting.md)
- [TimePerceiver: An Encoder-Decoder Framework for Generalized Time-Series Forecasting](timeperceiver_an_encoder-decoder_framework_for_generalized_time-series_forecasti.md)

<!-- RELATED:END -->
