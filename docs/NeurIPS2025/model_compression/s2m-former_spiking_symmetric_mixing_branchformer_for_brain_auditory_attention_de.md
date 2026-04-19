---
title: >-
  [论文解读] S2M-Former: Spiking Symmetric Mixing Branchformer for Brain Auditory Attention Detection
description: >-
  [NeurIPS 2025][模型压缩][脉冲神经网络] 提出 S2M-Former，一种脉冲驱动的对称混合 Branchformer 框架，通过空间-频率双分支的互补学习和轻量化 1D token 表示，在 EEG 听觉注意力检测任务上以仅 0.06M 参数实现了 SOTA 级精度，同时将能耗降低至双分支 ANN 模型的 1/5.8。
tags:
  - NeurIPS 2025
  - 模型压缩
  - 脉冲神经网络
  - 听觉注意力检测
  - 对称混合架构
  - 能效计算
  - EEG
---

# S2M-Former: Spiking Symmetric Mixing Branchformer for Brain Auditory Attention Detection

**会议**: NeurIPS 2025  
**arXiv**: [2508.05164](https://arxiv.org/abs/2508.05164)  
**代码**: [GitHub](https://github.com/JackieWang9811/S2M-Former)  
**领域**: 模型压缩  
**关键词**: 脉冲神经网络, 听觉注意力检测, 对称混合架构, 能效计算, EEG

## 一句话总结

提出 S2M-Former，一种脉冲驱动的对称混合 Branchformer 框架，通过空间-频率双分支的互补学习和轻量化 1D token 表示，在 EEG 听觉注意力检测任务上以仅 0.06M 参数实现了 SOTA 级精度，同时将能耗降低至双分支 ANN 模型的 1/5.8。

## 研究背景与动机

听觉注意力检测（AAD）旨在从 EEG 信号中解码听者在复杂声学环境中关注的说话人，这对开发神经驱动助听设备至关重要。现有方法面临以下核心挑战：

**孤立学习范式**：近期的双分支网络（如 DBPNet、M-DBPNet）虽然使用了多种 EEG 特征，但分支间仅通过简单拼接或求和融合，忽视了特征之间的互补学习潜力。

**计算开销过大**：为建模 EEG 的特定属性（如拓扑结构），这些方法使用 3D 卷积等高开销操作，参数量高达 0.88M-1.32M，不适合部署在低功耗可穿戴设备上。

**能效瓶颈**：助听器和 BCI 系统对电池寿命、延迟和计算资源有严格限制，现有 ANN 模型的高能耗难以满足实际需求。

S2M-Former 的核心思想是：利用脉冲神经网络（SNN）天然的低功耗特性（用稀疏 AC 操作替代 MAC），结合对称双分支设计实现空间-频率特征的互补学习，同时用 1D token 序列替代 3D 操作大幅减少参数。

## 方法详解

### 整体框架

S2M-Former 由三部分组成：(1) 分支特异性脉冲编码器（SBE/FBE）分别提取空间和频率域特征；(2) 脉冲对称混合（S2M）模块实现分支间互补融合；(3) 分类头输出预测。输入 EEG 先通过 CSP 提取空间特征 $E_S \in \mathbb{R}^{C \times T}$，通过 DE 提取频率特征 $E_F \in \mathbb{R}^{5 \times H \times W}$，经时间步扩展和脉冲编码后进入后续处理。

### 关键设计

1. **通道参数化 LIF (CPLIF) 神经元**：在标准 Parametric LIF 基础上，为每个通道分配独立的膜时间常数 $\tau_l[c]$ 和偏置 $\beta[c]$，实现通道级自适应时序建模。膜电位更新公式：

$$H[t,c,n] = V[t-1,c,n] + \frac{1}{\tau_l[c]}(X[t,c,n] - (V[t-1,c,n] - V_{reset})) + \beta[c]$$

设计动机：标准 LIF 所有通道共享同一时间常数，无法捕捉不同频带/通道的差异化时序特性，CPLIF 提供更精细的脉冲激活控制。

2. **双分支脉冲编码器（SBE + FBE）**：

    - **SBE（空间分支）**：通过级联时间卷积（kernel 8→16）逐步提取时间依赖，再用双路空间卷积（kernel $C \times 1$）聚合跨通道交互并残差相加。输出维度 $\mathbb{R}^{T_S \times D \times T}$。
    - **FBE（频率分支）**：使用 3 层膨胀卷积（dilation=2, kernel 3×3）处理 2D 脑拓扑图，通道维度 $D \to 4D \to 2D \to D$，配合最大池化和 1×1 残差卷积。核心创新在于将传统 3D 操作展平为 1D token 序列，参数量降低 14.7 倍。

3. **S2M 模块（四个子模块协同工作）**：

    - **SCSA（脉冲通道自注意力）**：将标准 SSA 的注意力矩阵从 $N \times N$ 改为 $D \times D$ 的通道维度，复杂度从 $O(N^2D)$ 降至 $O(ND^2)$。空间分支揭示电极相关性，频率分支揭示多频带关系。QKV 投影用 1D 深度可分离卷积（kernel=3）实现。
    - **SMSC（脉冲多尺度可分离卷积）**：三个并行深度卷积路径（kernel 1/3/5），捕捉多尺度局部模式。先通过 pointwise conv 扩展至 3D 通道，三路分别处理后求和，再用 channel shuffle 促进跨尺度信息交互，不增加额外参数。
    - **SGCM（脉冲门控通道混合器）**：将空间和频率分支 token 拼接，线性投影至 2D 通道，split 为 query/key 两部分。通过对 query 求和生成通道注意力向量 $A_c \in \mathbb{R}^{T_S \times (N_S+N_F) \times 1}$，对 key 做通道级掩码实现自适应融合。
    - **MPTM（膜电位感知 Token 混合器）**：用 GAP 聚合全局信息，按 $\alpha=0.5$ 比例混合融合分支和原始分支的全局摘要，构建引导表示 $R$，通过脉冲逐元素调制 $F = \mathcal{SN}(X_G) \odot R + X_G$ 实现跨分支融合并保留残差。

### 损失函数 / 训练策略

使用交叉熵损失进行二分类（左/右注意力方向）。所有数据集统一预处理流程（重参考、带通滤波、降采样至 128 Hz），特征提取在训练/验证/测试集上分别进行以防信息泄露。滑动窗口 50% 重叠进行 EEG 分段。模型在所有决策窗口长度下使用固定大小（0.06M），不像 M-DBPNet 那样随窗口变化（1.32M/1.00M/0.88M）。

## 实验关键数据

### 主实验（Within-trial 设置）

| 数据集 | 指标 (2s Acc%) | S2M-Former | DBPNet | M-DBPNet | DARNet | 参数量对比 |
|--------|---------------|------------|--------|----------|--------|-----------|
| KUL | Accuracy ± SD | 93.71 ± 8.14 | 93.66 ± 7.88 | 93.75 ± 6.34 | 92.81 ± 9.45 | **0.06M vs 0.88M** |
| DTU | Accuracy ± SD | **85.28 ± 6.01** | 83.93 ± 5.17 | 82.56 ± 8.01 | 81.30 ± 5.76 | 14.7× 更小 |
| AV-GC | Accuracy ± SD | **91.83 ± 6.66** | 90.78 ± 4.91 | 87.04 ± 7.76 | 89.17 ± 6.94 | 5.8× 低能耗 |

### 消融实验（DTU Within-trial / Cross-trial）

| 配置 | DTU Within-2s | DTU Cross-2s | 说明 |
|------|--------------|--------------|------|
| S2M-Former (完整) | **85.28** | **76.74** | 全部组件 |
| SM-Former (ANN替代) | 80.94 (-4.34) | 73.49 (-3.25) | SNN→ANN，精度显著下降 |
| CPLIF → LIF | 84.13 (-1.15) | 75.67 (-1.07) | 通道级参数化有效 |
| 去除 SGCM+MPTM | 82.98 (-2.30) | 74.81 (-1.93) | 互补学习模块关键 |
| 仅空间分支 | 82.28 (-3.00) | 73.58 (-3.16) | 双分支优于单分支 |
| 仅频率分支 | 70.48 (-14.80) | 70.11 (-6.63) | 空间特征更重要 |

### 能效分析

| 模型 | SNN | 参数(M) | FLOPs(G) | 能耗(mJ) |
|------|-----|---------|----------|----------|
| DBPNet | ✗ | 0.88 | 0.0984 | 0.4526 |
| M-DBPNet | ✗ | 1.32 | 0.1068 | 0.4913 |
| SM-Former (ANN) | ✗ | 0.06 | 0.0243 | 0.1116 |
| **S2M-Former** | ✓ | **0.06** | **0.0112** | **0.0779** |

### 关键发现

- S2M-Former 以 0.06M 参数在 18 个评估条件中赢得 11 项最高精度（61.1%），Top-3 覆盖率 83.33%。
- 能耗仅 0.0779 mJ，比 DBPNet（0.4526 mJ）低 5.8 倍，比 M-DBPNet（0.4913 mJ）低 6.3 倍。
- 跨被试（LOSO）验证中，S2M-Former 在 KUL（75.75%）和 DTU（59.75%）上均最优。
- SNN 版本全面优于 ANN 版本（SM-Former），证明脉冲驱动机制对 EEG 表征学习的有效性。
- 频率分支单独使用时（DE 特征 + FBE）已超过 QKFormer、SDT、Spikformer 三个 SNN 基线。

## 亮点与洞察

1. **对称设计的巧妙性**：空间和频率分支使用镜像模块结构，自然促进互补学习，避免了人工设计复杂融合策略的需要。
2. **1D token 替代 3D 操作**：这一轻量化策略是参数量大幅降低的关键，同时保持甚至提升了性能。
3. **SNN 的实际价值**：在 EEG 这种天然具有时序脉冲特性的信号上，SNN 不仅带来能效优势，还提供了更好的精度（+4.34% vs ANN），与许多领域中 SNN 精度不如 ANN 的常见认知不同。

## 局限与展望

- 在 KUL 和 AV-GC 数据集的跨试验设置下标准差较大（>18%），部分被试低于随机水平（50%），泛化性仍有提升空间。
- 仅验证二分类（左/右注意力）场景，尚未扩展到多说话人场景。
- 缺少在实际神经形态硬件（如 Loihi、SpiNNaker）上的部署验证，能耗估算基于理论计算。
- 极短决策窗口（0.1s）下与部分基线差距扩大。

## 相关工作与启发

- 与 DBPNet/M-DBPNet 的对比表明，高参数量不等于高性能，轻量化设计+合理架构创新可以四两拨千斤。
- CPLIF 的通道级参数化思路可推广到其他 SNN 架构中，为细粒度时序建模提供新方向。
- 对称混合架构的设计范式对多模态融合任务具有参考价值。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次将 SNN 对称混合框架用于 AAD，CPLIF 和 S2M 模块设计新颖
- **实验充分度**: ⭐⭐⭐⭐⭐ 三个数据集、三种评估设置、完整消融和能耗分析
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，公式展示规范
- **价值**: ⭐⭐⭐⭐⭐ 为低功耗 BCI/助听设备提供极具前景的解决方案

<!-- RELATED:START -->

## 相关论文

- [Spiking Brain Compression: Post-Training Second-Order Compression for Spiking Neural Networks](spiking_brain_compression_post-training_second-order_compression_for_spiking_neu.md)
- [Synergy between the Strong and the Weak: Spiking Neural Networks Are Inherently Superior in Temporal Processing](synergy_between_the_strong_and_the_weak_spiking_neural_networks_are_inherently_s.md)
- [BaRISTA: Brain-Scale Informed Spatiotemporal Representation of Human Intracranial EEG](barista_brain_scale_informed_spatiotemporal_representation_of_human_intracranial.md)
- [AI-Generated Video Detection via Perceptual Straightening](ai-generated_video_detection_via_perceptual_straightening.md)
- [DRAGON: Guard LLM Unlearning in Context via Negative Detection and Reasoning](dragon_guard_llm_unlearning_in_context_via_negative_detection_and_reasoning.md)

<!-- RELATED:END -->
