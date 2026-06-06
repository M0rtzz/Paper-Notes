---
title: >-
  [论文解读] Emulating Self-Attention with Convolution for Efficient Image Super-Resolution
description: >-
  [ICCV 2025][图像恢复][图像超分辨率] 观察到自注意力在相邻层之间的特征和注意力图高度相似（89%/87%），提出用共享大核卷积和动态卷积核组成的 ConvAttn 模块替代大部分自注意力，同时首次在轻量级超分辨率中引入 Flash Attention 将窗口扩展到 32×32…
tags:
  - "ICCV 2025"
  - "图像恢复"
  - "图像超分辨率"
  - "自注意力替代"
  - "大核卷积"
  - "注意力机制"
  - "轻量级网络"
---

# Emulating Self-Attention with Convolution for Efficient Image Super-Resolution

**会议**: ICCV 2025  
**arXiv**: [2503.06671](https://arxiv.org/abs/2503.06671)  
**代码**: [GitHub](https://github.com/dslisleedh/ESC)  
**领域**: 图像复原  
**关键词**: 图像超分辨率, 自注意力替代, 大核卷积, Flash Attention, 轻量级网络

## 一句话总结

观察到自注意力在相邻层之间的特征和注意力图高度相似（89%/87%），提出用共享大核卷积和动态卷积核组成的 ConvAttn 模块替代大部分自注意力，同时首次在轻量级超分辨率中引入 Flash Attention 将窗口扩展到 32×32，以极低延迟和内存代价实现了 SOTA 性能。

## 研究背景与动机

图像超分辨率（SR）任务中 Transformer 已展现出优于 CNN 的性能，但面临严峻的实际部署问题：

**内存访问瓶颈**：自注意力需要物化 score 矩阵 $S = QK^T$，加上 tensor reshape 和 window masking 等内存密集操作，导致极高的延迟和内存占用。SwinIR-light 比相同 FLOPs 的 CNN 慢 4.7倍、内存高 2倍。

**效率指标的误导**：现有方法主要关注降低 FLOPs 和参数量，但忽视了实际延迟和内存占用——这些才是部署中真正的瓶颈。

**关键发现**：论文通过实验分析了 SwinIR-light 中自注意力的**层间相似性**，发现：
   - 自注意力提取的特征 $F$ 在相邻层间的 CKA 相似度高达 **87%**
   - 注意力图 $A^{avg}$ 在相邻层间的余弦相似度高达 **89%**
   这说明自注意力在多层中提取了**大量重复特征**，存在显著的冗余。

基于此，论文提出了一个大胆的策略：在每个 block 中仅保留**一层**自注意力（用大窗口 + Flash Attention 强化），其余层全部用高效的卷积模块替代。

## 方法详解

### 整体框架

ESC 网络的整体结构包含四个主要部分：
- **浅层特征提取**：$3 \times 3$ 卷积将 LR 输入映射到 $C$ 维特征
- **深层特征提取器 $H$**：由 $N$ 个 ESCBlock 组成，所有 block 共享一个 $13 \times 13$ 的大核 $LK$
- **图像级跳跃模块 $S$**：并行处理 LR 输入
- **上采样器 $U$**：融合深层特征和跳跃特征生成 SR 图像

### 关键设计

1. **ConvAttn 模块（卷积注意力）**:

    - **功能**：用卷积模拟自注意力的两大优势——远程依赖建模和实例依赖加权。
    - **核心思路**：将特征按通道分为两部分：$F^{att} \in \mathbb{R}^{H \times W \times 16}$（前 16 个通道）和 $F^{idt} \in \mathbb{R}^{H \times W \times (C-16)}$（剩余通道）。仅对 $F^{att}$ 应用两种卷积：
        - **共享大核 $LK \in \mathbb{R}^{13 \times 13 \times 16 \times 16}$**：在整个网络中共享，捕捉长程交互（模拟注意力的远程依赖）
        - **动态核 $DK \in \mathbb{R}^{3 \times 3 \times 1 \times 16}$**：通过 GAP + MLP 从输入生成，实现实例依赖加权（模拟注意力的自适应性）
    $F^{res} = (F^{att} \circledast DK) + (F^{att} \circledast LK)$
      然后与 $F^{idt}$ 拼接后通过 $1 \times 1$ 卷积融合。
    - **设计动机**：基于层间特征高度相似的观察，远程交互模式在各层中不需要每次重新计算。共享 $LK$ 减少参数增长和优化难度，$DK$ 提供逐层的输入自适应性。仅对 16 个通道操作大大降低了内存访问开销。

2. **ESCBlock 结构**:

    - **功能**：每个 block 中仅使用一层自注意力，其余 $M$ 层使用 ConvAttn。
    - **核心思路**：
    $F_{i,0} = F_i^{in} + \text{SelfAttn}(\text{LN}(F_i^{in}))$
    $F_{i,j} = F_{i,j-1} + \text{ConvAttn}_j(\text{ConvFFN}_j(F_{i,j-1}), LK), \quad j=1,...,M$
      关键地，ConvFFN 放在自注意力**之前**，这样自注意力在提取特征时已考虑了局部信息，无需复杂的 QKV 投影。ConvAttn 层位于自注意力之后，负责提取窗口间特征（类似 shifted window 的功能）。
    - **设计动机**：既然自注意力特征在层间高度相似，只需一层自注意力建立全局关系，其余层用轻量级卷积维护即可。

3. **Flash Attention 集成**:

    - **功能**：首次在轻量级 SR 任务中成功引入 Flash Attention，将窗口大小扩展到 $32 \times 32$。
    - **核心思路**：Flash Attention 通过避免物化完整的 score 矩阵来减少内存占用。在 $32 \times 32$ 窗口下，相比普通实现可达 **16倍** 延迟降低和 **12.2倍** 内存节省。
    - **设计动机**：大窗口虽然只增加少量 FLOPs，却能显著提升性能。但传统实现中大窗口导致的 score 矩阵过大，Flash Attention 完美解决了这一问题。由于大部分自注意力已被 ConvAttn 替代，剩余的一层自注意力使用大窗口的开销可控。

4. **ESC-FP 变体（FLOPs 优先）**:

    - **功能**：在需要压缩 FLOPs 和参数量的场景下，对大核进行深度可分离分解。
    - **核心思路**：将 $LK$ 分解为逐点核 $LK^c \in \mathbb{R}^{1 \times 1 \times 16 \times 16}$ 和深度核 $LK^s \in \mathbb{R}^{13 \times 13 \times 1 \times 16}$，动态核可通过零填充与 $LK^s$ 合并：
    $F^{res} = (F^{att} \circledast LK^c) \circledast (ZP(DK) + LK^s)$

### 损失函数 / 训练策略

- 使用 $L_1$ 损失在 Y 通道上训练
- 在 DIV2K 数据集和大规模 DFLIP 数据集上分别训练评估
- 训练 patch size 为 $64 \times 64$
- 提供三个模型变体：ESC（延迟优先）、ESC-light、ESC-FP（FLOPs/参数优先）

## 实验关键数据

### 主实验（DIV2K 训练，×4 超分辨率）

| 方法 | Urban100 PSNR | 延迟(ms) | 内存(mb) | FLOPs(G) | 参数(K) |
|------|-------------|---------|---------|----------|---------|
| SwinIR-lt | 26.47 | 222.9 | 351 | 63.6 | 930 |
| ATD-lt | 26.97 | 189.7 | 753 | 100.1 | 769 |
| HiT-SRF | 26.80 | 82.1 | 1331 | 58.0 | 866 |
| MambaIRV2-lt | 26.82 | 153.4 | 748 | 75.6 | 790 |
| **ESC** | **27.07** | **21.9** | **215** | 149.2 | 968 |
| **ESC-FP** | 26.90 | 21.7 | 158 | 60.8 | 539 |

ESC 在 Urban100×4 上超越 ATD-light 0.1 dB PSNR，同时快 **8.7倍**。

### 消融实验

| 配置 | Set5×2 PSNR | Urban100×2 PSNR | 延迟(ms) | 说明 |
|------|-----------|----------------|---------|------|
| 仅自注意力 | 38.27 | 33.23 | 128.2 | 全用 SA，较多层 |
| 仅 ConvAttn | 38.18 | 32.91 | 126.3 | 无 SA，性能明显下降 |
| 9×9 LK | 38.33 | 33.42 | 117.1 | 核太小 |
| 17×17 LK | 38.32 | 33.40 | 126.6 | 核太大，效果反降 |
| 无 LK 共享+无 DK | 38.32 | 33.37 | 119.7 | 基线 |
| 无 DK | 38.31 | 33.36 | 119.7 | DK 贡献显著 |
| WS16+更多层 | 38.24 | 33.05 | 118.0 | 小窗口不如大窗口+少层 |
| **ESC (13×13 LK)** | **38.35** | **33.46** | 120.9 | 最优配置 |

### 关键发现

- 单用 SA 或单用 ConvAttn 都不如两者结合，**互补组合**是最优方案
- $13 \times 13$ 是最优核大小，过大过小都导致性能下降
- LAM 可视化证实 ESC 的感受野与纯 Transformer 相当甚至更大（扩散指数最高）
- 大规模数据训练（DFLIP）时 ESC 的优势更明显——超越 ATD-light 0.27 dB（×4 Urban100），证明保留了 Transformer 的数据可扩展性

## 亮点与洞察

- **层间冗余的发现**：定量证明自注意力在相邻层间有 87-89% 的相似度，为减少自注意力使用提供了坚实依据
- **极端效率**：ESC 在 HD 图像上仅需 21.9ms 延迟（ATD-light 需 189.7ms），接近 CNN 的推理速度
- **仅 16 通道参与注意力操作**：大胆地只对 16/60 通道进行长程建模，其余通道直通，同时不损失性能
- **共享大核**：全网络共享一个大核是一种优雅的参数高效设计，类似于全局的"通用远程交互模板"
- **Flash Attention 的首次 SR 应用**：突破了轻量级 SR 中窗口大小的瓶颈

## 局限与展望

- Flash Attention 目前需要 CUDA 优化，对仅有 CPU 的设备不友好
- 共享大核的 $13 \times 13$ 尺寸对不同任务可能不是最优的
- ConvAttn 中的动态核生成通过 GAP 进行全局池化，可能丢失空间信息
- 论文主要在超分辨率任务上验证，对其他low-level任务（去噪、去模糊）的效果需要进一步验证
- 训练时 patch size 仅 64×64，大窗口（32×32）的padding可能导致训练不稳定

## 相关工作与启发

- 与 ELAN 的注意力共享策略有思想联系，但 ESC 更进一步直接用卷积替代而非共享注意力图
- 与 RepLKNet 等大核 CNN 的差异在于加入了动态核和自注意力的互补设计
- 为"何时需要自注意力、何时卷积足够"的问题提供了定量答案

## 评分

- 新颖性: ⭐⭐⭐⭐ 层间冗余观察虽不全新，但将其转化为具体设计策略（共享大核+动态核+少量SA）是创新的
- 实验充分度: ⭐⭐⭐⭐⭐ 从 DIV2K 到 DFLIP，从 ×2 到 ×4，包含延迟/内存/FLOPs 全面对比，LAM 和特征可视化深入
- 写作质量: ⭐⭐⭐⭐ 动机分析清晰，图表信息量大，但部分公式中的下标较多
- 价值: ⭐⭐⭐⭐⭐ 解决了轻量级 SR 的实际部署痛点，Flash Attention 的引入和 ConvAttn 的设计具有很大的实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Spiking Meets Attention: Efficient Remote Sensing Image Super-Resolution with Attention Spiking Neural Networks](../../NeurIPS2025/image_restoration/spiking_meets_attention_efficient_remote_sensing_image_super-resolution_with_att.md)
- [\[ICCV 2025\] Outlier-Aware Post-Training Quantization for Image Super-Resolution](outlier-aware_post-training_quantization_for_image_super-resolution.md)
- [\[ICCV 2025\] Efficient Concertormer for Image Deblurring and Beyond](efficient_concertormer_for_image_deblurring_and_beyond.md)
- [\[ICCV 2025\] IM-LUT: Interpolation Mixing Look-Up Tables for Image Super-Resolution](im-lut_interpolation_mixing_look-up_tables_for_image_super-resolution.md)
- [\[ICCV 2025\] Self-Calibrated Variance-Stabilizing Transformations for Real-World Image Denoising](self-calibrated_variance-stabilizing_transformations_for_real-world_image_denois.md)

</div>

<!-- RELATED:END -->
