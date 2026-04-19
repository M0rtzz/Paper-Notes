---
title: >-
  [论文解读] UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution
description: >-
  [CVPR 2026][图像恢复][image_restoration] 提出 UCAN，一种统一卷积与注意力的轻量级超分网络，通过 Hedgehog Attention 突破线性注意力的低秩瓶颈，结合 Flash Attention 大窗口建模、大核蒸馏模块和跨层参数共享，在极低计算量下实现了与大模型可比的超分性能。
tags:
  - CVPR 2026
  - 图像恢复
  - image_restoration
  - super_resolution
  - lightweight
  - 注意力机制
---

# UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution

**会议**: CVPR 2026  
**arXiv**: [2603.11680](https://arxiv.org/abs/2603.11680)  
**代码**: 无  
**领域**: 图像恢复  
**关键词**: image_restoration, super_resolution, lightweight, attention, linear_attention  

## 一句话总结

提出 UCAN，一种统一卷积与注意力的轻量级超分网络，通过 Hedgehog Attention 突破线性注意力的低秩瓶颈，结合 Flash Attention 大窗口建模、大核蒸馏模块和跨层参数共享，在极低计算量下实现了与大模型可比的超分性能。

## 背景与动机

轻量级超分辨率面临核心矛盾：**扩大有效感受野**对捕获全局上下文至关重要，但扩展注意力窗口或卷积核会显著增加计算开销。

现有方法的局限：
- **CNN 方法**：局限于局部感受野
- **Transformer 方法**（SwinIR、ELAN）：窗口注意力限制全局建模
- **线性注意力**：虽然 $O(N)$ 复杂度，但特征多样性不足导致**秩坍塌**（rank collapse），表达能力受限
- **Mamba/SSM 方法**：全局建模仍存在效率问题

## 方法详解

### 整体架构

UCAN 采用分层设计：浅层特征提取 → Broad Effective Receptive Field Group (BERFG) 深层编码 → 重建模块（卷积 + PixelShuffle）。

### 1. Hedgehog Attention — 突破线性注意力低秩瓶颈

标准线性注意力将 softmax 替换为特征映射 $\phi$，复杂度从 $O(N^2)$ 降至 $O(N)$：

$$\mathbf{o}_i^L = \frac{\phi(\mathbf{q}_i)^\top \left(\sum_j \phi(\mathbf{k}_j) \mathbf{v}_j^\top\right)}{\phi(\mathbf{q}_i)^\top \left(\sum_j \phi(\mathbf{k}_j)\right)}$$

但 ReLU/ELU 等简单映射导致注意力矩阵秩严重不足。UCAN 采用 Hedgehog Feature Map (HFM)，通过对称指数对拼接：

$$\phi_H(\mathbf{X}) = [\exp(\mathbf{W}^\top \mathbf{X} + \mathbf{b}_1), \ldots, \exp(-\mathbf{W}^\top \mathbf{X} - \mathbf{b}_m)]$$

其中 $\mathbf{W}$ 为共享投影，$\{\mathbf{b}_i\}$ 为可学习偏置。HFM 的对称配对机制保留正负方向信息，实验显示秩恢复至 46（ReLU/ELU 远低于此）。

### 2. 半共享混合注意力 (Semi-Sharing Hybrid Attention)

BERFG 包含 Sharing Block 和 Receiving Block：

- **Sharing Block**：完整计算 Shared Window MHA 的注意力图 $\mathbf{A}_{qk}$ 和 Hedgehog Attention 的特征映射 $\mathbf{A}_{map}$
- **Receiving Block**：直接复用 Sharing Block 计算的注意力，仅重新计算值投影

Dual Fusion Layer 整合两个分支：
- **空间分支**（Hedgehog Attention）：$\mathbf{F}_{sb} = \phi(\mathbf{Q}) \phi(\mathbf{K})^\top \mathbf{V} + \mathbf{W}_d \mathbf{V}$
- **通道分支**（Channel Attention）：$\mathbf{F}_{cb} = \text{softmax}(\mathbf{Q}^\top \mathbf{K}) \mathbf{V}$

总复杂度为空间分辨率的线性函数：
$$\mathcal{O}_{\text{DFL}} = \underbrace{2C^2 HW}_{\text{channel}} + \underbrace{6HW \frac{C^2}{D} + 9HWC}_{\text{spatial}}$$

### 3. High Performance Attention (HPA)

使用 ConvMLP（7×7 核）捕获局部上下文，配合 **Flash Attention** 实现 32×32 大窗口注意力。Flash Attention 在 128×128 分辨率下比原生注意力快 **13.4 倍**。

### 4. Large Kernel Distillation (LKD)

将通道分为细粒度子集和粗粒度子集（$C_{fg} = \max(C/4, 16)$），仅对细粒度通道执行三分支特征提取（通道注意力 + 小核局部 + 分层大核膨胀卷积），粗粒度通道直接旁路，以极低开销扩展感受野。

## 实验结果

### 表1：轻量级超分 ×4 定量对比

| 方法 | 参数量 | MACs | Set5 PSNR | Urban100 PSNR | Manga109 PSNR |
|------|--------|------|-----------|---------------|---------------|
| OmniSR | 792K | 50.9G | 32.49 | 26.64 | 31.02 |
| ATD-light | 769K | 100.1G | 32.63 | 26.97 | 31.48 |
| MambaIRV2-lt | 790K | 75.6G | 32.51 | 26.82 | 31.24 |
| ESC | 968K | 149.2G | 32.68 | 27.07 | 31.54 |
| **UCAN** | **705K** | **38.1G** | **32.65** | 26.89 | 31.50 |
| **UCAN-L** | **902K** | **48.4G** | **32.68** | 27.06 | **31.63** |

### 表2：注意力机制效率对比（128×128 输入）

| 模块 | 延迟 (ms) | 参数量 |
|------|-----------|--------|
| Naive Self-Attention | 2576.75 (1.0×) | 0.082M |
| Flash Attention | 191.80 (13.4×) | - |
| Dual Fusion Layer | 1294.83 (2.0×) | 0.014M |

UCAN-L 在 Manga109 ×4 上以 48.4G MACs 达 31.63 dB，超越 MambaIRV2 0.39 dB 且计算量降低 36%。

## 亮点与创新

- **Hedgehog Attention** 从根本上解决线性注意力的秩坍塌问题，比 ReLU/ELU 显著提升特征多样性
- **半共享机制**在 Sharing/Receiving Block 间复用注意力计算，接近参数共享的效率但保留表达多样性
- **Flash Attention 大窗口**在轻量级 SR 中首次使用 32×32 窗口而不增加延迟
- **大核蒸馏**仅在 1/4 通道上执行重计算，高效扩展感受野
- 极致的参数效率：705K 参数 + 38.1G MACs 即可与 950K+ 参数方法竞争

## 不足与局限

- 仅在经典 SR（bicubic 降采样）上验证，未测试真实世界退化场景
- Hedgehog Feature Map 的训练开销（指数运算）未充分讨论
- 消融实验仅在 Set5 和 Urban100 上进行，覆盖面有限
- 未与同期其他线性注意力改进方法（如 FLatten Transformer）对比
- BSDS100 上 UCAN 表现不如几个竞争方法（27.79 vs 27.80），边缘场景优势不明显

## 评分

⭐⭐⭐⭐ — 技术贡献扎实，Hedgehog Attention 和半共享机制思路新颖实用；在极低计算预算下实现了出色的精度-效率平衡。

<!-- RELATED:START -->

## 相关论文

- [Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](toward_real-world_infrared_image_super-resolution_a_unified_autoregressive_frame.md)
- [Emulating Self-Attention with Convolution for Efficient Image Super-Resolution](../../ICCV2025/image_restoration/emulating_self-attention_with_convolution_for_efficient_image_super-resolution.md)
- [SpatioTemporal Difference Network for Video Depth Super-Resolution](../../AAAI2026/image_restoration/spatiotemporal_difference_network_for_video_depth_super-resolution.md)
- [Spiking Meets Attention: Efficient Remote Sensing Image Super-Resolution with Attention Spiking Neural Networks](../../NeurIPS2025/image_restoration/spiking_meets_attention_efficient_remote_sensing_image_super-resolution_with_att.md)
- [UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization](unirain_unified_image_deraining_with_rag-based_dataset_distillation_and_multi-ob.md)

<!-- RELATED:END -->
