---
title: >-
  [论文解读] FiDeSR: High-Fidelity and Detail-Preserving One-Step Diffusion Super-Resolution
description: >-
  [CVPR 2026][图像恢复][单步扩散超分] 提出 FiDeSR，一种高保真和细节保持的单步扩散超分框架，通过细节感知加权（DAW）、隐空间残差精炼块（LRRB）和潜在频率注入模块（LFIM）三个互补组件，同时解决单步扩散超分中的结构保真度退化和高频细节恢复不足问题。
tags:
  - CVPR 2026
  - 图像恢复
  - 单步扩散超分
  - 频率感知
  - 残差精炼
  - 细节加权
  - 高保真
---

# FiDeSR: High-Fidelity and Detail-Preserving One-Step Diffusion Super-Resolution

**会议**: CVPR 2026  
**arXiv**: [2603.02692](https://arxiv.org/abs/2603.02692)  
**代码**: [GitHub](https://github.com/Ar0Kim/FiDeSR)  
**领域**: 图像超分辨率  
**关键词**: 单步扩散超分, 频率感知, 残差精炼, 细节加权, 高保真

## 一句话总结

提出 FiDeSR，一种高保真和细节保持的单步扩散超分框架，通过细节感知加权（DAW）、隐空间残差精炼块（LRRB）和潜在频率注入模块（LFIM）三个互补组件，同时解决单步扩散超分中的结构保真度退化和高频细节恢复不足问题。

## 研究背景与动机

扩散模型在真实世界图像超分（Real-ISR）中表现出色，但多步扩散推理代价高。单步扩散方法（SinSR、OSEDiff）通过蒸馏压缩迭代过程，但面临两个核心问题：

**高保真度难以保持**：VAE 编码条件导致结构失真和低频不一致（如 AddSR 出现结构扭曲）

**高频细节恢复不足**：
   - 多步扩散通过迭代去噪逐步生成高频细节，单步扩散将此压缩为一步，高频恢复不充分（如 OSEDiff 过度平滑）
   - 最近的残差学习方法（PiSA-SR）仅预测单一全局残差，导致不稳定的高频重建和残差伪影（如 PiSA-SR 生成过多细节）

核心思路：从训练（DAW）、模型架构（LRRB）和推理（LFIM）三个阶段分别针对性解决保真和细节问题。

## 方法详解

### 整体框架

FiDeSR 基于 SD 2.1-base，采用 LoRA 微调的单步扩散框架：
1. **训练阶段**：LQ 图像编码为 $z_L$，U-Net 预测粗残差 $r$，LRRB 精炼得到 $z_r$，DAW 指导损失加权
2. **推理阶段**：单步扩散后，LRRB 精炼残差，LFIM 注入频率增强，VAE 解码输出

### 关键设计

1. **细节感知加权 (DAW)**：自适应地强调模型表现较差的细节丰富区域。构造两个空间图：

    - **细节图 $D$**：融合 Sobel（边缘锐度）、Laplacian（局部对比度）和 Variance（纹理方差）三种空间算子：
    $D = \frac{Sobel(x_H) + Laplacian(x_H) + Variance(x_H)}{3}$
    - **误差图 $E$**：融合像素误差和感知误差：$E = (1-p)E_{pix} + pE_{perc}$
    - **难度权重图**：$W_{DAW} = D \odot E$，对重建损失和 CSD 损失均施加此空间加权
    - 设计动机：让模型专注于边缘、纹理等视觉重要区域，而非在已重建好的平滑区域过拟合

2. **隐空间残差精炼块 (LRRB)**：弥补单步扩散中残差预测的不稳定性和高频恢复不足。基于 RRDB 架构，在隐空间操作：

    - 输入：$z_L$ 和 U-Net 初始残差 $r$ 的拼接
    - 学习校正量 $\Delta r$，精炼残差：$r' = r + \Delta r$
    - 最终隐变量：$z_r = z_L - r'$
    - 设计动机：将 U-Net 预测视为强初始估计，LRRB 学习更精确的校正；与像素域的 ESRGAN 不同，LRRB 专门针对扩散隐空间的残差不稳定性

3. **隐空间频率注入模块 (LFIM)**：推理时无需重训练即可灵活增强频率分量。对精炼隐变量 $z_r$ 进行 FFT 分解：

    - 基于 Butterworth 滤波器分离低频 $\Delta_{LP}$ 和高频 $\Delta_{HP}$
    - **空间门 $M_{sp}$**：基于细节图（Sobel, Laplacian, Variance）识别细节/平坦区域
    - **通道门 $M_{ch}$**：分析每个通道的频率能量比
    - 选择性注入：低频增强结构，高频增强纹理
    - 设计动机：推理时可灵活控制增强强度，无需重新训练

### 损失函数 / 训练策略

总损失 $\mathcal{L}_{total} = \mathcal{L}_{rec} + \mathcal{L}_{reg}$：
- **重建损失**：$\mathcal{L}_{rec} = \lambda_{mse} \cdot W_{DAW} \cdot \text{MSE} + \lambda_{lpips} \cdot W'_{DAW} \cdot \text{LPIPS}$
- **正则化损失**：DAW 加权的 CSD 损失（蒸馏预训练扩散模型的语义先验）
- $\lambda_{mse} = 1$, $\lambda_{lpips} = 2$
- 基座：SD 2.1-base，冻结 VAE 和 U-Net，LoRA rank=8
- 训练：2× H100，batch 8，AdamW，lr $5 \times 10^{-5}$，200K steps
- 文本提示由 RAM 提取

## 实验关键数据

### 主实验

| 数据集 | 指标 | FiDeSR (1s) | PiSA-SR (1s) | OSEDiff (1s) | SeeSR (50s) |
|--------|------|-------------|--------------|--------------|-------------|
| DRealSR | PSNR↑ | **28.90** | 28.32 | 27.92 | 28.14 |
| DRealSR | LPIPS↓ | **0.2836** | 0.2960 | 0.2967 | 0.3141 |
| DRealSR | MANIQA↑ | **0.6239** | 0.6161 | 0.5898 | 0.6016 |
| DRealSR | FID↓ | **127.97** | 130.48 | 135.45 | 146.98 |
| RealSR | LPIPS↓ | **0.2626** | 0.2672 | 0.3194 | 0.3004 |
| RealSR | FID↓ | **109.68** | 124.18 | 123.49 | 125.09 |
| DIV2K | DISTS↓ | **0.1845** | 0.1934 | 0.1975 | 0.1966 |

注：FiDeSR 仅用 **1 步**推理，在全参考和无参考指标上均优于多数单步和部分多步方法，FID 在所有方法中最低。

### 消融实验

| 配置 | CLIPIQA↑ | NIQE↓ | MUSIQ↑ | MANIQA↑ | 说明 |
|------|----------|-------|--------|---------|------|
| 无 LRRB + 无 DAW | 0.6611 | 4.7381 | 67.60 | 0.6237 | 基线 |
| 仅 DAW | 0.6641 | 4.7129 | 67.63 | 0.6236 | DAW 轻微提升 |
| 仅 LRRB | 0.6626 | 4.7340 | 67.95 | 0.6278 | LRRB 提升更显著 |
| DAW + LRRB | **0.6699** | **4.6300** | **68.29** | **0.6285** | 互补效果最佳 |

### 关键发现

- FiDeSR 是首个同时在全参考和无参考指标上达到最优平衡的单步扩散 SR 方法
- LRRB 将高频噪声预测误差平均降低 1.62%（DIV2K 1.24%, DRealSR 1.99%, RealSR 1.62%）
- LFIM 的低频注入提升 PSNR/SSIM（结构保真），高频注入提升 MUSIQ/MANIQA（感知质量），可灵活权衡
- 在所有数据集上 FID 最低，说明生成分布最接近真实图像分布

## 亮点与洞察

1. **问题分析精准**：清晰识别了单步扩散 SR 的两个核心瓶颈（保真 vs 细节），并从训练/架构/推理三个阶段针对性设计
2. **DAW 的双重引导**：同时利用细节图（"哪里重要"）和误差图（"哪里差"），比纯频率加权更智能
3. **LRRB 的设计合理性**：将 RRDB 的残差精炼思想引入扩散隐空间，专门解决扩散残差不稳定问题
4. **LFIM 的灵活性**：推理时可调节增强强度，无需重训练，实用性强
5. **感知-失真权衡的突破**：FiDeSR 在二者之间取得了比现有方法更好的平衡

## 局限与展望

1. 基于 SD 2.1-base，可能受限于基座模型的生成能力
2. LFIM 的频率分离依赖 Butterworth 滤波器参数的手动设定
3. DAW 的误差图计算增加了训练时的额外开销
4. 未探索更高效的单步蒸馏策略（如一致性模型）
5. 可扩展到视频超分或多模态修复任务

## 相关工作与启发

- **PiSA-SR**：残差学习的单步扩散 SR，FiDeSR 的 LRRB 直接改进其粗残差预测
- **OSEDiff**：基于 VSD + LoRA 的单步 SR，高频恢复不足
- **GuideSR**：通过全分辨率引导提升保真度，但感知质量受限
- **TFDSR**：将频率信息融入多步扩散，FiDeSR 将类似思想压缩到单步
- 启发：单步扩散的瓶颈不在于步数，而在于残差预测的精度和频率分量的控制

## 评分

- 新颖性: ⭐⭐⭐⭐ 三个组件各有新意，DAW 的双重引导和 LRRB 的扩散隐空间残差精炼设计独特
- 实验充分度: ⭐⭐⭐⭐ 3 个数据集、9 种指标、与 8 种方法对比、消融全面
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，三个组件的动机和关联讲解流畅
- 价值: ⭐⭐⭐⭐ 单步扩散 SR 的实用性强，同时解决保真和细节恢复是重要贡献

<!-- RELATED:START -->

## 相关论文

- [Disentangled Textual Priors for Diffusion-based Image Super-Resolution](disentangled_textual_priors_for_diffusion-based_image_super-resolution.md)
- [Detail-Preserving Latent Diffusion for Stable Shadow Removal](../../CVPR2025/image_restoration/detail-preserving_latent_diffusion_for_stable_shadow_removal.md)
- [One-Step Event-Driven High-Speed Autofocus](../../CVPR2025/image_restoration/one-step_event-driven_high-speed_autofocus.md)
- [Statistical Characteristic-Guided Denoising for Rapid High-Resolution Transmission Electron Microscopy Imaging](statistical_characteristic-guided_denoising_for_rapid_high-resolution_transmissi.md)
- [Bridging the Perception Gap in Image Super-Resolution Evaluation](bridging_the_perception_gap_in_image_super-resolution_evaluation.md)

<!-- RELATED:END -->
