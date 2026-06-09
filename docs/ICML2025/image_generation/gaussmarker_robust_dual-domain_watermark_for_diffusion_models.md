---
title: >-
  [论文解读] GaussMarker: Robust Dual-Domain Watermark for Diffusion Models
description: >-
  [ICML 2025][图像生成][扩散模型] 提出 GaussMarker——首个双域（空间+频率）扩散模型水印方法，通过流水线注入器在初始高斯噪声的空间域和频率域一致嵌入水印，配合模型无关的可学习高斯噪声修复器（GNR）增强对旋转/裁剪攻击的鲁棒性…
tags:
  - "ICML 2025"
  - "图像生成"
  - "扩散模型"
  - "dual-domain"
  - "Gaussian noise restorer"
  - "tuning-free"
  - "robustness"
---

# GaussMarker: Robust Dual-Domain Watermark for Diffusion Models

**会议**: ICML 2025  
**arXiv**: [2506.11444](https://arxiv.org/abs/2506.11444)  
**代码**: 无  
**领域**: 扩散模型 / 图像生成  
**关键词**: diffusion model watermark, dual-domain, Gaussian noise restorer, tuning-free, robustness

## 一句话总结

提出 GaussMarker——首个双域（空间+频率）扩散模型水印方法，通过流水线注入器在初始高斯噪声的空间域和频率域一致嵌入水印，配合模型无关的可学习高斯噪声修复器（GNR）增强对旋转/裁剪攻击的鲁棒性，在三个 Stable Diffusion 版本上八种图像扭曲下达到平均 TPR@1%FPR 0.997 的 SOTA 性能。

## 研究背景与动机

**领域现状**：扩散模型生成的图像日益逼真，版权保护和滥用检测需求迫切。水印技术是一种有前途的解决方案。现有的 in-processing 方法分为需要微调模型参数的和免微调的两类，免微调方法因不改变模型权重而更受青睐。

**现有痛点**：Gaussian Shading 等免微调 SOTA 方法仅在初始高斯噪声的单个域（空间域或频率域）注入水印。这导致其对某些简单图像编辑极其脆弱——例如仅旋转 3° 就使 Gaussian Shading 的检测准确率从 100% 暴跌至 64%，完全无法用于实际部署。

**核心矛盾**：空间域水印对旋转/裁剪敏感（像素位置被打乱），频率域水印对某些噪声攻击敏感。单域方法只能在一类攻击上鲁棒，无法同时应对多类攻击。

**本文目标**：设计同时在空间和频率域嵌入水印的免微调方案，并解决旋转/裁剪攻击下的鲁棒性难题。

**切入角度**：传统图像水印领域中双域水印已被证明比单域更鲁棒，但尚未有人将此思路应用到扩散模型水印中。同时作者发现旋转/裁剪对图像的操作可以近似映射到高斯噪声空间中的等效操作，因此可以训练一个修复网络来恢复退化后的噪声。

**核心 idea**：在初始高斯噪声的空间域和频率域同时嵌入水印，并用可学习的噪声修复网络增强旋转/裁剪鲁棒性。

## 方法详解

### 整体框架

GaussMarker 由三个核心模块组成：(1) 流水线水印注入器——依次在高斯噪声的空间域和频率域嵌入水印；(2) DDIM 反转+双域检测——从待检图像估计原始噪声后分别在两个域提取检测分数；(3) 高斯噪声修复器（GNR）——修复受攻击后退化的噪声信号。生成过程与标准 LDM 完全一致，不修改模型权重或推理流程。

### 关键设计

1. **空间域水印注入**:

    - 功能：在高斯噪声中嵌入 $l$-bit 多比特水印
    - 核心思路：将 $l$-bit 水印 $\omega$ 先上采样到与噪声同维度 $c \times w \times h$，再用确定性编码进行 Shuffle 得到信号图 $s \in \{0,1\}^{c \times w \times h}$。信号图指示各位置噪声的符号：$z_T^s = |z_T| \cdot (2s - 1)$，即保留噪声绝对值但用水印控制正负号。检测时通过逆 Shuffle + 均值池化投票恢复水印
    - 设计动机：符号控制不改变噪声的统计分布（仍是标准高斯），因此不影响生成质量。上采样后 Shuffle 保证了各维度的独立性和可逆性

2. **频率域水印注入**:

    - 功能：在空间已水印的噪声上再嵌入零比特频率域水印，增加检测冗余
    - 核心思路：对空间已水印的噪声 $z_T^s$ 做 Fourier 变换得到频率特征 $\hat{z}_T^s$，用预定义的圆环掩码 $M$ 将低频区域替换为预设的频率水印模式 $\omega^f$：$z_T^{s,f} = \mathcal{F}^{-1}(\hat{z}_T^s \cdot (1-M) + \omega^f \cdot M)$。小半径圆环仅改变少量频率分量，由 Parseval 定理保证空间域信号 $s$ 基本不变
    - 设计动机：频率域水印对旋转/缩放具有天然鲁棒性（Fourier 域的平移不变性），与空间域水印互补。零比特设计最小化对空间信号的干扰

3. **高斯噪声修复器（GNR）**:

    - 功能：修复经旋转/裁剪攻击后通过 DDIM 反转估计的噪声，恢复水印信号
    - 核心思路：核心观察是 DDIM 反转在图像级别的操作可以近似映射为噪声空间中的等效操作，即 $\text{Inversion}(\mathcal{T}(x^{s,f})) \approx \mathcal{T}(z_T^{s,f})$。因此可以在纯噪声上模拟攻击来训练 GNR，无需真正生成和反转图像。优化目标简化为 $\min_{\text{GNR}} \|{\text{GNR}}(\mathcal{T}(s_T^{s,f})) - s_T^{s,f}\|^2$，在信号图上操作。实际训练用 BCE 损失替换 MSE（因目标为二值），并加入负样本防止过拟合到固定输出
    - 设计动机：直接在噪声空间训练避免了高昂的 LDM 正向/反向传播成本，且修复网络与具体扩散模型无关——在 SD1.4 上训练可直接用于 SD2.0 检测

### 损失函数 / 训练策略

检测阶段融合两个域的分数。空间域分数 $r_s = -\|\tilde{\omega} - \omega\|^2$（水印比特误差），频率域分数 $r_f = -\|(\hat{\tilde{z}}_T - \omega^f) \cdot M\|^2$（频率模式匹配误差）。GNR 应用于反转噪声后再提取分数，最终融合两个域的 p-value 做判断。GNR 训练约需 1 小时，独立于 LDM。

## 实验关键数据

### 主实验（TPR@1%FPR / Bit Accuracy，SD V2.1）

| 方法 | Clean | Rotate | JPEG | Crop&Scale | Blur | Avg |
|------|-------|--------|------|------------|------|-----|
| Gaussian Shading | 1.000/1.000 | 0.018/0.512 | 0.999/0.986 | 0.081/0.540 | 1.000/0.999 | 0.788/0.874 |
| Tree-Ring | 1.000/- | 0.548/- | 0.998/- | 0.048/- | 1.000/- | 0.823/- |
| RingID | 1.000/- | 1.000/- | 1.000/- | 0.078/- | 1.000/- | 0.883/- |
| PRC | 1.000/1.000 | 0.010/0.500 | 0.926/0.962 | 0.020/0.501 | 0.993/0.996 | 0.638/0.816 |
| **GaussMarker** | **1.000/1.000** | **0.997/0.998** | **0.996/0.997** | **1.000/1.000** | **1.000/1.000** | **0.997/0.990** |

### 跨 SD 版本（平均 TPR@1%FPR / Bit Accuracy）

| SD 版本 | Gaussian Shading | Tree-Ring | **GaussMarker** |
|---------|-----------------|-----------|-----------------|
| SD V1.4 | 0.796/0.874 | 0.828/- | **0.997/0.990** |
| SD V2.0 | 0.796/0.874 | 0.822/- | **0.998/0.991** |
| SD V2.1 | 0.788/0.874 | 0.823/- | **0.997/0.990** |

### 关键发现

- 旋转攻击下差距最为惊人：Gaussian Shading TPR@1%FPR 仅 0.018，GaussMarker 达到 0.997——从几乎不可用到几乎完美
- Crop&Scale 同样是其他方法的致命攻击（TPR 0.02~0.08），GaussMarker 维持 1.000——GNR 的噪声修复是关键
- 跨三个 SD 版本的一致性证明了 GNR 模型无关设计的可靠性
- 在维持低 FPR（<1%）的前提下实现高 TPR，符合实际部署要求

## 亮点与洞察

- **双域水印从传统水印理论迁移到扩散模型**：空间域提供多比特容量，频率域提供旋转鲁棒性，两者互补。这一从经典理论到新应用的迁移角度值得借鉴
- **GNR 的模型无关设计**：利用 DDIM 反转的近似等变性在纯噪声上训练修复网络，彻底避免了对特定 LDM 的依赖。训练仅需采样高斯噪声并模拟攻击，极其高效
- **零比特频率域水印的最小干扰设计**：小半径圆环掩码保证对空间信号的扰动最小化，Parseval 定理提供了理论保证

## 局限性

- GNR 需要额外训练一个修复网络，增加部署复杂度
- 对极端组合攻击（旋转+裁剪+压缩同时施加）的鲁棒性未验证
- 频率域水印为零比特（仅检测有/无），多比特容量上限分析缺失
- 对非 SD 架构（DiT、Flux 等新一代扩散模型）的泛化性待测试

## 相关工作与启发

- **vs Gaussian Shading**：GS 仅在空间域嵌入，旋转下 TPR 从 1.0 降到 0.018；GaussMarker 双域+GNR 维持 0.997，鲁棒性提升 55 倍
- **vs Tree-Ring**：Tree-Ring 在频率域用同心圆环模式，对旋转有一定鲁棒性（TPR 0.548）但远不够；GaussMarker 的 GNR 进一步修复噪声退化
- **vs Stable Signature**：需修改解码器权重，是 tuning-based 方法；GaussMarker 完全免微调，即插即用
- **启发**：GNR 的噪声恢复思路可推广到音频水印、3D 模型水印等需要从退化信号恢复原始模式的场景

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个双域扩散模型水印+模型无关的 GNR 修复方案
- 实验充分度: ⭐⭐⭐⭐⭐ 8种扭曲+4种高级攻击+3个SD版本，对比7种方法
- 写作质量: ⭐⭐⭐⭐ 框架清晰，注入/检测流程图示直观
- 价值: ⭐⭐⭐⭐⭐ 解决扩散模型水印最大痛点（旋转/裁剪鲁棒性），实际部署价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] SleeperMark: Towards Robust Watermark against Fine-Tuning Text-to-Image Diffusion Models](../../CVPR2025/image_generation/sleepermark_towards_robust_watermark_against_fine-tuning_text-to-image_diffusion.md)
- [\[NeurIPS 2025\] T2SMark: Balancing Robustness and Diversity in Noise-as-Watermark for Diffusion Models](../../NeurIPS2025/image_generation/t2smark_balancing_robustness_and_diversity_in_noise-as-watermark_for_diffusion_m.md)
- [\[ICML 2025\] Angle Domain Guidance: Latent Diffusion Requires Rotation Rather Than Extrapolation](angle_domain_guidance_latent_diffusion_requires_rotation_rather_than_extrapolati.md)
- [\[ICML 2025\] Morse: Dual-Sampling for Lossless Acceleration of Diffusion Models](morse_dual-sampling_for_lossless_acceleration_of_diffusion_models.md)
- [\[ICCV 2025\] Dual Recursive Feedback on Generation and Appearance Latents for Pose-Robust Text-to-Image Diffusion](../../ICCV2025/image_generation/dual_recursive_feedback_on_generation_and_appearance_latents_for_pose-robust_tex.md)

</div>

<!-- RELATED:END -->
