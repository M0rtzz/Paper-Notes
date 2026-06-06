---
title: >-
  [论文解读] FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-Resolution
description: >-
  [CVPR 2025][图像生成][超分辨率] 提出 FaithDiff，首次释放（fine-tune）预训练扩散模型先验用于图像超分辨率，并设计对齐模块桥接退化图像特征与扩散噪声隐空间，通过联合优化 encoder 和扩散模型实现高保真结构恢复。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "超分辨率"
  - "扩散模型"
  - "alignment module"
  - "joint fine-tuning"
  - "faithful restoration"
---

# FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-Resolution

**会议**: CVPR 2025  
**arXiv**: [2411.18824](https://arxiv.org/abs/2411.18824)  
**代码**: [https://jychen9811.github.io/FaithDiff_page/](https://jychen9811.github.io/FaithDiff_page/)  
**领域**: 图像生成  
**关键词**: image super-resolution, diffusion prior, alignment module, joint fine-tuning, faithful restoration

## 一句话总结

提出 FaithDiff，首次释放（fine-tune）预训练扩散模型先验用于图像超分辨率，并设计对齐模块桥接退化图像特征与扩散噪声隐空间，通过联合优化 encoder 和扩散模型实现高保真结构恢复。

## 研究背景与动机

**领域现状**: 基于 LDM 的图像超分辨率方法（如 DiffBIR、SUPIR、SeeSR）已取得显著进展，但它们冻结预训练扩散模型，仅通过改进 encoder 提取退化鲁棒特征来引导扩散过程。

**现有痛点**:
- 冻结的扩散模型将 encoder 提取的任何错误特征误解为真实图像结构，导致恢复结果不忠实（如 Figure 1 中文字区域的失真）
- 退化图像的 LQ 特征与扩散模型的噪声隐表示之间存在显著 gap，直接拼接效果不佳
- 分别优化 encoder 和扩散模型会限制各模块协同工作的能力

**核心矛盾**: 现有方法仅依赖改进特征提取来引导冻结的扩散模型，但退化图像的特征提取本身就是不完美的，冻结的扩散模型无法分辨退化伪影和真实结构信息。

**本文目标**: 让 LDM 不仅能生成逼真的纹理，还能恢复与输入一致的忠实结构。

**切入角度**: 释放扩散模型（允许 fine-tune），使其学会从退化输入中识别有用信息；同时设计对齐模块让 encoder 特征与渐进式扩散过程对齐。

**核心 idea**: 不要冻结扩散先验——释放它，并让 encoder 和扩散模型在联合优化中互相适配，实现忠实超分。

## 方法详解

### 整体框架

1. 使用 VAE encoder 从 LQ 图像提取倒数第二层特征 $f^{LQ}$（512 通道，而非常规 8 通道）
2. 对齐模块将 $f^{LQ}$ 与当前扩散步 $t$ 的噪声隐表示 $x_t^{HQ}$ 融合，生成对齐特征 $f_t^a$
3. 扩散模型以 $f_t^a$ 和文本嵌入 $c$ 为条件进行去噪
4. 联合微调 VAE encoder + 对齐模块 + 扩散模型
5. 冻结的 VAE decoder 从精炼特征重建 HQ 图像

### 关键设计

**1. 深层 LQ 特征提取**
- **功能**: 使用 VAE encoder 倒数第二层的特征（512 通道）而非最后一层（8 通道）作为 LQ 特征 $f^{LQ}$。
- **核心思路**: 最后一层严重压缩通道维度，无法同时捕获退化因素和结构细节。倒数第二层保留了更丰富的信息，有利于后续扩散过程。
- **设计动机**: 实验证明 512 通道特征相比 8 通道在所有指标上更优（消融实验验证），信息量的增加对忠实恢复至关重要。

**2. 对齐模块（Alignment Module）**
- **功能**: 设计一个 Transformer-based 对齐模块，将 LQ 特征 $f^{LQ}$ 与噪声隐表示 $x_t^{HQ}$ 在每个扩散步动态对齐。
- **核心思路**: 分别对 $x_t^{HQ}$ 和 $f^{LQ}$ 进行卷积，拼接后经 2 层 Transformer block 交互，再通过残差连接和线性层输出对齐特征 $f_t^a$：$f_t^a = \text{Linear}(\text{Trans}(f_t^c) + f_t^x)$。
- **设计动机**: 随着扩散过程推进，$x_t^{HQ}$ 逐渐变清晰，但 $f^{LQ}$ 是固定的。直接相加会导致退化因素持续干扰生成。对齐模块通过 Transformer 交互自适应提取 $f^{LQ}$ 中与当前扩散步相关的有用信息。

**3. 统一特征优化（Unified Feature Optimization）**
- **功能**: 联合微调 VAE encoder、对齐模块和扩散模型三个组件，保持 VAE decoder 和文本编码器冻结。
- **核心思路**: 两阶段训练——(1) 先冻结 encoder 和扩散模型，仅预训练对齐模块；(2) 联合微调三个组件，让它们互相适配。
- **设计动机**: 分别优化限制了模块间的协同能力。联合优化使 encoder 能学习提取更适配扩散过程的特征，扩散模型也能学习从退化输入中分辨退化伪影和真实结构。

### 损失函数 / 训练策略

- L1 噪声预测损失: $L = \|\epsilon - \hat{\epsilon}_\theta(\sqrt{\bar{\alpha}_t} x_0^{HQ} + \sqrt{1-\bar{\alpha}_t} \epsilon, f^{LQ}, c, t)\|_1$
- 两阶段训练: 先预训练对齐模块建立连接，再联合微调整个系统
- 使用文本嵌入通过 cross-attention 辅助结构信息提取
- 基于 Stable Diffusion 架构

## 实验关键数据

### 主实验（DIV2K-Val + LSDIR-Val，三级退化）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | MUSIQ↑ | CLIPIQA+↑ |
|---|---|---|---|---|---|
| **Level-I (DIV2K)** | | | | | |
| Real-ESRGAN | 26.64 | 0.7737 | 0.1964 | 62.38 | 0.4649 |
| DiffBIR | 24.60 | 0.6595 | 0.2496 | 66.23 | 0.5407 |
| SUPIR | 25.09 | 0.7010 | 0.2139 | 65.49 | 0.5202 |
| SeeSR | 25.08 | 0.6967 | 0.2263 | 66.48 | 0.5336 |
| **FaithDiff** | 24.29 | 0.6668 | **0.2187** | **66.53** | **0.5432** |
| **Level-II (LSDIR)** | | | | | |
| SeeSR | 22.00 | 0.6026 | 0.2469 | 70.91 | 0.5837 |
| SUPIR | 21.30 | 0.5713 | 0.2733 | 70.59 | 0.5998 |
| **FaithDiff** | 20.88 | 0.5493 | **0.2469** | **71.15** | **0.6219** |

### 消融实验

1. **释放 vs 冻结扩散模型**: 释放后 MUSIQ +2.1, CLIPIQA+ +0.04
2. **联合 vs 分别优化**: 联合优化在感知质量指标上始终优于分别优化
3. **512 vs 8 通道 LQ 特征**: 512 通道在 LPIPS、MUSIQ、CLIPIQA+ 上全面领先
4. **对齐模块 vs 直接拼接**: Transformer-based 对齐显著优于简单拼接

### 关键发现

1. **冻结扩散模型是瓶颈**: 现有方法冻结扩散模型，导致其无法区分退化伪影和真实结构，释放后模型能主动抑制重建错误。
2. **联合优化的协同效应**: encoder 学习提取与扩散过程匹配的特征，扩散模型学习从退化特征中提取有用信息，二者协同提升忠实度。
3. **感知指标 vs 失真指标的权衡**: FaithDiff 在感知质量（MUSIQ、CLIPIQA+）上 SOTA，但 PSNR/SSIM 略低——这是因为方法优先恢复真实纹理而非逐像素拟合。
4. **退化越重优势越大**: 在 Level-III（重度退化）上，FaithDiff 相比 DiffBIR 的 CLIPIQA+ 提升最为显著。

## 亮点与洞察

- 首次挑战"冻结扩散先验"的范式，揭示了释放先验对忠实恢复的关键作用
- 对齐模块设计简洁但有效，核心思想是让 LQ 特征动态适配渐进式去噪过程
- 联合优化策略充分利用了三个模块的互补性
- 深层特征（512 通道）vs 浅层特征（8 通道）的分析为后续工作提供了指导

## 局限与展望

- PSNR/SSIM 等像素级指标不如传统方法，在需要精确像素重建的场景（如医学影像）可能不适用
- 联合微调增加了训练成本和复杂度
- 文本描述的质量会影响超分效果
- 对齐模块使用固定的 2 层 Transformer，可能不是最优架构
- 未探索在更大规模扩散模型（如 SDXL）上的效果

## 相关工作与启发

- DiffBIR 和 SUPIR 的两阶段方案（退化去除 + 细节生成）受限于退化去除精度；FaithDiff 通过端到端联合优化绕过了这一瓶颈
- 与 ControlNet 等条件注入方法不同，FaithDiff 直接修改扩散模型内部以适配 SR 任务
- 启发：在其他低级视觉任务（去噪、去模糊）中，释放扩散先验或许同样有效

## 评分

⭐⭐⭐⭐ — 释放扩散先验的核心洞察有价值，联合优化策略有效，但像素级指标的退步是一个 trade-off

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Uncertainty-guided Perturbation for Image Super-Resolution Diffusion Model](uncertainty-guided_perturbation_for_image_super-resolution_diffusion_model.md)
- [\[CVPR 2025\] Arbitrary-Steps Image Super-Resolution via Diffusion Inversion](arbitrary-steps_image_super-resolution_via_diffusion_inversion.md)
- [\[ECCV 2024\] XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution](../../ECCV2024/image_generation/xpsr_crossmodal_priors_for_diffusionbased_image_superresolut.md)
- [\[NeurIPS 2025\] Image Super-Resolution with Guarantees via Conformalized Generative Models](../../NeurIPS2025/image_generation/image_super-resolution_with_guarantees_via_conformalized_generative_models.md)
- [\[CVPR 2025\] Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models](diffusion-4k_ultra-high-resolution_image_synthesis_with_latent_diffusion_models.md)

</div>

<!-- RELATED:END -->
