---
title: >-
  [论文解读] Timestep-Aware Diffusion Model for Extreme Image Rescaling
description: >-
  [图像生成] 提出 TADM，在预训练 SD 的潜空间中执行极端图像缩放（16×/32×），通过解耦特征缩放模块和时间步自适应对齐策略，动态分配扩散模型的生成能力以应对空间非均匀退化。
tags:
  - 图像生成
---

# Timestep-Aware Diffusion Model for Extreme Image Rescaling

> **会议**: ICCV 2025
> **arXiv**: [2408.09151](https://arxiv.org/abs/2408.09151)
> **领域**: 图像缩放·扩散模型·超分辨率
> **关键词**: image rescaling, timestep alignment, diffusion model, decoupled feature rescaling, extreme downscaling

## 一句话总结

提出 TADM，在预训练 SD 的潜空间中执行极端图像缩放（16×/32×），通过解耦特征缩放模块和时间步自适应对齐策略，动态分配扩散模型的生成能力以应对空间非均匀退化。

## 研究背景与动机

超高分辨率图像的存储和传输需求使得极端缩放因子（16×、32×）变得越来越重要。图像缩放（Image Rescaling）与传统超分（SR）不同，可以联合优化下采样和上采样过程，保留关键信息用于高质量上采样。

然而，现有方法在极端因子下面临严峻挑战：

**INN 方法（IRN、HCFlow）的信息瓶颈**：可逆神经网络在 2×/4× 下表现优秀，但极端压缩下无法保留足够细节，产生过度平滑结果。

**GAN 先验方法的语义错误**：GRAIN（StyleGAN 先验）局限于人脸领域，VQIR（VQGAN 先验）虽适用于自然图像，但在人脸和文字等复杂结构上仍出现语义错误。

**空间非均匀退化**：不同图像和同一图像的不同区域，缩放引起的信息损失差异巨大（如纹理丰富区域 vs 平坦区域），但现有方法使用固定的恢复策略。

**TADM 的核心思路**：
- 在 SD 潜空间执行缩放操作（与 SD 先验对齐）
- 将缩放退化类比为扩散前向加噪过程
- 自适应预测"噪声密度"（时间步），动态分配扩散模型的生成能力

## 方法详解

### 整体框架（Fig. 1）

TADM 包含四个阶段：
1. **潜编码**：预训练 VAE 编码器 $\mathcal{E}$ 将 HR 图像 $x$ 编码为 $z$
2. **潜空间特征缩放**：DFRM 将 $z$ 缩放至目标尺寸，输出 LR 图像 $y$ 和缩放潜特征 $\hat{z}$
3. **去噪引导感知增强**：预训练 SD U-Net 对 $\hat{z}$ 执行单步去噪，获得增强特征 $\hat{z}_0$
4. **潜解码**：预训练 VAE 解码器 $\mathcal{D}$ 将 $\hat{z}_0$ 解码为重建图像 $\hat{x}$

### 关键设计 1：解耦特征缩放模块（DFRM）

传统方法直接从 $z$ 生成 LR 图像 $y$ 并从 $y$ 重建 $\hat{z}$，但导引损失 $\mathcal{L}_{gui}$ 和重建损失 $\mathcal{L}_{rec}$ 相互矛盾（一个趋向 SR 问题，一个趋向压缩问题）。

DFRM 将缩放解耦为两条独立变换链（Fig. 2）：
- **特征缩放链**：$(x, z) \rightarrow z_{lr} \rightarrow \hat{z}$，用 CNN 编码器 $G_e$ 和解码器 $G_d$ 完成
- **像素映射链**：$z_{lr} \leftrightarrow y$，用可逆神经网络 INN $F$ 完成特征域到像素域的双向映射

$$z_{lr} = G_e(x, z), \quad y = F(z_{lr}), \quad \hat{z} = G_d(F^{-1}(y))$$

重建损失同时考虑有/无量化操作的两条路径：

$$\mathcal{L}_{rec} = \|G_d(G_e(x,z)) - z\|_1 + \|G_d(F^{-1}(F(G_e(x,z)))) - z\|_1$$

引入**像素引导模块**改善 LR 图像视觉质量。

### 关键设计 2：时间步对齐策略

**核心观察**（Fig. 3）：缩放引入的 MSE 与扩散前向加噪的 MSE 呈对应关系——不同缩放因子和不同图像内容对应不同的扩散时间步。

- **时间步预测模块（TPM）**：轻量网络从 $\hat{z}$ 预测时间步 $t = \text{TPM}(\hat{z})$
- **混合时间调度器**：固定调度器（不可导）+ 可学习调度器（神经网络模拟，零初始化卷积稳定训练）：

$$\hat{z}_0 = \mathcal{S}_{fixed}(\hat{z}, \epsilon, t_0) + \mathcal{S}_{learned}(\hat{z}, \epsilon, t)$$

固定调度器使用标准公式 $\hat{z}_0 = (\hat{z} - \sqrt{1-\bar{\alpha}_t} \epsilon) / \sqrt{\bar{\alpha}_t}$，可学习调度器通过零初始化卷积逐步修正。

**分块推理**：超高分辨率图像分块处理，每块预测不同时间步，实现空间自适应的生成能力分配（Fig. 14）。

### 训练策略

三阶段训练：
1. 训练 DFRM（$\mathcal{L}_{res} = \lambda_{rec} \mathcal{L}_{rec} + \lambda_{gui} \mathcal{L}_{gui}$）
2. 联合训练 LoRA + TPM + 时间调度器（$\mathcal{L}_{enh} = \|x - \hat{x}\|_1 + \lambda_{pec}(\mathcal{L}_{lpips} + \mathcal{L}_{dists})$）
3. 小学习率联合微调全部模块

## 实验

### 主实验：极端缩放定量比较（Tab. 1）

**16× 缩放，DIV2K 数据集**：

| 方法 | PSNR ↑ | LPIPS ↓ | DISTS ↓ | MUSIQ ↑ | CLIPIQA ↑ |
|------|--------|---------|---------|---------|-----------|
| ESRGAN | 23.15 | 0.4478 | 0.2378 | 59.80 | 0.6161 |
| HCFlow | 26.66 | 0.4885 | 0.2866 | 46.43 | 0.2735 |
| VQIR | 23.91 | 0.3174 | 0.1024 | 64.04 | 0.6350 |
| S3Diff | 20.22 | 0.4033 | 0.1309 | 64.37 | 0.6228 |
| **TADM** | 23.98 | **0.2979** | **0.0886** | **66.56** | **0.7189** |

**32× 缩放，DIV2K 数据集**：

| 方法 | PSNR ↑ | LPIPS ↓ | DISTS ↓ | MUSIQ ↑ | CLIPIQA ↑ |
|------|--------|---------|---------|---------|-----------|
| HCFlow | 23.89 | 0.5816 | 0.3852 | 37.25 | 0.2792 |
| VQIR | 22.02 | 0.4568 | 0.2663 | 58.21 | 0.6293 |
| S3Diff | 17.81 | 0.4895 | 0.1810 | 67.92 | 0.6991 |
| **TADM** | **22.18** | **0.4221** | **0.1684** | **69.12** | **0.7204** |

TADM 在所有数据集的全部感知指标上达到最优。特别是 DIV2K 32× 下 DISTS 较第二名 VQIR 提升 **36.76%**。

### 消融实验：缩放空间与 SD 先验（Tab. 2）

| 潜空间缩放 | 像素缩放 | SD先验 | LPIPS ↓ | DISTS ↓ |
|-----------|---------|-------|---------|---------|
| ✗ | ✓ | ✓ | 0.3630 | 0.1154 |
| ✓ | ✗ | ✗ | 0.4675 | 0.3109 |
| ✓ | ✗ | ✓ | **0.2979** | **0.0886** |

关键发现：
- 潜空间缩放 vs 像素缩放：潜空间中操作与 SD 先验更对齐（DISTS 0.0886 vs 0.1154）
- SD 先验至关重要：去除 SD 增强后 DISTS 从 0.0886 暴涨至 0.3109

### 时间步对齐有效性（Fig. 13）

- 固定时间步 = 1：高保真但低感知质量
- 固定时间步 = 999：低保真但高感知质量
- **自适应时间步**：两者兼得，在保真度和感知质量上均最优

Fig. 14 展示了预测的时间步图：复杂纹理区域被分配更大时间步（更强生成能力），平坦区域分配更小时间步（更高保真度）。

## 亮点与洞察

1. **缩放=加噪** 的类比洞察极为精巧，为图像缩放问题建立了与扩散模型的自然联系
2. **解耦设计**解决了重建损失与引导损失的根本矛盾，INN 模块专注特征-像素映射
3. **时间步自适应**实现了空间非均匀退化的精细处理，分块推理中每块独立预测时间步
4. **单步去噪**高效利用 SD 先验，避免多步采样的高延迟

## 局限性

- LR 图像存在一定的振铃伪影和噪声
- 基于 SD 2.1-base 构建，对更新的 SD 版本需重新适配
- 分块推理中块边界可能引入不一致性

## 相关工作

- 图像缩放：IRN、HCFlow、CAR、VQIR、GRAIN
- 扩散超分：SR3、StableSR、S3Diff、SinSR、InvSR
- 单步扩散：OSEDiff、ResShift

## 评分

- **新颖性**: ★★★★★ — 时间步自适应对齐策略具有原创性
- **技术深度**: ★★★★★ — DFRM 解耦设计 + 混合调度器 + 分块推理，工程细节扎实
- **实验质量**: ★★★★★ — 4 个数据集 × 2 个缩放因子 × 6 个指标 + 多维消融
- **写作质量**: ★★★★☆ — 结构清晰，图示信息量大

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] FreeMorph: Tuning-Free Generalized Image Morphing with Diffusion Model](freemorph_tuning-free_generalized_image_morphing_with_diffusion_model.md)
- [\[ICCV 2025\] EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model](emoticrafter_text-to-emotional-image_generation_based_on_valence-arousal_model.md)
- [\[ICCV 2025\] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)
- [\[CVPR 2025\] DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation](../../CVPR2025/image_generation/dual-interrelated_diffusion_model_for_few-shot_anomaly_image_generation.md)
- [\[ICCV 2025\] LiT: Delving into a Simple Linear Diffusion Transformer for Image Generation](lit_delving_into_a_simple_linear_diffusion_transformer_for_image_generation.md)

<!-- RELATED:END -->
