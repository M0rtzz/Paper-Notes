---
title: >-
  [论文解读] DCTdiff: Intriguing Properties of Image Generative Modeling in the DCT Space
description: >-
  [ICML2025][图像生成][扩散模型] 提出 DCTdiff，首次在离散余弦变换（DCT）频域空间中进行端到端扩散图像生成，无需 VAE 即可无缝扩展至 512×512 分辨率，并在生成质量和训练效率上均优于像素空间扩散模型。 - 像素空间建模的冗余性：传统扩散模型直接在 RGB 像素空间建模，维度高、训练昂贵…
tags:
  - "ICML2025"
  - "图像生成"
  - "扩散模型"
  - "DCT"
  - "频域建模"
  - "频谱自回归"
---

# DCTdiff: Intriguing Properties of Image Generative Modeling in the DCT Space

**会议**: ICML2025  
**arXiv**: [2412.15032](https://arxiv.org/abs/2412.15032)  
**代码**: [forever208/DCTdiff](https://github.com/forever208/DCTdiff)  
**领域**: 图像生成  
**关键词**: 扩散模型, DCT, 频域建模, 图像生成, 频谱自回归

## 一句话总结
提出 DCTdiff，首次在离散余弦变换（DCT）频域空间中进行端到端扩散图像生成，无需 VAE 即可无缝扩展至 512×512 分辨率，并在生成质量和训练效率上均优于像素空间扩散模型。

## 研究背景与动机
- **像素空间建模的冗余性**：传统扩散模型直接在 RGB 像素空间建模，维度高、训练昂贵，难以直接扩展到高分辨率（如 UViT 在 FFHQ 256 像素空间训练 FID 高达 120）。
- **潜空间方法的代价**：Latent Diffusion（LDM）虽通过 SD-VAE 降维，但 VAE 本身需用 ~900 万图像训练、引入额外计算开销，且在高分辨率下 GFLOPs 急剧增长。
- **频域压缩的天然优势**：JPEG 编码表明 DCT 可将图像能量集中在少数低频系数中，实现接近无损的显著压缩，且完全无需训练。本文将此思路推广到扩散生成建模。

## 方法详解

### 整体流程
RGB 图像 → YCbCr 色彩空间转换 → 2× 色度下采样 → 分块 2D-DCT → Zigzag 展平 + 高频截断 → 频率 Token 化（4Y+1Cb+1Cr）→ ViT 扩散建模 → 逆 DCT 恢复图像。

### 1. 色彩空间转换与色度下采样
将 RGB 转为 YCbCr 后对 Cb/Cr 通道做 2× 下采样，信号量从 $3hw$ 降至 $1.5hw$（2× 压缩），利用人眼对亮度比色度更敏感的特性。

### 2. 分块 DCT 与高频截断
对 Y/Cb/Cr 通道分 $B \times B$ 块执行 Type-II DCT：

$$D(u,v) = \alpha(u)\alpha(v) \sum_{x=0}^{B-1}\sum_{y=0}^{B-1} A(x,y) \cos\!\left[\frac{(2x+1)u\pi}{2B}\right] \cos\!\left[\frac{(2y+1)v\pi}{2B}\right]$$

按 Zigzag 顺序展平后截去 $m^*$ 个高频系数，截断准则为：

$$m^* = \arg\max_m \{m : \text{rFID}(P_\text{data}, P_\text{dct\_data}(m)) < \gamma\}, \quad \gamma = 0.5$$

在 256×256 下可实现 ~4× 无损压缩，512×512 下达 ~7.1× 无损压缩。

### 3. 频率 Token 化
每个 Token 由 4 个 Y 块 + 1 个 Cb 块 + 1 个 Cr 块拼接而成，维度为 $6(B^2 - m^*)$，与 ViT patch size 关系为 $P = 2B$。

### 4. Entropy-Consistent Scaling
DCT 系数各频率上下界差异可达两个数量级。提出用 DC 分量（$D(0,0)$）的第 $\tau = 98.25$ 百分位作为统一缩放因子 $\eta$：

$$\eta = \max(|P_\tau|, |P_{100-\tau}|), \quad \bar{\mathbf{x}}_0 = \bar{\mathbf{x}}_0 / \eta$$

保持各频率分布形状不变，优于对每个频率独立缩放的"Naive Scaling"。

### 5. SNR Scaling
DCT 将能量集中于低频，使前向扩散过程中高频信号更快被噪声淹没。为补偿 block size $B$ 增大带来的 SNR 下降（$\eta$ 随 $B$ 翻倍而翻倍），对默认 noise schedule 进行 SNR 缩放调整。

### 6. Entropy-Based Frequency Reweighting（EBFR）
在训练损失中引入频率熵权重向量 $\mathbf{H}(B)$，赋予低频（高熵）信号更大权重：

$$\mathcal{L}_\text{EBFR}(\theta) = \mathbb{E}_t \lambda(t) \mathbb{E}_{\bar{\mathbf{x}}_0, \bar{\mathbf{x}}_t} \left[\mathbf{H}(B) \| \mathbf{s}_\theta(\bar{\mathbf{x}}_t, t) - \nabla_{\bar{\mathbf{x}}_t} \log P_{0t}(\bar{\mathbf{x}}_t | \bar{\mathbf{x}}_0) \|^2_2 \right]$$

## 实验关键数据

### UViT 框架下 FID-50k 对比（DPM-Solver）

| 数据集 | NFE | UViT (pixel) | DCTdiff |
|--------|-----|-------------|---------|
| CIFAR-10 | 100 | 5.80 | 5.28 |
| CelebA 64 | 100 | 1.57 | 1.71 |
| ImageNet 64 | 100 | 10.07 | 9.73 |
| FFHQ 128 | 100 | 9.18 | **6.25** |
| FFHQ 128 | 50 | 9.20 | **6.28** |

### 高分辨率无 VAE vs 有 VAE（DPM-Solver, NFE=100）

| 数据集 | UViT (latent+SD-VAE) | DCTdiff (无VAE) |
|--------|---------------------|-----------------|
| FFHQ 256 | **4.26** | 5.08 |
| FFHQ 512 | 10.89 | **7.07** |
| AFHQ 512 | 10.86 | **8.76** |

### 训练效率对比

| 数据集 | 模型 | 参数量 | GFLOPs | 收敛步数 |
|--------|------|--------|--------|----------|
| FFHQ 128 | UViT | 44M | 11 | 750k |
| FFHQ 128 | DCTdiff | 44M | 11 | **300k (2.5× 加速)** |
| AFHQ 512 | UViT (latent) | 131M+84M | 575 | 225k |
| AFHQ 512 | DCTdiff | 131M | 133 | 225k (**仅 1/4 训练成本**) |

### DiT 框架下 FID-50k（DDPM sampler, NFE=100）

| 数据集 | DiT | DCTdiff |
|--------|-----|---------|
| CelebA 64 | 5.11 | **3.84** |
| FFHQ 128 | 12.81 | **11.16** |

## 亮点与洞察
1. **频域扩散的理论贡献**：证明了"图像扩散即频谱自回归"——由于自然图像功率谱密度满足幂律 $|\hat{x}_0(\omega)|^2 = K|\omega|^{-\alpha}$，前向扩散先破坏高频再破坏低频，反向生成则先恢复低频再恢复高频，与 VAR 的 coarse-to-fine 生成一脉相承。
2. **无需 VAE 做高分辨率生成**：DCT 作为训练无关的确定性变换，在 512×512 上超越 SD-VAE latent diffusion，且完全避免了 VAE 的训练成本和领域适配问题。
3. **DCT 上采样定理**：证明了低分辨率 DCT 系数与高分辨率系数之间的近似关系 $\bar{D}(k,l) \approx \frac{1}{2}\cos(\frac{k\pi}{4B})\cos(\frac{l\pi}{4B}) D(k,l)$，DCT 上采样（FID 9.79）优于双三次插值（FID 12.53）。
4. **即插即用**：DCTdiff 不改变 Transformer 架构，可直接在 UViT/DiT 上使用，且在多种采样器（DDIM、DPM-Solver、DDPM）下均有效。

## 局限与展望
- 尚未在 text-to-image 等条件生成任务上验证，目前仅限无条件/类条件生成。
- 在低分辨率 CelebA 64 上优势不明显，甚至在 DPM-Solver 下略逊于 UViT。
- DCT block size $B$ 需要根据分辨率手动选择，缺乏自适应机制。
- Entropy-Consistent Scaling 的百分位阈值 $\tau=98.25$ 为经验值，对新域鲁棒性待验证。
- 高频截断准则 $\gamma=0.5$ 对纹理丰富场景（医学影像等）可能过于激进。
- 未与最新的 Flow Matching / Rectified Flow 方法比较。

## 相关工作与启发
- **DCTransformer** (Nash et al., 2021)：在 DCT 空间做自回归生成，但非扩散范式。
- **VAR** (Tian et al., 2024)：coarse-to-fine 自回归生成，本文从频谱角度给出了理论解释。
- **Latent Diffusion** (Rombach et al., 2022)：SD-VAE 压缩方案，本文证明 DCT 无需训练即可替代。
- **EDM** (Karras et al., 2022)：提供扩散模型设计空间解耦，本文将其扩展到频域设计空间。

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首次系统性地在 DCT 空间做扩散生成，理论与实践结合好
- 实验充分度: ⭐⭐⭐⭐ — 多框架(UViT/DiT)、多数据集、多采样器全面验证，消融实验充分
- 写作质量: ⭐⭐⭐⭐⭐ — 动机清晰，理论证明严谨，流程图直观
- 价值: ⭐⭐⭐⭐ — 提供了像素/潜空间之外的第三条可行路径，对高分辨率生成特别有价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Hessian Geometry of Latent Space in Generative Models](hessian_geometry_of_latent_space_in_generative_models.md)
- [\[ICML 2025\] Compositional Scene Understanding through Inverse Generative Modeling](compositional_scene_understanding_through_inverse_generative_modeling.md)
- [\[ICML 2025\] Action-Minimization Meets Generative Modeling: Efficient Transition Path Sampling with the Onsager-Machlup Functional](action-minimization_meets_generative_modeling_efficient_transition_path_sampling.md)
- [\[NeurIPS 2025\] Boosting Generative Image Modeling via Joint Image-Feature Synthesis](../../NeurIPS2025/image_generation/boosting_generative_image_modeling_via_joint_imagefeature_sy.md)
- [\[ICML 2025\] Efficient Generative Modeling with Residual Vector Quantization-Based Tokens](efficient_generative_modeling_with_residual_vector_quantization-based_tokens.md)

</div>

<!-- RELATED:END -->
