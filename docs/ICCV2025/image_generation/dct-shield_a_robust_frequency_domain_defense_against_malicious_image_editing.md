---
title: >-
  [论文解读] DCT-Shield: A Robust Frequency Domain Defense against Malicious Image Editing
description: >-
  [ICCV 2025][图像生成][图像免疫] 提出 DCT-Shield，在离散余弦变换（DCT）域中引入对抗扰动而非像素空间，使免疫噪声高度不可感知，并天然具备JPEG压缩鲁棒性，有效抵御基于扩散模型的恶意图像编辑。
tags:
  - ICCV 2025
  - 图像生成
  - 图像免疫
  - 对抗扰动
  - DCT频域
  - JPEG鲁棒性
  - 扩散模型防御
---

# DCT-Shield: A Robust Frequency Domain Defense against Malicious Image Editing

**会议**: ICCV 2025  
**arXiv**: [2504.17894](https://arxiv.org/abs/2504.17894)  
**代码**: 无  
**领域**: Image Generation / Adversarial Defense  
**关键词**: 图像免疫, 对抗扰动, DCT频域, JPEG鲁棒性, 扩散模型防御

## 一句话总结

提出 DCT-Shield，在离散余弦变换（DCT）域中引入对抗扰动而非像素空间，使免疫噪声高度不可感知，并天然具备JPEG压缩鲁棒性，有效抵御基于扩散模型的恶意图像编辑。

## 研究背景与动机

扩散模型（如Stable Diffusion、InstructPix2Pix）使得基于文本提示的图像编辑变得轻而易举，但也带来严重的安全隐患——恶意用户可利用公开图片进行未授权的编辑（换脸、添加/删除物体、篡改背景等）。

现有防御方法的核心思路是在图像中添加对抗噪声使扩散模型无法正常编辑（即"图像免疫"），但存在两个关键问题：

**噪声可见性**：在像素空间添加的对抗噪声（受 $L_\infty$ 约束）在放大后仍能被人眼识别，产生明显的伪影

**缺乏JPEG鲁棒性**：攻击者只需将免疫图像转为低质量JPEG即可去除对抗噪声，恢复可编辑性；增大像素预算虽可改善但会加剧噪声可见性

根本原因是这些方法在像素空间操作，而JPEG压缩本质上是在DCT频域进行信息去除——在像素空间添加的精心设计扰动被JPEG轻易量化掉。

## 方法详解

### 整体框架

DCT-Shield 的核心思想是直接在 DCT 系数上添加对抗扰动，而非像素空间。整个流程：

1. 将输入图像通过 JPEG 编码流程（RGB→YCbCr→色度下采样→8×8分块→DCT变换→量化）得到量化 DCT 系数 $\alpha$
2. 在量化后的 DCT 系数上添加扰动 $\delta$（绕过不可导的量化步骤）
3. 通过 JPEG 解码流程将扰动后的系数重建为免疫图像 $\mathbf{x}'$
4. 将 $\mathbf{x}'$ 送入 VAE 编码器计算损失
5. 使用 PGD 优化 $\delta$

### 关键设计

1. **DCT 域对抗优化**：将对抗扰动从像素空间转移到频域。

   核心优化问题：
    $\delta = \arg\min_{\|\delta\|_\infty \le \epsilon} \mathcal{L}(\mathcal{E}(\mathbf{x}')), \quad \mathbf{x}' = JPEG_D(\alpha + \delta; Q_{alg})$
   
   其中 $\alpha = JPEG_E(\mathbf{x}; Q_{alg})$ 是量化后的 DCT 系数，$\mathcal{E}$ 是 VAE 编码器。
   
    - 损失函数使用 VAE latent 的范数最小化：$\mathcal{L}(\delta) = \|\mathcal{E}(\mathbf{x}')\|_2$
    - 扰动 $\delta$ 在量化后添加，避免梯度被量化函数清零
    - JPEG 管线中的大部分操作（DCT/IDCT、色彩空间变换）是可微的
    - 默认设置：$Q_{alg}=0.95$，$\epsilon=1$，步长 $\gamma=0.1$，1000次迭代

2. **参数效率**：DCT 域操作将参数需求从 $O(3HW)$（像素空间）降至 $O(3HW/2)$（色度下采样后），部分变体仅需 $O(HW)$。这使得优化更高效、收敛更快。

3. **多种变体适配不同场景**：

    - **基础 DCT-Shield**：全通道（Y/Cb/Cr）DCT 系数扰动，通用编辑防护
    - **Mask-based DCT-Shield**：针对 inpainting 任务，将噪声集中在敏感区域
    - **Y-channel DCT-Shield**：仅在亮度通道添加扰动，进一步降低噪声可见性，增强高JPEG压缩下的鲁棒性

### 损失函数 / 训练策略

主要使用基于 VAE 编码器的目标函数（encoder attack），而非更昂贵的 U-Net 扩散攻击。实验证实 VAE 比 U-Net 更容易被对抗攻击，且 VAE-only 优化足以跨编辑模型迁移。

默认值：$Q_{alg} = 0.95$，$\epsilon = 1$，512×512 分辨率，1000 次 PGD 迭代。

## 实验关键数据

### 主实验

**编辑防护 + 噪声不可感知性对比**（OmniEdit 150样本，IP2P编辑模型）：

| 方法 | 噪声 LPIPS↓ | 噪声 FID↓ | 噪声 PSNR↑ | 防护 LPIPS↑ | 防护 FID↑ | 防护 PSNR↓ | 人类评价↑ |
|------|-----------|---------|----------|-----------|---------|----------|---------|
| AdvDM | 0.353 | 148.89 | 27.11 | 0.561 | 278.75 | 13.19 | 3.44 |
| MIST | 0.362 | 104.27 | 26.62 | 0.534 | 288.61 | 16.55 | 2.45 |
| PhotoGuard | 0.284 | 57.78 | 28.32 | 0.679 | 336.74 | 12.55 | 4.16 |
| SDS(-) | 0.335 | 86.36 | 27.84 | 0.681 | 313.74 | 12.70 | 4.02 |
| **DCT-Shield** | **0.267** | **35.02** | 27.61 | **0.684** | 316.36 | **12.25** | **4.35** |

DCT-Shield 在噪声不可感知性上大幅领先（FID 35.02 vs 次优57.78），同时防护效果也最优或持平。

**Inpainting 防护**（56样本，SD Inpainting 1.0）：

| 方法 | LPIPS↑ | FID↑ | CLIP↓ | 人类评价↑ |
|------|--------|------|-------|---------|
| AdvDM | 0.421 | 170.39 | 0.706 | 2.32 |
| PhotoGuard | 0.506 | 180.32 | 0.682 | 3.36 |
| DiffusionGuard | 0.518 | 194.93 | 0.664 | 3.96 |
| **DCT-Shield** | **0.547** | **199.08** | 0.674 | **4.12** |

### 消融实验

**JPEG鲁棒性**（不同压缩质量的防护保持能力）：

| JPEG质量 | SDS(-) LPIPS↑ | PhotoGuard LPIPS↑ | DCT-Shield LPIPS↑ |
|---------|-------------|-----------------|-----------------|
| 95% | ~0.55 | ~0.58 | ~0.60 |
| 85% | ~0.30 | ~0.35 | ~0.55 |
| 75% | ~0.20 | ~0.22 | ~0.50 |
| 65% | ~0.15 | ~0.18 | ~0.45 |

DCT-Shield 在所有JPEG压缩质量下均保持高防护能力（LPIPS > 0.45），而基线方法在低质量压缩后迅速失效。

DCT-Shield 对其他净化手段也表现出更强鲁棒性：AdvClean（LPIPS ~0.55 vs 基线 ~0.40-0.50）和 crop-and-resize。

**扰动限制-防护效果权衡**：$\epsilon$ 从0.8到1.4，噪声FID与防护FID之间的Pareto前沿表明 DCT-Shield 达到了显著优于像素空间方法的平衡。

### 关键发现

- DCT域扰动在人类感知上远优于像素空间扰动（FID从57-148降至35）
- JPEG管线的集成使免疫图像天然抵抗JPEG净化——这是之前所有方法的致命弱点
- 仅使用VAE编码器优化已足够，比U-Net扩散攻击计算量小得多
- Y-channel变体在高JPEG压缩下尤其有效，因亮度通道在JPEG中保留最完整
- 跨模型迁移性良好（从SD VAE优化的扰动在IP2P、Inpainting等不同U-Net上均有效）

## 亮点与洞察

- **从JPEG算法获得灵感**：JPEG本质上就是在DCT域做不可感知修改，DCT-Shield将同样的思路反过来用于对抗扰动，自然而然地获得了不可感知性和JPEG鲁棒性
- **绕过量化梯度问题**：在量化步骤之后添加扰动，避免了量化函数梯度为零的问题
- **$Q_{alg}$ 提供可调鲁棒性范围**：用户可通过调整算法质量因子来控制鲁棒性-不可感知性权衡，这是像素空间方法无法实现的
- **参数量减半**：利用JPEG的色度下采样特性，优化参数量从 $3HW$ 减至 $\sim 1.5HW$

## 局限与展望

- 需要1000次PGD迭代优化每张图，计算开销仍然较大（虽然仅需VAE前向/反向传播）
- 仅在512×512分辨率上验证，高分辨率图像需要更多DCT块和更长优化时间
- 对非JPEG格式的净化（如WebP、AVIF等现代格式）未做评估
- 当攻击者已知防御使用DCT-Shield时，可能设计针对性的自适应攻击
- 可进一步探索扩展到视频数据和更多编辑模型

## 相关工作与启发

- EditShield、PhotoGuard、MIST、AdvDM 等均在像素空间添加扰动，DCT-Shield 是首个在频域优化的方法
- DiffusionGuard 针对 inpainting 的扩散攻击计算昂贵，DCT-Shield 的 mask 变体以更低成本实现了更好效果
- Diff-Protect (SDS) 证实了 VAE 比 U-Net 更易受对抗攻击，DCT-Shield 充分利用了这一发现

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 将对抗扰动从像素空间转移到DCT域，思路新颖且自然，JPEG鲁棒性作为"副产品"而非强制约束
- **实验充分度**: ⭐⭐⭐⭐ 覆盖编辑/inpainting两类任务，多种净化方法评估，人类评价充分
- **写作质量**: ⭐⭐⭐⭐ JPEG相关背景介绍详实，方法动机推导流畅
- **价值**: ⭐⭐⭐⭐ 解决了图像免疫领域噪声可见性和JPEG不鲁棒两大核心痛点，实用性强

<!-- RELATED:START -->

## 相关论文

- [Towards Robust Defense against Customization via Protective Perturbation Resistant to Diffusion-based Purification](towards_robust_defense_against_customization_via_protective_perturbation_resista.md)
- [Robust-Wide: Robust Watermarking against Instruction-driven Image Editing](../../ECCV2024/image_generation/robust-wide_robust_watermarking_against_instruction-driven_image_editing.md)
- [TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models](trce_towards_reliable_malicious_concept_erasure_in_text-to-image_diffusion_model.md)
- [Efficient Input-Level Backdoor Defense on Text-to-Image Synthesis via Neuron Activation Variation](efficient_input-level_backdoor_defense_on_text-to-image_synthesis_via_neuron_act.md)
- [FICGen: Frequency-Inspired Contextual Disentanglement for Layout-driven Degraded Image Generation](ficgen_frequency-inspired_contextual_disentanglement_for_layout-driven_degraded_.md)

<!-- RELATED:END -->
