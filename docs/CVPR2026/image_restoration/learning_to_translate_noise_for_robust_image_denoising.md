---
title: >-
  [论文解读] Learning to Translate Noise for Robust Image Denoising
description: >-
  [CVPR 2026][图像恢复][图像去噪] 提出噪声翻译框架，通过轻量级噪声翻译网络将未知真实噪声转换为高斯噪声，再由预训练的高斯去噪网络处理，在 OOD 真实噪声基准上平均 PSNR 提升 1.5dB 以上，且翻译网络仅 0.29M 参数、可跨去噪器迁移。
tags:
  - CVPR 2026
  - 图像恢复
  - 图像去噪
  - 噪声翻译
  - 高斯噪声
  - 分布外泛化
  - Wasserstein距离
---

# Learning to Translate Noise for Robust Image Denoising

**会议**: CVPR 2026  
**arXiv**: [2412.04727](https://arxiv.org/abs/2412.04727)  
**代码**: [https://hij1112.github.io/learning-to-translate-noise/](https://hij1112.github.io/learning-to-translate-noise/)  
**领域**: Image Restoration / 图像恢复  
**关键词**: 图像去噪, 噪声翻译, 高斯噪声, 分布外泛化, Wasserstein距离

## 一句话总结

提出噪声翻译框架，通过轻量级噪声翻译网络将未知真实噪声转换为高斯噪声，再由预训练的高斯去噪网络处理，在 OOD 真实噪声基准上平均 PSNR 提升 1.5dB 以上，且翻译网络仅 0.29M 参数、可跨去噪器迁移。

## 研究背景与动机

基于深度学习的图像去噪方法在控制环境下表现出色，但面对分布外（OOD）真实噪声时泛化能力严重不足：

**合成噪声与真实噪声的分布差异**：早期方法假设高斯噪声，训练的模型在真实场景中表现大幅下降

**真实数据集的过拟合**：用真实 noisy-clean 图像对训练的模型会过拟合到训练数据特有的噪声-信号相关性，在新噪声类型上失效

**完整覆盖所有真实噪声分布不现实**：不同相机、传感器、环境产生的噪声千差万别

**现有泛化方法局限**：固定变换（如 Anscombe 变换）适应性差；测试时优化（如 LAN）计算开销大，不可扩展到大图像

**关键观察**：给真实噪声图像额外添加高斯噪声后，用高斯去噪器处理效果反而显著改善（PSNR 从 29.63dB 提升到 32.73dB）。这启发了"先翻译、再去噪"的策略。

## 方法详解

### 整体框架

两阶段框架：
1. **阶段一**：训练高斯去噪网络 $\mathcal{D}(\cdot; \boldsymbol{\theta})$，专门处理高斯噪声
2. **阶段二**：冻结去噪网络，训练噪声翻译网络 $\mathcal{T}(\cdot; \boldsymbol{\phi})$，将任意真实噪声转换为高斯噪声

推理流程：$\hat{I}_\mathcal{T} = \mathcal{D}(\mathcal{T}(I; \boldsymbol{\phi}); \boldsymbol{\theta}^*)$

### 关键设计

1. **隐式噪声翻译损失 $\mathcal{L}_{\text{implicit}}$**：

    - 功能：端到端优化翻译 + 去噪的整体效果
    - 核心思路：$\|\mathcal{D}(\mathcal{T}(I; \boldsymbol{\phi}); \boldsymbol{\theta}^*) - I_{\text{GT}}\|_1$
    - 设计动机：不直接约束翻译后噪声的形态，而是通过冻结去噪器的性能反向要求翻译网络输出适合高斯去噪器的输入

2. **显式噪声翻译损失 $\mathcal{L}_{\text{explicit}}$（两部分）**：

    - **空间域匹配 $\mathcal{L}_{\text{spatial}}$**：用 1-Wasserstein 距离匹配翻译噪声 $n_\mathcal{T}$ 与高斯参考噪声 $n_\mathcal{G}$ 的一维分布
        - 做法：将两者按通道展平排序，计算排序后元素的 L1 距离
        - 确保翻译噪声在元素级别服从高斯分布
    - **频域匹配 $\mathcal{L}_{\text{freq}}$**：用 1-Wasserstein 距离匹配两者傅里叶系数幅度的分布
        - 数学基础：空间不相关的高斯噪声，其傅里叶系数幅度服从 Rayleigh 分布
        - 做法：对翻译噪声和参考噪声分别做 FFT，匹配幅度分布
        - 确保翻译噪声是空间不相关的（消除结构化噪声模式）
    - 合并：$\mathcal{L}_{\text{explicit}} = \mathcal{L}_{\text{spatial}} + \beta \cdot \mathcal{L}_{\text{freq}}$

3. **高斯注入块（Gaussian Injection Block, GIBlock）**：

    - 功能：在 U-Net 的每层内部注入高斯噪声
    - 核心思路：不在输入端添加（会扭曲信号），而在网络子模块内部逐步施加高斯先验
    - 组成：NAFBlock + 高斯噪声注入 + 残差连接
    - 消融验证：GIBlock 是翻译网络在推理时可靠地将未见噪声映射到高斯分布的关键

### 损失函数 / 训练策略

总损失：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{implicit}} + \alpha \cdot \mathcal{L}_{\text{explicit}}$

- $\alpha = 5 \times 10^{-2}$，$\beta = 2 \times 10^{-3}$
- 高斯注入噪声水平 $\tilde{\sigma} = 100$
- 去噪网络训练数据：BSD400 + WED（加高斯噪声 σ=15）+ SIDD
- 翻译网络训练数据：仅 SIDD 真实 noisy-clean 对 + 随机高斯噪声 [0,15] 增强
- 翻译网络基于轻量级 U-Net 架构

## 实验关键数据

### 主实验（OOD 平均 PSNR，dB）

| 方法 | SIDD (ID) | OOD Avg↑ | 提升 |
|------|----------|---------|------|
| NAFNet | 39.97 | 38.43 | 基线 |
| **NAFNet + NTN** | 39.24 | **39.94** | **+1.51** |
| Xformer | 39.98 | 38.58 | 基线 |
| **Xformer + NTN** | 39.10 | **40.04** | **+1.46** |
| AFM (之前 SOTA) | 38.29 | 39.07 | — |
| Mask-Denoising | 38.91 | 38.56 | — |
| CLIP-Denoising | 38.03 | 38.53 | — |

### 消融实验

| 配置 | SIDD | OOD Avg | 说明 |
|------|------|---------|------|
| 基线翻译（仅 implicit） | 39.35 | 39.27 | 最简版本 |
| + GIBlock | 39.05 | 39.61 | +0.34 OOD |
| + Explicit loss | 39.33 | 39.61 | +0.34 OOD |
| + Both（完整版） | 39.24 | **39.94** | +0.67 OOD |

### 与简单加高斯噪声对比

| 输入 | SIDD | OOD Avg |
|------|------|---------|
| 原始噪声图 I | 37.77 | 17.89 |
| I + N(σ=5) | 38.15 | 22.93 |
| I + N(σ=10) | 38.76 | 39.22 |
| I + N(σ=15) | 39.16 | 38.95 |
| **翻译后 $I_\mathcal{T}$** | **39.24** | **39.94** |

### 关键发现

1. **固定加噪的局限性一目了然**：σ=10 在某些数据集好但在另一些差，σ=15 同理；不同图像/数据集需要不同噪声水平，而翻译网络可自适应
2. **翻译网络可跨去噪器迁移**：用 NAFNet 训练的翻译网络直接搭配 Xformer 使用，OOD 性能（39.94 dB）与专门为 Xformer 训练的（40.04 dB）几乎一致
3. **ID 性能并非下降而是去过拟合**：在 SIDD 上的微小 PSNR 下降是因为其他方法过拟合了训练集中的伪影（如拉链纹理），本文方法反而避免了这种过拟合
4. **计算开销极小**：翻译网络仅 0.29M 参数、1.07G MACs，对比 NAFNet 的 29.1M/16.23G 和 Xformer 的 25.1M/142.68G，几乎可忽略

## 亮点与洞察

1. **朴素观察 → 优雅框架**：从"加高斯噪声反而提升去噪效果"的观察出发，推导出完整的噪声翻译理论
2. **损失函数的数学动机极强**：
    - 空间域：1-Wasserstein 匹配确保元素级高斯分布
    - 频域：利用高斯噪声 FFT 幅度服从 Rayleigh 分布的数学性质，确保空间不相关性
3. **即插即用架构**：翻译网络与去噪网络完全解耦，一次训练可搭配任何预训练高斯去噪器
4. **不需要测试时优化**：与 LAN 等方法相比，推理时无需逐像素优化，可扩展到任意分辨率
5. **可视化说服力强**：翻译前后的噪声分布直方图清楚展示了从结构化噪声到高斯噪声的转变

## 局限与展望

1. 翻译网络仅在 SIDD 训练，遇到与 SIDD 差异极大的噪声类型时可能受限
2. ID 性能有微小下降（约 0.7dB），对追求极致 ID 性能的场景需额外微调
3. 去噪网络需预训练在同一高斯噪声水平（σ=15）上，噪声水平不匹配时效果待验证
4. 仅在图像去噪上验证，视频去噪和其他图像恢复任务（如去模糊、超分）的适用性待探索

## 相关工作与启发

- **DnCNN**：CNN 去噪先驱，频域训练提升泛化性
- **NAFNet / Restormer / KBNet**：强大的去噪骨干网络，但泛化能力有限
- **Anscombe 变换 / Pixel-Shuffle Down-sampling**：固定变换简化噪声，但适应性差
- **LAN**：测试时优化像素级偏移，有效但不可扩展（限 256×256）
- **AFM**：对抗训练提升鲁棒性，但仍受限于训练分布
- 核心启发："不要尝试去噪所有噪声类型，而是先将所有噪声统一翻译成一种你已经擅长处理的噪声"——这种分而治之的思路适用于更广泛的领域

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 噪声翻译的思路新颖且有坚实的数学动机
- 实验充分度: ⭐⭐⭐⭐⭐ — 9 个 OOD 基准、详尽消融、可视化分析、迁移性验证
- 写作质量: ⭐⭐⭐⭐⭐ — 从直觉到理论到实验，论述逻辑清晰
- 价值: ⭐⭐⭐⭐⭐ — 即插即用、轻量高效，对图像去噪领域有实际推动意义

<!-- RELATED:START -->

## 相关论文

- [Are Deep Speech Denoising Models Robust to Adversarial Noise?](../../ICLR2026/image_restoration/are_deep_speech_denoising_models_robust_to_adversarial_noise.md)
- [PNG: Diffusion-Based sRGB Real Noise Generation via Prompt-Driven Noise Representation Learning](diffusion-based_srgb_real_noise_generation_via_prompt-driven_noise_representatio.md)
- [NEC-Diff: Noise-Robust Event–RAW Complementary Diffusion for Seeing Motion in Extreme Darkness](nec-diff_noise-robust_event-raw_complementary_diffusion_for_seeing_motion_in_ext.md)
- [Classic Video Denoising in a Machine Learning World: Robust, Fast, and Controllable](../../CVPR2025/image_restoration/classic_video_denoising_in_a_machine_learning_world_robust_fast_and_controllable.md)
- [Blind Image Deblurring with Noise-Robust Kernel Estimation](../../ECCV2024/image_restoration/blind_image_deblurring_with_noise-robust_kernel_estimation.md)

<!-- RELATED:END -->
