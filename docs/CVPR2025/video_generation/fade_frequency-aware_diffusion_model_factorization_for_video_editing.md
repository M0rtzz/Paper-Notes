---
title: >-
  [论文解读] FADE: Frequency-Aware Diffusion Model Factorization for Video Editing
description: >-
  [CVPR 2025][video editing] 提出 FADE，一种免训练的视频编辑框架，通过频率感知的扩散模型分解和频谱引导调制，利用预训练 T2V 模型的视频先验实现高保真、时间一致的视频编辑。
tags:
  - CVPR 2025
  - video editing
  - diffusion model
  - frequency domain
  - training-free
  - temporal consistency
---

# FADE: Frequency-Aware Diffusion Model Factorization for Video Editing

**会议**: CVPR 2025  
**arXiv**: [2506.05934](https://arxiv.org/abs/2506.05934)  
**代码**: [https://github.com/EternalEvan/FADE](https://github.com/EternalEvan/FADE)  
**领域**: image_generation  
**关键词**: video editing, diffusion model, frequency-aware factorization, spectrum-guided modulation, training-free

## 一句话总结

提出 FADE，一种免训练的视频编辑方法，通过分析 T2V 模型中各 transformer block 的频率角色（sketching vs sharpening），利用频谱引导调制在频域中分离保留与编辑内容，实现高质量的外观和运动编辑。

## 研究背景与动机

**领域现状**: 扩散模型已将视频编辑能力提升到高保真和强文本对齐水平，但传统基于 T2I 模型的方法在处理视频动态（尤其是运动编辑）时表现不足。

**现有痛点**:
- T2I 模型缺乏视频先验，导致时间不一致和运动编辑能力有限
- Null-text inversion 方法需要大量迭代计算 null-text 嵌入，耗时严重
- 注意力特征注入方法（attention injection）内存需求高，且限制了编辑灵活性
- 直接在视频扩散模型上加噪声（如 CogVideoX-V2V）无法充分利用视频先验

**核心矛盾**: 视频扩散模型（T2V）拥有丰富的时空先验知识，但其庞大的计算需求（数十个 transformer block 的 full-attention）使得将以往 T2I 编辑技术直接迁移变得不可行。

**本文目标**: 设计一种高效、灵活的视频编辑策略，能充分利用预训练 T2V 模型中的视频先验，支持外观和运动两类编辑。

**切入角度**: 从频域视角分析 T2V 模型内部各 block 的功能分工，发现早期 block 负责勾勒低频空间布局和时间动态（sketching blocks），后期 block 精炼高频细节（sharpening blocks），基于此进行角色分解。

**核心 idea**: 通过频率感知的 block 分解 + 频谱引导调制，只用少量 sketching blocks 提供低频结构引导，既降低计算开销，又释放了视频扩散先验的编辑潜力。

## 方法详解

### 整体框架

1. 使用预训练 T2V 模型（CogVideoX, 48 层 DiT）对输入视频进行 DDIM inversion，得到噪声 $\boldsymbol{z}_T^*$ 和反转轨迹 $\{\boldsymbol{z}_t^*\}_{t=0}^T$
2. 在去噪采样过程中，对前 4 个 sketching blocks 提取源视频和目标视频的 full-attention 输出 $\boldsymbol{F}_t^*$ 和 $\boldsymbol{F}_t$
3. 对 attention 输出进行 3D DFT 变换，通过低通滤波器分离低频结构信息
4. 计算频谱引导项 $\mathcal{G}_t$，用其梯度调制 DDIM 采样轨迹

### 关键设计

**1. 频率感知的 T2V 模型分解（Frequency-Aware Factorization）**
- **功能**: 将 T2V 模型的 48 个 transformer block 分为 sketching blocks（前 4 层）和 sharpening blocks（后 44 层）。
- **核心思路**: 可视化分析发现 early blocks 的注意力图沿对角线密集对齐（主对角线=帧内空间结构，次对角线=帧间时间对应），spectrum 集中在低频，输出模糊——它们勾勒基础布局和运动。Late blocks 的注意力分布更稀疏均匀，处理高频纹理、颜色等细节。
- **设计动机**: 利用这一功能分工，编辑时只需操作 sketching blocks 进行结构重建引导，sharpening blocks 自由生成细节，既高效（减少计算量）又灵活（不限制高频编辑）。

**2. 频谱引导调制（Spectrum-Guided Modulation）**
- **功能**: 将 sketching blocks 的 attention 输出变换到频域，用低通滤波器提取低频成分，计算源视频与目标视频的低频差异作为引导信号。
- **核心思路**: 对 attention 输出 $\boldsymbol{F}_t$ 进行 3D DFT（空间 + 时间维度），得到 $\mathcal{F}_t$；低通滤波后计算频谱引导 $\mathcal{G}_t = \|\text{LP}(\mathcal{F}_t) - \text{LP}(\mathcal{F}_t^*)\|_2^2$；用 $\mathcal{G}_t$ 对 $\boldsymbol{z}_t$ 的梯度调制采样轨迹：$\boldsymbol{z}_{t-1} = \text{DDIM}(\boldsymbol{\epsilon}_\theta, \boldsymbol{z}_t, t, \boldsymbol{y}_{tgt}) - \lambda \text{Norm}(\nabla_{\boldsymbol{z}_t} \mathcal{G}_t)$。
- **设计动机**: 在频域而非特征域进行引导，避免了直接注入 attention 特征导致的信息泄漏（源视频高频细节不当保留），仅保留低频结构（基础空间布局+时间运动），给高频细节留出编辑空间。

**3. 双分支采样策略（Dual Branch Strategy）**
- **功能**: 在每个去噪步骤中，同时对源视频反转轨迹 $\boldsymbol{z}_t^*$ 和目标视频 $\boldsymbol{z}_t$ 运行 sketching blocks，使用相同的源提示 $\boldsymbol{y}_{src}$ 计算 attention 输出。
- **核心思路**: 源分支提供参考结构信息，目标分支使用编辑提示 $\boldsymbol{y}_{tgt}$ 进行完整去噪，两者的 sketching blocks 输出差异驱动频谱引导。
- **设计动机**: 避免了 null-text optimization 的迭代开销，也不需要直接交换或混合 attention map，提供了更灵活的引导机制。

### 损失函数 / 训练策略

- **免训练方法**，无需任何优化或微调
- 使用 DDIM 采样 $T=50$ 步，引导区间 $[0, 0.6T]$
- 引导权重 $\lambda$ 在 10-15 之间，根据编辑任务调整
- 使用 BLIP 等多模态语言模型自动生成源视频文本描述
- 低通滤波器保留约 2/3 的频率分量

## 实验关键数据

### 主实验（DAVIS 数据集 + 真实视频）

| 方法 | CLIP↑ | M.PSNR↑ | LPIPS↓ | OSV↓ | 人类偏好↑ |
|---|---|---|---|---|---|
| **外观编辑** | | | | | |
| Tune-A-Video | 0.3522 | 19.86 | 0.4625 | 35.01 | 0.12 |
| FateZero | 0.3562 | 20.65 | 0.3057 | 33.23 | 0.29 |
| CogVideoX-V2V | 0.3754 | 18.96 | 0.4811 | 31.45 | 0.09 |
| **FADE (Ours)** | **0.3762** | **20.69** | **0.3085** | **31.36** | **0.35** |
| **运动编辑** | | | | | |
| Tune-A-Video | 0.3281 | 18.68 | 0.4637 | 35.85 | 0.10 |
| FateZero | 0.3259 | 19.02 | 0.3712 | 34.47 | 0.13 |
| CogVideoX-V2V | 0.3678 | 18.17 | 0.4928 | 35.52 | 0.19 |
| **FADE (Ours)** | **0.3683** | **19.26** | **0.3692** | **32.28** | **0.43** |

### 消融实验

| 配置 | CLIP↑ | M.PSNR↑ | LPIPS↓ | OSV↓ | 时间 |
|---|---|---|---|---|---|
| Symm. blocks | 0.3659 | 20.73 | 0.3367 | 32.61 | 5 min |
| W/o factorization | 0.3691 | 20.94 | 0.3328 | 32.05 | 12 min |
| W/o filter | 0.3612 | 20.89 | 0.3364 | 32.28 | 3 min |
| **FADE (Ours)** | **0.3728** | 20.87 | 0.3352 | **31.77** | **3 min** |

### 关键发现

1. **Sketching blocks 足矣**: 仅用前 4 个 block（共 48 个）即可获得最佳编辑质量，加入 sharpening blocks 反而会误导模型，降低编辑性能。
2. **低通滤波的关键作用**: 去掉低通滤波器后，高频信息泄漏导致目标物体保留过多源特征，文本对齐度下降（CLIP 从 0.3728 降至 0.3612）。
3. **运动编辑的显著优势**: FADE 在运动编辑上的人类偏好得分（0.43）远超其他方法，得益于视频先验的充分利用。
4. **效率提升**: FADE 3 分钟完成编辑，传统方法需 15 分钟以上。

## 亮点与洞察

- 从频域视角揭示了 T2V 模型内部 block 的功能分工（sketching vs sharpening），这一发现具有独立价值
- 在频域而非特征域进行引导的设计巧妙地避免了信息泄漏问题
- 免训练设计使方法具有极强的实用性，可直接应用于各类 T2V 模型
- 同时支持外观和运动编辑的统一框架
- 反直觉发现：使用更少的 block 做引导反而能获得更好的编辑效果

## 局限与展望

- 编辑性能依赖底层 T2V 模型的生成能力
- 在严重遮挡场景下，需要高级时间推理能力，当前模型不足以应对
- 只探索了一种 T2V 模型（CogVideoX），未验证在其他架构上的泛化性
- 低通滤波器的频率截断比例（2/3）需要经验性调整
- 运动编辑的复杂度有限，难以处理大幅度的运动变化

## 相关工作与启发

- 与 FateZero 等注意力注入方法相比，频域引导更灵活，不受 attention map 交换的限制
- 与 CogVideoX-V2V 使用相同 T2V 模型但效果显著更好，证明了分解策略的价值
- 频率感知的思路可推广到其他扩散模型任务（如 3D 生成、音频编辑等）
- Block 功能分析方法论可用于理解其他大规模 Transformer 架构

## 评分

⭐⭐⭐⭐ — 创新的频域分析视角 + 免训练实用性强，但局限于特定 T2V 模型

<!-- RELATED:START -->

## 相关论文

- [Identity-Preserving Text-to-Video Generation by Frequency Decomposition](identity-preserving_text-to-video_generation_by_frequency_decomposition.md)
- [Improved Video VAE for Latent Video Diffusion Model](improved_video_vae_for_latent_video_diffusion_model.md)
- [Exploring Temporally-Aware Features for Point Tracking](exploring_temporally-aware_features_for_point_tracking.md)
- [Target-Aware Video Diffusion Models](../../ICLR2026/video_generation/target-aware_video_diffusion_models.md)
- [Training-free Motion Factorization for Compositional Video Generation](../../CVPR2026/video_generation/training-free_motion_factorization_for_compositional_video_generation.md)

<!-- RELATED:END -->
