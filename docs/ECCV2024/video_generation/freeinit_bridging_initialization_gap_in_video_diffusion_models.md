---
title: >-
  [论文解读] FreeInit: Bridging Initialization Gap in Video Diffusion Models
description: >-
  [ECCV 2024][视频扩散模型] 发现视频扩散模型存在训练-推理初始化差异（训练时低频信息泄露导致初始噪声具有时序相关性，而推理时使用无相关的高斯噪声），提出 FreeInit 通过迭代精炼初始噪声的时空低频成分来弥合该差异，显著提升视频生成的时序一致性。
tags:
  - ECCV 2024
  - 视频扩散模型
  - 噪声初始化
  - 时序一致性
  - 频率域
  - 视频生成
---

# FreeInit: Bridging Initialization Gap in Video Diffusion Models

**会议**: ECCV 2024  
**arXiv**: [2312.07537](https://arxiv.org/abs/2312.07537)  
**代码**: [项目主页](https://tianxingwu.github.io/pages/FreeInit/)  
**领域**: 视频生成  
**关键词**: 视频扩散模型, 噪声初始化, 时序一致性, 频率域, 推理策略

## 一句话总结

发现视频扩散模型存在训练-推理初始化差异（训练时低频信息泄露导致初始噪声具有时序相关性，而推理时使用无相关的高斯噪声），提出 FreeInit 通过迭代精炼初始噪声的时空低频成分来弥合该差异，显著提升视频生成的时序一致性。

## 研究背景与动机

基于扩散模型的视频生成取得了快速进展，大多数模型构建在预训练图像扩散模型（如 Stable Diffusion）之上，通过添加时序层和大规模视频数据训练实现视频生成。然而，生成视频仍普遍存在**时序不一致**和**不自然动态**的问题。

作者深入研究了视频扩散模型的**噪声初始化**问题，发现了一个此前被忽视的隐式训练-推理差异：

**核心发现一：信息泄露**
- 扩散过程无法完全将干净 latent 腐蚀为纯高斯噪声，尤其在**低频带**
- 低频成分的腐蚀速率远慢于高频成分（从可视化和 SNR 分析均可验证）
- 在最终扩散步 $t=1000$ 时，噪声 latent 仍包含来自输入视频的大量低频信息

**核心发现二：低频主导推理质量**
- 即使替换80%的高频成分，生成结果仍然基本不变
- 说明初始噪声的低频成分**决定了生成结果的整体分布**

**差异的后果**：
- 训练时：初始噪声从真实视频腐蚀而来，低频带保留时序相关性
- 推理时：使用 i.i.d. 高斯噪声，完全无时序相关性
- 结果：推理质量退化，时序不一致和不自然运动

## 方法详解

### 整体框架

FreeInit 是一种**推理时采样策略**，无需额外训练或微调。核心流程：

1. 初始化高斯噪声 $\epsilon$
2. DDIM 去噪采样得到干净 latent $z_0$
3. 用原始噪声 $\epsilon$ 通过前向扩散重新加噪得到 $z_T$（保留低频信息）
4. 噪声重初始化：融合 $z_T$ 的低频与新随机噪声 $\eta$ 的高频
5. 用重初始化的噪声 $z_T'$ 作为下一轮 DDIM 采样的起点
6. 迭代以上过程

### 关键设计

#### 1. 信息泄露的频率分析

**SNR 频率分布分析**：在前向扩散过程中，对 noisy latent $z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ 进行逐频段 SNR 分析。结果表明：

- 低频成分（蓝绿色曲线）SNR 衰减极慢
- 高频成分（红色曲线）SNR 快速降低
- 在 $t=1000$ 时，低频带 SNR 甚至大于 0 dB，意味着严重的信息泄露

**低频成分主导性验证**：从真实视频扩散得到 $z_T$，逐步替换其高频成分为随机高斯噪声：
- 即使只保留20%的原始低频信息，生成结果仍接近完整 $z_T$ 的结果
- 当全部替换为高斯噪声时，才出现明显的质量下降

#### 2. 去噪-加噪循环（Denoise and Diffuse）

从初始高斯噪声 $\epsilon$ 出发：
1. 通过 DDIM 采样得到初步清晰 latent $z_0$
2. 使用**同一噪声** $\epsilon$（关键细节！）通过 DDPM 前向过程重新加噪：

$$z_T = \sqrt{\bar{\alpha}_T} z_0 + \sqrt{1-\bar{\alpha}_T} \epsilon = \sqrt{\bar{\alpha}_T}(\text{DDIM}_{sample}(\epsilon)) + \sqrt{1-\bar{\alpha}_T} \epsilon$$

为什么使用同一噪声 $\epsilon$？因为随机采样新噪声会引入中频带的显著不确定性，破坏已有的时空相关性。

#### 3. 噪声重初始化（Noise Reinitialization）

通过时空频率滤波器融合 $z_T$ 的低频和随机噪声 $\eta$ 的高频：

$$\mathcal{F}_{z_T}^L = \text{FFT}_{3D}(z_T) \odot \mathcal{H}$$
$$\mathcal{F}_\eta^H = \text{FFT}_{3D}(\eta) \odot (1 - \mathcal{H})$$
$$z_T' = \text{IFFT}_{3D}(\mathcal{F}_{z_T}^L + \mathcal{F}_\eta^H)$$

其中：
- $\text{FFT}_{3D}$ 在时间和空间维度上操作，捕获时空频率信息
- $\mathcal{H}$ 是高斯低通滤波器（GLPF），归一化截止频率 $D_0 = 0.25$
- 保留 $z_T$ 的低频（含时序相关性）
- 引入 $\eta$ 的高频（提供视觉细节的随机性）

#### 4. 迭代精炼

上述过程可迭代多次执行：
- 每次迭代：低频成分通过去噪获得改进的时空一致性
- 同时：高频通过重初始化获得灵活性
- 初始噪声逐步向训练分布靠拢

默认使用4次 FreeInit 迭代。

### 损失函数 / 训练策略

FreeInit 是**纯推理方法**，不涉及任何训练或微调。关键配置：
- 低通滤波器类型：高斯低通滤波器（GLPF）
- 截止频率：$D_0 = 0.25$（归一化时空频率）
- 迭代次数：默认4次
- 各模型参数保持不变

## 实验关键数据

### 主实验

在三个公开模型上评估，使用 UCF-101 和 MSR-VTT 提示：

**时序一致性** (DINO ↑)：

| 方法 | UCF-101 | MSR-VTT |
|------|---------|---------|
| AnimateDiff | 85.24 | 83.24 |
| AnimateDiff+FreeInit | **92.01** | **91.86** |
| ModelScope | 88.16 | 88.95 |
| ModelScope+FreeInit | **91.11** | **93.28** |
| VideoCrafter | 85.62 | 84.68 |
| VideoCrafter+FreeInit | **89.27** | **88.72** |

FreeInit 在所有模型上将时序一致性提升 **2.92~8.62** 分。

**运动质量**：

| 方法 | FVD ↓ | MS差异 ↓ | DD差异 ↓ |
|------|-------|---------|---------|
| AnimateDiff | 1340.96 | 7.33 | 20.2 |
| AnimateDiff+FreeInit | **1032.47** | **0.04** | **1.53** |
| ModelScope | 785.30 | 1.64 | 3.71 |
| ModelScope+FreeInit | **702.15** | **0.35** | 8.22 |
| VideoCrafter | 730.04 | 6.14 | 15.79 |
| VideoCrafter+FreeInit | **675.39** | **3.19** | **6.44** |

FVD 全面改善，运动流畅度和动态程度更接近真实视频。

### 消融实验

1. **噪声重初始化与滤波器选择**：

| 方法 | UCF-101 DINO ↑ | MSR-VTT DINO ↑ |
|------|---------------|----------------|
| AnimateDiff (w/o NR) | 86.77 | 85.18 |
| AnimateDiff (ILPF) | 87.53 | 86.17 |
| AnimateDiff (GLPF) | **92.01** | **91.86** |
| ModelScope (w/o NR) | 88.20 | 90.90 |
| ModelScope (GLPF) | **91.11** | **93.28** |
| VideoCrafter (w/o NR) | 86.09 | 87.11 |
| VideoCrafter (GLPF) | **89.27** | **89.33** |

**关键结论**：
- 噪声重初始化至关重要
- 软过渡的GLPF显著优于硬截止的ILPF（理想低通滤波器）
- 在中低频引入适度随机性对质量改善很重要

2. **迭代次数影响**：
    - 时序一致性随迭代次数持续提升
    - **第1次迭代提升最显著**（首次注入低频信息，大幅缩小训练-推理差异）
    - 后续迭代带来递减的改善

3. **与同等推理步数的比较**：
    - 仅1次 FreeInit 迭代（总 $2n$ 步）的时序一致性已超过 $5n$ 步的普通 DDIM（2.5倍时间开销）
    - 证明 FreeInit 不等价于简单增加采样步数——**好的开始比对坏初始态挣扎更重要**

### 关键发现

1. **训练-推理初始化差异是影响视频生成质量的根本原因之一**
2. **低频成分决定全局分布**：视频扩散模型的生成结果主要由初始噪声的低频成分决定
3. **使用原始噪声加噪很关键**：新噪声会在中频引入不确定性
4. **GLPF 优于 ILPF**：软过渡滤波器允许在中低频引入适度变化，避免过度限制
5. **方法具有广泛适用性**：不仅限于视频扩散模型，也适用于图像扩散模型如 SDXL

## 亮点与洞察

1. **问题定义精准**：首次从频率域系统研究视频扩散模型的初始噪声问题，将模糊的"生成质量差"归因为具体的"训练-推理初始化差异"
2. **方法极其简洁**：整个算法仅涉及 FFT、DDIM 采样和前向扩散，无需修改模型或训练，几行代码即可集成
3. **"好的开始"哲学**：与大量关注去噪过程改进的工作不同，本文强调初始状态的重要性——让初始噪声更接近训练分布比在去噪过程中挣扎更有效
4. **通用性极强**：无需针对不同模型调参（GLPF $D_0=0.25$ 通用即可），即插即用

## 局限与展望

1. **推理时间增加**：默认4次迭代意味着5倍的采样时间，虽然效率比单纯增加步数高，但仍是瓶颈
2. **粗到细采样策略**：作者在补充材料中讨论了可通过先低分辨率后高分辨率的策略缓解效率问题
3. **低频带定义的固定性**：当前使用固定的 $D_0=0.25$，未探索自适应的频率分界策略
4. **依赖 DDIM 的确定性**：方法建立在确定性采样的基础上，对随机采样的适用性未充分讨论

## 相关工作与启发

- **PYoCo (ICCV 2023)**：同样关注视频噪声先验，但需要大规模微调；FreeInit 无需训练
- **Common Diffusion Noise Schedules (Lin et al.)**：指出噪声调度无法完全腐蚀信息的问题（图像域），FreeInit 将这一发现扩展到视频域
- **Resampled Diffusion (Lugmayr et al.)**：修复任务中的重采样操作启发了迭代精炼策略
- **启发**：初始化差异可能也存在于其他条件生成任务（如3D、音频），类似的频率域精炼策略值得探索

## 评分

- **创新性**：★★★★☆ — 训练-推理初始化差异的洞察深刻，频率域精炼方案新颖
- **实用性**：★★★★★ — 零修改即插即用，效果显著且稳定，极具部署价值
- **实验充分度**：★★★★★ — 三个模型、两个数据集、多维指标、详尽消融，非常全面
- **写作质量**：★★★★★ — 可视化清晰、频率分析严谨、叙述流畅

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Exploring Pre-trained Text-to-Video Diffusion Models for Referring Video Object Segmentation](exploring_pre-trained_text-to-video_diffusion_models_for_referring_video_object_.md)
- [\[ECCV 2024\] SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Latent Video Diffusion](sv3d_novel_multi-view_synthesis_and_3d_generation_from_a_single_image_using_late.md)
- [\[ECCV 2024\] Kalman-Inspired Feature Propagation for Video Face Super-Resolution](kalman-inspired_feature_propagation_for_video_face_super-resolution.md)
- [\[ECCV 2024\] VFusion3D: Learning Scalable 3D Generative Models from Video Diffusion Models](vfusion3d_learning_scalable_3d_generative_models_from_video_diffusion_models.md)
- [\[CVPR 2025\] VideoGuide: Improving Video Diffusion Models without Training Through a Teacher's Guide](../../CVPR2025/video_generation/videoguide_improving_video_diffusion_models_without_training_through_a_teachers_.md)

</div>

<!-- RELATED:END -->
