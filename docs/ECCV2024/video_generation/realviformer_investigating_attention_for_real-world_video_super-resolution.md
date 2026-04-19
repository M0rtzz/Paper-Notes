---
title: >-
  [论文解读] RealViformer: Investigating Attention for Real-World Video Super-Resolution
description: >-
  [ECCV 2024][图像恢复][超分辨率] 本文系统研究了空间注意力和通道注意力在真实世界视频超分辨率（RWVSR）中的行为差异，发现通道注意力对退化伪影更鲁棒但会导致特征冗余，据此提出了带有改进通道注意力（ICA）和通道注意力融合（CAF）模块的 RealViformer，以更少的参数和更快的速度达到 SOTA。
tags:
  - ECCV 2024
  - 图像恢复
  - 超分辨率
  - 注意力机制
  - artifact propagation
  - Transformer
  - covariance
---

# RealViformer: Investigating Attention for Real-World Video Super-Resolution

**会议**: ECCV 2024  
**arXiv**: [2407.13987](https://arxiv.org/abs/2407.13987)  
**代码**: [https://github.com/Yuehan717/RealViformer](https://github.com/Yuehan717/RealViformer)  
**领域**: 图像修复  
**关键词**: real-world video super-resolution, channel attention, artifact propagation, Transformer, covariance

## 一句话总结

本文系统研究了空间注意力和通道注意力在真实世界视频超分辨率（RWVSR）中的行为差异，发现通道注意力对退化伪影更鲁棒但会导致特征冗余，据此提出了带有改进通道注意力（ICA）和通道注意力融合（CAF）模块的 RealViformer，以更少的参数和更快的速度达到 SOTA。

## 研究背景与动机

**领域现状**：视频超分辨率（VSR）是低级视觉中的核心任务。标准 VSR 假设 LR 帧由 HR 帧通过已知核下采样得到，近年来 Transformer 架构（如基于 Swin 的方法）在标准 VSR 上已取代 CNN 成为 SOTA。真实世界 VSR 面临来自相机成像系统、压缩、网络传输等复杂退化，缺乏 LR/HR 的封闭对应关系。

**现有痛点**：(a) 循环式 VSR 模型会通过隐状态在时间维度上传播伪影，在真实世界退化下尤为严重；(b) 标准 VSR 中表现优异的空间注意力 Transformer（如 Swin-based）在真实世界场景下反而产生更多伪影（见 Fig.1），性能不及卷积模型 RealBasicVSR；(c) 现有 RWVSR 方法（如 RealBasicVSR、FastRealVSR）主要基于 CNN 设计，缺乏对注意力机制在退化条件下行为的系统分析。

**核心矛盾**：空间注意力善于空间匹配但对局部退化高度敏感；通道注意力对退化更鲁棒但导致通道间高协方差（特征冗余），限制了重建能力。两种机制各有优劣，如何取长补短是关键。

**本文目标**：(a) 回答为什么 Transformer 在标准 VSR 上有效但在 RWVSR 上不佳；(b) 揭示通道注意力对退化的鲁棒性及其冗余问题；(c) 设计一个有效的通道注意力 RWVSR Transformer。

**切入角度**：从注意力机制的协方差计算本质出发，通过实验对比空间注意力和通道注意力在退化查询下的输出稳定性，量化通道协方差指标，发现问题后用 squeeze-excite 和协方差重标定来修正。

**核心 idea**：通道注意力因在大空间范围上计算协方差而对退化更鲁棒，通过 squeeze-excite 和基于注意力图的通道权重重标定可缓解其特征冗余问题，从而构建高效的 RWVSR Transformer。

## 方法详解

### 整体框架

RealViformer 采用单向循环 Transformer 框架：
- **光流估计**：使用 SPyNet 估计 $s^f_{(t-1)\to t}$ 并将上一时刻隐状态 $h_{t-1}$ warp 到当前时刻
- **重建模块 $\mathcal{R}$**：接收当前帧 $I^L_t$ 和对齐后的隐状态 $\hat{h}_{t-1}$，先通过 CAF 模块进行时间信息融合，再通过带 ICA 模块的三级编码器-解码器 Transformer 块进行重建
- **上采样模块 $\mathcal{U}$**：对重建特征进行上采样输出 HR 帧
- 编码器-解码器三级结构：Level 1/2/3 分别有 [2,3,4] 个 Transformer 块，[48,96,192] 个通道，[1,2,4] 个注意力头，squeeze factor 均为 4

### 关键设计

1. **Channel Attention Fusion (CAF) 模块**：

    - **功能**：用通道注意力融合当前帧浅层特征 $f_t$ 和对齐的隐状态 $\hat{h}_{t-1}$，限制隐状态中传播的伪影
    - **核心思路**：Query 由 $f_t$ 经 LayerNorm + 3×3 卷积生成；Key/Value 由 $\hat{h}_{t-1}$ 经 LayerNorm + 1×1 卷积 + 3×3 深度卷积生成后 chunk 分割。注意力图 $A_t \in \mathbb{R}^{C \times C}$ 按通道注意力公式计算：$A_t = \text{softmax}(Q_t K_t^T / \alpha)$。最终输出 $O_t = K_{1\times1} * K^d_{3\times3} * K_{1\times1} * \mathbf{C}[A_t V_t; f_t]$
    - **设计动机**：通道注意力在大空间范围上计算协方差（特征尺寸为 $\mathbb{R}^{1 \times HW}$），对局部退化不敏感。实验表明在模糊/噪声/压缩退化下，通道注意力输出余弦相似度达 0.98-0.99（空间注意力仅 0.75-0.92）

2. **Improved Channel Attention (ICA) 模块**：

    - **功能**：在 Transformer 块中替代原始通道注意力进行自注意力特征重建，缓解通道冗余问题
    - **核心思路**：(a) Squeeze-and-Excite：先用 squeeze 卷积将输入通道压缩 $r$ 倍，在压缩空间做通道注意力（注意力图大小为 $\mathbb{R}^{C/r \times C/r}$），再用 excite 卷积恢复通道数，从而生成新的非冗余信息；(b) 基于协方差的通道重标定：从注意力图 $A_r$ 的行方向取平均值和最大值，经线性层+sigmoid 预测每个通道的标量权重 $\in \mathbb{R}^{C/r \times 1}$，对注意力输出进行通道加权
    - **设计动机**：通道注意力输出的每个通道是 Value 通道的加权求和，导致通道间协方差显著升高（$ac(O) = 0.87$ vs 输入 $\approx 0.15$）。高协方差意味着特征冗余，不利于学习。Squeeze-excite 在压缩空间做注意力产生低冗余特征；基于注意力图的权重预测则利用通道间关系信息，比 SE-Net 的 naive 池化更精确

3. **探索性实验体系**：

    - **功能**：系统验证空间 vs 通道注意力的敏感性差异和通道协方差问题
    - **核心思路**：(a) 构建敏感性对比实验（Fig.2）：用干净帧做注意力得 $O$，加退化后做注意力得 $O_{D_i}$，比较余弦相似度；(b) 将注意力模块插入循环 VSR baseline 的时间聚合位置（Fig.3），在合成退化下训练测试比较 PSNR/LPIPS 改善；(c) 量化通道协方差指标 $ac(Z) = \frac{1}{d}\sum_{i\neq j}|Cov(Z)|_{i,j}$
    - **设计动机**：为 RealViformer 的设计选择提供实验依据，而非凭直觉设计

### 损失函数 / 训练策略

- **两阶段训练**（沿用 RealBasicVSR 策略）：
    - **Stage 1**（300K iterations）：Charbonnier loss + SSIM loss
    - **Stage 2**（130K iterations）：Charbonnier loss + SSIM loss + Perceptual loss + GAN loss，权重分别为 1, 0.001, 1, 0.005
- **退化合成**：沿用 Real-ESRGAN 的随机退化管线（模糊、噪声、JPEG 压缩、视频压缩的随机组合）
- **训练细节**：REDS 数据集，15 帧序列，64×64 裁剪，batch size 16，4× Quadro RTX 8000 GPU，SPyNet 前 5K iterations 冻结

## 实验关键数据

### 主实验

| 方法 | Params(M) | Runtime(ms) | VideoLQ ILNIQE↓ | VideoLQ NRQM↑ | RealVSR ILNIQE↓ | RealVSR NRQM↑ | REDS4 PSNR↑ | REDS4 LPIPS↓ | UDM10 PSNR↑ | UDM10 LPIPS↓ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RealSR | 16.7 | 180 | 26.63 | 6.054 | 32.81 | 5.610 | 22.02 | 0.5991 | 25.37 | 0.4811 |
| Real-ESRGAN | 16.7 | 196 | 27.97 | 6.057 | 31.93 | 6.245 | 21.56 | 0.3533 | 24.96 | 0.3395 |
| BSRGAN | 16.7 | 180 | 27.49 | 6.156 | 32.65 | 6.152 | 22.94 | 0.3766 | 25.97 | 0.3388 |
| RealBasicVSR | 6.3 | 73 | 25.98 | 6.306 | 30.37 | 6.582 | 23.09 | 0.2991 | 25.96 | 0.3209 |
| **RealViformer** | **5.3** | **49** | **25.94** | **6.338** | **28.61** | **6.588** | **23.34** | **0.2877** | **26.42** | **0.3063** |

### 消融实验

| 方法 | CAF | ICA | VideoLQ NRQM↑ | UDM10 LPIPS↓ |
|:---|:---:|:---:|:---:|:---:|
| Sp-baseline（空间注意力） | - | - | 6.061 | 0.3482 |
| Ch-baseline（通道注意力） | ✗ | ✗ | 6.181 | 0.3085 |
| RealViformer⁻ | ✓ | ✗ | 6.196 | 0.2933 |
| **RealViformer** | **✓** | **✓** | **6.338** | **0.2877** |

### 关键发现

- **通道注意力对退化更鲁棒**：余弦相似度实验表明通道注意力在模糊/噪声/压缩下输出变化极小（0.98-0.99），空间注意力变化大（0.75-0.92），这源于通道注意力在全局空间范围 $\mathbb{R}^{1 \times HW}$ 上计算协方差
- **通道注意力导致特征冗余**：通道注意力输出的通道间协方差 $ac(O) = 0.87$，远高于输入（$\approx 0.15$）和空间注意力输出，在标准 VSR 中通道注意力 SSIM 低于空间注意力（0.8338 vs 0.8432）
- **ICA 有效降低冗余**：CAF+ICA 使传播信息的通道相关性从 0.436 降至 0.422，且高频分量功率增强（RPS 分析）
- **参数和速度优势大**：5.3M 参数（RealBasicVSR 6.3M），49ms 运行时间（RealBasicVSR 73ms），全面更快更轻量
- **用户研究验证**：30 名评估者对 85 帧的评分中，RealViformer 在 MOS 上超越所有对比方法

## 亮点与洞察

1. **系统的分析范式**：不是直接提新架构，而是先深入分析空间/通道注意力在退化条件下的行为差异，用控制变量实验给出量化证据，再基于发现设计模型，说服力强
2. **揭示通道注意力的冗余本质**：这个发现对整个低级视觉 Transformer 社区有广泛影响，因为 Restormer 等方法广泛使用通道注意力
3. **简洁有效的解决方案**：ICA 仅用 squeeze-excite + 注意力图权重预测两个简单修改，无需复杂架构设计
4. **更少参数+更快速度+更好性能**：同时满足三者在 SR 领域较为罕见

## 局限与展望

1. **单向循环框架**：只用前向传播，未利用后向信息，双向框架可能进一步提升性能
2. **流估计依赖 SPyNet**：SPyNet 较为轻量但精度有限，更好的光流或可变形对齐可能改善时间聚合
3. **ICA 的 squeeze factor 固定为 4**：不同层级可能需要不同的压缩率，可探索自适应策略
4. **仅验证了 ×4 超分**：未报告 ×2 或 ×8 的结果
5. **真实世界无参考评价指标有限**：ILNIQE/NRQM 虽比 NIQE 好但仍不完美，缺乏更鲁棒的感知质量评估

## 相关工作与启发

- **Restormer [Zamir et al., CVPR 2022]**：提出通道注意力用于图像修复，本文发现其通道冗余问题并提出改进
- **RealBasicVSR [Chan et al., CVPR 2022]**：RWVSR 的 CNN SOTA，本文延续其训练策略但将其替换为 Transformer
- **VICReg [Bardes et al., 2021]**：自监督学习中的方差-不变性-协方差正则化，启发了本文对通道协方差的量化分析
- **SE-Net [Hu et al., CVPR 2018]**：squeeze-excite 机制的原始提出者，本文在两个关键点上做了区分性改进
- **对视频超分中注意力的启发**：通道注意力适合时间聚合（限制伪影传播），空间注意力适合空间细节重建的互补思路可进一步探索

## 评分

- **新颖性**: ⭐⭐⭐⭐ 核心贡献在于系统性的分析发现而非架构创新，揭示了通道注意力在RWVSR中的优势和冗余问题，分析范式有价值
- **实验充分度**: ⭐⭐⭐⭐⭐ 敏感性分析、协方差量化、多数据集对比、消融实验、用户研究、RPS分析，实验体系非常完整
- **写作质量**: ⭐⭐⭐⭐ 叙事流畅，从发现问题到解决问题的逻辑清晰，exploration → finding → verification → model design 的结构很好
- **价值**: ⭐⭐⭐⭐ 对RWVSR和低级视觉Transformer设计都有启发，通道注意力冗余的发现可推广到更多任务

<!-- RELATED:START -->

## 相关论文

- [Kalman-Inspired Feature Propagation for Video Face Super-Resolution](kalman-inspired_feature_propagation_for_video_face_super-resolution.md)
- [VSRM: A Robust Mamba-Based Framework for Video Super-Resolution](../../ICCV2025/video_generation/vsrm_a_robust_mamba-based_framework_for_video_super-resolution.md)
- [BF-STVSR: B-Splines and Fourier—Best Friends for High Fidelity Spatial-Temporal Video Super-Resolution](../../CVPR2025/video_generation/bf-stvsr_b-splines_and_fourier---best_friends_for_high_fidelity_spatial-temporal.md)
- [Compressed-Domain-Aware Online Video Super-Resolution](../../CVPR2026/video_generation/compressed-domain-aware_online_video_super-resolution.md)
- [RealCam-I2V: Real-World Image-to-Video Generation with Interactive Complex Camera Control](../../ICCV2025/video_generation/realcam-i2v_real-world_image-to-video_generation_with_interactive_complex_camera.md)

<!-- RELATED:END -->
