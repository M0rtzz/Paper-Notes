---
title: >-
  [论文解读] SpecGuard: Spectral Projection-based Advanced Invisible Watermarking
description: >-
  [ICCV2025][AI安全][不可见水印] SpecGuard 提出将水印信息嵌入到小波分解后的高频子带的频谱域中（通过 FFT 近似的频谱投影），编码端用强度因子增强鲁棒性，解码端利用 Parseval 定理设计可学习阈值进行比特恢复，在保持高图像质量（PSNR>42dB）的同时实现了对畸变、再生成和对抗攻击的全面鲁棒性，超越了现有 SOTA 方法。
tags:
  - ICCV2025
  - AI安全
  - 不可见水印
  - 频谱投影
  - 小波变换
  - FFT
  - Parseval定理
---

# SpecGuard: Spectral Projection-based Advanced Invisible Watermarking

**会议**: ICCV2025  
**arXiv**: [2510.07302](https://arxiv.org/abs/2510.07302)  
**代码**: https://github.com/SpecGuard (已开源)  
**领域**: AI安全 / 数字水印  
**关键词**: 不可见水印, 频谱投影, 小波变换, FFT, Parseval定理

## 一句话总结
SpecGuard 提出将水印信息嵌入到小波分解后的高频子带的频谱域中（通过 FFT 近似的频谱投影），编码端用强度因子增强鲁棒性，解码端利用 Parseval 定理设计可学习阈值进行比特恢复，在保持高图像质量（PSNR>42dB）的同时实现了对畸变、再生成和对抗攻击的全面鲁棒性，超越了现有 SOTA 方法。

## 研究背景与动机

**领域现状**：随着 AI 图像生成和编辑工具的普及，数字内容的版权保护和真实性验证变得愈发紧迫。不可见水印是当前主流的认证机制，通过在图像中嵌入不可见信息来验证真实性。

**现有痛点**：
   - 传统变换域水印（DCT、DWT）容易被常见图像操作（缩放、裁剪、压缩、加噪）破坏
   - 深度学习方法（HiDDeN、StegaStamp、Stable Signature）在端到端嵌入上有进展，但面对对抗攻击、图像再生成（扩散模型重建）时仍然脆弱
   - 生成式水印方法（如与扩散模型结合）计算复杂度高且容易被针对性攻击

**核心矛盾**：不可见性（imperceptibility）与鲁棒性（robustness）之间存在根本性的权衡——嵌入强度越大越鲁棒但越容易被察觉；反之越不可见但越脆弱。

**本文目标**：设计一种同时在不可见性和鲁棒性上超越 SOTA 的水印方法，特别是要抵御三类攻击：畸变（rotation、crop、noise 等）、图像再生成（扩散模型重建）、对抗攻击。

**切入角度**：将水印嵌入到隐藏卷积层的频谱域（而非直接的空间域或简单的频域），通过"小波投影→高频子带→FFT 频谱投影"的层叠变换，让水印信息深度隐藏在高频频谱分量中。

**核心 idea**：在小波分解的高频子带上做 FFT 频谱投影，将水印嵌入到频谱域的高频区域，配合可学习阈值（基于 Parseval 定理）实现高精度比特提取。

## 方法详解

### 整体框架
SpecGuard 包含编码器和解码器两个模块：
- **编码器**：对原始图像做小波分解→取高频子带 $S_{HH}$→对 $S_{HH}$ 做 FFT 频谱投影→在频谱域的指定区域嵌入二进制水印→逆变换重建水印图像
- **解码器**：对水印图像重复同样的小波+频谱投影变换→在相同区域提取信号→用可学习阈值 $\theta$ 解码比特

### 关键设计

1. **小波投影（Wavelet Projection）**:

    - 功能：在嵌入前先将图像分解为多尺度多方向的子带
    - 核心思路：用 2D 离散小波变换将图像分解为 $S_{LL}$（低频近似）、$S_{LH}$、$S_{HL}$、$S_{HH}$（水平/垂直/对角高频细节）。分解层数 $\kappa = \lfloor\sqrt{\log(1+N)}\rfloor$，其中 $N$ 为总像素数
    - 设计动机：高频子带包含边缘和纹理细节，在此嵌入既不影响视觉质量又利于隐藏

2. **FFT 频谱投影近似**:

    - 功能：将高频子带 $S_{HH}$ 从空间域转换到频谱域
    - 核心思路：对 $S_{HH}$ 做对称扩展（$N \times N \to 2N \times 2N$），然后应用 2D FFT，取实部作为频谱投影系数的近似：$\zeta(u,v) \approx \text{Re}(F(u,v))$。对称扩展保证 FFT 结果仅含实值，简化后续处理
    - 设计动机：直接在频谱域嵌入比空间域更稳定，FFT 近似比精确频谱投影计算效率更高

3. **水印嵌入过程**:

    - 功能：将二进制消息嵌入到频谱域的指定高频区域
    - 核心思路：首先在 $S_{HH}$ 上通过 $k$ 层卷积+LeakyReLU 提取特征。然后创建以 $(h/2, w/2)$ 为中心、半径为 $r$ 的径向掩码，仅在掩码内的高频频谱区域嵌入消息：$S_{HH}^{(n+1)}[:,W_c,x_i,y_i] += M_{\text{expanded}}[:,W_c,i] \cdot s$，其中 $s$ 是控制嵌入强度的因子
    - 设计动机：径向掩码将嵌入限制在高频区域，减少感知影响；强度因子 $s$ 平衡不可见性和鲁棒性；不知道 $r$、$s$、$W_c$ 就难以定位水印，增强安全性（黑盒特性）

4. **基于 Parseval 定理的可学习阈值解码**:

    - 功能：用自适应阈值从频谱域提取水印比特
    - 核心思路：解码器对水印图像做相同的小波+频谱投影变换，提取掩码区域的系数，然后用可学习阈值 $\theta$ 判定每个 bit：$D_M[i] = 1 \text{ if Extracted}[i] > \theta$。$\theta$ 通过梯度下降优化：$\theta \leftarrow \theta - \eta \cdot \frac{\partial L_{\text{dec}}}{\partial \theta}$
    - 设计动机：Parseval 定理保证空间域和频谱域的总能量相等，但水印嵌入（强度因子 $s$）会改变局部频谱能量分布——嵌入"1"的位置能量更高。可学习阈值能适应这种能量分布变化，在各种攻击下动态调整判定边界

### 损失函数 / 训练策略
- 编码器损失 $L_{\text{enc}} = \|E_\theta(I, M) - I\|^2$（保证不可见性）
- 解码器损失 $L_{\text{dec}} = \|D_\theta(I_{\text{embedded}}) - M\|^2$（保证提取精度）
- 总损失 $L = \lambda_{\text{enc}} L_{\text{enc}} + \lambda_{\text{dec}} L_{\text{dec}}$，初始 $\lambda_{\text{enc}}=0.7$，$\lambda_{\text{dec}}=1.0$
- Adam 优化器，编码器 lr=$10^{-2}$，解码器 lr=$10^{-3}$（每 100 步减半），训练 300 epochs

## 实验关键数据

### 主实验：无攻击条件下的质量与精度对比

| 方法 | 会议 | BL=64 PSNR/BRA | BL=128 PSNR/BRA | BL=256 PSNR/BRA |
|------|------|----------------|-----------------|-----------------|
| HiDDeN | ECCV'18 | 32.01/0.98 | 31.80/0.85 | 31.50/0.82 |
| StegaStamp | CVPR'20 | 28.50/0.99 | 28.20/0.98 | 28.00/0.94 |
| EditGuard | CVPR'24 | 41.56/0.98 | 41.30/0.97 | 40.90/0.97 |
| MuST | AAAI'24 | 41.20/0.98 | 40.90/0.93 | 40.50/0.90 |
| **SpecGuard** | **ICCV'25** | **42.59/0.99** | **42.89/0.99** | **40.86/0.98** |

SpecGuard 在 128-bit 嵌入时达到 PSNR=42.89dB、SSIM=0.99、BRA=0.99，全面领先。

### 鲁棒性对比（Waves 框架评估）

| 攻击类型 | 指标 | Tree-Ring | Stable Sig | StegaStamp | SpecGuard |
|---------|------|-----------|------------|------------|-----------|
| Rotation | Avg P | 0.375 | 0.594 | 0.357 | **0.687** |
| Crop | Avg P | 0.332 | 0.995 | 0.540 | **0.998** |
| Regen-Diff | Avg P | 0.612 | 0.001 | 0.943 | **0.982** |
| Regen-VAE | Avg P | 0.832 | 0.516 | 1.000 | **0.995** |
| Adversarial | Avg P | 0.448 | - | - | **高** |

SpecGuard 在畸变、再生成和对抗三类攻击下均表现优异。

### 多分辨率质量评估

| 分辨率 | 数据集 | PSNR | SSIM | FID | MSE |
|--------|--------|------|------|-----|-----|
| 256×256 | CelebA-HQ | 40.361 | 0.9889 | 16.451 | 0.0002 |
| 512×512 | MS-COCO | 44.680 | 0.9927 | 17.020 | 0.0001 |
| 1024×1024 | MS-COCO | 48.081 | 0.9936 | 16.955 | 0.0001 |

分辨率越高质量越好（PSNR 最高达 48dB），因为水印能量在更多像素中分散。

### 关键发现
- **高 bit 容量下仍保持高精度**：256-bit 时 BRA 仍达 0.98，而多数方法在 256-bit 时大幅下降
- **再生成攻击下表现突出**：面对扩散模型重建（Regen-Diff），SpecGuard 的 Avg P = 0.982，而 Stable Signature 几乎完全失效（0.001）
- **不可见性极佳**：PSNR > 40dB 在所有测试分辨率上均成立
- **社交媒体平台鲁棒性**：上传到各平台后水印仍可恢复

## 亮点与洞察
- **层叠变换策略巧妙**：小波分解 → 高频子带 → FFT 频谱投影，每一步都将水印推向更难被感知和破坏的域。这种"变换中嵌入变换"的思路让攻击者难以在不知道变换参数的情况下定位水印
- **Parseval 定理的实用化**：将一个数学定理（空间域和频域总能量守恒）转化为可学习阈值的设计依据——嵌入"1"的位置有能量增益，$\theta$ 学会利用这个能量差异。这种理论指导设计的方式值得借鉴
- **黑盒安全性**：水印参数 $(r, s, W_c)$ 构成密钥空间，不知道这些参数就无法定位嵌入区域

## 局限与展望
- **计算复杂度**：小波变换 + FFT + 多层卷积的组合在高分辨率图像上的延迟未详细分析
- **固定嵌入策略**：强度因子 $s$ 和半径 $r$ 是固定超参数，未能根据图像内容自适应调整
- **评估的攻击种类**：虽然覆盖了三大类攻击，但未涉及带有模型知识的白盒攻击
- **可改进方向**：(a) 自适应强度因子——根据局部纹理复杂度动态调整 $s$；(b) 多尺度嵌入——在不同小波分解层同时嵌入以增强容错；(c) 与生成模型结合——在扩散模型的采样过程中直接嵌入

## 相关工作与启发
- **vs StegaStamp**: StegaStamp 端到端学习但 PSNR 仅约 28-29dB，远低于 SpecGuard 的 42+dB。SpecGuard 的频谱域嵌入在不可见性上有绝对优势
- **vs Stable Signature**: Stable Signature 嵌入到生成模型的解码器中，是 pre-processing 方法。对扩散模型再生成攻击几乎完全失效（Avg P = 0.001），而 SpecGuard 作为 post-processing 方法表现更鲁棒
- **vs EditGuard**: EditGuard 在 PSNR 上接近（41.56 vs 42.59），但 SpecGuard 在高 bit 容量（256-bit）时保持更高的 BRA（0.98 vs 0.97）
- **vs Tree-Ring**: Tree-Ring 将水印嵌入初始噪声的频谱中，与 SpecGuard 都使用频域思路，但 Tree-Ring 仅适用于扩散模型生成的图像

## 评分
- 新颖性: ⭐⭐⭐⭐ 小波+频谱投影的层叠变换和 Parseval 定理指导的可学习阈值有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖了畸变/再生成/对抗三大类攻击，多分辨率、多数据集、社交媒体测试
- 写作质量: ⭐⭐⭐⭐ 数学推导详尽但符号较多，整体结构清晰
- 价值: ⭐⭐⭐⭐ 在水印领域实现了全面超越 SOTA 的效果，实用价值高

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [INACTIVE: Invisible Backdoor Attack against Self-supervised Learning](../../CVPR2025/ai_safety/invisible_backdoor_attack_against_self-supervised_learning.md)
- [Accelerating Spectral Clustering under Fairness Constraints](../../ICML2025/ai_safety/accelerating_spectral_clustering_under_fairness_constraints.md)
- [Spectral Perturbation Bounds for Low-Rank Approximation with Applications to Privacy](../../NeurIPS2025/ai_safety/spectral_perturbation_bounds_for_low-rank_approximation_with_applications_to_pri.md)
- [Ghost in the Transformer: Detecting Model Reuse with Invariant Spectral Signatures](../../AAAI2026/ai_safety/ghost_in_the_transformer_detecting_model_reuse_with_invariant_spectral_signature.md)
- [Provable Watermarking for Data Poisoning Attacks](../../NeurIPS2025/ai_safety/provable_watermarking_for_data_poisoning_attacks.md)

<!-- RELATED:END -->
