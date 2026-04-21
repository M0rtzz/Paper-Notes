---
title: >-
  [论文解读] FIPER: Factorized Features for Robust Image Super-Resolution and Compression
description: >-
  [NeurIPS 2025][图像恢复][超分辨率] 提出 Factorized Features 统一表示——将图像分解为可学习的非均匀基与空间变化系数，配合锯齿坐标变换和多频调制，在 4× 超分辨率上 PSNR 相对提升 204.4%（HAT-L-F vs SwinIR），在图像压缩上 BD-rate 相比 VTM 降低 21.09%。
tags:
  - NeurIPS 2025
  - 图像恢复
  - 超分辨率
  - 图像压缩
  - 因子化特征
  - 基-系数分解
  - 坐标变换
  - 多频调制
  - 多帧压缩
---

# FIPER: Factorized Features for Robust Image Super-Resolution and Compression

**会议**: NeurIPS 2025  
**arXiv**: [2410.18083](https://arxiv.org/abs/2410.18083)  
**代码**: [项目主页](https://jayisaking.github.io/FIPER/)  
**领域**: 图像恢复 / 超分辨率与压缩  
**关键词**: 超分辨率, 图像压缩, 因子化特征, 基-系数分解, 坐标变换, 多频调制, 多帧压缩

## 一句话总结

提出 Factorized Features 统一表示——将图像分解为可学习的非均匀基与空间变化系数，配合锯齿坐标变换和多频调制，在 4× 超分辨率上 PSNR 相对提升 204.4%（HAT-L-F vs SwinIR），在图像压缩上 BD-rate 相比 VTM 降低 21.09%。

## 研究背景与动机

**超分辨率发展脉络**：从 CNN（EDSR、RCAN）→ GAN（SRGAN）→ Transformer（SwinIR、HAT、DAT）一路发展，现有方法主要关注网络架构设计（窗口注意力、稀疏注意力、密集跳连等），而忽略了图像内容表示本身的优化。

**频率分解方法的局限**：已有分解方法（如 Learnable Fourier Series、Factor Fields）通过固定的正弦/余弦函数建模频率成分，但存在两个核心问题：(a) 固定谐波函数的均匀分布限制了对非均匀视觉模式的适应性；(b) 低频成分能量更大，导致训练中模型偏向低频而牺牲高频细节，高低频收敛严重不同步。

**超分辨率与压缩的共性**：两个任务本质上都需要从低质量输入中恢复精细细节并保持感知保真度——超分辨率是增强分辨率，压缩是从有损编码中重建。这一共性为统一表示提供了动机。

**多帧压缩的冗余**：传统方法对每帧独立编码，忽略了不同图像之间共享的频率成分，编码效率低。

**核心问题**：能否设计一种统一的图像表示，超越简单的特征图输出，既能在超分辨率中精确重建细节，又能在压缩中高效编码？

## 方法详解

### 整体框架

FIPER 的核心是 Factorized Features 表示（公式 7），将图像重建为基函数 $b_i$ 和系数 $c_{ij}$ 的因子化乘积，并通过坐标变换 $\gamma$ 和多频调制 $(\alpha_j, \psi)$ 增强表达能力：

$$\hat{I}(x) = \mathcal{P}\left(\text{Concat}_{i=1}^{N} {}_{j=1}^{K} \left\{ c_{ij}(x) \odot \psi(\alpha_j \cdot b_i(\gamma_i(x))) \right\}\right)$$

框架包含三个核心模块：
- **Coefficient Backbone** $F_{\text{coeff}}$：从低分辨率输入提取特征 $X_{\text{coeff}}$，生成空间变化的系数图
- **Basis Swin Transformer** $F_{\text{basis}}$：从 $X_{\text{coeff}}$ 生成多尺度学习基 $\{b_1, \ldots, b_N\}$
- **Factorized Features Reconstruction**：将基和系数按公式 7 组合并投影为 RGB 输出

### 关键设计

**1. 空间变化系数（Spatially Variant Coefficient）**

图像不同区域的频率特性差异显著（平滑区域不需要高频成分），因此系数 $c_i(x)$ 设计为空间变化的：

$$f(x) = \sum_{i=1}^{N} c_i(x) \cdot b_i(x)$$

由 Coefficient Backbone 通过卷积 + PixelShuffle 上采样生成，实现了对不同空间位置的自适应加权。

**2. 学习非均匀基（Learned Non-uniform Basis）**

传统方法通过学习频率参数 $u, v$ 和相位 $\xi$ 来构造基函数，但固定的正弦函数形式限制了局部灵活性，且高频成分收敛明显滞后于低频。FIPER 直接生成完整的基图 $b_i = M_i \in \mathbb{R}^{T \times T}$，由 Basis Swin Transformer 生成。实验证明（图 2），学习的非均匀基收敛更同步、更稳定。

**3. 锯齿坐标变换（Sawtooth Coordinate Transformation）**

引入坐标变换函数 $\gamma(x) = x \bmod k$，在采样基函数之前对坐标进行周期性映射。锯齿变换将连续空间折叠为分段线性的三角波，显式强制模型学习 patch 级别的周期性重复模式（如建筑物的窗户网格、织物的纹理周期）。可视化（图 3b）显示，相比普通基-系数场（图 3a），坐标变换使特征明显呈现分块周期结构。

**4. 多频调制（Multi-frequency Modulation）**

神经网络存在低频偏置（spectral bias），倾向于先学习低频内容。多频调制通过不同的频率标量 $\alpha_j \in \{1, 4, 16, 64\}$ 和周期函数 $\psi \in \{\sin, \cos\}$ 强制每个基函数同时贡献高/低频成分：

较大的 $\alpha_j$ 使 $\psi$ 振荡更剧烈，迫使基函数在高频域也保持准确性——如果某个基只在低频准确，其高频输出将包含不期望的噪声，形成隐式的正则化。

**5. 系数优先的流水线顺序**

与传统 Fourier 分析"先基后系数"不同，FIPER 先通过 Coefficient Backbone 提取特征，再从特征生成基。消融实验 HAT-F-Basis-First 表明反转顺序导致巨大性能下降，因为 Coefficient Backbone 实际充当特征提取模块，精炼的特征有助于下游基的生成。

**6. 可合并基的多帧压缩**

不同图像经常共享相似的频率成分。FIPER 利用 Transformer $F_{\text{merge}}$ 将 $M$ 帧各自的基 $b_i^n$ 融合为单一共享基：

$$b_i(h,w) = F_{\text{merge}}(\{b_i^n(h,w) \mid n \in 1, \ldots, M\})$$

共享基单独传输，每帧仅编码各自的系数，利用帧间互信息减少编码冗余。随着 $M$ 增大，共享基愈加去噪和泛化。

### 损失函数 / 训练策略

- **超分辨率**：使用 $L_1$ 损失优化。ImageNet 预训练 300k iterations（lr=2e-4, batch=32），DF2K 微调 200k iterations（lr=1e-5）
- **图像压缩**：率失真损失 $L = R(\hat{y}) + R(\hat{z}) + \lambda \cdot D(x, \hat{x})$，其中 $R$ 为比特率代价，$D$ 为失真项，$\lambda$ 控制权衡。SR 先验先冻结训练压缩模块，再端到端联合微调
- 基-系数对数量 $N=6$，频率标量 $\alpha_j \in \{1, 4, 16, 64\}$

## 实验关键数据

### 主实验：4× 超分辨率（PSNR/SSIM）

| 方法 | Set5 | Set14 | BSD100 | Urban100 | Manga109 |
|------|------|-------|--------|----------|----------|
| SwinIR | 32.93/0.9043 | 29.15/0.7958 | 27.95/0.7484 | 27.56/0.8273 | 32.22/0.9273 |
| HAT† | 33.18/0.9073 | 29.38/0.8001 | 28.05/0.7534 | 28.37/0.8447 | 32.87/0.9319 |
| DAT-L† | 33.33/0.9084 | 29.40/0.8009 | 28.04/0.7543 | 28.49/0.8473 | 33.02/0.9321 |
| **DAT-F† (Ours)** | **33.45**/0.9094 | **29.60**/0.8039 | **28.13**/0.7560 | **28.75**/0.8520 | **33.23**/0.9339 |
| HAT-L† | 33.30/0.9083 | 29.47/0.8015 | 28.09/0.7551 | 28.60/0.8498 | 33.09/0.9335 |
| **HAT-F† (Ours)** | **33.53**/0.9100 | **29.65**/0.8050 | **28.18**/0.7569 | **28.79**/0.8527 | **33.33**/0.9342 |
| **HAT-L-F† (Ours)** | **33.75**/0.9116 | **29.87**/0.8091 | **28.31**/0.7597 | **29.51**/0.8637 | **33.36**/0.9343 |

### 图像压缩 BD-Rate（anchor = VTM）

| 方法 | BD-Rate (%) ↓ | 参数量 (M) |
|------|---------------|-----------|
| ELIC (CVPR'22) | -7.24 | 36.9 |
| TCM (CVPR'23) | -11.74 | 76.7 |
| LALIC (CVPR'25) | -15.26 | 63.2 |
| **TCM-HAT-L-F (Ours)** | **-21.09** | 110.3 |
| TCM-HAT-F-multi M=8 | -16.61 | 131.4 |
| TCM-HAT-F-multi M=24 | -20.97 | 131.4 |

### 消融实验

| 变体 | Urban100 PSNR | 说明 |
|------|---------------|------|
| HAT-L† (baseline) | 28.60 | 原始大模型 |
| HAT-F-Basis-First† | 28.57 | 反转系数/基顺序，性能大幅下降 |
| HAT-F-Concat† | 28.73 | 直接拼接基和系数，不用公式 7 |
| **HAT-F†** | **28.79** | 完整 Factorized Features |
| **HAT-L-F†** | **29.51** | 大模型 + Factorized Features |

### 关键发现

- **204.4% 相对 PSNR 提升**：以 SwinIR 为底线、HAT-L 为参照，HAT-L-F 在 Urban100 上提升从 27.56 → 29.51，远超参考模型的 28.60
- **每个组件都关键**：去掉坐标变换（图 3a vs 3b）、多频调制（图 3c vs 3d）、系数优先顺序（HAT-F vs HAT-F-Basis-First）均导致明显性能下降
- **多帧共享基的扩展性**：随着合并帧数 $M$ 从 1 增至 24，BD-rate 从 +27.96% 降至 -20.97%，共享基愈加泛化
- **参数效率**：DAT-F（40M）超越 DAT-L（43M），HAT-F（46M）超越 HAT-L（41M），Factorized Features 比简单增大模型更高效

## 亮点与洞察

- **表示 > 架构**：FIPER 的核心贡献不在于新的网络模块，而在于重新定义了图像的输出表示——从简单特征图到因子化的基-系数分解。这提供了一种正交于网络设计的改进路径
- **锯齿坐标变换的简洁性**：$\gamma(x) = x \bmod k$ 仅需 $O(1)$ 计算即实现了 patch 级周期约束，优雅地引导模型学习重复结构模式
- **超分辨率先验迁移到压缩**：训练好的 SR 模型直接作为压缩解码器的 synthesis transform，充分利用了两个任务的共性
- **多帧压缩的可扩展性**：共享基的数量增加带来持续的 BD-rate 改善，且解码时间随 $M$ 增大反而减少（共享基的去噪效果）

## 局限性

- **计算开销**：Basis Swin Transformer 引入额外参数和推理时间（TCM-HAT-L-F 解码 0.264s vs TCM 的 0.15s），在实时解码场景受限
- **语义信息缺失**：当前方法纯粹在像素/频率层面操作，未利用语义先验，在复杂场景下可能受限
- **多帧扩展的边际收益递减**：$M$ 从 16 增至 24 时 BD-rate 仅从 -19.88% 降至 -20.97%，共享基的容量存在上限
- **评估范围有限**：仅在标准学术基准（Set5/14, Urban100, Kodak 等）上评估，未验证在多样化真实场景下的鲁棒性

## 相关工作与启发

- **与 SwinIR/HAT/DAT 的关系**：这些方法作为 Coefficient Backbone 被即插即用地增强，说明 Factorized Features 具有良好的通用性
- **与 Factor Fields/TensoRF 的联系**：FIPER 将 3D 场景建模中的张量分解思想迁移到 2D 图像低级视觉任务
- **与 Fourier Feature Mapping 的区别**：后者使用固定的 Fourier 基映射坐标，FIPER 学习非均匀基并配合坐标变换
- **启发**：这种"表示优先"的思路可推广到去噪、去模糊、修复等其他低级视觉任务；多频调制机制可能对视频超分辨率中的时间频率建模有参考价值

## 评分

- 新颖性: ⭐⭐⭐⭐ 学习非均匀基+坐标变换+多频调制的组合新颖，"表示优先于架构"的角度独特
- 实验充分度: ⭐⭐⭐⭐⭐ 超分辨率 5 个基准 + 压缩 3 个基准 + 多帧压缩 + 充分的消融实验
- 写作质量: ⭐⭐⭐⭐ 从 Fourier 变换到 Factorized Features 的推导清晰有说服力
- 价值: ⭐⭐⭐⭐ 即插即用地提升多个现有 SR 模型，压缩上大幅超越 VTM，实用性强

<!-- RELATED:START -->

## 相关论文

- [AdcSR: Adversarial Diffusion Compression for Real-World Image Super-Resolution](../../CVPR2025/image_restoration/adversarial_diffusion_compression_for_real-world_image_super-resolution.md)
- [Audio Super-Resolution with Latent Bridge Models](audio_super-resolution_with_latent_bridge_models.md)
- [PIDSR: Complementary Polarized Image Demosaicing and Super-Resolution](../../CVPR2025/image_restoration/pidsr_complementary_polarized_image_demosaicing_and_super-resolution.md)
- [Progressive Focused Transformer for Single Image Super-Resolution](../../CVPR2025/image_restoration/progressive_focused_transformer_for_single_image_super-resolution.md)
- [Augmenting Perceptual Super-Resolution via Image Quality Predictors](../../CVPR2025/image_restoration/augmenting_perceptual_super-resolution_via_image_quality_predictors.md)

<!-- RELATED:END -->
