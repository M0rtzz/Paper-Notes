---
title: >-
  [论文解读] Learning Pixel-adaptive Multi-layer Perceptrons for Real-time Image Enhancement
description: >-
  [ICCV 2025][图像恢复][图像增强] 提出 BPAM 框架，将双边网格的空间建模能力与 MLP 的非线性映射能力相结合，通过为每个像素动态生成独特的微型 MLP 参数实现高质量、实时的图像增强。
tags:
  - ICCV 2025
  - 图像恢复
  - 图像增强
  - 双边网格
  - 像素自适应MLP
  - 实时处理
  - 色彩映射
---

# Learning Pixel-adaptive Multi-layer Perceptrons for Real-time Image Enhancement

**会议**: ICCV 2025  
**arXiv**: [2507.12135](https://arxiv.org/abs/2507.12135)  
**代码**: [GitHub](https://github.com/LabShuHangGU/BPAM)  
**领域**: 图像修复  
**关键词**: 图像增强, 双边网格, 像素自适应MLP, 实时处理, 色彩映射

## 一句话总结

提出 BPAM 框架，将双边网格的空间建模能力与 MLP 的非线性映射能力相结合，通过为每个像素动态生成独特的微型 MLP 参数实现高质量、实时的图像增强。

## 研究背景与动机

图像增强需要在质量和速度之间取得平衡。当前方法分为两大阵营，各有明显局限：

**端到端深度学习方法**：以 Restormer、NAFNet 为代表，增强质量高但计算开销大，难以实时处理高分辨率图像

**混合方法**：将深度学习与高效物理模型结合
   - **3D LUT 方法**：通过查找表实现快速色彩映射，速度极快（>1000 FPS），但问题在于 3D LUT 是纯图像级算子——仅基于 RGB 值进行颜色变换，完全忽略空间上下文。虽然 SA-3DLUT 等方法尝试引入空间感知，但色彩映射阶段仍然是 RGB 独占的
   - **双边网格方法**：以 HDRNet 为代表，天然编码了空间和强度域信息，可以通过切片操作高效地从低分辨率表示映射到全分辨率。但存在两个瓶颈：(a) 仅支持仿射变换，无法建模复杂非线性色彩关系；(b) 继承了灰度图像设计，将 RGB 三通道融合为单通道引导图来提取系数，导致色彩信息利用不充分

**全局 MLP 方法**：如 CSRNet 用 MLP 替代传统色彩变换，但参数在整個图像上共享，面对局部变化的色彩/光照时力不从心，需要很深的隐层才能获得跨区域的泛化能力

本文的核心洞察是：如果每个像素都能拥有自己独特的 MLP 参数，就可以用极小的 MLP（仅 3-8-3）实现高质量的空间自适应色彩变换。双边网格正好提供了高效生成空间变化参数的机制。

## 方法详解

### 整体框架

BPAM 的流程：(1) 使用三层 U-Net 风格的 NAFNet 骨干从下采样图像中提取特征；(2) 通过 Pixel Unshuffle 进一步减小网格尺寸并增加通道数；(3) 用 $1\times1$ 卷积生成两个双边网格（分别存储 MLP 第一层和第二层的参数）；(4) 生成多通道引导图，通过两次切片操作提取每个像素对应的完整 MLP 参数；(5) 将输入像素通过各自的微型 MLP 进行色彩变换。

### 关键设计

1. **像素自适应 MLP 学习（Pixel-Adaptive MLP）**:

    - 功能：为图像中每个像素生成一个独特的三层 MLP（3-8-3），将颜色变换从全局共享参数变为像素级自适应
    - 核心思路：主干网络生成两个双边网格，第一个网格每个格点包含 32 个参数（$\mathbf{W}_1 \in \mathbb{R}^{8\times3}$ 的 24 个权重 + $\mathbf{b}_1 \in \mathbb{R}^8$ 的 8 个偏置），第二个网格每个格点包含 27 个参数（$\mathbf{W}_2 \in \mathbb{R}^{3\times8}$ 的 24 个权重 + $\mathbf{b}_2 \in \mathbb{R}^3$ 的 3 个偏置）。色彩变换分两阶段：
        - 隐层：$\mathbf{z}(x,y) = \sigma(\mathbf{W}_1(x,y,s) \cdot \mathbf{I}(x,y) + \mathbf{b}_1(x,y,s))$
        - 输出：$O(x,y) = \mathbf{W}_2(x,y,s) \cdot \mathbf{z}(x,y) + \mathbf{b}_2(x,y,s)$
    - 设计动机：MLP 具有强大的非线性建模能力，但全局共享限制了其处理局部变化的能力。像素自适应使每个像素获得专属的非线性变换函数，仅需极小的 MLP 即可达到高性能

2. **网格分解策略（Grid Decomposition）**:

    - 功能：将双边网格拆分为多个子网格，每个子网格搭配独立的引导通道
    - 核心思路：认识到 MLP 参数天然可以按类别分组。对于第一个网格，将 32 个参数分为 4 个子网格（3 个颜色通道各自的权重 + 共享偏置）。对于第二个网格，将 27 个参数分为 9 个子网格（8 个隐层通道各自的权重 + 共享偏置）。相应生成 4 通道和 9 通道的引导图，每个通道从对应子网格中提取参数
    - 设计动机：传统双边网格将 RGB 三通道融合为单通道引导图来提取所有系数，这导致色彩信息在切片阶段丢失。分解策略让每个颜色通道拥有独立的引导和提取路径，实现了对色彩信息的充分利用，类似于 3D LUT 能利用全部颜色信息进行映射的优势

3. **双阶段引导图生成**:

    - 功能：顺序生成两组引导图，用于两次切片操作
    - 核心思路：第一个卷积网络以原始图像为输入生成第一组引导图用于提取隐层参数 → 计算隐层向量 → 第二个卷积网络以隐层向量为输入生成第二组引导图用于提取输出层参数
    - 设计动机：MLP 参数分为两部分且有顺序依赖，顺序生成引导图可以让第二阶段利用中间结果的信息，生成更精确的参数

### 损失函数 / 训练策略

- **总损失**：$\mathcal{L} = \mathcal{L}_2 + 0.5 \times \mathcal{L}_{ssim} + 0.005 \times \mathcal{L}_{per}$
    - $\mathcal{L}_2$：MSE 像素损失
    - $\mathcal{L}_{ssim}$：SSIM 结构损失
    - $\mathcal{L}_{per}$：基于预训练 VGG19 的感知损失
- **优化器**：Adam，余弦退火学习率 + 额外 0.1 衰减因子
- **网格尺寸自适应**：PPR10K 用 1/4 分辨率，FiveK 全分辨率用 1/32，其他用 1/8，深度固定为 8
- **高效实现**：切片操作和 MLP 参数应用通过 CUDA 扩展加速

## 实验关键数据

### 主实验

**FiveK 数据集 Tone Mapping（全分辨率）**

| 方法 | 参数量 | PSNR↑ | SSIM↑ | ΔE↓ |
|------|--------|-------|-------|-----|
| HDRNet | 482K | 24.17 | 0.919 | 8.91 |
| CSRNet | 37K | 24.23 | 0.920 | 8.75 |
| 3DLUT | 592K | 24.39 | 0.923 | 8.33 |
| LutBGrid | 464K | 24.57 | 0.931 | 8.03 |
| **BPAM（Ours）** | 624K | **25.12** | **0.934** | **7.73** |

**LCDP 数据集 曝光校正**

| 方法 | 参数量 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|--------|-------|-------|--------|
| LCDPNet | 960K | 23.24 | 0.842 | 0.137 |
| CoTF | 310K | 23.89 | 0.858 | 0.104 |
| LutBGrid | 464K | 22.71 | 0.803 | 0.154 |
| **BPAM（Ours）** | 624K | **24.22** | **0.872** | **0.097** |

### 消融实验

**组件贡献分析（FiveK 480p Tone Mapping）**

| 配置 | 仿射变换 | MLP | 网格分解 | PSNR | SSIM |
|------|---------|-----|---------|------|------|
| Setting 1 | ✓ | | | 25.53 | 0.935 |
| Setting 2 | | ✓ | | 25.70 | 0.939 |
| Setting 3 | ✓ | | ✓ | 25.63 | 0.937 |
| **Setting 4** | | ✓ | ✓ | **25.83** | **0.941** |

**推理速度对比**

| 方法 | 1080p 时间(ms) | 4K FPS |
|------|---------------|--------|
| 3DLUT | 0.89 | 862 |
| LutBGrid | 1.66 | 319 |
| HDRNet | 11.8 | 22.8 |
| **BPAM** | 10.2 | **36.0** |
| CSRNet | 16.2 | 15.3 |

### 关键发现

1. 从仿射变换切换到 MLP 带来 0.17 dB 提升，网格分解策略额外带来 0.13 dB 提升，两者互补
2. 在全分辨率 FiveK 上，BPAM 比 LutBGrid 高出 0.55 dB，优势在高分辨率下更加明显
3. BPAM 的 4K 处理速度超过 30 FPS，满足实时需求（3DLUT 更快但缺乏空间感知）
4. 在曝光校正任务上优势最为突出（+0.33 dB over CoTF），因为极端光照更需要空间自适应的非线性变换
5. 像素自适应的 3-8-3 微型 MLP 即可胜任，无需深层隐层或大通道数

## 亮点与洞察

- **参数维度的巧妙创新**：不是让 MLP 更大更深，而是让每个像素拥有自己的微型 MLP——以空间维度换取深度维度，思路新颖
- **双边网格的新用法**：从存储仿射系数扩展到存储 MLP 参数，打破了双边网格仅支持线性变换的瓶颈
- **网格分解弥合了两大阵营**：双边网格获得了类似 3D LUT 的多通道色彩利用能力，同时保持了空间感知的优势
- **实用性强**：仅 624K 参数，4K 30+ FPS，PSNR 领先，是少见的质量和速度兼顾的方案
- **CUDA 加速的工程意识**：切片操作和 MLP 应用通过自定义 CUDA 核实现，确保理论优势转化为实际速度

## 局限与展望

1. 相比 3D LUT 方法（>800 FPS），BPAM 的速度仍有差距，这是空间感知方法的通病
2. 3D 双边网格随分辨率变化需要手动设定网格尺寸比例，不够自动化
3. MLP 结构固定为 3-8-3，未探索自适应确定最优结构的可能
4. 仅在增强/色调映射/曝光校正任务上验证，是否适用于去噪、去模糊等其他修复任务未知
5. 感知损失的权重（0.005）感觉较具经验性，缺乏系统的超参数敏感性分析

## 相关工作与启发

- **HDRNet**：双边网格用于图像增强的开创性工作，但限于仿射变换
- **CSRNet**：用全局 MLP 做色彩变换的先驱，但参数全局共享
- **3D LUT 系列**（SA-3DLUT、LutBGrid）：高效色彩映射的主流方案，但空间感知受限
- 启发：好的表示结构（双边网格）+ 好的函数形式（MLP）的结合，可以以轻量级的方式超越各自的局限

## 评分

- 新颖性: ⭐⭐⭐⭐ 双边网格存储 MLP 参数的想法新颖，网格分解策略巧妙
- 实验充分度: ⭐⭐⭐⭐ 三个数据集三种任务，消融清晰，速度对比全面
- 写作质量: ⭐⭐⭐⭐ 方法动机清晰，公式推导完整，图示直观
- 价值: ⭐⭐⭐⭐ 在实时图像增强领域提供了有竞争力的新方案，工业应用前景好

<!-- RELATED:START -->

## 相关论文

- [Lightweight and Fast Real-time Image Enhancement via Decomposition of the Spatial-aware Lookup Tables](lightweight_and_fast_real-time_image_enhancement_via_decomposition_of_the_spatia.md)
- [MobileIE: An Extremely Lightweight and Effective ConvNet for Real-Time Image Enhancement on Mobile Devices](mobileie_an_extremely_lightweight_and_effective_convnet_for_real-time_image_enha.md)
- [Enhancing Image Restoration Transformer via Adaptive Translation Equivariance](enhancing_image_restoration_transformer_via_adaptive_translation_equivariance.md)
- [Metric Convolutions: A Unifying Theory to Adaptive Image Convolutions](metric_convolutions_a_unifying_theory_to_adaptive_image_convolutions.md)
- [CWNet: Causal Wavelet Network for Low-Light Image Enhancement](cwnet_causal_wavelet_network_for_low-light_image_enhancement.md)

<!-- RELATED:END -->
