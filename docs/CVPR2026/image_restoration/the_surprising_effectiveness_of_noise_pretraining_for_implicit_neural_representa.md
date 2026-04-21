---
title: >-
  [论文解读] The Surprising Effectiveness of Noise Pretraining for Implicit Neural Representations
description: >-
  [CVPR 2026][图像恢复][隐式神经表示] 本文通过系统的实验分析发现：用非结构化噪声（均匀/高斯分布）预训练 INR 可在图像拟合中达到惊人的 ~80dB PSNR，远超所有数据驱动初始化方法；而具有自然图像 $1/|f^\alpha|$ 频谱结构的噪声则在信号拟合和去噪之间实现最佳平衡，无需任何真实数据即可匹配 SOTA 数据驱动初始化性能。
tags:
  - CVPR 2026
  - 图像恢复
  - 隐式神经表示
  - 噪声预训练
  - 参数初始化
  - 信号拟合
  - 去噪
---

# The Surprising Effectiveness of Noise Pretraining for Implicit Neural Representations

**会议**: CVPR 2026  
**arXiv**: [2603.29034](https://arxiv.org/abs/2603.29034)  
**代码**: 有（项目页面已公开）  
**领域**: 图像恢复 / 隐式神经表示  
**关键词**: 隐式神经表示、噪声预训练、参数初始化、信号拟合、去噪

## 一句话总结

本文通过系统的实验分析发现：用非结构化噪声（均匀/高斯分布）预训练 INR 可在图像拟合中达到惊人的 ~80dB PSNR，远超所有数据驱动初始化方法；而具有自然图像 $1/|f^\alpha|$ 频谱结构的噪声则在信号拟合和去噪之间实现最佳平衡，无需任何真实数据即可匹配 SOTA 数据驱动初始化性能。

## 研究背景与动机

**领域现状**：隐式神经表示（INR）用 MLP 将空间坐标映射为信号值，在压缩、逆成像、神经渲染等领域广泛应用。INR 的收敛性能高度依赖参数初始化策略——数据驱动方法（如 meta-learning、Strainer）已被证明远优于标准随机初始化。

**现有痛点**：尽管数据驱动初始化效果显著，其成功的根本原因仍然不清楚——是编码了经典的统计信号先验？还是更复杂的数据特定特征？这种不清楚限制了在缺乏领域数据的科学成像等场景中的应用。

**核心矛盾**：数据驱动初始化需要先验真实信号，但许多应用领域（如科学成像）缺乏充足的领域数据。如果能理解初始化成功的根本机制，可能找到不依赖真实数据的高效替代方案。

**本文目标** (1) 数据驱动 INR 初始化的性能增益来源于什么层面的信号特性？(2) 是否可以用不含真实数据的噪声来替代数据驱动初始化？(3) 不同类型噪声的预训练对信号拟合和逆问题求解有何不同影响？

**切入角度**：受噪声预训练视觉分类网络的工作启发，用不同类别噪声预训练 INR，通过对比实验揭示初始化成功的底层机制——每种噪声类别有精确定义的性质，可作为控制变量分析。

**核心 idea**：用噪声代替真实数据预训练 INR，发现非结构化噪声是信号拟合之王而频谱噪声是全能选手，无需真实数据即可高效初始化 INR。

## 方法详解

### 整体框架

本文采用 Strainer 的 INR 框架：6层 MLP（sine 激活函数, $\omega=30$），前5层为共享编码器、最后一层为信号特定解码器。预训练阶段在 $N=10$ 个噪声样本上联合训练 5000 步。测试时保留编码器权重、随机初始化单个解码器头，在新信号上训练 2000 步。将此噪声预训练变体命名为 Snp (Strainer Noise Pretraining)。

### 关键设计

1. **多类噪声预训练 (Snp)**:

    - 功能：用不同统计特性的噪声替代真实数据进行 INR 参数初始化
    - 核心思路：考察两大类噪声——非结构化（均匀 $\mathcal{U}(0,1)$、高斯 $\mathcal{N}(0.5, 0.2)$）和结构化（Dead Leaves 系列：方块/有向/混合/纹理，以及统计模型系列：频谱/$1/|f^\alpha|$、频谱+颜色、小波边缘模型）。结构化噪声通过在随机噪声上施加自然图像统计特性生成。每类噪声只需 10 个样本即可完成预训练
    - 设计动机：每种噪声类别有精确定义的统计/频谱特性，作为控制变量可分离不同层面信号特征对初始化的贡献

2. **信号拟合与逆问题的权衡发现**:

    - 功能：揭示 INR 初始化中信号拟合能力与深度先验强度之间的根本性权衡
    - 核心思路：非结构化噪声（Uniform/Gaussian）预训练后，信号拟合达到惊人的 ~80dB PSNR，但去噪能力最差（23-24dB）。结构化频谱噪声（$1/|f^\alpha|$）在两个任务间实现最佳平衡——拟合接近 Strainer（56.4 vs 57.8 dB），去噪接近 Siren（27.6 vs 28.3 dB）。这揭示了一个根本性权衡：功能容量和深度先验强度不可兼得
    - 设计动机：仅评估信号拟合无法反映 INR 作为逆问题求解器的实际能力，需要同时考量两个维度

3. **NTK 与局部复杂度的机理解释**:

    - 功能：从理论角度解释为什么不同噪声预训练会导致截然不同的行为
    - 核心思路：NTK 分析显示 Snp:Uniform 用最少特征值就能捕获目标信号能量（解释其快速收敛）。局部复杂度分析揭示 Snp:Uniform 的初始层产生高度非线性的伪随机输入空间划分，类似于 Instant-NGP 的哈希编码——高容量但缺乏结构先验。Snp:Spectrum 和 Strainer 则展示出相似的损失地形，解释了二者相近的性能表现
    - 设计动机：不仅要发现现象，更要理解机制，为未来的方法设计提供理论指导

### 损失函数 / 训练策略

预训练和拟合均使用 L2 损失函数，Adam 优化器 (lr=1e-4)。预训练 5000 步，测试时拟合 2000 步。去噪任务通过早停（early stopping）确定最佳迭代次数。视频拟合使用 ResFields 框架，添加逐帧可学习残差更新，训练 100k 步。

## 实验关键数据

### 主实验

图像拟合 PSNR (dB), T=2000 步：

| 方法 | CelebA-HQ | AFHQ | OASIS-MRI |
|------|----------|------|-----------|
| Siren | 44.9 | 45.1 | 53.0 |
| Strainer | 57.8 | 58.0 | 62.8 |
| TransINR | 51.9 | 49.0 | 55.5 |
| IPC | 49.7 | 47.2 | 51.4 |
| **Snp: Uniform** | **85.7** | **79.9** | **79.3** |
| Snp: Gaussian | 80.0 | 77.0 | 79.1 |
| Snp: Spectrum | 56.4 | 56.2 | 60.0 |

### 消融实验

去噪 PSNR (dB) 与最佳迭代步数 (CelebA-HQ)：

| 方法 | PSNR | 最佳步数 |
|------|------|---------|
| Siren | **28.3** | 139 |
| Strainer | 27.3 | 70 |
| Snp: Spectrum | 27.6 | 78 |
| Snp: Uniform | 23.0 | 73 |
| Snp: Gaussian | 23.8 | 70 |

视频拟合与去噪 (Pexels, 100k 步)：

| 任务 | 方法 | 平均 PSNR |
|------|------|----------|
| 拟合 | Vanilla ResFields | 29.5 |
| 拟合 | **ResFields + Snp:Spectrum** | **31.1** |
| 去噪 | Vanilla ResFields | 27.0 |
| 去噪 | **ResFields + Snp:Spectrum** | **28.0** |

### 关键发现

- **非结构化噪声的惊人拟合能力**：Snp:Uniform 达到 85.7dB，比最强数据驱动方法 Strainer（57.8dB）高出近 28dB，这是一个极其出人意料的发现
- **信号拟合 vs 去噪的权衡**：非结构化噪声拟合能力最强但去噪最差，频谱噪声在二者间最佳平衡
- **频谱噪声≈真实数据**：$1/|f^\alpha|$ 噪声预训练性能几乎匹配用真实人脸数据预训练的 Strainer
- **与哈希编码的类比**：Snp:Uniform 的功能几何类似于 Instant-NGP 的哈希编码，高容量来自伪随机的输入空间划分
- **视频领域趋势不同**：频谱噪声在视频拟合中略优于均匀噪声，可能因为视频的时空连续性更受益于结构化先验

## 亮点与洞察

- **发现性贡献极强**：通过精心设计的控制实验揭示了 INR 初始化的核心机制，这是少有的"解释为什么"而非"做什么"的工作
- **实用价值明确**：$1/|f^\alpha|$ 噪声易于生成，可完全替代真实数据进行 INR 初始化——对数据缺乏的科学成像领域尤其有价值
- **权衡洞察深刻**：信号拟合容量和深度先验强度的不可兼得性是对 INR 社区的重要警示
- **跨域泛化**：发现从图像到视频、从人脸到 MRI 均成立，说明是 INR 的普遍性质而非数据特定现象
- **NTK/损失地形分析**：提供了理论层面的解释框架，不停留在经验观察

## 局限与展望

- 仅考察了正弦激活函数和固定层数，不同激活函数（ReLU、Gaussian、Wavelet）的结论可能不同
- 未探索不同网络深度与噪声类型的交互效应
- 去噪实验较简单（仅高斯噪声），更复杂的逆问题（如超分辨率、CT重建）值得验证
- 可探索自适应地混合不同噪声类型的预训练以同时优化拟合和先验

## 相关工作与启发

- Strainer 证明了跨域 INR 初始化的有效性（人脸→动物→MRI），本文进一步证明甚至不需要真实数据
- Looking at Noise 数据集提供了结构化噪声的系统分类，是本文控制实验的基础
- Instant-NGP 的哈希编码与 Snp:Uniform 的功能几何的类比为理解两种方法提供了统一视角

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 用噪声预训练这个反直觉的发现极具启发性，打破了 INR 社区的常规认知
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖多种噪声类型、多个数据集、多个任务、多种可解释性分析，实验设计极为系统
- **写作质量**: ⭐⭐⭐⭐⭐ — 叙事流畅、图表直观、从实验到机理解释的逻辑链完整
- **价值**: ⭐⭐⭐⭐ — 对 INR 社区有深远的理论指导意义，实用价值在数据缺乏的科学成像场景尤为突出

<!-- RELATED:START -->

## 相关论文

- [PNG: Diffusion-Based sRGB Real Noise Generation via Prompt-Driven Noise Representation Learning](diffusion-based_srgb_real_noise_generation_via_prompt-driven_noise_representatio.md)
- [Learning to Translate Noise for Robust Image Denoising](learning_to_translate_noise_for_robust_image_denoising.md)
- [Implicit Augmentation from Distributional Symmetry in Turbulence Super-Resolution](../../NeurIPS2025/image_restoration/implicit_augmentation_from_distributional_symmetry_in_turbulence_super-resolutio.md)
- [NEC-Diff: Noise-Robust Event–RAW Complementary Diffusion for Seeing Motion in Extreme Darkness](nec-diff_noise-robust_event-raw_complementary_diffusion_for_seeing_motion_in_ext.md)
- [Gyro-based Neural Single Image Deblurring](../../CVPR2025/image_restoration/gyro-based_neural_single_image_deblurring.md)

<!-- RELATED:END -->
