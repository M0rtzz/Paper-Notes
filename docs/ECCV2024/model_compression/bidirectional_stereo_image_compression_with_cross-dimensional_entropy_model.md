---
title: >-
  [论文解读] Bidirectional Stereo Image Compression with Cross-Dimensional Entropy Model
description: >-
  [ECCV2024][模型压缩][stereo image compression] 提出双向对称的立体图像压缩框架 BiSIC，采用 3D 卷积联合编解码器和跨维度熵模型，在 PSNR 和 MS-SSIM 上均超越传统标准和已有学习方法，同时消除了单向方法中左右视图压缩质量不平衡的问题。
tags:
  - ECCV2024
  - 模型压缩
  - stereo image compression
  - bidirectional coding
  - 3D convolution
  - cross-dimensional entropy model
  - learned image compression
---

# Bidirectional Stereo Image Compression with Cross-Dimensional Entropy Model

**会议**: ECCV2024  
**arXiv**: [2407.10632](https://arxiv.org/abs/2407.10632)  
**代码**: [GitHub](https://github.com/LIUZhening111/BiSIC)  
**领域**: model_compression  
**关键词**: stereo image compression, bidirectional coding, 3D convolution, cross-dimensional entropy model, learned image compression

## 一句话总结

提出双向对称的立体图像压缩框架 BiSIC，采用 3D 卷积联合编解码器和跨维度熵模型，在 PSNR 和 MS-SSIM 上均超越传统标准和已有学习方法，同时消除了单向方法中左右视图压缩质量不平衡的问题。

## 背景与动机

立体视觉模拟人类双目视觉，广泛应用于 3D 电影、自动驾驶、AR/VR 等场景。随着立体相机的普及，立体图像数据量急剧增长，高效压缩成为关键需求。

现有立体图像压缩方法存在两大问题：

1. **单向压缩导致质量不平衡**：主流方法先压缩一个视图，再以其为参考压缩另一个视图。这种单向依赖导致两个视图的重建质量差异显著（实验中 VVC 的两视图 PSNR 差距达 2.58 dB），不利于人眼观看和下游任务。
2. **现有双向方法性能受限**：仅有的双向方法 BCSIC 使用 2D 卷积分别处理两视图，无法捕获视图间的对齐特征；其熵模型仅利用空间上下文，忽略了通道维度的丰富依赖信息。

## 核心问题

如何设计一个对称双向的立体图像压缩框架，既能充分利用视图间的相关性实现高压缩率，又能保证两个视图的重建质量均衡？

## 方法详解

### 整体架构

BiSIC 由两个核心组件构成：联合编解码器（Joint Codec）和跨维度熵模型（Cross-Dimensional Entropy Model）。整个流程完全对称——左右视图被同等对待，不存在主从关系。

### 1. 3D 卷积编解码器

与以往使用 2D 卷积分别处理每个视图不同，BiSIC 采用 3D 卷积作为编解码器骨干。3D 卷积将左右两幅图像沿视图维度堆叠后联合处理，天然具备提取视图间相关性的能力。

- **编码器**：4 层 3D 卷积，将立体图像对从 $B \times 3 \times H \times W$ 下采样为紧凑的 latent 表示 $B \times N \times \frac{H}{16} \times \frac{W}{16}$
- **解码器**：对称的 3D 转置卷积结构，将量化后的 latent 恢复为重建图像
- **Hyper 编解码器**：同样基于 3D 卷积，生成辅助的 hyperprior 信息用于熵模型

### 2. 双向互注意力模块（Bidirectional Mutual Attention Block）

卷积层擅长局部建模但长距离依赖能力有限，因此在编码器的第 2 和第 4 层卷积之后插入互注意力模块来捕获全局特征。该模块包含两个阶段：

- **Cross-Key Attention**：用一个视图的 Key 和另一个视图的 Value 生成注意力图，再由当前视图的 Query 查询——用于发现跨视图的对齐特征和共同模式
- **Cross-Query Attention**：Key 和 Value 来自同一视图，由另一视图的 Query 查询——保留各视图自身特征的同时借助对方视图分配注意力权重

两阶段输出经自注意力增强后通过共享参数的 Combine Block 融合。为避免 $(H \times W)^2$ 的计算量，采用 Efficient Attention，将注意力图大小缩减为 $C_K \times C_V$，与输入尺寸无关。

### 3. 跨维度熵模型（Cross-Dimensional Entropy Model）

熵模型是压缩性能的关键——它估计 latent 的概率分布，分布估计越准确，熵编码产生的码流越紧凑。BiSIC 提出聚合四种条件信息的跨维度熵模型：

- **Hyperprior**：由 hyper 解码器生成的辅助信息，提供全局统计特征
- **通道上下文（Channel Context）**：将 latent 沿通道均分为 $K$ 个 slice，逐 slice 解码时用已解码的前序 slice（含两个视图）作为参考，通过互注意力模块聚合
- **空间上下文（Spatial Context）**：在每个 slice 内，用 Masked 3D 卷积从已编码的因果区域提取空间依赖（mask 确保只访问已解码的位置）
- **立体依赖（Stereo Dependency）**：上述通道和空间上下文均同时考虑两个视图的信息，3D 卷积天然捕获跨视图关联

最终通过聚合网络 $\mathbf{G}_{ag}$ 融合所有条件来估计高斯分布的均值 $\mu$ 和方差 $\sigma^2$。

### 4. 快速变体 BiSIC-Fast

逐像素的空间自回归推理耗时严重。BiSIC-Fast 将 Checkerboard 结构扩展为 Stereo-Checkerboard：

- 将 latent 分为 stereo anchor 和 stereo non-anchor 两部分
- Anchor 部分仅依赖 hyperprior 和通道上下文解码
- Non-anchor 部分额外利用已解码的 anchor 作为空间条件
- 将逐像素自回归简化为两步操作，大幅加速

### 损失函数

标准的 rate-distortion 损失：$\mathcal{L} = \lambda D + R$，其中 $D$ 为 MSE 或 MS-SSIM 失真，$R$ 为 latent 和 hyperprior 的比特率，$\lambda$ 控制率失真权衡。

## 实验关键数据

### 数据集

- **InStereo2K**：2060 对室内立体图像（2010 训练 / 50 测试），分辨率 1080×860
- **Cityscapes**：5000 对室外城市场景（2975 训练 / 1525 测试），分辨率 2048×1024

### RD 性能（BDBR，以 BPG 为基准，越低越好）

| 方法 | InStereo2K PSNR | InStereo2K MS-SSIM | Cityscapes PSNR | Cityscapes MS-SSIM |
|------|----------------|-------------------|----------------|-------------------|
| VVC | -35.31% | -31.05% | -56.25% | -44.04% |
| ECSIC | -43.71% | -55.65% | -52.06% | -64.96% |
| **BiSIC** | **-48.07%** | **-61.13%** | **-57.49%** | **-67.98%** |
| BiSIC-Fast | -45.35% | -59.36% | -51.96% | -65.56% |

- 相比 VVC，BiSIC 在 InStereo2K 上额外节省 12.76% 比特率（PSNR）和 30.08%（MS-SSIM）
- 相比最强双向基线 BCSIC，额外节省 6.5%–15% 比特率

### 运行时间

| 方法 | BPG | HEVC | VVC | SASIC | BCSIC | BiSIC | BiSIC-Fast |
|------|------|------|------|-------|-------|-------|------------|
| 时间 | 16.17s | 28.16s | 190.27s | 20.24s | 89.44s | 167.25s | **22.82s** |

BiSIC-Fast 运行时间仅 22.82s，接近 BPG/SASIC 水平，远快于 BiSIC 和 VVC。

### 消融实验（BD-PSNR，相对 BiSIC）

- 2D Conv 替换 3D Conv：-0.32 dB（说明 3D 卷积有效捕获视图间特征）
- 用 Minnen 熵模型替换跨维度熵模型：-0.35 dB（验证多维条件的必要性）
- 移除互注意力模块：-0.79 dB（影响最大，说明全局特征交换至关重要）

## 亮点

1. **对称双向设计**消除了单向方法固有的视图质量不平衡问题，实验中 BiSIC 的双视图 PSNR 差异极小，而 VVC 差距达 2.58 dB
2. **3D 卷积骨干**是一个简洁有效的设计选择，天然适配立体图像的视图维度，比分别用 2D 卷积处理再融合更为优雅
3. **跨维度熵模型**系统性地聚合 hyperprior、空间、通道、立体四种条件，全面且对称
4. **Stereo-Checkerboard 快速变体**在性能仅有轻微下降的前提下将运行时间从 167s 降至 23s，实用性大幅提升

## 局限与展望

1. BiSIC 的编解码时间（167.25s）仍然较长，实际部署受限；BiSIC-Fast 解决了速度但牺牲了部分性能
2. 仅在 InStereo2K 和 Cityscapes 两个数据集上验证，缺乏更多样化场景（如遥感、医学等）的测试
3. 熵模型中的 slice 数 $K$ 和卷积核大小等超参数的选择未做深入分析
4. 3D 卷积的参数量和显存开销相较 2D 卷积更大，未讨论模型复杂度与压缩性能的权衡
5. 未探索与感知质量指标（如 LPIPS）或下游任务性能的关联

## 与相关工作的对比

| 方法 | 方向 | 编码器 | 视图交互 | 熵模型条件 |
|------|------|--------|----------|-----------|
| HESIC+ | 单向 | 2D Conv | Homography warp | Hyperprior + 空间 |
| SASIC | 单向 | 2D Conv | 水平位移 | Hyperprior + 空间 |
| ECSIC | 单向 | 2D Conv | 无显式 warp | 单向条件熵 |
| BCSIC | 双向 | 2D Conv（分离） | Contextual transfer | Hyperprior + 空间 |
| LDMIC | 双向 | 2D Conv | 注意力 | Hyperprior + 空间 + 通道 |
| **BiSIC** | **双向** | **3D Conv（联合）** | **互注意力** | **Hyperprior + 空间 + 通道 + 立体** |

BiSIC 的核心优势在于用 3D 卷积实现联合处理（而非分别处理后交互），以及跨维度熵模型对多种条件的系统性整合。

## 启发与关联

- 3D 卷积处理立体对的思路可推广到多视图压缩和视频压缩中的多帧联合编码
- 跨维度熵模型的设计理念（多源条件聚合）对单图像压缩的熵模型改进也有参考价值
- Stereo-Checkerboard 的加速策略体现了"在自回归步数和条件丰富度之间寻找平衡"的通用思路
- 双向对称架构的均衡压缩特性对立体视觉下游任务（深度估计、3D 重建）友好

## 评分

- 新颖性: ⭐⭐⭐⭐ （3D 卷积骨干和跨维度熵模型的组合设计新颖，但各组件单独来看并非全新）
- 实验充分度: ⭐⭐⭐⭐ （消融完整，与多种基线对比充分，但数据集偏少）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，图示直观，公式表达规范）
- 价值: ⭐⭐⭐⭐ （在立体图像压缩领域取得 SOTA，双向对称设计思路具有实用意义）

<!-- RELATED:START -->

## 相关论文

- [Learned Image Compression with Dictionary-based Entropy Model](../../CVPR2025/model_compression/learned_image_compression_with_dictionary-based_entropy_model.md)
- [BaSIC: BayesNet Structure Learning for Computational Scalable Neural Image Compression](basic_bayesnet_structure_learning_for_computational_scalable_neural_image_compre.md)
- [Generative Video Compression with One-Dimensional Latent Representation](../../CVPR2026/model_compression/generative_video_compression_with_one-dimensional_latent_representation.md)
- [LALIC: Linear Attention Modeling for Learned Image Compression](../../CVPR2025/model_compression/linear_attention_modeling_for_learned_image_compression.md)
- [A*-Thought: Efficient Reasoning via Bidirectional Compression for Low-Resource Settings](../../NeurIPS2025/model_compression/a-thought_efficient_reasoning_via_bidirectional_compression_for_low-resource_set.md)

<!-- RELATED:END -->
