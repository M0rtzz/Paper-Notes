---
title: >-
  [论文解读] SPAI: Any-Resolution AI-Generated Image Detection by Spectral Learning
description: >-
  [CVPR 2025][图像生成][频谱学习] 提出 SPAI，通过掩码频谱学习（Masked Spectral Learning）建模真实图像的频率分布，引入频谱重建相似度（SRS）和频谱上下文注意力（SCA）检测 AI 生成图像作为分布外样本，在 13 种生成模型上平均 AUC 达 91.0%…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "频谱学习"
  - "自监督"
  - "频率重建"
  - "OOD检测"
  - "任意分辨率"
---

# SPAI: Any-Resolution AI-Generated Image Detection by Spectral Learning

**会议**: CVPR 2025  
**arXiv**: [2411.19417](https://arxiv.org/abs/2411.19417)  
**代码**: [https://mever-team.github.io/spai](https://mever-team.github.io/spai)  
**领域**: 图像生成  
**关键词**: 频谱学习, 自监督, 频率重建, OOD检测, 任意分辨率

## 一句话总结
提出 SPAI，通过掩码频谱学习（Masked Spectral Learning）建模真实图像的频率分布，引入频谱重建相似度（SRS）和频谱上下文注意力（SCA）检测 AI 生成图像作为分布外样本，在 13 种生成模型上平均 AUC 达 91.0%，比次优方法绝对提升 5.5%，且支持任意分辨率图像检测。

## 研究背景与动机

**领域现状**：AI 图像生成技术飞速发展，从 GAN 到扩散模型（Stable Diffusion、DALLE3、Midjourney 等），生成图像质量极高且模型数量激增。有害虚假图像在互联网上快速传播，迫切需要鲁棒的 AI 生成图像检测（AID）方法。

**现有痛点**：现有检测器通过学习捕获特定生成模型引入的伪影（如频谱中的异常模式、texture 不一致等）来工作。然而，不同生成模型引入的伪影差异巨大——即使是微小差异的模型也会产生截然不同的伪影。结果是，现有检测器在训练时见过的模型上表现优异，但在未见模型上灾难性失败。例如 NPR 在 Firefly 上 AUC 仅 38.0%，而 DMID 在 SD3 上仅 67.9%。

**核心矛盾**：学习生成模型的伪影本质上是追踪"不断变化"的目标——每当新模型出现，检测器就需要更新训练。而维护一个涵盖所有模型的训练集是不可行的。

**本文目标**：不依赖特定模型伪影，而是建模**真实图像**的不变特征，从而将 AI 生成图像作为分布外（OOD）样本检测。

**切入角度**：作者指出真实图像的**频谱分布**构成了一个对生成模型高度不变但具有强区分力的模式——它不受特定生成模型引入的影响，却能有效区分真实与合成图像。

**核心 idea**：用自监督的频率重建作为预训练任务学习真实图像的频谱模型，然后通过比较重建精度来检测 OOD 样本。

## 方法详解

### 整体框架
输入图像首先被分割为 $K$ 个 $224 \times 224$ 的 patch，每个 patch 经过频率掩码生成低频和高频分量，送入预训练的频谱 ViT 模型 $\mathcal{G}$ 提取多层特征，计算频谱重建相似度（SRS），结合频谱上下文向量（SCV），通过频谱上下文注意力（SCA）融合所有 patch 的信息，最终由 MLP 输出真假概率。

### 关键设计

1. **掩码频谱学习（Masked Spectral Learning）**:

    - 功能：自监督学习真实图像的频率分布
    - 核心思路：使用 2D DFT 将图像分解为低频 $x^l$ 和高频 $x^h$ 分量（通过半径 $r$ 的圆形频率掩码），随机选择一个作为输入，训练 ViT 模型 $\mathcal{G}$ 重建完整频谱。损失函数为频率域距离 $\mathcal{L}_{rec} = \mathcal{D}(\mathcal{F}(x), \mathcal{F}(\hat{x}))$。仅使用 ImageNet 的 120 万真实图像训练，权重在后续训练中冻结
    - 设计动机：真实图像的频率重建模型对真实图像重建更准确，而对 AI 生成图像（OOD 数据）重建误差更大，这成为检测的基础

2. **频谱重建相似度（Spectral Reconstruction Similarity, SRS）**:

    - 功能：量化图像频谱与学习到的真实图像频谱模型之间的对齐程度
    - 核心思路：将原始图像、低频分量、高频分量分别编码为每层 $N$ 个 transformer block 的特征表示，通过投影算子映射到统一空间后，计算三对余弦相似度（原始-低频 $\omega^{ol}$、原始-高频 $\omega^{oh}$、低频-高频 $\omega^{lh}$），对每对提取均值和标准差，拼接 $N$ 层的 $6N$ 个值形成 SRS 向量 $z^\lambda$
    - 设计动机：真实图像的三组分量特征高度对齐（高相似度），AI 生成图像因频谱异常而对齐度较低

3. **频谱上下文注意力（Spectral Context Attention, SCA）**:

    - 功能：将任意分辨率图像的多个 patch 级检测结果融合为图像级判断
    - 核心思路：定义可学习的查询向量 $q$，对所有 $K$ 个 patch 的频谱向量 $z_k^S$ 计算注意力权重 $\mathcal{A} = \text{softmax}(q \cdot (z^S W_K)^\top / \sqrt{D_h})$，加权融合得到图像级表示。计算复杂度仅为 $O(K)$，可高效处理百万像素级图像
    - 设计动机：简单的 resize 操作会丢弃高频信息，而高频信息正是检测的关键线索。SCA 允许在原始分辨率下处理图像的每个 patch，然后注意力加权融合

### 损失函数 / 训练策略
端到端训练使用二元交叉熵损失 $\mathcal{L}_{cls} = BCE(\hat{y}, y)$。训练时使用固定 $K_{\text{training}}=4$ 个随机增强视图代替实际 patch（解决训练数据尺寸限制），推理时使用实际 patch 数。训练数据仅包含来自单一 LDM 模型的 18 万低分辨率 (256×256) 合成图像和 18 万真实图像。

## 实验关键数据

### 主实验
在 13 种生成模型（覆盖 GAN、扩散模型、商业模型）上的跨模型检测 AUC：

| 模型分辨率 | 生成模型 | SPAI (Ours) | RINE (次优) | DMID | PatchCr. |
|---|---|---|---|---|---|
| < 0.5 MP | Glide | 90.2 | 95.6 | 73.1 | 78.4 |
| | SD1.3 | 99.6 | 99.9 | 100.0 | 95.7 |
| | SD1.4 | 99.6 | 99.9 | 100.0 | 96.2 |
| 0.5-1.0 MP | Flux | 83.0 | 93.0 | 97.2 | 86.9 |
| | DALLE2 | 91.1 | 93.0 | 54.3 | 81.8 |
| | SD2 | 96.5 | 96.6 | 99.7 | 95.7 |
| | SDXL | 97.4 | 99.3 | 99.6 | 96.7 |
| | SD3 | **75.9** | 39.1 | 67.9 | 33.8 |
| | GigaGAN | 85.4 | 92.9 | 67.9 | 98.0 |
| > 1.0 MP | MJv5 | 94.5 | 96.4 | 99.9 | 79.0 |
| | MJv6.1 | 84.0 | 81.2 | 94.4 | 96.1 |
| | DALLE3 | **90.2** | 41.8 | 41.3 | 28.1 |
| | Firefly | **96.0** | 82.9 | 90.2 | 79.1 |
| **平均** | | **91.0** | 85.5 | 83.5 | 80.4 |

SPAI 以 91.0% 平均 AUC 领先次优方法 5.5%，且在 SD3、DALLE3、Firefly 等高难度模型上优势巨大。

### 消融实验

| 配置 | AUC | 说明 |
|---|---|---|
| SPAI (完整) | 91.0 | 全部组件 |
| w/o spectral pretraining | 52.5 | 去掉频谱预训练，降 38.5% |
| w/o SRS | 71.0 | 去掉 SRS，降 20.0% |
| w/o SCV | 84.9 | 去掉上下文向量，降 6.1% |
| w/o SCA | 83.2 | 去掉注意力融合，降 7.8% |
| w/o SCA + TenCrop (mean) | 85.3 | 用均值替代注意力，降 5.7% |
| w/o JPEG augm. | 89.1 | 去掉 JPEG 增强 |
| w/o distortions | 84.2 | 去掉所有失真增强，降 6.8% |

| Backbone | 训练数据 | AUC |
|---|---|---|
| CLIP ViT-B/16 | 4 亿 | 87.6 |
| DINOv2 ViT-B/14 | 1.42 亿 | 87.5 |
| **MFM ViT-B/16 (Ours)** | **120 万** | **91.0** |

### 关键发现
- **频谱预训练是核心**：去掉后直接降至 52.5%，说明频率重建是整个方法的根基
- **数据效率极高**：仅用 120 万图像训练的频率重建 ViT 超过用 4 亿图像训练的 CLIP 和 1.42 亿的 DINOv2
- **鲁棒性出色**：在 JPEG (Q=50)、WebP (Q=50)、高斯模糊 (k=7)、高斯噪声 (σ=5)、缩放 (50%) 五种扰动下均优于所有对比方法
- **SD3 和 DALLE3 是最难检测的模型**：所有方法在这两个最新模型上表现最差，但 SPAI 在 SD3 上达 75.9%（次优仅 67.9%），DALLE3 达 90.2%（次优仅 41.8%）

## 亮点与洞察
- **范式转换**：从"学习伪影"到"建模真实分布"——这避免了追踪不断变化的生成模型伪影，是更可持续的检测策略。频谱域的选择尤为关键，因为真实图像的频谱比空间域更具不变性
- **任意分辨率处理**：SCA 模块通过单查询注意力机制以 O(K) 复杂度融合任意数量的 patch，避免了 resize 带来的高频信息损失。这对实际部署（处理百万像素照片）至关重要
- **自监督检测**：不需要标注，频率重建任务天然适合建模真实图像分布，且训练集只需真实图像——理论上永远不会过时

## 局限与展望
- 在 Flux 和 SD3 上 AUC 分别为 83.0% 和 75.9%，说明最新扩散模型在频谱上越来越接近真实图像
- SCA 推理时需要对每个 patch 独立前向 ViT，大分辨率图像的推理延迟可能较大
- 仅使用 ImageNet 预训练频谱模型，对特定领域（如医学图像）的泛化能力未知
- Backbone 固定为 ViT-B/16，更大模型（ViT-L/ViT-H）是否能进一步提升未探索

## 相关工作与启发
- **vs RINE**：同样使用预训练视觉模型，但 RINE 依赖 CLIP 的 ViT-L backbone（4 亿图像）仍只达 85.5%；SPAI 用 120 万图像训练的频谱 ViT 达 91.0%
- **vs DMID**：直接学习空间域伪影，在 SD1.3/1.4 上接近 100% 但对 DALLE3 仅 41.3%；SPAI 更均匀
- **vs NPR**：利用升采样层伪影，在 DALLE3 达 97.1% 但其他模型极差（Flux 19.8%）；"一招鲜"不如分布建模

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 频谱分布建模+OOD检测的范式转换非常有启发性
- 实验充分度: ⭐⭐⭐⭐⭐ 13种生成模型、5种真实图像源、5种扰动、完整消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰但部分数学符号略显冗余
- 价值: ⭐⭐⭐⭐⭐ 实际部署价值极高，支持任意分辨率且对未知模型泛化好

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] A Bias-Free Training Paradigm for More General AI-generated Image Detection](a_bias-free_training_paradigm_for_more_general_ai-generated_image_detection.md)
- [\[CVPR 2025\] Where's the Liability in the Generative Era? Recovery-Based Black-Box Detection of AI-Generated Content](wheres_the_liability_in_the_generative_era_recovery-based_black-box_detection_of.md)
- [\[NeurIPS 2025\] Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection](../../NeurIPS2025/image_generation/physics-driven_spatiotemporal_modeling_for_ai-generated_video_detection.md)
- [\[ECCV 2024\] Zero-Shot Detection of AI-Generated Images](../../ECCV2024/image_generation/zero-shot_detection_of_ai-generated_images.md)
- [\[AAAI 2026\] Aggregating Diverse Cue Experts for AI-Generated Image Detection](../../AAAI2026/image_generation/aggregating_diverse_cue_experts_for_ai-generated_image_detec.md)

</div>

<!-- RELATED:END -->
