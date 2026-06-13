---
title: >-
  [论文解读] Style Quantization for Data-Efficient GAN Training
description: >-
  [CVPR 2025][模型压缩][GAN] SQ-GAN 通过将 StyleGAN 的中间 style 空间离散量化为可学习码本，把稀疏连续潜变量空间压缩为紧凑结构化的离散代理空间，增强有限数据下判别器一致性正则化的效果，并利用 CLIP 嵌入+最优传输距离初始化码本，将外部语义知识注入码本，显著提升小样本 GAN 的生成质量。
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "GAN"
  - "风格空间量化"
  - "一致性正则化"
  - "码本学习"
  - "最优传输"
---

# Style Quantization for Data-Efficient GAN Training

**会议**: CVPR 2025  
**arXiv**: [2503.24282](https://arxiv.org/abs/2503.24282)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 小样本GAN训练, 风格空间量化, 一致性正则化, 码本学习, 最优传输

## 一句话总结

SQ-GAN 通过将 StyleGAN 的中间 style 空间离散量化为可学习码本，把稀疏连续潜变量空间压缩为紧凑结构化的离散代理空间，增强有限数据下判别器一致性正则化的效果，并利用 CLIP 嵌入+最优传输距离初始化码本，将外部语义知识注入码本，显著提升小样本 GAN 的生成质量。

## 研究背景与动机

**领域现状**：在数据有限（几百到几千张）的场景下训练 GAN，判别器极易过拟合，导致生成质量下降。现有解决方案包括数据增强（DiffAug、ADA）、模型正则化（LeCam、DigGAN、CR）以及利用预训练模型引入外部知识（KD-DLGAN）。其中一致性正则化（CR）通过强制判别器对 $z$ 和 $z+\epsilon$ 生成的图像给出一致评分来增强鲁棒性。

**现有痛点**：CR 在有限数据下面临根本性挑战——生成器对输入潜空间 $\mathcal{Z}$ 的探索不充分，导致潜空间中相邻变量 $z$ 和 $z+\epsilon$ 可能映射到真实感差异巨大的图像。如果强制判别器对这两张差异很大的图像给出一致评分，反而会损害判别器的判别能力。

**核心矛盾**：有限数据 vs 连续潜空间的充分探索之间存在根本冲突。连续潜空间太大太稀疏，有限样本无法覆盖，导致从中采样的点对可能落在"荒芜区域"，生成质量不可控。

**本文目标**：构造一个更紧凑、更结构化的离散代理空间替代原始连续潜空间，使有限数据下相邻潜变量更可靠地映射到相近质量的图像，从而真正发挥一致性正则化的优势。

**切入角度**：受向量量化（VQ-VAE、VQ-GAN）启发，但不是在数据空间做量化，而是在 StyleGAN 的中间潜空间（style space $\mathcal{W}$）上做量化——这个空间比原始 $\mathcal{Z}$ 更解耦，每个维度控制不同属性。

**核心 idea**：将中间潜变量 $w$ 分段量化到可学习码本上，形成离散代理空间 $\mathcal{W}^q$，在此空间上做一致性正则化；同时用 CLIP+最优传输将训练数据的语义知识注入码本初始化。

## 方法详解

### 整体框架

SQ-GAN 在 StyleGAN2 框架上增加三个关键组件：(1) 风格空间量化——将 $w$ 分段后量化到码本，用量化后的 $w^q$ 驱动合成网络；(2) 量化一致性正则化——在量化后的代理空间上执行 CR；(3) 知识增强码本初始化（CBI）——用 CLIP 特征和最优传输在训练前预对齐码本编码与训练数据语义。

### 关键设计

1. **风格空间离散量化 (Style Quantization)**:

    - 功能：将连续稀疏的 style 空间 $\mathcal{W}$ 压缩为紧凑结构化的离散代理空间 $\mathcal{W}^q$
    - 核心思路：将中间潜变量 $w \in \mathbb{R}^{d_w}$ 分割为 $s$ 个子向量 $\{\hat{w}_i\}_{i=1}^s$，每个 $\hat{w}_i \in \mathbb{R}^{d_w/s}$（设 $d_w/s=4$）。每个子向量量化到可学习码本 $\mathcal{C} \in \mathbb{R}^{k \times (d_w/s)}$ 的最近邻：$\hat{w}_i^q = \arg\min_{c_j \in \mathcal{C}} \|\hat{w}_i - c_j\|$。量化后的子向量拼接成 $w^q = [\hat{w}_1^q, ..., \hat{w}_s^q]$ 送入合成网络。代理空间是 $s$ 个码本的笛卡尔积 $\mathcal{W}^q = \mathcal{C}^1 \times ... \times \mathcal{C}^s$
    - 设计动机：原始连续 $\mathcal{W}$ 在有限数据下无法充分覆盖，离散化后的 $\mathcal{W}^q$ 空间更小、每个编码更具语义性。相邻量化编码更可能映射到质量一致的图像，从根本上解决 CR 的有效性问题。且在 style 空间（而非原始 $z$ 空间）做量化保证了解耦性——每个子向量控制不同属性

2. **量化一致性正则化 (Quantized CR) + 均匀性约束**:

    - 功能：在量化代理空间上增强判别器鲁棒性，同时防止码本坍缩
    - 核心思路：对 $z$ 加扰动 $\epsilon$ 后映射到 $w' = f_\mathcal{W}(z+\epsilon)$ 再量化得 $w'^q$，CR 损失为 $\mathcal{L}_{qcr} = \mathbb{E}[\|f_D(g(w^q)) - f_D(g(w'^q))\|^2]$。即使扰动后的 $w'$ 与 $w$ 不同，量化后可能落在相同码本编码上（输出完全一致），或落在相邻但同样高质量的编码上。为防止码本坍缩，将码本投影到单位超球面并最小化 RBF 核均匀性损失 $\mathcal{L}_{uf} = \log \mathbb{E}[\exp(-t\|\bar{c}_i - \bar{c}_j\|^2)]$
    - 设计动机：离散化天然提供了"量化鲁棒性"——小扰动不一定改变量化结果，从而让 CR 约束更合理。均匀性约束确保码本编码在超球面上均匀分布，避免退化到少数编码

3. **知识增强码本初始化 (CBI)**:

    - 功能：利用预训练基础模型的语义知识为码本提供有意义的初始化
    - 核心思路：用 CLIP 视觉编码器提取训练图像特征 $F = \{f_i\}$，用 CLIP 文本编码器处理量化后的离散编码（通过 MLP 变换为 token 嵌入）得到特征 $T = \{t_i\}$。两组特征之间计算基于 Sinkhorn 算法的最优传输距离并最小化：$\mathcal{L}_{ot} = \mathbb{E}[d(T, F) \cdot \gamma^*]$，其中 $\gamma^*$ 是最优传输计划。整个初始化优化 $\mathcal{L}_{sq} + \mathcal{L}_{uf} + \mathcal{L}_{ot}$
    - 设计动机：有限数据下从零学习语义丰富的码本很困难。CLIP 作为在大规模数据上预训练的基础模型，其特征空间携带了丰富的视觉-语义先验。通过最优传输将码本对齐到 CLIP 特征空间，相当于为码本预建了一个"语义词汇表"，大幅加速后续训练收敛

### 损失函数 / 训练策略

- 生成器损失：$\mathcal{L}(g, f_\mathcal{W}, \mathcal{C}, P) = \mathcal{L}_{adv}(g) + \lambda_{sq}(\mathcal{L}_{sq} + \mathcal{L}_{uf})$
- 判别器损失：$\mathcal{L}(f_D) = \mathcal{L}_{adv}(f_D) + \lambda_{qcr} \mathcal{L}_{qcr}$
- $\lambda_{sq} = 0.01$, $\lambda_{qcr} = 0.01$, 扰动强度 $\sigma = 0.1$
- 量化使用 straight-through gradient estimator 处理不可导操作
- CBI 阶段先用 $\mathcal{L}_{sq} + \mathcal{L}_{uf} + \mathcal{L}_{ot}$ 预训练码本，然后正式 GAN 训练
- 分辨率 256×256，StyleGAN2 架构

## 实验关键数据

### 主实验

| 数据集 | 指标 | SQ-GAN+CBI | CR | StyleGAN2 | 提升 (vs CR) |
|--------|------|------|----------|------|------|
| Oxford-Dog | FID↓ | **35.01** | 48.73 | 64.26 | -13.72 |
| Oxford-Dog | IS↑ | **12.44** | 10.47 | 9.69 | +1.97 |
| FFHQ-2.5K | FID↓ | **22.04** | 41.43 | 48.11 | -19.39 |
| FFHQ-2.5K | IS↑ | **4.20** | 4.06 | 3.50 | +0.14 |
| MetFaces (1.2K) | FID↓ | **35.44** | 48.89 | 53.21 | -13.45 |
| BreCaHAD (1.75K) | FID↓ | **42.42** | 80.72 | 97.06 | -38.30 |

结合 ADA 增强后效果进一步提升：MetFaces FID 降至 24.77（vs CR+ADA 29.91），BreCaHAD FID 降至 22.61（vs CR+ADA 22.69）。

### 消融实验

| 配置 | FID↓ (Oxford-Dog) | 说明 |
|------|---------|------|
| SQ-GAN + CBI | 35.01 | 完整模型 |
| SQ-GAN (无CBI) | 36.30 | CBI 提供额外 1.3 FID 提升 |
| 仅 CR (无量化) | 48.73 | 量化带来 ~12 FID 提升 |
| StyleGAN2 baseline | 64.26 | 基线 |
| SQ-GAN 无均匀性约束 | ~40 | 码本坍缩导致退化 |

### 关键发现

- **量化是最关键的贡献**：仅加入量化（无 CBI）就将 FID 从 48.73 降到 36.30，贡献了约 85% 的提升
- CBI 在所有数据集上都提供额外收益，在极小数据集（MetFaces 1.2K）上更为明显
- SQ-GAN 与 ADA 增强正交且可叠加，证明量化和增强解决的是不同问题
- 训练过程中 FID 曲线更平滑——量化后训练动态更稳定，判别器过拟合趋势减弱
- 在极小数据集（MetFaces 1.2K、BreCaHAD 1.75K）上改进最为显著，说明量化在数据越少时价值越大

## 亮点与洞察

- **从潜空间覆盖度角度分析 CR 失效原因**是一个深刻的洞察——CR 论文通常只关注正则化形式，而本文指出 CR 有效性的前提是相邻潜变量映射到"质量一致"的图像，这个前提在有限数据下不成立。量化正是从根本上修复了这个前提
- **在 style 空间而非 z 空间做量化**是关键的设计选择——$\mathcal{W}$ 空间比 $\mathcal{Z}$ 更解耦，每个分段自然对应不同属性，使量化后的组合仍然有意义
- **将码本类比为"语义词汇表"**是优雅的概念化：码本编码=词汇，量化后的组合=句子描述图像，CBI=用 CLIP 预建词汇表

## 局限与展望

- 仅在 256×256 分辨率的 StyleGAN2 上验证，未涉及更大模型（如 StyleGAN3）或更高分辨率
- 码本大小 k 和分段数 s 是需要调节的超参数，不同数据集可能需要不同配置
- CBI 依赖 CLIP 模型质量，对 CLIP 覆盖不好的专业领域（如医学图像 BreCaHAD）效果可能受限
- 改进方向：探索自适应码本大小调节；将量化思路应用于扩散模型的有限数据训练；结合码本实现可控属性操纵

## 相关工作与启发

- **vs CR (Zhao et al.)**: CR 直接在连续空间做一致性约束，在数据充足时有效但有限数据下失效。SQ-GAN 通过量化修复了 CR 的前提条件，可视为"CR 的正确打开方式"
- **vs VQ-GAN/VQ-VAE**: VQ-GAN 在数据空间做向量量化用于图像 tokenization，SQ-GAN 在潜空间做量化用于增强训练正则化——目标和位置都不同
- **vs KD-DLGAN**: KD-DLGAN 将预训练模型知识蒸馏到判别器特征中，SQ-GAN 的 CBI 将知识注入码本初始化。两者引入外部知识的方式互补
- **vs ADA**: ADA 通过自适应数据增强缓解过拟合，SQ-GAN 通过潜空间压缩改善正则化。两者正交可叠加

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将 style 空间量化用于增强有限数据 GAN 训练，概念新颖且有理论支撑
- 实验充分度: ⭐⭐⭐⭐ 4 个数据集、与多种 SOTA 方法对比、消融分析完整
- 写作质量: ⭐⭐⭐⭐ 动机分析深入，从 CR 失效原因推导出量化方案逻辑通顺
- 价值: ⭐⭐⭐⭐ 为有限数据下 GAN 训练提供了新范式，量化思路可迁移

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers](q-dit_accurate_post-training_quantization_for_diffusion_transformers.md)
- [\[ECCV 2024\] MetaAug: Meta-Data Augmentation for Post-Training Quantization](../../ECCV2024/model_compression/metaaug_meta-data_augmentation_for_post-training_quantization.md)
- [\[CVPR 2025\] QuartDepth: Post-Training Quantization for Real-Time Depth Estimation on the Edge](quartdepth_post-training_quantization_for_real-time_depth_estimation_on_the_edge.md)
- [\[CVPR 2025\] FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation](fima-q_post-training_quantization_for_vision_transformers_by_fisher_information_.md)
- [\[ACL 2025\] EfficientQAT: Efficient Quantization-Aware Training for Large Language Models](../../ACL2025/model_compression/efficientqat.md)

</div>

<!-- RELATED:END -->
