---
title: >-
  [论文解读] SeiT++: Masked Token Modeling Improves Storage-Efficient Training
description: >-
  [ECCV 2024][图像分割][存储高效训练] 在 SeiT 的 token 化训练框架上引入掩码 token 建模（MTM）自监督预训练，并设计 TokenAdapt 和 ColorAdapt 两种 token 专用数据增强策略，在仅 1% 存储空间（1.4GB）下将 ImageNet-1k 分类准确率从 74.0% 提升至 77.8%，有效解决了 token 域数据增强的难题。
tags:
  - ECCV 2024
  - 图像分割
  - 存储高效训练
  - 向量量化
  - 掩码建模
  - 数据增强
  - 自监督学习
---

# SeiT++: Masked Token Modeling Improves Storage-Efficient Training

**会议**: ECCV 2024  
**arXiv**: [2312.10105](https://arxiv.org/abs/2312.10105)  
**代码**: 有 (https://github.com/naver-ai/seit)  
**领域**: 分割 / 图像分类  
**关键词**: 存储高效训练, 向量量化, 掩码建模, 数据增强, 自监督学习

## 一句话总结

在 SeiT 的 token 化训练框架上引入掩码 token 建模（MTM）自监督预训练，并设计 TokenAdapt 和 ColorAdapt 两种 token 专用数据增强策略，在仅 1% 存储空间（1.4GB）下将 ImageNet-1k 分类准确率从 74.0% 提升至 77.8%，有效解决了 token 域数据增强的难题。

## 研究背景与动机

训练高性能视觉模型需要海量数据集，带来巨大的存储压力（如 LAION-5B 需要 240TB）。现有存储缩减方法包括：

| 方法 | 思路 | 局限性 |
|------|------|--------|
| 数据集蒸馏 | 将数据集压缩为小型合成集 | 计算复杂度高，不适用于大规模数据 |
| 采样方法 | 选择最具代表性的子集 | 低数据量时多样性不足 |
| 降低分辨率/JPEG压缩 | 减小单张图像大小 | 性能显著下降 |
| **SeiT** | 将图像转为离散 token 存储 | 90%性能，1%存储，但仅限全监督 |

SeiT 的突破性在于使用 ViT-VQGAN tokenizer 将每张图像压缩为 token 序列，使 ImageNet-1k 从 140GB 压缩至 1.4GB。但 SeiT 存在两个关键局限：

**仅探索了全监督学习**，未利用自监督预训练的潜力

**Token 域的数据增强受限**——像素级增强直接应用到 token 会导致严重失真

## 方法详解

### 整体框架

SeiT++ = Masked Token Modeling (MTM) + TokenAdapt + ColorAdapt

**数据准备**：使用 ViT-VQGAN tokenizer 将每张图像离线转换为 token 序列存储

**预训练阶段**：使用 MTM 进行自监督预训练（无需标签）

**微调阶段**：在 token 化的数据集上进行有监督微调（分类/分割）

### 关键设计

**Masked Token Modeling (MTM)**：

类比 BERT 的 Masked Language Modeling，MTM 从可见 token 预测被掩码的 token：

1. **掩码策略**：采用截断正态分布的可变掩码比率
2. **编码器**：仅处理可见 token 嵌入，减少训练时间和显存
3. **解码器**：将编码器输出用掩码 token 填充到原始长度，预测被掩码位置的原始 token
4. **训练目标**：交叉熵损失

$$\mathcal{L}_{recon} = \text{CE}(T'_M, T_M)$$

仅对被掩码的 token 计算损失。

**TokenAdapt（解决几何增强问题）**：

核心问题：token 化过程将 n×n 的 2D 图像块压缩为 1D 向量，**空间信息坍塌**使翻转等增强失效；token 嵌入间的**相互依赖性**使插值类增强（resize、crop）引入伪影。

解决方案：学习一个转换-增强-逆变换流水线

$$Z_T^{\mathbf{A}} = g(\mathbf{A}(f(Z_T))), \quad T^{\mathbf{A}} = \mathbf{q}_{\mathcal{Z}}(Z_T^{\mathbf{A}})$$

- $f$：将 token 嵌入转入增强兼容空间
- $\mathbf{A}$：标准像素级增强（翻转、裁剪、仿射等）
- $g$：逆变换回 token 嵌入空间
- $\mathbf{q}_{\mathcal{Z}}$：向量量化到码本索引

$f$ 和 $g$ 从 token 配对数据 $(T_x, T_{\mathbf{A}(x)})$ 学习，训练后可跨数据集、跨任务通用。

**ColorAdapt（解决颜色增强问题）**：

受 Adaptive Instance Normalization 启发，通过调整 token 嵌入的统计量改变颜色属性：

$$\mathcal{C}(Z_{T_1}, Z_{T_2}) = \sigma(Z_{T_2})\frac{Z_{T_1} - \mu(Z_{T_1})}{\sigma(Z_{T_1})} + \mu(Z_{T_2})$$

在保持目标结构的同时改变颜色表示。

### 损失函数 / 训练策略

- MTM 预训练损失：仅掩码位置的交叉熵
- TokenAdapt 训练损失：增强后 token 嵌入与真实增强图像 token 嵌入间的交叉熵
- 微调继承 SeiT 的训练配方，追加使用 CutMix 和 Emb-Noise

## 实验关键数据

### 主实验（表格）

存储高效 ImageNet-1k 分类（ViT-B/16）：

| 方法 | 输入 | 存储 | Top-1 Acc |
|------|------|------|-----------|
| Full pixels | 图像 | 140 GB | 81.8 |
| JPEG quality=5 | 图像 | 11 GB (8%) | 74.6 |
| 降分辨率 ×0.2 | 图像 | 9.6 GB (7%) | 75.2 |
| SeiT | Token | 1.4 GB (1%) | 74.0 |
| **SeiT++** | **Token** | **1.4 GB (1%)** | **77.8** |

不同存储预算下的对比：

| 存储 | SeiT | SeiT w/ MTM | SeiT++ w/o MTM | SeiT++ w/ MTM |
|------|------|------------|----------------|---------------|
| 1.4 GB | 74.0 | 75.1 | 75.5 | **77.8 (+3.8)** |
| 0.8 GB | 66.3 | 70.6 | 69.1 | **74.1 (+7.8)** |
| 0.3 GB | 47.2 | 53.9 | 51.2 | **60.6 (+13.4)** |

### 消融实验（表格）

各增强策略的独立贡献（ViT-S, ImageNet-100）：

| ColorAdapt | TokenAdapt | Top-1 Acc |
|------------|------------|-----------|
| ✘ | ✘ | 77.3 (SeiT baseline) |
| ✔ | ✘ | 78.3 (+1.0) |
| ✘ | ✔ | 80.4 (+3.1) |
| ✔ | ✔ | **81.4 (+4.1)** |

鲁棒性评估（ViT-B, 无 MTM）：

| 评估基准 | SeiT | SeiT++ | 提升 |
|----------|------|--------|------|
| Clean | 74.0 | 75.5 | +1.5 |
| Gaussian Noise | 50.7 | 58.6 | +7.9 |
| Gaussian Blur | 62.6 | 66.8 | +4.2 |
| ImageNet-R | 25.5 | 30.2 | +4.7 |
| Sketch | 22.6 | 27.7 | +5.1 |

### 关键发现

1. **存储越少，提升越大**：从 1.4GB 到 0.3GB，SeiT++ 相比 SeiT 的提升从 3.8% 扩大到 13.4%
2. MTM 和 token 增强**协同增效**：仅用 MTM 提升 1.1%，仅用增强提升 1.5%，组合提升 3.8%
3. TokenAdapt（几何增强）比 ColorAdapt（颜色增强）贡献更大（+3.1 vs +1.0）
4. ADE-20k 语义分割上也有 +4.2 mIoU 的显著提升
5. 方法可迁移至不同 tokenizer（VQGAN）甚至不同输入格式（DCT 系数）

## 亮点与洞察

1. **Token 域增强的系统分析**：首次深入分析像素级增强在 token 域失效的两个根因——空间信息坍塌和 token 间依赖性
2. **TokenAdapt 的巧妙设计**：不直接在 token 空间做增强，而是学习一个可逆变换到"增强兼容空间"，复用成熟的像素级增强策略
3. **自监督 + token 的首次结合**：证明 MLM 类自监督方法可以从离线 token 直接学习，无需原始像素图像
4. **极致的存储效率**：仅用 1GB 数据即可训练出 70%+ 准确率的 ViT 模型

## 局限与展望

1. 未在 ImageNet-21k 等大规模数据集上实验（受资源限制）
2. TokenAdapt 的转换/逆变换模块需要额外训练，增加了流水线复杂度
3. tokenizer（ViT-VQGAN）本身的重建质量限制了方法的上界
4. 仅探索了 ViT 架构，对 CNN 架构的适用性未知
5. 可探索更先进的 tokenizer（如 SDXL-VAE）进一步提升性能

## 相关工作与启发

- **SeiT**：本文的基础框架，首次证明 token 训练的可行性
- **BeiT / MAE**：掩码图像建模的先驱，本文将其思想迁移到离线 token
- **MAGE**：在生成式学习中使用 VQGAN token，但依赖在线 tokenization
- 启发：离散表示（token）不仅可以压缩存储，还天然适合 MLM 类自监督学习

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 新颖性 | 4 |
| 技术深度 | 4 |
| 实验充分性 | 5 |
| 写作质量 | 4 |
| 实用价值 | 4 |
| **综合** | **4.2** |

<!-- RELATED:START -->

## 相关论文

- [Harnessing Massive Satellite Imagery with Efficient Masked Image Modeling](../../ICCV2025/segmentation/harnessing_massive_satellite_imagery_with_efficient_masked_image_modeling.md)
- [Masked Representation Modeling for Domain-Adaptive Segmentation](../../CVPR2026/segmentation/masked_representation_modeling_for_domain-adaptive_segmentation.md)
- [ColorMAE: Exploring Data-Independent Masking Strategies in Masked AutoEncoders](colormae_exploring_data-independent_masking_strategies_in_masked_autoencoders.md)
- [DreamLIP: Language-Image Pre-training with Long Captions](dreamlip_language-image_pre-training_with_long_captions.md)
- [ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback](controlnet_improving_conditional_controls_with_efficient_consistency_feedback.md)

<!-- RELATED:END -->
