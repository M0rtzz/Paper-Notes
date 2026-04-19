---
title: >-
  [论文解读] Layer Consistency Matters: Elegant Latent Transition Discrepancy for Generalizable Synthetic Image Detection
description: >-
  [CVPR 2026][图像生成][AI生成图像检测] 提出LTD方法，发现真实图像在ViT中间层具有稳定的层间特征一致性，而生成图像呈现突变，利用层间转换差异实现跨GAN/DM的通用生成图像检测，在UFD/DRCT-2M/GenImage上均超越SOTA。
tags:
  - CVPR 2026
  - 图像生成
  - AI生成图像检测
  - CLIP
  - ViT
  - 层间一致性
---

# Layer Consistency Matters: Elegant Latent Transition Discrepancy for Generalizable Synthetic Image Detection

**会议**: CVPR 2026  
**arXiv**: [2603.10598](https://arxiv.org/abs/2603.10598)  
**代码**: https://github.com/yywencs/LTD  
**领域**: 图像生成 / AI生成图像检测  
**关键词**: 合成图像检测, 层间转换差异, CLIP-ViT, 跨域泛化, 动态层选择

## 一句话总结

发现真实图像在冻结CLIP ViT中间层的特征表示呈现稳定的层间过渡，而合成图像在中间层出现显著的注意力突变，提出Layer Transition Discrepancy (LTD) 方法建模该差异，在UFD上mean Acc达96.90%，DRCT-2M上达99.54%，GenImage上达91.62%，全面超越SOTA。

## 研究背景与动机

**领域现状**：生成模型（GAN、扩散模型）合成的图像越来越逼真，区分真假图像成为紧迫需求。现有检测方法分为三类：(1) 基于空间纹理/频率的方法（CNNSpot、NPR、FreqNet）——依赖特定模型伪影，跨域泛化差；(2) 扩散模型专用检测（DRCT、LaRE2）——对GAN生成的图像效果不佳；(3) 基于冻结CLIP的方法（UnivFD、RINE、FatFormer）——利用预训练语义特征。

**现有痛点**：低级伪影（频率、纹理）随生成模型的进化而变化，学到的是model-specific的bias；CLIP-based方法中，UnivFD仅用最后一层特征忽略低级信息，RINE等方法融合所有层特征但包含大量无关信息引入噪声。

**核心矛盾**：需要一种与具体生成模型无关的、通用的检测线索——既不依赖于特定伪影，又能捕捉真实与合成图像的本质差异。

**本文的发现**：通过分析CLIP ViT各层间特征的余弦相似度和L2距离，发现真实图像在中间层保持稳定的语义注意力一致性（层间特征平滑过渡），而合成图像在中间层出现前景/背景注意力的突然跳变（层间转换差异大）。这一现象可能源于生成模型优先优化像素级逼真度和高级语义一致性，但缺乏严格的物理约束，导致中间层整合纹理到结构时无法维持连续的空间相关性。

**核心idea**：利用Layer Transition Discrepancy (LTD)——中间层相邻层特征的差异——作为模型无关的检测信号，同时建模全局结构一致性和局部层间变化。

## 方法详解

### 整体框架

使用冻结的CLIP ViT-L/14作为骨干提取分层特征。通过动态层选择策略自适应选取最具判别力的连续中间层子集，计算相邻层特征差异得到LTD特征。设计双分支检测架构：一分支处理原始中间层特征建模整体一致性，另一分支处理LTD差异特征放大层间变化。两分支共享权重的Transformer block处理后拼接，输入MLP分类。

### 关键设计

1. **动态层选择策略（Dynamic Layer-wise Selection）**:
    - 功能：从ViT的24层中自适应选取最具判别力的 $n$ 个连续中间层
    - 核心思路：定义可学习logits $\boldsymbol{\pi} \in \mathbb{R}^C$（$C = l - n + 1$ 个候选窗口），使用Gumbel-Softmax确定最优起始层索引 $s$，在训练中保持可微分
    - 设计动机：不同图像的判别性层可能不同（实验发现层11-19最优），固定层选择不够灵活；Gumbel-Softmax实现端到端可微的离散选择

2. **层间转换差异特征（Layer Transition Discrepancy, LTD）**:
    - 功能：捕捉真实/合成图像在ViT中间层的层间过渡差异
    - 核心思路：对选定的 $n$ 个连续层 $\{\mathbf{f}_s^{(k)}\}_{k=1}^n$ 计算相邻层CLS token差异 $\mathbf{d}_s^{(k)} = \mathbf{f}_s^{(k+1)} - \mathbf{f}_s^{(k)}$，得到 $n-1$ 个LTD差异向量
    - 设计动机：相比直接使用原始特征，差异特征聚焦于层间变化模式，抑制无关冗余信息；真实图像差异小而稳定，合成图像差异大而突变

3. **双分支共享权重检测架构**:
    - 功能：同时建模全局结构一致性和局部层间变化
    - 核心思路：原始特征分支 $\mathbf{F}_s = [\mathbf{f}_s, \mathbf{f}_{cls}, \mathbf{f}_p]$ 和LTD分支 $\mathbf{D} = [\mathbf{d}, \mathbf{d}_{cls}, \mathbf{d}_p]$ 各添加CLS token和位置编码，通过权重共享的可训练Transformer block交互学习
    - 设计动机：权重共享强制特征对齐，将空间一致性和层间转换映射到统一语义空间，防止分布分散

### 损失函数 / 训练策略

标准二分类交叉熵损失。训练仅需2类ProGAN数据（chair + tvmonitor），5个epoch即可收敛。所有CLIP ViT参数冻结，仅训练层选择logits、位置编码和双分支Transformer + MLP。

## 实验关键数据

### 主实验
| 数据集 | 指标 | 本文LTD | ForgeLens | FatFormer | 提升 |
|--------|------|---------|-----------|-----------|------|
| UFD | Mean Acc | 96.90% | 95.56% | 95.98% | +0.92% |
| UFD | Mean AP | 99.51% | 99.11% | 99.15% | +0.36% |
| DRCT-2M | Mean Acc | 99.54% | 98.22% | - | +1.32% |
| DRCT-2M | Mean AP | 99.99% | 99.76% | - | +0.23% |
| GenImage | Mean Acc | 91.62% | 89.18% | 84.34% | +2.44% |
| GenImage | Mean AP | 97.17% | 96.76% | 95.01% | +0.41% |

### 消融实验
| 配置 | UFD Acc | DRCT-2M Acc | Mean Acc | 说明 |
|------|---------|-------------|----------|------|
| Raw ML. only | 84.92% | 92.75% | 88.84% | 仅原始中间层特征 |
| Raw ML. + Pos.Enc | 94.22% | 96.12% | 95.17% | 加位置编码 |
| LTD only | 86.42% | 93.50% | 89.96% | 仅LTD差异特征 |
| LTD + Pos.Enc | 92.43% | 94.01% | 93.22% | LTD加位置编码 |
| Full model | 96.90% | 99.54% | 98.22% | 双分支完整模型 |

### 关键发现
- 真实图像在ViT中间层（约Layer 11-19）保持稳定的注意力一致性，合成图像在同区间出现显著注意力跳变
- 浅层（Layer 0-7）和深层（Layer 16-23）对真假图像的区分能力有限，中间层（Layer 8-15）最强
- 最优窗口为5个连续层（层11起始），过多或过少都会降低性能
- 仅用2类训练数据即可泛化到16种不同GAN和DM生成器
- 对JPEG压缩（QF 60-100）和下采样（0.5x-1.0x）保持鲁棒

## 亮点与洞察
- **发现了一个此前未被注意的检测线索**：层间转换差异是模型无关的，不依赖于特定生成器的伪影，因此具有天然的跨域泛化能力
- **极高的训练效率**：仅需2类训练数据、5个epoch就能训练完成，在4090上几分钟完成
- **推理速度最快**：冻结CLIP骨干 + 轻量双分支，FPS显著高于FatFormer等方法
- **Physical prior的洞察**：生成模型重点优化像素级真实感和高级语义对齐，但中间层的结构连续性不受约束，因此成为泄露生成来源信息的"窗口"

## 局限与展望
- Midjourney在GenImage上的Acc仅62.97%，对部分高质量商业模型仍有提升空间
- 严重依赖CLIP ViT的预训练表示——如果CLIP被用于生成过程本身（如未来模型），检测效果可能下降
- 动态层选择用Gumbel-Softmax，推理时退化为固定选择，未能真正做到per-image adaptive
- 仅使用CLS token的LTD，未利用spatial token的局部层间差异信息

## 相关工作与启发
- **vs UnivFD**: UnivFD仅用CLIP最后一层做线性探测，忽略中间层信息；LTD利用中间层的层间差异，mean Acc提升11%
- **vs FatFormer/RINE**: 它们融合所有层特征但引入无关信息和噪声；LTD聚焦层间变化抑制冗余
- **vs NPR/FreqNet**: 依赖低级统计伪影（上采样指纹、频谱），对扩散模型泛化差；LTD利用层间结构一致性，对GAN和DM都有效
- **启发**：预训练大模型的中间层表示包含丰富的取证信号，层间动态比单层特征更具判别力

## 评分
- 新颖性: ⭐⭐⭐⭐ 发现层间转换差异这一新检测线索，观察深刻且启发性强
- 实验充分度: ⭐⭐⭐⭐⭐ 三大benchmark全面评测，16+生成器，鲁棒性/消融实验充分
- 写作质量: ⭐⭐⭐⭐ 动机分析清晰，可视化有说服力，但方法部分公式偏简单
- 价值: ⭐⭐⭐⭐ 高实用价值，训练简单高效，泛化性强，适合部署

<!-- RELATED:START -->

## 相关论文

- [From Inpainting to Layer Decomposition: Repurposing Generative Inpainting Models for Image Layer Decomposition](from_inpainting_to_layer_decomposition_repurposing_generative_inpainting_models_.md)
- [Diversity over Uniformity: Rethinking Representation in Generated Image Detection](diversity_over_uniformity_rethinking_representation_in_generated_image_detection.md)
- [CausalCLIP: Causally-Informed Feature Disentanglement and Filtering for Generalizable Detection of Generated Images](../../AAAI2026/image_generation/causalclip_causally-informed_feature_disentanglement_and_filtering_for_generaliz.md)
- [Image Diffusion Preview with Consistency Solver](image_diffusion_preview_with_consistency_solver.md)
- [TriDF: Evaluating Perception, Detection, and Hallucination for Interpretable DeepFake Detection](tridf_evaluating_perception_detection_and_hallucination_for_interpretable_deepfa.md)

<!-- RELATED:END -->
