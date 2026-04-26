---
title: >-
  [论文解读] Masked Representation Modeling for Domain-Adaptive Segmentation
description: >-
  [CVPR 2026][图像分割][掩码建模] 提出 Masked Representation Modeling (MRM)，在潜在空间而非像素空间进行掩码与重建，作为 UDA 分割的即插即用辅助任务，在 GTA→Cityscapes 上平均为 4 种 baseline 带来 +2.3 mIoU 提升。
tags:
  - CVPR 2026
  - 图像分割
  - 掩码建模
  - 表示重建
  - 域自适应分割
  - 辅助任务
  - 即插即用
---

# Masked Representation Modeling for Domain-Adaptive Segmentation

**会议**: CVPR 2026  
**arXiv**: [2509.13801](https://arxiv.org/abs/2509.13801)  
**代码**: [GitHub](https://github.com/Wenlve-Zhou/MRM)  
**领域**: 分割 / 无监督域自适应  
**关键词**: 掩码建模, 表示重建, 域自适应分割, 辅助任务, 即插即用

## 一句话总结

提出 Masked Representation Modeling (MRM)，在潜在空间而非像素空间进行掩码与重建，作为 UDA 分割的即插即用辅助任务，在 GTA→Cityscapes 上平均为 4 种 baseline 带来 +2.3 mIoU 提升。

## 研究背景与动机

无监督域自适应 (UDA) 语义分割旨在利用标注的源域数据和无标注的目标域数据，克服域偏移 (domain shift) 问题。辅助自监督任务是增强 UDA 的有效途径，其中对比学习已被广泛探索且效果显著，但另一类强大的自监督方法——掩码图像建模 (MIM)——在 UDA 分割中几乎未被探索。

MIM 未被采用的两个核心原因：

**输入结构约束**：MIM 需要掩码图像 patch，破坏了 DeepLab/DAFormer 等分割网络的输入结构

**优化冲突**：MIM 重建的是低层像素/token，与分割任务的高层语义目标不一致

作者的核心思路：**在特征空间（而非输入空间）做掩码和重建**，既不破坏输入 pipeline，又让重建目标与分割目标对齐——因为重建后的特征直接送入分割解码器做像素分类。

## 方法详解

### 整体框架

MRM 作为辅助任务插入现有 UDA pipeline。在分割模型的编码器 $E(\cdot)$ 输出特征 $f^t = E(x^t)$ 上执行掩码操作，用轻量 Rebuilder $R(\cdot)$ 重建被掩码特征，再送入解码器 $D(\cdot)$ 做分类。推理时 Rebuilder 完全移除，**零额外推理开销**。总优化目标 $\mathcal{L}_{overall} = \mathcal{L}_{sup} + \mathcal{L}_{uda} + \lambda\mathcal{L}_{mrm}$。

### 关键设计

1. **表示空间掩码重建 (MRM)**：将编码器输出特征 $f^t \in \mathbb{R}^{C \times H \times W}$ 先经过表示嵌入层缩放到 $C' \times H' \times W'$（统一不同架构的特征维度），再随机掩码 40% 的位置，用可学习的 mask token 填充。重建后经投影层恢复原始维度，再与原始特征融合 $f^r = M^s \odot f^o + (1 - M^s) \odot f^t$。**关键区别于 MIM**：重建目标不是像素值，而是让解码器在重建特征上做出正确的语义预测 $\mathcal{L}_{mrm} = -\sum_{i,j,c} \tilde{y}_{ijc} \log D(R(E(x^t)))_{ijc}$，使用伪标签 $\tilde{y}$ 监督。

2. **轻量 Rebuilder 模块**：受 MAE 解码器启发的非对称设计。包括：(a) 表示嵌入层——线性映射调整通道维度 + 双线性插值调整空间维度；(b) 掩码层——均匀随机采样生成二值掩码；(c) Transformer 块（仅 1-2 个）+ 绝对位置编码；(d) Projector——转置卷积恢复原始维度。整体非常轻量，仅需极少的 Transformer 块即可有效。

3. **多尺度模型适配**：对 DAFormer 等层级式架构，不在每个 stage 实例化单独的 Rebuilder（开销过大），而是仅使用最后一个 stage 的表示进行 Transformer 处理，再通过独立的上采样操作为每个目标尺度分别生成多尺度特征。此设计灵感源自 ViTDet 的发现：多尺度特征可通过简单上采样从最终表示获得。

### 损失函数 / 训练策略

- $\mathcal{L}_{mrm}$：交叉熵损失，使用目标域伪标签监督
- 权衡系数 $\lambda = 1.0$
- Rebuilder 配置：2 个 Transformer 块，embedding dim=512，$H'=W'=16, C'=512$
- 掩码率 40%（低于 MAE 的 60-75%，因 MRM 的可见 token 处理深度较浅）
- Rebuilder lr = $2 \times 10^{-4}$
- 单卡 NVIDIA RTX 3090 训练

## 实验关键数据

### 主实验

| 数据集 | 指标 (mIoU) | +MRM (在MIC上) | MIC 原始 | 提升 |
|--------|------|------|----------|------|
| GTA → Cityscapes | mIoU | **77.5** | 75.9 | +1.6 |
| Synthia → Cityscapes | mIoU | **68.1** | 67.3 | +0.8 |

| Baseline | GTA→City 原始 | +MRM | 提升 |
|---------|------|------|------|
| DACS | 52.1 | 55.9 | **+3.8** |
| DAFormer | 68.3 | 70.3 | **+2.0** |
| HRDA | 73.8 | 75.4 | **+1.6** |
| MIC | 75.9 | 77.5 | **+1.6** |

### 消融实验

| 配置 | mIoU | 说明 |
|------|---------|------|
| 掩码率 20% | 54.3 | 信息过多，重建任务太简单 |
| 掩码率 40% | **55.9** | 最优 |
| 掩码率 60% | 55.2 | 重建信号多样性下降 |
| 掩码率 80% | 54.1 | 过度掩码伤害语义一致性 |
| Transformer 块数 n=1 | 55.4 | 仅 1 块已有效 |
| Transformer 块数 n=2 | **55.9** | 最优 |
| Transformer 块数 n=4 | 55.6 | 过多块无进一步收益 |

### 关键发现

- MRM 是**模型无关**的：在 4 种不同 baseline (DACS/DAFormer/HRDA/MIC) 上均有一致提升
- MIC + MRM 达到 77.5 mIoU，超过此前所有 GTA→Cityscapes 的 SOTA (+1.4)
- 在 Synthia→Cityscapes 上同样有效（平均 +2.8 mIoU），说明跨域稳定
- 在细粒度类别（traffic sign, rider, motorbike）上提升尤为明显，说明 MRM 增强了解码器的高层语义判别
- 最优掩码率 40% 显著低于标准 MIM (60-75%)，反映了 MRM 在特征空间操作的特殊性

## 亮点与洞察

- **概念上的关键贡献**：将 MIM 从输入空间搬到特征空间，一举解决了与分割架构的兼容性和优化冲突两个难题
- **真正的即插即用**：Rebuilder 仅在训练时使用，推理时完全移除，零额外开销——这是实际部署中非常重要的属性
- MRM 与对比学习互补：对比学习增强编码器，MRM 同时增强编码器+解码器
- 轻量设计：仅 1-2 个 Transformer 块，证明特征空间重建不需要深度模型

## 局限与展望

- 依赖伪标签质量，伪标签噪声可能限制 MRM 的上限
- 多尺度适配方案虽然高效，但仅使用最后一层特征进行重建，可能丢失浅层细节
- 未探索与对比学习的联合使用——两者互补性值得验证
- 掩码策略为简单均匀随机，未利用语义或域特性指导掩码位置

## 相关工作与启发

- **MAE/ConvNeXtV2**：经典 MIM 方法，在输入空间重建，无法直接用于分割
- **PiPa/GANDA/QuadMix**：近期 SOTA UDA 方法，MRM 可作为辅助任务进一步增强它们
- **IFVD/CWD/CIRKD**：分割 KD 方法，主要增强编码器；MRM 独特之处在于同时增强解码器
- 启发：特征空间重建的思想可推广到其他密集预测任务（检测、深度估计）的域自适应

## 评分

- 新颖性: ⭐⭐⭐⭐ 将掩码建模从输入空间转移到表示空间，概念清晰、创新点明确
- 实验充分度: ⭐⭐⭐⭐ 两大 benchmark、4 种 baseline、详细消融
- 写作质量: ⭐⭐⭐⭐ 三种辅助任务的对比图非常直观，方法描述清楚
- 价值: ⭐⭐⭐⭐ 简洁有效的即插即用策略，对 UDA 分割领域有实用价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation](seeing_beyond_extrapolative_domain_adaptive_panoramic_segmentation.md)
- [\[CVPR 2026\] Heuristic Self-Paced Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions](heuristic_self-paced_learning_for_domain_adaptive_semantic_segmentation_under_ad.md)
- [\[CVPR 2026\] SARMAE: Masked Autoencoder for SAR Representation Learning](sarmae_masked_autoencoder_for_sar_representation_learning.md)
- [\[CVPR 2026\] REL-SF4PASS: Panoramic Semantic Segmentation with REL Depth Representation and Spherical Fusion](rel-sf4pass_panoramic_semantic_segmentation_with_rel_depth_representation_and_sp.md)
- [\[CVPR 2026\] Combining Boundary Supervision and Segment-Level Regularization for Fine-Grained Action Segmentation](boundary_segment_action_segmentation.md)

<!-- RELATED:END -->
