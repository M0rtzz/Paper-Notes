---
title: >-
  [论文解读] A3GS: Arbitrary Artistic Style into Arbitrary 3D Gaussian Splatting
description: >-
  [ICCV 2025][3D视觉][3D Stylization] 提出A³GS，首个前馈式零样本3DGS风格迁移网络，使用GCN自编码器+AdaIN注入任意风格，仅需10秒完成任意3D场景风格迁移。
tags:
  - ICCV 2025
  - 3D视觉
  - 3D Stylization
  - Zero-Shot
  - Feed-Forward
  - GCN
  - AdaIN
  - 3DGS
---

# A3GS: Arbitrary Artistic Style into Arbitrary 3D Gaussian Splatting

**会议**: ICCV 2025  
**arXiv**: 无  
**代码**: 无  
**领域**: 3D视觉 / 3D风格迁移  
**关键词**: 3DGS风格迁移, 零样本, 前馈网络, 图卷积网络, AdaIN

## 一句话总结
提出A3GS——首个前馈式零样本3DGS风格迁移框架，通过GCN自编码器将3DGS场景编码到潜在空间并用AdaIN注入任意风格特征，仅需10秒即可完成任意风格到任意3D场景的迁移，速度比优化方法快两个数量级。

## 研究背景与动机

**领域现状**：3D场景风格迁移是元宇宙、游戏等领域的重要需求。随着3D高斯溅射（3DGS）的出现，StylizedGS、G-Style等方法利用梯度优化对3DGS进行风格迁移，实现了实时渲染和较好的风格化效果。

**现有痛点**：基于优化的方法有两个严重问题：（1）每个场景和每种风格都需要几分钟到几小时的优化，无法快速切换风格；（2）随着场景中高斯基元数量增加，GPU显存需求线性增长，无法处理大规模场景——例如StyleGaussian在超过30万个高斯基元时就会显存溢出。

**核心矛盾**：风格迁移的质量依赖于对3D空间结构特征的充分理解，但高效的前馈方法又难以在非结构化的3DGS数据上提取有效的3D局部特征。单个高斯基元的信息不足以捕获3D结构风格特征，需要多个局部高斯基元的协同颜色分布。

**本文目标**：设计一个前馈网络，实现零样本（zero-shot）3DGS风格迁移——对任意新场景和新风格无需额外训练或优化。

**切入角度**：作者观察到，2D图像风格迁移中CNN可以在特征空间进行高效的零样本风格注入。类比地，如果能设计一种适合3DGS非结构化数据的特征提取网络，就能在3D特征空间实现类似的风格迁移。

**核心 idea**：用3D图卷积网络（3D-GCN）自编码器将3DGS场景中局部高斯基元聚合编码到潜在空间，在潜在空间用AdaIN注入风格图像的VGG特征，再解码回3DGS颜色，实现快速前馈式风格迁移。

## 方法详解

### 整体框架
输入为一个3DGS场景（N个高斯基元，每个有位置、旋转、缩放、不透明度和颜色属性）和一张任意风格参考图像。方法固定高斯基元的几何属性和不透明度，只修改颜色。Pipeline包含三个阶段：（1）GCN编码器将高斯基元颜色编码到潜在空间；（2）AdaIN风格注入器在潜在空间对准内容特征和风格特征；（3）GCN解码器将风格化特征解码回每个高斯基元的颜色。整个过程约10秒完成。

### 关键设计

1. **3D图卷积层（3D-GCN Layer）**:

    - 功能：从非结构化的3DGS数据中提取局部3D空间特征
    - 核心思路：为每个高斯基元定义基于空间距离的M个最近邻作为感受野。引入具有可学习形状和权重的卷积核 $K^S = \{k_C, k_s\}_{s=1}^S$，其中核心点 $k_C = (0,0,0)$，各支撑点 $k_s \in \mathbb{R}^3$ 可学习。这种自适应核形状使得网络能根据不同的3D几何结构调整卷积操作，类似于2D CNN中可变形卷积的思想
    - 设计动机：普通GCN缺乏3D空间感知能力（消融实验证实），MLP则完全没有局部特征聚合。3D-GCN的可学习核形状能更好地适配不规则3D高斯分布，解决了传统点云网络对几何变换敏感的问题

2. **GCN自编码器（GCN-based Autoencoder）**:

    - 功能：将大规模3DGS场景压缩到紧凑的潜在空间，同时保留局部聚合的3D结构信息
    - 核心思路：编码器交替使用3D图卷积层和池化层，池化层通过通道维max pooling聚合感受野内特征，再以采样率r进行下采样：$\text{Encoder}(P, C) = (\tilde{P}, F_c)$。解码器使用同类型卷积层，配合反距离加权插值进行上采样，将潜在特征映射回所有高斯基元的颜色：$\text{Decoder}(\tilde{P}, F_{cs}) = (P, C')$
    - 设计动机：直接在每个高斯基元上做风格迁移代价太高且效果差，因为单个基元信息不足以捕获风格所需的区域性颜色分布模式。自编码器将局部聚类的结构和颜色信息压缩到潜在向量中，为后续风格注入提供了合适的特征空间

3. **AdaIN风格注入器（AdaIN-based Stylizer）**:

    - 功能：在潜在空间中将风格图像的统计特征注入到3DGS内容特征中
    - 核心思路：由于3D内容特征空间和2D图像风格特征空间不对齐，先用MLP $\phi$ 将3D内容特征映射到图像特征空间，再用AdaIN对齐均值和方差：$F_{cs} = \psi(\text{AdaIN}(\phi(F_c), F_s))$，最后用MLP $\psi$ 映射回3D特征空间。风格特征 $F_s$ 由VGG网络从参考图像提取
    - 设计动机：消融实验显示，如果不做空间映射直接AdaIN会导致风格化失败；如果在每一层都应用AdaIN则会丢失细粒度细节。在潜在空间中间层做一次全局AdaIN是最佳平衡点

### 损失函数 / 训练策略
采用两阶段训练策略：

**第一阶段——自编码器训练**：冻结风格模块，用RGB重建损失 $L_{rgb} = \frac{1}{N}\sum_{i=1}^{N}(c_i - \hat{c}_i)^2$ 训练编码器和解码器学习有效的3DGS特征提取和颜色重建。

**第二阶段——风格化训练**：冻结自编码器，引入风格模块。随机选取3DGS场景和风格图像，渲染多视角图像后计算风格损失 $L_{style} = L_c + \lambda L_s$（内容损失+加权风格损失）。引入背景掩码机制过滤背景特征，防止背景颜色影响VGG特征的均值方差计算导致前景颜色偏移。

**数据集**：从Objaverse子集生成40,000个不同的3DGS物体（用TriplaneGaussian单图生成），风格图像使用WikiArt的90,000张艺术图片。

## 实验关键数据

### 主实验

| 指标 | StyleRF | StyleSplat | StyleGaussian | A3GS (Ours) |
|------|---------|------------|---------------|-------------|
| 短程LPIPS↓ | 0.092 | 0.033 | 0.035 | **0.033** |
| 短程RMSE↓ | 0.084 | 0.045 | **0.033** | 0.029 |
| 长程LPIPS↓ | 0.154 | 0.052 | 0.057 | 0.061 |
| 长程RMSE↓ | 0.186 | 0.062 | 0.060 | **0.057** |

| 指标 | StyleRF | StyleSplat | StyleGaussian | A3GS (Ours) |
|------|---------|------------|---------------|-------------|
| 风格一致性↑ | 2.1 | 3.9 | 3.5 | **4.1** |
| 内容保持↑ | 3.7 | **4.5** | 3.3 | **4.5** |
| 视觉自然度↑ | 3.5 | 4.3 | 3.6 | **4.5** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| Full model (3D-GCN + AdaIN + Mask) | 高质量风格化 | 完整模型 |
| MLP替代GCN | 产生伪影 | 缺乏局部特征感知 |
| 简单GCN替代3D-GCN | 内容结构丢失 | 无法有效提取3D结构特征 |
| 去掉AdaIN | 仅色调变化 | 无真正风格迁移，类似滤镜效果 |
| 去掉背景掩码 | 严重色偏 | 背景影响VGG特征统计量 |
| AdaIN应用于每层 | 细节丢失 | 过度风格化破坏细粒度结构 |

### 关键发现
- 3D-GCN的可学习核形状是关键——普通GCN和MLP都无法在非结构化3DGS上提取有效的局部3D特征
- A3GS在TNT数据集上对典型场景仅需10秒，是StyleSplat的30倍加速。当高斯基元超过4800万时StyleSplat显存溢出，而A3GS可通过分批处理（每批最多1500万基元）处理任意大规模场景
- 背景掩码对训练质量至关重要，因为Objaverse生成的物体不占满整个图像

## 亮点与洞察
- **GCN处理3DGS的范式**：将3DGS基元视为增强版点云，用图卷积聚合局部特征，这个思路可以推广到3DGS的编辑、压缩、语义分割等其他下游任务
- **两阶段训练策略**：先自编码器后风格化的解耦训练避免了两个目标的干扰，是处理"特征空间学习+特征空间操作"类问题的通用范式
- **分批处理突破规模限制**：利用前馈网络局部处理的特性，将大场景分批处理，彻底解决了优化方法的显存瓶颈

## 局限与展望
- 对训练分布外的罕见/极端艺术风格效果可能不佳，泛化能力有限
- 依赖局部特征聚合，可能在需要全局场景级上下文的情况下出现不一致
- 潜在空间变换过程中可能丢失精细的风格细节
- 未来可考虑引入全局注意力机制或多尺度风格注入来提升全局一致性

## 相关工作与启发
- **vs StyleGaussian**: StyleGaussian也是前馈方法但需要对每个场景训练数小时，且受限于场景规模。A3GS通过GCN自编码器实现了真正的零样本迁移
- **vs StyleSplat/G-Style**: 这些优化方法每个风格需要数百秒优化。A3GS速度快两个数量级，且无理论上的场景规模限制
- **vs 2D风格迁移**: A3GS的核心思路是将2D AdaIN范式推广到3D——关键创新在于用3D-GCN替代CNN解决了非结构化数据的特征提取问题

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个真正零样本前馈3DGS风格迁移，GCN处理3DGS的方案有创新性
- 实验充分度: ⭐⭐⭐⭐ 定量对比、用户研究、消融实验齐全，速度对比很有说服力
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示直观，方法描述完整
- 价值: ⭐⭐⭐⭐ 解决了3DGS风格迁移的效率瓶颈，对元宇宙/游戏等实时应用有直接价值

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] Arbitrary-Scale 3D Gaussian Super-Resolution](../../AAAI2026/3d_vision/arbitrary-scale_3d_gaussian_super-resolution.md)
- [\[ICCV 2025\] Tune-Your-Style: Intensity-Tunable 3D Style Transfer with Gaussian Splatting](tune-your-style_intensity-tunable_3d_style_transfer_with_gaussian_splatting.md)
- [\[NeurIPS 2025\] Styl3R: Instant 3D Stylized Reconstruction for Arbitrary Scenes and Styles](../../NeurIPS2025/3d_vision/styl3r_instant_3d_stylized_reconstruction_for_arbitrary_scenes_and_styles.md)
- [\[ICCV 2025\] Curve-Aware Gaussian Splatting for 3D Parametric Curve Reconstruction](curve-aware_gaussian_splatting_for_3d_parametric_curve_reconstruction.md)
- [\[ICCV 2025\] Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction](robust_and_efficient_3d_gaussian_splatting_for_urban_scene_reconstruction.md)

<!-- RELATED:END -->
