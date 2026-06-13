---
title: >-
  [论文解读] Geometry Meets Light: Leveraging Geometric Priors for Universal Photometric Stereo under Limited Multi-Illumination Cues
description: >-
  [AAAI 2026 Oral][3D视觉][光度立体] 提出 GeoUniPS，首次将大规模3D重建模型（VGGT）的几何先验引入通用光度立体网络，通过光照-几何双分支编码器，在多光照线索不足时利用几何先验补偿，同时引入透视投影训练数据集 PS-Perp 弥合正交投影假设与真实场景的差距。 领域现状 光度立体（Photom…
tags:
  - "AAAI 2026 Oral"
  - "3D视觉"
  - "光度立体"
  - "几何先验"
  - "3D重建基础模型"
  - "VGGT"
  - "透视投影"
---

# Geometry Meets Light: Leveraging Geometric Priors for Universal Photometric Stereo under Limited Multi-Illumination Cues

**会议**: AAAI 2026 Oral  
**arXiv**: [2511.13015](https://arxiv.org/abs/2511.13015)  
**代码**: [https://github.com/marcotam2002/geounips](https://github.com/marcotam2002/geounips)  
**领域**: 3D视觉 / 光度立体  
**关键词**: 光度立体, 几何先验, 3D重建基础模型, VGGT, 透视投影

## 一句话总结

提出 GeoUniPS，首次将大规模3D重建模型（VGGT）的几何先验引入通用光度立体网络，通过光照-几何双分支编码器，在多光照线索不足时利用几何先验补偿，同时引入透视投影训练数据集 PS-Perp 弥合正交投影假设与真实场景的差距。

## 研究背景与动机

### 领域现状

光度立体（Photometric Stereo, PS）通过固定相机、变化光照的多张图像恢复高保真法线图。其发展历程是对光照假设的逐步放松：

- **传统PS**：需校准方向光 + Lambertian BRDF假设
- **非校准PS**：去除光照校准要求，但仍假设方向光模型
- **通用PS**（Universal PS）：完全不需要光照模型，直接从图像中学习光照表示

当前SOTA的通用PS方法（SDM-UniPS, LINO-UniPS）已经消除了大部分光照假设。

### 现有痛点

1. **多光照线索不可靠时性能急剧下降**：通用PS从根本上依赖多光照变化作为主要线索。当某些区域光照不足（偏置光照、阴影、自遮挡），或装修复杂的真实室内环境中间接光照占主导时，方法缺乏补偿机制
2. **合成训练数据缺乏真实世界的几何上下文**：获取真实场景的高精度法线GT极其昂贵，因此PS模型只能在干净的合成数据上训练。合成数据的统计分布与真实场景（如"建筑立面通常是分段平面"这类先验）差异巨大
3. **正交投影假设不适用于真实透视相机**：现有训练数据集（PS-Wild, PS-Mix）全部使用正交投影渲染，无法处理透视畸变

### 核心矛盾

如何在**纯合成训练流程**中获取**真实世界的高层几何先验**，以补偿多光照线索不足的情况？

### 本文关键洞察

大规模3D重建模型（如 VGGT）在海量真实数据上预训练，已经隐式编码了丰富的真实场景几何知识。即使输入单张图像，也能恢复合理的3D形状——说明这些模型学到了超越低层多视角光度约束的**高层单目几何先验**。将这些先验注入PS管道，就能在光照不足时提供有意义的几何引导。

## 方法详解

### 整体框架

GeoUniPS采用标准的编码器-解码器两阶段设计：
1. **编码器**：从K张多光照图像中提取K个特征图（替代传统PS中的光照信息）
2. **解码器**：在随机采样的像素位置预测法线

核心创新在编码器端——**光照-几何双分支编码器**。

### 关键设计

#### 1. 光照-几何双分支编码器

**功能**：并行提取光照敏感特征和光照不变的几何先验特征，拼接形成统一表示。

$$\mathcal{F} = \text{Concat}(\mathcal{F}_{\text{Geo}}, \mathcal{F}_{\text{IL}})$$

**几何分支（EncoderGeo）**：
- 使用**冻结的 VGGT aggregator**（24层交替frame/global attention）
- 从层 [4, 11, 17, 23] 提取 token
- 通过可学习的 DPT head 融合为 128-dim 特征图（2×下采样）
- VGGT在MegaDepth, CO3D-v2, ScanNet, DL3DV等大规模真实数据上训练，编码了丰富的场景几何知识
- 冻结参数保留其几何知识不被合成训练数据破坏

**光照分支（EncoderIL）**：
- 类VGGT的Transformer架构（12层），但用轻量卷积代替DINOv2 tokenizer以捕获细粒度局部模式
- 交替使用帧内注意力（frame attention）和**光轴注意力**（light-axis attention，在相同空间位置的不同光照图像间做注意力）
- 从层 [2, 5, 8, 11] 提取并融合为 128-dim特征图
- 从零训练，学习光照变化中的法线线索

**设计动机**：
- 几何先验是光照不变的，能在任何光照条件下提供合理的全局形状估计
- 光照线索在光照充足时提供精细的表面细节
- 两者互补：几何先验补偿光照不足区域，光照线索提供高频细节

#### 2. 双尺度法线解码器

**功能**：在随机采样的像素位置，从特征和原始图像中预测法线。

**核心思路**：两级预测——

**低尺度阶段**：
- 5层 256-dim 光轴Transformer处理编码器特征
- 1层 384-dim 光轴Transformer + PMA 沿光轴聚合
- 2层 384-dim 像素采样Transformer 增强空间一致性
- MLP (384→192→3) 预测低频法线

**高尺度阶段**：
- 将原始RGB值通过 MLP (3→256) 嵌入高维空间
- 与编码器特征拼接后用 5层 256-dim 光轴Transformer处理
- PMA 聚合 + 与低尺度法线融合
- 2层像素采样Transformer + MLP (387→384→192→3) 预测精细法线

**设计动机**：低尺度捕获全局几何结构，高尺度从原始RGB恢复高频表面细节。MLP嵌入RGB比直接使用patch或pixel更高效。

#### 3. PS-Perp 透视投影数据集

**功能**：构建首个透视投影的通用PS合成训练数据集。

**核心思路**：
- 使用Blender Cycles渲染器，焦距从20mm到1000mm采样
- 60,297个多物体场景，44,220个使用<70mm焦距（强透视效应）
- 每场景10张16-bit图像，512×512分辨率
- 随机组合方向光、点光源、环境光
- 与PS-Mix共享资产库和场景组合策略

**设计动机**：SDM-UniPS在15mm焦距下MAE高达22.18°（仅用正交数据训练），而混合训练后降至6.98°。透视投影训练使模型能学习空间变化的视角方向。

### 损失函数 / 训练策略

- **损失**：MSE loss 计算预测与GT法线的 $\ell_2$ 误差，在两个尺度分别计算后求和
- 4×H100 GPU训练6天，AdamW优化器，初始学习率1e-4，每10 epoch衰减0.8
- FP32精度训练，每batch随机采样3-6张输入图像提升鲁棒性
- 训练采样2048像素，推理增至10000像素

## 实验关键数据

### 主实验

#### DiLiGenT 基准 (正交投影，96张图像)

| 方法 | Ball | Bear | Buddha | Cat | Goblet | Harvest | Avg MAE↓ |
|------|------|------|--------|-----|--------|---------|---------|
| SDM-UniPS | 1.45 | 3.50 | 7.54 | 5.19 | 7.69 | 10.76 | 5.80 |
| LINO-UniPS | 1.77 | 2.62 | 6.22 | 3.38 | 5.14 | 8.60 | 4.75 |
| **GeoUniPS** | 2.63 | **2.46** | **5.95** | **3.27** | **5.00** | **8.54** | **4.65** |

#### LUCES 基准 (透视投影，52张图像)

| 方法 | Ball | Die | Hippo | House | Avg MAE↓ |
|------|------|-----|-------|-------|---------|
| SDM-UniPS | 11.77 | 7.22 | 8.95 | 25.91 | 12.80 |
| LINO-UniPS | 9.65 | 6.25 | 5.82 | 22.69 | 9.46 |
| **GeoUniPS** | **7.59** | **3.79** | **5.62** | **21.84** | **9.42** |

### 消融实验

#### 训练数据效果（用SDM-UniPS验证）

| 训练数据 | 15mm MAE | 35mm MAE | 正交 MAE |
|---------|---------|---------|---------|
| PS-Mix（正交） | 22.18 | 14.09 | **5.52** |
| PS-Perp（透视） | 7.18 | 5.47 | 8.95 |
| PS-Perp + PS-Mix | **6.98** | **5.53** | 5.62 |

#### 编码器设计消融

| 配置 | DiLiGenT (K=96) | DiLiGenT (K=1) | LUCES (K=52) |
|------|----------------|----------------|--------------|
| EncoderIL only | 5.81 | 18.40 | 11.03 |
| EncoderGeo only (VGGT) | 6.97 | 11.67 | 11.48 |
| Dual-Branch (VGGT) | **5.75** | **12.86** | **9.42** |
| Dual-Branch (MoGe) | 5.98 | 14.26 | 9.78 |

### 关键发现

1. **K=1（单图像）时几何先验至关重要**：EncoderIL only在K=1时MAE=18.40°，加入几何分支后降至12.86°——多光照线索完全缺失时几何先验发挥补偿作用
2. **纯几何先验不随输入图像数量提升而扩展**：EncoderGeo only即使K增大也改善有限，证明在编码器中提取光照线索不可替代
3. **透视投影训练数据的互补性**：单独用PS-Mix在强透视下MAE=22.18°，混合训练后仅6.98°
4. **VGGT比MoGe做几何backbone更好**：VGGT的多视角训练使其比纯单目深度估计方法学到更强的几何先验
5. **Multi-Illumination数据集的定性结果惊艳**：在地板、墙壁、镜面、透明物体等传统PS完全失败的区域，GeoUniPS仍能生成合理的法线图

## 亮点与洞察

1. **"3D重建模型=视觉-几何基础模型"的洞察深刻**：将多视角3D重建模型视为编码了丰富几何知识的基础模型，而非仅仅是重建工具
2. **冻结+可学习投影头的策略**：保留预训练知识不被小规模合成数据破坏，同时允许适配下游任务
3. **光轴注意力 vs 全注意力的区别**：光照分支用光轴注意力（直觉：同一像素在不同光照下的变化才是关键），几何分支的全注意力移除后影响很小（因为VGGT的几何知识是光照不变的）
4. **PS-Perp数据集的实际价值**：弥合了该领域长期存在的正交-透视投影差距

## 局限与展望

1. **推理速度较慢**：16张512×512图像推理约13秒（H100），实时应用受限
2. **训练时只用3-6张图像**：评估时用更多图像可能因分布偏移导致性能下降
3. **对VGGT的依赖**：若VGGT编码器更新或有更好的3D重建模型出现，需要重新训练投影头
4. 可探索将几何先验引入到视频PS或动态场景PS中
5. 缺少对计算资源需求的详细分析（VGGT本身较大）

## 相关工作与启发

- **SDM-UniPS** (Ikehata, 2023)：通用PS的奠基工作，本文的直接baseline，提出了GLC概念
- **VGGT** (Wang et al., 2025)：前馈3D重建模型，能从单图推理3D形状，本文的几何先验来源
- **DPT** (Ranftl et al., 2021)：Dense Prediction Transformer，用作特征融合的投影头
- **LINO-UniPS** (Li et al., 2025)：通过小波细化和梯度损失提升细节，与本文互补
- **启发**：大规模预训练视觉基础模型的几何知识可以被"低成本"地迁移到依赖光度信号的任务中

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 将3D重建模型作为几何先验引入PS是全新思路，PS-Perp数据集填补空白
- 实验充分度: ⭐⭐⭐⭐ — DiLiGenT+LUCES定量 + Multi-illumination定性 + 消融全面，但缺少更多真实场景定量评估
- 写作质量: ⭐⭐⭐⭐⭐ — 动机论述非常优秀，"为什么需要几何先验"的逻辑链完整
- 价值: ⭐⭐⭐⭐⭐ — 开辟了将基础模型几何知识引入PS的新方向，启发其他依赖光度信号的任务

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Leveraging 3D Geometric Priors in 2D Rotation Symmetry Detection](../../CVPR2025/3d_vision/leveraging_3d_geometric_priors_in_2d_rotation_symmetry_detection.md)
- [\[AAAI 2026\] 3D-Free Meets 3D Priors: Novel View Synthesis from a Single Image with Pretrained Diffusion Guidance](3d-free_meets_3d_priors_novel_view_synthesis_from_a_single_image_with_pretrained.md)
- [\[AAAI 2026\] Generalized Geometry Encoding Volume for Real-time Stereo Matching](generalized_geometry_encoding_volume_for_real-time_stereo_matching.md)
- [\[CVPR 2025\] PS-EIP: Robust Photometric Stereo Based on Event Interval Profile](../../CVPR2025/3d_vision/ps-eip_robust_photometric_stereo_based_on_event_interval_profile.md)
- [\[CVPR 2026\] AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors](../../CVPR2026/3d_vision/anchorsplat_feed-forward_3d_gaussian_splatting_with_3d_geometric_priors.md)

</div>

<!-- RELATED:END -->
