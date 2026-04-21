---
title: >-
  [论文解读] Learning Efficient Fuse-and-Refine for Feed-Forward 3D Gaussian Splatting
description: >-
  [NeurIPS 2025][3D视觉][3DGS] 提出Fuse-and-Refine模块，通过混合Splat-Voxel表征将像素对齐的高斯基元聚合到粗到细的体素层次结构中，用稀疏体素Transformer在15ms内融合约20万基元并提升约2dB PSNR，且仅在静态场景训练即可零样本泛化到流式动态场景重建。
tags:
  - NeurIPS 2025
  - 3D视觉
  - 3DGS
  - Splat-Voxel
  - 前馈重建
  - 流式动态场景
  - 稀疏视图
---

# Learning Efficient Fuse-and-Refine for Feed-Forward 3D Gaussian Splatting

**会议**: NeurIPS 2025  
**arXiv**: [2503.14698](https://arxiv.org/abs/2503.14698)  
**代码**: [GitHub](https://19reborn.github.io/SplatVoxel/)  
**领域**: 3D视觉 / 前馈式3DGS  
**关键词**: 3DGS, Splat-Voxel, 前馈重建, 流式动态场景, 稀疏视图

## 一句话总结

提出Fuse-and-Refine模块，通过混合Splat-Voxel表征将像素对齐的高斯基元聚合到粗到细的体素层次结构中，用稀疏体素Transformer在15ms内融合约20万基元并提升约2dB PSNR，且仅在静态场景训练即可零样本泛化到流式动态场景重建。

## 研究背景与动机

前馈式3DGS方法（如PixelSplat、GS-LRM）通过大规模训练直接从稀疏视图预测高斯基元，实现了高效场景重建。然而这类方法存在三个核心问题：

**冗余性**：基元与输入图像像素对齐，当输入视图重叠时产生大量冗余基元

**位置受限**：基元被约束在输入射线上，无法在3D空间中自由放置，导致重建分辨率受限

**动态扩展困难**：像素对齐的设计无法自然处理时序帧间的冗余合并和新内容补充

已有的启发式基元合并方法（如层次化高斯）通常降低重建质量且需要额外优化。本文的核心思路是：**用学习的方式在规范3D空间中融合和精化基元，用体素作为结构化中间表征**。

## 方法详解

### 整体框架

输入：来自前馈3DGS模型（如GS-LRM）的像素对齐高斯基元集合 → Splat-to-Voxel转换 → 粗到细体素层次构建 → 稀疏体素Transformer处理 → 输出精化后的高斯基元

### 关键设计

1. **Splat-to-Voxel转换**:

    - **Splat沉积**：对于密集高分辨率体素网格，每个高斯基元根据位置用距离核函数沉积到最近的8个体素中，权重通过归一化核函数计算
    - **Splat融合**：每个体素用不透明度加权累加沉积的基元特征和属性，实现基于物理意义的特征聚合
    - 这一过程将点云式的无序基元转化为结构化的体素表征

2. **粗到细体素层次结构**:

    - 高分辨率体素按 $[d,h,w]$ 因子下采样生成低分辨率粗体素
    - 粗体素特征由对应的细体素特征经浅层MLP生成
    - **体素稀疏化**：按体素权重排序，仅保留top 20%的粗体素，将Transformer输入token数降到约1万
    - 这是效率的关键：粗级别做全局注意力（token数少），细级别做局部精化（保留细节）

3. **稀疏体素Transformer**:

    - 将粗体素特征作为1D token序列，送入Transformer进行全局上下文建模
    - 处理后的特征复制回细体素网格，再经浅层MLP结合初始细体素属性生成精化的高斯基元
    - 整个Fuse-and-Refine模块仅需15ms处理约20万基元

4. **零样本流式动态场景重建**:

    - **3D形变**：利用预训练2D点跟踪模型获取跨帧对应关系，通过三角化恢复3D运动
    - 使用嵌入式形变图(Embedded Deformation Graph)将稀疏锚点的运动传播到整个场景
    - **错误感知融合**：将历史关键帧的变形基元与当前帧新基元一起送入Splat-Voxel表征，用基于渲染误差的自适应权重过滤变形误差
    - 仅维护关键帧基元，旧关键帧被替换时平滑过渡

### 损失函数 / 训练策略

- 光度损失：MSE + λ·LPIPS，λ为多视图Transformer时取0.5，体素Transformer时取4.0
- 两阶段训练：先训练多视图Transformer 200K迭代，再联合微调体素Transformer 100K迭代
- 训练数据：DL3DV大规模场景数据集，batch size 128，共300K迭代
- 仅在静态场景上训练，动态场景推理无需额外训练

## 实验关键数据

### 主实验

| 数据集 | 指标 | SplatVoxel | GS-LRM (之前SOTA) | 提升 |
|--------|------|-----------|-------------------|------|
| RealEstate10K | PSNR↑ | **28.47** | 28.10 | +0.37 |
| RealEstate10K | SSIM↑ | **0.907** | 0.892 | +0.015 |
| RealEstate10K | LPIPS↓ | **0.078** | 0.114 | -31.6% |
| DL3DV | PSNR↑ | **30.61** | 28.59 | +2.02 |
| DL3DV | SSIM↑ | **0.935** | 0.925 | +0.010 |
| Neural3DVideo (流式) | PSNR↑ | **27.41** | 21.80 | +5.61 |
| Neural3DVideo (流式) | Flicker↓ | **2.916** | 5.714 | -49% |

### 消融实验

| 配置 | PSNR↑ | 时间(ms) | 说明 |
|------|-------|---------|------|
| GS-LRM基线 | 28.59 | 52.8 | 24层多视图Transformer |
| + 非学习融合 | 12.57 | 33.8 | 启发式合并导致严重退化 |
| + 无粗到细 | 20.62 | 48.6 | 全分辨率Transformer不可行 |
| + 无稀疏化 | 29.69 | 72.0 | 计算量太大 |
| + 3D CNN替代Transformer | 29.44 | 82.1 | 全局注意力优于局部卷积 |
| + 无Splat特征 | 29.40 | 49.2 | 特征信息重要 |
| + 完整方法 | **30.61** | **52.5** | 所有设计互相配合 |

### 关键发现

- 非学习的启发式融合方法导致灾难性退化(PSNR从28.59降到12.57)，验证了学习式融合的必要性
- 粗到细层次结构实现了质量和效率的最佳平衡——无它PSNR仅20.62
- 在流式动态场景上PSNR提升巨大(+5.61dB)，且闪烁大幅降低，说明历史信息融合非常有效
- 仅作4视图训练，可泛化到2/8/16视图，且在16视图时优势更大

## 亮点与洞察

- **Splat-Voxel混合表征**：将点云的灵活性与体素的结构性完美结合，既适合聚合也适合Transformer处理
- **粗到细+稀疏化**：在粗级别做全局注意力、细级别做局部精化的设计，是效率与质量的优雅平衡
- **静态训练、动态零样本**：核心模块（融合+精化）的能力是通用的，不需要动态场景数据即可泛化
- **15ms推理速度**：200K基元的融合精化仅需15ms，使得交互级(15fps)流式重建成为可能

## 局限与展望

- 流式重建依赖2D点跟踪模型（如TAPIR）的质量，跟踪失败会影响形变精度
- 体素分辨率和稀疏化比例是超参数，需要针对不同场景调整
- 当前仅在稀疏视图(2-4视图)设置下验证，密集视图场景的效果未知

## 相关工作与启发

- **vs GS-LRM**: GS-LRM直接从多视图Transformer预测像素对齐基元，缺乏3D空间的全局优化；SplatVoxel在此基础上加入3D融合精化
- **vs 4DGS**: 4DGS需要逐场景优化动态场景，SplatVoxel零样本泛化且运行速度快得多
- **vs EvolvSplat/S-Cube**: 这些方法也使用体素表征处理3DGS，但未扩展到动态场景且需要特定训练

## 评分

- 新颖性: ⭐⭐⭐⭐ Splat-Voxel混合设计和粗到细层次结构设计巧妙，零样本动态泛化是亮点
- 实验充分度: ⭐⭐⭐⭐⭐ 静态/动态场景全面评估，消融详细，不同视图数量的泛化实验充分
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图表质量高，论文结构合理
- 价值: ⭐⭐⭐⭐⭐ 为前馈式3DGS的基元融合提供了通用解决方案，流式重建应用前景广阔

<!-- RELATED:START -->

## 相关论文

- [Plana3R: Zero-shot Metric Planar 3D Reconstruction via Feed-Forward Planar Splatting](plana3r_zero-shot_metric_planar_3d_reconstruction_via_feed-forward_planar_splatt.md)
- [AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors](../../CVPR2026/3d_vision/anchorsplat_feed-forward_3d_gaussian_splatting_with_3d_geometric_priors.md)
- [ZPressor: Bottleneck-Aware Compression for Scalable Feed-Forward 3DGS](zpressor_bottleneck-aware_compression_for_scalable_feed-forward_3dgs.md)
- [SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting](../../CVPR2026/3d_vision/sr3r_rethinking_super-resolution_3d_reconstruction_with_feed-forward_gaussian_sp.md)
- [Fin3R: Fine-tuning Feed-forward 3D Reconstruction Models via Monocular Knowledge Distillation](fin3r_fine-tuning_feed-forward_3d_reconstruction_models_via_monocular_knowledge_.md)

<!-- RELATED:END -->
