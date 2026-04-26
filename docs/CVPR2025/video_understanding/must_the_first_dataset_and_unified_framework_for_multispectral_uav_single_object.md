---
title: >-
  [论文解读] MUST: The First Dataset and Unified Framework for Multispectral UAV Single Object Tracking
description: >-
  [CVPR 2025][视频理解][多光谱跟踪] 提出首个大规模多光谱无人机单目标跟踪数据集 MUST（250 序列、43K 帧、8 光谱波段），并设计 UNTrack 统一框架融合光谱、空间、时序特征，通过非对称 Transformer 和光谱提示编码器实现高效鲁棒跟踪。
tags:
  - CVPR 2025
  - 视频理解
  - 多光谱跟踪
  - 无人机目标跟踪
  - Transformer
  - 光谱提示
  - 背景消除
---

# MUST: The First Dataset and Unified Framework for Multispectral UAV Single Object Tracking

**会议**: CVPR 2025  
**arXiv**: [2503.17699](https://arxiv.org/abs/2503.17699)  
**代码**: https://github.com/q2479036243/MUST-Multispectral-UAV-Single-Object-Tracking (有)  
**领域**: 视频理解  
**关键词**: 多光谱跟踪, 无人机目标跟踪, 非对称Transformer, 光谱提示, 背景消除

## 一句话总结

提出首个大规模多光谱无人机单目标跟踪数据集 MUST（250 序列、43K 帧、8 光谱波段），并设计 UNTrack 统一框架融合光谱、空间、时序特征，通过非对称 Transformer 和光谱提示编码器实现高效鲁棒跟踪。

## 研究背景与动机

无人机跟踪面临独特挑战：
1. **小目标问题**：无人机飞行高度 20-250m，目标尺度仅占图像 $3 \times 10^{-5}$
2. **RGB 特征不足**：背景杂乱、颜色相似、遮挡等场景下，空间特征（颜色、纹理）无法有效区分目标和背景
3. **外观剧烈变化**：无人机视角变化大，目标外观可能与初始模板差异显著

**多光谱图像（MSI）的优势**：光谱信息反映目标的内在反射特性，即使空间特征相似，不同物质的光谱曲线差异显著，可用于有效区分。且光谱特征在跟踪过程中相对稳定。

**现有问题**：
- 缺乏大规模、高挑战度的多光谱无人机跟踪数据集
- 现有 MSI 跟踪数据集（HOT，仅 50 序列）规模小且为地面场景
- 现有无人机跟踪器多基于 Siamese 架构且仅处理 RGB 数据
- 已有方法将 RGB 跟踪器简单适配到 MSI，未充分挖掘光谱特征

## 方法详解

### 整体框架

UNTrack 统一处理三种输入：光谱提示（spectrum prompt）、初始模板（initial template）、连续搜索帧（sequential searches）。核心由三部分组成：(1) 统一非对称 Transformer 提取特征并建模关系；(2) 光谱提示编码器生成和更新光谱特征提示；(3) 双分支预测头输出目标位置。

### 关键设计

1. **统一非对称 Transformer（Unified Asymmetric Transformer）**：将光谱提示 P、模板 T、搜索帧 S 的 token 统一拼接为 $F = [P; T; S]$，在全局注意力中交互。但核心创新在于**非对称注意力**——分析发现完整 $9$ 块注意力图中，仅 P/T/S 各自的自注意力（块 1,5,9）和 S 对 P+T 的交叉注意力（块 7,8）对跟踪有用，而 P/T 间的交叉注意力不仅无用还引入噪声。因此剪枝为：P 自注意力 + T 自注意力 + S 对全体的注意力，大幅减少计算（FLOPs 降低 19%）同时提升 AUC 4.1%。

2. **光谱背景消除机制（Spectral Background Eliminate）**：利用搜索 token 的注意力图 $A_S$ 量化每个区域属于目标的置信度 $B = \|\text{AvgPool}(A_S)\|$，仅保留 top-$\rho$ 区域，渐进消除背景区域。$\rho$ 使用余弦退火策略在训练中动态调整。**效果**：进一步降低计算开销（106.9G vs 137.5G FLOPs），推理速度从 24.3 → 38.0 FPS。

3. **光谱提示编码器（Spectrum Prompt Encoder）**：将非对称 Transformer 输出的提示 token $\bar{P}$ 和模板全局池化 $\text{GPool}(\bar{T})$ 拼接，经两层 FC（压缩-激发操作，类似 SE 注意力）和 MLP，编码得到新的光谱提示 $\hat{P}$，记录目标的材质光谱特性。**核心**：光谱提示在帧间持续更新，为后续跟踪提供稳定的、不受外部干扰的光谱参考。

### 损失函数 / 训练策略

- **分类**：Focal Loss $\mathcal{L}_{cls}$
- **回归**：$\ell_1$ Loss + GIoU Loss：$\mathcal{L} = \mathcal{L}_{cls} + 5 \mathcal{L}_1 + 2 \mathcal{L}_{GIoU}$
- AdamW 优化器，初始学习率 $10^{-4}$，30 epoch 后衰减 10 倍，共 50 epoch
- 搜索帧 $384 \times 384$，模板 $192 \times 192$，batch size 24
- **参数重建策略**：将 ImageNet RGB 预训练参数通过插值扩展到 8 通道 MSI（平均提升 10%+ AUC）
- 连续搜索帧数设为 2（精度-效率最佳平衡）

## 实验关键数据

### 主实验 — MUST 数据集

| 方法 | 类型 | AUC ↑ | SR₀.₅ ↑ | Pre ↑ | PreN ↑ |
|------|------|-------|---------|-------|--------|
| SiamRPN | Siamese | 38.9 | 49.5 | 56.7 | 50.1 |
| OSTrack₃₈₄ | One-stream | 44.5 | 56.7 | 63.9 | 56.8 |
| ODTrack | One-stream | 46.3 | 58.4 | 67.6 | 60.4 |
| OSTrack*₃₈₄ | +参数重建 | 55.1 | 69.6 | 73.3 | 68.8 |
| **UNTrack*** | **+参数重建** | **59.7** | **75.8** | **79.2** | **74.8** |

UNTrack* 比 OSTrack*₃₈₄ 高 **4.6% AUC** 和 **5.9% Pre**。

### 跨数据集验证

| 方法 | HOT AUC ↑ | HOT Pre ↑ | FPS ↑ | GOT10K AO ↑ |
|------|-----------|-----------|-------|-------------|
| SiamHYPER | 67.8 | 94.5 | 27.7 | - |
| HANet | 68.8 | 94.8 | 21.2 | - |
| TMTNet | 69.9 | 92.8 | 12.6 | - |
| **UNTrack** | **70.4** | 93.7 | **37.0** | **77.3** |

HOT 数据集上 SOTA 精度 + 最快速度（37 FPS）。GOT10K 上与 SOTA 持平。

### 消融实验

| 配置 | AUC ↑ | FLOPs | FPS ↑ | 说明 |
|------|-------|-------|-------|------|
| 无 Prompt (基线) | 53.8 | 140.9G | 23.9 | 仅 T+S 自注意力 |
| + Prompt 全注意力 | 55.4 | 169.8G | 21.4 | 引入光谱提示但计算量大 |
| + 非对称注意力 | 59.5 | 137.5G | 24.3 | 剪枝后大幅提升 |
| + 背景消除 | **59.7** | **106.9G** | **38.0** | 保持精度，效率提升 58% |

| 光谱提示来源 | AUC ↑ | 参数量 | 说明 |
|-------------|-------|--------|------|
| 无提示 | 53.9 | 93.1M | 基线 |
| 随机初始化（不更新） | 53.7 | 101.3M | 引入无关信息 |
| 非对称 Transformer 输出 | 55.0 | 101.3M | 历史更新有效 |
| **光谱提示编码器** | **59.7** | 112.6M | 充分利用光谱特征 |

### 关键发现

- 参数重建策略（RGB→MSI 插值）平均提升各跟踪器 10%+ AUC，成本极低
- 非对称注意力剪枝同时降低 19% FLOPs 并提升 4.1% AUC，说明剪除的注意力块确实是噪声
- 连续 2 帧搜索比单帧提升 5.1% AUC，但 3 帧以上不再提升且计算翻倍
- 光谱信息在运动模糊（MB）、相机移动（CM）、相似颜色（SC）等挑战中优势最明显
- UNTrack 在目标短暂丢失（OV）后可通过历史提示重新定位，优于其他跟踪器

## 亮点与洞察

- **首个多光谱无人机跟踪数据集**：250 序列、43K 帧、8 波段（390-950nm 含可见光+近红外），12 种挑战属性，填补领域空白
- **非对称注意力的思路精巧**：通过分析注意力图的 9 宫格发现跟踪无关块，剪枝同时提速提精度
- **光谱提示的持续更新机制**：类似 memory bank 但更轻量，仅 1 个 token 即可编码目标光谱特性
- 参数重建策略简单实用：仅通过插值即可将 RGB 预训练扩展到 MSI

## 局限与展望

- 多光谱相机帧率仅 5 FPS，限制了高速运动场景
- 8 个光谱波段有限，更多波段可能带来更好表征
- 数据集仅 250 序列，与 RGB 跟踪数据集规模差距大
- 无 MSI 预训练参数，依赖 RGB→MSI 插值策略
- 未探讨多目标跟踪或光谱异常检测等扩展任务

## 相关工作与启发

- OSTrack / ODTrack：one-stream 跟踪器代表，是主要对比基线
- HOT 数据集（50 序列）：唯一现有 MSI 跟踪数据集，规模太小
- UAV123 / VisDrone：RGB 无人机跟踪数据集，缺少光谱维度
- 本文的非对称注意力思路可推广到其他多模态融合任务（如 RGB-T 跟踪）
- 光谱提示编码器的压缩-激发设计借鉴了 SENet，但应用于光谱维度

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首创数据集 + 专用框架，开辟多光谱无人机跟踪新方向
- **实验充分度**: ⭐⭐⭐⭐⭐ 数据集分析、主实验、跨数据集、多项消融、可视化全面
- **写作质量**: ⭐⭐⭐⭐ 数据集和方法描述详细，注意力分析有图示辅助
- **价值**: ⭐⭐⭐⭐⭐ 数据集和框架对无人机跟踪社区具有基础性贡献

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking](learning_occlusion-robust_vision_transformers_for_real-time_uav_tracking.md)
- [\[CVPR 2026\] UETrack: A Unified and Efficient Framework for Single Object Tracking](../../CVPR2026/video_understanding/uetrack_a_unified_and_efficient_framework_for_single_object_tracking.md)
- [\[CVPR 2025\] Similarity-Guided Layer-Adaptive Vision Transformer for UAV Tracking](similarity-guided_layer-adaptive_vision_transformer_for_uav_tracking.md)
- [\[ICCV 2025\] General Compression Framework for Efficient Transformer Object Tracking](../../ICCV2025/video_understanding/general_compression_framework_for_efficient_transformer_object_tracking.md)
- [\[CVPR 2025\] OmniTrack: Omnidirectional Multi-Object Tracking](omnidirectional_multi-object_tracking.md)

<!-- RELATED:END -->
