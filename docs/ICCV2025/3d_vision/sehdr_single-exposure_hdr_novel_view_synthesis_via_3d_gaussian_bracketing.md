---
title: >-
  [论文解读] SeHDR: Single-Exposure HDR Novel View Synthesis via 3D Gaussian Bracketing
description: >-
  [ICCV 2025][3D视觉][HDR新视角合成] 提出 SeHDR，首个从单曝光多视角 LDR 图像合成 HDR 新视角的框架，通过在 3D 高斯空间中生成包围曝光（Bracketed 3D Gaussians）并用可微神经曝光融合（NeEF）合并为 HDR 场景表示。 领域现状：HDR 新视角合成（HDR-NVS）从…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "HDR新视角合成"
  - "3D高斯"
  - "曝光包围"
  - "单曝光"
  - "可微渲染"
---

# SeHDR: Single-Exposure HDR Novel View Synthesis via 3D Gaussian Bracketing

**会议**: ICCV 2025  
**arXiv**: [2509.20400](https://arxiv.org/abs/2509.20400)  
**代码**: [https://github.com/yiyulics/SeHDR](https://github.com/yiyulics/SeHDR)  
**领域**: 3D视觉 / HDR成像  
**关键词**: HDR新视角合成, 3D高斯, 曝光包围, 单曝光, 可微渲染

## 一句话总结
提出 SeHDR，首个从单曝光多视角 LDR 图像合成 HDR 新视角的框架，通过在 3D 高斯空间中生成包围曝光（Bracketed 3D Gaussians）并用可微神经曝光融合（NeEF）合并为 HDR 场景表示。

## 研究背景与动机

**领域现状**：HDR 新视角合成（HDR-NVS）从多视角 LDR 图像重建 HDR 场景。现有方法（HDR-NeRF、HDRGS）需要多视角+多曝光输入——不同视角需提供不同曝光的图像来互补过曝/欠曝区域的信息。

**现有痛点**：(1) 多曝光采集在实际中很难——同一视角不同曝光需要完美对齐（运动模糊风险），不同视角不同曝光的相机标定更困难；(2) 直接将现有多曝光 HDR-NVS 方法应用于单曝光输入无效（缺少互补曝光信息）；(3) 将单图 HDR 重建方法逐帧应用后再做 3DGS 会引入多视图不一致性（产生浮点伪影和模糊）。

**核心矛盾**：单曝光输入中过曝/欠曝区域的信息已被量化和饱和截断，是一个病态问题——但多曝光采集在实际中太困难。

**本文目标**：从标准的单曝光多视角 LDR 图像中学习 HDR 场景表示，用于 HDR 新视角合成。

**切入角度**：借鉴计算摄影中的曝光包围（exposure bracketing）技术——在 3D 高斯空间中合成不同曝光的高斯，然后融合为 HDR，避免了在 2D 图像空间做 HDR 重建时的多视图不一致问题。

**核心 idea**：先学习线性颜色空间的基础 3D 高斯 → 通过曝光操控生成多组相同几何不同曝光的"包围曝光 3D 高斯"→ 用可微神经曝光融合（NeEF）在球谐函数空间内融合为 HDR 高斯。

## 方法详解

### 整体框架
单曝光 LDR 多视角输入 → 学习基础 3D 高斯（SH 系数在线性颜色空间中参数化）→ 曝光操控生成包围曝光高斯 → NeEF 在 SH 空间中融合为 HDR 高斯 → HDR 新视角渲染。

### 关键设计

1. **线性颜色空间 3D 高斯**:

    - 功能：将 3DGS 的颜色表示从 sRGB 空间转到线性辐射度空间
    - 核心思路：估计相机响应函数（CRF），将 LDR 输入逆映射到线性辐射度。SH 系数在线性空间中参数化，使得曝光操控（乘以标量）在物理上有意义
    - 设计动机：在 sRGB 空间中的曝光变化是非线性的，不能简单缩放；线性空间中曝光变化等价于标量乘法

2. **包围曝光 3D 高斯 (Bracketed 3D Gaussians)**:

    - 功能：从单曝光基础高斯生成多组不同曝光的 3D 高斯
    - 核心思路：保持高斯的几何属性（位置、协方差）不变，仅调整线性颜色值来模拟不同曝光。采样比输入更高和更低的曝光值，生成多组 bracketed 高斯
    - 设计动机：计算摄影中 exposure bracketing 的 3D 扩展——在 3D 空间操作天然保证多视图一致性，避免了 2D 单图 HDR 方法的多视图不一致问题

3. **可微神经曝光融合 (NeEF)**:

    - 功能：将多组包围曝光高斯融合为一个 HDR 高斯
    - 核心思路：在球谐函数（SH）参数空间内操作——对不同曝光的 SH 系数学习加权融合策略。融合网络直接在 SH 空间中运行，输出 HDR 的 SH 系数。整个过程可微分，端到端训练
    - 设计动机：在 SH 空间而非像素空间融合，保持了 3DGS 的视角依赖性建模能力且更高效

### 损失函数 / 训练策略
不需要 HDR 真值监督。使用渲染的 LDR 图像与输入 LDR 图像之间的光度损失进行端到端训练。

## 实验关键数据

### 主实验

| 方法 | 输入要求 | HDR 质量 (PSNR) | 备注 |
|------|---------|----------------|------|
| SeHDR | 单曝光 | **14.3dB 优于 baseline** | 无需 HDR 真值 |
| HDRGS | 多曝光 | 单曝光时失败 | 依赖多曝光互补 |
| 单图 HDR + 3DGS | 单曝光 | 模糊/浮点伪影 | 多视图不一致 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 完整 SeHDR | 最优 | 线性空间+括号曝光+NeEF |
| 无线性化 | 下降 | sRGB 空间曝光操控不合理 |
| 直接 SH 拼接（无 NeEF） | 下降 | 简单拼接不如学习性融合 |
| 不同括号曝光数量 | 适中更优 | 过多曝光收益递减 |

### 关键发现
- 在 3D 空间做曝光包围天然解决了多视图一致性问题——这是相比 2D 单图 HDR 方法的核心优势
- 无需 HDR 真值的自监督训练是一大优势，仅靠输入 LDR 图像的光度损失即可训练
- 线性颜色空间是正确处理曝光变化的关键前提

## 亮点与洞察
- **Exposure bracketing 的 3D 推广**是非常自然且巧妙的想法——将计算摄影的经典概念提升到 3D 高斯空间
- 对任何标准的多视图数据集都可直接应用，无需特殊的多曝光采集——大幅降低了 HDR-NVS 的使用门槛
- 在 SH 空间做融合保持了视角依赖的光照建模能力

## 局限与展望
- CRF 估计的精度会影响线性化质量
- 对极度过曝/欠曝区域（信息完全丢失）的恢复能力有限
- 需要 3DGS 的标准训练条件（多视角、SfM 位姿估计）
- 目前仅处理静态场景

## 相关工作与启发
- **vs HDR-NeRF**: 需要多曝光输入+隐式表示；SeHDR 单曝光+显式 3DGS
- **vs HDRGS (Cai et al.)**: 同为 3DGS 但需要多曝光；SeHDR 首次实现单曝光 HDR-NVS
- **vs 单图 HDR 方法 (LDR2HDR等)**: 2D 方法导致多视图不一致；SeHDR 在 3D 空间操作避免了此问题

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个单曝光 HDR-NVS 框架，exposure bracketing 的 3D 推广概念新颖
- 实验充分度: ⭐⭐⭐⭐ 与多种方法和精心设计的 baseline 对比，14.3dB 提升显著
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法流程自然流畅
- 价值: ⭐⭐⭐⭐⭐ 大幅降低 HDR-NVS 的采集门槛，适用于任何标准多视图数据集

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] High Dynamic Range Novel View Synthesis with Single Exposure](../../ICML2025/3d_vision/high_dynamic_range_novel_view_synthesis_with_single_exposure.md)
- [\[CVPR 2026\] Physically Inspired Gaussian Splatting for HDR Novel View Synthesis](../../CVPR2026/3d_vision/physically_inspired_gaussian_splatting_for_hdr_novel_view_synthesis.md)
- [\[ICCV 2025\] Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)
- [\[ICCV 2025\] BillBoard Splatting (BBSplat): Learnable Textured Primitives for Novel View Synthesis](billboard_splatting_bbsplat_learnable_textured_primitives_fo.md)
- [\[ICCV 2025\] HumanOLAT: A Large-Scale Dataset for Full-Body Human Relighting and Novel-View Synthesis](humanolat_a_large-scale_dataset_for_full-body_human_relighting_and_novel-view_sy.md)

</div>

<!-- RELATED:END -->
