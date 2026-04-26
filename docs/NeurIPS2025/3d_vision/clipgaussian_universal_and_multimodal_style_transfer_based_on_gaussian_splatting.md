---
title: >-
  [论文解读] CLIPGaussian: Universal and Multimodal Style Transfer Based on Gaussian Splatting
description: >-
  [NeurIPS 2025][3D视觉][Gaussian Splatting] CLIPGaussian 提出首个基于 Gaussian Splatting 的统一风格迁移框架，支持文本和图像引导的 2D 图像、视频、3D 物体和 4D 动态场景的风格化，作为即插即用模块集成到现有 GS 管线中，无需大规模生成模型或从头重训，且不改变模型大小。
tags:
  - NeurIPS 2025
  - 3D视觉
  - Gaussian Splatting
  - 风格迁移
  - CLIP
  - 多模态
  - 3D/4D
---

# CLIPGaussian: Universal and Multimodal Style Transfer Based on Gaussian Splatting

**会议**: NeurIPS 2025  
**arXiv**: [2505.22854](https://arxiv.org/abs/2505.22854)  
**代码**: 即将开源（论文承诺）  
**领域**: 3D视觉 / 风格迁移  
**关键词**: Gaussian Splatting, 风格迁移, CLIP, multi-modal, 3D/4D

## 一句话总结

CLIPGaussian 提出首个基于 Gaussian Splatting 的统一风格迁移框架，支持文本和图像引导的 2D 图像、视频、3D 物体和 4D 动态场景的风格化，作为即插即用模块集成到现有 GS 管线中，无需大规模生成模型或从头重训，且不改变模型大小。

## 研究背景与动机

**领域现状**：Gaussian Splatting (GS) 已成为从 2D 图像渲染 3D 场景的高效表示方法，并已扩展到图像、视频和 4D 动态内容。风格迁移作为视觉内容编辑的重要任务，在 2D 领域已相当成熟，但在 GS 表示上的风格迁移仍面临挑战。

**现有痛点**：
   - 现有 GS 风格迁移方法（StyleGaussian、ReGS、InstantStyleGaussian 等）仅修改颜色和不透明度，无法改变几何结构
   - G-Style 虽然优化外观和几何，但会显著增加模型大小（Gaussian 数量翻倍以上）
   - 基于扩散模型的方法（如 Morpheus、Style3D）需要大模型，计算成本高且难以保证多视角一致性
   - 没有统一框架能同时处理 2D、视频、3D 和 4D 场景的风格迁移
   - 文本引导的 GS 编辑方法（I-GS2GS、DGE）主要针对编辑而非风格迁移

**核心矛盾**：GS 风格迁移在外观修改与几何变形之间的 trade-off——修改几何通常需要增加 Gaussian 数量，破坏原始模型的紧凑性

**本文解决什么**：
   - 如何在不改变 Gaussian 数量的前提下同时优化颜色和几何？
   - 如何支持跨 2D/视频/3D/4D 多模态的统一风格迁移？
   - 如何同时支持文本和图像两种风格引导方式？

**切入角度**：将风格迁移视为 GS 参数的微调问题，利用 CLIP 的跨模态对齐能力统一文本和图像作为风格条件，在保持原始 Gaussian 数量不变的前提下优化所有参数。

**核心idea**：将 CLIPStyler 的 patch-based CLIP loss 思想扩展到 GS 框架中，通过全局方向性 CLIP loss 和局部 patch CLIP loss 的组合，实现不改变模型大小的多模态风格迁移。

## 方法详解

### 整体框架

CLIPGaussian 采用两阶段训练流程：

**阶段一**：使用标准 GS 方法（3DGS/D-MiSo/MiRaGe/VeGaS）在输入数据上训练基础模型，获得场景的 Gaussian 表示 $\mathcal{G} = \{(\mathcal{N}(m_i, \Sigma_i), \sigma_i, c_i, \theta_i)\}_{i=1}^n$，其中 $m_i$ 为均值位置，$\Sigma_i$ 为协方差矩阵，$\sigma_i$ 为不透明度，$c_i$ 为球谐颜色，$\theta_i$ 为模态相关的额外参数。

**阶段二**：在冻结 Gaussian 数量的前提下，使用风格条件 $\mathcal{S}$（图像或文本）通过 CLIP 和 VGG 引导微调所有 Gaussian 参数。每步选择一张训练图像 $I_l$，渲染重建 $R_\mathcal{G}(I_l)$，提取随机 patch 并应用随机透视增强，然后通过多组分损失更新 $\mathcal{G}$ 的参数。

### 关键设计

**统一 Gaussian 表示**：不同模态使用不同的 GS 基础模型：
- 3D 场景：标准 3DGS
- 4D 动态场景：D-MiSo（多 Gaussian + 变形网络）
- 2D 图像：MiRaGe（平面 Gaussian + 3D 潜空间）
- 视频：VeGaS（3D 折叠 Gaussian）

关键在于，所有模态都可以统一为"一组 Gaussian 参数 + 渲染器"的形式，而风格迁移本质上是对这些参数的优化问题。

**即插即用设计**：CLIPGaussian 不修改基础模型的架构，不进行 densification 或 pruning，仅微调已有的 Gaussian 参数。这意味着：
- 风格化后的模型大小与原始模型完全相同
- 可以通过线性插值 Gaussian 参数实现风格插值
- 对基础模型无侵入性，任何 GS 方法都可以作为基础

**联合颜色与几何优化**：3D 和 4D 场景中，CLIPGaussian 同时优化位置、颜色、缩放、旋转和不透明度参数，实现真正的几何变形而非仅改变颜色。视频模态中仅优化颜色以保持时间一致性。

### 损失函数 / 训练策略

总损失函数为四项加权组合：

$$L_{total} = \lambda_d L_d + \lambda_p L_p + \lambda_c L_c + \lambda_b L_b$$

1. **内容损失 $L_c$**：使用 VGG-19 的 conv4_2 和 conv5_2 特征计算原始图像 $I_l$ 与渲染图像 $R_\mathcal{G}(I_l)$ 之间的 MSE，保持内容结构：

$$L_c(R_\mathcal{G}(I_l), I_l) = MSE(\Phi_{VGG}(R_\mathcal{G}(I_l)), \Phi_{VGG}(I_l))$$

2. **方向性 CLIP 损失 $L_d$**（全局风格）：衡量渲染图与原图的 CLIP 嵌入变化方向是否与风格条件和"Photo"负提示之间的方向一致：

$$L_d = 1 - \cos(\Phi_{CLIP}(R_\mathcal{G}(I_l)) - \Phi_{CLIP}(I_l), \Phi_{CLIP}(\mathcal{S}) - \Phi_{CLIP}(\text{"Photo"}))$$

3. **Patch CLIP 损失 $L_p$**（局部风格）：对渲染图的随机 patch 应用透视增强后计算方向性 CLIP 损失的均值，关注局部细节：

$$L_p = \frac{1}{n}\sum_{i=1}^n L_d(p_i(R_\mathcal{G}(I_l)), I_l)$$

4. **背景损失 $L_b$**：约束背景区域不被风格化污染，计算背景 mask 区域的 L1 距离。

默认超参数：$\lambda_b=1000$, $\lambda_p=90$, $\lambda_d=5$, $\lambda_c=0.8$, patch_size=128, num_patch=64, 训练 5000 步，无 densification/pruning。

## 实验关键数据

### 主实验

**3D 文本引导风格迁移**（NeRF-Synthetic + Mip-NeRF 360）：

| 方法 | CLIP-S ↑ | CLIP-SIM ↑ | CLIP-CONS ↑ | CLIP-F ↑ | 模型大小变化 |
|------|---------|----------|------------|---------|-----------|
| I-GS2GS | 16.80 | 12.03 | 99.19 | 13.53 | -36% |
| DGE | 17.59 | 12.27 | 99.31 | 12.46 | -5% |
| **CLIPGaussian** | **26.86** | **26.31** | 98.80 | 2.34 | **+0%** |

**3D 图像引导风格迁移**：

| 方法 | CLIP-S ↑ | CLIP-SIM ↑ | CLIP-CONS ↑ | CLIP-F ↑ | 模型大小变化 |
|------|---------|----------|------------|---------|-----------|
| StyleGaussian | 63.69 | 13.07 | 98.87 | 1.36 | +0% |
| G-Style | 76.94 | 24.94 | 98.94 | 1.31 | **+126%** |
| **CLIPGaussian** | 72.65 | 20.72 | 98.78 | 1.77 | **+0%** |

**视频风格迁移**（DAVIS 数据集，图像引导）：

| 方法 | CLIP-S ↑ | CLIP-SIM ↑ | CLIP-CONS ↑ | CLIP-F ↑ |
|------|---------|----------|------------|---------|
| CCPL | 18.89 | 8.20 | 97.92 | -0.02 |
| UniST | 15.93 | 3.85 | 99.36 | 5.16 |
| **CLIPGaussian** | **74.31** | **17.60** | 99.18 | 1.27 |

**视频风格迁移**（文本引导）：

| 方法 | CLIP-S ↑ | CLIP-SIM ↑ | CLIP-CONS ↑ | CLIP-F ↑ |
|------|---------|----------|------------|---------|
| Rerender | 19.40 | 9.83 | 98.23 | -0.03 |
| Text2Video | 26.05 | 24.99 | 93.63 | 0.03 |
| **CLIPGaussian** | **26.25** | 24.53 | **99.00** | 1.92 |

### 消融实验

**Feature learning rate 的影响**（3D，CLIP 指标）：

| feature_lr | CLIP-S ↑ | CLIP-SIM ↑ | CLIP-CONS ↑ | CLIP-F |
|-----------|---------|----------|------------|--------|
| 32 | 18.53 | 19.51 | 98.08 | 15.02 |
| 128 | 25.60 | 29.78 | 97.87 | 6.93 |
| 256 | 27.45 | 32.65 | 97.96 | 4.42 |

**$\lambda_p$ 和 $\lambda_d$ 参数影响**：

| $\lambda_p$ | CLIP-S | CLIP-SIM | CLIP-CONS | CLIP-F |
|------------|--------|---------|----------|--------|
| 0 | 16.00 | 14.12 | 98.74 | 11.03 |
| 90 (默认) | 23.67 | 25.07 | 97.95 | 2.36 |
| 180 | 24.36 | 25.27 | 97.78 | 1.48 |

**训练时间对比**：

| 场景 | Gaussian数量 | 风格化时间 |
|------|------------|-----------|
| hotdog | 0.14M | 11m29s |
| lego | 0.31M | 11m36s |
| bonsai | 1.35M | 11m37s |
| garden | 4.48M | 21m03s |

### 关键发现

1. **文本引导占优**：CLIPGaussian 在文本引导风格迁移中显著优于所有基线（CLIP-S 提升 50%+），主要因为 CLIP 的文本-图像对齐天然适合此任务
2. **零模型膨胀**：与 G-Style (+126% Gaussian 数量) 不同，CLIPGaussian 保持原始模型大小不变，具有实际部署优势
3. **时间一致性优势**：Gaussian 表示天然跨帧共享，风格修改自动传播到所有相关帧，比帧间传播方法更一致
4. **用户研究验证**：正式用户研究中（CLICKworker 平台，30 名参与者/调查），CLIPGaussian 在文本引导场景中排名最高

## 亮点与洞察

- **统一性设计理念**：首次在单一框架中统一 2D/视频/3D/4D 四种模态的风格迁移，证明 GS 作为通用基底的潜力
- **即插即用的实用性**：不修改基础架构、不改变模型大小的设计非常适合工程部署
- **风格插值能力**：由于 Gaussian 数量不变，可以通过线性插值参数实现平滑的风格过渡
- **Patch-based CLIP loss 的扩展**：将 CLIPStyler 的 2D 方法优雅地扩展到 3D/4D 领域

## 局限与展望

- 2D 图像风格化质量不如大模型或扩散模型方法（如 ChatGPT-4o），这是 GS 方法的固有限制
- 风格化效果依赖基础模型的重建质量——如果基础模型重建差，风格化可能产生不合理结果
- 训练时间（~11 分钟/场景）虽然合理但非实时
- CLIP 模型的语义理解有限，对于复杂抽象风格描述可能不够精准
- 4D 场景的对比实验受限（4DStyleGaussian 当时无代码，仅做视觉对比）

## 相关工作与启发

- **3DGS 系列**：CLIPGaussian 的基础框架，核心贡献在于将风格迁移引入 GS 生态
- **CLIPStyler / FastCLIPstyler**：2D 文本引导风格迁移，CLIPGaussian 的 patch CLIP loss 直接受其启发
- **AdaIN / StyTr2**：经典图像风格迁移方法，作为 2D 基线
- **G-Style**：最接近的 3D 竞争对手，但需要增加 Gaussian 数量
- **D-MiSo / VeGaS / MiRaGe**：CLIPGaussian 使用的各模态基础 GS 模型

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首个跨 4 种模态的统一 GS 风格迁移框架
- **技术深度**: ⭐⭐⭐ — 核心技术是 CLIP loss 的组合，相对直接
- **实验充分度**: ⭐⭐⭐⭐⭐ — 四种模态全面评估，用户研究，大量消融
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，可视化丰富
- **实用性**: ⭐⭐⭐⭐ — 即插即用设计有实际应用价值

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Tune-Your-Style: Intensity-Tunable 3D Style Transfer with Gaussian Splatting](../../ICCV2025/3d_vision/tune-your-style_intensity-tunable_3d_style_transfer_with_gaussian_splatting.md)
- [\[NeurIPS 2025\] LODGE: Level-of-Detail Large-Scale Gaussian Splatting with Efficient Rendering](lodge_level-of-detail_large-scale_gaussian_splatting_with_efficient_rendering.md)
- [\[ICCV 2025\] A3GS: Arbitrary Artistic Style into Arbitrary 3D Gaussian Splatting](../../ICCV2025/3d_vision/a3gs_arbitrary_artistic_style_into_arbitrary_3d_gaussian_spl.md)
- [\[NeurIPS 2025\] Instant Video Models: Universal Adapters for Stabilizing Image-Based Networks](instant_video_models_universal_adapters_for_stabilizing_image-based_networks.md)
- [\[NeurIPS 2025\] IBGS: Image-Based Gaussian Splatting](ibgs_image-based_gaussian_splatting.md)

<!-- RELATED:END -->
