---
title: >-
  [论文解读] UniPre3D: Unified Pre-training of 3D Point Cloud Models with Cross-Modal Gaussian Splatting
description: >-
  [CVPR 2025][3D视觉][3D预训练] UniPre3D 提出了首个统一的3D预训练方法，通过预测高斯原语并利用可微高斯溅射渲染图像来提供像素级监督，同时引入尺度自适应的跨模态融合策略，使得预训练方法能同时适用于物体级和场景级的任意尺度点云与任意架构的3D模型。 领域现状：3D点云的表示学习分为点基方法（Point…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D预训练"
  - "点云"
  - "高斯溅射"
  - "跨模态融合"
  - "统一学习"
---

# UniPre3D: Unified Pre-training of 3D Point Cloud Models with Cross-Modal Gaussian Splatting

**会议**: CVPR 2025  
**arXiv**: [2506.09952](https://arxiv.org/abs/2506.09952)  
**代码**: [https://github.com/wangzy22/UniPre3D](https://github.com/wangzy22/UniPre3D)  
**领域**: 3D视觉  
**关键词**: 3D预训练, 点云, 高斯溅射, 跨模态融合, 统一学习

## 一句话总结

UniPre3D 提出了首个统一的3D预训练方法，通过预测高斯原语并利用可微高斯溅射渲染图像来提供像素级监督，同时引入尺度自适应的跨模态融合策略，使得预训练方法能同时适用于物体级和场景级的任意尺度点云与任意架构的3D模型。

## 研究背景与动机

**领域现状**：3D点云的表示学习分为点基方法（PointNet++ 等，擅长物体级细粒度局部结构）和体素基方法（SparseUNet 等，擅长场景级长程关系建模）。当前的3D预训练方法也因此分成两条路线：物体级主要使用掩码自编码（MAE）范式，场景级主要使用对比学习范式。

**现有痛点**：MAE 方法在场景级数据上效果差，因为 Chamfer Distance 损失对大规模数据计算代价高且监督不精确；对比学习在物体级数据上容易过快饱和，预训练效果有限。两类方法各有适用场景，无法统一。

**核心矛盾**：点云的尺度多样性是根本挑战——场景点云可包含比物体点云多上百倍的点，导致同一种预训练范式无法同时适应两种尺度。然而在2D图像领域，物体图像和场景图像的信息密度差异并不大。

**本文目标**：设计一种统一的3D预训练方法，能够：(1) 适用于任意尺度的点云（物体级和场景级）；(2) 适用于任意架构的3D模型（点基和体素基）。

**切入角度**：作者观察到2D图像域的尺度差异远小于3D点云域，因此提出用图像域作为中间媒介来缩小点云数据的尺度差异。从3D数据渲染投影图像作为预训练任务，其难度可以自动适应数据的尺度。

**核心 idea**：用3D高斯溅射（3DGS）从点云预测高斯原语并渲染图像，通过像素级图像重建损失端到端训练3D backbone，并融合预训练图像模型的2D特征来调控任务复杂度。

## 方法详解

### 整体框架

UniPre3D 的 pipeline 由两个模态分支组成。**3D 分支**包括点云 backbone、轻量高斯预测器和可微图像渲染器。**2D 分支**包括预训练图像特征提取器、2D-to-3D 几何投影器和尺度自适应融合模块。整体前向传播分三个阶段：提取（Extract）、融合（Fuse）、渲染（Render）。

输入为点云 $P \in \mathbb{R}^{N \times 3}$ 和参考视角图像 $I_{\text{ref}}$，输出为从预测的高斯原语渲染的图像 $I_r$，通过与真实图像的 MSE 损失进行端到端优化。

### 关键设计

1. **高斯原语预测与渲染**:

    - 功能：将3D点云预训练任务转化为图像渲染任务，实现像素级精确监督
    - 核心思路：3D backbone 提取点云特征后，通过轻量 MLP 头预测每个点对应的高斯原语参数（位置偏移、不透明度、缩放、旋转四元数、球谐系数，共23维），然后利用可微高斯溅射技术渲染多视角图像。高斯原语的协方差属性决定了其有效区域，小规模点云学到较大协方差（轻微模糊可接受），大规模点云有更密集的高斯带来更多细节
    - 设计动机：相比 NeRF（Ponder 系列使用），3DGS 有三大优势：(1) 尺度自适应性——协方差自动调节；(2) 轻量设计——只需 MLP 头，避免预训练知识被辅助组件吸收；(3) 效率——可全图监督，速度约为 PonderV2 的2倍

2. **物体级特征融合（Feature Fusion）**:

    - 功能：在物体级预训练中融合2D图像特征与3D特征，提供颜色和纹理信息
    - 核心思路：由于物体级数据集缺少深度图，通过将3D点投影到2D平面建立对应关系。对每个点计算其在参考图像上的像素坐标 $(u, v)$，取最小深度的点作为表面点对齐像素网格。然后将对齐后的2D特征与 backbone 最后解码层的3D特征拼接，通过 MLP 进行融合：$F_{\text{fuse}} = \text{MLP}(\text{cat}(F_{3D}, \hat{F}_{2D}))$
    - 设计动机：物体级点云无颜色但渲染目标有颜色，直接学习会导致 backbone 错误提取与下游无关的颜色特征。融合参考视角图像特征提供颜色线索，使 backbone 专注于几何结构学习

3. **场景级点融合（Point Fusion）**:

    - 功能：在场景级预训练中通过增加伪点云来增强视觉引导，降低预训练任务难度
    - 核心思路：利用场景数据集的深度图，将2D像素通过相机内外参反投影到3D空间，得到伪点云 $P_{2D}$。将其与 backbone 第一编码层输出的 $P_{3D}$ 合并为跨模态元点云，经体素化采样后通过剩余网络提取融合特征。此操作使高斯原语数量增加约70%
    - 设计动机：场景点云过于稀疏且几何关系复杂，仅用点云作输入使预训练任务过难。特征融合策略在场景级表现不佳（消融实验证实），而点融合通过增补密集的2D信息降低了优化难度

### 损失函数 / 训练策略

- **损失函数**：采用像素级 MSE 损失。物体级预训练引入前景/背景加权：$\mathcal{L}^{\text{obj}} = \omega_{\text{fg}} \mathcal{L}(I_r^{\text{fg}}, I_{\text{gt}}^{\text{fg}}) + \omega_{\text{bg}} \mathcal{L}(I_r^{\text{bg}}, I_{\text{gt}}^{\text{bg}})$，其中 $\omega_{\text{fg}}=4, \omega_{\text{bg}}=1$
- **渲染策略**：渲染视角与参考视角不重叠以防信息泄露；场景级限制参考/渲染视角间距（<5帧）以提高图像知识利用率
- **训练细节**：物体级在 ShapeNet 上预训练50 epoch，1张3090Ti；场景级在 ScanNetV2 上预训练100 epoch，8张3090Ti。2D分支使用 Stable Diffusion autoencoder

## 实验关键数据

### 主实验

| 任务 | Backbone | 数据集 | 指标 | 无预训练 | UniPre3D |
|------|----------|--------|------|----------|----------|
| 分类 | Std. Transformer | ScanObjectNN PB_T50_RS | OA(%) | 77.24 | **87.93** |
| 分类 | Mamba3D | ScanObjectNN PB_T50_RS | OA(%) | 92.6 | **93.4** |
| 部件分割 | PointMLP | ShapeNetPart | mIoU_C | 84.6 | **85.5** |
| 语义分割 | SparseUNet | ScanNet200 | mIoU | 25.0 | **28.3** |
| 语义分割 | PTv3 | ScanNet200 | mIoU | 35.2 | **36.4** |

### 消融实验

| 消融内容 | ScanNet200 mIoU |
|----------|----------------|
| 无2D融合 | 26.8 |
| 特征融合（场景级） | 27.0 |
| 点融合（场景级） | **28.3** |
| 无前/背景加权（物体级） | OA降约0.5% |

### 关键发现

- UniPre3D 在物体级和场景级任务上均取得一致性提升，验证了统一预训练的有效性
- 对已有高性能的 backbone（如 Mamba3D 的92.6%、PTv3 的77.45%），UniPre3D 仍能带来提升
- 场景级的点融合策略显著优于特征融合策略，因为场景数据更稀疏且复杂
- 3DGS 渲染的图像虽然在场景级较模糊，但已足够学习基本几何关系

## 亮点与洞察

- 首次实现统一的3D预训练，打破了物体级和场景级预训练范式的壁垒
- 利用图像域作为"中间媒介"来消弭点云尺度差异的思路非常巧妙
- 针对不同尺度数据设计不同的融合策略（特征融合 vs 点融合），而非一刀切
- 整体设计轻量高效，预训练附加组件不会抢占 backbone 的学习能力

## 局限与展望

- 场景级渲染图像质量较低（模糊），可能限制了预训练上限
- 2D分支使用固定的 Stable Diffusion autoencoder，未探索更强的视觉基础模型
- 未探索更大规模数据集和更长预训练对效果的影响
- 可考虑结合深度/法线等几何信号增强渲染监督

## 相关工作与启发

- **Ponder/PonderV2**：同样使用生成式预训练但基于 NeRF，UniPre3D 用 3DGS 替代实现了2倍速度提升和全图监督
- **TAP**：另一种生成式预训练方法，使用 attention-based 预测器较笨重，UniPre3D 的 MLP 头更轻量
- **Point-MAE 系列**：MAE 范式在物体级有效但在场景级失效，是本文要解决的核心问题

## 评分

- **新颖性**: 8/10 — 首个统一3D预训练方法，用 3DGS 桥接2D-3D的思路新颖
- **实验充分度**: 9/10 — 覆盖多个 backbone、多个任务、多个数据集，消融实验充分
- **写作质量**: 8/10 — 逻辑清晰，设计动机阐述到位
- **价值**: 8/10 — 对3D预训练领域有推动作用，但场景级渲染质量仍有提升空间

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] GEAL: Generalizable 3D Affordance Learning with Cross-Modal Consistency](geal_generalizable_3d_affordance_learning_with_cross-modal_consistency.md)
- [\[CVPR 2025\] CrossOver: 3D Scene Cross-Modal Alignment](crossover_3d_scene_cross-modal_alignment.md)
- [\[ICCV 2025\] Towards More Diverse and Challenging Pre-training for Point Cloud Learning: Self-Supervised Cross Reconstruction with Decoupled Views](../../ICCV2025/3d_vision/towards_more_diverse_and_challenging_pre-training_for_point_cloud_learning_self-.md)
- [\[CVPR 2025\] Feat2GS: Probing Visual Foundation Models with Gaussian Splatting](feat2gs_probing_visual_foundation_models_with_gaussian_splatting.md)
- [\[CVPR 2026\] CMHANet: A Cross-Modal Hybrid Attention Network for Point Cloud Registration](../../CVPR2026/3d_vision/cmhanet_a_cross-modal_hybrid_attention_network_for_point_cloud_registration.md)

</div>

<!-- RELATED:END -->
