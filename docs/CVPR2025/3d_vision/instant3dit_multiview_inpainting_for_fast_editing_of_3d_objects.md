---
title: >-
  [论文解读] Instant3dit: Multiview Inpainting for Fast Editing of 3D Objects
description: >-
  [CVPR 2025][3D视觉][3D editing] 将 3D 编辑问题转化为多视角一致的 2D inpainting 问题，通过微调 SDXL-inpainting 模型在 2×2 视角网格上同时生成一致的填充内容，再用 LRM 重建 3D，实现约 3 秒完成高质量 3D 编辑——比 SDS 方法快数百倍。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D editing
  - 图像修复
  - 3D masking
  - 扩散模型
  - SDXL
  - mesh editing
---

# Instant3dit: Multiview Inpainting for Fast Editing of 3D Objects

**会议**: CVPR 2025  
**arXiv**: [2412.00518](https://arxiv.org/abs/2412.00518)  
**代码**: [项目页面](https://amirbarda.github.io/Instant3dit.github.io/)  
**领域**: 3d_vision  
**关键词**: 3D editing, multiview inpainting, 3D masking, diffusion model, SDXL, mesh editing

## 一句话总结

将 3D 编辑问题转化为多视角一致的 2D inpainting 问题，通过微调 SDXL-inpainting 模型在 2×2 视角网格上同时生成一致的填充内容，再用 LRM 重建 3D，实现约 3 秒完成高质量 3D 编辑——比 SDS 方法快数百倍。

## 研究背景与动机

**领域现状**: 3D 生成编辑（如在已有模型上添加/修改部件）通常依赖 SDS (Score Distillation Sampling) 或 IDU (Iterative Dataset Update) 逐步优化 3D 表示，耗时数十分钟到数小时。

**现有痛点**:
1. **极慢的运行时间**: SDS 需要反复渲染+扩散模型推理+反传梯度，即使用 GaussianSplatting 加速仍需 ~15 分钟
2. **低质量输出**: SDS 倾向于寻找特定分布模式，导致生成结果"缺乏多样性"且纹理过饱和/模糊
3. **多视角不一致**: 现有 2D inpainting 模型不具备 3D 一致性，直接 per-view inpainting 后难以重建

**核心矛盾**: 要实现快速 3D 编辑需要避免 SDS 迭代优化，但直接用 2D 生成模型会丧失多视角一致性。

**本文目标**: 设计一套快速、高质量、表示无关的 3D inpainting 方法，使用户可以通过简单的 3D mask + 文本提示完成局部 3D 编辑。

**切入角度**: 不在 3D 空间优化，而是训练一个能同时 inpaint 2×2 多视角图像网格的扩散模型，然后用现成 LRM 重建 3D。

## 方法详解

### 整体框架

1. 用户在 3D 物体上绘制 mask $M$（粗糙的 3D 几何体），提供文本提示 $y$
2. 从 4 个规范视角渲染被 mask 遮挡的物体 $I_c(S,M)$ 和 mask $I_b(M,S)$，拼成 2×2 网格
3. 多视角 inpainting 扩散模型 $\epsilon_\theta$ 在网格上一次性生成 4 个一致的 inpainted 视图
4. 用现成的 LRM（NeRF-LRM / MeshLRM / GS-LRM）从多视角图重建新的 3D 表示

### 关键设计

**1. 多视角一致 Inpainting 微调策略**
- **功能**: 从 SDXL-inpainting 模型出发，在 3D 多视角 inpainting 数据上微调，使其同时具备 inpainting 能力和多视角一致性
- **核心思路**: 
    - 训练数据用 Objaverse 筛选的 5K 高质量 3D 物体，每个物体渲染 4 视角 2×2 网格
    - 输入 9 通道: noise latent (4ch) + mask image latent (4ch) + downsampled mask (1ch)
    - 10% 的时间随机丢弃 mask 条件，退化为纯多视角生成训练
    - 文本条件由 LLaVA 对 3D 物体生成高质量 caption
- **设计动机**: 相比反向路线（从多视角生成模型 Instant3D 微调 inpainting 能力），从 inpainting 模型微调更容易——因为多视角一致性可以通过少量 3D 数据学到，而 inpainting 需要在大规模数据上训练

**2. 3D Mask 设计（三种编辑模式）**
- **功能**: 设计三种类型的 3D mask 数据集，模拟用户实际编辑行为
- **Type I (Coarse Edit)**: 用随机平面切割物体，取部分的凸包放大 20% 作为 mask — 适合粗糙替换
- **Type II (Mesh Sculpting)**: 用随机平面切割，直接选择被切面以上的物体表面 — 适合精确雕刻
- **Type III (Surface Editing)**: 在物体表面采样一点，用多个随机椭圆柱体选取局部表面 patch — 适合纹理修改
- **设计动机**: Mask 设计是 inpainting 训练的关键——训练 mask 分布越接近测试时用户行为，效果越好；3D 一致的 mask（考虑遮挡关系）比随机 2D mask 效果显著更好
- 每个物体 30 个 mask（三种各 10 个），总计 ~150K 训练样本

**3. 表示无关的 3D 重建**
- **功能**: inpainted 的多视角图可输入不同 LRM 得到不同 3D 表示
- **核心思路**: 
    - NeRF: 用 NeRF-LRM，毫秒级重建
    - Mesh: 用 MeshLRM + ROAR 自适应重网格化（~20秒），保留原始 mesh 的 UV、拓扑等属性
    - Gaussian Splat: 用 GS-LRM
    - 法线估计器直接从扩散输出估计法线，用于 mesh 优化中保持细节
- **设计动机**: 通过在 2D 图像层面做编辑，天然解耦了 3D 表示的选择，使系统灵活适配不同下游需求

### 损失函数 / 训练策略

- 标准 latent diffusion 的 v-prediction 损失
- 基础模型: SDXL-inpainting
- 训练数据: Objaverse 5K 高质量物体，150K 多视角-mask 样本对
- 推理: Euler scheduler, 29 步，~3 秒/编辑 (A100)
- 各重建器耗时: NeRF-LRM ~0.7s, Mesh ~3s, Mesh+ROAR ~20s

## 实验关键数据

### 主实验（500 样本 benchmark）

| 方法 | ClipL↑ | ClipG↑ | SSIM↑ | LPIPS↓ | DreamSim↓ | FID↓ |
|---|---|---|---|---|---|---|
| SDXL | 低 | 低 | 低 | 高 | 高 | 高 |
| SDXL-inpainting | 低 | 低 | 低 | 高 | 高 | 高 |
| Instant3D | 中 | 中 | **最高** | **最低** | **最低** | 高 |
| **Ours (SDXL-inp.)** | **最高** | **最高** | 次高 | 次低 | 次低 | **最低** |

Instant3D 在多视角一致性上最好（本身就是多视角模型），但在 prompt 遵循和视觉质量上不如本方法。

### Mask 消融实验

| 训练 Mask | Type I FID↓ | Type II FID↓ | Type III FID↓ | All FID↓ | User Mask ClipG↑ |
|---|---|---|---|---|---|
| Random 2D | 差 | 差 | 差 | 差 | 最差 |
| Type I | 最好 | 中 | 中 | 中 | 中 |
| Type II | 中 | 最好 | 中 | 中 | 中 |
| Type III | 中 | 中 | 最好 | 中 | 中 |
| **I+II+III** | 次好 | 次好 | 次好 | **最好** | **最好** |

### 用户偏好研究

在 208 对比较中（15 位用户），Instant3dit vs NeRFiller:
- **Instant3dit: 86% 偏好** vs NeRFiller: 14%

### 速度对比

| 方法 | 编辑时间 |
|---|---|
| Vox-E | ~1h |
| Progressive3D | ~1h |
| NeRFiller | ~30K steps |
| MVEdit | ~30min |
| **Instant3dit** | **~3s + reconstruction** |

### 关键发现

1. **从 inpainting 微调比从 multiview 微调更有效**: SDXL-inpainting 出发在 prompt adherence 和视觉质量上大幅优于 Instant3D 出发，说明已有的大规模 inpainting 先验是关键优势
2. **3D-aware mask 至关重要**: Random 2D mask 训练的模型在所有指标上显著差于 3D mask 训练的模型
3. **混合 mask 训练最通用**: 单一 mask 类型训练在对应类型上最优但泛化差，I+II+III 混合训练在任意 mask 上表现最均衡
4. **模型可泛化到新视角**: 虽然训练只用固定视角，但由于 inpainting 的自注意力机制和原始模型的先验，可以泛化到任意方位角偏移
5. **SDS 的根本局限**: SDS 不仅慢，而且在 inpainting 场景下质量明显更差（趋向模式塌缩），证实了 DreamFusion 原文的分析

## 亮点与洞察

- "将 3D 编辑转化为多视角 2D inpainting"的思路是优雅的范式转换，同时解决了速度和质量两个问题
- 3D mask 数据集设计巧妙：三种 mask 类型对应三种实际编辑操作（粗糙替换、精确雕刻、局部纹理），使训练分布贴近真实使用
- 表示无关性是重要的工程优势：同一套多视角 inpainting 可直接对接 NeRF / Mesh / GS 的 LRM
- 微调策略的实验消融（从 inpainting vs 从 multiview）提供了有价值的设计指导

## 局限与展望

- 很细的 mask 可能被忽略（2D inpainting 的通病，模型倾向于重绘更大区域）
- 训练数据只有白色背景，大 mask 且无强 inductive bias 时可能生成白色背景而非遵循 prompt
- 仅支持 4 视角 2×2 网格，对于复杂遮挡或大型场景可能不够
- 依赖 LRM 的重建质量，当前 LRM 的 triplane 分辨率有限导致几何过度平滑
- Objaverse 训练数据仅 5K 物体，覆盖的物体类别和编辑类型有限

## 相关工作与启发

- Instant3D 证明了 2×2 网格生成+LRM 重建是快速 3D 生成的可行路线；本文将其扩展到有条件的 inpainting
- NeRFiller 用单视角 inpainting + IDU 优化做 3D inpainting，但多视角不一致且慢；本文首次实现多视角一致 inpainting
- MagicClay 提出了 mesh 编辑的局部约束方案（ROAR），本文在 mesh 重建后复用了这一技术
- 启发：从"2D 基础模型微调到 3D 一致"可能比"从 3D 数据训练"更高效，因为 2D 预训练可以传递丰富的视觉先验

## 评分

⭐⭐⭐⭐ — 范式上的创新（3D 编辑 → 多视角 inpainting）带来了速度和质量的双重飞跃，mask 设计和微调策略有充分的消融支撑；主要局限在训练数据规模和 LRM 重建质量的瓶颈。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] PrEditor3D: Fast and Precise 3D Shape Editing](preditor3d_fast_and_precise_3d_shape_editing.md)
- [\[CVPR 2025\] 3D Gaussian Inpainting with Depth-Guided Cross-View Consistency](3d_gaussian_inpainting_with_depth-guided_cross-view_consistency.md)
- [\[CVPR 2025\] UnCommon Objects in 3D](uncommon_objects_in_3d.md)
- [\[CVPR 2025\] Generative Multiview Relighting for 3D Reconstruction under Extreme Illumination Variation](generative_multiview_relighting_for_3d_reconstruction_under_extreme_illumination.md)
- [\[ICCV 2025\] 3D Mesh Editing using Masked LRMs](../../ICCV2025/3d_vision/3d_mesh_editing_using_masked_lrms.md)

<!-- RELATED:END -->
