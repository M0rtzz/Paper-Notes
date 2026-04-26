---
title: >-
  [论文解读] iSegMan: Interactive Segment-and-Manipulate 3D Gaussians
description: >-
  [CVPR 2025][3D视觉][3D高斯泼溅] iSegMan提出了一个无需场景特定训练的交互式3DGS分割与操作框架，通过极线引导的交互传播(EIP)和基于可见性的高斯投票(VGV)实现精确的3D区域控制，配合完整的操作工具箱支持语义编辑、上色、缩放、复制粘贴、组合和删除等多种功能。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D高斯泼溅
  - 交互式分割
  - 场景操作
  - 极线约束
  - SAM
---

# iSegMan: Interactive Segment-and-Manipulate 3D Gaussians

**会议**: CVPR 2025  
**arXiv**: [2505.11934](https://arxiv.org/abs/2505.11934)  
**代码**: https://zhao-yian.github.io/iSegMan  
**领域**: 3D视觉  
**关键词**: 3D高斯泼溅, 交互式分割, 场景操作, 极线约束, SAM

## 一句话总结
iSegMan提出了一个无需场景特定训练的交互式3DGS分割与操作框架，通过极线引导的交互传播(EIP)和基于可见性的高斯投票(VGV)实现精确的3D区域控制，配合完整的操作工具箱支持语义编辑、上色、缩放、复制粘贴、组合和删除等多种功能。

## 研究背景与动机

1. **领域现状**：3D高斯泼溅因其高效渲染和显式表示，正在推动3D场景操作领域的快速发展。现有方法如GaussianEditor通过文本驱动编辑，Instruct-GS2GS通过指令进行全局编辑。

2. **现有痛点**：(1) 现有3DGS操作方法难以精确控制操作区域，容易影响无关区域；(2) 基于文本提示的区域控制（如GaussianEditor）受限于文本描述的粗粒度，无法分割细粒度区域；(3) 现有3D交互式分割方法（SA3D、SAGA等）都需要场景特定的参数训练（特征蒸馏），影响效率和灵活性。

3. **核心矛盾**：场景操作需要精确的区域控制，但现有方法要么控制粗糙（文本驱动），要么需要耗时的预训练（特征蒸馏）。如何实现"零训练、精准控制、交互反馈"成为关键挑战。

4. **本文目标** 构建一个无需场景特定训练的3D区域控制模块，支持用户2D点击的任意视角交互，并在此基础上提供完整的场景操作工具箱。

5. **切入角度**：利用极线几何约束替代暴力特征匹配来传播用户交互到其他视角，利用高斯泼溅的alpha混合可见性信息替代特征训练来完成3D区域提取。

6. **核心 idea**：极线约束缩小搜索空间+可见性加权投票替代特征蒸馏=零训练的精准3D分割。

## 方法详解

### 整体框架
iSegMan接受用户在任意视角提供的2D点击（正/负），输出分割出的3D高斯子集。整体pipeline为：EIP将用户点击传播到多个视角 → SAM在各视角生成2D分割mask → VGV根据mask和高斯可见性投票提取3D区域 → 操作工具箱对选中区域执行用户指定功能。全过程不需要任何场景特定训练。

### 关键设计

1. **极线引导的交互传播 (EIP)**:

    - 功能：将用户在单一视角的2D点击高效、鲁棒地传播到其他视角
    - 核心思路：给定视角 $v$ 的2D点击 $p_v$，利用相机内外参将其投影为3D射线 $r_{p_v}$，然后计算该射线在新视角 $\tilde{v}$ 上的极线 $e_{p_v}^{\tilde{v}}$。匹配搜索被限制在极线上，使用DINO特征在极线上进行相似度匹配：$p_{\tilde{v}} = \text{Upsample}(I_{\tilde{v}}[\text{argmax}(\mathcal{A}_{p_v}^{\tilde{v}})])$，其中亲和度 $\mathcal{A}$ 只在极线采样点上计算
    - 设计动机：直接在全图做特征匹配搜索空间大、噪声多、不鲁棒。极线约束将2D搜索空间严格缩减为1D线段，大幅提高效率和准确性。使用Bresenham算法高效采集极线上的离散特征

2. **基于可见性的高斯投票 (VGV)**:

    - 功能：根据多视角2D分割mask提取目标3D高斯，无需任何特征训练
    - 核心思路：将2D像素视为"投票参与者"，3D高斯视为"候选者"。每个参与者的投票权重由该像素对该高斯的可见性（alpha混合权重）决定：$\Upsilon_{i,j} = \sigma_i \cdot \alpha_i \prod_{k=1}^{i-1}(1-\alpha_k)$。所有视角所有像素的投票求和得到每个高斯的得票数 $\Psi_j$，超过阈值的高斯被选中。此外引入迭代检查机制(IIM)：检查SAM预测mask与当前选中高斯的渲染mask是否相交，排除因遮挡或离开视野导致的错误分割
    - 设计动机：现有方法通过训练3D语义特征场来实现2D-3D映射，耗时且不灵活。高斯泼溅的alpha混合渲染天然提供了像素到高斯的贡献权重，直接利用这一信息做投票，完全省去训练步骤。投票权重的非对称性（可见性高→投票权大）自然处理了遮挡关系

3. **操作工具箱 (Manipulation Toolbox)**:

    - 功能：提供完整的3D场景操作功能集合
    - 核心思路：(1) 语义编辑：利用InstructPix2Pix编辑渲染视图，通过L1+感知损失迭代更新选中区域的高斯参数，配合退火策略提高稳定性；(2) 上色：直接修改高斯颜色属性，支持颜色替换和均衡上色两种模式；(3) 缩放：同时缩放方向向量和缩放因子保持刚体变换不变性；(4) 复制粘贴/组合/删除：直接操作高斯集合
    - 设计动机：精确的区域控制为各种操作功能提供了基础，工具箱化的设计使框架高度可扩展

### 损失函数 / 训练策略
语义编辑的更新损失：$\nabla_\theta \Theta_s = \mathbb{E}_v[(\frac{\partial \|I_v^e - I_v\|_1}{\partial I_v} + \frac{\partial \mathcal{D}(I_v, I_v^e)}{\partial I_v}) \cdot \frac{\partial I_v}{\partial \theta}]$，包括L1图像损失和LPIPS感知距离。使用退火策略逐步减小更新步长。VGV模块本身无需训练。

## 实验关键数据

### 主实验：交互式3D分割（SPIn-NeRF数据集）

| 方法 | 需训练 | mIoU(%) | mAcc(%) | 特征时间 | 分割时间 |
|------|--------|---------|---------|---------|---------|
| SA3D | ✓ | 91.9 | 98.8 | 5min | 30s |
| SAGA | ✓ | 88.0 | 98.5 | ~1.5h | 10ms |
| LangSplat | ✓ | 69.5 | 94.5 | ~2.5h | - |
| **iSegMan** | ✗ | **92.4** | **99.1** | **52s** | **6s** |

### 语义编辑定量对比

| 指标 | Instruct-GS2GS | GaussianEditor | iSegMan |
|------|----------------|----------------|---------|
| User study↑ | 2.10±0.20 | 3.32±0.40 | **4.52±0.20** |
| CLIP dir↑ | 0.1647 | 0.2071 | **0.2189** |

### 关键发现
- iSegMan在分割性能上最优（mIoU 92.4%），同时完全不需要场景特定训练
- 特征提取时间仅52秒（vs SAGA的1.5h和SA3D的5min），因为不需要训练特征场
- 在语义编辑中，iSegMan通过精确区域控制避免了irrelevant区域被影响的问题
- GaussianEditor基于文本的区域控制精度不足，容易导致编辑artifact
- 迭代检查机制(IIM)有效排除了因遮挡/离开视野导致的错误SAM mask

## 亮点与洞察
- **极线约束的巧妙应用**：将多视立体视觉中的经典几何约束用于2D交互传播，将2D匹配问题降维为1D极线上的搜索。这个思路可以迁移到任何需要跨视角传播2D标注的任务
- **投票机制替代特征训练**：利用高斯泼溅的alpha混合权重作为投票权重，完全省去特征蒸馏步骤。这揭示了3DGS的渲染过程本身就蕴含了丰富的2D-3D对应信息
- **交互式编辑循环**：支持渐进式编辑复杂需求（如"把人变成穿绿衬衫黄裤子的铜像"），且可复用中间结果提高效率

## 局限与展望
- 语义编辑部分依赖InstructPix2Pix，受限于2D编辑器的能力
- 极线匹配使用DINO特征，在纹理重复或弱纹理区域可能失败
- 投票阈值需要手动设定，不同场景可能需要不同阈值
- 目前仅支持2D点击交互，不支持更丰富的交互形式（如涂鸦、文本+点击混合）
- 可改进：将VGV与SAM 2结合处理视频/动态场景

## 相关工作与启发
- **vs SA3D**: SA3D通过mask逆渲染+跨视角自提示交替训练3D mask，需要训练步骤。iSegMan用投票替代训练，更快且性能更好
- **vs SAGA**: SAGA先用SAM获取所有视角的mask再蒸馏3D特征，预处理耗时约1.5h。iSegMan只处理用户交互传播到的视角，按需计算
- **vs GaussianEditor**: GaussianEditor用文本定位编辑区域，粒度粗。iSegMan用点击交互，可精确到任意细粒度区域

## 评分
- 新颖性: ⭐⭐⭐⭐ EIP和VGV的设计新颖实用，零训练分割是重要的实用创新
- 实验充分度: ⭐⭐⭐⭐ 分割+编辑两个任务的定量和定性评估全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，投票机制的表述形式化且直观
- 价值: ⭐⭐⭐⭐⭐ 高实用价值，为3DGS场景操作提供了真正可交互的工具

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] Click-Gaussian: Interactive Segmentation to Any 3D Gaussians](../../ECCV2024/3d_vision/click-gaussian_interactive_segmentation_to_any_3d_gaussians.md)
- [\[CVPR 2025\] WonderWorld: Interactive 3D Scene Generation from a Single Image](wonderworld_interactive_3d_scene_generation_from_a_single_image.md)
- [\[CVPR 2025\] IAAO: Interactive Affordance Learning for Articulated Objects in 3D Environments](iaao_interactive_affordance_learning_for_articulated_objects_in_3d_environments.md)
- [\[CVPR 2025\] SplineGS: Robust Motion-Adaptive Spline for Real-Time Dynamic 3D Gaussians from Monocular Video](splinegs_robust_motion-adaptive_spline_for_real-time_dynamic_3d_gaussians_from_m.md)
- [\[CVPR 2025\] DropoutGS: Dropping Out Gaussians for Better Sparse-view Rendering](dropoutgs_dropping_out_gaussians_for_better_sparse-view_rendering.md)

<!-- RELATED:END -->
