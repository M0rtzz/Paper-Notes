---
title: >-
  [论文解读] Towards Realistic Example-Based Modeling via 3D Gaussian Stitching
description: >-
  [CVPR 2025][3D视觉][3D高斯拼接] 提出首个基于 3D 高斯表示的真实感示例建模方法，通过采样式克隆（S-phase）和聚类式调优（T-phase）实现多个 3D 高斯场的无缝拼接与和谐外观融合，支持交互式实时编辑。 示例建模（Example-based Modeling）是计算机图形学的经典方法论——从不同…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D高斯拼接"
  - "示例建模"
  - "外观融合"
  - "交互式编辑"
  - "采样式克隆"
---

# Towards Realistic Example-Based Modeling via 3D Gaussian Stitching

**会议**: CVPR 2025  
**arXiv**: [2408.15708](https://arxiv.org/abs/2408.15708)  
**代码**: 无  
**领域**: 3D视觉 / 神经渲染  
**关键词**: 3D高斯拼接, 示例建模, 外观融合, 交互式编辑, 采样式克隆

## 一句话总结

提出首个基于 3D 高斯表示的真实感示例建模方法，通过采样式克隆（S-phase）和聚类式调优（T-phase）实现多个 3D 高斯场的无缝拼接与和谐外观融合，支持交互式实时编辑。

## 研究背景与动机

示例建模（Example-based Modeling）是计算机图形学的经典方法论——从不同模型中选取部件组合创建新模型。随着 NeRF/3DGS 技术的发展，从真实世界场景中直接进行真实感的示例建模成为可能。然而，现有方法面临以下挑战：

1. **外观不和谐**：简单地将多个神经场放在一起不处理外观融合，导致拼接处色调、纹理不一致
2. **SeamlessNeRF 的局限性**：唯一尝试解决无缝拼接的工作使用基于梯度传播的策略 + 网格表示，存在三个问题——无法交互式编辑、真实场景下产生明显伪影、无法传播复杂结构特征
3. **3DGS 的离散挑战**：3D 高斯是离散、不规则分布的点表示，不像 NeRF 的隐式网格那样支持梯度传播，需要全新的融合策略

本文的核心洞察是：利用 3DGS 的点云特性实现精细编辑和实时预览，同时通过采样策略代替梯度传播来实现无缝的特征传播。

## 方法详解

### 整体框架

pipeline 包含四个步骤：(1) 使用定制 GUI 将多个预训练的 3DGS 场景进行分割和刚体变换，获得语义合理的组合；(2) KNN 分析识别源场和目标场交叉区域的边界点；(3) S-phase（采样式克隆）：通过邻域采样显式传播边界的颜色和结构特征到目标场全体点；(4) T-phase（聚类式调优）：使用从源场提取的颜色调色板进行全局色调和谐化。所有步骤可在 GUI 中交互操作和实时预览。

### 关键设计

1. **KNN 边界条件识别**:
    - 功能：组合后自动识别目标场中位于交界区域的边界点
    - 核心思路：对目标场 $\mathcal{T}$ 中的每个高斯点 $a$，搜索其在源场 $\mathcal{S}$ 中的 K 近邻 $\{b_i\}_K$。当平均距离 $\frac{1}{K}\sum|b_i - a| < \beta$ 且不透明度 $o(a) > \tau$ 时标记为边界点。边界点的目标特征由源场 K 近邻的 SH 系数均值决定，通过 $\mathcal{L}_{feature}$ 损失优化
    - 设计动机：边界条件是和谐化的起始状态，精确的边界提取直接决定后续优化质量。$\tau=0.95$ 过滤低不透明度的噪声点，$\beta=0.05 \times L$（$L$ 为组合体尺寸）保证边界宽度合理

2. **采样式克隆（S-phase）**:
    - 功能：将边界处的颜色和结构特征无缝传播到目标场的所有非边界点
    - 核心思路：对每个非边界点 $a$，通过映射函数 $\phi(x) = x + \sin(\gamma \cdot \delta x)$（$\gamma=10$, $\delta x$ 为到最近边界点的距离）扰动位置后在边界点集中搜索 K 近邻作为"驱动点"。将驱动点的视角依赖 SH 颜色作为 $a$ 的优化目标（color loss）。同时在目标场的局部空间内用 Sobel 算子预计算 2D 梯度，施加梯度保持损失保留原始纹理内容
    - 设计动机：朴素的 Laplacian 方法（用邻域差作正则）梯度太弱无法启动传播。显式采样通过 color loss 提供足够驱动力。正弦扰动引入随机性使纹理传播更自然，高 $\gamma$ 适合高频结构

3. **聚类式调优（T-phase）**:
    - 功能：全局和谐化组合体的色调、亮度和饱和度
    - 核心思路：从源场多视角渲染中通过流式聚类提取颜色调色板（聚类中心 $c_i$ + 占比 $w_i$）。初始 3 个 bin，逐步扩展，中心通过新样本平均更新，20 次无足够投票的中心过期。对目标场渲染的每个像素匹配最近调色板中心，施加加权 L2 损失。仅对高不透明度（$\alpha > 0.95$）像素施加
    - 设计动机：S-phase 保证局部一致性但可能出现全局色调不均（亮度/色相偏移），T-phase 通过调色板匹配实现全局颜色对齐

### 损失函数 / 训练策略

总损失：$\mathcal{L}_{total} = \mathcal{L}_{feature} + \mathcal{L}_{color} + \lambda_1\mathcal{L}_{grad} + \lambda_2\mathcal{L}_{tune}$

$\lambda_1 = \lambda_2 = 2$。T-phase 损失在 S-phase 进行一段时间后加入（联合优化，S-phase 损失始终保持）。相机中心从以组合体原点为中心的球面上均匀采样。

## 实验关键数据

### 主实验

| 方法 | VQA 平均分↑ |
|------|------------|
| SeamlessNeRF | 0.753 |
| **Ours** | **0.784** |

实验使用 21 个组合结果，涉及 39 个部件模型：17 个 BlendedMVS + 4 个 Mip360 + 16 个 SeamlessNeRF + 2 个自建。

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 无 color loss (S-phase) | 传播无法启动 | feature loss 梯度太弱 |
| 仅 S-phase | 边界无缝但全局色调不均 | 局部和谐、全局失调 |
| 仅 T-phase | 缺乏结构传播 | 仅色调调整不够 |
| S+T (完整) | 最优 | 两阶段互补 |
| 高权重梯度损失 | 保留更多内容但阻碍和谐化 | 需平衡 |
| 无 $\phi$ 扰动 | 纹理传播机械 | 随机性提升真实感 |
| 2D Sobel vs 3D 梯度 | 2D 更有效 | 聚焦表面更合理 |
| 随机 vs 策略性视角采样 | 策略性更好保持视角依赖效果 | 正确传播视角相关颜色 |

### 关键发现

- S-phase 中 color loss 不可或缺——仅靠 feature loss 无法启动传播
- 2D 屏幕空间 Sobel 梯度损失比 3D 空间梯度更有效（聚焦可见表面）
- 正弦扰动函数 $\phi$ 使纹理传播更自然，避免机械复制
- SeamlessNeRF 在所有真实场景中均失败，证明梯度传播方法在复杂场景下的局限性
- 2D 风格迁移方法也无法实现无缝拼接效果

## 亮点与洞察

- **首个 3DGS 真实感部件组合方法**：填补了 3DGS 示例建模的空白
- **采样式克隆**巧妙解决了 3DGS 离散不规则分布无法梯度传播的难题
- **S+T 两阶段设计**分解了局部无缝（S）和全局和谐（T）两个层面的问题
- **交互式 GUI**的实用性强，支持分割、变换、边界识别、优化全流程实时预览
- **视角依赖正确传播**：球面采样 + SH 颜色采样确保反射等效果的一致性

## 局限与展望

- 仅支持刚体变换，不支持非刚体变形（如 ARAP），限制了创意灵活性
- 未考虑光照一致性，强光照场景下组合质量可能下降
- 缺少标准化定量评估指标和 GT，VQA 评估有局限
- 未来可与变形方法和光照估计方法结合

## 相关工作与启发

- **vs SeamlessNeRF**: SeamlessNeRF 用梯度传播+网格表示在真实场景失败；本文采样策略+点云表示全面克服
- **vs Neural Imposter**: Neural Imposter 仅放置无融合，本文实现和谐无缝拼接
- **vs SNeRF 风格迁移**: 2D 风格迁移无法实现 3D 一致的无缝效果

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个 3DGS 示例建模方法，采样式克隆是核心创新
- 实验充分度: ⭐⭐⭐ 消融详尽但定量评估受限于缺乏 benchmark
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，消融可视化直观
- 价值: ⭐⭐⭐⭐ 为 3DGS 编辑开辟新方向，交互式设计实用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Vid2Sim: Realistic and Interactive Simulation from Video for Urban Navigation](vid2sim_realistic_and_interactive_simulation_from_video_for_urban_navigation.md)
- [\[CVPR 2025\] EnvGS: Modeling View-Dependent Appearance with Environment Gaussian](envgs_modeling_view-dependent_appearance_with_environment_gaussian.md)
- [\[CVPR 2025\] Textured Gaussians for Enhanced 3D Scene Appearance Modeling](textured_gaussians_for_enhanced_3d_scene_appearance_modeling.md)
- [\[ICCV 2025\] From Gallery to Wrist: Realistic 3D Bracelet Insertion in Videos](../../ICCV2025/3d_vision/from_gallery_to_wrist_realistic_3d_bracelet_insertion_in_videos.md)
- [\[CVPR 2025\] RigGS: Rigging of 3D Gaussians for Modeling Articulated Objects in Videos](riggs_rigging_of_3d_gaussians_for_modeling_articulated_objects_in_videos.md)

</div>

<!-- RELATED:END -->
