---
title: >-
  [论文解读] Recovering Dynamic 3D Sketches from Videos
description: >-
  [CVPR 2025][3D视觉][动态草图] Liv3Stroke 提出了首个从视频中提取动态 3D 草图的方法，用可变形的三维 Bézier 曲线集合抽象表示物体运动，通过学习点云运动引导和逐笔画变形实现视点一致的运动草图重建。
tags:
  - CVPR 2025
  - 3D视觉
  - 动态草图
  - 三维运动抽象
  - Bézier曲线
  - 可微渲染
  - 运动表征
---

# Recovering Dynamic 3D Sketches from Videos

**会议**: CVPR 2025  
**arXiv**: [2503.20321](https://arxiv.org/abs/2503.20321)  
**代码**: https://jaeah.me/liv3stroke_web (项目页面)  
**领域**: 3D视觉  
**关键词**: 动态草图, 三维运动抽象, Bézier曲线, 可微渲染, 运动表征

## 一句话总结

Liv3Stroke 提出了首个从视频中提取动态 3D 草图的方法，用可变形的三维 Bézier 曲线集合抽象表示物体运动，通过学习点云运动引导和逐笔画变形实现视点一致的运动草图重建。

## 研究背景与动机

**领域现状**：从视频中理解 3D 运动是计算机视觉的核心问题。现有方法要么用非结构化的密集运动向量表示运动（如光流场），要么依赖预定义的关节模板（如 SMPL）。动态场景的神经辐射场方法（D-NeRF、4DGS）虽然能实现逼真重建，但求解的是高维复杂变量，且在外观变化下容易出错。

**现有痛点**：密集运动表示计算开销大且难以直接分析运动的核心结构；模板方法只适用于特定物体类型；动态 NeRF/3DGS 方法追求像素级精确重建，但无法提供运动的抽象理解。

**核心矛盾**：需要一种既紧凑又能表达任意拓扑运动的中间表示，它既不像密集场那样冗余，也不像模板那样受限于特定物体。

**本文目标**：用少量可变形的 3D 曲线（笔画）来抽象表达视频中物体的核心运动特征。

**切入角度**：受草图绘制的启发——人可以用几笔简单的线条就传达一个场景的核心信息。最近 CLIPasso 等工作已证明可以从图像中自动生成抽象草图。作者将此思路扩展到"用可变形的 3D 笔画抽象运动"。

**核心 idea**：将草图定义为一组可变形的 3D Bézier 曲线，通过每条曲线的平移/旋转和控制点调整来表达运动，并用感知损失（LPIPS + CLIP）在潜在空间中对齐草图与真实帧。

## 方法详解

### 整体框架

Liv3Stroke 的流水线分为两个阶段：(1) 从视频帧中学习 3D 运动引导——用点云和变形 MLP 获取粗略的 3D 运动布局；(2) 基于运动引导，初始化 3D 曲线并学习其变形——通过旋转/平移 MLP 和控制点调整 MLP 来拟合每条笔画在每个时间步的位置和形状。最终输出是每帧一组 3D Bézier 曲线，可从任意视角渲染为 2D 草图。

### 关键设计

1. **3D 运动引导学习**:

    - 功能：从视频帧中提取粗略的 3D 运动场，为后续笔画初始化和变形提供引导
    - 核心思路：用 10k 个 3D 点云表示场景，用 MLP 变形网络（输入为位置编码+时间编码）预测每个点在每个时间步的位移。关键是使用 LPIPS 感知损失（而非像素级 L1/L2 损失）来对齐渲染的点云图与真实帧，因为点云渲染的是灰度图而非自然图像。增加速度连续性正则和刚性正则确保运动平滑
    - 设计动机：LPIPS 损失能捕捉结构性差异而非像素细节，实验证明它比 L1/L2 损失能更好地保持物体边界和形状完整性。这一阶段不追求精确重建，而是获取"运动在哪里、怎么动"的粗略信息

2. **动态 3D 笔画变形**:

    - 功能：将运动表达为每条 Bézier 曲线的刚性变换和形状变形的组合
    - 核心思路：每条笔画的运动被分解为两个层次：(1) 每笔画的旋转 $R_i^t$ 和平移 $T_i^t$（通过 $\mathcal{M}_R$ 和 $\mathcal{M}_T$ 两个 MLP 预测），(2) 每个控制点的局部位移 $\Delta p_i^j$（通过 $\mathcal{M}_L$ MLP 预测）。采用粗到精策略：先在低分辨率下学习全局旋转/平移，再在高分辨率下学习控制点微调。$\mathcal{M}_T$ 使用运动引导阶段预训练的权重初始化
    - 设计动机：将刚性运动和非刚性变形分离可以更好地控制运动的不同层次——全局位移由旋转/平移 MLP 处理，局部形状变化（如翅膀扇动的细节）由控制点调整处理

3. **感知损失驱动的草图优化**:

    - 功能：在语义特征空间中对齐渲染的草图与输入视频帧
    - 核心思路：综合 LPIPS 损失（捕捉结构相似性）和 CLIP 距离损失（捕捉语义相似性）：$\mathcal{L}_{frame}^s = \lambda_s \rho(LPIPS(\mathcal{I}, \mathcal{S})) + dist(CLIP(\mathcal{I}), CLIP(\mathcal{S}))$。加上速度连续性正则 $\mathcal{L}_{temp}^s$ 和运动幅度正则 $\mathcal{L}_{reg}$。还使用修正函数 $\xi(\mathbf{x})$ 来抑制静止期间的微小运动
    - 设计动机：草图是抽象表达，不可能在像素空间与自然图像对齐，必须在语义/感知空间建立对应。CLIP 损失确保草图保持语义正确性，LPIPS 损失确保结构对齐

### 损失函数 / 训练策略

运动引导阶段的总损失为 $\mathcal{L}_{guidance} = 0.1 \cdot \mathcal{L}_{frame}^g + 0.05 \cdot \mathcal{L}_{temp}^g + 10^{-4} \cdot \mathcal{L}_{rigid}$。草图生成阶段的总损失为 $\mathcal{L}_{sketch} = \mathcal{L}_{frame}^s + \mathcal{L}_{temp}^s + \mathcal{L}_{reg}$。使用 Adam 优化器，所有 MLP 网络独立训练。

## 实验关键数据

### 主实验

| 方法 | 结构对齐 (新视角↑) | 结构对齐 (固定视角↑) | 运动提示相似度 (新视角↑) | 运动提示相似度 (固定视角↑) |
|------|-----------------|-----------------|---------------------|---------------------|
| CLIPasso | 0.760±0.107 | 0.740±0.127 | 0.659±0.007 | 0.664±0.011 |
| Sketch Video Syn. | 0.663±0.115 | 0.657±0.135 | 0.654±0.011 | 0.658±0.011 |
| Suggestive Contours | 0.784±0.102 | 0.750±0.119 | 0.661±0.013 | 0.656±0.016 |
| **Liv3Stroke** | 0.693±0.096 | 0.683±0.108 | **0.656±0.006** | **0.656±0.008** |

3D 运动引导精度评估：

| 方法 | Chamfer 距离↓ | 运动速度误差↓ |
|------|-------------|-------------|
| 4DGS | 0.205±0.046 | 4.60±3.00 |
| DG-Mesh | 0.277±0.059 | 4.10±2.70 |
| **Liv3Stroke** | 0.252±0.049 | **4.16±2.34** |

### 消融实验

| 配置 | Chamfer 距离↓ | 运动速度误差↓ |
|------|-------------|-------------|
| w/o $\mathcal{L}_{temp}^g$ | 0.258±0.043 | 6.81±5.41 |
| w/o $\mathcal{L}_{rigid}$ | 0.253±0.048 | 5.45±3.74 |
| 用 L2 替代 $\mathcal{L}_{rigid}$ | 0.253±0.045 | 4.61±2.94 |
| 完整模型 | 0.252±0.049 | 4.16±2.34 |

### 关键发现

- Liv3Stroke 在所有指标上展现最小的标准差，说明方法在不同场景和视角下最为稳定
- 直接优化笔画（不经引导）会产生毫无结构的断裂线段；缺少粗阶段引导则丢失全局结构（如鸟的翅膀）
- 速度连续性正则对运动质量至关重要——去掉后运动速度误差从 4.16 暴涨到 6.81
- 尽管 Liv3Stroke 不追求逼真重建，其点云运动精度与专做精确网格重建的 DG-Mesh 相当

## 亮点与洞察

- **"以简驭繁"的表达哲学**最令人印象深刻——用几十条线段就能抓住复杂 3D 运动的精髓。这不仅是技术创新，更提出了"运动可以用什么最简形式表达"这个深层问题
- **LPIPS 损失用于非自然图像的结构对齐**是一个可复用的 trick：在任何需要将抽象表示与自然图像对齐的场景中，感知损失都优于像素损失
- 刚性运动+控制点变形的分离策略可迁移到任何需要表达可变形物体运动的场景，如机器人操作中的柔性物体跟踪

## 局限与展望

- 只考虑视角无关的笔画，无法渲染基于视角的轮廓（如圆形物体的侧影）
- 笔画数量需要用户手动设定，没有自适应机制决定最优笔画数
- 对于纯平移的运动表达效果好，但高度复杂的非刚性变形（如流体）可能超出 Bézier 曲线的表达能力
- 未来可将笔画的物理属性（如刚度、质量）引入以实现基于笔画的物理模拟控制

## 相关工作与启发

- **vs CLIPasso**: CLIPasso 仅处理单帧 2D 草图，时序一致性差。Liv3Stroke 在 3D 空间中定义笔画，天然保证多视角一致性
- **vs Sketch Video Synthesis**: 该方法基于 layered neural atlas，无法处理大幅运动且始终在 2D 空间操作。Liv3Stroke 升维到 3D 并通过变形 MLP 处理大运动
- **vs 3Doodle/EMAP**: 这些方法处理静态场景的 3D 草图，Liv3Stroke 首次将其扩展到动态场景。关键区别在于引入了变形 MLP 框架

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个动态 3D 草图重建方法，问题定义本身就很新颖
- 实验充分度: ⭐⭐⭐⭐ 定量和定性评估充分，但缺少用户研究评估草图的可理解性
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详细
- 价值: ⭐⭐⭐⭐ 开创了运动抽象这一新方向，但实际应用场景还需更多探索

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera](dyn-hamr_recovering_4d_interacting_hand_motion_from_a_dynamic_camera.md)
- [\[CVPR 2025\] FreeGave: 3D Physics Learning from Dynamic Videos by Gaussian Velocity](freegave_3d_physics_learning_from_dynamic_videos_by_gaussian_velocity.md)
- [\[CVPR 2025\] MegaSaM: Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos](megasam_accurate_fast_and_robust_structure_and_motion_from_casual_dynamic_videos.md)
- [\[CVPR 2025\] MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds](mosca_dynamic_gaussian_fusion_from_casual_videos_via_4d_motion_scaffolds.md)
- [\[CVPR 2025\] Generating 3D-Consistent Videos from Unposed Internet Photos](generating_3d-consistent_videos_from_unposed_internet_photos.md)

<!-- RELATED:END -->
