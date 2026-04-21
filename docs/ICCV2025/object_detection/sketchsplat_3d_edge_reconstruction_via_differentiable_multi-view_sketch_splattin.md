---
title: >-
  [论文解读] SketchSplat: 3D Edge Reconstruction via Differentiable Multi-view Sketch Splatting
description: >-
  [ICCV 2025][目标检测][3D边缘重建] 提出 SketchSplat，将 3D 边缘表示为参数化 sketch（直线+Bézier曲线），通过从 sketch 采样高斯点进行可微渲染来直接优化边缘参数，同时提出自适应拓扑控制和改进的 2D 边缘检测器，在 CAD 数据集上实现 SOTA 的准确性、完整性和紧凑性。
tags:
  - ICCV 2025
  - 目标检测
  - 3D边缘重建
  - 可微渲染
  - 高斯溅射
  - 参数化曲线
  - CAD建模
---

# SketchSplat: 3D Edge Reconstruction via Differentiable Multi-view Sketch Splatting

**会议**: ICCV 2025  
**arXiv**: [2503.14786](https://arxiv.org/abs/2503.14786)  
**代码**: https://oceanying.github.io/SketchSplat  
**领域**: object_detection  
**关键词**: 3D边缘重建, 可微渲染, 高斯溅射, 参数化曲线, CAD建模

## 一句话总结

提出 SketchSplat，将 3D 边缘表示为参数化 sketch（直线+Bézier曲线），通过从 sketch 采样高斯点进行可微渲染来直接优化边缘参数，同时提出自适应拓扑控制和改进的 2D 边缘检测器，在 CAD 数据集上实现 SOTA 的准确性、完整性和紧凑性。

## 研究背景与动机

3D 边缘重建是计算机视觉和图形学的基础任务，支撑 CAD 建模、SLAM、自动驾驶等应用。现有方法分两类：

**传统方法**（线匹配+三角化）：准确但无法同时处理直线和曲线，多视图匹配不鲁棒

**可微渲染方法**（NEF、EMAP、EdgeGS）：先优化 3D 边缘点集，再拟合参数化边缘。问题在于拟合步骤仅依赖重建的 3D 点，不参考 2D 边缘图像，因此：
   - 点集中的噪声导致拟合边缘断裂
   - 拟合后的 3D 边缘与 2D 图像不对齐

核心动机：能否**直接优化参数化 3D 边缘**，让优化过程通过可微渲染与 2D 边缘图像对齐？

## 方法详解

### 整体框架

输入多视角 RGB 图像 → 2D 边缘检测（提出 2DGS-SN 方法）→ 用 EdgeGS 初始化 3D 边缘点 → 拟合为参数化 sketch → 从 sketch 采样高斯点 → 可微光栅化 → 计算 L1 损失 → 反向传播优化 sketch 参数 + 拓扑操作 → 输出优化后的参数化 3D 边缘

### 关键设计

1. **Sketch 表示与可微优化**：

    - 3D 边缘表示为两种 sketch：直线 $l \in \mathbb{R}^{2 \times 3}$（2个控制点）和三阶 Bézier 曲线 $c \in \mathbb{R}^{4 \times 3}$（4个控制点）
    - 每个 sketch 附带可优化的 opacity $o$ 和局部 scale $s \in \mathbb{R}^3$
    - **核心创新**：采样步骤——以固定步长（5mm）从 sketch 上采样点，每个点继承对应 sketch 的 opacity/scale，主方向取 sketch 局部切线方向
    - 采样得到的高斯点通过 3DGS 的光栅化流程渲染到 2D，与 GT 边缘图计算 L1 损失
    - 梯度反传到 sketch 参数 $S_i = \{l_i \text{ or } c_i, o_i, s_i\}$，实现端到端优化
    - 设计动机：3D 高斯作为中间表示，天然支持可微渲染，且易于扩展到不同类型的曲线

2. **自适应拓扑控制**：在优化过程中穿插四种拓扑操作，提升紧凑性和连通性：

    - **端点合并（End-point Merging）**：两个 sketch 端点距离 < 10mm 时合并为一个点（通过维护可优化点集 $P$，控制点通过索引引用）
    - **重叠 Sketch 合并（Overlapping Merging）**：若一个 sketch 被另一个覆盖超过 80%（采样点邻近比例），则将较小的合并到较大的
    - **共线 Sketch 合并（Co-linear Merging）**：两条方向相近（< 5°）、偏移小（< 10mm）、间隙小的直线合并为更长的直线
    - **Sketch 过滤（Filtering）**：训练后移除在超过 90% 视角中不可见的 sketch，解决遮挡导致的错误重建
    - 使用 AABB 快速过滤不满足条件的 sketch 对以加速

3. **2DGS-SN 边缘检测器**：

    - 现有神经网络边缘检测器（PiDiNet、DexiNed）在 CAD 物体上不准确，存在系统性偏移和纹理误检
    - 提出利用几何线索的检测方法：结合前景 mask $A$、深度图梯度 $g(D)$（来自 2DGS 重建的深度）和法线图梯度 $g(N)$（来自法线估计器）
    $E = G_f * (A | (g(D) > t_d) | (g(N) > t_n))$
    - 用 Sobel 算子计算梯度，阈值化后 OR 合并，再高斯滤波平滑

### 损失函数 / 训练策略

- L1 损失：渲染边缘图与 GT 边缘图之间，前景背景等比例采样
- Adam 优化器，每场景 1000 epochs，每 epoch 所有训练视角的损失累积后更新一次
- 场景归一化到 1m³ 包围盒内，拓扑操作阈值按场景尺寸 1% 设定
- 初始化 + sketch 优化各约 5 分钟/场景，总计约 10 分钟

## 实验关键数据

### 主实验（ABC-NEF 数据集，82 个 CAD 模型）

| 方法 | 检测器 | A↓(mm) | C↓(mm) | R5↑ | P5↑ | F5↑ | R10↑ | F10↑ |
|------|--------|--------|--------|-----|-----|-----|------|------|
| LIMAP | LSD | 9.9 | 18.7 | 36.2 | 43.0 | 39.0 | 84.3 | 90.4 |
| NEF | PiDiNeT | 11.9 | 16.9 | 11.4 | 15.7 | 13.0 | 64.0 | 93.3 |
| EMAP | 2DGS-SN | 8.8 | 7.9 | 63.5 | 70.4 | 66.3 | 90.4 | 95.1 |
| EdgeGS | 2DGS-SN | 7.4 | 7.2 | 75.7 | 86.9 | 80.3 | 92.9 | 95.8 |
| **SketchSplat** | **2DGS-SN** | **6.8** | **5.8** | **90.8** | **92.9** | **91.3** | **95.4** | **96.5** |

首个在 A、C 均 < 7mm 且 R5/P5/F5 均 > 90% 的方法。使用 2DGS-SN 检测器时，F5 比 EdgeGS 提升 11 个百分点。

### 消融实验

| 配置 | A↓ | C↓ | R5↑ | P5↑ | F5↑ | 边缘数 |
|------|-----|-----|-----|-----|-----|--------|
| 完整 SketchSplat | 6.8 | 5.8 | 90.8 | 92.9 | 91.3 | 44.28 |
| w/o 2DGS-SN（用 DexiNed） | 8.5 | 8.6 | 47.5 | 54.1 | 50.3 | 33.52 |
| w/o Merge（无拓扑合并） | 6.6 | 5.7 | 91.8 | 92.7 | 91.7 | **140.72** |
| w/o Filter | 6.8 | 5.8 | 90.9 | 92.9 | 91.3 | 44.37 |

不同初始化方法消融：

| 初始化 | A↓ | C↓ | F5↑ | 边缘数 |
|--------|-----|-----|-----|--------|
| EdgeGS → SketchSplat | 6.8 | 5.8 | 91.3 | 44.28 |
| EMAP → SketchSplat | 6.9 | 6.2 | 89.3 | 31.70 |

### 关键发现

- 2DGS-SN 检测器对所有方法（EMAP、EdgeGS、SketchSplat）都有显著提升，说明 2D 边缘质量是瓶颈
- 拓扑合并将边缘数量从 140 降到 44，大幅提升紧凑性而几乎不损失其他指标
- SketchSplat 从不同初始化出发均能大幅提升过初始化方法，证明可微优化的通用有效性
- 与 EdgeGS（140.8 条边缘）相比，SketchSplat 仅需 44.3 条即达到更高精度
- 训练效率：10 分钟/场景，与 EdgeGS 相当，远快于 NEF（1.5h）和 EMAP（2.5h）

## 亮点与洞察

- **架构创新**：用 3D 高斯作为中间表示桥接参数化边缘与 2D 图像，巧妙利用了 3DGS 的可微渲染能力
- **问题定义精准**：识别出"先重建点集再拟合"的两阶段范式的根本缺陷——拟合与图像监督脱节
- **端到端优化**：sketch 参数直接受 2D 图像损失约束，保证了 3D 边缘与 2D 的一致性
- **工程设计精巧**：拓扑操作在优化中穿插而非后处理，保证结果仍与图像一致

## 局限与展望

- 2DGS-SN 检测器依赖深度图和法线图的质量，需要预先用 2DGS 重建场景
- 仅处理直线和三阶 Bézier 曲线，对更复杂曲线类型未探索
- 排除了包含圆柱和球面的 CAD 模型（轮廓边在多视角下不一致）
- 初始化仍依赖现有方法（EdgeGS/EMAP），未实现完全端到端的从零初始化

## 相关工作与启发

- 与 DiffVG（2D 可微矢量图形优化）思路相似但推广到 3D，且用高斯采样避免了为每种曲线定制微分方式
- 2DGS-SN 检测器的思路（利用重建的深度/法线）可推广到其他需要精确边缘的任务
- sketch 表示可与 CAD 重建流水线整合，potential 用于 CAD 逆向工程

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次通过可微 sketch splatting 直接优化参数化 3D 边缘
- **实验充分度**: ⭐⭐⭐⭐ ABC-NEF 上结果令人印象深刻，DTU/Replica 也有定性评估
- **写作质量**: ⭐⭐⭐⭐ 动机清晰、方法描述详实、图例质量高
- **价值**: ⭐⭐⭐⭐ 为 3D 边缘重建开辟了新范式，F5 提升超过 10%

<!-- RELATED:START -->

## 相关论文

- [ViewSRD: 3D Visual Grounding via Structured Multi-View Decomposition](viewsrd_3d_visual_grounding_via_structured_multi-view_decomposition.md)
- [3D-MOOD: Lifting 2D to 3D for Monocular Open-Set Object Detection](3dmood_lifting_2d_to_3d_for_monocular_openset_object_detecti.md)
- [YOLO-Count: Differentiable Object Counting for Text-to-Image Generation](yolo-count_differentiable_object_counting_for_text-to-image_generation.md)
- [REXO: Indoor Multi-View Radar Object Detection via 3D Bounding Box Diffusion](../../AAAI2026/object_detection/rexo_indoor_multi-view_radar_object_detection_via_3d_bounding_box_diffusion.md)
- [Pippo: High-Resolution Multi-View Humans from a Single Image](../../CVPR2025/object_detection/pippo_high-resolution_multi-view_humans_from_a_single_image.md)

<!-- RELATED:END -->
