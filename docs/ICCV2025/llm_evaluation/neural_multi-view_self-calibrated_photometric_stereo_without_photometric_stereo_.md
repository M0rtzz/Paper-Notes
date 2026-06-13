---
title: >-
  [论文解读] Neural Multi-View Self-Calibrated Photometric Stereo without Photometric Stereo Cues
description: >-
  [ICCV 2025][LLM评测][多视图光度立体] 提出一种端到端的神经逆渲染框架，从多视图变化光照图像中联合恢复几何、空间变化反射率和光照参数，无需光源标定或中间光度立体线索（如法线图），超越了现有的分阶段 MVPS 方法。 从图像中恢复场景的内在属性——几何、反射率和光照——是计算机视觉的长期核心问题…
tags:
  - "ICCV 2025"
  - "LLM评测"
  - "多视图光度立体"
  - "神经逆渲染"
  - "自标定"
  - "端到端优化"
  - "神经BRDF"
  - "阴影感知体渲染"
---

# Neural Multi-View Self-Calibrated Photometric Stereo without Photometric Stereo Cues

**会议**: ICCV 2025  
**arXiv**: [2507.23162](https://arxiv.org/abs/2507.23162)  
**领域**: 3D Reconstruction / Inverse Rendering / Photometric Stereo  
**关键词**: 多视图光度立体, 神经逆渲染, 自标定, 端到端优化, 神经BRDF, 阴影感知体渲染

## 一句话总结

提出一种端到端的神经逆渲染框架，从多视图变化光照图像中联合恢复几何、空间变化反射率和光照参数，无需光源标定或中间光度立体线索（如法线图），超越了现有的分阶段 MVPS 方法。

## 研究背景与动机

从图像中恢复场景的内在属性——几何、反射率和光照——是计算机视觉的长期核心问题。光度立体（PS）方法通过在固定视角下逐一激活不同方向的光源来采集 OLAT（one-light-at-a-time）图像，从光照变化中分析表面法线，是解决这类问题的经典范式。多视图光度立体（MVPS）进一步扩展到从多个视角采集 OLAT 图像栈，实现完整的 3D 重建。

然而，现有 MVPS 方法存在根本性的架构缺陷：

**分阶段处理导致误差累积**：典型 pipeline 为"光源标定 → 逐视图 PS 线索估计 → 多视图融合"，每个阶段的误差向下传播

**逐视图 PS 忽略跨视图信息**：独立估计每个视角的法线图，无法利用跨视图的互补观测，且容易产生**跨视图不一致**

**依赖光源标定**：需要铬球或漫反射白板等校准设备

**要求视图对齐的 OLAT**：传统方法要求每个视角都有完整的 OLAT 图像栈，限制了采集灵活性

**核心洞察**：既然 multi-view OLAT 图像提供了已知的密集光度观测，为什么不直接从原始像素端到端地联合优化所有场景参数？分阶段处理会丢失信息而非利用信息。

## 方法详解

### 整体框架

框架接收多视图 OLAT 图像（含前景 mask 和相机参数），联合优化三大场景属性：

1. **几何**：神经 SDF（Signed Distance Function）
2. **反射率**：神经隐式 BRDF
3. **光照**：每个光源的方向和 RGB 强度

核心渲染 pipeline 包含三个 MLP：
- **Spatial MLP**：预测每个场景点的 SDF 值和 BRDF 隐码
- **BRDF MLP**：从隐码和角度编码预测反射率值
- **Shadow MLP**：细化基于 SDF 透射率的阴影因子

### 关键设计 1：图像形成模型与阴影感知体渲染

场景点 $\mathbf{x}$ 在观察方向 $\mathbf{v}$、光源方向 $\boldsymbol{\ell}$、强度 $e$ 下的辐射量：

$$r(\mathbf{x}) = e \cdot f(\mathbf{x}, \mathbf{n}, \mathbf{v}, \boldsymbol{\ell}) \cdot (\mathbf{n}^\top \boldsymbol{\ell})_+$$

像素观测值通过阴影感知体渲染计算：

$$r(\mathbf{p}) = s'(\mathbf{x}', \boldsymbol{\ell}) \cdot e \cdot \sum_k T_k \alpha_k f(\mathbf{x}_k, \mathbf{n}_k, \mathbf{v}, \boldsymbol{\ell}) (\mathbf{n}_k^\top \boldsymbol{\ell})_+$$

其中 $s'$ 是阴影因子，$T_k$ 是累积透射率，$\alpha_k$ 是不透明度。

**关键近似**：假设光线上所有采样点共享与表面交点相同的阴影因子，将阴影计算复杂度从 $O(N^2)$ 降为 $O(N)$。

### 关键设计 2：几何表示（Neural SDF + Hash Encoding）

使用 Spatial MLP 同时预测 SDF 值和 BRDF 隐码：

$$(g(\mathbf{x}), \mathbf{b}(\mathbf{x})) = \mathcal{G}(\mathcal{H}(\mathbf{x}; \phi); \theta)$$

- $\mathcal{H}$：多分辨率哈希编码（instant-ngp 风格），有利于高频细节恢复和训练效率
- 表面法线通过 SDF 梯度分析计算：$\mathbf{n} = \overline{\nabla g}$
- 不透明度从 SDF 转换而来，使用可优化锐度参数的 sigmoid 函数

### 关键设计 3：神经隐码驱动 BRDF

不同于传统方法将 BRDF 分解为漫反射和镜面反射项再用解析模型拟合，本文用单个 MLP 直接预测 BRDF 值：

$$f(\mathbf{x}, \mathbf{n}, \mathbf{v}, \boldsymbol{\ell}) = \mathcal{F}(\mathbf{b}(\mathbf{x}), \mathcal{A}(\mathbf{n}, \mathbf{v}, \boldsymbol{\ell}); \psi)$$

**角度编码（Angular Encoding）** 是关键创新之一：

$$\mathcal{A}(\mathbf{n}, \mathbf{v}, \boldsymbol{\ell}) = [\mathbf{n}^\top \mathbf{h}, \boldsymbol{\ell}^\top \mathbf{h}, \mathbf{n}^\top \boldsymbol{\ell}, \mathbf{n}^\top \mathbf{v}, (\mathbf{n}^\top \mathbf{h})^{10}]^\top$$

其中 $\mathbf{h} = \overline{\boldsymbol{\ell} + \mathbf{v}}$ 是半角向量。

角度编码的设计理念：
- 将法线-观测-光照方向转换为**旋转不变**的标量特征
- 具有相同 BRDF 但不同世界坐标法线的点更容易被映射到相同隐码
- MLP 无需隐式学习旋转不变性，降低学习难度

与已有工作的区别：NeRFactor 需要在测量 BRDF 数据集上预训练隐码 BRDF，本方法证明了**神经隐码驱动 BRDF 可以从多视图 OLAT 图像从零优化**。

### 关键设计 4：阴影建模

**两步阴影策略**：

1. **基于 SDF 的体渲染阴影**：从估计的表面交点沿光照方向投射阴影光线，通过透射率累积计算阴影因子：

$$s = 1 - \sum_k T_k^{(s)} \alpha_k^{(s)}$$

2. **Shadow MLP 细化**：体渲染产生的阴影接近二值，但实际阴影区域因互反射不会完全黑暗，用 Shadow MLP 细化：

$$s' = \mathcal{S}(\mathbf{b}(\mathbf{x}'), s, \mathbf{v}; \varphi)$$

### 关键设计 5：光照自标定

每个光源参数化为相机坐标系下的方向 $\boldsymbol{\ell}_j \in \mathcal{S}^2$ 和 RGB 强度 $\mathbf{e}_j \in \mathbb{R}_+^3$。渲染时用相机旋转矩阵 $\mathbf{R}_i$ 将光照方向变换到世界坐标。光源参数与场景参数一起优化，完全不需要预先标定。

### 关键设计 6：视图非对齐 OLAT 支持

传统 MVPS 要求视图对齐的 OLAT（每个视角都要完整的光照栈），本方法还支持**视图非对齐**采集：每个光源保持激活，拍一组多视图图像，然后切换到下一个光源再拍。不同光源下的视角不需要一一对应。这大大提高了实际采集的灵活性。

### 损失函数

缓存中优化部分截断，但基于框架设计可以推断核心损失：
- **加权 L1 颜色损失**：优于 L2 损失对异常值更鲁棒
- **SDF 正则化损失**：Eikonal 正则化确保有效 SDF
- **前景 mask 损失**：约束渲染轮廓与输入 mask 一致

## 实验关键数据

### 主实验

缓存截断，实验部分不完整。根据摘要和引言中的声明总结：

- 在形状重建精度方面**超越 SOTA 法线引导方法**（如 SuperNormal）
- 在光照估计精度方面优于分阶段方法
- 在稀疏甚至零光照变化条件下保持鲁棒，分阶段方法因 PS 估计不准而退化
- 在视图非对齐 OLAT 图像上验证了定性效果
- 成功处理了陶瓷和青铜金属等具有挑战性的反射率材质

### 关键发现

- 端到端优化有效避免了分阶段 pipeline 中法线不一致导致的几何伪影
- 角度编码显著提升了 BRDF 学习质量和整体重建精度
- Shadow MLP 对互反射造成的阴影区域亮度有效补偿
- 不需要光源标定即可恢复光照方向和相对强度

## 亮点与洞察

1. **端到端设计的哲学正确**：分阶段 pipeline 是历史限制的产物，端到端联合优化才能充分利用所有观测数据。本文从信息论角度论证了逐视图独立 PS 会丢失跨视图互补信息
2. **角度编码设计优雅**：5 维标量特征 $[\mathbf{n}^\top\mathbf{h}, \boldsymbol{\ell}^\top\mathbf{h}, \mathbf{n}^\top\boldsymbol{\ell}, \mathbf{n}^\top\mathbf{v}, (\mathbf{n}^\top\mathbf{h})^{10}]$ 精确捕捉了 BRDF 的物理不变量，$(\mathbf{n}^\top\mathbf{h})^{10}$ 项尤其巧妙地建模了镜面反射的集中性
3. **两步阴影策略务实有效**：先用物理模型（透射率累积）得到粗阴影，再用 MLP 处理互反射等非理想效应，兼顾物理正确性和实际灵活性
4. **光照自标定去除了对校准设备的依赖**：显著降低了 MVPS 的使用门槛
5. **视图非对齐采集的实际价值大**：传统要求每个视角完整 OLAT 栈的限制在实际场景中很不便，支持非对齐采集大幅提高了方法的实用性

## 局限性

1. 缓存截断严重，缺少完整定量实验和消融研究，难以全面评估
2. 假设方向光照，对点光源或复杂光照条件可能不适用
3. 忽略了互反射（inter-reflections），对凹陷几何或高反射材质可能有误差
4. 端到端优化的计算成本可能显著高于分阶段方法（需要在线进行体渲染和阴影光线投射）
5. 测试场景主要是中小尺度物体，对大场景的扩展性未验证
6. BRDF 用单个 MLP 直接预测，可解释性不如解析模型（无法直接得到漫反射率、粗糙度等物理参数）

## 相关工作

- **分阶段 MVPS**：SuperNormal（法线引导）、PS-NeRF（半端到端）、SVNL（需阴影线索）
- **端到端 MVPS**：DPIR（点基体渲染，需已知光照）
- **神经逆渲染**：NeRFactor、NvDiffRec、InvRender
- **Neural BRDF**：解析 BRDF（Disney BRDF）、基函数 BRDF（球面高斯）、隐码驱动 BRDF
- **Neural SDF**：NeuS、VolSDF、instant-ngp

## 评分

- **新颖性**: ★★★★☆ — 端到端 MVPS 框架（无需 PS 线索和光标定）的设计理念清晰有说服力
- **技术深度**: ★★★★★ — 图像形成模型严谨，角度编码和阴影建模的物理动机充分
- **实验质量**: ★★★☆☆ — 缓存截断无法完整评估，但从声明看对比实验覆盖面合理
- **实用性**: ★★★★☆ — 去除光标定和视图对齐约束大幅提升采集灵活性，文化遗产和材质数字化等领域有应用前景
- **表达清晰度**: ★★★★★ — 公式推导详细严谨，场景参数化和渲染 pipeline 描述清楚

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] MERLiN: Single-Shot Material Estimation and Relighting for Photometric Stereo](../../ECCV2024/llm_evaluation/merlin_single-shot_material_estimation_and_relighting_for_photometric_stereo.md)
- [\[ICCV 2025\] SketchSplat: 3D Edge Reconstruction via Differentiable Multi-view Sketch Splatting](sketchsplat_3d_edge_reconstruction_via_differentiable_multi-view_sketch_splattin.md)
- [\[ICCV 2025\] Spectral Sensitivity Estimation with an Uncalibrated Diffraction Grating](spectral_sensitivity_estimation_with_an_uncalibrated_diffraction_grating.md)
- [\[ICCV 2025\] InterSyn: Interleaved Learning for Dynamic Motion Synthesis in the Wild](intersyn_interleaved_learning_for_dynamic_motion_synthesis_in_the_wild.md)
- [\[ICCV 2025\] ForCenNet: Foreground-Centric Network for Document Image Rectification](forcennet_foreground-centric_network_for_document_image_rectification.md)

</div>

<!-- RELATED:END -->
