---
title: >-
  [论文解读] EnvGS: Modeling View-Dependent Appearance with Environment Gaussian
description: >-
  [CVPR 2025][3D视觉][高斯溅射] 本文提出EnvGS，用一组环境高斯原语（Environment Gaussian）作为显式3D表示来捕获场景反射，通过基于GPU RT Core的可微光线追踪渲染器联合优化环境高斯和基础高斯，在真实场景中首次实现了实时（26+ FPS）且高质量的镜面反射新视角合成，显著超越所有实时方法。
tags:
  - CVPR 2025
  - 3D视觉
  - 高斯溅射
  - 反射建模
  - 光线追踪
  - 2D高斯
  - 新视角合成
---

# EnvGS: Modeling View-Dependent Appearance with Environment Gaussian

**会议**: CVPR 2025  
**arXiv**: [2412.15215](https://arxiv.org/abs/2412.15215)  
**代码**: [https://zju3dv.github.io/envgs](https://zju3dv.github.io/envgs)  
**领域**: 3D视觉  
**关键词**: 高斯溅射, 反射建模, 光线追踪, 2D高斯, 新视角合成

## 一句话总结
本文提出EnvGS，用一组环境高斯原语（Environment Gaussian）作为显式3D表示来捕获场景反射，通过基于GPU RT Core的可微光线追踪渲染器联合优化环境高斯和基础高斯，在真实场景中首次实现了实时（26+ FPS）且高质量的镜面反射新视角合成，显著超越所有实时方法。

## 研究背景与动机

1. **领域现状**：3D高斯溅射（3DGS）通过显式高斯原语和光栅化实现了实时新视角合成，但使用球谐函数（SH）建模视角依赖效果，表达能力有限，面对强镜面反射时效果差。

2. **现有痛点**：GaussianShader和3DGS-DR等方法引入环境贴图来增强反射建模，但存在两个根本问题——(a) 环境贴图假设远场光照，无法建模近场反射（如桌面上物体的倒影）；(b) 低频环境贴图表达力不足以捕获高频反射细节。

3. **核心矛盾**：需要一种表示既能建模任意复杂度的反射（近场+远场+高频），又要保持实时渲染速度。NeRF-based方法（如NeRF-Casting）质量好但速度太慢（<0.1 FPS），高斯方法快但反射质量差。

4. **本文目标**：(1) 如何用显式3D表示建模复杂反射？(2) 如何高效渲染反射方向的查询？(3) 如何联合优化反射表示和场景几何？

5. **切入角度**：与其用2D环境贴图"投影"地表示环境光照，不如用另一组高斯原语在3D空间中显式表示反射内容。高斯原语天然支持高频细节和近场位置信息。

6. **核心 idea**：用环境高斯替代环境贴图来显式建模3D反射 + 基于OptiX RT Core的可微光线追踪来高效渲染。

## 方法详解

### 整体框架
场景由两组2D高斯原语表示：基础高斯 $\mathbf{P}_{base}$ 建模几何和漫反射外观，环境高斯 $\mathbf{P}_{env}$ 建模反射。渲染分三步：(1) 光栅化基础高斯得到法线 $\mathbf{n}$、基础颜色 $\mathbf{c}_{base}$、混合权重 $\beta$ 和表面位置 $\mathbf{x}$；(2) 沿反射方向 $\mathbf{d}_{ref}$ 从表面点 $\mathbf{x}$ 发射光线，通过光线追踪渲染环境高斯得到反射颜色 $\mathbf{c}_{ref}$；(3) 混合得最终颜色 $\mathbf{c} = (1-\beta) \cdot \mathbf{c}_{base} + \beta \cdot \mathbf{c}_{ref}$。

### 关键设计

1. **环境高斯表示 (Environment Gaussian)**:

    - 功能：在3D空间中显式建模场景的反射内容
    - 核心思路：与基础高斯共享相同的2DGS参数化（中心位置、不透明度、切向向量、缩放、SH系数），但代表的是环境/反射信息而非场景几何。每个基础高斯额外学一个混合权重 $\beta$ 控制反射/漫反射的混合比例。环境高斯初始化为将场景包围盒分成 $32^3$ 个子格，每格随机采样5个原语，在bootstrapping阶段后与基础高斯联合优化。
    - 设计动机：相比环境贴图，高斯原语有精确的3D位置信息，天然支持近场反射；高斯原语数量可以自适应增长（通过densification），理论上可以表示任意高频的反射细节。

2. **可微光线追踪渲染器 (Differentiable Gaussian Tracer)**:

    - 功能：高效渲染从每个表面点沿反射方向看到的环境高斯颜色
    - 核心思路：将每个2D高斯用两个三角形包围（4个顶点变换到世界空间），构建BVH加速结构。使用OptiX的raygen和anyhit可编程入口点实现分chunk渲染——anyhit收集k=16个最近交点并排序，raygen按深度顺序做体积渲染积分。当累计透射率低于阈值或无更多交点时停止。反向传播以与前向相同的front-to-back顺序重新发射光线计算梯度，关键计算 $\partial \mathcal{L}/\partial \mathbf{o}$ 和 $\partial \mathcal{L}/\partial \mathbf{d}$（对光线原点和方向的梯度），实现基础和环境高斯的联合优化。在RTX 4090上可达30 FPS渲染200万2DGS。
    - 设计动机：反射渲染中每个像素对应唯一的反射光线（不同原点和方向），无法用标准光栅化。利用现代GPU的RT Core硬件加速光线-三角形求交是同时满足速度和灵活性的唯一选择。

3. **单目法线监督与联合优化**:

    - 功能：解决反射/折射场景中几何歧义问题，确保基础高斯重建出正确的表面法线
    - 核心思路：除标准的法线一致性约束 $\mathcal{L}_{norm}$（渲染法线vs深度图梯度法线）外，引入预训练单目法线估计模型的预测 $\mathbf{N}_m$ 作为额外监督 $\mathcal{L}_{mono} = \frac{1}{N_p} \sum (1 - \mathbf{n}_i^\top \mathbf{N}_m)$。联合优化时，光线追踪的梯度通过反射方向反传到基础高斯的法线参数，形成几何-外观的闭环优化。
    - 设计动机：反射/折射面的法线对反射方向极为敏感，微小的法线偏差导致反射方向偏移很大。单目法线提供了与视角无关的prior，避免高斯为了拟合反射而扭曲几何（"foggy" geometry问题）。

### 损失函数 / 训练策略
$\mathcal{L} = \mathcal{L}_{rgb} + 0.04 \cdot \mathcal{L}_{norm} + 0.01 \cdot \mathcal{L}_{mono} + 0.01 \cdot \mathcal{L}_{perc}$，其中 $\mathcal{L}_{rgb}$ 为0.8*L1+0.2*D-SSIM，$\mathcal{L}_{perc}$ 为VGG-16感知损失。先单独训练基础高斯（bootstrapping），再初始化环境高斯开始联合训练。使用3DGS的自适应密度控制+3DGS-DR的法线传播和颜色破坏策略。

## 实验关键数据

### 主实验

| 数据集 | 方法类型 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FPS |
|--------|---------|------|-------|-------|--------|-----|
| Ref-Real | 实时 | 3DGS | 23.70 | 0.641 | 0.262 | 182 |
| Ref-Real | 实时 | 2DGS | 23.80 | 0.654 | 0.281 | 159 |
| Ref-Real | 实时 | GaussianShader | 22.88 | 0.622 | 0.314 | 28 |
| Ref-Real | 实时 | **EnvGS** | **24.62** | **0.671** | **0.241** | 26 |
| Ref-Real | 非实时 | NeRF-Casting | 24.67 | 0.659 | 0.246 | <0.1 |
| NeRF-Cast Scenes | 实时 | **EnvGS** | **30.44** | **0.886** | **0.148** | 26 |
| NeRF-Cast Scenes | 非实时 | NeRF-Casting | 31.02 | 0.889 | 0.128 | <0.1 |

### 消融实验

| 配置 | PSNR | SSIM | 说明 |
|------|------|------|------|
| Full EnvGS | 24.62 | 0.671 | 完整模型 |
| w/ 环境贴图替代环境高斯 | - | - | 无法捕获近场反射 |
| w/o 单目法线约束 | - | - | 几何噪声大，反射重建不准 |
| w/o 联合优化 | - | - | 法线不准导致反射方向偏移 |

### 关键发现
- EnvGS在所有实时方法中排名第一，在Ref-Real上PSNR超第二名2DGS约0.8 dB，LPIPS改善14%
- 与非实时SOTA NeRF-Casting性能非常接近（PSNR差0.05），但速度快100倍以上
- 在前景和近场区域，EnvGS优势更明显（前景PSNR 33.30 vs 3DGS 31.68，近场PSNR 46.39 vs 44.16）
- 联合优化至关重要——不联合优化会导致法线不准确，进而反射方向偏移
- 训练时间约2.5小时（RTX 4090），比NeRF-Casting（>47h）快20倍

## 亮点与洞察
- **用高斯表示环境而非环境贴图**：这个概念转换非常自然但效果巨大——同样的表示用于场景和环境，统一优化，且天然支持近场反射。
- **OptiX RT Core加速的可微光线追踪**：将硬件加速光线追踪引入高斯渲染是技术上的重要贡献，chunk-by-chunk的前向渲染 + 重投射的反向传播策略优雅地解决了内存和速度限制。
- **实时反射渲染的里程碑**：据作者所知是首个在真实场景中实现实时高质量镜面反射合成的方法，对VR/AR应用有直接价值。

## 局限与展望
- FPS约26，虽然实时但比不做反射的3DGS（182 FPS）慢很多，对性能敏感的应用仍有差距
- 环境高斯初始化为均匀网格采样，对场景尺度敏感，极大或极小场景可能需要调参
- 仅处理镜面反射，未考虑散射、折射等更复杂的光学现象
- 依赖较新的GPU（RTX 4090的RT Core），旧GPU上性能可能大幅下降

## 相关工作与启发
- **vs GaussianShader/3DGS-DR**: 这些方法用2D环境贴图建模反射，只能处理远场光照；EnvGS用3D高斯替代，自然支持近场+远场
- **vs NeRF-Casting**: NeRF-Casting沿反射方向做ray marching+MLP解码，质量最好但极慢；EnvGS用显式高斯+硬件加速光线追踪达到接近的质量但快100倍
- **vs 3iGS**: 3iGS用张量分解建模照明场，但限于有界场景；EnvGS的显式高斯天然支持无界场景

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 环境高斯概念新颖且优雅，可微光线追踪渲染器有技术深度
- 实验充分度: ⭐⭐⭐⭐ 多数据集对比+消融+前景/近场单独评估，较充分
- 写作质量: ⭐⭐⭐⭐⭐ 方法描述清晰，图示信息量大，结构合理
- 价值: ⭐⭐⭐⭐⭐ 实时反射渲染的重要突破，对3D重建和VR/AR领域有高实用价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Textured Gaussians for Enhanced 3D Scene Appearance Modeling](textured_gaussians_for_enhanced_3d_scene_appearance_modeling.md)
- [\[CVPR 2025\] ReCap: Better Gaussian Relighting with Cross-Environment Captures](recap_better_gaussian_relighting_with_cross-environment_captures.md)
- [\[ICML 2025\] LaGa: Tackling View-Dependent Semantics in 3D Language Gaussian Splatting](../../ICML2025/3d_vision/tackling_view-dependent_semantics_in_3d_language_gaussian_splatting.md)
- [\[CVPR 2025\] Towards Realistic Example-Based Modeling via 3D Gaussian Stitching](towards_realistic_example-based_modeling_via_3d_gaussian_stitching.md)
- [\[NeurIPS 2025\] WildCAT3D: Appearance-Aware Multi-View Diffusion in the Wild](../../NeurIPS2025/3d_vision/wildcat3d_appearance-aware_multi-view_diffusion_in_the_wild.md)

<!-- RELATED:END -->
