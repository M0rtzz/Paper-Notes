---
title: >-
  [论文解读] SpectroMotion: Dynamic 3D Reconstruction of Specular Scenes
description: >-
  [CVPR 2025][3D视觉][3D高斯溅射] SpectroMotion 基于 3DGS 框架，通过可变形高斯 MLP 建模动态物体、可变形反射 MLP 建模时变光照效果，并结合规范环境贴图和粗到细的三阶段训练策略，首次实现了对动态镜面场景的高质量 3D 重建和实时渲染。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D高斯溅射
  - 动态场景
  - 镜面反射
  - 环境光照
  - 可变形高斯
---

# SpectroMotion: Dynamic 3D Reconstruction of Specular Scenes

**会议**: CVPR 2025  
**arXiv**: [2410.17249](https://arxiv.org/abs/2410.17249)  
**代码**: 无  
**领域**: 3D视觉 / 动态场景重建  
**关键词**: 3D高斯溅射, 动态场景, 镜面反射, 环境光照, 可变形高斯

## 一句话总结

SpectroMotion 基于 3DGS 框架，通过可变形高斯 MLP 建模动态物体、可变形反射 MLP 建模时变光照效果，并结合规范环境贴图和粗到细的三阶段训练策略，首次实现了对动态镜面场景的高质量 3D 重建和实时渲染。

## 研究背景与动机

1. **领域现状**：3D 高斯溅射（3DGS）在静态场景的新视角合成中取得了突破性进展，同时 Deformable 3DGS 等方法将其扩展到了动态场景。另一方面，GaussianShader 和 GS-IR 等工作处理了静态场景中的镜面反射。但是，同时处理动态运动和镜面反射的交叉问题仍然是一个未解决的挑战。
2. **现有痛点**：(1) Deformable 3DGS、4DGS 等动态方法使用球谐函数（SH）建模颜色，无法准确表示视角相关的镜面反射；(2) GaussianShader、GS-IR 等镜面反射方法仅处理静态场景，无法应对物体运动和时变光照；(3) NeRF-DS 虽然专门针对动态镜面场景，但基于 NeRF 体积渲染，速度慢且质量有限。
3. **核心矛盾**：动态场景中的镜面物体，其反射外观不仅随视角变化，还随物体运动和环境光照的时间变化而改变，存在几何变形、材质属性和光照条件的三重耦合。
4. **本文目标**：在 3DGS 框架下，统一建模动态物体运动和镜面反射效果，实现高质量渲染和可靠的几何/材质分解。
5. **切入角度**：将最终颜色分解为漫反射和镜面反射两部分，分别用不同机制建模；用分阶段的训练策略逐步引入几何变形、法线优化和镜面反射能力。
6. **核心 idea**：结合可变形高斯 MLP（处理物体运动）+ 规范环境贴图（时不变光照基准）+ 可变形反射 MLP（时变光照偏差），并采用由粗到细的三阶段训练策略（静态→动态→镜面），稳定地优化所有组件。

## 方法详解

### 整体框架

输入为单目视频序列，输出为动态场景的 3DGS 表示。3D 高斯在规范空间定义，通过可变形高斯 MLP 预测每个时间步的位置、旋转和缩放偏移。颜色表示被分解为 $c_{\text{final}} = c_{\text{diffuse}} + c_{\text{specular}}$，其中漫反射部分用零阶球谐函数，镜面反射部分通过查询环境贴图得到基础反射颜色，再由可变形反射 MLP 预测时变光照偏移。训练分三个阶段逐步进行。

### 关键设计

1. **可变形高斯 MLP**:
    - 功能：建模场景中物体的动态运动，预测每个 3D 高斯在不同时间步的变形（位置、旋转、缩放偏移）
    - 核心思路：遵循 Deformable 3DGS 的设计，输入为 3D 高斯的空间坐标和时间信息，经过 8 层 FC（256 维隐藏层，ReLU 激活）得到 256 维特征向量，再分三个分支分别输出位置、旋转和缩放的偏移量。第 4 层采用跳跃连接（类似 NeRF），拼接输入与中间特征。
    - 设计动机：将动态建模从颜色表示中解耦出来，使得后续的镜面反射建模可以在稳定的几何基础上进行。无需 mask 监督即可自动区分动态和静态物体。

2. **规范环境贴图 + 可变形反射 MLP**:
    - 功能：分别建模时不变的基础光照和时变的光照效果
    - 核心思路：环境贴图使用 $6 \times 128 \times 128$ 的可学习 cubemap 参数，表示场景的规范（平均/基准）光照条件。给定高斯的法线方向和相机视角，通过物理反射方程计算反射方向，查询环境贴图获得基础镜面颜色。可变形反射 MLP 学习从时间到光照偏移的映射，捕获因物体运动导致的反射外观变化。最终镜面颜色结合镜面色调（specular tint）和粗糙度（roughness）属性。
    - 设计动机：将光照分解为时不变基准 + 时变偏差，减少了学习难度。仅靠 SH 无法准确建模镜面高光，环境贴图配合物理反射模型能更精确地表示视角相关的反射效果。

3. **粗到细三阶段训练策略**:
    - 功能：解决动态、几何和镜面反射之间的优化耦合问题
    - 核心思路：**静态阶段**（3k iter）：训练标准 3DGS 稳定静态几何。**动态阶段**（6k iter）：引入可变形高斯 MLP，前 3k iter 优化基本变形，后 3k iter 加入法线损失 $\mathcal{L}_{\text{normal}}$ 同时优化法线和深度。**镜面阶段**（31k iter）：将 SH 颜色切换为 $c_{\text{final}}$，冻结可变形高斯 MLP 和大部分参数，仅优化零阶 SH、镜面色调、粗糙度，6k iter 后解冻所有参数。前 2k iter 仅优化规范环境贴图，之后引入可变形反射 MLP。总共 40k iter。
    - 设计动机：如果所有组件同时优化，不完整的颜色表示会破坏已学到的几何。分阶段策略确保每个新引入的组件有稳定的基础。先学动态再学反射，避免动态运动和镜面效果的优化冲突。

### 损失函数 / 训练策略

- 标准 3DGS 重建损失（L1 + SSIM）
- 法线一致性损失 $\mathcal{L}_{\text{normal}}$：约束渲染法线与深度推导法线一致
- Adam 优化器，总 40,000 iterations
- 自适应高斯密度化和剪枝策略

## 实验关键数据

### 主实验

| 方法 | NeRF-DS Mean PSNR↑ | Mean SSIM↑ | Mean LPIPS↓ |
|------|-------------------|-----------|------------|
| Deformable 3DGS | 19.66 | 0.5826 | 0.3181 |
| 4DGS | 18.09 | 0.4649 | 0.4078 |
| GaussianShader | 14.98 | 0.3681 | 0.6121 |
| GS-IR | 15.05 | 0.3678 | 0.5856 |
| NeRF-DS | 18.74 | 0.5151 | 0.4337 |
| HyperNeRF | 16.23 | 0.5007 | 0.4420 |
| **SpectroMotion** | **20.08** | **0.5909** | **0.3094** |

### 消融实验

| 场景 | SpectroMotion PSNR | Deformable 3DGS PSNR | 提升 | 说明 |
|------|-------------------|---------------------|------|------|
| As | 24.51 | 24.14 | +0.37 | 镜面效果较弱的场景 |
| Bell | 19.60 | 19.42 | +0.18 | 包含较强镜面反射 |
| Cup | 20.13 | 20.10 | +0.03 | 差距小但仍最优 |
| Plate | 16.53 | 16.12 | +0.41 | 强镜面场景提升明显 |
| Press | 21.70 | 19.64 | +2.06 | 最大提升，复杂镜面动态 |
| Sieve | 20.36 | 20.74 | -0.38 | 唯一被 Deformable 3DGS 超越的场景 |

### 关键发现

- 在动态镜面物体的专项评估中（使用 Track Anything 生成的动态镜面 mask），SpectroMotion 全面领先所有方法
- 无需 mask 监督即可自动区分动态和静态物体（通过可变形高斯 MLP 的变形幅度可视化验证）
- 漫反射/镜面反射分解结果在视觉上合理——镜面部分集中在光滑金属表面
- 大多数场景高斯数 <200k，可达到 ≥30 FPS 实时渲染
- 训练时间约 1-2 小时（RTX 4090），远快于 NeRF-DS
- 局限：在剧烈场景变化（如手臂进出画面）时会产生浮块

## 亮点与洞察

- **首次在 3DGS 框架下统一动态和镜面反射**：此前动态 3DGS 和镜面 3DGS 是分开研究的两个方向，SpectroMotion 将两者优雅地统一在一个框架中，这本身就是一个重要贡献。
- **分阶段训练策略的稳健性**：从静态→动态→镜面的渐进式训练，巧妙地避免了多个学习目标之间的冲突。特别是镜面阶段先冻结几何再解冻的设计，平衡了新旧组件的优化。
- **时变光照的分解思路**：将光照分解为规范环境贴图（基准值）+ 可变形反射 MLP（时变偏差），这个设计可以迁移到其他需要处理时变外观的任务中。

## 局限与展望

- 无法处理剧烈的场景变化（如新物体进入/离开场景），依赖稳定的前景物体
- 仅用单目视频，在几何复杂区域可能出现歧义
- 环境贴图假设全局光照，无法建模局部遮挡造成的阴影变化
- 未来改进：结合 4DGS 的时空体素表示处理剧烈运动、引入物理约束的材质模型（BRDF）、扩展到多视角输入

## 相关工作与启发

- **vs Deformable 3DGS**: 仅建模动态但用 SH 表示颜色，无法处理镜面反射。SpectroMotion 在其基础上增加了镜面反射的完整建模方案，同时保持了动态建模能力。
- **vs NeRF-DS**: 唯一之前处理动态镜面场景的方法，但基于 NeRF 体积渲染，速度慢（无法实时）。SpectroMotion 用 3DGS 实现了实时渲染，同时质量更好（PSNR +1.34，LPIPS -0.1243）。
- **vs GaussianShader/GS-IR**: 仅处理静态镜面场景。SpectroMotion 证明了他们的静态镜面表示方案在动态场景下会完全失效（PSNR 仅 14-15）。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次在 3DGS 中统一动态和镜面反射，但各组件（deformable MLP、环境贴图）是已有技术的组合
- 实验充分度: ⭐⭐⭐ 仅在 NeRF-DS 数据集上评估，缺少 HyperNeRF 等其他数据集的完整评估
- 写作质量: ⭐⭐⭐⭐ 训练策略描述清晰，但方法公式化描述较少
- 价值: ⭐⭐⭐⭐ 填补了动态镜面 3DGS 重建的空白，有明确的应用场景

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] ODHSR: Online Dense 3D Reconstruction of Humans and Scenes from Monocular Videos](odhsr_online_dense_3d_reconstruction_of_humans_and_scenes_from_monocular_videos.md)
- [\[NeurIPS 2025\] D$^2$USt3R: Enhancing 3D Reconstruction for Dynamic Scenes](../../NeurIPS2025/3d_vision/d2ust3r_enhancing_3d_reconstruction_for_dynamic_scenes.md)
- [\[CVPR 2025\] IRIS: Inverse Rendering of Indoor Scenes from Low Dynamic Range Images](iris_inverse_rendering_of_indoor_scenes_from_low_dynamic_range_images.md)
- [\[CVPR 2025\] Horizon-GS: Unified 3D Gaussian Splatting for Large-Scale Aerial-to-Ground Scenes](horizon-gs_unified_3d_gaussian_splatting_for_large-scale_aerial-to-ground_scenes.md)
- [\[CVPR 2025\] LIM: Large Interpolator Model for Dynamic Reconstruction](lim_large_interpolator_model_for_dynamic_reconstruction.md)

<!-- RELATED:END -->
