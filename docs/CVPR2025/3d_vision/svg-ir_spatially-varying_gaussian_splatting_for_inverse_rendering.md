---
title: >-
  [论文解读] SVG-IR: Spatially-Varying Gaussian Splatting for Inverse Rendering
description: >-
  [CVPR 2025][3D视觉][逆渲染] 本文提出 SVG-IR 框架，通过引入空间变化高斯（SVG）表示让单个高斯原语拥有空间变化的材质和法线参数，并结合基于物理的间接光照模型，在保持实时渲染速度的同时，重光照质量超越 NeRF 方法 2.5 dB、超越现有高斯方法 3.5 dB。
tags:
  - CVPR 2025
  - 3D视觉
  - 逆渲染
  - 3D高斯溅射
  - 空间变化材质
  - 物理间接光照
  - 重光照
---

# SVG-IR: Spatially-Varying Gaussian Splatting for Inverse Rendering

**会议**: CVPR 2025  
**arXiv**: [2504.06815](https://arxiv.org/abs/2504.06815)  
**代码**: https://github.com/learner-shx/SVG-IR  
**领域**: 3D视觉  
**关键词**: 逆渲染, 3D高斯溅射, 空间变化材质, 物理间接光照, 重光照

## 一句话总结

本文提出 SVG-IR 框架，通过引入空间变化高斯（SVG）表示让单个高斯原语拥有空间变化的材质和法线参数，并结合基于物理的间接光照模型，在保持实时渲染速度的同时，重光照质量超越 NeRF 方法 2.5 dB、超越现有高斯方法 3.5 dB。

## 研究背景与动机

**领域现状**：从多视角图像重建 3D 资产（逆渲染）是计算机图形学和视觉的长期任务。3D 高斯溅射（3DGS）在新视角合成（NVS）上表现出色，训练和渲染速度快。一些方法将其扩展到重光照，通过将辐射量分解为 BRDF 材质参数和光照。

**现有痛点**：现有基于高斯的逆渲染方法（如 GS-IR、Relightable 3DGS）在重光照时存在明显伪影。根本原因是每个高斯具有恒定的材质参数和法线，导致在给定视角和光照方向下产生均匀颜色。然而实际中，一个高斯可能覆盖多个像素，这些像素应有不同颜色。此外，这些方法建模间接光照时缺乏物理约束，使用不受约束的球谐函数学习残差光照，导致不自然的间接照明，在新光源下也不会改变。

**核心矛盾**：单个高斯的"恒定属性"假设与真实场景中材质空间变化的事实之间存在根本冲突；无物理约束的间接光学习无法支持新光源下的真实重光照。

**本文目标**：(1) 设计更强大的高斯表示以支持空间变化的材质；(2) 引入物理约束的间接光照模型以实现自然的重光照效果。

**切入角度**：受计算机图形学中从平面着色（flat shading）到插值着色（interpolated shading）演进的启发，在高斯上定义多个"高斯顶点"（Gaussian vertices），使材质和法线在高斯内部可以空间变化。

**核心 idea**：用带多个顶点的空间变化高斯替代恒定参数高斯，配合类似三角形渲染中顶点/片元着色的 SVG splatting 流程，并通过在辐射场中光线追踪计算物理间接光照。

## 方法详解

### 整体框架

SVG-IR 是一个多阶段逆渲染框架。首先用标准 Gaussian Surfels 初始化高斯属性（几何、颜色、不透明度）。然后构建空间变化高斯（SVG），继承预训练属性并为每个高斯添加 M=4 个高斯顶点，每个顶点有独立的 albedo、roughness 和法线偏移。框架通过 SVG splatting 渲染图像，使用基于物理的光照模型（含直接和间接光照），通过损失函数优化高斯顶点属性。

### 关键设计

1. **空间变化高斯表示（Spatially-varying Gaussian, SVG）**:

    - 功能：突破每个高斯只能有恒定材质的限制，使单个高斯原语内部可以有不同的材质和法线
    - 核心思路：在基于 2D 高斯 surfel 的切线空间中为每个高斯定义 M 个代表性位置（Gaussian vertices），每个顶点有独立的 albedo $\boldsymbol{a}_i$、roughness $r_i$ 和法线偏移 $\Delta N_i$。查询高斯内任意位置的属性时，通过双线性插值在顶点间插值。恒定高斯是 SVG 的特例（M=1）
    - 设计动机：如图 2 所示，恒定高斯需要多个高斯才能拟合一个 BRDF 分布，而 SVG 只需一个即可通过不同位置的不同 BRDF lobe 完成拟合，表达能力更强

2. **SVG Splatting（高斯顶点/片元着色）**:

    - 功能：实现空间变化高斯的高效渲染，支持每像素的材质插值
    - 核心思路：类比三角形渲染管线，分为顶点着色和片元着色两个阶段。顶点着色阶段：使用物理光照模型计算每个高斯顶点的辐射量，得到 M 个颜色 $\mathbf{c}^{\{M\}_i}$。片元着色阶段：对于每条相机光线，在高斯切线空间中计算像素对应的 $(u, v)$ 坐标，用双线性插值得到该像素处的颜色，再通过 alpha blending 混合所有高斯的贡献。关键公式为 $\mathbf{c}_i = \text{BilinearInterp}(\mathbf{c}_i^{\{M\}}, u, v)$
    - 设计动机：直接利用计算机图形学中成熟的顶点/片元着色范式，既与 3DGS 的 rasterization 流程兼容，又能高效处理空间变化属性

3. **基于物理的间接光照模型**:

    - 功能：通过在辐射场中进行光线追踪，显式建模光传输过程，实现物理合理的间接光照
    - 核心思路：对每个高斯在上半球均匀采样 K 个方向，将高斯看作椭圆构建 BVH 加速光线追踪。沿每条射线累积透射率和辐射量，得到间接入射辐射量 $L_i^{ind}$。入射光被分解为：$L_i(x, \omega_i) = L_i^{dir}(x, \omega_i)V(x, \omega_i) + L_i^{ind}(x, \omega_i)$，其中 $V$ 为可见性，由透射率阈值判断。重光照时，用一跳间接光照替换训练时学到的间接光照，通过查询二级高斯并在新光源下计算其出射辐射量实现动态间接光照
    - 设计动机：现有方法用不受约束的球谐函数学习间接光照，虽然提升 NVS 质量，但间接光照不自然且无法随新光源变化。物理约束使材质和光照解耦更合理，支持新光源下的真实间接照明

### 损失函数 / 训练策略

总损失包含多项：$\mathcal{L} = \lambda_1\mathcal{L}_1 + \lambda_{ssim}\mathcal{L}_{ssim} + \lambda_{rc}\mathcal{L}_{rc} + \lambda_n\mathcal{L}_n + \lambda_{s,a}\mathcal{L}_{s,a} + \lambda_{s,r}\mathcal{L}_{s,r} + \lambda_{reg,n}\mathcal{L}_{reg,n}$。

其中 $\mathcal{L}_1$ 和 $\mathcal{L}_{ssim}$ 为渲染图像与真值的 L1 和 SSIM 损失；$\mathcal{L}_{rc}$ 为辐射一致性损失，利用光线追踪得到的间接辐射量监督二级高斯的出射辐射量；$\mathcal{L}_n$ 为法线一致性损失；$\mathcal{L}_{s,a}$ 和 $\mathcal{L}_{s,r}$ 为 albedo 和 roughness 的总变分平滑正则；$\mathcal{L}_{reg,n}$ 为法线偏移的 L2 正则项。

## 实验关键数据

### 主实验（重光照 PSNR）

| 方法 | TensoIR 平均 | ADT 平均 | 类型 |
|------|------------|---------|------|
| MII | 27.76 | 29.74 | NeRF |
| TensoIR | 28.53 | 30.58 | NeRF |
| GS-Shader | 19.89 | 23.04 | GS |
| GS-IR | 24.69 | 26.95 | GS |
| RelightGS | 27.60 | 32.84 | GS |
| **Ours** | **31.10** | **34.69** | GS |

超越最好的 NeRF 方法 2.5 dB，超越最好的 GS 方法 3.5 dB，训练时间仅 1 小时（NeRF 方法需 5-6h）。

### 消融实验

| SVG | PBI | 重光照 PSNR | NVS PSNR |
|-----|-----|-----------|---------|
| ✗ | ✗ | 28.614 | 34.640 |
| ✓ | ✗ | 29.447 | 35.794 |
| ✓ | ✓ | **31.087** | **36.709** |

### 关键发现

- SVG 表示单独即可提升重光照 PSNR 约 0.8 dB，结合物理间接光照再提升 1.6 dB
- Relightable 3DGS 的间接光照在不同环境光下保持不变（导致暗光下异常明亮），本方法可动态适应新光源
- NVS 任务上同样超越所有对比方法，TensoIR Synthetic 上 PSNR 达 36.71（vs TensoIR 的 35.17）
- GPU 内存开销增加 50%-80%（相比 3DGS/2DGS），10%-20%（相比 Relightable 3DGS）

## 亮点与洞察

- **从图形学经典范式获取灵感**：将 flat shading → interpolated shading 的演进迁移到 3DGS，在单个高斯内实现空间变化材质，概念简洁且高效
- **物理间接光照的实用方案**：通过在已有辐射场上做光线追踪获得间接光照，避免了额外训练参数，且支持新光源下的动态更新
- **辐射一致性损失妙用**：将间接辐射量反过来作为二级高斯出射辐射的监督，从额外视角提供指导

## 局限与展望

- 未引入几何先验（如 SDF），对高镜面反射物体的恢复效果不理想
- 高斯顶点带来额外的 GPU 内存开销
- 间接光照仅考虑一跳反射，复杂多跳场景可能不够精确
- 未来可结合 GS-ROR 等引入 SDF 改善几何精度

## 相关工作与启发

- NeRF 系的逆渲染（TensoIR、InvRender）质量高但速度慢，3DGS 系方法速度快但质量受限于恒定材质假设
- 本文的 SVG 思路对所有基于高斯的方法都有启发：任何需要表达高斯内空间变化属性的任务都可以参考这种 vertex-based 插值设计
- 物理间接光照的一跳方案在实时性和质量间取得了较好平衡

## 评分

- **新颖性**: 9/10 — 空间变化高斯的概念新颖且与图形学范式优雅对应
- **实验充分度**: 8/10 — 合成和真实数据集全面评测，消融清晰
- **写作质量**: 8/10 — 逻辑流畅，图示直观
- **价值**: 9/10 — 对 3DGS 逆渲染方向有显著推动

<!-- RELATED:START -->

## 相关论文

- [PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields](pbr-nerf_inverse_rendering_with_physics-based_neural_fields.md)
- [IRIS: Inverse Rendering of Indoor Scenes from Low Dynamic Range Images](iris_inverse_rendering_of_indoor_scenes_from_low_dynamic_range_images.md)
- [GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering](../../ICCV2025/3d_vision/geosplatting_towards_geometry_guided_gaussian_splatting_for_physically-based_inv.md)
- [Gaussian Splatting with Discretized SDF for Relightable Assets](../../ICCV2025/3d_vision/gaussian_splatting_with_discretized_sdf_for_relightable_assets.md)
- [HRAvatar: High-Quality and Relightable Gaussian Head Avatar](hravatar_high-quality_and_relightable_gaussian_head_avatar.md)

<!-- RELATED:END -->
