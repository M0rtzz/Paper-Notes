---
title: >-
  [论文解读] From Orbit to Ground: Generative City Photogrammetry from Extreme Off-Nadir Satellite Images
description: >-
  [CVPR 2026][3D视觉][城市重建] 提出从稀疏卫星图像重建城市级 3D 模型的两阶段方法：用 Z-Monotonic SDF 建模几何保证建筑结构完整性，再用微调 FLUX 扩散模型做"确定性修复"从退化贴图合成写实纹理，实现从轨道到地面近 90° 视点外推。
tags:
  - "CVPR 2026"
  - "3D视觉"
  - "城市重建"
  - "卫星图像"
  - "2.5D SDF"
  - "纹理修复"
  - "视点外推"
---

# From Orbit to Ground: Generative City Photogrammetry from Extreme Off-Nadir Satellite Images

**会议**: CVPR 2026  
**arXiv**: [2512.07527](https://arxiv.org/abs/2512.07527)  
**代码**: [项目页面](https://pku-vcl-geometry.github.io/Orbit2Ground/)  
**领域**: 3D Vision  
**关键词**: 城市重建, 卫星图像, 2.5D SDF, 纹理修复, 视点外推

## 一句话总结

提出从稀疏卫星图像重建城市级 3D 模型的两阶段方法：用 Z-Monotonic SDF 建模几何保证建筑结构完整性，再用微调 FLUX 扩散模型做"确定性修复"从退化贴图合成写实纹理，实现从轨道到地面近 90° 视点外推。

## 研究背景与动机

**领域现状**：NeRF/3DGS 在物体和街景级重建取得成功，但城市级重建面临数据采集难题——地面/无人机采集成本高且覆盖有限。卫星图像提供廉价的城市级覆盖。

**现有痛点**：卫星图像的极端挑战——源图（俯视）与目标（地面视角）存在近 90° 视点鸿沟；建筑立面严重透视缩短；大气畸变和传感器限制导致纹理退化。NeRF/3DGS 依赖密集视差和清晰光度，在此场景下完全失败。

**核心矛盾**：卫星图像几乎不提供垂直结构的视差信息（MVS 只能恢复屋顶和地面点云），而我们需要从中合成地面级逼真渲染。

**切入角度**：解耦为几何和外观两个子问题——几何用城市特有的 2.5D 先验约束（建筑基本是垂直拉伸），外观用生成模型先验补偿。

**核心 idea**：Z-Monotonic SDF 保证几何结构正确 → 确定性图像修复网络补偿纹理质量。

## 方法详解

### 整体框架

两阶段设计：
- **Stage 1 (Geometry)**：优化 Z-Monotonic SDF → 可微分等值面提取 → 高保真水密网格
- **Stage 2 (Appearance)**：基础纹理反投影 → FLUX 微调修复网络 → 迭代纹理精化

### 关键设计

1. **Z-Monotonic SDF**：

    - 核心约束：SDF 值沿 Z 轴单调不递减 $\frac{\partial s(x,y,z)}{\partial z} \geq 0$
    - **物理意义**：连续表面（屋顶/地面）→ $s=0$ 有唯一解定义高度；建筑边缘 → $s=0$ 在 $[z_{ground}, z_{roof}]$ 区间形成竖直平台，自动生成垂直立面
    - 实现：256×256 2D 网格，每格存储可学习参数 $h_j$，通过 $\tanh$ 激活函数构造单调曲线 + 空间插值
    - 优化目标：$\mathcal{L}_{geo} = \sum_{p} \|p_z - m^*(p)_z\|_1 + \lambda_{Lap}\mathcal{L}_{Lap} + \lambda_{Nrm}\mathcal{L}_{Nrm}$
    - **设计动机**：为什么选 2.5D 而非全 3D？卫星数据天然是俯视的，建筑绝大多数是垂直拉伸结构。牺牲非单调结构（如桥梁）的建模能力，换取对几何歧义的决定性鲁棒性

2. **确定性图像修复网络**：

    - 基于 FLUX-Schnell 微调，10 万对航拍图像训练
    - 关键设计：学习退化渲染→清晰目标的**确定性映射**（而非生成多个候选取平均）
    - 损失：$\mathcal{L}_{restorer} = \mathcal{L}_{LPIPS}(\hat{I}, I_{high}) + \lambda_{CHAR}\mathcal{L}_{CHAR}(\hat{I}, I_{high})$
    - **设计动机**：为什么做确定性修复而非 SDS 蒸馏？生成模型的随机性会产生跨视图冲突细节，确定性修复保证全局一致性，且效率更高（单次通过 vs 多次采样平均）

3. **迭代纹理精化**：

    - 循环：渲染退化视图 → 修复网络增强 → 作为监督优化纹理 → 下一轮输入质量更好
    - 相机路径：150m 均匀网格采样，海拔 450m，俯仰 45°，四个基本方向
    - 仅迭代 2 轮即足够

### 训练策略

- 几何与纹理分别优化，不联合训练
- 几何优化用 Adam (lr=0.01)，纹理用 MSE + SSIM 损失
- 修复网络训练 10K 步，batch 96
- **整体效率**：单 A6000 GPU，1 km² 仅需 1.5 小时

## 实验关键数据

### 主实验

| 方法 | MatrixCity F1↑ | MatrixCity CD↓ | DFC PSNR↑ | GoogleEarth PSNR↑ |
|------|---------------|---------------|-----------|-------------------|
| Mip-Splatting | 0.377 | 0.161 | 10.289 | 12.214 |
| 2DGS | 0.556 | 0.073 | 7.366 | 11.022 |
| CityGS-X | 0.189 | 0.227 | FAIL | 12.674 |
| Skyfall-GS | 0.296 | 0.359 | 12.460 | 12.456 |
| **Ours** | **0.643** | **0.036** | **13.059** | **12.770** |

F1 提升 0.09，Chamfer Distance 降低 50%，DFC 数据集提升尤为显著。

### 消融实验

| 配置 | F1↑ | CD↓ | PSNR↑ |
|------|-----|-----|-------|
| Naive MC 128 | 0.279 | 0.0749 | 16.800 |
| Naive MC 256 | 0.412 | 0.0757 | 17.002 |
| w/o Regularization | 0.637 | 0.0364 | 17.115 |
| **Full Model** | **0.643** | **0.0357** | **17.153** |
| w/o Image Restoration | - | - | 17.038 |

### 关键发现

- Z-Monotonic SDF 显著优于朴素 Marching Cubes 方案（F1: 0.643 vs 0.279/0.412）
- 修复网络贡献不可忽略（PSNR: 17.153 vs 17.038），且定性效果差异巨大
- 其他方法在卫星场景下产生碎片化、漂浮几何或模糊纹理，本方法鲁棒稳定

## 亮点与洞察

- **问题定义清晰**：将卫星→地面定义为"极端视点外推"，抓住了问题核心
- **2.5D 表征巧妙**：利用城市结构的领域先验,将不适定问题转化为约束优化
- **确定性修复 vs 生成式蒸馏**：更实用、更一致、更高效的选择
- **规模化验证**：4 km² 真实城市区域的重建展示了实用性

## 局限与展望

- 2.5D 假设无法处理桥梁、立交桥等非单调结构
- 修复网络需要大量配对航拍训练数据
- 纹理质量最终受限于修复模型的幻觉能力，可能与真实外观不一致
- 分辨率受限于 256×256 网格

## 相关工作与启发

- **GaussianShading / Skyfall-GS**：卫星重建近期工作
- **FLUX**：修复网络的基座扩散模型
- **DreamFusion / SDS**：生成先验的另一路线（本文论证其在此场景下的不适用性）

## 评分

- 新颖性: ⭐⭐⭐⭐ Z-Monotonic SDF 设计精巧，两阶段解耦合理
- 实验充分度: ⭐⭐⭐⭐⭐ 合成+真实+大规模验证全面
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机和方法阐述非常清晰
- 价值: ⭐⭐⭐⭐ 卫星重建实用价值高，城市规划/仿真应用明确

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Gaussian Splatting for Efficient Satellite Image Photogrammetry (EOGS)](../../CVPR2025/3d_vision/gaussian_splatting_for_efficient_satellite_image_photogrammetry.md)
- [\[ICCV 2025\] Sat2City: 3D City Generation from A Single Satellite Image with Cascaded Latent Diffusion](../../ICCV2025/3d_vision/sat2city_3d_city_generation_from_a_single_satellite_image_with_cascaded_latent_d.md)
- [\[CVPR 2026\] Yo'City: Personalized and Boundless 3D Realistic City Scene Generation via Self-Critic Expansion](yocity_personalized_and_boundless_3d_realistic_city_scene_generation_via_self-cr.md)
- [\[CVPR 2026\] Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting](off_the_grid_detection_of_primitives_for_feed-forward_3d_gaussian_splatting.md)
- [\[CVPR 2026\] HumanOrbit: 3D Human Reconstruction as 360° Orbit Generation](humanorbit_3d_human_reconstruction_as_360_orbit_generation.md)

</div>

<!-- RELATED:END -->
