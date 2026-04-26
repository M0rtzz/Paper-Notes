---
title: >-
  [论文解读] RNG: Relightable Neural Gaussians
description: >-
  [CVPR 2025][3D视觉][3D高斯溅射] 提出可重光照神经高斯 (RNG) 框架，通过学习每个高斯点的潜向量并以视角和光照方向为条件，结合阴影线索和混合前向-延迟优化策略，实现软边界物体的高质量重光照。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D高斯溅射
  - 重光照
  - 神经高斯
  - 阴影映射
  - 混合渲染
---

# RNG: Relightable Neural Gaussians

**会议**: CVPR 2025  
**arXiv**: [2409.19702](https://arxiv.org/abs/2409.19702)  
**代码**: https://whois-jiahui.fun/project_pages/RNG  
**领域**: 3D视觉 / 重光照  
**关键词**: 3D高斯溅射, 重光照, 神经高斯, 阴影映射, 混合渲染

## 一句话总结

提出可重光照神经高斯 (RNG) 框架，通过学习每个高斯点的潜向量并以视角和光照方向为条件，结合阴影线索和混合前向-延迟优化策略，实现软边界物体的高质量重光照。

## 研究背景与动机

创建可重光照的3D资产对内容创作至关重要，但光照、几何和材质的分解本质上是病态问题。现有方法大多依赖解析着色模型和表面约束（如有效法线假设），无法处理毛发、织物等模糊边界物体。NRHints 虽支持软边界物体重光照但基于 NeRF，训练/渲染慢且过度平滑。并行工作 GS3 使用 3DGS 提升效率但几何精度不足导致阴影质量差。核心挑战是：如何在避免解析着色假设的同时，利用 3DGS 的高效性实现精确阴影？本文通过神经潜向量代替解析模型，阴影映射+深度精化网络提升阴影质量，前向-延迟混合优化平衡几何与外观。

## 方法详解

### 整体框架

每个高斯点携带一个潜向量表示反射率，由神经解码器 $\Theta$ 以视角 $\omega_o$、光照方向 $\omega_i$ 和阴影线索 $V$ 为条件解码为颜色。训练分两阶段：第一阶段前向着色（先解码再混合）获取几何和潜向量初始化；第二阶段延迟着色（先混合特征再解码）启用阴影线索优化阴影质量。推理在 RTX 4090 上达 60 FPS。

### 关键设计

1. **可重光照神经高斯**：每个高斯点存储潜向量（特征向量）而非球谐系数或解析 BRDF 参数。反射率表示为 $\rho(\mathbf{x}, \omega_o, \omega_i) = \Theta(\mathbf{x} | \omega_o, \omega_i, V)$，其中 $\Theta$ 是 MLP 解码器。这完全避免了对着色模型类型（如 Disney BRDF）或表面约束（如有效法线）的假设，使框架能学习不符合简单解析模型的外观——特别适合毛发、织物等模糊材质。

2. **阴影线索 + 深度精化**：点光源产生尖锐阴影，MLP 易过度平滑。通过在 3DGS 框架下执行阴影映射获取阴影线索：先溅射获取相机深度，经深度精化网络 $\bar{z}' = \bar{z} \cdot \Phi(\omega_o)$ 修正（因加权深度和可能不准确），定位着色点 $P$。再从光源位置设虚拟相机溅射获取阴影深度找到交叉点 $Q$，记录 $|PQ|$ 作为阴影线索。该线索作为解码器的额外输入，显著提升阴影清晰度和一致性。

3. **混合前向-延迟优化**：前向着色 $C_{\text{forward}} = \sum \Theta(\mathbf{x}_i | \omega_o, \omega_i) \alpha_i \prod(1-\alpha_j)$ 先解码再混合，几何好但阴影模糊（混合后模糊高频）。延迟着色 $C_{\text{defer}} = \Theta(\sum \mathbf{x}_i \alpha_i \prod(1-\alpha_j) | \omega_o, \omega_i, V)$ 先混合特征再解码，阴影锐利但可能产生浮块。两阶段策略：第一阶段前向着色优化几何+潜向量，第二阶段延迟着色+阴影线索精化外观。

### 损失函数 / 训练策略

- L1 + SSIM 图像重建损失
- 训练约 1.3 小时（RTX 4090）
- 第一阶段不启用阴影线索（初期高斯形状不佳，错误阴影信息会干扰训练）
- 第一阶段潜向量作为第二阶段初始化加速收敛
- 输入为移动点光源下的多视角图像

## 实验关键数据

### 主实验

| 方法 | 框架 | PSNR↑ | SSIM↑ | LPIPS↓ | 训练时间 | 渲染 FPS |
|------|------|-------|-------|--------|---------|---------|
| NRHints | NeRF | 27.38 | 0.860 | 0.133 | ~24h | ~1 |
| GS3 | 3DGS | 接近 | 接近 | 接近 | ~1.5h | ~60 |
| **RNG** | **3DGS** | **最优/次优** | **最优/次优** | **最优/次优** | **~1.3h** | **~60** |

### 消融实验

| 组件 | PSNR 变化 | 阴影质量 |
|------|----------|---------|
| 无阴影线索 | 下降 | 差（模糊/不一致） |
| 无深度精化 | 略降 | 中（阴影位置偏移） |
| 仅前向着色 | 接近 | 差（阴影模糊） |
| 仅延迟着色 | 下降 | 好但有浮块 |
| **完整RNG** | **最高** | **最佳** |

### 关键发现

- RNG 在大多数场景上取得最优或次优指标，平均 PSNR/SSIM/LPIPS 均最佳
- 训练速度比 NRHints 快约 18 倍，渲染速度快约 60 倍
- 阴影质量显著优于 GS3（得益于阴影线索和深度精化）
- 混合优化策略有效平衡了几何质量和阴影清晰度
- 对软边界物体（毛发、织物）的效果明显优于使用表面约束的方法

## 亮点与洞察

- 用潜向量代替解析着色模型是正确的设计选择——真实材质往往不符合简单模型
- 在 3DGS 框架下实现阴影映射的思路新颖：利用两次溅射（相机+光源视角）模拟光线追踪
- 深度精化网络解决了高斯加权深度不准确的实际问题
- 前向-延迟混合的设计背后有清晰的物理直觉：混合操作的位置决定了频率保持能力

## 局限与展望

- 目前仅支持单点光源重光照，环境光照需要积分开销
- 深度精化假设线性修正，复杂几何可能不足
- 阴影映射的分辨率可能限制极细阴影细节
- 可扩展到多光源、环境图和动态场景

## 相关工作与启发

- **vs NRHints**: 同样支持软边界但基于 NeRF 且过度平滑，RNG 用 3DGS 保留更多细节且快 18 倍
- **vs GS3**: 同样基于 3DGS 但用解析近似，RNG 的阴影线索+深度精化实现更高阴影质量
- **vs 3DGS 逆渲染方法**: 如 GaussianShader 等依赖表面法线和解析 BRDF，不适用软边界物体

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 神经高斯+阴影映射+混合优化的组合有新意
- **实验充分度**: ⭐⭐⭐⭐ — 多场景对比，消融充分
- **写作质量**: ⭐⭐⭐⭐ — 方法阐述清晰，设计选择有充分论证
- **实用价值**: ⭐⭐⭐⭐ — 1.3小时训练+60FPS渲染，实用性强

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] HRAvatar: High-Quality and Relightable Gaussian Head Avatar](hravatar_high-quality_and_relightable_gaussian_head_avatar.md)
- [\[CVPR 2025\] ARM: Appearance Reconstruction Model for Relightable 3D Generation](arm_appearance_reconstruction_model_for_relightable_3d_generation.md)
- [\[NeurIPS 2025\] BecomingLit: Relightable Gaussian Avatars with Hybrid Neural Shading](../../NeurIPS2025/3d_vision/becominglit_relightable_gaussian_avatars_with_hybrid_neural_shading.md)
- [\[ICCV 2025\] Gaussian Splatting with Discretized SDF for Relightable Assets](../../ICCV2025/3d_vision/gaussian_splatting_with_discretized_sdf_for_relightable_assets.md)
- [\[CVPR 2025\] Textured Gaussians for Enhanced 3D Scene Appearance Modeling](textured_gaussians_for_enhanced_3d_scene_appearance_modeling.md)

<!-- RELATED:END -->
