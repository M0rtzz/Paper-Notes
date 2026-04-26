---
title: >-
  [论文解读] AniGS: Animatable Gaussian Avatar from a Single Image with Inconsistent Gaussian Reconstruction
description: >-
  [CVPR 2025][3D视觉][3D人体重建] 从单张图像生成可动画 3D 人体——先用适配的 CogVideo 生成多视角标准姿态图像（含法线），再将多视角不一致性建模为 4DGS 中的时序变化来提取一致的 canonical 空间高斯模型，最后通过 SMPL-X 蒙皮驱动动画。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D人体重建
  - 高斯溅射
  - 单图生成
  - 视频扩散
  - 4DGS
---

# AniGS: Animatable Gaussian Avatar from a Single Image with Inconsistent Gaussian Reconstruction

**会议**: CVPR 2025  
**arXiv**: [2412.02684](https://arxiv.org/abs/2412.02684)  
**代码**: https://lingtengqiu.github.io/2024/AniGS/  
**领域**: 3D视觉  
**关键词**: 3D人体重建、高斯溅射、单图生成、视频扩散、4DGS

## 一句话总结
从单张图像生成可动画 3D 人体——先用适配的 CogVideo 生成多视角标准姿态图像（含法线），再将多视角不一致性建模为 4DGS 中的时序变化来提取一致的 canonical 空间高斯模型，最后通过 SMPL-X 蒙皮驱动动画。

## 研究背景与动机

**领域现状**：从单图创建可动画 3D 人体是虚拟角色/数字人的核心需求。现有方法如 CharacterGen 用多视角生成+标准 3DGS 重建，但多视角图像不可避免地存在视角间不一致（衣物细节、颜色、几何）。

**现有痛点**：(1) 视频生成模型产生的多视角图像存在视角不一致——不同角度的外观细节不完全匹配。(2) 标准 3DGS 假设多视角图像完美一致，不一致输入导致重建模糊和伪影。(3) 生成标准姿态图像需要姿态控制但现有视频生成模型不支持精确姿态条件。

**核心矛盾**：多视角生成的不一致性是不可避免的（生成模型的固有局限），但 3D 重建要求多视角一致性。

**本文目标** 在多视角不一致输入下仍能提取一致的 3D canonical 空间表征。

**切入角度**：将多视角不一致性重新解释为"时间维度的变化"——每个视角看作不同时间帧——然后用 4DGS 的形变场建模这种"时变"外观，canonical 空间自然提取出一致的 3D 模型。

**核心 idea**：把多视角不一致看作时序变化，用 4DGS 的形变场吸收不一致性，canonical 空间得到一致重建。

## 方法详解

### 整体框架
输入单图 → CogVideo（加参考图引导 + SMPL-X 姿态条件 + RGB/法线联合生成）→ 多视角标准姿态图像 + 法线图 → 4DGS 重建（HexPlane 形变场处理不一致性）→ canonical 空间 3DGS 模型 → SMPL-X 蒙皮动画。

### 关键设计

1. **参考图引导的多视角生成**:

    - 功能：从单图生成 360° 多视角标准姿态图像
    - 核心思路：适配 CogVideo 加入参考图 guidance + SMPL-X 渲染的法线/深度作为姿态条件。DiT 的前 3 块和后 3 块分别处理 RGB 和法线模态，中间 24 块共享。在 100K 野外视频上预训练+6124 合成扫描微调
    - 设计动机：联合生成 RGB+法线使 3D 重建有额外的几何监督。野外视频预训练比纯合成数据泛化更好

2. **4DGS 不一致性处理**:

    - 功能：在不一致多视角输入下提取一致 canonical 空间
    - 核心思路：用 HexPlane + MLP 形变场为每个"视角帧"预测 per-Gaussian 的形状/颜色偏移。形变场吸收了视角间的不一致性，剩余的 canonical 空间是多视角的"共识"
    - 设计动机：标准 3DGS 无法处理不一致输入会产生模糊。4DGS 的形变场为不一致性提供了显式的建模容器

3. **形状正则化**:

    - 功能：防止动画时出现尖刺伪影
    - 核心思路：法线 L1 损失约束表面平滑 + 各向异性正则化限制高斯椭球的极端扁平形态
    - 设计动机：标准 3DGS 的高斯容易退化为极薄椭球，在新姿态下产生尖刺

### 损失函数 / 训练策略
RGB L1 + SSIM + 法线 L1 + 各向异性正则化。~5 分钟生成 + ~5 分钟优化，实时渲染。

## 实验关键数据

### 主实验

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| CharacterGen | 17.570 | 0.644 | 0.205 |
| En3D | 19.244 | 0.751 | 0.174 |
| **AniGS** | **21.475** | **0.857** | **0.137** |

### 消融实验

| 配置 | 效果 |
|------|------|
| 无法线正则化 | 表面噪声多 |
| 无各向异性正则化 | 新姿态有尖刺 |
| 随机点初始化 vs 粗网格初始化 | 粗网格显著更好 |
| 标准 3DGS vs 4DGS | 4DGS 更清晰（处理不一致性） |

### 关键发现
- **4DGS 处理不一致性至关重要**：PSNR 从标准 3GS 的 ~19 提升到 4DGS 的 21.5
- **野外视频预训练 > 纯合成**：泛化到真实人像时质量更好
- **10 分钟端到端**（5min 生成 + 5min 优化），实时渲染

## 亮点与洞察
- **"不一致→时变→4DGS"的概念转换**极其优雅——不试图消除不一致，而是拥抱它并用形变场吸收
- **联合 RGB+法线生成**为 3D 重建提供了免费的几何先验

## 局限与展望
- 手部和脚部细节仍有不足（SMPL-X 的手指精度有限）
- CogVideo 生成有时会产生较大不一致（如衣物颜色变化）
- 动画质量受限于 SMPL-X 蒙皮的表达能力

## 相关工作与启发
- **vs CharacterGen**：用标准 3DGS 重建不一致多视角→模糊。AniGS 的 4DGS 方案 PSNR 高 4 个点
- **vs En3D**：需要 SDS 优化耗时 30+min。AniGS 仅需 10 分钟

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用 4DGS 处理多视角不一致性的概念转换非常出色
- 实验充分度: ⭐⭐⭐⭐ 多方法对比、消融详细
- 写作质量: ⭐⭐⭐⭐ 方法动机清楚
- 价值: ⭐⭐⭐⭐ 对单图人体重建有重要贡献

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] GAS: Generative Avatar Synthesis from a Single Image](../../ICCV2025/3d_vision/gas_generative_avatar_synthesis_from_a_single_image.md)
- [\[CVPR 2025\] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](instanthdr_single-forward_gaussian_splatting_for_high_dynamic_range_3d_reconstru.md)
- [\[CVPR 2025\] High-fidelity 3D Object Generation from Single Image with RGBN-Volume Gaussian Reconstruction Model](high-fidelity_3d_object_generation_from_single_image_with_rgbn-volume_gaussian_r.md)
- [\[CVPR 2025\] HRAvatar: High-Quality and Relightable Gaussian Head Avatar](hravatar_high-quality_and_relightable_gaussian_head_avatar.md)
- [\[CVPR 2025\] Wonderland: Navigating 3D Scenes from a Single Image](wonderland_navigating_3d_scenes_from_a_single_image.md)

<!-- RELATED:END -->
