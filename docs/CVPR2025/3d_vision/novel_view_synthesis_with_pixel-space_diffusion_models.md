---
title: >-
  [论文解读] Novel View Synthesis with Pixel-Space Diffusion Models
description: >-
  [CVPR 2025][3D视觉][新视角合成] VIVID 用 EDM2 像素空间扩散模型实现端到端新视角合成，通过双 U-Net 编解码器+交叉注意力转移几何信息、简单的位姿嵌入（而非复杂几何编码）和基于同形变换的单视角数据增强，在 RealEstate10K 上 FID 2.89（比 GenWarp 低 51%），PSNR 17.36（+29%）。
tags:
  - CVPR 2025
  - 3D视觉
  - 新视角合成
  - 像素空间扩散
  - 双U-Net
  - 单视角增强
  - 同形变换
---

# Novel View Synthesis with Pixel-Space Diffusion Models

**会议**: CVPR 2025  
**arXiv**: [2411.07765](https://arxiv.org/abs/2411.07765)  
**代码**: 项目页面  
**领域**: 3D视觉  
**关键词**: 新视角合成、像素空间扩散、双U-Net、单视角增强、同形变换

## 一句话总结

VIVID 用 EDM2 像素空间扩散模型实现端到端新视角合成，通过双 U-Net 编解码器+交叉注意力转移几何信息、简单的位姿嵌入（而非复杂几何编码）和基于同形变换的单视角数据增强，在 RealEstate10K 上 FID 2.89（比 GenWarp 低 51%），PSNR 17.36（+29%）。

## 研究背景与动机

1. **领域现状**：新视角合成（NVS）方法分为基于 3D 表示的（NeRF/3DGS，需多视角训练）和基于生成的（扩散模型，可单视角推理）。现有扩散方法多在潜空间工作（如 Zero-1-to-3、GeoGPT）。
2. **现有痛点**：(1) 潜空间扩散的 VAE 编解码引入重建损失——细节丢失尤其在大视角变化时严重；(2) 现有方法需要复杂的几何编码（极线特征、深度 warp）增加工程复杂度；(3) 训练仅在多视角视频数据上，对域外场景泛化差。
3. **核心矛盾**：潜空间高效但有重建瓶颈；像素空间保真但训练困难（分辨率高、收敛慢）。
4. **本文目标**：用像素空间扩散直接做 NVS，避免潜空间重建损失，同时通过简单位姿嵌入和单视角增强降低复杂度。
5. **切入角度**：EDM2 架构的进步使像素空间扩散的训练效率大幅提升——使像素空间 NVS 成为可能。
6. **核心 idea**：双 U-Net + 交叉注意力几何转移 + 位姿嵌入 + 同形旋转增强。

## 方法详解

### 整体框架

源视角图像 → 编码器 U-Net 提取特征 → 联合自注意力+交叉注意力将源视角特征转移到目标视角 → 目标位姿嵌入（扁平化外参+内参）→ 解码器 U-Net 去噪生成目标视角图像。级联设计：低分辨率基模型 + 超分辨率模型。

### 关键设计

1. **简单位姿嵌入**

    - 功能：将相机位姿信息注入扩散过程
    - 核心思路：直接将外参矩阵+焦距+主点扁平化后归一化（$\mu=0, \sigma=1$）作为嵌入。消融显示简单位姿嵌入（FID 3.00）已接近位姿+极线的复杂编码（FID 2.87）
    - 设计动机：复杂几何编码（极线特征、深度 warp）增加工程复杂度但收益有限。交叉注意力本身已能学习隐式几何对应

2. **交叉注意力几何转移**

    - 功能：从源视角特征转移几何和外观信息到目标视角
    - 核心思路：联合自注意力+交叉注意力——query 来自目标视角，key/value 来自源和目标的联合特征
    - 设计动机：比 warp 操作更灵活——warp 在遮挡区域失效，注意力可以从可见区域"借用"信息

3. **单视角同形增强**

    - 功能：用单视角图像模拟多视角数据增强
    - 核心思路：对单视角图像施加随机旋转同形变换 $H_{rot} = K_{dst} R_{dst} R_{src}^{-1} K_{src}^{-1}$，生成源-目标图像对。10% 混合比例最优
    - 设计动机：训练数据（RealEstate10K）主要是室内房屋游览视频，域外（如户外自然场景）泛化差。单视角增强引入域外图像的外观多样性

### 损失函数 / 训练策略

标准 EDM2 扩散损失。CFG 系数 1.5（域内）/ 2.0（域外）。级联两阶段：基模型 + 超分辨率。

## 实验关键数据

### 主实验

| 方法 | 中距离 FID↓ | 中距离 PSNR↑ | 长距离 FID↓ | 长距离 PSNR↑ |
|------|-----------|-------------|-----------|-------------|
| GeoGPT | 6.43 | 14.06 | 7.22 | 13.13 |
| GenWarp | 5.91 | 13.43 | 7.38 | 12.10 |
| PhotoNVS | 7.12 | 13.32 | 9.22 | 12.05 |
| **VIVID** | **2.89** | **17.36** | **3.89** | **15.21** |

### 消融实验

| 几何编码 | FID↓ | PSNR↑ | 说明 |
|---------|------|-------|------|
| 无编码 | 5.75 | 13.39 | 基线 |
| 极线编码 | 4.14 | 17.43 | 几何有帮助 |
| **位姿嵌入** | **3.00** | **21.11** | 简单更好 |
| 位姿+极线 | 2.87 | 21.15 | 边际提升 |

### 关键发现

- FID 比 GenWarp 低 51%（5.91→2.89），PSNR 高 29%——像素空间的保真度优势显著
- 简单位姿嵌入（FID 3.00）vs 位姿+极线（FID 2.87）差距微小——复杂几何编码不必要
- 10% 单视角增强使域外 FID 从 36.14 降至 31.98（-11.4%），但 25% 过度增强反而退化
- 域外泛化仍是主要瓶颈（域内 FID 2.89 vs 域外 >30）

## 亮点与洞察

- **"简单位姿就够了"的发现**：挑战了"NVS 必须用复杂几何编码"的共识——注意力机制可以隐式学习几何对应
- **像素空间 vs 潜空间的实证**：首次在 NVS 上系统比较，证明像素空间在保真度上有本质优势
- **同形变换增强的巧妙**：仅处理旋转（不处理平移），但已足够引入有用的域外多样性

## 局限与展望

- 同形增强仅处理旋转不处理平移——缺少深度变化的增强
- 像素空间扩散计算量大于潜空间方法
- 域外泛化仍有显著掉点（FID >30）
- RealEstate10K 主要是室内场景——户外场景需要更多数据

## 相关工作与启发

- **vs GenWarp**: 使用 warp 操作做几何对应，遮挡区域失败。VIVID 用注意力替代，更鲁棒
- **vs GeoGPT**: 潜空间方法 FID 6.43。像素空间 VIVID FID 2.89——保真度差距源于 VAE 重建损失

## 评分

- 新颖性: ⭐⭐⭐⭐ 像素空间NVS+位姿简化的组合有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 多距离+域外+详细消融+多指标
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐ 为NVS提供了新的架构选择

<!-- RELATED:START -->

## 相关论文

- [MVGD: Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion](zero-shot_novel_view_and_depth_synthesis_with_multi-view_geometric_diffusion.md)
- [DiffPortrait360: Consistent Portrait Diffusion for 360° View Synthesis](diffportrait360_consistent_portrait_diffusion_for_360_view_synthesis.md)
- [CoMapGS: Covisibility Map-based Gaussian Splatting for Sparse Novel View Synthesis](comapgs_covisibility_map-based_gaussian_splatting_for_sparse_novel_view_synthesi.md)
- [SoundVista: Novel-View Ambient Sound Synthesis via Visual-Acoustic Binding](soundvista_novel-view_ambient_sound_synthesis_via_visual-acoustic_binding.md)
- [MOVIS: Enhancing Multi-Object Novel View Synthesis for Indoor Scenes](movis_enhancing_multi-object_novel_view_synthesis_for_indoor_scenes.md)

<!-- RELATED:END -->
