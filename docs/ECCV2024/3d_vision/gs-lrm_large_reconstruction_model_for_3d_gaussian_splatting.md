---
title: >-
  [论文解读] GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting
description: >-
  [ECCV 2024][3D视觉][大规模重建模型] 本文提出GS-LRM，一个极其简洁的基于Transformer的大规模重建模型，将多视角图像patch化后通过自注意力直接回归逐像素3D高斯参数，在物体级（超Triplane-LRM 4dB PSNR）和场景级（超pixelSplat 2.2dB PSNR）重建中均大幅超越SOTA，单张A100上0.23秒完成推理。
tags:
  - ECCV 2024
  - 3D视觉
  - 大规模重建模型
  - 3D高斯溅射
  - Transformer
  - 稀疏视角重建
  - 前馈3D重建
---

# GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting

**会议**: ECCV 2024  
**arXiv**: [2404.19702](https://arxiv.org/abs/2404.19702)  
**代码**: https://sai-bi.github.io/project/gs-lrm/  
**领域**: 3D视觉  
**关键词**: 大规模重建模型, 3D高斯溅射, Transformer, 稀疏视角重建, 前馈3D重建

## 一句话总结

本文提出GS-LRM，一个极其简洁的基于Transformer的大规模重建模型，将多视角图像patch化后通过自注意力直接回归逐像素3D高斯参数，在物体级（超Triplane-LRM 4dB PSNR）和场景级（超pixelSplat 2.2dB PSNR）重建中均大幅超越SOTA，单张A100上0.23秒完成推理。

## 研究背景与动机

**领域现状**：3D重建是计算机视觉的核心问题。传统方法依赖密集多视角输入和复杂的SfM/MVS管线。近年来，NeRF和3D高斯溅射（3DGS）在每场景优化范式下取得了出色的渲染质量，但速度慢且需要大量输入视角。基于Transformer的大规模重建模型（LRM）通过在大规模3D数据上训练，实现了前馈式稀疏视角3D重建。

**现有痛点**：(1) 现有LRM使用Triplane NeRF表示，受限于固定的低分辨率三平面（通常32），难以保留高频细节；(2) 体渲染计算昂贵，限制了训练分辨率；(3) Triplane NeRF天然只适合物体级重建，难以扩展到大规模场景；(4) 模型设计复杂，如需要额外的triplane token和精心设计的解码器。

**核心矛盾**：要构建一个通用、可扩展、高效的3D重建模型，需要同时满足：高质量细节保留、快速渲染、从物体到场景的尺度适应性。Triplane NeRF在这三个维度上都存在瓶颈。

**本文目标** (1) 如何设计一个更简洁高效的LRM架构？(2) 如何让重建模型同时处理物体和场景？(3) 如何保留高频细节并实现快速渲染？

**切入角度**：用3D高斯溅射替代Triplane NeRF作为输出表示，设计一个纯Transformer模型直接预测逐像素的3D高斯参数。像素对齐的高斯预测将2D输入和3D输出对齐在同一空间中，简化架构的同时天然支持高分辨率输入和大规模场景。

**核心 idea**：用纯Transformer将多视角图像token化后直接回归逐像素3D高斯原语，实现极简、可扩展、物体/场景通用的前馈式高质量3D重建。

## 方法详解

### 整体框架

GS-LRM的架构极其简洁：输入 $N$ 张多视角图像及相机参数 → 将每张图与Plücker射线坐标拼接（9通道）→ patchify为不重叠的patch token → 多视角token拼接后送入 $L$ 层Transformer块（自注意力+MLP）→ 每个输出token通过线性层解码为 $p^2$ 个3D高斯参数 → unpatchify后得到逐像素高斯 → 合并所有视角的高斯作为最终输出。

### 关键设计

1. **逐像素3D高斯预测（Per-pixel Gaussian Prediction）**:
    - 功能：将2D图像像素与3D高斯一一对应，简化网络并保留高频细节
    - 核心思路：每个2D像素对应一个3D高斯，参数包括3通道RGB、3通道尺度、4通道旋转四元数、1通道不透明度和1通道射线距离（共12维）。高斯中心通过 $xyz = ray_o + t \cdot ray_d$ 从射线距离和相机参数计算。总输出 $N \times H \times W$ 个高斯
    - 设计动机：像素对齐创建了输入RGB到输出颜色的快捷通道，便于网络学习精确的逐高斯颜色；随输入分辨率自动扩展高斯数量，适应高分辨率和大场景

2. **Plücker射线坐标位置编码**:
    - 功能：为每个patch提供唯一的空间和视角信息，替代传统位置编码
    - 核心思路：将每个像素的Plücker射线坐标（6维）与RGB（3维）拼接为9通道输入，patchify后通过线性层映射为 $d$ 维token。由于Plücker坐标天然包含了像素位置和视角信息，不需要额外的位置编码或视角编码
    - 设计动机：消除了需要设计位置编码和视角编码的复杂性，同时让自注意力能够基于几何射线信息进行多视角对应推理

3. **纯自注意力多视角融合**:
    - 功能：通过全局自注意力在所有视角的所有patch间建立对应关系
    - 核心思路：将所有视角的所有patch token拼接为一个长序列，送入标准Transformer块。每个自注意力层都在所有视角间和视角内的patch上全局计算，自然学习到多视角对应关系和重建先验
    - 设计动机：相比pixelSplat等方法需要专门设计极线特征聚合，全局自注意力更简洁且能利用所有像素的信息（而非仅极线上的子集），在大规模数据训练下效果更优

### 损失函数 / 训练策略

损失为MSE和感知损失的加权和：$\mathcal{L} = \frac{1}{M}\sum_{i'} (MSE(\hat{I}_{i'}^*, I_{i'}^*) + \lambda \cdot Perceptual(\hat{I}_{i'}^*, I_{i'}^*))$，$\lambda=0.5$。使用基于VGG-19的感知损失（比LPIPS训练更稳定）。模型24层Transformer，隐藏维度1024，16头注意力，MLP维度4096，patch大小8×8，共约300M参数。256分辨率预训练2天，512分辨率微调1天，64×A100 (40G)。

## 实验关键数据

### 主实验

**物体级重建（GSO数据集）：**

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | 输入分辨率 |
|------|-------|-------|--------|-----------|
| Triplane-LRM | 26.54 | 0.893 | 0.064 | 512 |
| GS-LRM (Res-512) | **30.52** | **0.952** | **0.050** | 512 |
| LGM | 21.44 | 0.832 | 0.122 | 256 |
| GS-LRM (Res-256) | **29.59** | **0.944** | **0.051** | 256 |

**场景级重建（RealEstate10K）：**

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| pixelNeRF | 20.43 | 0.589 | 0.550 |
| GPNR | 24.11 | 0.793 | 0.255 |
| pixelSplat | 25.89 | 0.858 | 0.142 |
| GS-LRM | **28.10** | **0.892** | **0.114** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| vs Triplane-LRM (GSO) | +3.98dB PSNR | 像素对齐高斯远优于三平面 |
| vs LGM (GSO, 256) | +8.15dB PSNR | 纯Transformer远优于U-Net |
| vs pixelSplat (RE10K) | +2.21dB PSNR | 全局自注意力优于极线采样 |
| 4视角输入物体 | 30.52 PSNR | 标准配置 |
| 2视角输入场景 | 28.10 PSNR | 场景级同样出色 |

### 关键发现

- 像素对齐的高斯预测是质量提升的核心——为输入RGB到输出颜色建立了直接映射
- Triplane-LRM难以重建文本等高频细节和细薄结构，GS-LRM则能忠实再现
- LGM的U-Net架构在相同计算量下显著逊于Transformer（8dB差异）
- 纯自注意力无需极线等3D归纳偏置，在大数据训练下能自动学习多视角对应
- 推理速度约0.23秒（单A100），支持交互式应用

## 亮点与洞察

- **架构极简至优雅**：仅有patchify→Transformer→unpatchify三步，无需triplane token、额外解码器或3D归纳偏置
- **像素对齐是关键洞察**：将输入输出对齐在同一像素空间，创建了颜色学习的捷径
- **物体-场景通用**：同架构仅换训练数据即可切换，展示了极强的通用性
- **可扩展性强**：随分辨率自然扩展高斯数量和token长度，支持512+分辨率

## 局限与展望

- 当前最高分辨率约512×904，可探索扩展到1K-2K分辨率
- 需要已知相机参数，实际应用中可能不可用（如手机拍摄4张照片）
- 像素对齐表示只能建模视锥内的表面，不可见区域无法重建
- 未使用高阶球谐函数，视角依赖效果受限
- 可探索与DUSt3R结合实现免位姿输入

## 相关工作与启发

- **LRM/Instant3D**：首个基于Transformer的大规模重建模型，采用Triplane NeRF
- **pixelSplat**：同期工作，也预测像素对齐高斯，但使用极线特征聚合而非全局自注意力
- **LGM**：同期工作，U-Net架构预测多视角高斯，聚焦物体生成
- 启发：简洁的纯Transformer+大数据路线在3D领域同样有效，呼应了NLP和2D视觉的scaling law

## 评分

- 新颖性: ⭐⭐⭐⭐ 架构极简但效果卓越，像素对齐高斯的LRM是新颖组合
- 实验充分度: ⭐⭐⭐⭐⭐ 物体/场景双场景+多种对比+下游生成应用+高分辨率展示
- 写作质量: ⭐⭐⭐⭐⭐ 极其清晰简洁，方法描述精准
- 价值: ⭐⭐⭐⭐⭐ 奠定了基于GS的大规模前馈重建的基线，影响广泛

<!-- RELATED:START -->

## 相关论文

- [GRM: Large Gaussian Reconstruction Model for Efficient 3D Reconstruction and Generation](grm_large_gaussian_reconstruction_model_for_efficient_3d_reconstruction_and_gene.md)
- [CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization](cor-gs_sparse-view_3d_gaussian_splatting_via_co-regularization.md)
- [Texture-GS: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing](texture-gs_disentangling_the_geometry_and_texture_for_3d_gaussian_splatting_edit.md)
- [Pixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian Splatting](pixel-gs_density_control_with_pixel-aware_gradient_for_3d_gaussian_splatting.md)
- [Surface Reconstruction from 3D Gaussian Splatting via Local Structural Hints](surface_reconstruction_from_3d_gaussian_splatting_via_local_structural_hints.md)

<!-- RELATED:END -->
