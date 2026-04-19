---
title: >-
  [论文解读] Generating 3D-Consistent Videos from Unposed Internet Photos
description: >-
  [CVPR 2025][3D视觉][视频生成] 本文提出KFC-W，一种从无位姿互联网照片生成3D一致视频的自监督方法，通过在视频扩散模型上联合训练多视角修复和视角插值两个目标，无需任何3D标注（如相机参数），生成的视频在几何和外观一致性上超越商业模型Luma Dream Machine。
tags:
  - CVPR 2025
  - 3D视觉
  - 视频生成
  - 3D一致性
  - 互联网照片
  - 多视角学习
  - 自监督
---

# Generating 3D-Consistent Videos from Unposed Internet Photos

**会议**: CVPR 2025  
**arXiv**: [2411.13549](https://arxiv.org/abs/2411.13549)  
**代码**: 无（项目网页有demo）  
**领域**: 3D视觉  
**关键词**: 视频生成, 3D一致性, 互联网照片, 多视角学习, 自监督

## 一句话总结
本文提出KFC-W，一种从无位姿互联网照片生成3D一致视频的自监督方法，通过在视频扩散模型上联合训练多视角修复和视角插值两个目标，无需任何3D标注（如相机参数），生成的视频在几何和外观一致性上超越商业模型Luma Dream Machine。

## 研究背景与动机
1. **领域现状**：视频基础模型迅速发展（Sora/CogVideoX/Lumiere等），从文本/图像生成视频已相当成熟。3D重建方法（NeRF/3DGS）在受控场景下效果好，但需要密集视角和相机参数。
2. **现有痛点**：现有视频模型（如Luma Dream Machine）在给定稀疏互联网照片时无法理解场景布局，会臆造新建筑/结构来"过渡"关键帧，本质上做的是形变而非真实相机运动。帧插值方法（如FILM）只在小基线下有效，无法处理大视角差异。
3. **核心矛盾**：通用视频模型缺乏3D感知能力，而引入3D监督（相机位姿）又需要SfM预处理，计算昂贵且容易失败（MegaScenes中仅1/5场景SfM成功）。
4. **本文目标** 在不使用任何3D标注的情况下，训练一个能从稀疏互联网照片生成3D一致视频的模型。
5. **切入角度**：视频和多视角图像是互补的——视频提供时间连续性但视角变化小，互联网照片提供大视角多样性但无时序关系。联合利用两者可以获得3D感知能力。
6. **核心 idea**：用多视角修复（从互联网照片学3D先验）和视角插值（从视频学时序连贯）两个自监督目标联合训练视频扩散模型，实现零3D标注的场景级3D理解。

## 方法详解

### 整体框架
基于latent Diffusion Transformer (DiT)微调，输入为2-5张无位姿互联网照片，输出为相邻关键帧间各15帧的插值视频。训练时交替使用两个目标：(1) 在MegaScenes多视角照片上做多视角修复，(2) 在RealEstate10k/DL3DV视频上做视角插值。两个目标统一在同一个去噪框架下，模型根据输入格式自动识别任务。

### 关键设计

1. **多视角修复（Multiview Inpainting）**:
    - 功能：从无序互联网照片中学习3D几何先验，无需相机位姿
    - 核心思路：给定n张同场景条件图和1张目标图，对目标图80%的patches加噪，保持条件图和目标20%不变。模型通过self-attention从条件图中提取结构信息和场景身份，从目标图可见部分获取光照和布局，生成完整目标图。使用MegaScenes（8M互联网照片，430K场景）训练，仅需"同场景"标签，不需相机参数。用冻结的语义分割模型过滤行人/车辆等瞬态物体。
    - 设计动机：受CroCo启发但有三大改进：支持任意数量输入（CroCo仅两张）、最小化数据标注需求、使用概率生成模型（vs MAE的确定性重建）。学到的先验如对称性、深度感知对大基线视频生成至关重要。

2. **视角插值（View Interpolation）**:
    - 功能：学习从起始帧到结束帧之间平滑一致的相机运动
    - 核心思路：从视频中随机采样 $16(n-1)+1$ 帧，每第16帧为条件帧（保持干净），其余15帧为目标帧（加噪）。所有patches添加帧索引嵌入（0, 1, 2, ..., 通过线性层）。训练模型在保持首尾帧不变的情况下生成中间帧的平滑过渡。
    - 设计动机：这个目标单独使用只能处理小基线视频（如Re10k），无法应对互联网照片的大视角差异和光照变化。但每个条件图通过self-attention为所有帧提供信息，使模型学会生成时间上连贯的序列。

3. **CLIP嵌入光照控制**:
    - 功能：在互联网照片间保持一致的光照条件
    - 核心思路：训练时每张图用冻结CLIP编码器提取全局特征，reshape后与图像patch token相加。对条件帧施加极端的ColorJitter颜色增强——迫使模型从CLIP嵌入（而非条件帧本身）获取光照信息。推理时用第一张输入图的CLIP嵌入统一所有插值帧的光照。
    - 设计动机：互联网照片间光照差异大（晴天vs阴天），不加控制的模型会生成光照不一致的帧。CLIP嵌入捕捉粗粒度光照信息（"cloudy"/"sunny"），虽不够精细但对保持一致性已足够。

### 损失函数 / 训练策略
- 标准扩散去噪目标：仅在噪声patches上计算loss
- 两个目标统一：模型根据输入的帧数和格式自动判断任务，不需要任务标志
- 训练数据：MegaScenes（多视角修复）+ RealEstate10k + DL3DV（视角插值）
- 训练配置：32台A100 80G GPU训练3天
- 推理：50步DDIM采样，仅解码中间帧

## 实验关键数据

### 主实验（用户研究, Ours(Full)优胜率）

| 对比方法 | 一致性 | 相机运动 | 美观度 | 数据集 |
|---------|--------|---------|--------|--------|
| FILM | 100.0% | 100.0% | 100.0% | Phototourism |
| FLAVR | 100.0% | 100.0% | 100.0% | Phototourism |
| Ours(Video-Only) | 100.0% | 96.7% | 96.7% | Phototourism |
| **Luma Dream Machine** | **60.2%** | **73.6%** | **60.2%** | **Phototourism** |
| Luma Dream Machine | 83.7% | 84.9% | 68.3% | Re10k |

### 消融实验（COLMAP重建成功率）

| 配置 | SfM成功率 | 注册图像率 | 说明 |
|------|----------|----------|------|
| 仅原始稀疏照片 | 45% (115/252) | 43% (378/882) | 基准 |
| +生成帧(Full) | **93% (235/252)** | **84% (741/882)** | 几何一致性强 |
| +生成帧(Video-Only) | 71% (179/252) | 67% (589/882) | 多视角修复关键 |

### 关键发现
- 帧插值方法（FILM/FLAVR/LDMVFI）在大基线互联网照片上完全失败（0%优胜率），因其仅擅长小运动
- 仅用视角插值训练的模型在Re10k上表现好但在Phototourism上失败，证明多视角修复目标是处理wild场景的关键
- 用更宽基线视频训练（Long-Video消融）也无法替代多视角修复，因为互联网照片包含视频中罕见的极端旋转/缩放
- 生成帧使COLMAP成功率从45%跃升到93%，说明帧具有真实的几何一致性
- 3DGS在生成帧上训练的PSNR/SSIM/LPIPS全面优于在原始稀疏照片上训练

## 亮点与洞察
- **自监督多任务学习的涌现能力**令人惊艳：模型从未见过"互联网照片作输入、一致视频作输出"的配对，但通过多视角修复+视角插值的联合训练，自然涌现了这一能力
- **互联网照片与视频的互补性**是核心洞察：照片提供视角多样性（极端旋转/缩放），视频提供时序连贯性，两者结合超越了各自单独能达到的效果
- **ColorJitter强增强迫使模型使用CLIP条件**是巧妙的工程trick：通过破坏条件帧的颜色信息，模型被迫从CLIP嵌入中获取光照，实现了推理时的光照可控
- 该范式可迁移到其他需要3D理解但缺乏3D标注的任务，如自动驾驶场景生成、室内导航

## 局限与展望
- 不处理动态物体，仅适用于静态场景
- CLIP嵌入只提供粗粒度光照控制，无法精确控制太阳角度等物理属性
- 生成帧数固定为每对关键帧15帧，扩展到更长或更密需要更多计算
- 基于内部text-to-video模型（非开源），但方法可适配任何DiT架构
- 输入关键帧数限制在2-5张，更多输入的扩展性未验证
- 与Luma的对比不完全公平（Luma分辨率更高、模型更大）

## 相关工作与启发
- **vs Luma Dream Machine**: 商业级通用视频模型，但在此任务上做"创意形变"而非真实相机运动；本文通过3D感知训练目标实现了场景布局理解
- **vs CroCo**: CroCo在双视图上做masked modeling学3D先验但仅支持两张图且为确定性；本文扩展到多图+概率生成
- **vs Zero-1-to-3/ZeroNVS**: 需要相机位姿进行条件化；本文完全不需要位姿
- **vs VFusion3D**: 利用视频模型的一致性生成3D资产；本文则反过来，利用3D感知训练使视频模型更一致
- 该工作表明：纯2D数据（视频+多视角照片）可以替代昂贵的3D标注进行场景级3D学习

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 任务定义新颖，自监督方案巧妙，联合多视角修复和视角插值的设计有独创性
- 实验充分度: ⭐⭐⭐⭐ 用户研究+COLMAP+3DGS多角度验证，但缺乏定量的逐帧评估指标
- 写作质量: ⭐⭐⭐⭐⭐ 叙述流畅，图示清晰，动机阐述充分
- 价值: ⭐⭐⭐⭐⭐ 提出了可扩展的3D学习范式，对场景理解和视频生成社区都有启发

<!-- RELATED:START -->

## 相关论文

- [DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos](depthcrafter_generating_consistent_long_depth_sequences_for_open-world_videos.md)
- [LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos](../../ICCV2025/3d_vision/longsplat_robust_unposed_3d_gaussian_splatting_for_casual_long_videos.md)
- [ERUPT: Efficient Rendering with Unposed Patch Transformer](erupt_efficient_rendering_with_unposed_patch_transformer.md)
- [Auto-Regressively Generating Multi-View Consistent Images](../../ICCV2025/3d_vision/auto-regressively_generating_multi-view_consistent_images.md)
- [HOI3DGen: Generating High-Quality Human-Object-Interactions in 3D](hoi3dgen_generating_high-quality_human-object-interactions_in_3d.md)

<!-- RELATED:END -->
