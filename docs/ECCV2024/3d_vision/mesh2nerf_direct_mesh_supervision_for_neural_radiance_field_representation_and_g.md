---
title: >-
  [论文解读] Mesh2NeRF: Direct Mesh Supervision for Neural Radiance Field Representation and Generation
description: >-
  [ECCV 2024][3D视觉][NeRF] 提出Mesh2NeRF，通过解析解直接从纹理网格(textured mesh)构造GT辐射场，用occupancy函数建模密度场、用反射模型建模颜色场，为NeRF表示与生成任务提供精确的3D逐点监督。
tags:
  - ECCV 2024
  - 3D视觉
  - NeRF
  - 网格监督
  - 辐射场生成
  - 3D生成
  - 扩散模型
---

# Mesh2NeRF: Direct Mesh Supervision for Neural Radiance Field Representation and Generation

**会议**: ECCV 2024  
**arXiv**: [2403.19319](https://arxiv.org/abs/2403.19319)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: NeRF, 网格监督, 辐射场生成, 3D生成, 扩散模型

## 一句话总结

提出Mesh2NeRF，通过解析解直接从纹理网格(textured mesh)构造GT辐射场，用occupancy函数建模密度场、用反射模型建模颜色场，为NeRF表示与生成任务提供精确的3D逐点监督。

## 研究背景与动机

**领域现状**: NeRF作为3D生成的表示已被广泛使用，但训练生成式NeRF需要GT辐射场样本；目前的GT获取方式是从mesh渲染多视角图片再用NeRF拟合

**现有痛点**: 
   - 多视角渲染→NeRF拟合的流程冗余且引入信息损失（遮挡、欠拟合）
   - NeRF的像素级渲染损失是弱监督：每个像素颜色需同时监督光线上所有采样点的密度和颜色
   - 视角覆盖不均衡时，这种弱监督容易导致辐射场不准确

**核心矛盾**: 明明拥有精确的3D mesh数据，却要绕道2D渲染再重建回3D，既浪费了mesh的精确几何信息，又引入了不必要的误差

**本文目标**: 跳过多视角渲染环节，直接从textured mesh解析地导出辐射场的密度和颜色值，作为NeRF的逐点3D监督

**切入角度**: 从体渲染的数学推导出发，用top-hat函数逼近Dirac delta的密度分布，将mesh表面编码为occupancy-based的alpha值

**核心 idea**: 用解析解把mesh变成radiance field，实现逐采样点的直接3D监督

## 方法详解

### 整体框架

Mesh2NeRF分两部分：(1) 从textured mesh解析推导辐射场（密度场+颜色场）；(2) 将推导出的辐射场作为直接监督信号用于NeRF单场景拟合或NeRF生成模型训练。

### 关键设计

1. **密度场建模 — Occupancy-based Alpha**:

    - 理想状态下mesh表面的密度是Dirac delta函数（仅表面处密度无穷大），但这对神经网络不可行
    - 用top-hat函数 $\Delta_n(t)$ 逼近Dirac delta：

    $\Delta_n(t) = \begin{cases} n/2, & \text{if } |t| < 1/n \\ 0, & \text{otherwise} \end{cases}$

    - 核心洞察：不直接用密度 $\sigma$（值会趋向无穷），而用alpha值 $\alpha_i = 1 - \exp(-\sigma_i \delta_i)$
    - 对于大 $n$，alpha值可简化为基于距离的occupancy函数：

    $\alpha = \begin{cases} 1, & \text{if } d < h \\ 0, & \text{otherwise} \end{cases}$

      其中 $d$ 是到mesh表面的距离，$h$ 是表面半厚度
    - 物理含义：射线上第一个与表面相交的采样点 $i_m$ 的 $\alpha_{i_m}=1$，对应颜色即为纹理mesh交点颜色，其余点 $\alpha=0$

2. **颜色场建模 — BRDF反射模型**:

    - 射线上所有采样点的颜色定义为射线与mesh表面第一个交点的颜色 $\mathbf{c}_i$
    - 使用Phong反射模型计算视角相关颜色（也可替换为任意BRDF）
    - 考虑了mesh几何（法线）、纹理和环境光照的综合影响
    - 体渲染结果 $\hat{C}(\mathbf{y}) = \alpha_{i_m} \mathbf{c}_{i_m}$，与GT mesh渲染高度一致

3. **Mesh2NeRF作为NeRF监督**:

    - 提供逐采样点的3D监督，而非传统的逐像素2D监督
    - Alpha损失：$\mathcal{L}_{alpha} = \sum_{i=1}^{N} |\hat{\alpha}_i - \alpha_i|^2$
    - 颜色损失：$\mathcal{L}_{color} = \sum_{i=1}^{N} \|\hat{\mathbf{c}}_i - \mathbf{c}_i\|_2^2$
    - 积分损失（可选）：$\mathcal{L}_{integral}$ 约束ray颜色积分
    - 总损失：$\mathcal{L} = \mathcal{L}_{alpha} + w_{color}\mathcal{L}_{color} + w_{integral}\mathcal{L}_{integral}$

4. **高效采样策略**:

    - 使用Embree库的BVH加速结构进行射线-mesh求交
    - 对与mesh相交的射线：在空场景空间和表面附近窄带（距离 $h$ 内）做分层采样
    - 对不与mesh相交的射线：沿射线随机采样

### 在NeRF生成任务中的应用

- 基于SSDNeRF框架（triplane NeRF auto-decoder + triplane latent diffusion model）
- 将SSDNeRF的渲染损失 $\mathcal{L}_{rend}$ 替换为Mesh2NeRF损失 $\mathcal{L}$
- 训练阶段用Mesh2NeRF的3D直接监督，条件生成推理阶段仍用渲染损失（因为推理时没有mesh）
- 总训练目标：$\mathcal{L}_{ssdnerf} = w_{rend}\mathcal{L} + w_{diff}\mathcal{L}_{diff}$

### 损失函数 / 训练策略

- 单场景拟合：支持NeRF/TensoRF/Instant NGP三种编码方式，分别对应Mesh2NeRF NeRF/TensoRF/NGP
- 生成任务：使用SSDNeRF官方实现，相同视点数和训练设置，仅替换损失函数，确保公平比较
- 无需额外射线采样开销

## 实验关键数据

### 单场景拟合 — ABO & Poly Haven

| 方法 | ABO PSNR↑ | ABO SSIM↑ | ABO LPIPS↓ | Poly Haven PSNR↑ |
|------|-----------|-----------|------------|-------------------|
| NeRF | 25.09 | 0.882 | 0.137 | 21.24 |
| TensoRF | 31.33 | 0.944 | 0.032 | 23.32 |
| Instant NGP | 30.16 | 0.928 | 0.039 | 24.39 |
| Mesh2NeRF NeRF | 32.40 | 0.942 | 0.044 | 22.75 |
| Mesh2NeRF TensoRF | 32.00 | 0.957 | 0.024 | 23.97 |
| **Mesh2NeRF NGP** | **33.28** | **0.969** | **0.018** | **25.30** |

### 条件生成 — ShapeNet Cars

| 方法 | 1-view PSNR↑ | 2-view PSNR↑ | 3-view PSNR↑ | 4-view PSNR↑ |
|------|-------------|-------------|-------------|-------------|
| SSDNeRF | 21.09 | 24.67 | 25.71 | 26.54 |
| **Ours** | **21.78** | **24.98** | **25.89** | **26.51** |

ShapeNet Chairs 在2-view条件下提升最显著：PSNR从19.65→22.22 (+2.57dB)

### 关键发现

- 单场景拟合：Mesh2NeRF NGP比Instant NGP提升 $+3.12$ dB PSNR（ABO数据集），证明逐点3D监督远优于像素级2D监督
- 条件生成：尤其在非常规形状和真实数据(KITTI)上优势更明显——SSDNeRF在车身颜色与背景相近时失败，Mesh2NeRF仍能重建
- 无条件生成(Objaverse Mugs)：从NeRF提取mesh时，Mesh2NeRF的几何质量远好于SSDNeRF（如杯子是否封口）
- 生成模型的3D监督比2D监督更能学习到正确的几何先验

## 亮点与洞察

- **从第一性原理出发**: 不是启发式设计，而是从体渲染公式数学推导出mesh到辐射场的转换，具有理论保证
- **消除冗余流程**: 省去"mesh→渲染→拟合NeRF"的间接路径，直接"mesh→radiance field"，减少信息损失
- **即插即用**: 可以替换任何使用渲染损失的NeRF方法中的损失函数，兼容NeRF/TensoRF/Instant NGP等多种编码
- **3D逐点监督 vs 2D像素监督**: 实验充分证明逐点监督的优越性，尤其在遮挡和视角稀疏时

## 局限与展望

1. 颜色建模使用Phong模型，将光照bake进外观，不支持光照解耦
2. 采样策略依赖渲染图像的射线方案，没有充分利用已知mesh几何做更高效的采样
3. 仅在ShapeNet/Objaverse等合成数据集上验证生成任务，缺乏真实场景的大规模验证
4. 未探索与更多先进3D生成框架（如3DGS-based方法）的结合
5. 表面厚度参数 $h$ 的选择对结果有影响，需要tuning

## 相关工作与启发

- **NeRF2Mesh** (反方向): 从NeRF重建mesh ↔ Mesh2NeRF从mesh构建NeRF，互为逆过程
- **SSDNeRF**: 作为生成框架baseline，Mesh2NeRF的即插即用特性在此得到验证
- **depth-supervised NeRF**: 深度先验是弱形式的几何监督 → Mesh2NeRF提供了强形式（完整occupancy + color）
- **启发**: 当我们拥有精确的3D数据时，应该充分利用其精度优势做直接监督，而非降维到2D再间接学习

## 评分

- **新颖性**: ⭐⭐⭐⭐ (从数学推导建立mesh到radiance field的解析映射，思路优雅)
- **实验充分度**: ⭐⭐⭐⭐ (单场景/条件生成/无条件生成/真实数据都有验证)
- **写作质量**: ⭐⭐⭐⭐ (公式推导清晰，因果链明确)
- **价值**: ⭐⭐⭐⭐ (为mesh-based 3D数据集的NeRF应用提供了新范式)

<!-- RELATED:START -->

## 相关论文

- [Dynamic Neural Radiance Field from Defocused Monocular Video](dynamic_neural_radiance_field_from_defocused_monocular_video.md)
- [MALD-NeRF: Taming Latent Diffusion Model for Neural Radiance Field Inpainting](taming_latent_diffusion_model_for_neural_radiance_field_inpainting.md)
- [Lagrangian Hashing for Compressed Neural Field Representations](lagrangian_hashing_for_compressed_neural_field_representations.md)
- [GVGEN: Text-to-3D Generation with Volumetric Representation](gvgen_text-to-3d_generation_with_volumetric_representation.md)
- [S³D-NeRF: Single-Shot Speech-Driven Neural Radiance Field for High Fidelity Talking Head Synthesis](s3d-nerf_single-shot_speech-driven_neural_radiance_field_for_high_fidelity_talki.md)

<!-- RELATED:END -->
