---
title: >-
  [论文解读] NeAR: Coupled Neural Asset–Renderer Stack
description: >-
  [CVPR 2026][3D视觉][神经渲染] NeAR 提出将神经资产创作和神经渲染联合设计为一个耦合栈，通过光照均匀化的结构化 3D 潜变量（LH-SLAT）消除输入图像中的烘焙光照，再用光照感知的神经解码器实时合成可重光照的 3D 高斯场，在前向渲染、重建、重光照和新视角重光照四类任务上超越现有方法。
tags:
  - CVPR 2026
  - 3D视觉
  - 神经渲染
  - 光照均匀化
  - 3D高斯溅射
  - 可重光照
  - 资产-渲染器联合设计
---

# NeAR: Coupled Neural Asset–Renderer Stack

**会议**: CVPR 2026  
**arXiv**: [2511.18600](https://arxiv.org/abs/2511.18600)  
**代码**: https://near-project.github.io/ (项目页)  
**领域**: 3D视觉 / 扩散模型  
**关键词**: 神经渲染, 光照均匀化, 3D高斯溅射, 可重光照, 资产-渲染器联合设计

## 一句话总结
NeAR 提出将神经资产创作和神经渲染联合设计为一个耦合栈，通过光照均匀化的结构化 3D 潜变量（LH-SLAT）消除输入图像中的烘焙光照，再用光照感知的神经解码器实时合成可重光照的 3D 高斯场，在前向渲染、重建、重光照和新视角重光照四类任务上超越现有方法。

## 研究背景与动机

1. **领域现状**：当前神经图形学有两条独立路线——神经资产创作（用生成模型合成3D资产）和神经渲染（将资产映射到图像）。它们通常独立开发：资产生成假设固定渲染器，渲染器针对静态资产分布训练。

2. **现有痛点**：(a) 基于 PBR 分解的方法（如 Hunyuan3D）容易产生材质误判（如将木头判为金属），分解误差在非线性渲染流程中被放大，导致烘焙阴影和光照不一致；(b) 基于扩散的 2D 重光照方法（如 DiLightNet、IC-Light）缺乏3D一致性，计算开销大；(c) 现有 SLAT 表示盲目编码外观，包含阴影和高光，无法直接用于重光照。

3. **核心矛盾**：资产中的阴影、高光和互反射与几何和材质固有纠缠，脆弱的显式 PBR 逆向分解在实际场景中不可靠；而完全黑盒的神经生成又缺乏可控性。

4. **本文目标** 如何在避免不稳定 PBR 逆向的同时实现高质量、可控的单图像可重光照 3D 生成？

5. **切入角度**：作者提出"先均匀化后合成"（homogenize-then-synthesize）的策略——将资产表示和渲染过程共同设计，让它们通过共享的光照均匀化潜空间建立稳健的"契约"。

6. **核心 idea**：联合设计光照均匀化的 3D 潜变量表示和光照感知的神经渲染器，形成耦合的资产-渲染器栈，替代传统解耦的资产生成+独立渲染范式。

## 方法详解

### 整体框架
NeAR 分两个阶段：**Stage 1** 用 LoRA 微调 rectified-flow 模型，将任意光照下的单张输入图像提升到光照均匀化的 SLAT（LH-SLAT），消除烘焙阴影和不稳定高光；**Stage 2** 用前馈解码器将 LH-SLAT 在目标光照和视角条件下合成可重光照的 3D 高斯溅射场。整个流程无需逐物体优化，支持实时推理。

### 关键设计

1. **光照均匀化结构化 3D 潜变量（LH-SLAT）**:
    - 功能：将任意光照的输入映射到标准光照空间，生成光照无关的资产表示
    - 核心思路：定义均匀光照 $E_h$ 为白色均匀环境光。先用预训练 SLAT 流模型 $f_s$ 从输入图像生成带阴影的 SLAT $Z_s$，再用 LoRA 适配的模型 $f_\theta$ 在稀疏体素空间中将 $Z_s$ 引导到光照均匀化的 $Z_{\text{lh}} = f_\theta(Z_s, I_{\text{in}})$。训练数据通过多视角渲染3D资产在均匀光照下获得真值，再在随机光照下渲染获得输入对。可选地，对高反射材质额外提取 Base Color SLAT $Z_{\text{bc}}$ 拼接。
    - 设计动机：直接做 PBR 分解是病态问题；通过流模型在潜空间中学习光照去除，避免了显式材质分解的不稳定性。LH-SLAT 保留了几何-材质-光照交互的本质信息，为下游渲染提供稳定基础。

2. **光照 Tokenizer**:
    - 功能：将 HDR 环境光图编码为紧凑的光照条件 token
    - 核心思路：将环境光图分解为 LDR 色调映射图 $\mathbf{E}_{\text{ldr}}$、归一化对数强度图 $\mathbf{E}_{\text{log}}$ 和相机空间方向编码 $\mathbf{E}_{\text{dir}}$。用 ConvNeXt 提取视觉特征，然后用 NeRF 位置编码处理 $\mathbf{E}_{\text{dir}}$，通过空间交叉注意力将方向信息与视觉特征融合，最终经自注意力得到光照条件 token $C_L \in \mathbb{R}^{4096 \times 768}$。
    - 设计动机：相比用 VAE 压缩整个光图，显式嵌入方向信息使得切换视角时光照方向可编辑。

3. **内在感知解码器（IAD）+ 光照感知解码器（LAD）**:
    - 功能：IAD 解码视角无关、光照不变的内在特征；LAD 注入光照和视角条件生成最终的光照相关特征
    - 核心思路：**IAD** 用 Transformer + shifted window attention 处理 LH-SLAT，加入 16 个 register token 通过全局交叉注意力捕获全局上下文，输出内在特征 $\boldsymbol{h}$。**LAD** 先计算观察视角嵌入（射线距离 + 射线方向的 NeRF 位置编码），加到内在特征上得到 $\boldsymbol{h}^v$，然后用堆叠的交叉注意力块注入光照条件 $C_L$，输出光照感知特征 $\boldsymbol{h}^e$。
    - 设计动机：将解码分为光照无关和光照相关两部分，既保证了内在属性的稳定性，又允许灵活的光照控制。放弃球谐函数改用显式视角注入，更好地建模视角相关的镜面高光。

4. **神经 3D 高斯溅射**:
    - 功能：从内在特征和光照感知特征回归 3DGS 参数
    - 核心思路：内在特征 $\boldsymbol{h}$ 解码为位置偏移、基础颜色、粗糙度、金属度、缩放、旋转和不透明度等光照无关参数；光照特征 $\boldsymbol{h}^e$ 解码为 48 维颜色特征、光照缩放和阴影。用浅层 MLP 结合法线位置编码和颜色特征预测辐射值，通过可微光栅化得到 HDR 预测图像。
    - 设计动机：将高斯参数分为内在和光照相关两组，实现了解耦的材质-光照表示。

### 损失函数 / 训练策略

**Stage 1**：条件流匹配损失 $\mathcal{L}_{\text{stage1}} = \mathbb{E}\|\boldsymbol{v}_\theta(\boldsymbol{z}, Z_s, I_{\text{in}}, t) - (\boldsymbol{\epsilon} - \boldsymbol{z}_0)\|_2^2$。

**Stage 2**：HDR 重建损失（对数空间 L1 + LPIPS + D-SSIM）$\mathcal{L}_{\text{hdr}}$ + PBR 材质辅助监督 $\mathcal{L}_{\text{pbr}}$ + 阴影监督 $\mathcal{L}_{\text{shadow}}$。

训练数据：87K 个带 PBR 纹理的 3D 资产 + 2K 个 4K 分辨率 HDR 环境光图，用 Blender EEVEE Next 渲染。

## 实验关键数据

### 主实验（四项任务 PSNR↑ 对比）

| 任务 | 方法 | ADT | DTC | Objaverse | Glossy Syn. |
|------|------|-----|-----|-----------|-------------|
| G-buffer前向渲染 | DiffusionRenderer | 24.41 | 27.16 | 27.09 | 25.46 |
| | **NeAR** | **29.15** | **31.59** | **32.23** | **30.47** |
| 随机光照重建 | DiLightNet | 21.11 | 23.53 | 25.65 | 24.09 |
| | **NeAR** | **22.89** | **24.68** | **26.53** | **25.32** |
| 未知光照重光照 | DiffusionRenderer | 21.91 | 22.99 | 23.75 | 22.13 |
| | **NeAR** | **21.95** | **23.47** | **24.38** | **22.61** |
| 新视角重光照 | Hunyuan3D-2.1 | 22.30 | 24.89 | 25.47 | 22.26 |
| | **NeAR** | **22.87** | **25.53** | **25.97** | **22.94** |

### 消融实验

| 输入 SLAT 类型 | PSNR↑ | SSIM↑ | LPIPS↓ |
|---------------|-------|-------|--------|
| Shaded SLAT | 28.95 | 0.9281 | 0.0813 |
| Base Color SLAT | 30.38 | 0.9541 | 0.0564 |
| LH-SLAT | 32.02 | 0.9631 | 0.0494 |
| **LH + Base Color** | **32.54** | **0.9649** | **0.0442** |

| LAD 层数 (IAD=12) | PSNR | FPS |
|-------------------|------|-----|
| 1 层 | 31.56 | 48 |
| 3 层 | 32.35 | 38 |
| **6 层** | **32.54** | **30** |
| 9 层 | 32.56 | 23 |

### 关键发现
- LH-SLAT 比 Shaded SLAT PSNR 高 3+ dB，证实光照均匀化的必要性
- LH-SLAT + Base Color SLAT 组合效果最佳，Base Color 补充了高反射材质的信息
- LAD 6 层是性能与速度的最佳权衡点（PSNR 32.54、30 FPS）
- 先注入视角信息再 bake 光照的设计（图9中架构c+d+g）显著优于先 bake 光照再考虑视角
- HY3D-2.1 会将木头错判为金属（金属度图错误），NeAR 的 LH-SLAT 能正确恢复材质

## 亮点与洞察
- **资产-渲染器联合设计范式**：不再把资产生成和渲染看作独立组件，而是通过共享潜空间形成"契约"。这个思路对整个神经图形学栈的设计有启发意义。
- **光照均匀化策略**：不做显式 PBR 分解，而是在潜空间中学习光照去除，巧妙避开了逆渲染的病态性问题。
- **纹理风格迁移应用**：LH-SLAT 可以接受风格化图像输入，实现语义一致的风格迁移 + 真实感重光照，展示了表示的灵活性。

## 局限与展望
- 在透明物体上仍有困难（如头盔），虽然神经渲染器能部分缓解但并未完全解决
- 训练需要大量 3D 资产的多光照渲染，数据准备成本较高
- 仅评估了静态物体，动态场景和人体还未涉及
- 30 FPS 实时性能对于某些应用可能仍不够

## 相关工作与启发
- **vs Trellis**: Trellis 使用 SLAT 但盲目编码光照，NeAR 提出 LH-SLAT 显式消除光照，形成更适合重光照的稳定表示
- **vs DiffusionRenderer**: DiffusionRenderer 基于 G-buffer 做 2D 渲染，缺乏 3D 结构信息，NeAR 的 3D 高斯场在阴影和高光细节上更准确
- **vs HY3D-2.1**: HY3D 解耦PBR分解导致材质误估（如金属度错误），NeAR 避免了显式分解的脆弱性

## 评分
- 新颖性: ⭐⭐⭐⭐ 资产-渲染器耦合设计理念新颖，LH-SLAT 表示有独创性
- 实验充分度: ⭐⭐⭐⭐⭐ 四个子任务、四个数据集、多方法对比、消融实验、应用展示全面
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，对比分析到位
- 价值: ⭐⭐⭐⭐ 对神经渲染和 3D 生成领域的设计范式有重要启发

<!-- RELATED:START -->

## 相关论文

- [Easy3E: Feed-Forward 3D Asset Editing via Rectified Voxel Flow](easy3e_feed-forward_3d_asset_editing_via_rectified_voxel_flow.md)
- [Indoor Asset Detection in Large Scale 360° Drone-Captured Imagery via 3D Gaussian Splatting](indoor_asset_detection_in_large_scale_360_drone-captured_imagery_via_3d_gaussian.md)
- [NTK-Guided Implicit Neural Teaching](ntk-guided_implicit_neural_teaching.md)
- [PhysX-3D: Physical-Grounded 3D Asset Generation](../../NeurIPS2025/3d_vision/physx-3d_physical-grounded_3d_asset_generation.md)
- [Unblur-SLAM: Dense Neural SLAM for Blurry Inputs](unblur-slam_dense_neural_slam_for_blurry_inputs.md)

<!-- RELATED:END -->
