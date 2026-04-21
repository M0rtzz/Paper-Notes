---
title: >-
  [论文解读] DMAligner: Enhancing Image Alignment via Diffusion Model Based View Synthesis
description: >-
  [CVPR2026][3D视觉][image alignment] 提出 DMAligner，将图像对齐问题从传统的光流 warp 范式转化为"对齐导向的视图合成"任务，利用条件扩散模型直接生成对齐后的完整图像，配合专门构建的 DSIA 合成数据集和动态感知掩码模块（DMP），有效避免了 warp 方法固有的 ghosting 和遮挡伪影，在多个基准上全面超越现有方法。
tags:
  - CVPR2026
  - 3D视觉
  - image alignment
  - 扩散模型
  - view synthesis
  - dynamic scenes
  - occlusion handling
---

# DMAligner: Enhancing Image Alignment via Diffusion Model Based View Synthesis

**会议**: CVPR2026  
**arXiv**: [2602.23022](https://arxiv.org/abs/2602.23022)  
**代码**: [boomluo02/DMAligner](https://github.com/boomluo02/DMAligner)  
**领域**: 3d_vision  
**关键词**: image alignment, diffusion model, view synthesis, dynamic scenes, occlusion handling

## 一句话总结

提出 DMAligner，将图像对齐问题从传统的光流 warp 范式转化为"对齐导向的视图合成"任务，利用条件扩散模型直接生成对齐后的完整图像，配合专门构建的 DSIA 合成数据集和动态感知掩码模块（DMP），有效避免了 warp 方法固有的 ghosting 和遮挡伪影，在多个基准上全面超越现有方法。

## 背景与动机

图像对齐（Image Alignment）是计算机视觉中的基础任务，目标是将不同视角或不同时刻拍摄的两幅图像对齐到统一的坐标系中。它在视频稳定、全景拼接、超分辨率、多帧去噪等应用中至关重要。

传统图像对齐方法的流程通常包括：

1. **光流估计**：计算像素级的运动场（optical flow），如 RAFT、FlowFormer 等
2. **图像 warp**：利用估计的光流对源图像进行反向变换，生成对齐结果
3. **后处理**：融合或修补 warp 后的伪影

这一范式存在两个根本性问题：

- **遮挡区域无法处理**：当目标视角中的某些区域在源图像中被遮挡时，光流无法提供有效的对应关系，warp 后产生空洞或 ghosting 伪影
- **动态物体干扰**：场景中运动物体改变了几何关系，导致光流估计不准确，warp 结果在动态区域出现严重的重影

核心动机：**能否绕过"估计光流→warp"的间接范式，直接生成对齐后的完整图像？** 受扩散模型在图像生成方面的强大能力启发，作者提出将对齐问题重新定义为条件图像生成问题。

## 核心问题

- 传统 warp-based 对齐方法在遮挡和动态区域不可避免地产生 ghosting 伪影
- 生成式方法需要大量高质量训练数据——现有数据集缺少面向对齐任务的配对标注（时间 t2 + 相机位姿 P1 的 ground truth 图像）
- 扩散模型需要学会区分动态前景与静态背景，以正确处理场景中的运动物体

## 方法详解

### 任务重新定义

传统对齐：给定参考图 $I_{ref}$（时间 $t_1$，相机 $P_1$）和源图 $I_{src}$（时间 $t_2$，相机 $P_2$），将 $I_{src}$ warp 到 $P_1$ 坐标系。

DMAligner 的视角：直接生成目标图像 $I_{gt}$（时间 $t_2$，相机 $P_1$），即保留 $t_2$ 时刻的场景内容但从 $P_1$ 的视角观察。这实质上是一个条件视图合成问题。

### DSIA 数据集构建

为了获取 $I_{gt}$（时间 $t_2$ + 相机 $P_1$）这种在真实场景中无法采集的 ground truth，作者使用 Blender 构建了 **DSIA**（Dynamic Scene Image Alignment）合成数据集：

- **场景多样性**：25 种人物角色 + 100+ 种物体模型 + 多种相机运动轨迹
- **动态场景模拟**：角色执行行走、跑步、挥手等动作，物体进行平移、旋转等运动
- **数据渲染**：对每个场景，分别渲染 $I_{ref}(t_1, P_1)$、$I_{src}(t_2, P_2)$ 和 $I_{gt}(t_2, P_1)$
- **数据规模**：共 1033 个场景，生成 30K+ 高质量图像对
- **相机运动类型**：前移、后移、左移、右移、旋转等多种类型，覆盖各种对齐场景

### Dynamics-aware Diffusion Training

基于 Latent Diffusion Model (LDM) 框架进行条件生成训练：

**编码阶段**：将参考图 $I_{ref}$ 和源图 $I_{src}$ 分别通过 VAE 编码器映射到潜空间，得到 $z_{ref}$ 和 $z_{src}$

**前向扩散**：对 ground truth 的潜表示 $z_{gt}$ 逐步添加高斯噪声

$$z_t = \sqrt{\bar{\alpha}_t} z_{gt} + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**条件去噪**：U-Net 以拼接的 $[z_t, z_{ref}, z_{src}]$ 作为输入，学习预测噪声 $\epsilon_\theta$

$$\mathcal{L} = \mathbb{E}_{z_{gt}, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(z_t, t, z_{ref}, z_{src}) \|_2^2 \right]$$

### DMP（Dynamics-aware Mask Producing）模块

DMP 模块是从 U-Net 的中间隐特征中提取动态感知掩码的关键组件：

1. **特征提取**：从 U-Net decoder 的多尺度特征中提取中间表示 $F_{mid}$
2. **掩码预测**：通过轻量级卷积头将 $F_{mid}$ 映射为二值掩码 $M_{dyn}$，区分动态前景和静态背景
3. **动态区域增强**：利用 $M_{dyn}$ 对去噪过程施加空间注意力指导——动态区域需要更多生成能力，静态区域主要依赖几何变换
4. **辅助损失**：使用光流不一致性作为伪标签监督掩码预测

$$\mathcal{L}_{mask} = \text{BCE}(M_{dyn}, M_{pseudo})$$

DMP 增强了网络对动态信息的显式建模能力，使模型能够根据区域属性自适应地分配生成资源。

### 推理流程

1. 输入参考图 $I_{ref}$ 和源图 $I_{src}$
2. VAE 编码到潜空间
3. 从纯高斯噪声出发，以 $z_{ref}$ 和 $z_{src}$ 为条件执行 DDIM 去噪
4. DMP 模块在去噪过程中提供动态感知引导
5. VAE 解码得到最终对齐图像 $I_{align}$

## 实验关键数据

### DSIA 测试集

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| RAFT + Warp | 22.31 | 0.782 | 0.189 |
| FlowFormer + Warp | 23.15 | 0.801 | 0.172 |
| LoFTR + Warp | 21.87 | 0.764 | 0.203 |
| **DMAligner** | **27.43** | **0.893** | **0.078** |

在合成数据集上 PSNR 提升 4+ dB，LPIPS 降低约 50%。

### MPI Sintel 评估

| 方法 | 遮挡区域 PSNR↑ | 动态区域 PSNR↑ |
|------|---------------|---------------|
| RAFT + Warp | 18.7 | 17.2 |
| **DMAligner** | **23.1** | **22.6** |

在遮挡和动态区域的优势尤为显著，验证了生成式方法在这些困难区域的本质优势。

### DAVIS 视频序列

在真实动态视频上的定性评估表明：DMAligner 生成的对齐结果没有 ghosting 伪影，即使在大运动和严重遮挡的情况下也能产生视觉上自然的结果。

### 消融实验

- **去掉 DMP 模块**：PSNR 从 27.43 降至 26.15（-1.28），证明动态感知引导的有效性
- **去掉 DSIA 数据预训练**：PSNR 降至 24.87，说明合成数据对学习对齐先验至关重要
- **仅用静态场景训练**：动态区域 PSNR 大幅下降 3.2 dB，验证动态场景数据的必要性

## 亮点

- **范式转换**：从"光流估计→warp"的间接范式转向"条件生成"的直接范式，从根本上避免了遮挡和 ghosting 问题
- **DSIA 数据集设计精巧**：利用 Blender 渲染 $I_{gt}(t_2, P_1)$ 这种真实世界无法采集的 GT，巧妙解决了训练数据问题
- **DMP 模块轻量有效**：从隐特征中提取动态掩码，无需额外的动态检测模块，即插即用地增强扩散模型对动态场景的处理能力
- **无需光流估计**：端到端地完成图像对齐，避免了光流估计误差的级联传播

## 局限与展望

- 基于扩散模型的生成方式推理速度较慢，DDIM 采样仍需多步迭代，实时性不足
- DSIA 合成数据集的域差距（domain gap）可能限制真实场景的泛化能力
- 30K+ 的训练数据规模相对有限，场景多样性（25 角色 + 100 物体）可进一步扩展
- 未探索在大分辨率（如 4K）图像上的对齐效果和效率
- 与 NeRF/3DGS 等神经场景表示结合的可能性未被探索

## 与相关工作的对比

| 维度 | 传统 warp 方法 | 深度 homography | DMAligner |
|------|--------------|----------------|-----------|
| 核心操作 | 光流估计→像素 warp | 学习全局变换矩阵 | 条件扩散生成 |
| 遮挡处理 | 无法处理，产生空洞 | 依赖 inpainting 后处理 | 生成式天然填充 |
| 动态物体 | ghosting 严重 | 假设静态场景 | DMP 显式建模 |
| 训练数据 | 无需训练 / 光流 GT | 图像对 + 变换矩阵 | DSIA 合成数据 |
| 推理速度 | 快（单次前向） | 快（单次前向） | 较慢（多步去噪） |

核心差异在于 DMAligner 将对齐重新定义为生成问题，用扩散模型的生成能力弥补了 warp 方法在遮挡/动态区域的固有缺陷。

## 启发与关联

- "将 X 问题转化为条件生成问题"的思路具有通用性，可推广到其他存在遮挡困难的视觉任务（如立体匹配中的遮挡区域、视频修复等）
- DSIA 数据集的构建思路——用渲染引擎生成真实世界无法采集的 GT——可借鉴到其他需要特殊标注的任务
- DMP 模块的"从隐特征中挖掘辅助信息"的设计思路，类似于 DiffRefiner 中的 FGSIM，均在生成过程融入语义理解
- 后续可探索蒸馏或 consistency model 加速推理，使生成式对齐方法达到实时性

## 评分

- 新颖性: 8/10 — 将对齐问题重新建模为视图合成，范式转换清晰且合理
- 实验充分度: 7/10 — 合成和真实数据均有评估，但缺少与更多最新生成式方法的对比
- 写作质量: 8/10 — 问题定义清晰，方法阐述流畅，DSIA 数据集构建描述详尽
- 价值: 7/10 — 提供了对齐问题的新思路，但推理效率限制了即时应用前景

<!-- RELATED:START -->

## 相关论文

- [PR-IQA: Partial-Reference Image Quality Assessment for Diffusion-Based Novel View Synthesis](pr-iqa_partial-reference_image_quality_assessment_for_diffusion-based_novel_view.md)
- [VLM-Guided Group Preference Alignment for Diffusion-based Human Mesh Recovery](vlm-guided_group_preference_alignment_for_diffusion-based_human_mesh_recovery.md)
- [PolarAnything: Diffusion-based Polarimetric Image Synthesis](../../ICCV2025/3d_vision/polaranything_diffusion-based_polarimetric_image_synthesis.md)
- [SIR-DIFF: Sparse Image Sets Restoration with Multi-View Diffusion Model](../../CVPR2025/3d_vision/sir-diff_sparse_image_sets_restoration_with_multi-view_diffusion_model.md)
- [NanoSD: Edge Efficient Foundation Model for Real Time Image Restoration](nanosd_edge_efficient_foundation_model_for_real_time_image_restoration.md)

<!-- RELATED:END -->
