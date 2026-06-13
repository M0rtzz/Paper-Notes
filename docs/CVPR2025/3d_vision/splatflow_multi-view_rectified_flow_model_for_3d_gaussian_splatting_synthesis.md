---
title: >-
  [论文解读] SplatFlow: Multi-View Rectified Flow Model for 3D Gaussian Splatting Synthesis
description: >-
  [CVPR 2025][3D视觉][3D高斯溅射] 提出 SplatFlow 框架，由多视图整流流（RF）模型和高斯溅射解码器（GSDecoder）组成，在潜空间中联合生成多视图图像、深度和相机位姿，并通过免训练反演和修复技术实现统一的 3DGS 生成与编辑。 - 3DGS 已成为高保真实时渲染的主流方案…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D高斯溅射"
  - "整流流模型"
  - "文本到3D生成"
  - "3D编辑"
  - "多视图生成"
---

# SplatFlow: Multi-View Rectified Flow Model for 3D Gaussian Splatting Synthesis

**会议**: CVPR 2025  
**arXiv**: [2411.16443](https://arxiv.org/abs/2411.16443)  
**代码**: [项目主页](https://gohyojun15.github.io/SplatFlow/)  
**领域**: 3D视觉  
**关键词**: 3D高斯溅射, 整流流模型, 文本到3D生成, 3D编辑, 多视图生成

## 一句话总结

提出 SplatFlow 框架，由多视图整流流（RF）模型和高斯溅射解码器（GSDecoder）组成，在潜空间中联合生成多视图图像、深度和相机位姿，并通过免训练反演和修复技术实现统一的 3DGS 生成与编辑。

## 研究背景与动机

- 3DGS 已成为高保真实时渲染的主流方案，但现有 3DGS 生成和编辑方法各自独立，缺乏统一框架
- 3DGS 生成方面：SDS 方法需要耗时的逐场景优化；直接生成方法多限于合成物体级数据集，无法处理真实世界场景中多变的场景尺度和相机轨迹
- 3DGS 编辑方面：利用 2D 扩散模型指导编辑需要额外阶段（纹理调整、精细化）或复杂的跨视图一致性模块
- 2D 扩散模型已展示通过反演实现免训练编辑的能力，但这一范式尚未扩展到 3DGS
- 真实世界场景的尺度和相机轨迹各异，需要在生成模型中联合学习相机位姿分布
- 受 2D 扩散模型启发，直接建模 3DGS 的生成模型应该也能通过反演和修复技术实现免训练编辑

## 方法详解

### 整体框架

SplatFlow 包含两个主要组件：（1）多视图整流流（RF）模型在潜空间中条件于文本提示，联合生成多视图图像 latent、深度 latent 和 Plücker 射线坐标（表示相机位姿）；（2）GSDecoder 将这些潜表示转换为像素对齐的 3DGS 表示。利用 SD3 的冻结编码器共享潜空间，并结合免训练的 SDEdit 反演和 RePaint 修复技术支持 3DGS 编辑和多种 3D 任务。

### 关键设计

**1. 多视图整流流模型**

- **功能**：从文本提示联合生成多视图一致的图像、深度和相机位姿
- **核心思路**：将每个视图的图像 latent $\mathcal{E}(\bm{I}_i)$、深度 latent $\mathcal{E}(\bm{D}_i)$ 和 Plücker 射线 $\bm{r}_i$ 沿通道维度拼接为 $\bm{X}_i \in \mathbb{R}^{(2n+6) \times h \times w}$，$K$ 个视图组成输入 $Y_0 \in \mathbb{R}^{K \times (2n+6) \times h \times w}$。在此上训练条件流匹配目标。采样时在每步预测 $t=0$ 处的结果并回投至射线流形以保持相机位姿精度。可融合 SD3 的向量场提升单视图质量
- **设计动机**：联合建模而非分别建模相机位姿和图像的好处是：(1) 可通过修复技术灵活处理多种任务（已知部分约束预测未知部分），(2) 真实场景需要自适应相机位姿

**2. 高斯溅射解码器（GSDecoder）**

- **功能**：将多视图潜表示高效转换为像素对齐的 3DGS
- **核心思路**：基于前馈式 3DGS 重建方法设计，输入为 $K$ 个视图的图像 latent、深度 latent 和相机位姿。引入深度 latent 集成增强 3D 结构信息，使用 DepthAnythingV2 提取深度图。对抗损失（vision-aided loss）在收敛后期加入以提升视觉质量而不破坏训练稳定性。架构基于 SD3 解码器初始化，增加跨视图注意力
- **设计动机**：冻结编码器虽然保证了与 2D 生成模型的兼容性，但可能丢失细粒度空间细节。深度 latent 补充 3D 结构信息，对抗损失提升感知质量

**3. 免训练反演与修复编辑**

- **功能**：在仅训练生成任务的情况下实现 3DGS 编辑和多种 3D 任务
- **核心思路**：3DGS 编辑：对输入多视图 latent 使用 SDEdit 反演到 $t_k$，后用目标文本条件重新采样生成编辑后的 latent。3D 任务：利用联合建模的特性，将已知数据（如多视图图像+深度）作为约束，通过 RePaint 修复推理未知部分（如相机位姿），实现相机位姿估计和新视图合成
- **设计动机**：2D 扩散模型已证明反演和修复是强大的免训练编辑工具，将其扩展到多视图 3D 模型是自然延伸

### 损失函数 / 训练策略

- RF 模型：条件流匹配损失 $\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t,Y_t,Y_1}[\|u_t(Y_t|Y_1) - u_\theta(Y_t,t)\|_2^2]$
- GSDecoder：LPIPS + MSE + vision-aided 对抗损失（延迟启用）
- $K=8$ 视图设定，基于 SD3 微调，调整输入输出通道并加入跨视图注意力
- 训练数据：MVImgNet + DL3DV-7K 子集，使用 Llava-One Vision Qwen 7B 生成文本描述

## 实验关键数据

### 主实验

文本到 3DGS 生成（MVImgNet / DL3DV）：

| 方法 | MVImgNet FID↓ | MVImgNet CLIP↑ | DL3DV FID↓ | DL3DV CLIP↑ |
|------|-------------|---------------|------------|-------------|
| Director3D | 39.55 | 30.48 | 88.44 | 30.04 |
| Director3D+SDS++ | 41.80 | 31.00 | 95.88 | 31.68 |
| **SplatFlow** | **34.85** | **31.43** | **79.91** | 30.06 |
| SplatFlow+SDS++ | 35.46 | **32.30** | 85.31 | **31.90** |

### 消融实验

GSDecoder 组件消融：

| 配置 | PSNR↑ | LPIPS↓ |
|------|-------|--------|
| 仅图像 latent | 20.3 | 0.32 |
| +深度 latent | 22.1 | 0.26 |
| +对抗损失 | **23.5** | **0.21** |

### 关键发现

1. SplatFlow 在更小的训练数据集上超越 Director3D（FID 34.85 vs 39.55），证明了联合建模的优势
2. 深度 latent 集成显著提升 GSDecoder 的收敛速度和重建质量
3. 免训练编辑在 3DGS 场景中效果良好，无需额外的跨视图一致性模块
4. 采样过程中的射线流形约束对相机位姿估计精度至关重要
5. 可直接通过修复技术实现新视图合成和相机位姿估计

## 亮点与洞察

- 首次实现了 3DGS 生成与编辑的统一框架，仅需训练生成模型即可免训练执行编辑和多种 3D 任务
- 联合建模图像+深度+相机位姿的设计优雅，使修复技术可灵活推理任何缺失模态
- 共享 SD3 编码器的设计实现了与 2D 生成模型的兼容性，可在采样时融合 SD3 知识
- 射线流形约束是针对整流流模型的新颖技术洞察

## 局限与展望

- 训练数据规模有限（MVImgNet + DL3DV-7K 子集），扩展到更大数据集可能进一步提升质量
- 8 视图设定可能不足以覆盖复杂大场景
- 免训练编辑的效果受限于 RF 模型的生成先验质量
- 未来可扩展到动态 3DGS 场景的生成与编辑
- 引入更精细的编辑控制（如局部编辑、物理约束）

## 相关工作与启发

- **Director3D**: 从文本生成相机位姿再生成多视图图像，SplatFlow 改为联合学习分布获得更好效果
- **LucidDreamer / DreamScene**: 基于单视图修复或 SDS 的场景生成方法，在大轨迹变化时不稳定
- **SDEdit / RePaint**: 2D 扩散模型中的免训练编辑/修复技术，本文将其扩展到整流流和 3D 场景
- 启发：将 2D 生成模型的多样化免训练能力（编辑、修复、反演）提升到 3D 是一个有前途的研究方向

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 统一生成编辑框架和联合建模思路新颖
- **实验充分度**: ⭐⭐⭐⭐ — 两个真实世界数据集评测，涵盖生成、编辑、NVS、位姿估计
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，技术细节充分
- **价值**: ⭐⭐⭐⭐ — 为 3DGS 生成与编辑提供了简洁统一的解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Sharp-It: A Multi-view to Multi-view Diffusion Model for 3D Synthesis and Manipulation](sharp-it_a_multi-view_to_multi-view_diffusion_model_for_3d_synthesis_and_manipul.md)
- [\[CVPR 2025\] CoMapGS: Covisibility Map-based Gaussian Splatting for Sparse Novel View Synthesis](comapgs_covisibility_map-based_gaussian_splatting_for_sparse_novel_view_synthesi.md)
- [\[CVPR 2025\] RainyGS: Efficient Rain Synthesis with Physically-Based Gaussian Splatting](rainygs_efficient_rain_synthesis_with_physically-based_gaussian_splatting.md)
- [\[CVPR 2025\] S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting](s2gaussian_sparse-view_super-resolution_3d_gaussian_splatting.md)
- [\[CVPR 2025\] MVGD: Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion](zero-shot_novel_view_and_depth_synthesis_with_multi-view_geometric_diffusion.md)

</div>

<!-- RELATED:END -->
