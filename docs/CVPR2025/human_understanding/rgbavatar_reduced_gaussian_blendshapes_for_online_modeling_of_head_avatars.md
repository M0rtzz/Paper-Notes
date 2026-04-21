---
title: >-
  [论文解读] RGBAvatar: Reduced Gaussian Blendshapes for Online Modeling of Head Avatars
description: >-
  [CVPR 2025][人体理解][头部虚拟形象] RGBAvatar提出"精简高斯混合形状"表示，仅用20个可学习基底即可高效表征可动画头部虚拟形象，配合批量并行渲染和颜色初始化策略，首次实现在线实时（边拍边建）的头部虚拟形象重建。
tags:
  - CVPR 2025
  - 人体理解
  - 头部虚拟形象
  - 高斯混合形状
  - 实时重建
  - 在线建模
  - 面部动画
---

# RGBAvatar: Reduced Gaussian Blendshapes for Online Modeling of Head Avatars

**会议**: CVPR 2025  
**arXiv**: [2503.12886](https://arxiv.org/abs/2503.12886)  
**代码**: [github.com/gapszju/RGBAvatar](https://github.com/gapszju/RGBAvatar)  
**领域**: Human Understanding / 3D Vision  
**关键词**: 头部虚拟形象, 高斯混合形状, 实时重建, 在线建模, 面部动画

## 一句话总结

RGBAvatar提出"精简高斯混合形状"表示，仅用20个可学习基底即可高效表征可动画头部虚拟形象，配合批量并行渲染和颜色初始化策略，首次实现在线实时（边拍边建）的头部虚拟形象重建。

## 研究背景与动机

3DGS极大推动了可动画头部虚拟形象的重建质量。现有方法（如GaussianBlendshapes）用3DMM的预定义blendshape基底线性混合高斯属性，取得了较好效果。但存在两个问题：

- **参数量随基底数线性增长**：FLAME有50+个blendshape基底，每个都对应完整的高斯属性集合，导致训练慢、内存消耗大
- **预定义基底不够个性化**：通用3DMM基底无法最优地捕捉特定个体的面部细节

核心创新：用MLP隐式学习一组**精简的、个性化的**blendshape基底（仅20个），而非绑定到3DMM的固定基底。同时通过GPU优化实现前所未有的训练速度（630帧/秒）和渲染速度（400 FPS）。

## 方法详解

### 整体框架

输入单目视频，FLAME追踪得到参数 $\theta$，MLP $\mathcal{F}$ 将 $\theta$ 映射为精简权重 $\psi \in \mathbb{R}^K$（$K=20$），线性混合基底模型 $G^\psi = G_0 + \sum_{k=1}^K \psi_k \Delta G_k$，最后根据FLAME mesh变形将高斯变换到目标空间渲染。

### 关键设计

**1. 精简高斯混合形状（Reduced Gaussian Blendshapes）**

- **功能**：用极少量可学习基底（$K=20$）高效表示任意面部表情
- **核心思路**：不使用3DMM的固定blendshape基底，而是让MLP $\mathcal{F}: \mathbb{R}^H \rightarrow \mathbb{R}^K$ 学习从FLAME参数到精简权重的映射。基底 $\{\Delta G_k\}$ 和MLP同时优化，模型自适应发现紧凑的基底组合
- **设计动机**：通用3DMM基底是为全人群设计的，对特定个体存在冗余。精简到20个基底不仅减少参数量，还能通过端到端优化获得比50个3DMM基底更好的重建质量

**2. 颜色初始化估计 + 批量并行高斯光栅化**

- **功能**：将训练吞吐量从约100帧/秒提升到630帧/秒，80秒完成重建
- **核心思路**：颜色初始化将投影到2D的高斯视为高斯核，直接通过加权卷积估计初始颜色 $\mathbf{c}^{\text{init}} = \frac{\sum w_{ij} \mathbf{I}_{ij}}{\sum w_{ij}}$。批量并行渲染将多个样本的preprocess和rasterize分离为两阶段，仅需一次GPU-CPU同步，并利用CUDA Streams实现100% Stream Processor利用率
- **设计动机**：头部虚拟形象通常<100k高斯，传统单样本训练导致GPU利用率<60%，瓶颈在GPU-CPU同步而非计算

**3. 局部-全局采样的在线重建策略**

- **功能**：实现边拍摄边重建的在线模式，质量接近离线重建
- **核心思路**：维护局部采样池 $\mathcal{M}_l$（大小150，FIFO存放新帧）和全局采样池 $\mathcal{M}_g$（大小1000，水库采样存放历史帧）。每批次按 $\eta=0.7$ 比例从两个池采样，平衡快速适配新数据和防止遗忘
- **设计动机**：在线数据流的核心矛盾是新帧快速收敛 vs 旧帧遗忘防止。局部池保证快速适配，全局池+水库采样保证每帧均等保留概率

### 损失函数

L1颜色重建损失 + 随机背景颜色正则（约束高斯在头部区域内）。

## 实验关键数据

### 主实验：INSTA & GaussianBlendShapes数据集

| 方法 | INSTA平均PSNR↑ | GBS平均PSNR↑ | 训练时间 | 渲染FPS |
|------|--------------|-------------|---------|--------|
| GaussianAvatars | 29.6 | 32.8 | 慢 | ~200 |
| FlashAvatar | 28.1 | 31.2 | 中等 | ~300 |
| GaussianBlendShapes | 30.4 | 33.4 | 慢 | ~300 |
| **RGBAvatar (K=20)** | **31.2** | **34.0** | **80秒** | **~400** |

### 消融实验：基底数量影响

| 基底数K | PSNR↑ | 训练时间↓ |
|--------|-------|---------|
| 10 | 30.5 | ~60秒 |
| 20 | 31.2 | ~80秒 |
| 50 | 31.3 | ~150秒 |
| GBS-50基底 | 30.4 | ~300秒 |

### 关键发现

- **仅20个精简基底即超越50个3DMM基底**的重建质量（PSNR提升~0.8dB），同时训练速度快3-4倍
- 批量并行渲染将GPU利用率从60%提升到100%，训练吞吐量提升6倍
- 在线重建质量（逐帧到达）接近离线重建（全数据集），证明局部-全局采样策略有效
- 颜色初始化仅执行一次（首次splatting权重超阈值时），但显著加速早期收敛
- 方法兼容多种3DMM追踪器，不依赖特定追踪方案

## 亮点与洞察

1. **个性化基底替代通用基底**是一个简洁但有力的insight：特定个体的面部变化空间远小于全人群，20维即可充分表达
2. **系统级GPU优化**（CUDA Streams批量渲染+单次同步）对所有3DGS训练场景都有启发价值
3. 首次实现**在线实时**头部虚拟形象重建，使实时视频通话中即时生成虚拟形象成为可能

## 局限与展望

- 颜色初始化依赖于高斯投影可见性，对极端角度的初始化可能不稳定
- 在线模式下的水库采样策略较简单，未考虑样本重要性差异
- 未处理头发和配饰等非FLAME建模区域的细节
- 未来可探索更自适应的基底数量选择和流式数据的增量基底学习

## 相关工作与启发

- **与GaussianBlendshapes的关系**：直接改进——用精简可学习基底替代固定3DMM基底
- **与FlashAvatar的关系**：同样将高斯绑定到FLAME mesh上，但RGBAvatar用线性混合替代MLP偏移
- **启发**：在参数化人体建模中，"少而精"的可学习基底比"多而泛"的通用基底更高效

## 评分

⭐⭐⭐⭐

精简基底的核心创新简洁有效，配合系统级GPU优化实现了80秒重建+400FPS渲染的impressive速度。在线重建是一个有实际应用价值的新能力。整体技术完成度高，代码开源。

<!-- RELATED:START -->

## 相关论文

- [Avat3r: Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars](../../ICCV2025/human_understanding/avat3r_large_animatable_gaussian_reconstruction_model_for_hi.md)
- [VASA-3D: Lifelike Audio-Driven Gaussian Head Avatars from a Single Image](../../NeurIPS2025/human_understanding/vasa-3d_lifelike_audio-driven_gaussian_head_avatars_from_a_single_image.md)
- [FATE: Full-head Gaussian Avatar with Textural Editing from Monocular Video](fate_full-head_gaussian_avatar_with_textural_editing_from_monocular_video.md)
- [WildAvatar: Learning In-the-Wild 3D Avatars from the Web](wildavatar_learning_in-the-wild_3d_avatars_from_the_web.md)
- [GGTalker: Talking Head Synthesis with Generalizable Gaussian Priors and Identity-Specific Adaptation](../../ICCV2025/human_understanding/ggtalker_talking_head_systhesis_with_generalizable_gaussian_priors_and_identity-.md)

<!-- RELATED:END -->
