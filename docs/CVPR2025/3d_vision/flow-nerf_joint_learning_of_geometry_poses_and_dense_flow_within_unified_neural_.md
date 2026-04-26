---
title: >-
  [论文解读] Flow-NeRF: Joint Learning of Geometry, Poses, and Dense Flow within Unified Neural Representations
description: >-
  [CVPR 2025][3D视觉][神经辐射场] 提出 Flow-NeRF，首次在无位姿 NeRF 框架中将场景几何、相机位姿和密集光流作为统一的联合优化目标，通过共享点采样、位姿条件化双射映射和特征消息传递机制，在新视角合成和深度估计上大幅超越先前方法，同时首次定义并实现了新视角光流估计。
tags:
  - CVPR 2025
  - 3D视觉
  - 神经辐射场
  - 无位姿NeRF
  - 光流
  - 联合优化
  - 新视角流
---

# Flow-NeRF: Joint Learning of Geometry, Poses, and Dense Flow within Unified Neural Representations

**会议**: CVPR 2025  
**arXiv**: [2503.10464](https://arxiv.org/abs/2503.10464)  
**代码**: https://zhengxunzhi.github.io/flownerf/  
**领域**: 3D视觉  
**关键词**: 神经辐射场, 无位姿NeRF, 光流, 联合优化, 新视角流

## 一句话总结
提出 Flow-NeRF，首次在无位姿 NeRF 框架中将场景几何、相机位姿和密集光流作为统一的联合优化目标，通过共享点采样、位姿条件化双射映射和特征消息传递机制，在新视角合成和深度估计上大幅超越先前方法，同时首次定义并实现了新视角光流估计。

## 研究背景与动机

1. **领域现状**：NeRF 通常需要 COLMAP 等 SfM 管线提供相机位姿。无位姿 NeRF（如 BARF、Nope-NeRF）可将位姿与场景几何联合优化，但缺乏跨帧一致性约束导致重建质量差。
2. **现有痛点**：部分方法利用光流监督来约束位姿学习（如 LocalRF），但仅将光流视为位姿优化的正则化项，未探索光流对新视角合成和场景几何的增益潜力；同时现有方法依赖预训练的光流模型（如 RAFT），受制于其质量上限。
3. **核心矛盾**：光流提供丰富的跨视角密集对应信息，但现有框架未能将其作为一等优化目标纳入统一表示，导致光流信息未被充分利用。
4. **本文目标** (1) 如何在无位姿NeRF中联合学习光流？(2) 如何让光流学习反过来提升几何重建？(3) 如何实现新视角之间的密集对应估计（新视角光流）？
5. **切入角度**：作者的关键观察是——几何场和光流场应该共享底层场景表示，因为它们本质上建模的是同一个物理3D场景。通过将位姿作为帧标识符来条件化光流预测，可实现对训练视角之外的新视角光流推理。
6. **核心 idea**：用统一的神经场景表示同时学习几何、位姿和密集光流，并通过规范空间到世界空间的特征消息传递让光流学习增强几何重建。

## 方法详解

### 整体框架
输入为一组无位姿图像序列，框架包含几何分支和光流分支。每次迭代选取两帧 $I_i$ 和 $I_j$，通过共享点采样确保两个分支处理的是同一物理场景点。几何分支用标准 NeRF 的体渲染学习外观和深度；光流分支通过双射映射（Real-NVP）学习帧间2D-2D对应关系。两个分支通过特征消息传递机制耦合，规范空间特征增强世界空间表示。

### 关键设计

1. **共享点采样机制 (Shared Points Sampling)**:

    - 功能：确保几何分支和光流分支处理同一组物理场景点
    - 核心思路：每次迭代在帧 $i$ 上随机采样 $N=1024$ 个2D像素点，这些点被共享给两个分支。几何分支用已知内参 $K$ 和可学习位姿 $T$ 将其反投影到世界空间 $\mathbf{p}_i = z_i \mathbf{r}$（$\mathbf{r} = TK^{-1}[u,v]$）；光流分支则用内参反投影到相机空间 $K^{-1}[u,v]d_i$，两者通过固定比例 $z_i = \alpha d_i$（$\alpha=0.2$）保持对应关系。
    - 设计动机：使用相机空间投影（而非简单的正交投影）保留了场景的透视关系，实验证明这对深度估计至关重要。

2. **位姿条件化双射映射 (Pose-Conditioned Bijective Mapping)**:

    - 功能：学习帧间像素级2D-2D对应关系，并支持新视角光流推理
    - 核心思路：使用可逆神经网络 Real-NVP 参数化双射映射 $\epsilon$，将帧 $i$ 相机空间中的3D点 $\mathbf{O}_i$ 映射到规范3D空间中的点 $\mathbf{r}$，再映射回帧 $j$ 相机空间的点 $\mathbf{O}_j$。关键创新是用6-DoF位姿向量 $[r_1,r_2,r_3,t_1,t_2,t_3]$ 作为帧标识符，而非时间索引。位姿随优化过程动态变化，且携带物理几何信息，因此可以推广到训练视角之外的新视角。
    - 设计动机：时间条件化只能在训练视角推理光流（因为时间不携带相机运动的物理信息），而位姿条件化使得模型可以用任意位姿查询对应的光流，实现了前所未有的新视角光流估计。

3. **特征消息传递 (Feature Message Passing)**:

    - 功能：将光流分支的规范空间特征传递给几何分支，增强场景几何重建
    - 核心思路：使用3层256维的 Gabornet 从规范空间3D点提取128维特征，直接拼接到几何分支 MLP $F_{\theta1}$ 中跳跃连接之后的中间层。两个分支使用不同的损失函数优化（光度一致性 vs 光流监督），因此学到互补的特征表示。光流分支提供的2D对应信息能约束更准确的几何估计。
    - 设计动机：基于一个洞察——规范空间特征和世界空间特征虽来自不同分支、用不同损失优化，但表示的是同一物理场景，因此是互补的。将规范空间的运动一致性知识注入几何分支，可大幅改善新视角合成和深度预测。

### 损失函数 / 训练策略
- **RGB渲染损失**：$L_{rgb} = \frac{1}{N}\sum ||\hat{\mathbf{C}}(\mathbf{p}) - \mathbf{C}(\mathbf{p})||_1$，标准光度一致性损失
- **光流损失**：$L_{flow} = \frac{1}{N}\sum ||\hat{\mathbf{p}}_j - \mathbf{p}_j||_1$，预测2D点与 RAFT 提供的伪光流之间的 L1 距离
- 此外还包含3D点云损失和2D光度 warping 损失，两个分支联合端到端优化

## 实验关键数据

### 主实验

| 数据集 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|------|-------|-------|--------|
| Tanks & Temples (均值) | BARF | 23.42 | 0.61 | 0.54 |
| Tanks & Temples (均值) | Nope-NeRF | 26.34 | 0.74 | 0.39 |
| Tanks & Temples (均值) | **Flow-NeRF** | **28.73** | **0.82** | **0.29** |
| ScanNet (均值) | BARF | 31.41 | 0.82 | 0.39 |
| ScanNet (均值) | Nope-NeRF | 31.86 | 0.83 | 0.38 |
| ScanNet (均值) | **Flow-NeRF** | **32.55** | **0.85** | **0.34** |

### 消融实验（长程光流EPE，Sintel数据集）

| 方法 | 帧间隔=1 | 帧间隔=8 | 帧间隔=16 |
|------|---------|---------|----------|
| RAFT（预训练+微调） | **0.455** | 1.567 | 2.089 |
| Flow-NeRF | 0.689 | **1.318** | **1.683** |

### 关键发现
- **特征消息传递贡献最大**：从规范空间向世界空间传递特征使得 Tanks & Temples 上 PSNR 平均提升超过2点，ScanNet 上提升0.8点，深度指标全面改善。
- **长程光流显著优于 RAFT**：虽然模型仅用连续帧前向光流训练，但在帧间隔16时 EPE 为 1.683（RAFT 为 2.089），且能推理前向和后向长程光流。
- **位姿条件化是新视角光流的关键**：时间条件化只能在训练视角推理，位姿条件化实现了真正的新视角光流估计能力。

## 亮点与洞察
- **光流从正则化项升级为一等优化目标**：之前工作仅用光流约束位姿，本文证明将其作为显式目标能反过来增强几何重建——这体现了"多任务联合学习"在3D场景理解中的深层价值。
- **新视角光流的定义与实现**：首次定义了"新视角光流"这一新任务——推理任意未见视角之间的密集对应，为全面的场景建模开辟了新方向。
- **规范空间→世界空间的隐式蒸馏**：通过特征消息传递实现了从光流学习到几何学习的知识迁移，思路简洁但效果显著，可迁移到任何多分支神经场架构。

## 局限与展望
- **仅处理静态场景**：假设所有光流由相机运动引起，无法处理动态物体。
- **依赖 RAFT 伪光流**：训练监督的质量受限于 RAFT 的性能上限，特别是在纹理稀疏区域。
- **计算开销**：双分支结构增加了训练时间和内存需求。
- **可改进**：扩展到动态场景（引入物体运动建模）；用自监督方式替代 RAFT 伪标签；将消息传递机制用于3D Gaussian Splatting 等更高效的场景表示。

## 相关工作与启发
- **vs BARF**：BARF 用由粗到细的位置编码解决位姿-几何联合优化，但缺乏跨帧对应约束，在复杂场景表现差。Flow-NeRF 通过光流提供强跨帧约束，PSNR 提升5+点。
- **vs Nope-NeRF**：Nope-NeRF 用单目深度先验正则化位姿，但依赖预训练深度模型质量。Flow-NeRF 通过联合学习光流获得更强的几何约束。
- **vs Omnimotion**：Omnimotion 用时间条件化学习对应场但只能查询训练视角。Flow-NeRF 的位姿条件化设计实现了新视角泛化。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将光流作为NeRF的联合优化目标，并定义了新视角光流任务
- 实验充分度: ⭐⭐⭐⭐ 多数据集多任务评估（NVS、深度、位姿、光流），消融全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机明确，技术细节完整
- 价值: ⭐⭐⭐⭐ 统一框架的思路有重要启发意义，新视角光流可能催生新的下游应用

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Floxels: Fast Unsupervised Voxel Based Scene Flow Estimation](floxels_fast_unsupervised_voxel_based_scene_flow_estimation.md)
- [\[CVPR 2025\] Zero-Shot Monocular Scene Flow Estimation in the Wild](zero-shot_monocular_scene_flow_estimation_in_the_wild.md)
- [\[CVPR 2025\] End-to-End Implicit Neural Representations for Classification](end-to-end_implicit_neural_representations_for_classification.md)
- [\[CVPR 2025\] A Unified Image-Dense Annotation Generation Model for Underwater Scenes](a_unified_image-dense_annotation_generation_model_for_underwater_scenes.md)
- [\[CVPR 2025\] PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields](pbr-nerf_inverse_rendering_with_physics-based_neural_fields.md)

<!-- RELATED:END -->
