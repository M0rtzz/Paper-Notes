---
title: >-
  [论文解读] On Denoising Walking Videos for Gait Recognition
description: >-
  [CVPR 2025][3D视觉][步态识别] 提出 DenoisingGait，结合"知识驱动去噪"（利用生成式扩散模型在特定 timestep 下滤除步态无关信息）和"几何驱动去噪"（Feature Matching 模块将多通道扩散特征压缩为二维方向向量），生成全新的 Gait Feature Field 表示，在多个 RGB 步态数据集上取得 SOTA。
tags:
  - CVPR 2025
  - 3D视觉
  - 步态识别
  - 扩散模型
  - 特征去噪
  - 光流场
  - 衣物变化鲁棒性
---

# On Denoising Walking Videos for Gait Recognition

**会议**: CVPR 2025  
**arXiv**: [2505.18582](https://arxiv.org/abs/2505.18582)  
**代码**: https://github.com/ShiqiYu/OpenGait  
**领域**: 3D视觉  
**关键词**: 步态识别, 扩散模型, 特征去噪, 光流场, 衣物变化鲁棒性

## 一句话总结

提出 DenoisingGait，结合"知识驱动去噪"（利用生成式扩散模型在特定 timestep 下滤除步态无关信息）和"几何驱动去噪"（Feature Matching 模块将多通道扩散特征压缩为二维方向向量），生成全新的 Gait Feature Field 表示，在多个 RGB 步态数据集上取得 SOTA。

## 研究背景与动机

**领域现状**：步态识别是一种非侵入式生物特征识别方法，通过行走视频中的体型和肢体运动来识别身份。现有方法主要分为"硬去噪"（使用轮廓、骨架、SMPL 等预定义表示来去除背景和纹理干扰）和"软去噪"（直接在 RGB 视频上使用人体先验抑制非步态信息）。

**现有痛点**：硬去噪方法（轮廓、骨架等）的输入稀疏且信息量少，丢失了许多有利于身份识别的结构细节；软去噪方法虽然保留了更多信息，但仍然难以完全去除衣物纹理和颜色等步态无关因素。特别是在**换装场景**下，RGB 编码的纹理和颜色信息成为识别的主要噪声源。

**核心矛盾**：步态识别需要提取"对衣物和背景不变、但对体型和动作敏感"的特征，而 RGB 图像天然编码了大量身份无关的视觉信息。如何在保留结构信息的同时去除这些"噪声"是核心挑战。

**本文目标**：设计一个兼具知识驱动和几何驱动的去噪框架，从 RGB 视频中提取纯净的步态表示。

**切入角度**：受"what I cannot create, I do not understand"启发，探索生成式扩散模型作为步态表示学习器的潜力。发现通过控制扩散模型的 timestep $t$，可以选择性地滤除不同粒度的 RGB 细节——较大 $t$ 保留整体形状，较小 $t$ 重建精细纹理。在 $t=700$ 时步态识别效果最佳（CCPG 上提升 5.3%），但仍有残留的 RGB 噪声，需要进一步的几何驱动去噪。

**核心 idea**：用扩散模型做粗去噪（selectively filter RGB details），再用 Feature Matching 模块做精去噪（compress to direction vectors），生成类似光流的 Gait Feature Field 作为最终表示。

## 方法详解

### 整体框架

DenoisingGait 的 pipeline：(1) 输入 RGB 帧经 VAE 编码器投影到潜空间，用预训练 Stable Diffusion 在 timestep $t$ 做一步去噪得到扩散特征 $F_l$；(2) Feature Matching 模块将 $F_l$ 通过帧内匹配（$\Delta l=0$）和帧间匹配（$\Delta l>0$）分别生成静态和动态 Gait Feature Field；(3) 两个 Feature Field 并行送入 GaitBase 进行步态识别，用 triplet loss + cross-entropy loss 训练。

### 关键设计

1. **知识驱动去噪（Diffusion-based Denoising）**:

    - 功能：利用预训练扩散模型将 RGB 图像中的步态无关细节滤除
    - 核心思路：给定帧 $I_l$，先用 VAE 编码器得到潜变量 $z = \mathcal{E}(I_l)$，然后用预训练 SD 1.5 的 UNet $\epsilon_\theta$ 在不加随机噪声的情况下做一步去噪：$F_l = \epsilon_\theta(\mathcal{E}(I_l), t)$。关键在于 timestep $t$ 的选择：$t$ 过大则过度模糊丢失结构，$t$ 过小则保留过多纹理细节。实验找到 $t=700$ 为最优。
    - 设计动机：扩散模型在不同 timestep 捕获不同粒度的信息——早期 timestep 对应整体形状，后期对应精细纹理。步态识别需要的恰好是中等粒度的形状信息。这种"用生成模型做判别任务"的思路很有创意。

2. **几何驱动去噪（Feature Matching + Gait Feature Field）**:

    - 功能：将多通道扩散特征压缩为二维方向向量，进一步去除 RGB 编码的噪声
    - 核心思路：对于查询像素 $\langle i,j \rangle$ 的特征 $f^Q_{\langle i,j \rangle}$，在邻域中搜索键特征 $\mathcal{M}^K_{\langle i,j \rangle}$，计算 Softmax 相似度分布 $\mathcal{P}$，然后用固定的方向模板 $\mathcal{T}$（包含各邻居相对位移 $[\hat{i}, \hat{j}]$）加权求和得到方向向量 $G_{\langle i,j \rangle} = \mathcal{P} \cdot \mathcal{T}$。帧内匹配（$\Delta l=0$）得到静态 Gait Feature Field（类似 SIFT 梯度场），帧间匹配（$\Delta l>0$）得到动态 Gait Feature Field（类似光流场）。背景通过轮廓 mask 去除。
    - 设计动机：从多通道特征到二维方向向量的压缩天然过滤了 RGB 编码的高维纹理信息，只保留局部结构和运动的方向性特征。灵感来自 SIFT 描述子和光流估计，但完全是端到端可学习的。

3. **纹理抑制操作（Texture Suppression）**:

    - 功能：在训练时随机屏蔽高纹理区域，鼓励模型学习纹理不变的步态特征
    - 核心思路：发现静态 Gait Feature Field 中方向向量的大小 $\|G^{\text{Static}}_{\langle i,j \rangle}\|_2$ 能反映纹理强度。训练时以概率 $p$ 将大于阈值 $m=0.5$ 的像素置零，促使模型不依赖纹理信息做识别。
    - 设计动机：换装场景下纹理信息是最大干扰源。这个操作相当于告诉模型"纹理是不可靠的"，迫使其聚焦于体型和动作等稳定特征。

### 损失函数 / 训练策略

- 使用 triplet loss + cross-entropy loss 的标准组合训练 GaitBase
- SGD 优化器，初始学习率 0.1，权重衰减 0.0005
- 采用有序采样策略，每个训练步骤处理 20 帧
- CCPG 数据集上训练 60k 步，batch size (8, 4)

## 实验关键数据

### 主实验

| 方法 | 输入 | CCPG-CL (换装) | CCPG-Mean | 协议 |
|------|------|----------------|-----------|------|
| GaitBase | Sils | 71.6 | 75.5 | Gait |
| DeepGaitV2 | Sils | 78.6 | 83.3 | Gait |
| BigGait | RGB | 82.6 | 87.2 | Gait |
| SkeletonGait++ | Sils+Skeleton | 79.1 | 83.7 | Gait |
| MultiGait++ | Sils+Parsing+Flow | 83.9 | 87.6 | Gait |
| **DenoisingGait** | **RGB+Sils** | **84.0** | **89.5** | **Gait** |
| **DenoisingGait** | **RGB+Sils** | **91.8** | **95.7** | **ReID** |

### 消融实验

| 配置 | CCPG-CL | 说明 |
|------|---------|------|
| 扩散基线 (无 ϵθ) | ~78.7 | 仅 VAE 编码+GaitBase |
| 扩散基线 (t=700) | ~84.0 | 加入扩散去噪，+5.3% |
| + Feature Matching (静态) | 提升 | 加入几何驱动去噪 |
| + Feature Matching (动态) | 进一步提升 | 加入运动场信息 |
| + Texture Suppression | 最终 | 纹理抑制进一步增强鲁棒性 |

### 关键发现

- timestep $t$ 的选择至关重要：$t=700$ 是 CCPG 上的最优点，过大过小都会降低性能，验证了扩散模型的多粒度特性
- 静态 Gait Feature Field 自动避开纹理丰富区域（如衣服图案），聚焦于身体轮廓和关节结构
- 动态 Gait Feature Field 的激活焦点集中在运动的肢体部位，与步态的运动学特征高度一致
- 跨域评估（在一个数据集训练，另一个测试）同样表现出色，说明扩散特征的泛化能力强

## 亮点与洞察

- **扩散模型 timestep 的语义含义应用于判别任务**：发现 timestep 可以作为"信息粒度控制器"，这个洞察可以迁移到其他需要多粒度特征的判别任务（如行人重识别、细粒度分类）
- **Gait Feature Field 的设计**：将多通道特征压缩为二维方向向量的思路既优雅又有效——它天然过滤了高维的纹理信息，只保留了结构和运动的方向性信息。这个 Feature Matching 模块可以作为通用的特征去噪工具
- **纹理强度与方向向量大小的关联发现**：$\|G^{\text{Static}}\|_2$ 能反映纹理强度，这个发现为设计纹理抑制操作提供了自然的度量

## 局限与展望

- 依赖预训练 SD 1.5 模型做特征提取，推理计算量较大（每帧需要一次扩散前向），不适合实时场景
- timestep $t=700$ 是在 CCPG 上调参得到的，换到其他数据集可能需要重新寻找
- 仅测试了上半身换装场景，对于更极端的外观变化（如雨衣、头盔等）的鲁棒性有待验证
- 轮廓 mask 的准确性会影响背景去除效果，在遮挡严重时可能降低性能

## 相关工作与启发

- **vs BigGait**: BigGait 也是 RGB 端到端方法，通过特征平滑做软去噪。DenoisingGait 的扩散+几何双驱动去噪更彻底，CL 场景提升 1.4%
- **vs SkeletonGait++**: 骨架+轮廓的双模态融合很有效（83.7%），但受限于预提取表示的信息损失。DenoisingGait 直接从 RGB 提特征，信息更丰富
- **vs MultiGait++**: 使用轮廓+Parsing+光流三种表示的融合在 Gait 协议上接近 DenoisingGait（87.6 vs 89.5），但 DenoisingGait 的方法更简洁，且在 ReID 协议上大幅领先

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将扩散模型用于步态识别，Gait Feature Field 表示新颖
- 实验充分度: ⭐⭐⭐⭐ 三个数据集、跨域评估、充分的消融和可视化分析
- 写作质量: ⭐⭐⭐⭐ 动机清晰，从扩散模型到 Feature Matching 的推导流畅
- 价值: ⭐⭐⭐⭐ 为步态识别提供了全新范式，扩散特征的多粒度控制思路有广泛启发

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Walking the Schrödinger Bridge: A Direct Trajectory for Text-to-3D Generation](../../NeurIPS2025/3d_vision/walking_the_schrödinger_bridge_a_direct_trajectory_for_text-to-3d_generation.md)
- [\[ICCV 2025\] Hierarchical Material Recognition from Local Appearance](../../ICCV2025/3d_vision/hierarchical_material_recognition_from_local_appearance.md)
- [\[CVPR 2025\] DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos](depthcrafter_generating_consistent_long_depth_sequences_for_open-world_videos.md)
- [\[CVPR 2025\] Recovering Dynamic 3D Sketches from Videos](recovering_dynamic_3d_sketches_from_videos.md)
- [\[CVPR 2025\] GenFusion: Closing the Loop between Reconstruction and Generation via Videos](genfusion_closing_the_loop_between_reconstruction_and_generation_via_videos.md)

<!-- RELATED:END -->
