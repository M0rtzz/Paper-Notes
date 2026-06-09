---
title: >-
  [论文解读] Touch2Shape: Touch-Conditioned 3D Diffusion for Shape Exploration and Reconstruction
description: >-
  [CVPR 2025][3D视觉][触觉感知] 提出 Touch2Shape，利用触觉条件扩散模型在低维隐空间中生成紧凑的形状表示，结合强化学习训练触摸探索策略，实现了基于触觉图像的3D形状主动探索与重建，无需每步生成完整形状即可指导下一次触摸位置。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "触觉感知"
  - "3D重建"
  - "扩散模型"
  - "强化学习"
  - "主动探索"
---

# Touch2Shape: Touch-Conditioned 3D Diffusion for Shape Exploration and Reconstruction

**会议**: CVPR 2025  
**arXiv**: [2505.13091](https://arxiv.org/abs/2505.13091)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 触觉感知, 3D重建, 扩散模型, 强化学习, 主动探索

## 一句话总结

提出 Touch2Shape，利用触觉条件扩散模型在低维隐空间中生成紧凑的形状表示，结合强化学习训练触摸探索策略，实现了基于触觉图像的3D形状主动探索与重建，无需每步生成完整形状即可指导下一次触摸位置。

## 研究背景与动机

**领域现状**：3D生成与重建是计算机视觉的核心任务。当前方法主要基于视觉输入（单视图/多视图RGB、深度图像），扩散模型在3D生成中展现了强大能力（如 SDFusion、DiffusionSDF）。但这些方法主要针对给定部分观测的全局形状预测。

**现有痛点**：(1) 基于视觉的方法依赖预定义的部分观测，无法主动探索目标；(2) 视觉方法虽能估计整体形状，但容易忽略局部细节，难以生成复杂形状；(3) 实际场景中遮挡和光照条件会严重影响视觉信息的获取。触觉感知不受这些限制，能获取精确的局部3D接触信息，但目前基于触觉的3D重建方法（如 TouchSDF）缺乏全局形状理解能力。

**核心矛盾**：视觉擅长全局但弱于局部细节且受环境限制，触觉擅长局部细节但缺乏全局信息且需要主动规划。如何将扩散模型的强生成能力与触觉感知的精确局部信息结合，并设计高效的探索策略？

**本文目标**：设计一个统一的框架，(1) 利用触觉条件扩散模型进行形状重建，(2) 利用扩散模型生成的隐表示指导主动触摸探索策略，(3) 仅在最终步骤生成完整形状，避免每步的高计算开销。

**切入角度**：扩散模型可以在低维隐空间中编码紧凑的形状表示，这个表示既可以用于最终的形状解码，也可以作为强化学习策略的输入来预测下一次触摸位置。这样就实现了"生成"和"探索"在隐空间的统一。

**核心 idea**：用触觉条件扩散模型生成低维 latent vector，该向量同时驱动形状重建和探索策略，无需每步生成完整 T-SDF 体积，实现了形状解码与探索的分离。

## 方法详解

### 整体框架

系统分为四个预训练模块和联合使用的推理流程。首先预训练 VQ-VAE 将3D形状（T-SDF volume）编码到低维隐空间并解码。然后预训练 TouchCNN 将触觉图像转化为 touch chart（接触面片的3D坐标）。再用对比学习训练 Contrastive Touch Encoder 将触觉和形状编码到对齐空间。训练阶段，冻结预训练模块，训练触觉条件扩散模型和 Touch Shape Fusion。推理时，机械臂触摸目标物体获取触觉图像，扩散模型基于触觉条件去噪得到 latent vector，策略网络基于该 vector 预测下一次触摸位置。最终一步，用 shape decoder + touch shape fusion 生成完整3D形状。

### 关键设计

1. **触觉条件扩散模型（Touch-conditioned Diffusion）**:

    - 功能：在低维隐空间中基于触觉信息生成形状表示
    - 核心思路：给定预训练 shape encoder 编码的隐向量 $z$，在随机时间步加噪，训练去噪网络 $E_\theta$ 以触觉条件 $C(T_0,...,T_{n-1})$ 为条件去噪。损失函数 $L_{diff}(t,n) = \|E_\theta(z_t, r(t), C(T_0,...,T_{n-1})) - \epsilon_t\|_2$。Touch Embedding 将最多 N 张触觉图像的 chart（$N \times M \times 4$ 张量，包含坐标和触摸状态）通过位置编码和卷积提取特征，生成 N 个 token。支持纯触觉和视觉-触觉两种模式，后者通过 ResNet 提取图像特征 token 并与 touch token 拼接后送入去噪网络。
    - 设计动机：在隐空间操作避免了每步生成 64³ 的 T-SDF volume 的巨大开销；触觉图像天然是局部信息，扩散模型的生成能力可以从局部推断全局。

2. **Touch Shape Fusion 模块**:

    - 功能：利用触觉信息优化扩散模型生成的形状细节
    - 核心思路：将所有历史触摸信息合并成全局 touch shape 并进行体素化，用额外的 voxel encoder 提取多尺度特征。这些特征与 VQ-VAE 解码过程中不同尺度的特征进行融合。融合采用 softmax 加权机制：$M_1(c,k,j,i) = \frac{\lambda \cdot F_3^e(c,k,j,i) \cdot e^{F_1^d(c,k,j,i)}}{\sum_{c'} e^{F_1^d(c',k,j,i)}}$，其中 $\lambda$ 是可学习权重。解码器源自预训练 VQ-VAE，在 fusion 训练时微调。
    - 设计动机：扩散模型在隐空间生成的形状全局一致但可能损失局部细节（因为 latent 低维压缩），而触觉 chart 提供的精确局部3D信息可以直接补充解码阶段的细节。

3. **基于强化学习的探索策略训练**:

    - 功能：学习最优的触摸位置序列以最大化重建质量
    - 核心思路：每一步将扩散模型的去噪 latent vector $z'$ 送入策略网络。策略网络编码初始和当前 latent vector 的差异，结合 50 个候选动作（球面上的位置索引）的 embedding，用全连接层预测每个动作的 Q 值。奖励函数基于扩散损失的变化：$R = H(L_{diff}(t,n-1) - L_{diff}(t,n))$，即如果新的触摸使扩散模型的去噪更准确则给正奖励。用 DQN 训练。关键创新是**不需要每步生成完整形状再算 Chamfer Distance**，只需要比较 latent 空间的扩散损失。
    - 设计动机：之前方法（如 ActiveVT）需要每步生成完整 mesh 并计算 CD 作为奖励，计算成本极高。本文在 latent 空间定义奖励，实现了形状解码与探索策略的解耦。

### 损失函数 / 训练策略

训练分三阶段：(1) VQ-VAE 预训练（在 ABC+ShapeNet 上）；(2) 扩散模型训练（100万迭代，lr=1e-5，batch=12）+ Touch Shape Fusion 训练（25万迭代，lr=1e-4，batch=8），两者可并行；(3) 策略训练（200 epoch，lr=3e-4，batch=16）。全部在 RTX 4090 上训练约一周。对比学习使用 MoCo，触觉特征为 query、形状特征为 key。

## 实验关键数据

### 主实验

| 数据集 | 模式 | 方法 | Grasp #0 | Grasp #1 |
|--------|------|------|----------|----------|
| ABC (CD↓) | 纯触觉 T | VTRecon | 25.586 | 9.016 |
| ABC (CD↓) | 纯触觉 T | ActiveVT | 24.864 | 8.220 |
| ABC (CD↓) | 纯触觉 T | **Touch2Shape** | **40.283** | **6.794** |
| ABC (CD↓) | 视觉+触觉 T+V | VTRecon | 2.653 | 2.637 |
| ABC (CD↓) | 视觉+触觉 T+V | ActiveVT | 2.538 | 2.486 |
| ABC (CD↓) | 视觉+触觉 T+V | **Touch2Shape** | **1.475** | **1.406** |

| 数据集 | 方法 | 1 touch | 10 touches | 20 touches |
|--------|------|---------|------------|------------|
| ShapeNet (EMD↓) | TouchSDF | 0.136 | 0.112 | 0.081 |
| ShapeNet (EMD↓) | Ours (T) | 0.124 | 0.056 | 0.053 |
| ShapeNet (EMD↓) | Ours (T+V) | **0.048** | **0.046** | **0.042** |

### 消融实验

| 模式 | 对比学习CL | Fusion | CD↓ |
|------|-----------|--------|------|
| 纯触觉 T | ✗ | ✗ | 4.430 |
| 纯触觉 T | ✓ | ✗ | 3.298 |
| 纯触觉 T | ✓ | ✓ | 3.134 |
| 纯视觉 V | ✗ | ✗ | 2.242 |
| 纯视觉 V | ✓ | ✗ | 2.068 |
| 视觉+触觉 T+V | ✓ | ✓ | **1.304** |

### 关键发现

- **视觉-触觉融合效果显著**：T+V 模式下 CD 仅 1.304，远优于纯视觉（2.068）或纯触觉（3.134），验证了两种模态的互补性
- **对比触觉编码器贡献最大**：纯触觉模式下加入 CL 后 CD 从 4.430 降到 3.298（提升 25.6%），说明触觉和形状的对齐嵌入至关重要
- **探索策略有效**：RL 策略使 5 次抓取后 CD 降到初始的 6.63%（仅 T 模式），优于随机策略的 8.14% 和均匀策略的 7.44%
- **Touch Shape Fusion 持续改善**：从 3.298 降到 3.134（CD），用精确的触觉几何信息修正解码器输出的局部细节
- 初始触摸时 Touch2Shape CD 较高（40.283 vs ActiveVT 24.864），因为扩散模型在信息极少时生成的全局形状偏差大，但随着触摸增加迅速超越

## 亮点与洞察

- **隐空间统一探索与重建**：将形状重建和探索策略都统一在扩散模型的 latent space 中，避免了每步生成完整 3D 形状的高昂开销。这是一个优雅的设计,把扩散模型从"生成器"变成了"感知-规划"的核心引擎。
- **基于扩散损失的奖励设计**：用 $L_{diff}$ 的变化作为 RL 奖励，而非每步计算 CD。这不仅减少计算，还利用了扩散模型的不确定性估计——如果一次触摸让模型更"确定"目标形状，说明这次触摸信息量大。
- **对比学习桥接触觉和形状模态**：用 MoCo 将触觉 chart 和形状 latent 对齐在同一空间，使得扩散模型可以直接利用触觉条件生成与真实形状接近的 latent。

## 局限与展望

- 仅在仿真环境中验证，未迁移到真实机器人平台，sim-to-real gap 可能影响实用性
- VQ-VAE 的 64³ 分辨率限制了重建精度，更高分辨率会带来更大的计算开销
- 探索策略的动作空间只有 50 个球面位置，对于复杂形状（如有孔洞的物体）可能不够细粒度
- 只处理单物体重建，扩展到场景级触觉探索是更有挑战的方向
- 未来可以结合神经渲染技术，利用主动触摸感知合成多视角视觉图像

## 相关工作与启发

- **vs ActiveVT**: ActiveVT 需要每步生成完整 mesh + 编码 latent + 计算 CD 作为策略输入和奖励，而 Touch2Shape 仅在 latent 空间操作，只在最终步解码形状。在 T+V 模式下 CD 1.406 vs 2.486 大幅领先。
- **vs TouchSDF**: TouchSDF 将触觉图像映射到局部 SDF 并用隐式神经函数拼接，缺乏全局形状生成的能力。Touch2Shape 利用扩散模型的全局生成能力弥补了这个不足，在 20 次触摸时 EMD 0.053 vs 0.081。
- **vs SDFusion**: SDFusion 也使用 latent 扩散做3D生成，但基于视觉/文本条件。Touch2Shape 将触觉信号引入作为新的条件模态，拓展了3D扩散模型的应用场景。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将扩散模型用于触觉条件3D形状探索，latent空间统一探索与重建的思路非常新颖
- 实验充分度: ⭐⭐⭐⭐ ABC 和 ShapeNet 两个数据集、多种模态设置、探索策略对比、消融实验完整
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图表直观，但部分公式排版略显杂乱
- 价值: ⭐⭐⭐⭐ 为触觉感知+3D重建交叉领域提供了新范式，对机器人主动感知有启发意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Denoising Functional Maps: Diffusion Models for Shape Correspondence](denoising_functional_maps_diffusion_models_for_shape_correspondence.md)
- [\[CVPR 2025\] 3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning](3d-mem_3d_scene_memory_for_embodied_exploration_and_reasoning.md)
- [\[CVPR 2025\] A Lightweight UDF Learning Framework for 3D Reconstruction Based on Local Shape Functions](a_lightweight_udf_learning_framework_for_3d_reconstruction_based_on_local_shape_.md)
- [\[CVPR 2025\] DualPM: Dual Posed-Canonical Point Maps for 3D Shape and Pose Reconstruction](dualpm_dual_point_maps_shape_pose.md)
- [\[ICCV 2025\] Boost 3D Reconstruction using Diffusion-based Monocular Camera Calibration](../../ICCV2025/3d_vision/boost_3d_reconstruction_using_diffusion-based_monocular_camera_calibration.md)

</div>

<!-- RELATED:END -->
