---
title: >-
  [论文解读] SpatialDreamer: Self-supervised Stereo Video Synthesis from Monocular Input
description: >-
  [CVPR 2025][3D视觉][立体视频合成] 提出 SpatialDreamer，一种基于视频扩散模型的自监督立体视频合成框架：通过深度引导的视频数据生成模块 (DVG) 解决立体视频训练数据不足问题…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "立体视频合成"
  - "自监督学习"
  - "视频扩散模型"
  - "时空一致性"
  - "新视角合成"
---

# SpatialDreamer: Self-supervised Stereo Video Synthesis from Monocular Input

**会议**: CVPR 2025  
**arXiv**: [2411.11934](https://arxiv.org/abs/2411.11934)  
**代码**: 无  
**领域**: 视频生成  
**关键词**: 立体视频合成, 自监督学习, 视频扩散模型, 时空一致性, 新视角合成

## 一句话总结

提出 SpatialDreamer，一种基于视频扩散模型的自监督立体视频合成框架：通过深度引导的视频数据生成模块 (DVG) 解决立体视频训练数据不足问题，通过 RefinerNet 框架和一致性控制模块（立体偏差强度 + 时序交互学习 TIL）确保生成立体视频的几何与时间一致性，性能超越 Apple Vision Pro 3D 转换器。

## 研究背景与动机

立体视频合成从单目输入生成目标视角视频，在 3D 电影制作、Apple Vision Pro 等 VR 设备中有广泛应用。该任务面临两个核心挑战：(1) **缺乏高质量立体视频对用于训练**——传统方法需要双摄像头采集，成本高昂；(2) **时空一致性难以保持**——生成的视频帧间容易出现抖动和闪烁。

现有方法主要将单图新视角合成 (NVS) 技术直接应用到视频上，但这些方法难以有效表示动态场景。层级方法（如 3D-photography、MPI）存在深度离散化伪影；NeRF/3DGS 方法在稀疏视角下质量有限；即使 Apple Vision Pro 的 3D 照片转换器在处理视频时也会产生内容闪烁和不一致。

本文的核心思路是：**利用视频扩散模型 (SVD) 的强大时空建模能力，通过自监督框架同时解决数据不足和时空不一致问题**。

## 方法详解

### 整体框架

SpatialDreamer 基于 Stable Video Diffusion (SVD) 构建，包含四个核心模块：(1) DVG 模块通过前向-后向渲染生成配对训练视频；(2) RefinerNet 提取参考视角的空间特征并注入去噪 U-Net；(3) 时序交互学习模块 (TIL) 融合长时帧的特征增强时间一致性；(4) 立体偏差强度指标控制 3D 效果强弱。输入为单目视频和立体位姿，输出为立体视频对。

### 关键设计

**1. Depth-based Video Generation (DVG) — 深度引导视频数据生成**

- **功能**: 自监督地从单目视频生成配对立体视频训练数据，解决数据不足问题
- **核心思路**: 采用前向-后向渲染机制：(a) 估计视频深度；(b) 将参考视角图像渲染到目标视角产生遮挡mask；(c) 用 inpainting 模型填充遮挡区域得到目标视角图像 $x_2$；(d) 将 $x_2$ 反向渲染回原始视角。关键改进是利用光流 $u, v$ 和前向-后向一致性置信度 $C$ 在相邻帧间传播遮挡信息：$m^t(i,j) = 1$ 当 $\sum_{k \in \{t-1,t+1\}} m^k(i+u,j+v) \cdot C(i,j) \geq 1$
- **设计动机**: 直接逐帧渲染会导致帧间遮挡mask不一致引起抖动。通过光流建立帧间像素对应关系并融合相邻帧遮挡信息，使遮挡区域在时间维度上更平滑连贯

**2. RefinerNet + 空间注意力 — 参考视角特征注入**

- **功能**: 学习配对视角间的特征空间分布差异，将参考视角信息注入去噪过程
- **核心思路**: RefinerNet 采用与去噪 U-Net 相同的架构（不含时间层），用 SD2.1 权重初始化。将去噪 U-Net 的特征 $z_t$ 和 RefinerNet 的特征 $z_r$ 沿高度维度拼接后做 self-attention，取前半部分作为输出。这使得 U-Net 能自适应地从 RefinerNet 学习同一特征空间中的关联特征
- **设计动机**: 相比 ControlNet 架构，RefinerNet 与 U-Net 共享相同网络结构和权重初始化，使两者处于同一特征空间，配对视角的特征融合更自然（实验证明 RefinerNet 优于 ControlNet）

**3. Consistency Control Module — 一致性控制**

- **功能**: 同时确保几何一致性和时间一致性
- **核心思路**: 包含两个子模块：(a) **TIL (Temporal Interaction Learning)**: 将参考帧特征 $z_r^t$ 与 $N_r$ 个相邻帧特征融合：$aug_r^t = \lambda \cdot \text{Attn}_{r,r} + (1-\lambda) \cdot \frac{1}{N_r}\sum_{i=1}^{N_r}\text{Attn}_{r,i}$，平衡自注意力与跨帧注意力；(b) **Stereo Deviation Strength**: 量化两视角间的隐空间差异 $s(z) = |z_0 - z_{ref}|$，作为位置嵌入加入残差块，并用 stereo-aware loss $l_d = \|s(z_0) - s(\hat{z}_0)\|_2^2$ 直接监督
- **设计动机**: 仅靠 RefinerNet 的逐帧空间引导不足以保证时间一致性。TIL 通过全局长时帧信息增强时间连贯性；而立体偏差强度使模型能根据场景深度自适应控制 3D 效果，不同场景即使视角相同也应有不同的立体感

### 损失函数 / 训练策略

- **总损失**: $l = l_\epsilon + \lambda \cdot l_d$
    - $l_\epsilon$: SVD 的 v-prediction MSE 损失
    - $l_d$: 立体感知损失，监督生成视频与真实视频间的立体偏差一致性
- 训练时参考视角和目标视角互换（左→右和右→左都作为训练对）
- DVG 生成的数据具有几何和时间先验，支持高效的自监督训练

## 实验关键数据

### 主实验

RealEstate10K 立体图像质量对比：

| 方法 | SSIM ↑ | PSNR ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| 3D-photography | 0.855 | 23.93 | 0.112 |
| Deep3D | 0.808 | 22.94 | 0.183 |
| MVSplat | 0.863 | 25.89 | 0.132 |
| **SpatialDreamer** | **0.916** | **32.26** | **0.038** |

立体视频质量对比（FVD↓ / $E_{warp}^*$↓）：

| 方法 | FVD ↓ | $E_{warp}^*$ ↓ |
|------|-------|-----------------|
| 3D-photography | 155.0 | 3.418 |
| AVP (Apple Vision Pro) | 99.92 | 3.446 |
| NVS-Solver | 249.1 | 5.842 |
| **SpatialDreamer** | **67.09** | **3.374** |

### 消融实验

各组件贡献（图像级 / 视频级）：

| 组件 | SSIM ↑ | PSNR ↑ | LPIPS ↓ | FVD ↓ |
|------|--------|--------|---------|-------|
| U-Net only | 0.880 | 23.73 | 0.06 | - |
| + ControlNet | 0.855 | 24.04 | 0.183 | - |
| + RefinerNet | 0.895 | 30.20 | 0.043 | - |
| + RefinerNet + SDS | 0.916 | 32.26 | 0.038 | - |
| + RN + SDS (video) | - | - | - | 184.0 |
| + RN + SDS + TIL | - | - | - | 123.5 |
| + RN + SDS + DVG | - | - | - | 85.21 |
| + RN + SDS + TIL + DVG | - | - | - | **67.09** |

### 关键发现

1. **RefinerNet 显著优于 ControlNet**：PSNR 从 24.04 提升至 30.20，说明共享架构的特征融合比控制信号注入更适合立体合成
2. **DVG 是视频质量提升最大的组件**：FVD 从 184.0→85.21（无 TIL），光流引导的遮挡mask融合有效改善时间一致性
3. **超越 Apple Vision Pro 3D 转换器**：FVD 67.09 vs 99.92，且 warp error 也更低，证明方法的工业应用潜力
4. 对不同深度估计方法（DepthAnything、Marigold、MiDaS 等）均鲁棒，表明框架的通用性

## 亮点与洞察

1. **自监督范式避免了配对立体视频数据的获取瓶颈**：DVG 的前向-后向渲染 + 光流refinement 是优雅的数据生成方案
2. **立体偏差强度的引入很有实用价值**：同一视角在不同深度场景中应产生不同的 3D 效果，这个可控性对 VR 应用很重要
3. **超越商业产品 AVP 3D converter** 具有说服力，证明学术方法在工程上的可行性

## 局限与展望

1. 依赖深度图进行渲染，深度估计误差会影响立体质量
2. 模型参数量较大，难以实时运行
3. 未来可探索隐式深度表示替代显式深度图
4. 可拓展到更多 VR/AR 应用场景

## 相关工作与启发

- **SVD (Stable Video Diffusion)**: 提供了强大的视频生成基座模型，SpatialDreamer 在其上添加立体视角控制
- **AdaMPI / SinMPI**: 基于多平面图像的 NVS 方法，存在深度离散化伪影，SpatialDreamer 通过扩散模型避免此问题
- **ControlNet**: 被作为对比架构，RefinerNet 证明了共享特征空间比控制信号注入更适合立体合成
- **DepthCrafter**: 视频深度估计方法，为 DVG 提供更好的时间一致深度图

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 自监督立体视频合成范式新颖，DVG + RefinerNet + 一致性控制的组合设计完整
- **实验充分度**: ⭐⭐⭐⭐ — 与多种方法对比（含商业 AVP），消融全面，且测试了多种深度估计方法
- **写作质量**: ⭐⭐⭐ — 整体清晰但部分公式符号使用不够统一
- **价值**: ⭐⭐⭐⭐ — 对 VR 内容制作有直接应用价值，超越商业产品的表现令人印象深刻

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] RayZer: A Self-supervised Large View Synthesis Model](../../ICCV2025/3d_vision/rayzer_a_self-supervised_large_view_synthesis_model.md)
- [\[CVPR 2025\] Sonata: Self-Supervised Learning of Reliable Point Representations](sonata_self-supervised_learning_of_reliable_point_representations.md)
- [\[CVPR 2025\] Consistency-aware Self-Training for Iterative-based Stereo Matching](consistency-aware_self-training_for_iterative-based_stereo_matching.md)
- [\[CVPR 2025\] Regularizing INR with Diffusion Prior for Self-Supervised 3D Reconstruction of Neutron CT Data](regularizing_inr_with_diffusion_prior_self-supervised_3d_reconstruction_of_neutr.md)
- [\[CVPR 2025\] 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](4dequine_disentangling_motion_and_appearance_for_4d_equine_reconstruction_from_m.md)

</div>

<!-- RELATED:END -->
