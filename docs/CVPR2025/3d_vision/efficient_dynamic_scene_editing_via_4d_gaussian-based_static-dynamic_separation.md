---
title: >-
  [论文解读] Instruct-4DGS: Efficient Dynamic Scene Editing via 4D Gaussian-based Static-Dynamic Separation
description: >-
  [CVPR 2025][3D视觉][4D高斯] 提出 Instruct-4DGS，利用 4D 高斯 (4DGS) 中静态 3D 高斯和 Hexplane 变形场的内在可分离性，仅编辑静态典范高斯即可实现高效动态场景编辑，再通过 Coherent-IP2P 驱动的分数蒸馏精炼时序对齐以消除运动伪影，将编辑时间缩短一半以上且仅需单 GPU。
tags:
  - CVPR 2025
  - 3D视觉
  - 4D高斯
  - 动态场景编辑
  - 静态-动态分离
  - InstructPix2Pix
  - Score Distillation
  - Hexplane
---

# Instruct-4DGS: Efficient Dynamic Scene Editing via 4D Gaussian-based Static-Dynamic Separation

**会议**: CVPR 2025  
**arXiv**: [2502.02091](https://arxiv.org/abs/2502.02091)  
**代码**: https://hanbyelcho.info/instruct-4dgs/  
**领域**: 3D视觉 / 场景编辑  
**关键词**: 4D高斯, 动态场景编辑, 静态-动态分离, InstructPix2Pix, Score Distillation, Hexplane

## 一句话总结

提出 Instruct-4DGS，利用 4D 高斯 (4DGS) 中静态 3D 高斯和 Hexplane 变形场的内在可分离性，仅编辑静态典范高斯即可实现高效动态场景编辑，再通过 Coherent-IP2P 驱动的分数蒸馏精炼时序对齐以消除运动伪影，将编辑时间缩短一半以上且仅需单 GPU。

## 研究背景与动机

**领域现状**：基于用户指令的 3D 场景编辑已取得显著进展，InstructPix2Pix (IP2P) 能根据文本指令编辑 2D 图像，结合 NeRF/3DGS 实现了空间一致的 3D 编辑。但 4D 动态场景编辑仍然探索不足。

**现有痛点**：现有方法（如 Instruct 4D-to-4D）需要编辑用于动态场景合成的"所有2D图像"（$T \times \mathcal{M}$ 张，T为时间步数，M为相机数），然后通过迭代数据集更新+额外训练循环来更新整个场景，单次编辑需要数小时处理、2块GPU并行。更关键的是，这种方法的编辑时间随时间维度线性增长，无法扩展到长视频。

**核心矛盾**：动态场景编辑的目标是修改外观但保持运动不变，但现有方法将外观和运动耦合在一起统一更新，效率极低且不必要。

**本文目标** 如何高效编辑 4D 动态场景的外观，使编辑时间不随时间步数线性增长？

**切入角度**：4DGS 天然将场景分离为静态典范3D高斯（外观）和 Hexplane 变形场（运动），既然只需编辑外观，为何不只修改静态高斯而保持变形场冻结？

**核心 idea**：利用 4DGS 中静态-动态的可分离性，仅用首帧多视角图像编辑静态高斯实现快速粗编辑，再用分数蒸馏进行无需额外图像编辑的时序精炼。

## 方法详解

### 整体框架

Instruct-4DGS 分三步进行：(1) 用多相机视频数据训练目标 4DGS 场景 $\{\mathcal{G}_{canon}^{opt}, \mathcal{E}^{opt}, \mathcal{D}^{opt}\}$；(2) Stage 1——仅编辑首时间步的多视角图像（用 IP2P），以 L1 RGB loss 监督更新静态典范高斯 $\mathcal{G}_{canon}$ 的 SH 颜色和位置，得到伪编辑场景；(3) Stage 2——用 Coherent-IP2P 驱动的分数蒸馏 (SDS) 精炼静态高斯与变形场的对齐，消除运动伪影。全程仅修改静态高斯参数，变形场保持冻结。

### 关键设计

1. **静态高斯编辑策略（Stage 1）**:

    - 功能：通过仅编辑最小但充分的组件实现高效动态场景外观编辑
    - 核心思路：提取首时间步 $t=0$ 的多视角图像，用 IP2P 根据用户指令编辑这些图像作为监督目标。然后冻结 Hexplane 变形场的所有参数，仅用 L1 RGB loss 优化静态典范高斯 $\mathcal{G}_{canon}$ 的 SH 颜色和位置属性，约 800-1000 次迭代即可完成。关键在于：整个动态场景的外观信息完全由静态高斯承载（变形场只处理位置/尺度/旋转的变化），因此只需编辑静态高斯。
    - 设计动机：对比基线需要编辑 $T \times \mathcal{M}$ 张图像，本方法仅编辑 $\mathcal{M}$ 张（首帧多视角），编辑工作量与时间步数 T 完全解耦，实现了时序维度的可扩展性。

2. **分数蒸馏时序精炼（Stage 2）**:

    - 功能：消除 Stage 1 中编辑静态高斯导致的运动伪影，使编辑后的高斯与原始变形场重新对齐
    - 核心思路：Stage 1 的编辑会引入两类伪影——(a) 高斯位置微移导致从 Hexplane 查询的体素特征偏移，变形失真；(b) 只有首帧可见面的 SH 颜色被更新，后续时间步高斯旋转后暴露未编辑的 SH 值。为解决这些问题，利用 SDS 将 IP2P 的 2D 先验蒸馏到 4D 空间：每次迭代随机采样相机和时间步渲染伪编辑场景图像，通过 $\nabla \mathcal{L}_{SDS} = \mathbb{E}[(\epsilon_\theta(\tilde{I}, c_I, c_T) - \epsilon) \frac{\partial \tilde{I}}{\partial \mathcal{G}_{canon}^{edit}}]$ 更新静态高斯参数，约 800 次迭代。
    - 设计动机：SDS 不需要编辑额外的2D图像，直接利用扩散模型的编辑先验来指导跨时序的一致性优化。由于只是精炼而非从头生成，所需迭代次数少，SDS固有的 Janus 问题也不严重。

3. **Coherent-IP2P 一致性编辑**:

    - 功能：确保多视角/多时步编辑的空间时序一致性，避免SDS蒸馏中不一致引导导致的模糊
    - 核心思路：受 MVDream 和 Tune-a-Video 启发，将 IP2P 中 U-Net 的 2D 自注意力替换为 3D 交叉注意力——同一 batch 中的多张图像共享注意力权重，使编辑引导在图像间保持一致。在 Stage 1 中用于多视角编辑的空间一致性，在 Stage 2 中用于多时步精炼的时序一致性。
    - 设计动机：原始 IP2P 对不同图像独立生成不一致的编辑引导，累积到SDS损失中会导致模糊输出。Coherent-IP2P 通过共享注意力实现图像间的协同编辑，显著提升细节保持和语义一致性。

### 损失函数

- **4DGS 训练损失**：$\mathcal{L}_{4DGS} = |\hat{I}_{M,t} - I_{M,t}| + \mathcal{L}_{TV}$，L1 渲染重建 + 变形场的总变差平滑正则
- **Stage 1 编辑损失**：标准 L1 RGB 损失，监督编辑后的首帧渲染与 IP2P 编辑图像
- **Stage 2 SDS 损失**：使用 Classifier-Free Guidance 的双条件（图像条件 $s_I$ + 文本条件 $s_T$）分数蒸馏

## 实验关键数据

### 主实验表（编辑质量定量对比）

| 方法 | PSNR↑ | SSIM↑ | LPIPS_VGG↓ | CLIP sim↑ | 编辑时间 | GPU数量 |
|:--|:--|:--|:--|:--|:--|:--|
| Instruct 4D-to-4D (avg) | 20.40 | 0.736 | 0.491 | 0.230 | ~2h | 2 GPUs |
| **Instruct-4DGS (avg)** | 19.25 | **0.783** | **0.303** | **0.249** | **~40min** | **1 GPU** |

### 消融实验

| 方法变体 | 用户偏好（1st占比）| 关键问题 |
|:--|:--|:--|
| Fully SDS（无Stage 1） | 最低 | 运动平滑但指令对齐差，低保真度 |
| Refine w/ original IP2P | 较低 | 严重视觉伪影 |
| Ours w/ refine {E,D} | 中等 | 精炼变形场反而引入时序不一致 |
| **Ours w/o refine {E,D}** | **最高** | 最高编辑保真度+时序一致性 |

### 关键发现

- **LPIPS一致性优势**：虽然 PSNR 略低于基线（因为不直接以全部编辑图像为优化目标），但 LPIPS 在所有场景上一致大幅优于基线（0.303 vs 0.491），表明感知质量更优
- **效率优势显著**：编辑时间从 2h（2 GPU）降至 40min（1 GPU），提速 2-3 倍
- **变形场不应被精炼**：消融实验明确表明精炼变形场参数会引入更多时序不一致和运动伪影
- **Coherent-IP2P 至关重要**：使用原始 IP2P 做 SDS 精炼会导致严重的视觉伪影和模糊

## 亮点与洞察

1. **静态-动态分离的深刻利用**：核心洞察是 4DGS 中外观和运动的解耦是编辑效率的关键——只编辑外观载体（静态高斯），不动运动载体（变形场），编辑与时间维度解耦
2. **两阶段互补设计**：Stage 1 提供高保真但时序受限的粗编辑，Stage 2 提供跨时序的一致性精炼但保真度依赖 Stage 1——两阶段缺一不可
3. **反直觉的消融发现**：精炼变形场反而有害——说明对齐问题的根源在静态高斯而非变形场
4. **实用性强**：单GPU、40分钟、文本驱动、多种编辑风格，接近可用的工具形态

## 局限性与可改进方向

1. **依赖 IP2P 能力**：编辑质量受限于 IP2P 的生成质量和指令理解能力
2. **无法直接编辑运动**：只能编辑外观，不能修改物体的运动模式（如改变速度/轨迹）
3. **整体编辑而非局部编辑**：无法选择性地编辑场景中的部分物体（需要额外的分割步骤）
4. **4D表示自身局限**：某些场景的4DGS重建质量不够好时，即使经过时序精炼仍可能存在运动伪影
5. **仅在前向面视角数据上测试**：DyNeRF 和 Technicolor 都是前向面数据集，360°场景的泛化性未验证

## 相关工作与启发

- **Instruct 4D-to-4D**：唯一的前序工作，通过迭代更新所有2D图像实现4D编辑，是本文的对标基线
- **DreamFusion (SDS)**：分数蒸馏的首次提出，本文将其用于编辑精炼而非生成
- **MVDream / Tune-a-Video**：多视角/视频一致性编辑的思想，启发了 Coherent-IP2P 的设计
- **4D Gaussian Splatting (4DGS)**：静态3D高斯+Hexplane变形场的4D表示，本文充分利用了其结构特性
- **启发**：在4D/视频编辑任务中，应尽可能利用表示本身的结构特性来分解问题，而非暴力端到端处理

## 评分

⭐⭐⭐⭐ — 核心思路简洁优雅，静态-动态分离的利用很有洞察力；效率提升显著且实验完整；但方法依赖4DGS的特定结构，且无法编辑运动属性是本质局限。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Ctrl-D: Controllable Dynamic 3D Scene Editing with Personalized 2D Diffusion](ctrl-d_controllable_dynamic_3d_scene_editing_with_personalized_2d_diffusion.md)
- [\[CVPR 2025\] MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds](mosca_dynamic_gaussian_fusion_from_casual_videos_via_4d_motion_scaffolds.md)
- [\[ICCV 2025\] MEGA: Memory-Efficient 4D Gaussian Splatting for Dynamic Scenes](../../ICCV2025/3d_vision/mega_memory-efficient_4d_gaussian_splatting_for_dynamic_scenes.md)
- [\[CVPR 2026\] Catalyst4D: High-Fidelity 3D-to-4D Scene Editing via Dynamic Propagation](../../CVPR2026/3d_vision/catalyst4d_high-fidelity_3d-to-4d_scene_editing_via_dynamic_propagation.md)
- [\[CVPR 2025\] Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera](dyn-hamr_recovering_4d_interacting_hand_motion_from_a_dynamic_camera.md)

<!-- RELATED:END -->
