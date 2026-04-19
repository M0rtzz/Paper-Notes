---
title: >-
  [论文解读] MinCD-PnP: Learning 2D-3D Correspondences with Approximate Blind PnP
description: >-
  [ICCV 2025][3D视觉][图像-点云配准] 本文提出 MinCD-PnP，通过三重近似将计算昂贵的 Blind PnP 简化为最小化 2D-3D 关键点间 Chamfer 距离的问题，设计轻量级多任务学习模块 MinCD-Net 集成到现有 I2P 配准框架中，在跨场景和跨数据集设置下显著提升内点率和配准召回率。
tags:
  - ICCV 2025
  - 3D视觉
  - 图像-点云配准
  - 2D-3D 对应关系
  - PnP
  - Chamfer距离
  - 多任务学习
---

# MinCD-PnP: Learning 2D-3D Correspondences with Approximate Blind PnP

**会议**: ICCV 2025  
**arXiv**: [2507.15257](https://arxiv.org/abs/2507.15257)  
**代码**: https://github.com/anpei96/mincd-pnp-demo (有)  
**领域**: 3D视觉  
**关键词**: 图像-点云配准, 2D-3D 对应关系, PnP, Chamfer距离, 多任务学习

## 一句话总结

本文提出 MinCD-PnP，通过三重近似将计算昂贵的 Blind PnP 简化为最小化 2D-3D 关键点间 Chamfer 距离的问题，设计轻量级多任务学习模块 MinCD-Net 集成到现有 I2P 配准框架中，在跨场景和跨数据集设置下显著提升内点率和配准召回率。

## 研究背景与动机

图像-点云（I2P）配准是计算机视觉中的基础任务，目标是建立图像像素与点云中三维点之间的 2D-3D 对应关系，进而通过 PnP 算法估计 6DoF 相机位姿。这一任务广泛应用于视觉定位、导航、SLAM 和三维重建等领域。

**现有方法的局限性**：当前主流的深度学习 I2P 配准方法（如 P2-Net、2D3D-MATR）通过像素到点的特征匹配来建立对应关系，但这种纯特征层面的匹配忽视了 2D-3D 对应关系的投影几何约束（即有效的对应 ⟨q,p⟩ 必须满足 q = π(Tp)），导致难以有效剔除离群点。

**Differential PnP 的问题**：为引入几何约束，近期方法采用可微 PnP（如 EPro-PnP）来监督对应关系学习。然而，可微 PnP 对预测对应关系中的噪声和离群点高度敏感——当网络不可避免地产生错误对应时，PnP 估计的位姿不可靠，反过来阻碍了对应关系学习的有效性。

**Blind PnP 的启示**：Blind PnP 通过同时优化变换矩阵 T 和对应关系矩阵 C 来实现对噪声和离群点的鲁棒性。但其计算复杂度极高（涉及 M×N 的布尔矩阵搜索），无法直接用于深度学习框架中的梯度反传。

**核心切入点**：能否保留 Blind PnP 的鲁棒性，同时降低其计算复杂度，使其适用于端到端的对应关系学习？本文给出了肯定的答案：通过三重近似策略，将 Blind PnP 转化为最小化 Chamfer 距离的可优化问题。

## 方法详解

### 整体框架

MinCD-PnP 由两部分组成：(1) 理论层面的三重近似（Triple Approximation），将 Blind PnP 转化为 MinCD-PnP 问题；(2) 实现层面的 MinCD-Net 模块，通过多任务学习有效求解 MinCD-PnP。MinCD-Net 可即插即用地集成到现有 I2P 配准架构（如 2D3D-MATR）中。

### 关键设计

1. **近似 I：从内点最大化到 Chamfer 距离最小化**

    - 功能：消除 Blind PnP 中的布尔对应矩阵 C，将内点计数最大化问题转化为 Chamfer 距离最小化
    - 核心思路：通过不等式推导，证明 $\max_{\mathbf{T},\mathbf{C}} \kappa(\mathbf{T},\mathbf{C}) \leq \max_{\mathbf{T}} \kappa^{\star}(\mathbf{T})$，其中 $\kappa^{\star}$ 是基于 Chamfer 距离形式的上界。利用这一上界关系，将原始的离散组合优化松弛为连续的 Chamfer 距离最小化：$L_{\text{Chamfer}}(\mathbf{T}|\mathbf{S}_I, \mathbf{S}_P) = \sum_{q \in S_I} \min_{p \in S_P} \|q - \pi(Tp)\|^2 + \sum_{p \in S_P} \min_{q \in S_I} \|q - \pi(Tp)\|^2$
    - 设计动机：M×N 的布尔矩阵搜索复杂度过高，而 Chamfer 距离只需对每个点找最近邻，复杂度大幅降低，且可微分

2. **近似 II：关键点采样降低 Chamfer 距离计算量**

    - 功能：从全量像素和点云中采样代表性关键点，将 Chamfer 距离矩阵从 M×N（~10¹¹）降至 M₀×N₀（~10⁶）
    - 核心思路：从像素集 S_I 和点集 S_P 中分别采样关键点集 K_I 和 K_P，用 $L_{\text{Chamfer}}(\mathbf{T}|\mathbf{K}_I, \mathbf{K}_P)$ 替代原始全量计算。2D/3D 关键点数量约为 10³ 量级，矩阵规模缩小约 10⁵ 倍
    - 设计动机：即使消除了布尔矩阵，全量像素和点的 Chamfer 距离计算仍然不可行，关键点采样在保持代表性的同时极大降低了计算开销

3. **近似 III：2D 关键点引导的 3D 关键点学习**

    - 功能：将联合学习 K_I 和 K_P 简化为仅学习 K_P，K_I 由预训练检测器提供
    - 核心思路：使用 Shi-Tomasi 角点检测器预先提取 2D 关键点 K_I。对每个 2D 关键点 q，通过特征匹配找到最近的 3D 点 $p_q^{\star} = \arg\min_{p} d(\mathbf{f}_q^{2D}, \mathbf{f}_p^{3D})$，并用 IoU 形式的损失 $L_{\text{key}}(q)$ 监督 3D 关键点学习。为过滤低置信度匹配，引入阈值 $s_{th} = e^{-0.4}$
    - 设计动机：联合学习 2D 和 3D 关键点使其足够多的内点是困难的，而成熟的 2D 关键点检测器已能提供良好的图像空间代表性

4. **MinCD-Net 多任务学习模块**

    - 功能：将 MinCD-PnP 的三个近似统一为一个可端到端训练的轻量模块
    - 核心思路：最终优化目标为 $\varphi^{\star} = \arg\min_\varphi \left( L_{\text{corr}} + \lambda_1 \sum_{q \in K_I} L_{\text{key}}(q) + \lambda_2 \min_T L_{\text{Chamfer}}(T|K_I, K_P) \right)$。使用 Point Transformer 编码 2D/3D 关键点特征，生成全局特征后通过 MLP 预测位姿 T（se(3)→SE(3)）
    - 设计动机：三个损失函数协同工作——L_corr 负责特征空间匹配，L_key 确保 3D 关键点逼近 2D 关键点，L_Chamfer 施加全局几何一致性约束

### 损失函数 / 训练策略

总损失函数包含三项：
- **L_corr**：Circle Loss，用于像素-点特征匹配，缓解内外点极端不平衡
- **L_key**：关键点学习损失，基于 IoU 形式，确保学到的 3D 关键点在投影误差阈值 τ 内逼近 2D 关键点
- **L_Chamfer**：Chamfer 距离损失，通过 Point Transformer + MLP 预测位姿后计算投影重投影 Chamfer 距离

训练策略：前 20 epoch 仅训练 L_corr（λ₁=λ₂=0），后续 20 epoch 加入 L_key 和 L_Chamfer（λ₁=0.2, λ₂=0.0001）。阈值 τ 根据相机参数和 RR 阈值自适应设定。

## 实验关键数据

### 主实验

跨场景泛化实验（7-Scenes 数据集，训练 Office→测试其他场景）：

| 方法 | IR (AVG) | RR (AVG) | 提升(IR) | 提升(RR) |
|------|----------|----------|----------|----------|
| P2-Net | 0.393 | 0.510 | - | - |
| MATR | 0.456 | 0.491 | - | - |
| MATR+Diff.PnP | 0.463 | 0.501 | +0.007 | +0.010 |
| MATR+BPnPNet | 0.486 | 0.578 | +0.030 | +0.087 |
| **MATR+MinCD-Net** | **0.568** | **0.647** | **+0.112** | **+0.156** |

跨数据集泛化实验（训练 Kitchen→测试 RGBD-V2）：

| 方法 | IR (AVG) | RR@0.1 (AVG) |
|------|----------|--------------|
| MATR | 0.291 | 0.692 |
| MATR+Diff.PnP | 0.303 | 0.708 |
| MATR+BPnPNet | 0.315 | 0.799 |
| **MATR+MinCD-Net** | **0.371** | **0.870** |

### 消融实验

| 配置 | IR (AVG) | RR (AVG) | 说明 |
|------|----------|----------|------|
| MATR (baseline) | 0.456 | 0.491 | 无PnP约束 |
| + Diff.PnP | 0.463 | 0.501 | 可微PnP，提升有限 |
| + BPnPNet | 0.486 | 0.578 | 加权Blind PnP，改善但梯度受限 |
| + MinCD-Net (仅L_key) | ~0.52 | ~0.58 | 仅关键点学习 |
| + MinCD-Net (完整) | **0.568** | **0.647** | 三损失协同，最佳 |

### 关键发现

- MinCD-Net 相对 Diff.PnP 在跨场景设置下 IR 平均提升约 +0.10，RR 提升约 +0.15，说明 Blind PnP 的鲁棒性优势得到保留
- 在最具挑战性的 Stairs 场景（纹理匮乏），MinCD-Net 的 RR 从 0.226（Diff.PnP）提升到 0.571，提升幅度约 150%
- 跨数据集实验中 MinCD-Net 同样表现最优，证明其泛化能力
- 使用预训练的多数据集模型 MinCD-Net†，RGBD-V2 上 IR 达到 0.581，RR 达到 0.914

## 亮点与洞察

- 将 Blind PnP 这一经典但计算昂贵的鲁棒估计方法通过三级理论近似成功引入深度学习框架，是理论驱动与工程实践的优秀结合
- MinCD-Net 的即插即用设计允许其集成到任意 I2P 配准架构中，而非绑定特定网络
- 关键点引导策略巧妙地利用成熟的 2D 检测器解决了联合学习的难题
- 使用 Chamfer 距离替代精确 PnP 求解，天然对噪声和离群点具有鲁棒性

## 局限与展望

- 2D 关键点依赖预训练检测器（Shi-Tomasi），在纹理极度匮乏场景下可能不够理想，可考虑学习型关键点检测
- 三重近似的理论界是上界而非紧界，最优性有待进一步分析
- 目前仅在室内场景验证，大规模室外场景（如自动驾驶中的 LiDAR-相机配准）的表现有待探索
- Point Transformer 编码关键点的方式可能在关键点数量较大时计算开销增大

## 相关工作与启发

- 2D3D-MATR：当前 I2P 配准的强基线，本文在此基础上集成 MinCD-Net 实现提升
- EPro-PnP：端到端概率 PnP，对噪声有一定鲁棒性但难以处理大量离群点
- BPnPNet：此前唯一将 Blind PnP 用于对应关系学习的工作，但使用 RANSAC 过滤后的梯度信息有限
- 启发：将经典鲁棒估计方法"软化"并嵌入深度学习的思路可推广到其他几何估计任务

## 评分

- 新颖性: ⭐⭐⭐⭐ （三重近似思路新颖，但基于已有理论框架）
- 实验充分度: ⭐⭐⭐⭐⭐ （4个数据集，跨场景+跨数据集，多方法对比）
- 写作质量: ⭐⭐⭐⭐ （理论推导清晰，但部分数学符号较重）
- 价值: ⭐⭐⭐⭐ （I2P 配准的实用改进，即插即用模块设计值得借鉴）

<!-- RELATED:START -->

## 相关论文

- [3×2: 3D Object Part Segmentation by 2D Semantic Correspondences](../../ECCV2024/3d_vision/3x2_3d_object_part_segmentation_by_2d_semantic_correspondenc.md)
- [Learning 3D Object Spatial Relationships from Pre-trained 2D Diffusion Models](learning_3d_object_spatial_relationships_from_pre-trained_2d_diffusion_models.md)
- [Self-supervised Learning of Hybrid Part-aware 3D Representations of 2D Gaussians and Superquadrics](self-supervised_learning_of_hybrid_part-aware_3d_representations_of_2d_gaussians.md)
- [Amodal3R: Amodal 3D Reconstruction from Occluded 2D Images](amodal3r_amodal_3d_reconstruction_from_occluded_2d_images.md)
- [Towards Scalable Spatial Intelligence via 2D-to-3D Data Lifting](towards_scalable_spatial_intelligence_via_2d-to-3d_data_lifting.md)

<!-- RELATED:END -->
