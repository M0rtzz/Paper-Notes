---
title: >-
  [论文解读] IAAO: Interactive Affordance Learning for Articulated Objects in 3D Environments
description: >-
  [CVPR 2025][articulated objects][3DGS] 提出 IAAO 框架，利用 3DGS 构建层次化语义特征场，结合大基础模型实现铰接物体的交互式 affordance 检测与运动参数恢复，支持复杂室内场景的多部件操控。
tags:
  - CVPR 2025
  - 3D Gaussian Splatting
  - articulated objects
  - affordance
  - scene understanding
  - foundation model
---

# IAAO: Interactive Affordance Learning for Articulated Objects in 3D Environments

**会议**: CVPR 2025  
**arXiv**: [2504.06827](https://arxiv.org/abs/2504.06827)  
**代码**: [https://lulusindazc.github.io/IAAOproject/](https://lulusindazc.github.io/IAAOproject/)  
**领域**: 3d_vision  
**关键词**: articulated objects, affordance, 3D Gaussian Splatting, foundation models, motion recovery, scene understanding

## 一句话总结

构建基于 3DGS 的层次化语义特征场，融合 CLIP/SAM/DINOv2 的语义信息，实现铰接物体的交互式 affordance 预测和跨状态运动参数恢复，支持任意类别、多可动部件的复杂室内场景。

## 研究背景与动机

**领域现状**: 理解铰接物体（如柜门、剪刀等）的形状、姿态和关节运动是机器人、AR/VR 的核心需求。Ditto 和 PARIS 等方法通过两个关节状态的观测构建数字孪生。

**现有痛点**: (1) Ditto 需要特定类别预训练，泛化能力差；(2) PARIS 基于 NeRF 隐式表示，对初始化敏感且不稳定；(3) 现有方法假设静态部分已对齐、相机位姿已知，不适用于真实场景；(4) 缺乏细粒度 affordance 检测能力（如识别门把手这类小型功能元素）。

**核心矛盾**: 智能体需要同时理解物体的 what（语义）、where（定位）和 how（交互方式），但现有方法通常只关注单一方面，且限于简单的两部件物体。

**本文切入角度**: 利用显式 3DGS 表示 + 大基础模型（CLIP、SAM、DINOv2）的语义蒸馏，构建可交互的 3D 语义场。

## 方法详解

### 整体框架

三阶段流程：
1. **语义场景重建**：对两个关节状态分别构建带层次化特征的 3DGS
2. **Affordance 与运动预测**：在 3D Gaussian 原语上直接进行物体/部件级查询，估计全局变换和局部关节参数
3. **场景融合**：基于估计的变换合并两状态的 3DGS 模型，填补遮挡区域

### 关键设计

**1. 视角一致的 Mask 聚类与层次化特征场**
- **功能**: 使用 SAM 生成逐视图的类不可知 mask，通过 mask 图聚类将 2D mask 关联为 3D 一致实例；同时用低维特征场 + 解码器蒸馏 MaskCLIP 和 DINOv2 特征。
- **核心思路**: 构建 mask 图 $\mathcal{G}^t_0 = (\mathcal{V}^t_0, \mathcal{E}^t_0)$，每个节点是 2D mask，通过特征相似性聚类得到跨视角一致的 3D 实例。解码器 $\mathcal{D}$ 用小型 MLP 投影渲染特征到三个分支（instance/part-level CLIP + DINO）。
- **设计动机**: 直接嵌入高维 2D 特征内存开销大；mask 图聚类解决了 SAM 跨视角不一致、过分割等问题。
- **损失函数**: 特征蒸馏损失 $\mathcal{L}_{feat} = \|\mathcal{D}(\hat{F}^t(I^t_i)) - F^t(I^t_i)\|_2^2$ + 标签场的交叉熵损失 $\mathcal{L}_{label}$

**2. 全局-局部运动恢复**
- **功能**: 先用 GeoTransformer 从静态部分的 3D Gaussians 估计粗糙全局对齐 $\xi_g^t = (s_g^t, R_g^t, T_g^t)$，再对每个铰接部件估计局部变换 $\xi_o^t = (R_o^t, T_o^t)$。
- **核心思路**: 通过 DINO 特征计算 3D-to-2D 对应关系：对采样的 Gaussian 点与目标状态的 mask 像素计算特征相似度矩阵 $\alpha_{p \to o}$，经 softmax 得到权重 $\beta_{p \to o}$，加权求和获得对应 2D 像素位置。
- **关键损失**:
    - 匹配损失: $\mathcal{L}_{match} = \|\pi^{t'}_n(p^{t \to t'}) - s^{t'}_{p \to n}(I^{t'}_n)\|_2^2$
    - Mask 特征损失: $\mathcal{L}_{mask}$（粗糙引导变换）
    - RGB 一致性损失: $\mathcal{L}_{rgb}$
- **设计动机**: 3DGS 的点云稀疏且有噪声，直接做 3D-3D 配准困难；通过 3D-to-2D 投影和特征匹配间接建立对应关系更鲁棒。

**3. 功能 Affordance 预测**
- **功能**: 给定任务描述（如"open the drawer"），用 CLIP 文本编码器编码后与特征场计算相似度，定位对应的功能区域（如抽屉把手）。
- **核心思路**: 显式 3DGS 原语允许直接在 3D 空间做特征查询，无需渲染。支持语言、点、mask 等多种提示形式。
- **设计动机**: 不同于 NeRF 需要渲染整张图再提取特征，显式 Gaussian 原语可直接操作，高效实现物体/部件级定位。

### 损失函数

总损失: $\mathcal{L} = \lambda_{cons}(\mathcal{L}_{rgb} + \mathcal{L}_{mask} + \mathcal{L}_{label}) + \lambda_{match}\mathcal{L}_{match}$

## 实验关键数据

### 主实验 — Articulation 参数估计

**PARIS 两部件数据集（10 合成 + 2 真实物体）**:

| 方法 | Axis Ang↓ | Axis Pos↓ | Part Motion↓ |
|---|---|---|---|
| PARIS* | 11.14 | 1.79 | 50.52 |
| DigitalTwinArt | 0.14 | 0.01 | 0.12 |
| **IAAO** | **0.11** | **0.01** | **0.10** |

真实场景:

| 方法 | Axis Ang↓ |
|---|---|
| PARIS* | 16.00 |
| DigitalTwinArt | 10.11 |
| **IAAO** | **8.86** |

### 消融实验

| 设置 | Axis Ang↓ | Axis Pos↓ | Part Motion↓ |
|---|---|---|---|
| w/o semantic association | 0.25 | 0.03 | 0.20 |
| w/o mask loss | 0.18 | 0.02 | 0.15 |
| w/o match loss | 0.32 | 0.05 | 0.28 |
| Full IAAO | **0.11** | **0.01** | **0.10** |

### 关键发现

1. **全面超越 SOTA**: 在所有指标上优于 DigitalTwinArt（当前最强基线），关节轴角误差仅 0.11°。
2. **强泛化能力**: 可处理从未见过的物体类别，不依赖预训练，对多部件物体同样有效。
3. **室内场景扩展**: 在 OmniSim 室内场景中也能工作，支持复杂背景和多个铰接物体。
4. **DINOv2 特征对运动恢复至关重要**: DINO 的密集特征提供了 3D-2D 精细对应关系，而 CLIP 特征用于粗糙的 mask 级关联。

## 亮点与洞察

- 显式 3DGS + 基础模型蒸馏的范式具有通用性，可直接在 3D 原语上做查询而非渲染后处理
- 全局-局部分级变换估计策略巧妙绕过了两状态未对齐的常见难题
- 支持任意类别、任意数量可动部件，远优于需要类别预训练的方法
- Mask 图聚类解决 SAM 跨视角不一致性的思路具有启发性

## 局限与展望

- 目前仅支持旋转/平移两种关节类型，未涵盖更复杂的运动（如柔性变形）
- 依赖两个已知关节状态的多视角图像，获取成本高
- GeoTransformer 的全局对齐在部分遮挡严重时可能失败
- 3DGS 重建质量受 SfM 初始化影响，低纹理区域可能退化
- 尚未讨论推理速度和实时交互能力

## 相关工作与启发

- **PARIS**: 基于 NeRF 的两部件铰接物体重建，不需预训练但对初始化敏感
- **DigitalTwinArt**: 两阶段方法（先重建形状再恢复关节），是最强基线
- **GaussianGrouping / SAGA**: 基于 3DGS 的分割方法，提供了 mask 聚类的基础
- **启发**: 3DGS 的显式点云表示天然适合与基础模型特征融合和交互操控

## 评分

⭐⭐⭐⭐ — 方法设计完整、实验全面，在铰接物体交互理解领域取得 SOTA，但实验规模仍以小型合成场景为主。
