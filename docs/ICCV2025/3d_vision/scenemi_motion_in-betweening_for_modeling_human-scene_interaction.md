---
title: >-
  [论文解读] SceneMI: Motion In-betweening for Modeling Human-Scene Interactions
description: >-
  [ICCV 2025][3D视觉][人体-场景交互] 首次正式研究场景感知运动插值（scene-aware motion in-betweening）问题，提出 SceneMI 框架，通过双层场景描述符（全局体素 + 局部 BPS）全面编码场景上下文，利用扩散模型的去噪特性处理含噪关键帧…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "人体-场景交互"
  - "运动插值"
  - "扩散模型"
  - "场景编码"
  - "关键帧动画"
---

# SceneMI: Motion In-betweening for Modeling Human-Scene Interactions

**会议**: ICCV 2025  
**arXiv**: [2503.16289](https://arxiv.org/abs/2503.16289)  
**代码**: [inwoohwang.me/SceneMI](http://inwoohwang.me/SceneMI)  
**领域**: 3D视觉  
**关键词**: 人体-场景交互, 运动插值, 扩散模型, 场景编码, 关键帧动画

## 一句话总结

首次正式研究场景感知运动插值（scene-aware motion in-betweening）问题，提出 SceneMI 框架，通过双层场景描述符（全局体素 + 局部 BPS）全面编码场景上下文，利用扩散模型的去噪特性处理含噪关键帧，在 TRUMANS 上碰撞帧率降低 56.9%，在真实世界 GIMO 上脚部滑动减少 37.5%、抖动减少 56.5%。

## 研究背景与动机

### 问题定义

给定 3D 场景 $\mathcal{G}$ 和稀疏关键姿态集 $\mathbf{s} \in \mathbb{R}^{N \times D}$（关键帧指示器 $\mathbf{m} \in \{0,1\}^N$），合成完整运动序列 $\mathbf{x} = \{x^n\} \in \mathbb{R}^{N \times D}$，同时满足关键帧约束和 3D 场景的环境约束（如避障、物理合理性）。

### 已有方法的不足

**场景感知运动合成方法缺乏可控性**：HUMANISE、TRUMANS 等方法从文本/动作标签生成运动，但用户无法精确控制运动细节（如何时在哪个位置做什么动作）

**运动插值方法忽视场景**：传统运动插值方法（RNN、Transformer 等）假设空旷环境，直接应用于 3D 场景会导致人体穿透物体

**现有场景感知方法局限性强**：少数考虑场景的方法（如目标姿态到达）使用条件 VAE，表达能力和可扩展性受限

**真实世界数据的噪声问题**：真实场景中关键帧可能来自不精确的运动捕捉传感器或视频估计，现有方法假设精确关键帧

### 核心动机

**关键洞察**：运动插值是比完全生成更可控、更实用的任务——关键帧提供了充分的约束来降低任务复杂度。场景感知运动插值支持多种实际应用：(1) 动画师可在 3D 场景中通过关键帧创建角色动画；(2) 自动修复真实世界人-场景交互数据中的运动伪影；(3) 增强从单目视频重建的人-场景交互质量。同时，扩散模型天然具有去噪能力，可类比处理含噪关键帧。

## 方法详解

### 整体框架

从 3D 场景提取全局体素特征 $\mathbf{c}_g$ 和基于关键帧的局部 BPS 特征 $\mathbf{c}_l$。训练时对运动序列中的关键帧进行 imputation（替换），模型整合场景特征、体形特征 $\mathbf{b}$ 和扩散时间步 $t$，合成满足关键帧和场景约束的完整运动序列。

### 关键设计

#### 1. **双层场景编码**

- **功能**：紧凑而全面地编码全局和局部场景上下文
- **核心思路**：

  **全局场景特征**：将整个场景表示为粗糙的占据体素栅格 $\mathbf{c}_g \in \{0,1\}^{d_x \times d_y \times d_z}$（0.1m/体素），以第一帧的根位置和朝向为中心。通过 ViT 编码器处理（$48 \times 24 \times 48$ 输入 → 512 维特征向量）。提供全局空间布局，指导整体运动轨迹规划。

  **关键帧中心局部场景特征**：使用 Basis Point Set (BPS) 方法。在 T-pose SMPL 网格表面通过 FPS 确定 64 个锚点（索引固定）。对每个关键帧，计算场景中最近点到每个锚点的偏移向量 $\mathbf{c}_l^n \in \mathbb{R}^{64 \times 3}$。经 MLP 嵌入后与关键帧特征在对应帧拼接。

- **设计动机**：全局特征捕获"大画面"（房间布局、家具位置），局部特征编码"关键帧处的最近障碍物"。BPS 特征对点序列、网格拓扑和分辨率不敏感，增强了对不同场景源（手工 vs. 扫描）的泛化能力。两者互补：全局引导路径规划，局部防止细粒度碰撞。

#### 2. **场景感知运动插值扩散模型**

- **功能**：利用扩散模型生成满足关键帧和场景约束的运动序列
- **核心思路**：

  **运动表示**：每帧包含全局关节位置 $J \in \mathbb{R}^{22 \times 3}$、6D 根朝向 $\phi \in \mathbb{R}^6$ 和 SMPL 姿态参数 $\psi \in \mathbb{R}^{21 \times 6}$，共 201 维。体形特征 $\mathbf{b} \in \mathbb{R}^7$ 编码代表性关节对距离。

  **训练时 imputation**：随机选择 $k$ 个关键帧（含首尾帧），在噪声样本 $\mathbf{x}_t$ 的关键帧位置替换为干净值：
  $$\mathbf{x}_t' = \mathbf{m} \odot \mathbf{x}_0 + (1-\mathbf{m}) \odot \mathbf{x}_t$$
  局部场景特征仅在关键帧位置可见：$\mathbf{c}_l' = \mathbf{m} \odot \mathbf{c}_l$

  **模型架构**：U-Net + Adaptive Group Normalization + 1D 卷积。AdaGN 动态适应归一化，1D 卷积学习序列运动数据。

  **推理**：每个去噪步进行 imputation，并使用 classifier-free guidance（$\mathbf{w}=2.5$）：
  $$\hat{\mathbf{x}}_0 = \mathbf{w} \cdot \mathcal{D}_\theta(\tilde{\mathbf{x}}_t, t, \mathbf{b}, \mathbf{c}_g) + (1-\mathbf{w}) \cdot \mathcal{D}_\theta(\tilde{\mathbf{x}}_t, t, \mathbf{b}, \emptyset)$$

- **设计动机**：imputation 是在扩散框架中引入关键帧约束最自然的方式——直接将干净信号注入噪声序列的特定位置。U-Net + AdaGN 在全局运动表示上已被验证效果好。

#### 3. **含噪关键帧处理**

- **功能**：利用扩散模型的去噪特性处理不精确的真实世界关键帧
- **核心思路**：

  将扩散/采样时间步分为两个阶段：
  - **阶段 1** $[T, T^*+1]$：用含噪关键帧 $\mathbf{s}^{\text{noisy}}$ 做 imputation，引导整体运动结构
  - **阶段 2** $[T^*, 1]$：停止 imputation，让扩散模型自由去噪整个序列（包括关键帧位置），修正噪声

  $$\mathbf{x}_t' = \begin{cases} \mathbf{m} \odot \mathbf{x}_0^{\text{noisy}} + (1-\mathbf{m}) \odot \mathbf{x}_t, & t \in [T, T^*+1] \\ \mathbf{x}_t, & t \in [T^*, 1] \end{cases}$$

  训练时，从干净运动加入随机噪声（$l \sim \mathcal{U}(0, 1.0)$）生成含噪训练数据，模型学习恢复干净运动。

- **设计动机**：含噪关键帧中的噪声可类比扩散过程中的加性噪声。$T^*=20$（$T=1000$）是最优分界点——较大的 $T^*$ 使运动更平滑但降低关键帧精度，较小的 $T^*$ 保持关键帧精度但去噪不充分。

### 损失函数 / 训练策略

总训练损失：
$$\mathcal{L} = \mathcal{L}_{\text{simple}} + \lambda_{\text{joints}} \mathcal{L}_{\text{joints}} + \lambda_{\text{vel}} \mathcal{L}_{\text{vel}}$$

- $\mathcal{L}_{\text{simple}}$：标准扩散重建损失（预测 $\mathbf{x}_0$）
- $\mathcal{L}_{\text{joints}}$：通过 FK 计算的 3D 关节位置损失
- $\mathcal{L}_{\text{vel}}$：关节速度损失（增强运动平滑度）
- $\lambda_{\text{joints}}=2.0$, $\lambda_{\text{vel}}=10.0$

训练 $T=1000$ 步，单张 RTX 3090 GPU。训练数据：TRUMANS。

## 实验关键数据

### 主实验

**TRUMANS 无噪声关键帧评估**（关键帧间隔 60 帧）：

| 方法 | FID↓ | Foot Skating↓ | Jerk↓ | MJPE Key(m)↓ | MJPE All(m)↓ | 碰撞帧率↓ | 穿透Max(m)↓ |
|------|------|-------------|-------|------------|------------|----------|-----------|
| MDM | 1.422 | 0.316 | 0.972 | 0.568 | 0.576 | 0.317 | 0.112 |
| OmniControl | 0.371 | 0.294 | 0.274 | 0.217 | 0.294 | 0.211 | 0.081 |
| CondMDI | 0.943 | 0.281 | 0.305 | 0.452 | 0.457 | 0.262 | 0.087 |
| **SceneMI** | **0.123** | **0.248** | **0.194** | **0.006** | **0.023** | **0.113** | **0.043** |

**真实世界 GIMO 评估**（关键帧间隔 15 帧）：

| 方法 | Foot Skating↓ | Accel↓ | Jerk↓ | 碰撞帧率↓ | 穿透Max(m)↓ |
|------|-------------|-------|-------|----------|-----------|
| GIMO 原始数据 | 0.261 | 0.347 | 0.573 | 0.057 | 0.048 |
| CondMDI | 0.312 | 0.359 | 0.498 | 0.091 | 0.083 |
| **SceneMI** | **0.163** | **0.165** | **0.249** | **0.060** | **0.047** |

### 消融实验

**场景编码组件消融**（TRUMANS 无噪声）：

| 配置 | FID↓ | MJPE All(m)↓ | 碰撞帧率↓ |
|------|------|------------|----------|
| w/o 场景感知 ($\mathbf{c}_g, \mathbf{c}_l$) | 0.136 | 0.059 | 0.131 |
| w/o 全局特征 $\mathbf{c}_g$ | 0.138 | 0.051 | 0.128 |
| w/o 局部特征 $\mathbf{c}_l$ | 0.125 | 0.036 | 0.119 |
| **完整模型** | **0.123** | **0.023** | **0.113** |

**噪声感知 $T^*$ 值消融**（TRUMANS 含噪声，间隔 3 帧）：

| $T^*$ | FID↓ | Foot Skating↓ | Jerk↓ | MJPE Key(m)↓ | 碰撞帧率↓ |
|-------|------|-------------|-------|------------|----------|
| 0（无噪声感知） | 0.157 | 0.265 | 0.230 | 0.015 | 0.119 |
| 10 | 0.123 | 0.253 | 0.199 | 0.013 | 0.110 |
| **20** | **0.118** | **0.247** | **0.198** | **0.013** | **0.108** |
| 40 | 0.121 | 0.249 | 0.187 | 0.014 | 0.112 |
| 60 | 0.122 | 0.250 | 0.189 | 0.015 | 0.114 |

**GIMO 上的功能分解**：

| 配置 | Foot Skating↓ | Jerk↓ | 碰撞帧率↓ |
|------|-------------|-------|----------|
| w/o 场景感知 | 0.192 | 0.245 | **0.082** |
| w/o 噪声感知 | 0.391 | 0.301 | 0.072 |
| **完整模型** | **0.163** | **0.249** | 0.060 |

### 关键发现

1. **碰撞率大幅降低**：SceneMI 在 TRUMANS 上碰撞帧率仅 0.113（vs. CondMDI 0.262、MDM 0.317）
2. **关键帧对齐精度极高**：MJPE Key 仅 0.006m，比 OmniControl (0.217m) 低两个数量级
3. **GIMO 真实数据修复效果显著**：足滑减少 37.5%（0.261→0.163），抖动减少 56.5%（0.573→0.249）
4. **噪声感知至关重要**：在 GIMO 上，w/o 噪声感知的足滑恶化到 0.391（甚至比原始数据更差），说明直接用噪声关键帧做 imputation 反而有害
5. **场景感知与噪声感知互补**：场景感知主要减少碰撞，噪声感知主要提升运动质量

## 亮点与洞察

1. **问题定义有实际价值**：将 HSI 建模重新定义为运动插值任务，既降低了问题复杂度又增加了实用性（动画制作、数据增强、视频重建）
2. **BPS 局部场景编码巧妙**：固定锚点 + 最近场景点偏移，对场景表示形式不敏感，是泛化到真实世界扫描场景的关键
3. **扩散去噪类比含噪关键帧**：巧妙利用扩散模型的天然去噪能力，将含噪关键帧处理转化为"提前停止 imputation + 让模型自由去噪"
4. **从合成到真实的泛化**：仅在 TRUMANS（手工制作场景 + MoCap 数据）上训练，即可泛化到 GIMO（手机扫描场景 + IMU 数据）——得益于 BPS 和体素的场景无关性
5. **首个单目视频 HSI 重建管线**：组合图像到 3D、深度估计、人体姿态估计和 SceneMI，形成完整管线

## 局限与展望

1. **依赖全姿态关键帧**：当只有部分关节位置可用时灵活性受限
2. **缺少文本条件**：无法通过文本指导运动风格或语义
3. **特征级融合的局限**：人-场景交互主要通过特征拼接实现，模型级融合可能更具表现力
4. **推理速度**：$T=1000$ 步采样较慢，DDIM 等加速策略未被探索
5. **手部交互缺失**：22 个关节不包含手指，无法建模精细的物体交互

## 相关工作与启发

- 与 CondMDI 的区别：CondMDI 依赖运动速度特征作为输入（实际应用中难以获取），SceneMI 仅需关键帧姿态
- 与 SceneDiffuser 的区别：SceneDiffuser 结合扩散+强化学习从起始姿态扩展，不具备关键帧对齐能力
- BPS (Basis Point Set) 的巧妙应用：从人体表面锚点到场景点的偏移，是一种点序列无关的场景-人体相对关系编码

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次正式研究场景感知运动插值，含噪关键帧处理设计巧妙
- **实验充分度**: ⭐⭐⭐⭐⭐ — 无噪声 + 含噪声 + 真实世界 + 视频重建，消融全面
- **写作质量**: ⭐⭐⭐⭐ — 问题动机清晰，实验组织有序
- **价值**: ⭐⭐⭐⭐⭐ — 实际应用价值高，为 HSI 领域提供了实用的新范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Guiding Human-Object Interactions with Rich Geometry and Relations](../../CVPR2025/3d_vision/guiding_human-object_interactions_with_rich_geometry_and_relations.md)
- [\[ICCV 2025\] ATLAS: Decoupling Skeletal and Shape Parameters for Expressive Parametric Human Modeling](atlas_decoupling_skeletal_and_shape_parameters_for_expressive_parametric_human_m.md)
- [\[ICCV 2025\] Global Motion Corresponder for 3D Point-Based Scene Interpolation under Large Motion](global_motion_corresponder_for_3d_point-based_scene_interpolation_under_large_mo.md)
- [\[CVPR 2025\] Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions](../../CVPR2025/3d_vision/reconstructing_in-the-wild_open-vocabulary_human-object_interactions.md)
- [\[ICCV 2025\] HIS-GPT: Towards 3D Human-In-Scene Multimodal Understanding](his-gpt_towards_3d_human-in-scene_multimodal_understanding.md)

</div>

<!-- RELATED:END -->
