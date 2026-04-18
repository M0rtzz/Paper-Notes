---
title: >-
  [论文解读] EAG3R: Event-Augmented 3D Geometry Estimation for Dynamic and Extreme-Lighting Scenes
description: >-
  [NeurIPS 2025][3D视觉][事件相机] EAG3R 将事件相机的异步事件流融入 MonST3R 点图重建框架，通过 Retinex 增强模块 + SNR 感知融合机制 + 事件光度一致性损失，在极端低光动态场景下实现鲁棒的深度估计、位姿跟踪和 4D 重建，零样本迁移夜间场景即可大幅超越 RGB-only 方法。
tags:
  - NeurIPS 2025
  - 3D视觉
  - 事件相机
  - 3D几何估计
  - 低光照
  - 点图重建
  - 动态场景重建
---

# EAG3R: Event-Augmented 3D Geometry Estimation for Dynamic and Extreme-Lighting Scenes

**会议**: NeurIPS 2025  
**arXiv**: [2512.00771](https://arxiv.org/abs/2512.00771)  
**代码**: 待确认  
**领域**: 3D视觉  
**关键词**: 事件相机, 3D几何估计, 低光照, 点图重建, 动态场景重建

## 一句话总结

EAG3R 将事件相机的异步事件流融入 MonST3R 点图重建框架，通过 Retinex 增强模块 + SNR 感知融合机制 + 事件光度一致性损失，在极端低光动态场景下实现鲁棒的深度估计、位姿跟踪和 4D 重建，零样本迁移夜间场景即可大幅超越 RGB-only 方法。

## 研究背景与动机

**领域现状**: DUSt3R/MonST3R 利用 Transformer 直接回归稠密 pointmap 实现 pose-free 三维重建，掀起了处理长序列、动态场景等挑战性场景的研究热潮。

**现有痛点**: 在自动驾驶等真实场景中，快速运动和急剧变化的光照导致 RGB 图像出现模糊、过曝、欠曝等退化，现有 RGB-only 方法在这些条件下性能严重下降。

**核心矛盾**: RGB 相机依赖长曝光成像，天然不适合极端光照和快速运动场景；而事件相机具备高时间分辨率和高动态范围，但尚未被集成到现代学习驱动的几何估计流水线中。

**本文目标**: 如何将事件相机数据有效融入 pointmap-based 重建框架，使其在极端低光动态场景中保持鲁棒。

**切入角度**: 在 MonST3R 骨干上添加轻量事件适配器和 SNR 引导的自适应融合，同时用事件流构建全局优化中的光度一致性约束。

**核心 idea**: 通过信噪比感知融合在高SNR区域信任RGB、在低SNR区域信任事件特征，并用事件的亮度变化作为全局优化的额外监督信号。

## 方法详解

### 整体框架

EAG3R 在 MonST3R 的基础上进行两方面增强：(1) 特征提取阶段——Retinex 增强 + 轻量事件适配器 + SNR 感知融合；(2) 全局优化阶段——事件光度一致性损失。输入为低光视频和对应事件流，输出包含每帧深度图、相机位姿和全局动态点云。

### 关键设计

1. **Retinex 图像增强模块**:

    - **功能**: 恢复低光图像的可见性，并生成 SNR 置信图
    - **为什么**: 直接使用退化的低光图像会导致特征提取失败；同时需要一个像素级的"可靠性"指标来指导后续融合
    - **怎么做**: 浅层网络估计照度图 $L_{\text{illum}}^t$，增强图像为 $I_{\text{lu}}^t = I^t \odot L_{\text{illum}}^t$。然后计算 SNR 图：
    $\mathcal{M}_{\text{snr}}^t = \frac{\widetilde{I}_g^t}{|I_g^t - \widetilde{I}_g^t| + \epsilon}$
   其中 $\widetilde{I}_g^t$ 为灰度图的均值滤波结果
    - **区别**: 相比独立的低光增强预处理（如 RetinexFormer），本模块与下游任务联合训练，且额外输出 SNR 图用于融合

2. **轻量事件适配器**:

    - **功能**: 从稀疏事件流中提取高保真特征
    - **为什么**: 事件数据在低光场景下仍能捕捉结构信息，但需要专门的编码器
    - **怎么做**: 使用预训练的 Swin Transformer 作为事件编码器，将事件体素化后提取层级特征 $\{F_{\text{evt},l}^t\}_{l=1}^4$。在每个层级通过交叉注意力与图像特征交互：
    $F'_{\text{evt},l} = \text{CrossAttn}(Q=F_{\text{evt},l}^t, K=F_{\text{img},l}^t, V=F_{\text{img},l}^t)$
    - **区别**: 图像编码器冻结，仅训练事件适配器，避免破坏预训练的图像特征

3. **SNR 感知特征融合**:

    - **功能**: 自适应地结合图像和事件特征
    - **为什么**: 不同区域的图像质量差异很大——亮区 RGB 可靠，暗区事件更可靠
    - **怎么做**: 用归一化 SNR 图加权融合：
    $F_{\text{cat}}^t = (F_{\text{img-final}}^t \odot \hat{\mathcal{M}}_{\text{snr}}^t) \| (F_{\text{evt-final}}'^t \odot (1 - \hat{\mathcal{M}}_{\text{snr}}^t))$
   高 SNR 区域偏向图像特征，低 SNR 区域偏向事件特征
    - **区别**: 相比简单的均匀融合或 attention-based 融合，SNR 引导提供了物理上有意义的先验

4. **事件光度一致性损失**:

    - **功能**: 在全局优化中引入基于事件的时空一致性约束
    - **为什么**: MonST3R 原有的光流约束在低光下不可靠，事件流提供了更稳定的运动信号
    - **怎么做**: 在 Harris 角点处定义显著 patch，比较事件观测的亮度增量 $\Delta L_{\mathcal{P}_m}(u)$ 和从图像梯度+运动场预测的亮度增量 $\Delta \hat{L}_{\mathcal{P}_m}(u; X_{\text{global}})$。归一化后计算 L2 残差：
    $\mathcal{L}_{\text{event}} = \sum_{\mathcal{P}_m} \sum_{u \in \mathcal{P}_m} \left\| \frac{\Delta L_{\mathcal{P}_m}(u)}{\|\Delta L_{\mathcal{P}_m}\|} - \frac{\Delta \hat{L}_{\mathcal{P}_m}(u; X_{\text{global}})}{\|\Delta \hat{L}_{\mathcal{P}_m}\|} \right\|^2$
    - **区别**: 归一化消除了未知的对比度灵敏度阈值 $C$，使损失对不同事件传感器具有不变性

### 损失函数 / 训练策略

联合优化目标：
$$X_{\text{global}}^* = \arg\min \left(\mathcal{L}_{\text{align}} + w_{\text{smooth}}\mathcal{L}_{\text{smooth}} + w_{\text{flow}}\mathcal{L}_{\text{flow}} + w_{\text{event}}\mathcal{L}_{\text{event}}\right)$$

- 训练仅使用 MVSEC 的 outdoor_day2 序列（白天），测试在 outdoor_night1-3（夜间）上进行零样本评估
- 微调 MonST3R 的 ViT-Base 解码器、DPT head、增强网络和事件适配器
- 25 个 epoch，4×RTX 3090 训练约 24 小时

## 实验关键数据

### 主实验——单目深度估计（MVSEC Night1-3）

| 方法 | Night1 Abs Rel↓ | Night1 δ<1.25↑ | Night2 Abs Rel↓ | Night2 δ<1.25↑ | Night3 Abs Rel↓ | Night3 δ<1.25↑ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| DUSt3R | 0.407 | 0.393 | 0.415 | 0.384 | 0.463 | 0.335 |
| MonST3R | 0.370 | 0.373 | 0.309 | 0.469 | 0.317 | 0.453 |
| DUSt3R (LightUp) | 0.425 | 0.351 | 0.462 | 0.347 | 0.525 | 0.293 |
| MonST3R (Finetune) | 0.376 | 0.426 | 0.328 | 0.472 | 0.302 | 0.509 |
| **EAG3R** | **0.353** | **0.491** | **0.307** | **0.518** | **0.288** | **0.533** |

### 主实验——相机位姿估计（MVSEC Night1-3）

| 方法 | Night1 ATE↓ | Night2 ATE↓ | Night3 ATE↓ |
|------|:---:|:---:|:---:|
| DUSt3R | 1.474 | 3.921 | 4.109 |
| MonST3R | 0.559 | 0.626 | 0.733 |
| MonST3R (Finetune) | 0.580 | 0.467 | 0.402 |
| Easi3R_monst3r (Finetune) | 0.540 | 0.448 | 0.394 |
| **EAG3R** | **0.482** | **0.428** | 0.409 |

### 消融实验（Night3 深度估计）

| 方法 | Abs Rel↓ | δ<1.25↑ | RMSE log↓ |
|------|:---:|:---:|:---:|
| MonST3R (Baseline) | 0.317 | 0.453 | 0.418 |
| MonST3R (Finetune) | 0.302 | 0.509 | 0.401 |
| + Event | 0.297 | 0.518 | 0.396 |
| + Event + LightUp | 0.291 | 0.523 | 0.388 |
| + Event + LightUp + SNR Fusion (Full) | **0.288** | **0.533** | **0.371** |

### 关键发现

- **事件流贡献最大**: 仅加事件输入即从 0.302→0.297（Abs Rel），验证了事件信号在低光场景的核心价值
- **LightUp 增强有效但有限**: 独立使用 RetinexFormer 预处理反而可能恶化结果（DUSt3R LightUp 比原始更差），说明增强必须与下游任务联合优化
- **SNR 融合是关键**: 最后一步加 SNR 融合使 RMSE log 从 0.388→0.371，表明自适应权重分配比简单拼接更有效
- **零样本夜间泛化**: 仅在白天数据训练即可在夜间大幅超越所有 baseline，证明事件数据的跨场景泛化能力

## 亮点与洞察

- **首个事件增强的 pointmap-based 重建框架**: 将事件相机引入 DUSt3R/MonST3R 范式，开辟了事件+几何基础模型的新方向
- **SNR 引导融合**: 用信噪比作为物理先验指导多模态融合，比纯学习的注意力机制更可解释且更高效
- **事件光度一致性损失**: 巧妙地利用事件的亮度变化模型作为全局优化的约束，归一化处理使其对传感器参数不敏感
- **零样本夜间泛化**: 仅用白天数据训练就能在夜间显著优于所有方法，这是事件相机高动态范围特性的直接体现

## 局限与展望

- 仅在 MVSEC 单一数据集上验证，该数据集规模较小且场景有限（仅室外驾驶）
- 训练数据中使用的事件数据来自真实事件相机，V2E 合成事件导致梯度爆炸——这限制了在更大数据集上的可扩展性
- 事件适配器使用 Swin Transformer，参数量和计算开销未详细报告
- 未评估在正常光照条件下事件流是否会引入噪声干扰
- 动态重建部分仅有定性结果，缺乏定量指标

## 相关工作与启发

- **DUSt3R/MonST3R**: 本文的骨干架构，pointmap-based pose-free 重建的开创性工作
- **RetinexFormer**: Retinex 增强模块的灵感来源，但本文证明独立预处理不如端到端联合训练
- **EvLight**: 自适应事件-图像特征融合的先驱工作
- **启发**: SNR 感知融合的思想可推广到其他多模态任务（如 RGB-LiDAR、RGB-Thermal 融合），物理先验引导融合权重比纯数据驱动更鲁棒

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将事件相机集成到 pointmap 重建框架，SNR 感知融合有创新
- 实验充分度: ⭐⭐⭐ 三个任务+消融实验完整，但仅单一数据集偏弱
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式推导完整，图示信息量大
- 价值: ⭐⭐⭐⭐ 为极端条件下的 3D 重建提供了实用方案，事件+基础模型方向有潜力
