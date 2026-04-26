---
title: >-
  [论文解读] Video Diffusion Models Excel at Tracking Similar-Looking Objects Without Supervision
description: >-
  [NeurIPS 2025][视频扩散模型] 发现预训练视频扩散模型在高噪声去噪阶段天然学到了适合追踪的运动表示，提出 TED 框架融合运动和外观特征，在追踪外观相似物体时比现有自监督方法提升多达 10 个百分点。
tags:
  - NeurIPS 2025
  - 视频扩散模型
  - 自监督追踪
  - 运动表示
  - 相似外观物体
  - 标签传播
---

# Video Diffusion Models Excel at Tracking Similar-Looking Objects Without Supervision

**会议**: NeurIPS 2025  
**arXiv**: [2512.02339](https://arxiv.org/abs/2512.02339)  
**代码**: 无  
**领域**: 图像生成 / 目标追踪  
**关键词**: 视频扩散模型, 自监督追踪, 运动表示, 相似外观物体, 标签传播

## 一句话总结

发现预训练视频扩散模型在高噪声去噪阶段天然学到了适合追踪的运动表示，提出 TED 框架融合运动和外观特征，在追踪外观相似物体时比现有自监督方法提升多达 10 个百分点。

## 研究背景与动机

1. **领域现状**: 视频标签传播（将首帧标注传播到后续帧）是视频理解的核心任务。有监督方法（如 SAM2）表现优秀但依赖大量标注。自监督方法学习帧表示进行像素级匹配。

2. **现有痛点**: 现有自监督追踪方法过度依赖外观特征——当多个物体外观相似（如两只鹿、两个相同颜色的球）时追踪失败。即使使用时间信号训练（如 cycle-consistency），推理时仍用 2D 图像编码器独立处理每帧，丢失了帧间运动信息。

3. **核心矛盾**: 外观特征是区分不同类物体的捷径，但在区分同类相似物体时反而是障碍。运动是唯一可靠的区分线索，但现有方法未能有效利用。

4. **本文目标**: 在无追踪标注的前提下，实现对外观相似物体的鲁棒追踪。

5. **切入角度**: 预训练视频扩散模型在生成连贯视频时必须隐式建模帧间动力学——直接利用其内部特征作为运动感知表示，无需任何追踪特定训练。

6. **核心 idea**: 视频扩散模型在高噪声去噪阶段自然编码运动信息（因为外观已不可见），这些特征可直接用于区分外观相同但运动不同的物体。

## 方法详解

### 整体框架

TED (Temporal Enhanced Diffusion) 框架：(1) 从视频扩散模型提取运动特征 $\mathbf{R}_m$ → (2) 从图像扩散模型提取外观特征 $\mathbf{R}_a$ → (3) 融合 $\mathbf{R}_f = \text{concat}(\lambda \cdot \mathbf{R}_m, (1-\lambda) \cdot \mathbf{R}_a)$ → (4) 标签传播。

### 关键设计

**1. 运动感知表示提取**

- **功能**: 从预训练视频扩散模型中获取编码帧间运动的特征
- **核心思路**: 给定视频 $X$，加入高强度噪声得到 $\mathbf{X}^\tau$（$\tau$ 大表示更多噪声），送入视频扩散模型的 3D UNet 做一次前向传播，从第三个 block 提取特征作为 $\mathbf{R}_m$。关键：整个视频序列作为整体输入（利用时间注意力和 3D 卷积），而非逐帧独立处理。对长视频使用滑动窗口（窗口大小 $L$，带重叠帧保证运动一致性）。
- **设计动机**: 高噪声级别下外观信息几乎被破坏，模型被迫从全局运动模式中去噪，因此特征编码了丰富的运动信息。实验证实 $\tau=600$-$900$ 时运动特征追踪效果最佳——此时外观特征已完全失效。

**2. 运动+外观特征融合**

- **功能**: 结合运动和外观的互补信息
- **核心思路**: 受 Two-Stream ConvNets 启发，L2 归一化后拼接：$\mathbf{R}_f = \text{concat}(\lambda \cdot \frac{\mathbf{R}_m}{\|\mathbf{R}_m\|_2}, (1-\lambda) \cdot \frac{\mathbf{R}_a}{\|\mathbf{R}_a\|_2})$。外观特征来自图像扩散模型（ADM）逐帧独立计算。$\lambda$ 控制权重：对完全相同物体 $\lambda=1.0$ 最优，对真实世界视频 $\lambda \approx 0.5$ 最优。
- **设计动机**: 纯运动特征对外观完全相同的物体足够，但真实世界中运动+外观共同作用效果更好。

**3. 标签传播追踪**

- **功能**: 基于特征相似度将首帧标签传播到后续帧
- **核心思路**: 对目标帧中每个像素 $i$，在参考帧的空间邻域 $\mathcal{S}(i)$ 内计算特征点积相似度 $A_{tr}(i,j) = \mathbf{R}_f^t(i) \cdot \mathbf{R}_f^r(j)$，保留 top-K 值，加权聚合标签。采用递归传播（首帧 + 前 $m$ 帧作为参考帧）。
- **设计动机**: 标准的视频标签传播协议，使方法与现有方法公平比较。

### 损失函数 / 训练策略

- **无训练**: TED 完全利用预训练模型的特征，不进行任何追踪特定训练
- 默认使用 I2VGen-XL 作为视频扩散模型，ADM 作为图像扩散模型
- 噪声步数 $\tau$ 和特征层 $n_v$ 根据经验选择

## 实验关键数据

### 主实验

| 方法 | 时间训练 | DAVIS $\mathcal{J\&F}_m$ | Youtube-Similar $\mathcal{J\&F}_m$ | Kubric-Similar $\mathcal{J}_m$ |
|------|---------|------------------------|----------------------------------|------------------------------|
| MoCo | ✕ | 65.9 | 48.0 | 51.6 |
| SFC | ✕ | 71.2 | 55.5 | 43.1 |
| DIFT | ✕ | 75.7 | 60.7 | 52.7 |
| CRW | ✓ | 67.6 | 52.0 | 49.7 |
| Spa-then-Temp | ✓ | 74.1 | 59.6 | 44.0 |
| SMTC | ✓ | 73.0 | 57.5 | 64.7 |
| **TED (Ours)** | **✕** | **77.6** | **66.0** | **87.2** |

### 消融实验

| 消融项 | DAVIS | Youtube-Similar | Kubric-Similar |
|--------|-------|----------------|---------------|
| 仅外观 $\mathbf{R}_a$ ($\lambda$=0) | 75.7 | 60.7 | ~50 |
| 仅运动 $\mathbf{R}_m$ ($\lambda$=1) | ~73 | ~63 | **87.2** |
| **融合 $\mathbf{R}_f$ ($\lambda$=0.5)** | **77.6** | **66.0** | ~85 |
| 无重叠帧 ($l$=0) | - | ~62 | - |
| 重叠帧 ($l$=2) | - | **~66** | - |

**不同噪声级别 $\tau$ 的追踪性能**:

| $\tau$ | $\mathbf{R}_a$ 性能 | $\mathbf{R}_m$ Youtube-Similar | 解释 |
|--------|-------------------|-------------------------------|------|
| 200 | 高 | 较低 | 外观信息丰富，运动信息被掩盖 |
| 400 | 中等 | 上升 | 模型开始侧重运动去噪 |
| **600** | 很低 | **峰值** | 外观几乎不可用，**运动主导** |
| 900 | ~0 | 高 (Kubric 峰值) | 几乎纯噪声，仅运动可学 |

### 关键发现

- 在 Kubric-Similar（两个完全相同的球）上，大多数方法 $\mathcal{J}_m$ 约 50%（等于随机猜测），TED 达到 **87.2%** — 质的飞跃
- 在 DAVIS 标准基准上超越所有自监督方法 6 个点以上
- **核心发现**: 视频扩散模型在高噪声 $\tau$ 学到的是运动，低噪声 $\tau$ 学到的是外观——这揭示了扩散模型去噪的层级结构
- 少量重叠帧（$l=2$）即可保证运动一致性
- 方法是模型无关的：换用 Stable Video Diffusion 同样有效

## 亮点与洞察

- **洞察极深**: 揭示了视频扩散模型去噪过程的层级分工——高噪声=运动，低噪声=外观
- 零追踪标注，零额外训练——纯复用预训练生成模型的知识
- PCA 可视化直观展示了运动特征对相似物体的区分能力（图 4）
- 控制实验设计精妙：Kubric-Similar 的两个完全相同球完美隔离了运动 vs 外观特征的作用

## 局限与展望

- 依赖预训练视频扩散模型的窗口长度限制（I2VGen-XL 为 16 帧），长视频需要滑动窗口处理
- 推理速度较慢（需要运行整个 3D UNet 前向传播）
- $\lambda$ 需要根据场景类型调整（全相同物体 vs 真实世界）
- 未与有监督方法（如 SAM2）进行直接对比

## 相关工作与启发

- DIFT 从图像扩散模型提取外观特征用于追踪，TED 将此扩展到视频扩散模型的运动特征
- 与 Track4Gen（需追踪标注训练视频扩散模型）不同，TED 直接利用预训练模型无需额外训练
- 启发：生成模型的内部表示蕴含丰富的物理世界理解，等待被挖掘

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 发现视频扩散模型的"高噪声=运动"洞察非常原创
- **实验充分度**: ⭐⭐⭐⭐⭐ 标准/相似/完全相同三种难度基准，噪声级别消融，模型消融全面
- **写作质量**: ⭐⭐⭐⭐⭐ 动机清晰，控制实验设计精妙，可视化有说服力
- **价值**: ⭐⭐⭐⭐⭐ 既有深刻洞察又有实用价值，为生成模型的表示复用开辟新方向

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] VideoGuide: Improving Video Diffusion Models without Training Through a Teacher's Guide](../../CVPR2025/video_generation/videoguide_improving_video_diffusion_models_without_training_through_a_teachers_.md)
- [\[CVPR 2025\] Exploring Temporally-Aware Features for Point Tracking](../../CVPR2025/video_generation/exploring_temporally-aware_features_for_point_tracking.md)
- [\[CVPR 2025\] Articulated Kinematics Distillation from Video Diffusion Models](../../CVPR2025/video_generation/articulated_kinematics_distillation_from_video_diffusion_models.md)
- [\[CVPR 2025\] InterDyn: Controllable Interactive Dynamics with Video Diffusion Models](../../CVPR2025/video_generation/interdyn_controllable_interactive_dynamics_with_video_diffusion_models.md)
- [\[NeurIPS 2025\] VMDT: Decoding the Trustworthiness of Video Foundation Models](vmdt_decoding_the_trustworthiness_of_video_foundation_models.md)

<!-- RELATED:END -->
