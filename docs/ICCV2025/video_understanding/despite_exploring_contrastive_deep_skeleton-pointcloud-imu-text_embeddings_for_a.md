---
title: >-
  [论文解读] DeSPITE: Exploring Contrastive Deep Skeleton-PointCloud-IMU-Text Embeddings for Action Recognition
description: >-
  [ICCV 2025][视频理解][多模态对比学习] DeSPITE 提出了一种隐私保护的多模态对比预训练模型，将 LiDAR 点云、骨架姿态、IMU 和文本四种模态对齐到统一嵌入空间，实现了跨模态匹配、检索以及人体活动识别的预训练范式。
tags:
  - ICCV 2025
  - 视频理解
  - 多模态对比学习
  - LiDAR点云
  - IMU
  - 骨架姿态
  - 联合嵌入空间
---

# DeSPITE: Exploring Contrastive Deep Skeleton-PointCloud-IMU-Text Embeddings for Action Recognition

**会议**: ICCV 2025  
**arXiv**: [2506.13897](https://arxiv.org/abs/2506.13897)  
**代码**: 即将发布  
**领域**: 多模态学习 / 人体动作理解  
**关键词**: 多模态对比学习, LiDAR点云, IMU, 骨架姿态, 联合嵌入空间

## 一句话总结

DeSPITE 提出了一种隐私保护的多模态对比预训练模型，将 LiDAR 点云、骨架姿态、IMU 和文本四种模态对齐到统一嵌入空间，实现了跨模态匹配、检索以及人体活动识别的预训练范式。

## 研究背景与动机

现有多模态对比学习方法（如 ImageBind、IMU2CLIP、MotionCLIP）在人体活动理解任务中取得了成功，但它们无一例外地依赖 RGB 视频作为主视觉模态。然而，RGB 摄像头在医疗、监控等隐私敏感场景中的部署面临严重的伦理和法律限制。

LiDAR 作为一种天然的隐私保护传感器，已被证明在人体活动识别（HAR）和人体姿态估计（HPE）中具有优异能力，但 LiDAR 点云在多模态对比学习空间中的位置尚未被探索。具体来说：

**RGB 依赖问题**：所有现有方法都将 RGB 视频/图像作为"锚定模态"来绑定其他模态（如 IMU、骨架），这严重限制了在隐私敏感场景中的适用性

**跨模态匹配空白**：LiDAR 点云 ↔ IMU、LiDAR 点云 ↔ 骨架之间的对应关系此前从未被研究过

**预训练不足**：由于数据量不足，LiDAR 点云 HAR 的通用预训练模型尚不存在，现有预训练主要靠自监督在同一小数据集上进行

本文的核心研究问题是：**如果完全放弃 RGB，只依赖 LiDAR 作为主视觉模态进行多模态对比学习，会发生什么？**

## 方法详解

### 整体框架

DeSPITE 学习一个联合嵌入空间，将点云序列 $X_{pc}$、IMU 序列 $X_{imu}$、骨架姿态序列 $X_{pose}$ 和文本描述 $X_{text}$ 对齐。每种模态由独立的编码器映射到 $\mathbb{R}^e$ 维嵌入向量：

- **点云编码器** $f_{pc}$：PST-Transformer + SimCLR 投影头
- **IMU 编码器** $f_{imu}$：2 层 LSTM
- **骨架编码器** $f_{pose}$：ACTOR 编码器
- **文本编码器**：冻结的 CLIP 文本编码器

### 关键设计

1. **双层对比损失设计**：不同于简单地将所有模态对齐到文本空间，DeSPITE 采用了两层损失策略：

    - **文本对齐损失** $\mathcal{L}_{text}$：将有文本标注的子集中每种传感器模态与 CLIP 文本嵌入对齐（使用布尔掩码 $tm$ 处理无文本标注的样本）
    - **传感器间对齐损失** $\mathcal{L}_M$：直接对齐三种传感器模态的所有两两组合 $(pc, imu)$, $(pc, pose)$, $(imu, pose)$
    - 设计动机：主要目标不是学习文本对齐（已有现成方法），而是利用传感器模态间的自然对应关系实现 LiDAR 点云的跨模态应用

2. **灵活的模态组合训练**：系统地训练了所有可能的模态子集组合（DeSPIE、DeSPE、DePIE 等），通过修改模态集合 $M$ 和配对集 $M^*$ 来分析每种模态对联合嵌入空间的贡献

3. **LIPD-Babel 数据集构建**：通过将 LIPD 数据集与 Babel 文本标注进行时间对齐（将 Babel 从 30FPS 降采样到 10FPS），构建了首个包含点云-IMU-骨架-文本四模态的大规模数据集，分为 v1（匹配/检索评估）和 v2（HAR 评估）两个版本

### 损失函数 / 训练策略

总损失函数为：

$$\mathcal{L}_{total} = \alpha \mathcal{L}_{text} + \beta \mathcal{L}_M$$

其中 $\alpha = \beta = 0.5$。每个方向的对比损失基于 InfoNCE：

$$\mathcal{L}_{a \to b}^i = -\log \frac{\exp(\text{sim}(z_a^i, z_b^i) / \tau)}{\sum_{j=1}^B \exp(\text{sim}(z_a^i, z_b^j) / \tau)}$$

训练配置：512 维嵌入，Adam 优化器（lr=1e-4），batch size 1024，训练 145 个 epoch，24 帧窗口，256 点 FPD 下采样。

## 实验关键数据

### 主实验 — MSR-Action3D HAR 结果

| 方法 | 预训练方式 | Acc@1 (%) |
|------|-----------|-----------|
| PST-Transformer (baseline) | 无 | 93.73 |
| PST-Transformer† (复现) | 无 | 92.33 |
| PSTNet + PointCMP | 单模态自监督 | 93.27 |
| PST-Transformer + MaST-Pre | 单模态自监督 | 94.08 |
| PST-Transformer + M2PSC | 单模态自监督 | 94.84 |
| PST-Transformer + DePITE | **多模态对比 (Ours)** | 95.12 |
| PST-Transformer + DeSPIE | **多模态对比 (Ours)** | **95.47** |
| PST-Transformer + DeSPITE | **多模态对比 (Ours)** | **95.47** |

DeSPITE/DeSPIE 超越所有现有预训练方法，比最强单模态方法 M2PSC 高 0.63%，接近 KAN-HyperpointNet (95.59%) 的性能。

### 消融实验 — HMPEAR 数据集结果

| 方法 | 模态 | Acc(Seg) (%) |
|------|------|-------------|
| PST-Transformer† | PC | 65.94 |
| PEAR-Proj (BestAR) | RGB+PC | 66.0 |
| PST-Transformer + DeSPITE | PC | 69.18 (+3.24) |
| PST-Transformer + DeSPIE | PC | 70.26 (+4.32) |
| PST-Transformer + DePITE | PC | **70.65 (+4.71)** |

DeSPITE 系列预训练方法在 HMPEAR 上实现了新 SOTA，超越了所有先前的纯点云、纯 RGB 和多模态方法，预训练带来了近 4% 的提升。

### 关键发现

1. **文本对匹配/检索有害但对 HAR 有益**：包含文本模态的模型（DeSPITE、DePITE 等）在匹配和时序检索任务中几乎总是弱于不含文本的模型（DeSPIE、DeSPE 等），但在 HAR 微调中表现更好
2. **更多模态 = 更好的 HAR 预训练**：DeSPITE（四模态）、DeSPIE（三模态无文本）和 DePITE（三模态无骨架）在 MSR-Action3D 和 HMPEAR 上始终取得最佳性能
3. **跨模态匹配可行性**：IMU ↔ 骨架匹配最容易，点云 ↔ 骨架次之，IMU ↔ 点云最困难

## 亮点与洞察

- 首次将 LiDAR 点云序列纳入多模态对比学习框架，打开了隐私保护场景下多模态人体活动理解的新方向
- 发现文本嵌入在不同下游任务中的作用截然相反（匹配/检索负面、HAR 正面），揭示了联合嵌入空间的复杂性
- 通过穷举所有模态子集组合进行训练和评估，提供了极其系统的实验分析

## 局限与展望

- LIPD 数据集的真实 LiDAR 数据有限，大部分为合成数据，可能影响泛化性
- 仅使用 24 帧窗口（约 2.4 秒），对长时序活动的建模能力有限
- IMU ↔ 点云匹配性能相对较弱，未来可探索更强的对齐策略

## 相关工作与启发

- 与 ImageBind 的核心区别：不再以 RGB 为中心，而是以 LiDAR 为主视觉模态
- 可以启发面向隐私保护的通用 LiDAR 基础模型的构建
- 为 AR/VR 场景中 IMU 信号的可解释性提供了新工具（通过检索骨架/点云来可视化 IMU 信号含义）

## 评分

- 新颖性: ⭐⭐⭐⭐ （首次探索 LiDAR 为中心的多模态对比学习）
- 实验充分度: ⭐⭐⭐⭐⭐ （穷举模态组合 + 多数据集 + 多任务）
- 写作质量: ⭐⭐⭐⭐ （清晰系统）
- 价值: ⭐⭐⭐⭐ （为隐私保护人体活动理解开辟新方向）

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Adaptive Hyper-Graph Convolution Network for Skeleton-Based Human Action Recognition](adaptive_hyper-graph_convolution_network_for_skeleton-based_human_action_recogni.md)
- [\[ICCV 2025\] Frequency-Semantic Enhanced Variational Autoencoder for Zero-Shot Skeleton-based Action Recognition](frequency-semantic_enhanced_variational_autoencoder_for_zero-shot_skeleton-based.md)
- [\[ICCV 2025\] Adaptive Hyper-Graph Convolution Network for Skeleton-based Human Action Recognition with Virtual Connections](adaptive_hyper_graph_convolution_network_skeleton_action_recognition.md)
- [\[ECCV 2024\] Masked Video and Body-worn IMU Autoencoder for Egocentric Action Recognition](../../ECCV2024/video_understanding/masked_video_and_body-worn_imu_autoencoder_for_egocentric_action_recognition.md)
- [\[ICCV 2025\] Beyond Label Semantics: Language-Guided Action Anatomy for Few-shot Action Recognition](beyond_label_semantics_language-guided_action_anatomy_for_few-shot_action_recogn.md)

<!-- RELATED:END -->
