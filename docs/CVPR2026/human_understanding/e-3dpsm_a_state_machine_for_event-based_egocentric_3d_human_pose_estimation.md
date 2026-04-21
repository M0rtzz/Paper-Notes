---
title: >-
  [论文解读] E-3DPSM: A State Machine for Event-Based Egocentric 3D Human Pose Estimation
description: >-
  [CVPR 2026][人体理解][事件相机] 提出 E-3DPSM，一种基于事件相机的自我中心 3D 人体姿态状态机，将姿态估计建模为连续时间状态演化过程，通过双向 SSM 时序建模和可学习的卡尔曼式融合模块融合直接预测与增量预测，实现 80Hz 实时推理，MPJPE 降低 19%、时序稳定性提升 2.7 倍。
tags:
  - CVPR 2026
  - 人体理解
  - 事件相机
  - 自我中心姿态估计
  - 状态空间模型
  - 3D人体姿态
  - 时间一致性
---

# E-3DPSM: A State Machine for Event-Based Egocentric 3D Human Pose Estimation

**会议**: CVPR 2026  
**arXiv**: [2604.08543](https://arxiv.org/abs/2604.08543)  
**代码**: [https://4dqv.mpi-inf.mpg.de/E-3DPSM/](https://4dqv.mpi-inf.mpg.de/E-3DPSM/)  
**领域**: Human Understanding  
**关键词**: 事件相机, 自我中心姿态估计, 状态空间模型, 3D人体姿态, 时间一致性

## 一句话总结

提出 E-3DPSM，一种基于事件相机的自我中心 3D 人体姿态状态机，将姿态估计建模为连续时间状态演化过程，通过双向 SSM 时序建模和可学习的卡尔曼式融合模块融合直接预测与增量预测，实现 80Hz 实时推理，MPJPE 降低 19%、时序稳定性提升 2.7 倍。

## 研究背景与动机

自我中心 3D 人体姿态估计是 VR/AR 的核心能力（实时化身控制、健身追踪、远程呈现等），但面临几个关键挑战：

**RGB 相机的固有局限**：低光环境下噪声大、快速头部运动导致运动模糊、高分辨率视频流对可穿戴设备的带宽/功耗压力大

**事件相机的优势**：毫秒级时间分辨率、高动态范围、几乎无运动模糊，天然适合快速运动和自遮挡场景

现有事件相机方法 EventEgo3D/EventEgo3D++ 存在以下问题：

- **架构未充分适配事件流特性**：仅通过帧缓冲区存储前一帧信息，未充分利用事件数据的异步、连续、变化驱动特性
- **依赖 2D 热力图**：引入量化误差
- **需要预测分割掩码**：增加额外的误差来源
- **时间抖动和漂移**：在自遮挡场景下 3D 精度不足

核心洞察有三个：
- 事件天然编码 2D 空间变化，应对应 3D 空间的变化（delta poses）
- 事件的连续性适合建模为连续过程（SSM）
- 可以去掉 2D 热力图和分割掩码等中间监督

## 方法详解

### 整体框架

E-3DPSM 分三阶段处理：

1. **事件流预处理**：原始事件转换为 LNES 帧 $\{\mathbf{L}_t\}_{t=1}^N$（20ms 窗口，192×256×2）
2. **时空姿态编码（SPEM）**：提取时序感知的关节特征
3. **姿态回归模块（PRM）**：预测直接姿态 $\mathbf{P}_t^D$ 和增量姿态 $\mathbf{P}_t^\Delta$，通过可学习融合得到最终 3D 姿态 $\mathbf{P}_t$

### 关键设计

1. **时空姿态编码模块（SPEM）**：

    - **多阶段卷积编码**：4 级层级结构，每级含两个残差块 + 下采样卷积，逐步提取空间特征
    - **可变形注意力（Deformable Attention）**：每个阶段末端加入可变形注意力块，自适应聚焦于姿态关键区域，处理鱼眼镜头带来的畸变和自遮挡
    - **双向 SSM 时序建模**：在第 2 和第 4 阶段插入 S5 层（event-specific），在每个空间位置独立聚合长程时序上下文。训练时双向运行利用完整上下文，推理时可切换为因果模式支持实时部署
    - **关节查询解码器**：16 个可学习关节查询嵌入 $\mathbf{U}=\{\mathbf{u}_1,\ldots,\mathbf{u}_{16}\}$，通过 Transformer Decoder 与编码器最后一级特征交互，输出关节感知特征 $\mathbf{F}_t \in \mathbb{R}^{16 \times 192}$

2. **姿态回归模块（PRM）**：三个组件

    - **直接姿态回归**：MLP 将关节查询特征映射到 $\mathbf{P}_t^D \in \mathbb{R}^{16 \times 3}$，作为全局锚点防止漂移
    - **增量姿态回归**：将当前特征与前一帧姿态嵌入拼接，预测帧间位移 $\mathbf{P}_t^\Delta \in \mathbb{R}^{16 \times 3}$。事件流天然编码变化量，因此增量回归比绝对位置回归更容易学习
    - **可学习卡尔曼融合**：核心创新。维护内部状态 $\mathbf{X}_t$ 和协方差 $\Sigma_t$，通过运动更新步骤（用 delta pose）和测量更新步骤（用 direct pose）自适应融合。$\mathbf{Q}$（过程噪声）和 $\mathbf{R}$（观测噪声）为可学习参数，端到端训练后固定

3. **关键设计动机**：

    - 简单相加 $\mathbf{P}_t = \mathbf{P}_{t-1}^D + \mathbf{P}_t^\Delta$ 会累积漂移
    - 后处理卡尔曼滤波缺乏任务适应性
    - 可学习的噪声协方差让系统自动权衡"信任增量更新多少" vs "信任直接预测多少"

### 损失函数 / 训练策略

多项损失联合监督，权重设置：$\lambda_{3D} = \lambda_\Delta = \lambda_{2D} = 0.01$，$\lambda_{BL} = \lambda_{BA} = 10^{-3}$

$$\mathcal{L}_{total} = \lambda_{3D}\mathcal{L}_{3D} + \lambda_\Delta\mathcal{L}_\Delta + \lambda_{2D}\mathcal{L}_{2D} + \lambda_{BL}\mathcal{L}_{BL} + \lambda_{BA}\mathcal{L}_{BA}$$

- $\mathcal{L}_{3D}$：3D 关节位置 MSE
- $\mathcal{L}_\Delta$：增量姿态与 GT 帧间位移的 MSE
- $\mathcal{L}_{2D}$：2D 投影误差
- $\mathcal{L}_{BL}$：骨骼长度 L1 损失，保持人体比例
- $\mathcal{L}_{BA}$：骨骼方向余弦损失，保持解剖合理性

训练策略：
- **不需要合成数据预训练**（之前方法需要在 EE3D-S 上预训练）
- Adam 优化器，batch size 32
- EE3D-R 上训练 15 epoch（$\eta=10^{-3}$），EE3D-W 上微调 10 epoch（$\eta=10^{-4}$）
- 4 张 A40 GPU 训练 34 小时

## 实验关键数据

### 主实验

| 方法 | EE3D-R MPJPE↓ | EE3D-R PA-MPJPE↓ | EE3D-R $e_{smooth}$↓ | EE3D-W MPJPE↓ | EE3D-W PA-MPJPE↓ | EE3D-W $e_{smooth}$↓ |
|------|-------------|-----------------|---------------------|-------------|-----------------|---------------------|
| EgoPoseFormer | 151.66 | 96.99 | 66.50 | 220.40 | 130.45 | 79.23 |
| EventEgo3D | 110.39 | 84.52 | 27.06 | 195.50 | 108.20 | 45.29 |
| EventEgo3D++ | 103.28 | 77.06 | 22.93 | 172.43 | 98.41 | 40.87 |
| **Ours (Causal)** | **84.45** | **62.64** | **8.40** | **158.86** | **93.46** | **23.57** |
| **Ours (Non-Causal)** | **81.32** | **60.21** | **6.65** | **155.82** | **90.85** | **22.65** |

**关键提升**：MPJPE 降低 ~19%（EE3D-R），$e_{smooth}$ 提升 2.7×（从 22.93 到 8.40）

**遮挡关节专项评估**：

| 方法 | EE3D-R Occ MPJPE↓ | EE3D-R Occ PA-MPJPE↓ |
|------|-------------------|---------------------|
| EventEgo3D++ | 88.43 | 49.53 |
| **Ours (Causal)** | **67.49** | **41.85** |

### 消融实验

**SPEM 模块消融**：

| 配置 | MPJPE↓ | PA-MPJPE↓ | $e_{smooth}$↓ | 说明 |
|------|--------|----------|-------------|------|
| w/o SSM Blocks | 118.53 | 87.48 | 16.94 | 无时序建模，严重退化 |
| Single SSM (Stage 4) | 90.18 | 66.06 | 7.84 | 早期时序很重要 |
| w/o Deformable Attn | 88.27 | 64.98 | 7.71 | 空间自适应有帮助 |

**PRM 模块消融**：

| 配置 | MPJPE↓ | PA-MPJPE↓ | $e_{smooth}$↓ | 说明 |
|------|--------|----------|-------------|------|
| w/o Fusion (简单相加) | 141.22 | 84.17 | 10.14 | 漂移严重 |
| Direct Pose Only | 91.26 | 65.43 | 17.22 | 抖动大 |
| Static Fusion | 88.31 | 65.63 | 9.93 | 不如自适应 |
| **Full model** | **84.45** | **62.64** | **8.40** | 最优 |

### 关键发现

1. **因果模式接近非因果**：训练时非因果、推理时因果仍能取得竞争性能，说明模型泛化能力好
2. **实时性强**：A6000 上 80Hz，3050Ti 上也能 52Hz
3. **不需要合成数据预训练**：比之前方法更简洁
4. **骨骼/方向损失很重要**：保持解剖合理性避免非物理预测
5. 即使对 baseline 添加后处理卡尔曼平滑，仍然不如本文方法，说明改进来自架构设计而非平滑技巧

## 亮点与洞察

1. **事件流特性的深度利用**：将"事件编码变化"与"3D delta pose"自然对应，是非常优雅的模态-任务对齐
2. **SSM 用于事件相机**：首次将 S5 层引入自我中心 3D 姿态估计，SSM 的连续状态演化与事件流的异步特性天然契合
3. **可学习卡尔曼融合**：比简单相加或固定参数滤波都好，让模型自动学习信任哪个信号源
4. **去除中间监督**：不需要 2D 热力图和分割掩码，简化了流水线，也消除了潜在的误差源

## 局限与展望

1. **强遮挡+高动态仍有挑战**：论文承认在这些极端场景下精度仍有提升空间
2. **仅验证了鱼眼自我中心视角**：能否推广到第三人称视角未知
3. **EE3D 数据集较小**：实验室采集的真实数据有限，in-the-wild 数据更有难度
4. **16 关节体的局限**：手部和面部关节未覆盖，完整身体姿态需要更多关节
5. **可探索 Mamba 替代 S5**：更新的 SSM 架构可能带来进一步提升

## 相关工作与启发

- **EventEgo3D/EventEgo3D++**：直接前驱，本文的改进目标
- **S5 / Mamba**：SSM 在事件相机领域的成功应用
- **EgoPoseFormer**：RGB 方法适配事件输入的对比 baseline
- 启发：事件相机的"变化编码"特性在 3D 视觉中可以有更多应用（如场景流、深度估计），delta prediction + fusion 的范式值得推广

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 将姿态估计建模为连续状态机，delta+direct 融合设计优雅
- 实验充分度: ⭐⭐⭐⭐ — 两个 benchmark 全面消融，但数据集偏小
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，公式推导完整，图示直观
- 价值: ⭐⭐⭐⭐ — 为事件相机 3D 视觉提供了新范式，80Hz 实时性有实际应用价值

<!-- RELATED:START -->

## 相关论文

- [EgoPoseFormer v2: Accurate Egocentric Human Motion Estimation for AR/VR](egoposeformer_v2_accurate_egocentric_human_motion_estimation_for_arvr.md)
- [Event-based Head Pose Estimation: Benchmark and Method](../../ECCV2024/human_understanding/event-based_head_pose_estimation_benchmark_and_method.md)
- [3D Hand Pose Estimation in Everyday Egocentric Images](../../ECCV2024/human_understanding/3d_hand_pose_estimation_in_everyday_egocentric_images.md)
- [UniDex: A Robot Foundation Suite for Universal Dexterous Hand Control from Egocentric Human Videos](unidex_a_robot_foundation_suite_for_universal_dexterous_hand_control_from_egocen.md)
- [HiPART: Hierarchical Pose AutoRegressive Transformer for Occluded 3D Human Pose Estimation](../../CVPR2025/human_understanding/hipart_hierarchical_pose_autoregressive_transformer_for_occluded_3d_human_pose_e.md)

<!-- RELATED:END -->
