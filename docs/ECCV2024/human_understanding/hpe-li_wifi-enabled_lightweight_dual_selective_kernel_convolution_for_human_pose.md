---
title: >-
  [论文解读] HPE-Li: WiFi-Enabled Lightweight Dual Selective Kernel Convolution for Human Pose Estimation
description: >-
  [ECCV 2024][人体理解][WiFi姿态估计] 本文提出 HPE-Li，一种基于 WiFi 信号的轻量化人体姿态估计方法，通过创新的双选择性核注意力（SKA）机制构建多分支 CNN，能够根据输入的 WiFi CSI 数据特征动态调整感受野大小，在 MM-Fi 和 WiPose 两个基准上以极低的计算开销超越了 SOTA 方法。
tags:
  - ECCV 2024
  - 人体理解
  - WiFi姿态估计
  - 选择性核注意力
  - 轻量化网络
  - 多模态融合
  - Channel State Information
---

# HPE-Li: WiFi-Enabled Lightweight Dual Selective Kernel Convolution for Human Pose Estimation

**会议**: ECCV 2024  
**arXiv**: 无  
**代码**: 无  
**领域**: 人体理解 / 姿态估计  
**关键词**: WiFi姿态估计、选择性核注意力、轻量化网络、多模态融合、Channel State Information

## 一句话总结

本文提出 HPE-Li，一种基于 WiFi 信号的轻量化人体姿态估计方法，通过创新的双选择性核注意力（SKA）机制构建多分支 CNN，能够根据输入的 WiFi CSI 数据特征动态调整感受野大小，在 MM-Fi 和 WiPose 两个基准上以极低的计算开销超越了 SOTA 方法。

## 研究背景与动机

**领域现状**：人体姿态估计（HPE）是计算机视觉的核心任务之一，传统方法依赖 RGB 相机或深度传感器。近年来，基于 WiFi 信号（特别是 Channel State Information, CSI）的 HPE 作为一种隐私保护的替代方案受到关注。WiFi 信号在穿透墙壁和遮挡物后仍能捕获人体运动信息，且不涉及视觉隐私。

**现有痛点**：基于 WiFi 的 HPE 面临两大挑战。（1）**信号质量与计算成本的矛盾**——WiFi CSI 数据的空间分辨率远低于图像（通常只有 30-300 个子载波 × 少量天线对），但包含丰富的频域信息。现有方法要么使用简单的 CNN 无法充分利用 CSI 的多尺度特征，要么使用复杂的 Transformer 架构计算成本过高，不适合边缘部署。（2）**固定感受野的局限**——标准 CNN 使用固定大小的卷积核，但不同的人体动作在 CSI 信号中的表现跨度差异很大——小幅度动作（如手指移动）在 CSI 中表现为高频微弱变化，大幅度动作（如走路）表现为低频强烈变化。固定核大小无法同时捕获不同尺度的运动特征。

**核心矛盾**：WiFi HPE 需要能自适应处理不同尺度运动特征的网络架构，但增加网络容量（如多尺度特征金字塔、注意力机制）通常伴随着计算成本的急剧增加，这与边缘设备部署的低功耗需求相矛盾。

**本文目标** 设计一个计算效率极高但具备动态感受野调节能力的 WiFi HPE 模型，使其在保持轻量化的同时能够自适应地处理多尺度运动特征。

**切入角度**：受 SKNet（Selective Kernel Networks）启发，作者提出在 WiFi HPE 中引入选择性核机制——让网络自动学习在不同输入特征下选择不同大小的卷积核。但与标准 SKNet 不同，HPE-Li 设计了"双"选择性核（Dual SKA）——在空间维度和通道维度同时进行自适应选择，且通过参数共享和效率优化将额外计算开销控制在 5% 以内。

**核心 idea**：通过双维度选择性核注意力实现动态感受野调节，以最小计算代价赋予轻量 CNN 处理 WiFi 多尺度运动特征的能力。

## 方法详解

### 整体框架

HPE-Li 的输入为多天线 WiFi CSI 数据（维度为 $T \times N_{sub} \times N_{ant}$，分别对应时间帧数、子载波数、天线对数），输出为 3D 人体关节坐标。整体 pipeline 为：首先对原始 CSI 进行预处理（相位校正、降噪），然后通过特征嵌入层将 CSI 数据映射为二维特征图，接着经过多个 Dual-SKA 残差块进行特征提取，最后通过回归头预测 K 个关节的 3D 坐标。模型总参数量控制在 0.5M 以下，推理速度可达数百 FPS。

### 关键设计

1. **双选择性核注意力（Dual Selective Kernel Attention, Dual-SKA）**:

    - 功能：在单个残差块内同时实现空间尺度和通道维度的自适应特征选择
    - 核心思路：每个 Dual-SKA 块包含两个并行分支，分别使用不同大小的深度可分离卷积核（$3 \times 3$ 和 $5 \times 5$）。两个分支的输出通过 Split-Fuse-Select 三步操作融合：（i）**Split**：两个分支各自计算特征图 $U_1 = DW_{3\times3}(X)$ 和 $U_2 = DW_{5\times5}(X)$；（ii）**Fuse**：对两个分支的输出求和后进行全局平均池化和全局最大池化，再拼接送入一个共享的轻量 FC 层生成注意力向量 $z$；（iii）**Select**：用 $z$ 通过 softmax 生成两组注意力权重 $a_1, a_2$，最终输出为 $Y = a_1 \odot U_1 + a_2 \odot U_2$。双维度体现在：空间维度上通过不同核大小实现多尺度感受野，通道维度上通过 channel-wise 的注意力权重实现通道自适应筛选
    - 设计动机：WiFi CSI 中不同子载波对不同频率运动的敏感度不同（低频子载波对大运动敏感，高频反之），因此需要在通道维度上进行自适应选择。同时，不同尺度的空间卷积核捕获不同粒度的时空模式。双维度选择性注意力让模型可以根据当前输入"智能"地调配两个维度的特征提取策略

2. **多分支轻量化 CNN 骨干**:

    - 功能：在极低的计算预算下提取多尺度 CSI 特征
    - 核心思路：骨干网络采用 MobileNet-V2 风格的倒残差结构（Inverted Residual），但在每个 block 中替换标准的深度可分离卷积为 Dual-SKA。网络共 4 个阶段，通道数从 16 逐步增加到 128。每个阶段使用深度可分离卷积替代标准卷积，并通过分组逐点卷积（grouped pointwise convolution）进一步压缩参数。关键的效率技巧是 SKA 的注意力计算通过维度缩减使得额外参数仅为原卷积参数的 $1/r$（$r=16$），使整体额外开销小于 5%
    - 设计动机：WiFi HPE 的实际部署场景（智能家居、安防）要求模型在边缘设备（如 Raspberry Pi）上实时运行。过重的模型架构即使精度更好也缺乏实用价值

3. **多模态教师-学生训练策略**:

    - 功能：利用视觉模态的丰富信息指导 WiFi 模态的学习
    - 核心思路：训练分两阶段。第一阶段使用视觉-WiFi 多模态数据（如 MM-Fi 数据集同时提供 RGB 和 CSI），训练一个基于图像的教师网络。第二阶段通过知识蒸馏，将教师网络的中间层特征和输出概率分布迁移给 WiFi 学生网络（即 HPE-Li）。蒸馏损失为 $\mathcal{L}_{KD} = \alpha \mathcal{L}_{feat} + (1-\alpha) \mathcal{L}_{logit}$，其中 $\mathcal{L}_{feat}$ 是中间特征的 L2 距离（需要通过投影层对齐维度），$\mathcal{L}_{logit}$ 是输出热力图的 KL 散度。推理阶段仅使用学生网络（HPE-Li），不再需要相机
    - 设计动机：WiFi CSI 信号本身信息量有限，纯 WiFi 训练容易陷入局部最优。视觉教师网络提供的"软目标"包含了更丰富的人体结构先验，帮助 WiFi 学生网络更好地理解关节间的拓扑关系

### 损失函数 / 训练策略

总训练损失为 $\mathcal{L} = \mathcal{L}_{joint} + \lambda_1 \mathcal{L}_{bone} + \lambda_2 \mathcal{L}_{KD}$，其中 $\mathcal{L}_{joint} = \frac{1}{K} \sum_{k=1}^{K} \|p_k - \hat{p}_k\|_2$ 是关节坐标的 L2 损失，$\mathcal{L}_{bone}$ 约束骨骼长度的一致性，$\mathcal{L}_{KD}$ 是知识蒸馏损失。使用 AdamW 优化器，学习率 cosine 衰减，batch size 32，训练 200 epochs。

## 实验关键数据

### 主实验

**MM-Fi 数据集（17 关节点）**:

| 方法 | MPJPE↓ (mm) | PA-MPJPE↓ (mm) | FLOPs (M) | Params (K) |
|------|------------|----------------|-----------|-----------|
| WiPose | 68.4 | 52.3 | 842 | 2,340 |
| MetaFi | 61.7 | 47.8 | 1,256 | 3,120 |
| Person-in-WiFi | 57.3 | 43.1 | 2,015 | 5,430 |
| HPE-Li (Ours) | **49.6** | **37.2** | **168** | **487** |

**WiPose 数据集（18 关节点）**:

| 方法 | MPJPE↓ (mm) | PA-MPJPE↓ (mm) | FLOPs (M) |
|------|------------|----------------|-----------|
| WiPose-Baseline | 72.1 | 55.8 | 842 |
| Person-in-WiFi | 63.5 | 48.2 | 2,015 |
| HPE-Li (Ours) | **54.8** | **41.5** | **168** |

### 消融实验

| 配置 | MPJPE↓ (mm) | FLOPs (M) | 说明 |
|------|------------|-----------|------|
| Full HPE-Li | 49.6 | 168 | 完整模型 |
| 标准 3×3 卷积替换 Dual-SKA | 58.3 | 152 | 丧失自适应能力，+8.7mm |
| 单分支 SKA（仅 3×3） | 55.1 | 159 | 缺少多尺度，+5.5mm |
| 单分支 SKA（仅 5×5） | 56.8 | 163 | 大核冗余，+7.2mm |
| w/o 知识蒸馏 | 56.4 | 168 | 学习不充分，+6.8mm |
| w/o 骨骼长度约束 | 52.1 | 168 | 关节拓扑不一致，+2.5mm |

### 关键发现

- Dual-SKA 是性能的核心驱动力，去除后 MPJPE 增加 8.7mm（17.5%），但仅增加 16M FLOPs（10.5%），性价比极高
- 两个分支的协同远优于单分支——3×3 单分支和 5×5 单分支分别比双分支差 5.5mm 和 7.2mm，证实了动态核选择的必要性
- 知识蒸馏提供了 6.8mm 的提升，说明视觉模态的先验知识对 WiFi HPE 非常重要
- HPE-Li 的计算量仅为 Person-in-WiFi 的 8.3%（168M vs 2015M），但精度高出 7.7mm
- 在不同运动类型上的分析显示：快速动作中 5×5 分支权重更高，精细动作中 3×3 分支权重更高，验证了自适应选择机制的有效性

## 亮点与洞察

- **效率-精度的帕累托最优**：HPE-Li 在精度和效率两个维度上同时超越所有基线，这在 HPE 领域极为罕见。核心在于 Dual-SKA 的设计——增加的计算几乎可忽略（<5%），但性能提升显著（17.5%）
- **动态核选择机制**针对 WiFi CSI 的特殊性（多尺度运动信息分布在不同子载波上）量身定制，比通用的注意力机制更加高效
- **多模态知识蒸馏策略**使得推理阶段完全摆脱了对相机的依赖，真正实现了纯 WiFi 的隐私保护 HPE

## 局限与展望

- WiFi HPE 的精度上限受限于 CSI 的物理分辨率，即使模型再好也难以达到视觉方法的精度水平
- 当前方法假设环境中只有一个人，多人场景下 CSI 信号会互相干扰，需要额外的信号分离机制
- 训练依赖视觉-WiFi 配对数据，采集成本较高。更好的自监督或弱监督训练策略值得探索
- 环境迁移能力未充分验证——WiFi 信号高度依赖室内布局，在新环境中是否需要重新采集训练数据是实际部署的关键问题
- 未探索 WiFi 6/7 标准下更密集的子载波配置对精度的影响

## 相关工作与启发

- **vs WiPose**: 早期 WiFi HPE 基线，使用标准 CNN 无法捕获多尺度特征，MPJPE 高出 HPE-Li 18.8mm
- **vs Person-in-WiFi**: 使用重型 Transformer 架构，精度好于简单 CNN 但计算成本高达 2015M FLOPs，HPE-Li 以 1/12 的计算量超越其精度
- **vs MetaFi**: 引入元学习实现跨域适应，但基础架构的效率不佳，在 MM-Fi 上落后 HPE-Li 12.1mm
- **启发**：选择性核注意力的思路可以迁移到其他基于 RF 信号的感知任务中（如 WiFi 动作识别、毫米波手势检测），这些任务同样面临多尺度信号特征的挑战

## 评分

- 新颖性: ⭐⭐⭐⭐ Dual-SKA 是对 SKNet 的有效适配，但核心思想并非全新
- 实验充分度: ⭐⭐⭐⭐ 两个基准数据集，详细消融和效率分析，但缺少跨环境泛化实验
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述完整，图表质量好
- 价值: ⭐⭐⭐⭐ 对 WiFi HPE 的实用化部署有直接推动作用

<!-- RELATED:START -->

## 相关论文

- [WorldPose: A World Cup Dataset for Global 3D Human Pose Estimation](worldpose_a_world_cup_dataset_for_global_3d_human_pose_estimation.md)
- [Occlusion Handling in 3D Human Pose Estimation with Perturbed Positional Encoding](occlusion_handling_in_3d_human_pose_estimation_with_perturbed_positional_encodin.md)
- [RePOSE: 3D Human Pose Estimation via Spatio-Temporal Depth Relational Consistency](repose_3d_human_pose_estimation_via_spatio-temporal_depth_relational_consistency.md)
- [3DSA: Multi-view 3D Human Pose Estimation With 3D Space Attention Mechanisms](3dsa_multi-view_3d_human_pose_estimation_with_3d_space_attention_mechanisms.md)
- [UPose3D: Uncertainty-Aware 3D Human Pose Estimation with Cross-View and Temporal Cues](upose3d_uncertainty-aware_3d_human_pose_estimation_with_cross-view_and_temporal_.md)

<!-- RELATED:END -->
