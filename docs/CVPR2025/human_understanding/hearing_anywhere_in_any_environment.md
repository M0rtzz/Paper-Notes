---
title: >-
  [论文解读] Hearing Anywhere in Any Environment
description: >-
  [CVPR 2025][人体理解][跨房间RIR预测] 提出 xRIR，一个可跨房间泛化的声脉冲响应（RIR）预测统一模型，结合全景深度图的几何特征提取器和少量参考 RIR 的声学编码器，配合新构建的 AcousticRooms 数据集（260 个房间、30 万+ RIR），在已见/未见模拟环境和真实环境中均大幅超越基线方法。
tags:
  - CVPR 2025
  - 人体理解
  - 跨房间RIR预测
  - 空间声学
  - 几何特征提取
  - 参考RIR编码
  - 仿真到真实迁移
---

# Hearing Anywhere in Any Environment

**会议**: CVPR 2025  
**arXiv**: [2504.10746](https://arxiv.org/abs/2504.10746)  
**代码**: 无  
**领域**: 人体理解 / 3D视觉  
**关键词**: 跨房间RIR预测, 空间声学, 几何特征提取, 参考RIR编码, 仿真到真实迁移

## 一句话总结

提出 xRIR，一个可跨房间泛化的声脉冲响应（RIR）预测统一模型，结合全景深度图的几何特征提取器和少量参考 RIR 的声学编码器，配合新构建的 AcousticRooms 数据集（260 个房间、30 万+ RIR），在已见/未见模拟环境和真实环境中均大幅超越基线方法。

## 研究背景与动机

**领域现状**：房间声脉冲响应（RIR）描述了声源到接收器之间声音的反射、吸收和扩散过程，是混合现实中实现沉浸式声学体验的关键。传统方法需要在房间中密集采样数以百计的 RIR 测量。近年来基于隐式神经网络的方法可以将密集 RIR 测量"压缩"到单个模型中，但仅能用于训练时的单个房间。

**现有痛点**：现有深度学习 RIR 预测方法（如 INRAS、NAFs）是为单一房间设计的——每换一个新房间就需要用大量密集 RIR 数据重新训练，无法泛化到几何和材料不同的新环境。这严重限制了在 VR/AR 等需要快速适配多样环境的应用场景中的实用性。

**核心矛盾**：不同房间的几何结构和墙面材料千差万别，而这两个因素共同决定了房间独特的声学特征。单一模型要同时处理几何和材料的巨大差异是极具挑战性的。

**本文目标** (1) 如何从容易获取的视觉表征中提取房间几何信息；(2) 如何用少量 RIR 测量快速捕捉房间的材料声学特性；(3) 需要大规模高保真多房间 RIR 数据集来支撑跨房间泛化预训练。

**切入角度**：几何和材料是互补的信息源——全景深度图编码几何结构，少量参考 RIR 隐式编码材料属性（通过能量衰减和混响模式）。将两者融合可以在不显式建模材料系数的情况下近似完整的声学信息。

**核心 idea**：用全景深度图编码房间几何、用少量参考 RIR 编码声学特性，通过融合加权模块预测目标 RIR，实现单模型跨房间泛化。

## 方法详解

### 整体框架

xRIR 由三个组件构成：(1) 几何特征提取器——包含直达路径模块（编码声源-接收器的直线路径特征）和反射模块（利用全景深度图建模声音经墙壁反射的路径）；(2) 参考 RIR 编码器——用 ResNet-18 从 K 个参考 RIR 的对数幅度谱中提取声学特征；(3) 融合与加权模块——将几何特征和声学特征融合后，通过注意力机制和时间对齐加权矩阵，对参考 RIR 频谱图进行加权组合生成目标 RIR。

### 关键设计

1. **基于全景深度图的反射模块**:

    - 功能：编码声源到墙壁反射再到接收器的路径特征
    - 核心思路：(a) 将接收器处的全景深度图通过等距柱面投影转换为 3D 坐标图 $I_{coord}$，每个像素代表一个墙壁边界点；(b) 对每个声源，计算其与所有边界点的差值图 $I_{s,rf} = P_{rel,s} - I_{coord}$（类似地计算接收器和参考声源的差值图），这些差值图编码了声源→边界点→接收器的反射路径信息；(c) 使用 Vision Transformer（6层、8头、512维）处理差值图的 patch 特征，建模 patch 间的空间依赖关系，最终投影为紧凑的几何特征向量
    - 设计动机：不同于 INRAS 使用固定的弹跳点集合（无法跨房间统一），全景深度图提供了跨房间通用的几何表征；ViT 的全局注意力在此自然地建模了多路径反射之间的空间关系

2. **参考 RIR 编码器**:

    - 功能：从少量 RIR 测量中捕捉房间材料相关的声学特征
    - 核心思路：对 K 个参考 RIR 分别计算 STFT 对数幅度谱 $\mathbf{S}_{ref,k} = \log(\|\text{STFT}(A_{ref,k})\|)$，然后通过 ResNet-18 提取特征，使用最后一层的 mean pooling 特征 $f_a^{(k)}$ 表示每个参考 RIR 的声学特征
    - 设计动机：材料属性难以直接观测，但 RIR 的能量衰减模式和混响特征隐式编码了材料信息，用少量参考 RIR 作为"声学采样"来捕捉这些特性

3. **融合与时间对齐加权模块**:

    - 功能：整合几何和声学特征，生成时间对齐的参考 RIR 加权组合
    - 核心思路：(a) 将每个参考声源的几何特征（直达路径+反射路径+接收器反射）和声学特征拼接为联合表征 $\mathbf{h}_{ref}^{(k)}$；(b) 对目标声源也拼接其几何特征为 $\mathbf{h}_t$，然后计算目标与各参考之间的注意力：$\mathbf{Z} = \text{softmax}(\mathbf{H}_{ref} \cdot \mathbf{h}_t^T / \sqrt{C}) \odot \mathbf{H}_{ref}$；(c) 引入时间基向量 $\mathbf{T}_b$（sinusoidal 编码时间步），通过外积生成时间对齐的加权矩阵 $\mathbf{W} = \mathbf{Z} \cdot \mathbf{T}_b^T$，使得不同时间步可以有不同的加权——这对 RIR 特别重要，因为早期反射和晚期混响的特征截然不同；(d) 最终预测 $\mathbf{S}_{pred} = \sum_k \mathbf{W}_k \odot \mathbf{S}_{ref,k}$
    - 设计动机：不同参考 RIR 在不同时间段对目标 RIR 的贡献不同——空间上近的参考在早期反射上更有参考价值，而晚期混响更依赖房间整体特性

### 损失函数 / 训练策略

总损失为频谱 L1 损失加能量衰减损失：$\mathcal{L}_{total} = \mathcal{L}_{STFT} + \lambda \mathcal{L}_{ED}$，其中 $\mathcal{L}_{STFT} = \|\exp(\mathbf{S}_{pred}) - \exp(\mathbf{S}_{gt})\|_1$，$\mathcal{L}_{ED} = \|\text{EDC}(\mathbf{S}_{pred}) - \text{EDC}(\mathbf{S}_{gt})\|_1$。推理时通过 Griffin-Lim 算法将预测频谱图转回波形。

## 实验关键数据

### 主实验 (AcousticRooms 跨房间预测)

| 方法 | Seen EDT↓ | Seen C50↓ | Seen T60↓ | Unseen EDT↓ | Unseen C50↓ | Unseen T60↓ |
|------|-----------|-----------|-----------|-------------|-------------|-------------|
| Few-shot RIR (K=8) | 0.174 | 4.451 | 32.71% | 0.187 | 4.470 | 21.15% |
| Nearest Neighbor (K=8) | 0.064 | 1.717 | 8.94% | 0.090 | 2.667 | 11.64% |
| **xRIR (K=8)** | **0.038** | **0.940** | **8.13%** | **0.055** | **1.457** | **10.53%** |

### 仿真到真实迁移 (4个真实房间)

| 方法 | 教室 EDT↓ | 走廊 EDT↓ | 复杂房间 EDT↓ |
|------|-----------|-----------|---------------|
| Diff-RIR (K=12) | 0.113 | 0.160 | 0.115 |
| Nearest Neighbor (K=8) | 0.108 | 0.068 | 0.091 |
| **xRIR (K=8)** | **0.093** | **0.062** | **0.077** |

### 关键发现

- xRIR 在已见环境中 C50 误差仅 0.940 dB，比 Nearest Neighbor 低 45%，证明模型学到了超越简单空间插值的声学规律
- 在未见环境上 xRIR 依然保持强劲表现，EDT 误差 0.055s 远低于 Few-shot RIR 的 0.187s，验证了跨房间泛化能力
- 参考 RIR 数量从 K=1 增加到 K=8 带来持续改善，但在 K=4 时已经达到了相当好的性能
- 仿真到真实迁移成功：xRIR 用仿真数据预训练后在真实房间上表现优于专门在真实房间训练的 Diff-RIR（且 Diff-RIR 用了 12 个参考而 xRIR 只用 8 个）
- T60 指标上 xRIR 有时不如 Nearest Neighbor，因为 T60 是全局指标，对低信噪比的晚期波形敏感，学习方法在仿真→真实转移中受 SNR 差异影响

## 亮点与洞察

- **参考 RIR 作为材料"探针"**：不需要显式建模材料属性，仅用少量 RIR 测量就隐式捕捉了房间的声学特性。这是一个巧妙的 workaround——材料属性难以直接测量，但它们的效果反映在 RIR 中
- **时间对齐加权矩阵**：不是对参考 RIR 做整体加权，而是在每个时间步独立加权。这符合声学直觉——早期反射（取决于几何）和晚期混响（取决于材料+几何）需要不同的参考权重
- **AcousticRooms 数据集**：使用先进的 DG（间断 Galerkin）方法模拟 RIR，比 GWA 数据集的 PFFDTD 方法保真度更高，332 种材料的随机分配确保了声学多样性

## 局限与展望

- 当前仅预测单通道（全向）RIR，未处理空间音频（如双耳 HRTF），而这对 VR 沉浸感至关重要
- 依赖全景深度图作为输入，获取成本高于普通 RGB 图像
- T60 指标上表现不够理想，可能需要专门的晚期混响建模（例如引入扩散模型来生成波形而非频谱加权）
- 推理时使用 Griffin-Lim 恢复相位，会引入很多伪影，更好的相位估计或直接波形生成方法值得探索
- AcousticRooms 的 260 个房间类别虽多但仍有限，扩大到更多样的建筑类型可进一步提升泛化

## 相关工作与启发

- **vs INRAS/NAFs**: 这些方法为单个房间训练单独的隐式神经网络，需要密集 RIR 数据，无法跨房间泛化。xRIR 用一个统一模型处理所有房间，仅需少量参考 RIR
- **vs Few-Shot RIR**: Few-Shot RIR 也尝试跨房间泛化，但其 UNet 解码器在高保真 RIR 重建上不够好，且原始设计使用双耳回声而非分离声源-接收器位置的 RIR，导致在 AcousticRooms 上表现差
- **vs Diff-RIR**: Diff-RIR 用可微渲染框架学习材料系数，但每个房间需单独训练，计算成本高。xRIR 作为预训练单一模型，仅需少量微调即可适应新房间

## 评分

- 新颖性: ⭐⭐⭐⭐ 跨房间 RIR 泛化是新问题，全景深度图+参考 RIR 的组合方案合理有效
- 实验充分度: ⭐⭐⭐⭐⭐ 模拟 seen/unseen + 真实环境 sim-to-real，基线全面
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，方法描述详尽，图示质量高
- 价值: ⭐⭐⭐⭐ AcousticRooms 数据集 + xRIR 框架对空间音频社区有实际贡献

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] One2Any: One-Reference 6D Pose Estimation for Any Object](one2any_one-reference_6d_pose_estimation_for_any_object.md)
- [\[CVPR 2025\] VTON 360: High-Fidelity Virtual Try-On from Any Viewing Direction](vton_360_high-fidelity_virtual_try-on_from_any_viewing_direction.md)
- [\[NeurIPS 2025\] MDReID: Modality-Decoupled Learning for Any-to-Any Multi-Modal Object Re-Identification](../../NeurIPS2025/human_understanding/mdreid_modality-decoupled_learning_for_any-to-any_multi-modal_object_re-identifi.md)
- [\[AAAI 2026\] Modelling the Effects of Hearing Loss on Neural Coding in the Auditory Midbrain with Variational Conditioning](../../AAAI2026/human_understanding/modelling_the_effects_of_hearing_loss_on_neural_coding_in_the_auditory_midbrain_.md)
- [\[CVPR 2026\] RAM: Recover Any 3D Human Motion in-the-Wild](../../CVPR2026/human_understanding/ram_recover_any_3d_human_motion_in-the-wild.md)

<!-- RELATED:END -->
