---
title: >-
  [论文解读] Occlusion Handling in 3D Human Pose Estimation with Perturbed Positional Encoding
description: >-
  [ECCV 2024][人体理解][3D人体姿态估计] 针对人体关节遮挡导致2D骨架图边缺失、传统图拉普拉斯位置编码失效的问题，提出PerturbPE方法，利用瑞利-薛定谔微扰定理多次随机扰动并求平均来提取图拉普拉斯特征基的一致性部分作为位置编码，在完整骨架上优于MöbiusGCN，在边缺失场景下性能提升达12%。
tags:
  - ECCV 2024
  - 人体理解
  - 3D人体姿态估计
  - 图卷积网络
  - 位置编码
  - 遮挡处理
  - 瑞利-薛定谔微扰理论
---

# Occlusion Handling in 3D Human Pose Estimation with Perturbed Positional Encoding

**会议**: ECCV 2024  
**arXiv**: [2405.17397](https://arxiv.org/abs/2405.17397)  
**代码**: 无  
**领域**: 人体理解  
**关键词**: 3D人体姿态估计, 图卷积网络, 位置编码, 遮挡处理, 瑞利-薛定谔微扰理论

## 一句话总结

针对人体关节遮挡导致2D骨架图边缺失、传统图拉普拉斯位置编码失效的问题，提出PerturbPE方法，利用瑞利-薛定谔微扰定理多次随机扰动并求平均来提取图拉普拉斯特征基的一致性部分作为位置编码，在完整骨架上优于MöbiusGCN，在边缺失场景下性能提升达12%。

## 研究背景与动机

3D人体姿态估计的目标是从2D关节位置提升(lifting)到3D坐标。图卷积网络(GCN)因参数量小且精度高成为主流方案，特别是MöbiusGCN仅用0.042M参数就取得了优秀结果。为提升GCN的表达力，利用图拉普拉斯的特征向量作为位置编码是有效手段。

**核心痛点**：在实际场景中，人体经常被遮挡(自遮挡或物体遮挡)，导致2D姿态估计器检测不到某些关节，即输入图中部分边缺失。此时图拉普拉斯矩阵改变，其特征向量不再适用作为位置编码。现有的位置编码方法(如Laplacian PE、SignNet/BasisNet)都假设完整图输入，无法处理边缺失的情况。从理论角度看，部分边缺失后的子图匹配是NP完全问题。

**切入角度**：图拉普拉斯的特征基可分解为"规则部分"和"不规则部分"。网络的规则性体现在随机移除少量链路前后结构特征的一致性上。因此可以通过多次随机扰动并取平均来提取一致的、鲁棒的特征基部分。

**核心idea**：用瑞利-薛定谔微扰理论(RSPT)高效计算扰动后的特征向量（不需要重新计算整个特征基），多次独立随机去边并扰动求平均，得到一致性位置编码PerturbPE。

## 方法详解

### 整体框架

PerturbPE建立在MöbiusGCN之上。整体流程：(1) 给定可能有边缺失的2D人体骨架图，计算其图拉普拉斯矩阵；(2) 独立进行κ次扰动——每次随机移除一些边，用RSPT计算扰动后的特征向量；(3) 对κ次扰动的特征向量取平均，得到一致性位置编码P；(4) 将P通过MLP融合到每一层的节点特征中，输入MöbiusGCN进行3D姿态预测。

### 关键设计

1. **瑞利-薛定谔微扰理论(RSPT)计算扰动特征向量**:

    - 功能：给定原始图拉普拉斯 $\mathbf{A}_0$ 和扰动矩阵 $\mathbf{A}_1$（由被移除的边构成），高效计算扰动后的特征对
    - 核心思路：$\mathbf{A}(\epsilon) = \mathbf{A}_0 + \epsilon \mathbf{A}_1$，通过级数展开 $\mathbf{v}_i(\epsilon) = \sum_{k=0}^{\infty} \epsilon^k \mathbf{v}_i^{(k)}$ 逐阶逼近。实验中设 $\epsilon=1, k=1$（一阶微扰），利用Moore-Penrose伪逆和QR分解高效求解。对简并(特征值重复)和非简并情况分别处理
    - 设计动机：RSPT的核心优势是不需要重新计算整个特征分解，只需计算伪逆向量积，对于人体骨架图(17个节点)这种小图来说计算开销可忽略不计(推理时间仅从0.009s增到0.010s)

2. **多次扰动取平均提取一致性位置编码(PerturbPE)**:

    - 功能：独立执行κ次RSPT扰动，每次随机移除不同的边集，将得到的扰动特征向量取平均
    - 核心思路：
      $\mathbf{p} = \frac{\sum_{i=1}^{\kappa} \mathbf{v}_i}{\kappa}$
      其中 $\mathbf{v}_i$ 是第i次扰动后的特征向量。平均操作过滤掉随机扰动引入的不规则分量，保留图结构的一致性信息
    - 设计动机：基于图拉普拉斯特征基可分解为规则(consistent)和不规则(irregular)部分的理论。规则部分反映了不受少量边变化影响的结构特征，正是我们需要的位置编码。多次取平均是一种简洁的蒙特卡洛近似

3. **位置特征融合与Masked Condition训练策略**:

    - 功能：将PerturbPE编码与节点特征相加后通过MLP融合，嵌入MöbiusGCN的每一层
    - 核心思路：$\mathbf{X}^{\ell} = \sigma(f(\mathbf{Z}^{\ell} + \mathbf{P}))$，其中f是MLP。完整的MöbiusGCN块变为：
      $\mathbf{Z}^{\ell+1} = \sigma(2\Re\{\mathbf{U} \operatorname{Möbius}(\Lambda) \mathbf{U}^\top \sigma(f(\mathbf{Z}^{\ell} + \mathbf{P})) \mathbf{W}^{\ell+1}\} + \mathbf{b})$
    - 训练时假设每个样本随机缺失0-2条边(Masked Condition Strategy)，使模型学会处理各种缺失模式
    - 设计动机：一个网络适应所有缺失情况，比为每种缺失模式训练独立模型更实用

### 损失函数 / 训练策略

使用标准MSE损失：$\mathcal{L}(\mathcal{Y}, \hat{\mathcal{Y}}) = \sum_{i=1}^{k}(\mathcal{Y}_i - \hat{\mathcal{Y}}_i)^2$

训练细节：Adam优化器，初始学习率0.001，batch size 64。使用8个MöbiusGCN块，中间层通道数128(0.16M参数)或192(0.66M参数)。2D输入归一化到[-1,1]，推理时通过骨骼长度归一化校准尺度。

## 实验关键数据

### 主实验

**完整骨架(无遮挡)**：

| 数据集 | 指标 | PerturbPE | MöbiusGCN | 提升 |
|--------|------|-----------|-----------|------|
| Human3.6M (GT输入) | MPJPE(mm) | 32.7 | 34.1 | -1.4mm |
| MPI-INF-3DHP | PCK | 82.0 | 80.0 | +2.0 |

**缺失1条边**：

| 配置 | MPJPE(mm) | 说明 |
|------|-----------|------|
| 特征向量标记(baseline) | 55.0 | 标准拉普拉斯PE |
| +解决简并性 | 51.4 | 提升6.5% |
| +1边扰动 | 49.0 | 提升10.9% |
| +2边扰动 | 48.0 | 提升12.7% |

**缺失2条边**：

| 配置 | MPJPE(mm) | 说明 |
|------|-----------|------|
| MöbiusGCN | 60.0 | 无位置编码 |
| PerturbPE | 54.0 | 降低10% |

### 消融实验

| 配置 | MPJPE | 说明 |
|------|-------|------|
| 训练数据减少至3个subject | 42.9 vs 44.7 | 仍优于MöbiusGCN |
| 训练数据减少至2个subject | 48.9 vs 50.9 | 数据少时优势保持 |
| 训练数据减少至1个subject | 66.4 vs 67.4 | 边际但一致的提升 |

### 关键发现

- 在遮挡场景(边缺失)下，PerturbPE的优势大幅放大，1条边缺失提升可达12%
- 单网络训练即可处理所有边缺失组合，优于GFPose针对特定缺失模式训练独立网络的策略
- 具体遮挡部位实验显示：双腿缺失(52.4)优于双臂缺失(58.6)，可能因为腿部骨骼更规则、更容易从其他部位推断
- PerturbPE不增加模型参数(仅增加预处理计算)，推理时间几乎不变

## 亮点与洞察

- 将量子力学中的瑞利-薛定谔微扰理论引入图神经网络的位置编码，跨学科联结很有创新性
- 从图拉普拉斯特征基的规则/不规则分解角度理解遮挡问题，提供了理论支撑
- 方法极其轻量——不增加参数、不改变架构、推理时间几乎不变，纯粹通过更好的位置编码提升性能

## 局限与展望

- 假设关节数已知（只是边缺失），实际遮挡可能导致关节也不可见
- 扰动次数κ增加会线性增加预处理时间，论文未讨论κ的最优选择策略
- 仅在MöbiusGCN上验证，与其他GCN架构(如SemGCN初步验证)的通用性未充分展开
- 仅用Hourglass和GT作为2D输入，未在更现代的2D检测器(如HRNet)上系统测试

## 相关工作与启发

- MöbiusGCN用Möbius变换替代标准的图卷积滤波器，是目前最轻量级的3D HPE方法，本文以此为基础架构
- SignNet/BasisNet解决了图拉普拉斯位置编码的符号模糊和特征值重复问题，但未考虑边缺失
- GFPose为每种遮挡模式训练独立网络，本文用单一网络的PerturbPE统一处理所有情况更优雅
- 该方法的思想具有通用性——任何图结构中部分边不确定/缺失时，都可尝试PerturbPE来获得鲁棒位置编码

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将RSPT引入GCN位置编码是全新视角，理论与实践结合紧密
- 实验充分度: ⭐⭐⭐⭐ 完整骨架和多种缺失场景下都有实验，但2D输入源偏少
- 写作质量: ⭐⭐⭐⭐ 理论推导完整，但数学符号较重，阅读门槛高
- 价值: ⭐⭐⭐⭐ 解决了一个重要且常被忽视的实际问题，思想可迁移

<!-- RELATED:START -->

## 相关论文

- [RePOSE: 3D Human Pose Estimation via Spatio-Temporal Depth Relational Consistency](repose_3d_human_pose_estimation_via_spatio-temporal_depth_relational_consistency.md)
- [3DSA: Multi-view 3D Human Pose Estimation With 3D Space Attention Mechanisms](3dsa_multi-view_3d_human_pose_estimation_with_3d_space_attention_mechanisms.md)
- [WorldPose: A World Cup Dataset for Global 3D Human Pose Estimation](worldpose_a_world_cup_dataset_for_global_3d_human_pose_estimation.md)
- [UPose3D: Uncertainty-Aware 3D Human Pose Estimation with Cross-View and Temporal Cues](upose3d_uncertainty-aware_3d_human_pose_estimation_with_cross-view_and_temporal_.md)
- [3D Hand Pose Estimation in Everyday Egocentric Images](3d_hand_pose_estimation_in_everyday_egocentric_images.md)

<!-- RELATED:END -->
