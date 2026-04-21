---
title: >-
  [论文解读] Efficient Hybrid SE(3)-Equivariant Visuomotor Flow Policy via Spherical Harmonics
description: >-
  [CVPR 2026][3D视觉][SE(3) Equivariance] 提出E3Flow，首个基于球谐表示的等变flow matching策略框架，通过特征增强模块（FEM）动态融合点云和图像两种模态的视觉信息，结合rectified flow实现高效等变动作生成，在MimicGen 8个任务上平均成功率超过最强基线SDP 3.12%的同时推理速度提升7倍。
tags:
  - CVPR 2026
  - 3D视觉
  - SE(3) Equivariance
  - Spherical Harmonics
  - Rectified Flow
  - Robot Policy Learning
  - 多模态
---

# Efficient Hybrid SE(3)-Equivariant Visuomotor Flow Policy via Spherical Harmonics

**会议**: CVPR 2026  
**arXiv**: [2603.23227](https://arxiv.org/abs/2603.23227)  
**代码**: [https://github.com/zql-kk/E3Flow](https://github.com/zql-kk/E3Flow)  
**领域**: 3D Vision / Robot Manipulation  
**关键词**: SE(3) Equivariance, Spherical Harmonics, Rectified Flow, Robot Policy Learning, Multi-Modal Fusion

## 一句话总结

提出E3Flow，首个基于球谐表示的等变flow matching策略框架，通过特征增强模块（FEM）动态融合点云和图像两种模态的视觉信息，结合rectified flow实现高效等变动作生成，在MimicGen 8个任务上平均成功率超过最强基线SDP 3.12%的同时推理速度提升7倍。

## 研究背景与动机

扩散策略（Diffusion Policy）在机器人策略学习中成效显著，但面临两大核心挑战：

**数据效率低**：依赖大量高质量专家演示数据，而数据采集成本高昂

**等变方法的瓶颈**：嵌入对称先验可大幅提升数据效率和泛化能力，但现有等变扩散策略存在：
   - **计算密集**：需要大量迭代去噪步骤，叠加到已经复杂的等变网络上更加缓慢
   - **单模态依赖**：仅使用点云或图像，缺少精细视觉细节
   - **快采样不兼容**：直接将one-step diffusion或flow matching应用于等变策略会导致不稳定和性能下降

核心gap：**尚无方法将等变学习的数据效率与flow matching的推理速度统一起来。**

如图1(a)所示，当桌上玩具旋转到未见过的姿态时，非等变DP无法抓取，而等变的E3Flow可通过对应旋转原始轨迹成功执行。

## 方法详解

### 整体框架

E3Flow（图2）的流程：

1. **多模态编码**：手眼相机图像→ResNet提取不变特征；单视角点云→EquiformerV2提取等变特征
2. **FEM融合**：将图像语义注入到点云的球谐表示中
3. **球谐条件构建**：融合视觉特征 + 本体感知状态映射到球谐空间
4. **Rectified Flow生成**：基于ODE求解高效生成等变动作序列

### 关键设计

1. **球谐视觉表示**：用球谐函数实现严格SO(3)等变

    - 球谐函数 $Y_l^m(\theta,\phi)$ 是单位球面上的正交基，在旋转R下，同一阶l的球谐函数发生线性混合：$Y_l^m(R^{-1}\hat{\mathbf{r}}) = \sum_{m'} D_{mm'}^{(l)}(R) Y_l^{m'}(\hat{\mathbf{r}})$
    - 使用EquiformerV2编码点云获得多阶球谐特征（标量 $f_{pcd}^{(0)}$ + 高阶 $f_{pcd}^{(>0)}$），保留精细方向和旋转细节
    - 相比EquiBot的vector neurons方法，EquiformerV2编码高阶系数能捕获更精细的方向信息
    - 相比SDP仅用稀疏点云，本方法引入混合视觉输入

2. **Feature Enhancement Module (FEM)**：跨模态动态融合

    - 仅对标量分量（Type-0）注入图像特征，保持高阶分量的等变性不受破坏
    - 核心公式：$f_{fused} = \Pi[\Lambda(\mathcal{A}(f_{pcd}^{(0)}, f_{img}), f_{pcd}^{(0)}) \| f_{pcd}^{(>0)}]$
    - $\mathcal{A}$：跨模态注意力，将点云标量特征作为query、图像特征作为key/value
    - $\Lambda$：门控机制，自适应平衡图像贡献——与简单拼接（cat）导致的性能下降（72.36%→79.00%）形成对比
    - $\Pi$：投影回球谐空间
    - 设计动机：纯粹拼接不同模态特征会破坏等变结构；FEM仅在不变子空间（Type-0）操作，优雅地保持等变性

3. **等变Rectified Flow**：高效动作生成

    - 学习从噪声分布到动作分布的直线插值路径：$x_t = (1-t)x_0 + ta$
    - 训练损失：$\mathcal{L}_{RF}(\theta) = \mathbb{E}_{t,x_0,x_1}[\|v_\theta(x_t,t,s,v) - (a-x_0)\|^2]$
    - 由于速度场网络是等变的，满足 $v_\theta(\rho*x_t, t, \rho*s, \rho*v) = \rho * v_\theta(x_t, t, s, v)$
    - 训练目标是等变动作的线性变换，损失在群作用下不变，因此存在等变最优解
    - 默认10步采样，推理时间0.51s，比SDP(DDPM)的3.73s快7倍

### 损失函数 / 训练策略

- 优化器：AdamW，学习率 $1 \times 10^{-4}$，批大小64
- EMA衰减率0.95
- 单卡NVIDIA H20 GPU训练500 epochs
- 每20 epochs评估一次，每次50个episode，报告最大成功率
- 动作表示：3D位置+6D旋转+1D夹爪状态（位置和旋转为等变信息）

## 实验关键数据

### 主实验

**MimicGen 8任务成功率** (Table 1, 100个专家演示)

| 方法 | 等变类型 | Coffee_D2 | Nut_Asm | Square | Stack3 | **平均** |
|------|----------|-----------|---------|--------|--------|----------|
| DP | 无 | 44 | 54 | 10 | 32 | 47.50 |
| EquiDiff(voxel) | SO(2) | 65 | 67 | 39 | 76 | 68.50 |
| SDP(DDPM) | SE(3) | 63 | 92 | 64 | 98 | 75.88 |
| **E3Flow** | **SE(3)** | **64** | **94** | **70** | **100** | **79.00** |

**推理时间** (Table 2)

| 方法 | 平均推理时间(s) | 相对E3Flow |
|------|----------------|-----------|
| EquiBot | 2.03 | 4.0× |
| DP | 0.95 | 1.9× |
| EquiDiff(img) | 2.51 | 4.9× |
| EquiDiff(voxel) | 1.10 | 2.2× |
| SDP(DDPM) | 3.73 | **7.3×** |
| SDP(DDIM) | 0.46 | 0.9× |
| **E3Flow** | **0.51** | **1.0×** |

注：SDP用DDIM加速后成功率下降6.13%（75.88→69.75），而E3Flow以相近速度获得更高成功率。

### 消融实验

**组件分析** (Table 4)

| 输入 | 融合 | 生成方式 | 平均成功率 |
|------|------|----------|----------|
| PCD | - | RF | 75.88 |
| PCD | - | Diffusion | 75.23 |
| PCD+Img | cat | RF | 72.36 |
| PCD+Img | FEM | Diffusion | 77.58 |
| **PCD+Img** | **FEM** | **RF** | **79.00** |

**Flow方法对比** (Table 5)

| 方法 | 步数 | 推理时间 | 平均成功率 |
|------|------|----------|----------|
| MeanFlow | 1 | 0.17s | 54.50 |
| AlphaFlow | 1 | 0.17s | 64.62 |
| RF-1 | 1 | 0.16s | 69.00 |
| RF-5 | 5 | 0.28s | 71.00 |
| **RF-10** | **10** | **0.51s** | **79.00** |

### 关键发现

- 简单拼接多模态特征（cat）反而降低性能（79.00→72.36），说明模态对齐至关重要
- FEM通过仅操作不变子空间（Type-0特征）优雅解决了等变性与多模态融合的矛盾
- 等变学习在复杂任务上优势明显：DP在Square_D2上仅10%，E3Flow达70%
- 一步采样对等变模型不适用（MeanFlow仅54.50%），因单次前传不足以让高度抽象的等变特征引导精细动作
- SE(3)变换泛化实验（Table 3）：E3Flow在10°倾斜零样本测试中全面领先SDP
- 数据效率：E3Flow用100个演示即可达到其他方法用200个演示的效果（图5）

## 亮点与洞察

1. **等变性 + Flow Matching的首次成功统一**：证明了rectified flow可自然适配等变网络——因为训练目标是等变动作的线性变换，损失在群作用下不变
2. **FEM的精妙设计**：仅在Type-0不变子空间注入图像语义，不破坏高阶等变特征，解决了"多模态融合 vs 等变性保持"的两难
3. **对one-step方法的深入分析**：揭示了等变模型需要多步采样的原因——高度抽象的等变特征需要更多步来解码为精细动作
4. **端到端等变证明**：从输入到输出的完整等变链条有严格数学保证
5. **实际部署潜力**：0.51s推理时间 + 100演示数据效率 → 适合真实机器人场景

## 局限与展望

- EquiformerV2的单次前传虽比ET-SEED快，但仍是推理瓶颈所在
- 仅验证了SE(3)等变，对更一般的对称群（如尺度变换SIM(3)）未探索
- 点云下采样到1024点可能丢失细节，对精密装配任务可能不够
- 真实环境实验仅4个任务，规模有限
- 未讨论sim-to-real gap和域随机化的影响
- FEM的图像编码器（ResNet）未使用预训练权重（如CLIP），可能限制语义理解

## 相关工作与启发

- 在SDP框架上扩展：SDP使用球谐表示+扩散，E3Flow替换为rectified flow并增加图像输入
- 与EquiDiff的对比揭示了连续等变（SO(3)球谐）vs 离散等变（SO(2)卷积）的差异
- FEM的设计思路可推广到其他需要在等变表示中注入不变信息的场景
- Rectified flow在机器人策略中的应用值得进一步研究（如更少采样步、蒸馏等）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次统一等变学习和flow matching是重要贡献，但核心组件均建立在已有方法上
- **实验充分度**: ⭐⭐⭐⭐ — 8个仿真任务+4个真实任务，丰富的消融和基线对比
- **写作质量**: ⭐⭐⭐⭐ — 数学推导清晰，图表专业
- **价值**: ⭐⭐⭐⭐⭐ — 解决了等变策略的推理效率瓶颈，对机器人学习社区有直接实用价值

<!-- RELATED:START -->

## 相关论文

- [SE(3)-Equivariant Diffusion Policy in Spherical Fourier Space](../../ICML2025/3d_vision/se3-equivariant_diffusion_policy_in_spherical_fourier_space.md)
- [DropAnSH-GS: Dropping Anchor and Spherical Harmonics for Sparse-view Gaussian Splatting](dropping_anchor_and_spherical_harmonics_for_sparse-view_gaussian_splatting.md)
- [ECKConv: Learning Coordinate-based Convolutional Kernels for Continuous SE(3) Equivariant Point Cloud Analysis](learning_coordinate-based_convolutional_kernels_for_continuous_se3_equivariant_a.md)
- [From Pairs to Sequences: Track-Aware Policy Gradients for Keypoint Detection](from_pairs_to_sequences_track-aware_policy_gradients_for_keypoint_detection.md)
- [Ada3Drift: Adaptive Training-Time Drifting for One-Step 3D Visuomotor Robotic Manipulation](ada3drift_adaptive_trainingtime_drifting_for_onest.md)

<!-- RELATED:END -->
