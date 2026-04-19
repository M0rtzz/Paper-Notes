---
title: >-
  [论文解读] SegmentDreamer: Towards High-Fidelity Text-to-3D Synthesis with Segmented Consistency Trajectory Distillation
description: >-
  [ICCV 2025][3D视觉][文本到3D生成] 本文提出SegmentDreamer，通过分段一致性轨迹蒸馏（SCTD）重新表述SDS损失，解决了现有一致性蒸馏（CD）方法中自一致性和交叉一致性之间的不平衡问题，在单张A100 GPU上仅需~32分钟即可通过3DGS生成高保真3D资产。
tags:
  - ICCV 2025
  - 3D视觉
  - 文本到3D生成
  - 一致性蒸馏
  - Score Distillation
  - 3D高斯泼溅
  - 扩散模型
---

# SegmentDreamer: Towards High-Fidelity Text-to-3D Synthesis with Segmented Consistency Trajectory Distillation

**会议**: ICCV 2025  
**arXiv**: [2507.05256](https://arxiv.org/abs/2507.05256)  
**代码**: [https://zjhJOJO.github.io/segmentdreamer](https://zjhJOJO.github.io/segmentdreamer)  
**领域**: 3D视觉  
**关键词**: 文本到3D生成, 一致性蒸馏, Score Distillation, 3D高斯泼溅, 扩散模型

## 一句话总结
本文提出SegmentDreamer，通过分段一致性轨迹蒸馏（SCTD）重新表述SDS损失，解决了现有一致性蒸馏（CD）方法中自一致性和交叉一致性之间的不平衡问题，在单张A100 GPU上仅需~32分钟即可通过3DGS生成高保真3D资产。

## 研究背景与动机

文本到3D生成是计算机视觉与图形学的前沿方向，核心方案是通过Score Distillation Sampling (SDS) 从预训练的2D文本-图像扩散模型中"蒸馏"出3D表示。近年来，一致性蒸馏（Consistency Distillation, CD）被引入以改进SDS，代表工作包括CDS（Consistent3D）和GCS（ConnectCD）。

然而，现有CD-based方法存在根本性缺陷——**自一致性与交叉一致性之间的不平衡**：

- **CDS** 仅强制自一致性（同一ODE轨迹上的点映射到相同端点），完全忽略了交叉一致性（无条件与有条件ODE轨迹之间的对齐），导致缺乏有效的条件引导，生成语义不一致的细节
- **GCS** 试图同时强制两种一致性，但其一致性函数 $\boldsymbol{G}_\theta$ 存在固有缺陷（噪声预测模型缺少目标时间步），且在整个ODE轨迹上强制交叉一致性导致过度条件引导，产生过曝和伪影

此外，两者的蒸馏误差上界都较大：CDS为 $\mathcal{O}(\Delta_t)T$，GCS为 $\mathcal{O}(\Delta_t)(T-e)$，限制了生成质量。

本文的核心idea：**将PF-ODE轨迹分段，在每个子轨迹内分别强制自一致性和交叉一致性，同时显式定义两者的关系，从而获得显著更紧的蒸馏误差上界 $\mathcal{O}(\Delta_t)(s_{m+1}-s_m)$**。

## 方法详解

### 整体框架

SegmentDreamer采用以下pipeline：(1) 用Point-E初始化3D高斯；(2) 每次迭代随机渲染一批相机视角 $\mathbf{z}_0$；(3) 扩散到 $\mathbf{z}_{s_m}$（使用固定噪声 $\epsilon^*$）；(4) 通过无条件确定性采样获得 $\tilde{\mathbf{z}}_t^{\Phi}$；(5) 通过条件确定性采样获得 $\hat{\mathbf{z}}_s^{\Phi}$；(6) 计算SCTD损失优化3D表示 $\theta$。

### 关键设计

1. **分段一致性轨迹蒸馏（SCTD）**：

    - 功能：将整个时间步范围 $[0, T]$ 划分为 $N_s$ 个子区间，在每个子轨迹 $[s_m, s_{m+1})$ 内分别强制自一致性和交叉一致性
    - 核心思路：通过对SDS进行等价变换，将其分解为三部分：
    $\mathcal{L}_\text{SDS} = \mathbb{E}_{t,s}[b(t)||\underbrace{G^m_\theta(\hat{\mathbf{z}}_s, s, \emptyset) - G^m_\theta(\tilde{\mathbf{z}}_t, t, \emptyset)}_{\text{self-consistency}} + (\omega+1)\underbrace{(G^m_\theta(\tilde{\mathbf{z}}_t, t, \emptyset) - G^m_\theta(\tilde{\mathbf{z}}_t, t, \mathbf{y}))}_{\text{cross-consistency}} + \underbrace{\mathbf{z}_{s_m} - G^m_\theta(\hat{\mathbf{z}}_s, s, \emptyset)}_{\text{generative prior}}||_2^2]$
    - 设计动机：显式定义自一致性与交叉一致性的关系，避免CDS缺失交叉一致性和GCS过度条件引导的问题

2. **SCTD采样方法**：

    - 功能：去掉generative prior项，对自一致性和交叉一致性施加更严格的独立约束
    - 核心思路：最终SCTD损失包含两个独立的L2范数项，分别约束自一致性和交叉一致性，使用stop-gradient阻止梯度传播
    - 设计动机：直接最小化合并项 $||x_1 - x_2 + \omega(y_1 - y_2)||^2 = 0$ 无法保证 $x_1 = x_2$ 和 $y_1 = y_2$ 同时成立，因此需要独立约束

3. **轨迹分段策略**：

    - 功能：提出等分分段和单调递增分段两种方案
    - 核心思路：等分将 $[0,T]$ 均匀分为 $N_s$ 段；单调递增使较大噪声水平的段更长，公式为 $s_{m+1} - s_m = t_\tau + \frac{2m(T - N_s t_\tau)}{N_s(N_s-1)}$
    - 设计动机：由于 $t, s$ 在训练中随机采样，两种策略效果差异不大，$N_s = 5$ 在多数情况下表现最佳

4. **快速稳定优化管线**：

    - **动态采样步数调整**：当 $t > t_\tau$ 时使用两步确定性采样获取 $\tilde{\mathbf{z}}_t^{\Phi}$，否则使用一步采样，实现质量与速度的权衡
    - **一致性函数近似**：将 $G^m_\theta(\tilde{\mathbf{z}}_t^{\Phi}, t, \emptyset)$ 近似为 $\mathbf{z}_{s_m}$，利用PF-ODE的理论可逆性。这不仅省去U-Net Jacobian计算，还提升了优化稳定性

### 损失函数 / 训练策略

- 基础模型：Stable Diffusion 2.1
- 3D表示：3D Gaussian Splatting，Point-E初始化
- 优化器：Adam，5000次迭代
- 时间步范围：$t \sim \mathcal{U}(20, 500 + t_{\text{warm}})$，$t_{\text{warm}}$ 在前1500个epoch从480线性衰减到0
- 训练时间：~32分钟（CFG），~38分钟（Perp-Neg）

## 实验关键数据

### 主实验

| 方法 | CLIP-L↑ | IR↑ | FID↓ | 时间(min)↓ | 用户偏好Q1↓ | Q2↓ | Q3↓ |
|------|---------|-----|------|-----------|-----------|-----|-----|
| DreamFusion | 28.47 | -0.004 | 140.84 | 60 | 4.73 | 4.87 | 4.90 |
| LucidDreamer | 29.99 | 0.006 | 121.80 | 45 | 2.93 | 2.98 | 2.93 |
| Consistent3D (CDS) | 30.60 | 0.004 | 113.61 | 140 | 4.14 | 3.88 | 3.93 |
| ConnectCD (GCS) | 30.73 | 0.018 | 112.61 | 80 | 1.63 | 1.80 | 1.93 |
| **SegmentDreamer** | **30.88** | **0.020** | **110.45** | **38** | **1.57** | **1.47** | **1.33** |

### 消融实验

| 配置 | 关键影响 | 说明 |
|------|---------|------|
| $N_s = 1$ | 输出过度平滑 | 等同于CD，误差上界大 |
| $N_s = 5$ | 最佳平衡 | 细节丰富且表示清晰 |
| $N_s = 10$ | 细节增加但表示模糊 | 类似LucidDreamer的问题 |
| 等分 vs 单调递增 | 差异很小 | 因为t,s随机采样 |
| $t_\tau$ 大 | 过度平滑 | 采样步数不足难以保留 $\mathbf{z}_0$ 信息 |
| 有/无近似策略 | 无近似有轻微过曝 | 近似策略反而更好，类似省略Jacobian |

### 关键发现

- SCTD在所有定量指标（CLIP、IR、FID）上均优于CDS和GCS
- 生成时间仅38分钟，是Consistent3D的1/4，ConnectCD的1/2
- 用户研究中在文本对齐、物体真实性和细节三个维度均排名第一
- CDS在正常CFG scale下无法提供有效条件引导，GCS虽然改善但存在过曝和伪影

## 亮点与洞察

1. **理论贡献扎实**：证明了SCTD的蒸馏误差上界为 $\mathcal{O}(\Delta_t)(s_{m+1}-s_m)$，显著紧于CDS和GCS的上界
2. **问题诊断精准**：准确定位CD-based方法的核心问题在于自一致性与交叉一致性的不平衡，而非简单的CFG scale调整
3. **工程优化巧妙**：一致性函数近似策略看似违反理论（PF-ODE一阶求解器非精确可逆），但实践中反而改善了稳定性，这与DreamFusion省略Jacobian有异曲同工之妙
4. **通用性**：SCTD可无缝应用于3D头像生成、3D肖像生成等多种3D生成任务

## 局限与展望

- 主要聚焦单实例生成，多实例场景下效果次优
- 基于3DGS，可能继承3DGS在某些几何拓扑上的限制
- 可能被用于创建误导性内容（潜在负面影响）
- 未与基于多视图一致性的方法（如MVDream、Zero123++等）进行比较

## 相关工作与启发

- CSD [Yu et al.] 首先发现SDS中"classifier score"的作用，本文在此基础上进一步分析了CD框架中的对应组件
- 分段一致性模型（sCM, sCT）为本文的核心分段思想提供了理论基础
- 启示：在将2D生成先验蒸馏到3D时，理解并正确平衡不同一致性约束比简单堆叠技术更重要

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从理论上重新分析SDS与CD的关系，提出分段一致性蒸馏，思路新颖
- 实验充分度: ⭐⭐⭐⭐ 定量+定性+用户研究+消融，但40个prompt的测试规模偏小
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，但公式密集导致可读性稍有下降
- 价值: ⭐⭐⭐⭐⭐ 速度和质量双优，推动CD-based text-to-3D走向实用

<!-- RELATED:START -->

## 相关论文

- [Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis](gaussian_variation_field_diffusion_for_high-fidelity_video-to-4d_synthesis.md)
- [Advancing Text-to-3D Generation with Linearized Lookahead Variational Score Distillation](advancing_text-to-3d_generation_with_linearized_lookahead_variational_score_dist.md)
- [Walking the Schrödinger Bridge: A Direct Trajectory for Text-to-3D Generation](../../NeurIPS2025/3d_vision/walking_the_schrödinger_bridge_a_direct_trajectory_for_text-to-3d_generation.md)
- [Stable Score Distillation](stable_score_distillation.md)
- [JointDreamer: Ensuring Geometry Consistency and Text Congruence in Text-to-3D Generation via Joint Score Distillation](../../ECCV2024/3d_vision/jointdreamer_ensuring_geometry_consistency_and_text_congruence_in_text-to-3d_gen.md)

<!-- RELATED:END -->
