---
title: >-
  [论文解读] VeCoR — Velocity Contrastive Regularization for Flow Matching
description: >-
  [CVPR 2026][图像生成][Flow Matching] 提出 VeCoR（速度对比正则化），在标准 Flow Matching 训练中引入"负速度"对比信号，通过同时指导模型"该往哪走"和"不该往哪走"，实现更稳定的轨迹演化和更高的感知保真度——在 ImageNet-1K 上 SiT-XL/2 和 REPA-SiT-XL/2 分别获得 22% 和 35% 的 FID 相对降低。
tags:
  - CVPR 2026
  - 图像生成
  - Flow Matching
  - 对比学习
  - 速度场正则化
  - 负样本引导
---

# VeCoR — Velocity Contrastive Regularization for Flow Matching

**会议**: CVPR 2026  
**arXiv**: [2511.18942](https://arxiv.org/abs/2511.18942)  
**代码**: 有（项目页面）  
**领域**: 图像生成  
**关键词**: Flow Matching, 对比学习, 速度场正则化, 负样本引导, 图像生成

## 一句话总结
提出 VeCoR（速度对比正则化），在标准 Flow Matching 训练中引入"负速度"对比信号，通过同时指导模型"该往哪走"和"不该往哪走"，实现更稳定的轨迹演化和更高的感知保真度——在 ImageNet-1K 上 SiT-XL/2 和 REPA-SiT-XL/2 分别获得 22% 和 35% 的 FID 相对降低。

## 研究背景与动机
**领域现状**：Flow Matching (FM) 已成为扩散模型的强力替代，通过学习时间相关的速度场将先验分布传输到数据分布。FM 具有理论优雅性和计算效率。

**现有痛点**：标准 FM 仅提供单侧正向监督——训练模型"往正确方向走"，但缺乏"不要往错误方向走"的反馈。在轻量模型或低步数配置下，速度场的微小不一致会累积误差，导致样本偏离数据流形。

**核心矛盾**：FM 的监督是方向不对称的（只有吸引力，没有排斥力），在数据/模型容量受限时，某些区域的学到的流缺乏充分正则化，出现颜色偏移、几何失真、模糊和伪影等问题。

**切入角度**：受对比学习启发——既然能构造正样本对齐表示，为什么不也构造"负速度"来排斥不良流方向？

**核心idea**：将 FM 从纯吸引式目标扩展为"吸引-排斥"双侧训练信号，通过在图像/潜空间/速度空间上构造增强式负样本来正则化速度场。

## 方法详解

### 整体框架
VeCoR 是一个即插即用的训练方案，不改变 FM 的核心 ODE 公式。在标准 FM 目标基础上增加对比正则项：对每个训练样本，构造正速度 $\hat{v}_+$（标准 GT 速度）和一组负速度 $\hat{v}_-$（通过增强扰动获得），训练时同时拉近正速度、推远负速度。

### 关键设计

1. **VeCoR 对比损失**:

    - 功能：为 FM 增加双侧速度空间监督
    - 核心思路：在标准 FM MSE 损失基础上添加排斥项：
    $\hat{\mathcal{L}}^{(\text{VeCoR})} = \frac{1}{N}\sum_{i=1}^N \left[\|v_\theta - \hat{v}_+^{(i)}\|_2^2 - \lambda \sum_{j=1}^K \|v_\theta - \hat{v}_-^{(ij)}\|_2^2\right]$
      第一项吸引预测速度向 GT 对齐，第二项排斥预测速度远离负样本方向。$\lambda \in (0,1)$ 控制排斥强度。
    - 设计动机：正向目标只告诉"该去哪"，排斥项告诉"别去哪"——在有限数据/模型容量下，补充正则化那些欠约束的区域

2. **负速度候选集构造**:

    - 功能：构造语义一致但动力学不一致的负速度样本
    - 核心思路：在三个层次施加增强式扰动——
        - **图像空间(I)**：对训练图像做随机裁剪/颜色抖动等变换 → 编码为扰动潜表示 → 计算负速度
        - **潜空间(II)**：直接对潜表示施加扰动 → 计算负速度
        - **速度空间(III)**：直接对正速度施加通道打乱/加噪等操作 → 得到负速度
    - 设计动机：受 SimCLR 分类学启发（空间/几何变换 + 外观变换），但迁移到速度空间。实验表明空间/几何变换（尤其通道打乱）比外观变换更有效——因为结构扰动产生更有信息量的动力学负样本

3. **默认配置与扩展性**:

    - 功能：提供高效默认设置和 CFG 集成方案
    - 核心思路：默认用速度空间通道打乱（Random Channel Shuffle, RCS），$K=1$，$\lambda=0.05$。与 CFG 集成时需要修正冲突——VeCoR 的对比目标可解释为将预测速度远离负轨迹均值，与 CFG 的无条件引导方向可能冲突，采用 ΔFM 的修正策略
    - 设计动机：通道打乱改变了特征通道间的结构对应关系，产生结构上错误但大致合理的速度方向，是最有效且开销最低的负样本

### 训练策略
- 完全即插即用：不需额外数据或架构修改
- 仅增加一个负速度计算开销，与标准 FM 训练相比额外开销极小
- 支持任意 FM 变体（SiT、REPA-SiT、MMDiT）

## 实验关键数据

### 主实验 — ImageNet-1K 256×256（50 NFEs）

| 模型 | FID↓ | IS↑ | sFID↓ | Prec.↑ | Rec.↑ |
|------|------|-----|-------|--------|-------|
| SiT-XL/2 | 20.01 | 74.15 | 8.45 | 0.63 | 0.63 |
| +ΔFM | 16.32 | 78.07 | 5.08 | 0.66 | 0.63 |
| **+VeCoR** | **15.56** | **80.96** | **4.70** | **0.67** | 0.62 |
| REPA-SiT-XL/2 | 11.14 | 115.83 | 8.25 | 0.67 | 0.65 |
| **+VeCoR** | **7.28** | **127.90** | **5.17** | **0.71** | 0.64 |

### MS-COCO T2I 实验

| 方法 | ODE (Heun) CFG=2.0 | SDE (E-M) CFG=2.0 |
|------|---------------------|---------------------|
| M+R (baseline) | 5.03 | 6.03 |
| +ΔFM | 5.16 | 4.78 |
| **+VeCoR (RCR)** | **4.82** | **4.55** |

### 消融实验 — 扰动空间与操作类型

| 扰动空间 | 最佳操作 | FID (SiT-S/2) |
|---------|---------|--------------|
| 速度空间 | 通道打乱 | **55.13** (baseline 64.26) |
| 潜空间 | 随机裁剪 | ~57 |
| 图像空间 | 随机裁剪 | ~58 |

### 关键发现
- 在小模型上增益最大（SiT-S/2: FID 64→55，相对-14%），说明对容量受限模型尤为有效
- 空间/几何扰动比外观扰动效果好——颜色抖动等仅引入浅层变化，不足以提供有效的动力学对比信号
- $K=2$ 负样本是最佳数量，再多则边际收益递减
- 与 CFG 结合后达到 FID=1.94 的 SOTA，证明 VeCoR 学到了更鲁棒的速度场

## 亮点与洞察
- **将对比学习思想迁移到速度场空间**：不是在表示空间做对比（如 SimCLR），而是在 ODE 速度场上做对比——学到的流更稳定、更紧凑地贴合数据流形
- **极简设计**：默认只需一个通道打乱操作作为负样本、一个超参 $\lambda=0.05$，就能带来显著且一致的提升
- **与 ΔFM 的互补性**：ΔFM 关注条件间语义判别力，VeCoR 关注单条轨迹的几何稳定性——两者互补

## 局限与展望
- Recall 略有下降（0.63→0.62），对比排斥可能轻微限制了生成多样性
- 当前负样本均为启发式增强，可探索学习式负样本挖掘
- 仅在 ImageNet 和 COCO 上验证，缺少高分辨率（512+）和视频生成实验
- $\lambda$ 固定为 0.05，可能在不同数据集和模型规模下需要调整

## 相关工作与启发
- **vs ΔFM (Contrastive FM)**: ΔFM 在条件间做对比以增强语义判别性；VeCoR 在轨迹内做对比以增强几何稳定性——互补关系
- **vs REPA**: REPA 通过表示对齐加速收敛；VeCoR 在 REPA 基础上进一步提升（11.14→7.28 FID），说明速度场正则化和表示对齐正交
- **可迁移性**：VeCoR 框架可推广到任何基于 ODE/SDE 的生成模型——扩散模型、连续归一化流等

## 评分
- 新颖性: ⭐⭐⭐⭐ 将对比学习引入速度场的想法新颖但直觉上自然，增强式负样本设计实用
- 实验充分度: ⭐⭐⭐⭐⭐ 多种backbone、多种规模、消融详尽、包含 T2I 和 CFG 组合
- 写作质量: ⭐⭐⭐⭐ 动机分析清晰，图示直观，公式推导严谨
- 价值: ⭐⭐⭐⭐ 通用即插即用方法，对 FM 社区有广泛适用性

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Contrastive Flow Matching (ΔFM)](../../ICCV2025/image_generation/contrastive_flow_matching.md)
- [\[CVPR 2026\] COT-FM: Cluster-wise Optimal Transport Flow Matching](cot-fm_cluster-wise_optimal_transport_flow_matching.md)
- [\[CVPR 2026\] EgoFlow: Gradient-Guided Flow Matching for Egocentric 6DoF Object Motion Generation](egoflow_gradient-guided_flow_matching_for_egocentric_6dof_object_motion_generati.md)
- [\[CVPR 2026\] LeapAlign: Post-Training Flow Matching Models at Any Generation Step by Building Two-Step Trajectories](leapalign_post_training_flow_matching_models_at_any_generation_step.md)
- [\[CVPR 2026\] Frequency-Aware Flow Matching for High-Quality Image Generation](freqflow_frequency_aware_flow_matching.md)

<!-- RELATED:END -->
