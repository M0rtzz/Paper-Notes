---
title: >-
  [论文解读] Homogeneous Dynamics Space for Heterogeneous Humans
description: >-
  [CVPR 2025][人体理解][人体动力学] 本文提出 HDyS（Homogeneous Dynamics Space），通过聚合来自生物力学和强化学习的异构人体运动数据，训练一个同构潜空间来统一不同运动学和动力学表征，实现了从运动学到动力学的高质量双向映射，并在逆动力学估计、地面反力预测等下游任务上展现了有效性。
tags:
  - CVPR 2025
  - 人体理解
  - 人体动力学
  - 异构表征统一
  - 逆动力学
  - 前向动力学
  - 跨域对齐
---

# Homogeneous Dynamics Space for Heterogeneous Humans

**会议**: CVPR 2025  
**arXiv**: [2412.06146](https://arxiv.org/abs/2412.06146)  
**代码**: https://foruck.github.io/HDyS  
**领域**: 人体理解  
**关键词**: 人体动力学, 异构表征统一, 逆动力学, 前向动力学, 跨域对齐

## 一句话总结

本文提出 HDyS（Homogeneous Dynamics Space），通过聚合来自生物力学和强化学习的异构人体运动数据，训练一个同构潜空间来统一不同运动学和动力学表征，实现了从运动学到动力学的高质量双向映射，并在逆动力学估计、地面反力预测等下游任务上展现了有效性。

## 研究背景与动机

1. **领域现状**：计算机视觉在人体运动学方面已取得巨大进展（人体重建、动作识别、运动生成等），但人体动力学——即运动的产生机制（关节力矩、肌肉激活、肌电信号等）——仍然研究不足。

2. **现有痛点**：理解人体动力学面临双重异构性问题。(1) 表征异构：运动学有 markers、骨架关键点、关节角度、SMPL 参数等不同表示，动力学有关节力矩、肌肉动作、表面肌电（sEMG）等层次化表示，且不同表征之间转换困难。(2) 域异构：生物力学数据来自优化求解（高质量但动作简单、仅限实验室），强化学习数据来自物理仿真（动作丰富但存在 sim-to-real gap），两者的运动学模板和数据格式完全不同。

3. **核心矛盾**：现有数据源各有优缺点但无法互通——不同表征格式阻碍了数据聚合，不同域之间存在显著差异使得直接迁移效果很差。

4. **本文目标** 在异构人体运动表征和数据之间挖掘同构性，构建一个统一的潜空间实现运动学↔动力学的双向映射。

5. **切入角度**：虽然表面上表征和数据源各不相同，但底层描述的都是同一个事实——人体运动。笛卡尔坐标系的运动学表征（如 markers、关键点）在不同体系间差异较小，而不同层次的动力学表征（力矩、肌肉、肌电）虽不可直接转换，但共享相似的运动协调模式。

6. **核心 idea**：聚合异构数据 + 逆/前向动力学自编码器 + 对比对齐损失 = 同构潜空间。

## 方法详解

### 整体框架

HDyS 是由多个自编码器组成的聚合架构，对应逆动力学和前向动力学两个方向。输入为四种运动学表征（markers、骨架关键点、Rajagopal 关节角度、SMPL 参数）中的任意一种或多种，输出为多种动力学表征（关节力矩、肌肉激活、sEMG）。所有编码器的输出共享同一个 128 维潜空间，通过重建损失和对比对齐损失进行训练。

### 关键设计

1. **逆动力学自编码器（IDAE）**:

    - 功能：从运动学编码到潜空间，再从潜空间解码出动力学
    - 核心思路：对 markers 和骨架关键点使用 3 层 Transformer 编码器（无位置编码，以适应不同数量的 markers/points），对关节角度和 SMPL 参数使用 3 层 MLP 编码器，统一输出 128 维潜向量。解码器是一个共享的 Transformer 处理连续帧的时间上下文，然后接独立的 MLP 回归头分别解码不同类型的动力学输出（Rajagopal 力矩、SMPL 力矩、肌肉激活、sEMG）。
    - 设计动机：Transformer 编码器可以处理任意数量的 markers/points，实现了对不同数据集的统一处理。共享 Transformer 解码器让不同运动学输入的潜空间被推向同构，而独立的 MLP 头则保留了不同动力学表征的特异性。

2. **前向动力学自编码器（FDAE）**:

    - 功能：从动力学 + 运动学（不含加速度）编码到潜空间，再解码出运动学加速度
    - 核心思路：与 IDAE 类似的编码器分别编码运动学（去掉加速度分量）和动力学。运动学潜向量与动力学潜向量拼接后经共享 MLP composer 融合，再用独立 MLP 解码器预测骨架关键点加速度、SMPL 加速度和关节角加速度。这对应了物理中的牛顿第二定律：知道当前状态和力就能求加速度。
    - 设计动机：前向动力学自编码器迫使潜空间同时保留运动学和动力学信息，增强了潜空间的物理一致性。没有 FDAE 的消融实验显示了其对部分任务的正面贡献。

3. **对比对齐损失**:

    - 功能：将同一帧不同表征的潜向量拉近，不同帧的潜向量推远
    - 核心思路：使用 InfoNCE 损失，对一个 batch 中所有来自同一帧的潜向量（无论来自哪种运动学/动力学编码器）进行正对配对，不同帧的潜向量作为负样本。总损失为 $\mathcal{L} = \alpha_1 L_{recon} + \alpha_2 L_{align}$，其中重建损失为 L1 损失。
    - 设计动机：仅靠重建损失无法保证不同编码器的潜空间对齐（各自可能学到独立的表示）。对比对齐损失是实现"同构"的关键——确保无论输入何种表征，相同运动帧映射到潜空间中相近的位置。

### 损失函数 / 训练策略

总损失 $\mathcal{L} = 0.01 \cdot L_{recon} + 0.05 \cdot L_{align}$。使用 AdamW 优化器，学习率 1e-3，batch size 9600 帧，训练 1000 epochs。为消除不同数据集规模影响，采用平衡采样策略：每个 epoch 从每个数据集随机采样 3000 个序列。

## 实验关键数据

### 主实验

**逆动力学性能（表1）**：

| 数据集 | 指标 | HDyS (avg/best) | 单数据集训练 | 之前SOTA |
|--------|------|-----------------|-------------|----------|
| ImDy | mPJE↓ | 0.5765/0.4674 | 0.6854/0.5403 | 0.6300 |
| AddBiomechanics | mPJE↓ | 0.1189/0.1243 | 0.1695/0.1691 | 0.1626 |
| MinT | RMSE↓ | 0.0614/0.0615 | 0.0637/0.0640 | - |
| MiA | RMSE↓ | 11.8/11.6 | 13.6/13.5 | 13.3 |

### 消融实验

| 配置 | ImDy mPJE↓ | AddBio mPJE↓ | MiA RMSE↓ |
|------|-----------|-------------|-----------|
| Full HDyS | 0.5765 | 0.1189 | 11.8 |
| w/o 对齐损失 | 0.6575 | 0.1270 | 13.7 |
| w/o FDAE | 0.5776 | 0.1198 | 13.6 |
| w/o AMASS | 0.5797 | 0.1217 | 14.7 |
| 32维潜空间 | 0.7390 | 0.1401 | 16.7 |

**规模 vs 异构性分解（表2）**：

| 配置 | AddBio mPJE↓ | MiA RMSE↓ |
|------|-------------|-----------|
| 单数据集-50% | 0.1707 | 16.2 |
| 50%目标+50%异构 | 0.1284 | 14.5 |
| 单数据集-100% | 0.1695 | 13.5 |

### 关键发现

- 聚合异构数据集始终优于单数据集训练，验证了异构数据中存在同构动力学知识。
- 对齐损失是最关键组件：去掉后 ImDy 上 mPJE 从 0.5765 上升到 0.6575，证明了对比学习对同构空间的重要性。
- 相似动力学表征的数据集更互补——肌肉相关数据集（MiA 和 MinT）互相受益更大，力矩相关数据集（AddBiomechanics 和 ImDy）互相受益更大。
- 50%目标数据+50%异构数据甚至可以优于 100% 目标数据的单数据集训练，说明异构性带来的多样性价值可能超过同类数据的规模增加。
- AMASS 虽然不含动力学标签，但其高质量和多样的运动学数据仍对逆动力学有正面贡献。

## 亮点与洞察

- **"异构中挖同构"的哲学**：不同社区的数据格式虽然不同，但描述的是同一个物理现实。这种思想可以迁移到其他多源异构数据融合场景，如多模态医学数据（CT/MRI/超声）的统一表征。
- **逆-前向动力学双向自编码器**：不仅学习从运动到力的映射，还要从力回推加速度，形成一个物理一致的闭环。这种双向训练策略使潜空间更具物理意义。
- **规模 vs 异构性的精巧实验设计（表2）**：通过 50/50 分割实验巧妙地分解了数据规模和数据异构性的各自贡献，是一个值得借鉴的实验范式。

## 局限与展望

- 作者承认了 sim-to-real gap 的存在——RL 仿真数据（ImDy）和真实生物力学数据之间仍有域差异。
- 当前只处理下半身（Rajagopal 模型只用了 23 个关节），全身动力学建模仍是挑战。
- sEMG 数据噪声大且个体差异显著，模型泛化到新受试者的能力未充分验证。
- 潜空间维度固定为 128，未探索自适应维度或层次化潜空间。
- 未考虑外力（如地面反力）作为显式输入，可能限制了在需要外力信息的场景中的应用。

## 相关工作与启发

- **vs ImDyS**: ImDyS 只用 RL 仿真数据学习逆动力学，受限于 sim-to-real gap。HDyS 通过聚合真实生物力学数据弥补了这一缺陷，mPJE 从 0.63 降到 0.47。
- **vs AddBiomechanics**: AddBiomechanics 聚合了大量生物力学数据集但只能处理步态等简单动作。HDyS 通过引入 RL 数据扩展了动作覆盖范围。
- **vs MiA**: MiA 直接从视频/运动捕捉预测 sEMG，但数据规模和动作多样性有限。HDyS 利用跨域数据提升了泛化能力，RMSE 从 13.3 降到 11.6。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次系统地分析并统一人体运动的多源异构数据，逆-前向自编码器加对比对齐的设计思路新颖且有物理动机
- 实验充分度: ⭐⭐⭐⭐⭐ 四个数据集覆盖全部动力学层次，消融实验细致（组件、数据源、维度、规模/异构性分解）
- 写作质量: ⭐⭐⭐⭐ 问题分析清晰，但涉及大量生物力学背景知识，门槛较高
- 价值: ⭐⭐⭐⭐ 对人体动力学领域有基础性贡献，但应用场景相对专业

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] A Generalizable Physics-Enhanced State Space Model for Long-Term Dynamics Forecasting in Complex Environments](../../ICML2025/human_understanding/a_generalizable_physics-enhanced_state_space_model_for_long-term_dynamics_foreca.md)
- [\[CVPR 2025\] HOP: Heterogeneous Topology-based Multimodal Entanglement for Co-Speech Gesture Generation](hop_heterogeneous_topology-based_multimodal_entanglement_for_co-speech_gesture_g.md)
- [\[NeurIPS 2025\] BEDLAM2.0: Synthetic Humans and Cameras in Motion](../../NeurIPS2025/human_understanding/bedlam20_synthetic_humans_and_cameras_in_motion.md)
- [\[CVPR 2025\] Learning Phase Distortion with Selective State Space Models for Video Turbulence Mitigation](learning_phase_distortion_with_selective_state_space_models_for_video_turbulence.md)
- [\[ICML 2025\] Reactivation: Empirical NTK Dynamics Under Task Shifts](../../ICML2025/human_understanding/reactivation_empirical_ntk_dynamics_under_task_shifts.md)

<!-- RELATED:END -->
