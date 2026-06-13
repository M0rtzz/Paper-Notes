---
title: >-
  [论文解读] ScoreHOI: Physically Plausible Reconstruction of Human-Object Interaction via Score-Guided Diffusion
description: >-
  [ICCV 2025][图像生成][人物交互重建] ScoreHOI 利用 score-based 扩散模型作为优化器，结合 DDIM 逆向-正向采样与物理约束（接触、穿透、地面接触）引导去噪过程，并通过接触驱动的迭代细化策略，从单目图像实现物理合理的人体-物体交互三维重建，在 BEHAVE 上接触 F-Score 提升 9%。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "人物交互重建"
  - "扩散模型"
  - "score-guided sampling"
  - "物理约束"
  - "接触预测"
---

# ScoreHOI: Physically Plausible Reconstruction of Human-Object Interaction via Score-Guided Diffusion

**会议**: ICCV 2025  
**arXiv**: [2509.07920](https://arxiv.org/abs/2509.07920)  
**代码**: [https://github.com/RammusLeo/ScoreHOI.git](https://github.com/RammusLeo/ScoreHOI.git)  
**领域**: 图像生成  
**关键词**: 人物交互重建, 扩散模型, score-guided sampling, 物理约束, 接触预测

## 一句话总结
ScoreHOI 利用 score-based 扩散模型作为优化器，结合 DDIM 逆向-正向采样与物理约束（接触、穿透、地面接触）引导去噪过程，并通过接触驱动的迭代细化策略，从单目图像实现物理合理的人体-物体交互三维重建，在 BEHAVE 上接触 F-Score 提升 9%。

## 研究背景与动机

**领域现状**：从单目图像联合重建人体和交互物体的三维网格是一个重要但困难的任务。现有方法主要分两类：(a) 优化方法（如 CHORE）用 Adam 迭代优化物理约束，但过度强调物理约束而忽视图像特征，导致重建偏差大且速度慢；(b) 回归方法（如 CONTHO）通过前向网络单步预测，但单步细化鲁棒性差，尤其在严重遮挡或深度模糊场景下。

**现有痛点**：优化方法缺乏对人物交互的先验知识指导，容易陷入局部最优；回归方法虽然速度快，但单步前向缺乏迭代细化能力，在困难场景下表现不稳定。

**核心矛盾**：如何兼得先验知识驱动和物理约束满足？传统优化器没有数据分布先验，而回归网络缺乏可控的迭代优化机制。

**本文目标** (1) 将人物交互的先验知识融入优化过程；(2) 在采样过程中用物理约束监督生成方向。

**切入角度**：扩散模型通过 score function 描述数据分布的梯度场，支持条件引导采样。可以将扩散模型作为"具有丰富先验的优化器"，在去噪过程中注入物理约束实现有引导的细化。

**核心 idea**：用 score-based 扩散模型替代传统优化器，在 DDIM 采样过程中融入接触、穿透和地面约束作为物理引导，并通过迭代更新接触掩码提升重建的物理合理性。

## 方法详解

### 整体框架
输入为单目 RGB 图像 $I$、人/物分割掩码 $S_h, S_o$ 和物体模板 $P_o$。首先通过 **Affordance-Aware Regressor** 提取图像特征 $\mathcal{F}$ 并粗估 SMPL-H 参数 $\theta, \beta$ 和物体姿态 $R_o, t_o$。然后将这些参数送入 **Contact-Driven Iterative Refinement** 模块：通过 DDIM 逆向将初始估计转化为噪声潜变量，再在正向 DDIM 采样中注入物理约束引导，同时迭代更新接触掩码。最终通过双分支 Transformer 进一步细化人体和物体网格。

### 关键设计

1. **Affordance-Aware Regressor（可供性感知回归器）**:

    - 功能：从输入图像和物体模板粗估人体姿态与物体位姿
    - 核心思路：利用预训练的 PointNeXt 提取物体的 affordance 特征（即物体"可以怎么被使用"的先验），将其注入图像特征提取过程。这样即使物体形状不在训练集中，模型也能通过 affordance 泛化
    - 设计动机：传统方法用类别 ID 注入物体信息，但无法处理训练集外的物体。Affordance 概念提供了跨类别的通用先验

2. **Score-Guided Physical Optimization（分数引导的物理优化）**:

    - 功能：在扩散模型的去噪采样中注入物理约束引导
    - 核心思路：定义优化目标 $\bm{x} = \{\theta, \beta, R_o, t_o\} \in \mathbb{R}^{331}$。先通过 DDIM inversion 将初始估计 $\bm{x}^{\text{init}}$ 映射到噪声空间 $\bm{x}_\tau$，再在 DDIM 采样过程中将条件 score 修改为：$\nabla_{\bm{x}_t}\log p(\bm{x}_t|\bm{c},\mathcal{P}) = \nabla_{\bm{x}_t}\log p(\bm{x}_t|\bm{c}) + \nabla_{\bm{x}_t}\log p(\mathcal{P}|\bm{c},\hat{\bm{x}_0}(\bm{x}_t))$，其中第二项通过去噪后的 $\hat{\bm{x}_0}$ 近似计算物理约束损失的梯度
    - 修改后的噪声预测：$\epsilon'_\phi = \epsilon_\phi(\bm{x}_t, t, \bm{c}) + \rho\sqrt{1-\alpha_t}\nabla_{\bm{x}_t}L_\mathcal{P}$
    - 设计动机：直接在噪声空间计算物理约束梯度困难，通过 Tweedie 公式用 $\hat{\bm{x}_0}$ 近似使约束可计算

3. **物理约束损失函数**:

    - 总损失：$L_\mathcal{P} = \lambda_{ho}L_{ho} + \lambda_{of}L_{of} + \lambda_{pt}L_{pt}$
    - **人-物接触** $L_{ho}$：接触区域上人体和物体顶点的欧氏距离应为零
    - **物-地接触** $L_{of}$：物体底部接触顶点的高度应为零
    - **穿透避免** $L_{pt}$：利用物体的 SDF 函数惩罚人体顶点穿透物体的情况

4. **Contact-Driven Iterative Refinement（接触驱动迭代细化）**:

    - 功能：迭代更新接触掩码以提高接触预测精度
    - 核心思路：在每次迭代 $n$ 中：(1) 根据当前参数 $\bm{x}_0^n$ 从图像特征采样人/物特征 $\mathcal{F}_h, \mathcal{F}_o$；(2) 更新接触掩码 $\mathbf{M}_h, \mathbf{M}_o, \mathbf{M}_f$；(3) 执行 DDIM inversion + guided sampling 得到 $\bm{x}_0^{n+1}$。迭代 $N=10$ 次
    - 设计动机：单次前向预测接触掩码容易出错（特别是在严重遮挡下），迭代更新可以让接触预测和姿态优化相互促进

5. **IG-Adapter（图像-几何适配器）**:

    - 功能：为扩散模型注入图像观察和物体几何先验
    - 核心思路：引入额外的 cross-attention 块和线性融合头，将图像特征条件 $\bm{c}_I$（来自 $\mathcal{F}$ 的平均池化）和几何特征条件 $\bm{c}_G$（来自预训练 PointNeXt）融合
    - 训练目标：$L_{DM} = \mathbb{E}_{\bm{x}_0,\epsilon,t,\bm{c}_I,\bm{c}_G}\|\epsilon - \epsilon_\theta(\bm{x}_t, t, \bm{c}_I, \bm{c}_G)\|^2$

### 训练策略
训练分两阶段：(1) 训练图像骨干、接触预测器和顶点优化模块，50 epochs，LR $10^{-4}$；(2) 冻结图像骨干，训练扩散模型（加入 IMHD 数据集增强生成能力），30 epochs。使用 4 块 RTX 4090 训练约 1.5 天。

## 实验关键数据

### 主实验

| 数据集 | 方法 | CD_human↓ | CD_object↓ | Contact_prec↑ | Contact_recall↑ | Contact_F-S↑ |
|--------|------|-----------|-----------|--------------|----------------|-------------|
| BEHAVE | PHOSA | 12.17 | 26.62 | 0.393 | 0.266 | 0.317 |
| BEHAVE | CHORE | 5.58 | 10.66 | 0.587 | 0.472 | 0.523 |
| BEHAVE | CONTHO | 4.99 | 8.42 | 0.628 | 0.496 | 0.554 |
| BEHAVE | **ScoreHOI** | **4.85** | **7.86** | **0.634** | **0.586** | **0.609** |
| InterCap | PHOSA | 11.20 | 20.57 | 0.228 | 0.159 | 0.187 |
| InterCap | CHORE | 7.01 | 12.81 | 0.339 | 0.253 | 0.290 |
| InterCap | CONTHO | 5.96 | 9.50 | 0.661 | 0.432 | 0.522 |
| InterCap | **ScoreHOI** | **5.56** | **8.75** | 0.627 | **0.590** | **0.578** |

在 BEHAVE 上，ScoreHOI 的接触 F-Score 达到 0.609，比 CONTHO 提升约 **9%**。

### 消融实验

| 配置 | CD_human↓ | CD_object↓ | Contact_F-S↑ | 说明 |
|------|-----------|-----------|-------------|------|
| w/o diffusion | 5.03 | 8.48 | 0.588 | 去掉扩散模块 |
| w/o CDIR | 4.93 | 7.98 | 0.577 | 去掉迭代细化 |
| No condition | 4.94 | 8.23 | 0.585 | 无条件引导 |
| w/o $\bm{c}_G$ | 4.87 | 7.99 | 0.591 | 去掉几何条件 |
| w/o $\bm{c}_I$ | 4.88 | 8.03 | 0.597 | 去掉图像条件 |
| No guidance | 4.93 | 8.01 | 0.570 | 无物理引导 |
| w/o $L_{ho}$ | 4.87 | 7.95 | 0.574 | 去掉人物接触 |
| w/o $L_{pt}$ | 4.87 | 7.93 | 0.592 | 去掉穿透约束 |
| w/o $L_{of}$ | 4.89 | 7.95 | 0.602 | 去掉地面接触 |
| **Full model** | **4.85** | **7.86** | **0.609** | 完整模型 |

### 效率对比

| 方法 | CD_human↓ | CD_object↓ | FPS↑ |
|------|-----------|-----------|------|
| CHORE | 5.58 | 10.66 | 0.0035 |
| VisTracker | 5.24 | 7.89 | 0.0359 |
| ScoreHOI-Faster(N=2) | 4.87 | 7.95 | 2.008 |
| ScoreHOI | 4.85 | 7.86 | 0.290 |

### 关键发现
- **扩散模型贡献显著**：去掉扩散模块后 Contact_F-S 从 0.609 降到 0.588，说明扩散先验提供了有价值的分布知识
- **CDIR 是关键组件**：移除迭代细化后 F-Score 从 0.609 降到 0.577，证明迭代更新接触掩码对提升接触质量至关重要
- **人-物接触约束最关键**：去掉 $L_{ho}$ 后 recall 大幅下降，说明显式的接触引导是必要的
- **效率优势明显**：ScoreHOI 比 CHORE 快约 80 倍，且性能更优。快速版本（N=2）实现 2 FPS 实时级推理，仅有微小性能损失

## 亮点与洞察
- **扩散模型作为优化器**是一个巧妙的视角转换：不是用扩散模型生成，而是利用它学到的数据分布先验来指导优化过程。这一思路可迁移到任何需要"带先验的优化"的重建任务
- **DDIM inversion + guided sampling** 的组合允许从任意初始点出发，在保持先验知识的同时注入任务特定约束，比直接从噪声采样更适合细化任务
- **接触驱动迭代细化**体现了"预测-更新"的循环设计哲学，接触掩码和姿态参数形成了相互促进的闭环

## 局限性
- 需要预定义的物体 canonical pose 模板，无法处理未见过的物体
- 训练数据有限（BEHAVE 仅 20 类物体），泛化能力受限
- 迭代 10 次×DDIM 采样的推理开销仍然较大（0.29 FPS）

## 相关工作与启发
- **vs CONTHO**: CONTHO 用回归式交叉注意力单步细化，ScoreHOI 用扩散模型多步迭代细化，在接触质量上显著更优
- **vs CHORE**: CHORE 用 Adam 优化器+物理约束，ScoreHOI 用扩散先验替代 Adam，速度快 80 倍且精度更高
- **vs ScoreMDM/ScoreHMR**: 这些工作将扩散模型用于 HMR，ScoreHOI 首次扩展到人物交互场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 扩散模型作为带先验的优化器思路新颖，但 score-guided sampling 本身不算全新
- 实验充分度: ⭐⭐⭐⭐⭐ 消融全面（模块/条件/引导/超参数），多数据集验证
- 写作质量: ⭐⭐⭐⭐ 条理清晰，公式推导详细
- 价值: ⭐⭐⭐⭐ 对 HOI 重建领域有实际推进，方法论可迁移

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] HOI-IDiff: An Image-like Diffusion Method for Human-Object Interaction Detection](../../CVPR2025/image_generation/an_image-like_diffusion_method_for_human-object_interaction_detection.md)
- [\[CVPR 2026\] ViHOI: Human-Object Interaction Synthesis with Visual Priors](../../CVPR2026/image_generation/vihoi_human-object_interaction_synthesis_with_visual_priors.md)
- [\[ICCV 2025\] A Unified Framework for Motion Reasoning and Generation in Human Interaction](a_unified_framework_for_motion_reasoning_and_generation_in_human_interaction.md)
- [\[ICCV 2025\] HPSv3: Towards Wide-Spectrum Human Preference Score](hpsv3_towards_wide-spectrum_human_preference_score.md)
- [\[ICLR 2026\] Does FLUX Already Know How to Perform Physically Plausible Image Composition?](../../ICLR2026/image_generation/does_flux_already_know_how_to_perform_physically_plausible_image_composition.md)

</div>

<!-- RELATED:END -->
