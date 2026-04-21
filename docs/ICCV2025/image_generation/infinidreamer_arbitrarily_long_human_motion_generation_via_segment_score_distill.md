---
title: >-
  [论文解读] InfiniDreamer: Arbitrarily Long Human Motion Generation via Segment Score Distillation
description: >-
  [ICCV 2025][图像生成][长序列运动生成] InfiniDreamer 通过将预训练的短序列运动扩散模型作为先验，提出 Segment Score Distillation (SSD) 优化方法，对粗初始化的长运动序列中的重叠短片段进行迭代优化，实现了无需额外长序列训练数据的任意长度人体运动生成。
tags:
  - ICCV 2025
  - 图像生成
  - 长序列运动生成
  - Score Distillation
  - 滑动窗口
  - 免训练
  - 运动扩散模型
---

# InfiniDreamer: Arbitrarily Long Human Motion Generation via Segment Score Distillation

**会议**: ICCV 2025  
**arXiv**: [2411.18303](https://arxiv.org/abs/2411.18303)  
**代码**: 待确认  
**领域**: 人体运动生成 / 扩散模型  
**关键词**: 长序列运动生成, Score Distillation, 滑动窗口, 免训练, 运动扩散模型

## 一句话总结
InfiniDreamer 通过将预训练的短序列运动扩散模型作为先验，提出 Segment Score Distillation (SSD) 优化方法，对粗初始化的长运动序列中的重叠短片段进行迭代优化，实现了无需额外长序列训练数据的任意长度人体运动生成。

## 研究背景与动机

**领域现状**：文本到运动生成（Text-to-Motion）在 MDM、MLD、T2M-GPT 等方法的推动下已经可以生成约 10 秒的高质量短序列运动。然而实际应用（游戏动画、影视制作、AR/VR）通常需要数分钟甚至数小时的连续运动。

**现有痛点**：
   - **数据瓶颈**：高质量长序列运动数据极其稀缺，现有数据集（HumanML3D、BABEL）以短序列为主
   - **自回归方法**（TEACH、Multi-Act）会累积误差，导致运动漂移、重复模式、"冻结"现象
   - **扩散 infilling 方法**（PriorMDM/DoubleTake、DiffCollage）在段落边界处强行修改，容易导致突兀过渡、运动扭曲，甚至覆盖已生成内容

**核心矛盾**：需要生成超出训练数据长度的运动序列，但缺乏长序列监督信号，且现有组合方案在边界处处理粗暴。

**本文目标** 给定一系列文本描述，生成语义一致、过渡自然、任意长度的连续运动序列。

**切入角度**：从 DreamFusion 的 Score Distillation Sampling (SDS) 获得灵感——SDS 的优势在于渐进式、平滑的蒸馏过程，可以在不同视角间保持一致性。将这个优势迁移到时序维度，可以实现长运动的平滑过渡。

**核心 idea**：把长运动序列参数化为可微变量，用滑动窗口采样短片段，通过 Score Distillation 将每个短片段对齐到预训练运动扩散先验的分布，从而实现局部逼真 + 全局连贯。

## 方法详解

### 整体框架
输入是一系列文本提示 $Y = \{y_1, y_2, ..., y_n\}$，输出是长运动序列 $M = \{m_1, t_1, m_2, t_2, ..., m_n\}$，其中 $m_i$ 是对应文本的子运动，$t_i$ 是过渡段。框架包含三个模块：运动序列初始化、运动片段采样、段落 Score Distillation。

### 关键设计

1. **Motion Sequence Initialization（运动序列初始化）**

    - 功能：构建粗糙的长运动序列作为优化起点
    - 核心思路：先随机初始化整个长序列，然后用预训练 MDM 为每个子运动 $m_i$ 生成对应文本 $y_i$ 的短运动片段填充进去。过渡段 $t_i$ 保持随机初始化。相邻区域通过线性插值平滑化
    - 设计动机：给定初始结构+随机过渡段，为后续优化提供合理起点。梯度掩码 $Mask_l = 0.1$（子运动区域更新慢）和 $Mask_h = 0.8$（过渡区域更新快）控制不同区域的优化力度

2. **Motion Segment Sampling（运动片段采样）**

    - 功能：从长序列中迭代采样重叠的短片段供优化
    - 核心思路：使用大小为 $W$ 的滑动窗口以步长 $S$ 沿长序列移动，采样重叠的短片段 $x_0^i$。对于跨越多个子运动的窗口，随机选择其中一个子运动的文本条件（等概率）
    - 设计动机：重叠确保相邻片段之间的连续性，滑动窗口策略使优化覆盖整个序列

3. **Segment Score Distillation (SSD)**

    - 功能：利用预训练运动扩散模型的先验分布优化每个短片段
    - 核心思路：对每个采样片段 $x_0^i$，随机采样时间步 $t$，加噪得到 $x_t^i$，用扩散模型预测去噪结果 $\hat{x}_0^i = \phi(x_t^i; t, \varnothing)$，然后通过对齐损失优化：
    $\mathcal{L}_{align} = \mathbf{E}_{t,\epsilon}[w(t) \|\hat{x}_0^i - x_0^i\|_2^2]$
    - 配合三个几何正则化损失：位置约束 $\mathcal{L}_{pos}$（通过正向运动学保持关节位置准确）、足部接触约束 $\mathcal{L}_{foot}$（防止滑步）、速度正则化 $\mathcal{L}_{vel}$（鼓励平滑过渡）
    - 设计动机：与 infilling 方法在边界进行强修改不同，SSD 是渐进式的全局优化，每次更新都很小，因此不会破坏已有的运动结构。重叠窗口确保过渡段在多次优化中被反复打磨

### 损失函数 / 训练策略
总损失为 $\mathcal{L}_{ssd} = \mathcal{L}_{align} + \lambda_{pos}\mathcal{L}_{pos} + \lambda_{foot}\mathcal{L}_{foot} + \lambda_{vel}\mathcal{L}_{vel}$，其中 HumanML3D 上 $\lambda = 0$，BABEL 上 $\lambda = 0.1$。使用 AdamW 优化器，学习率 0.002，优化 20000 次迭代。窗口大小 $W=120$，步长 $S=30$。

## 实验关键数据

### 主实验 (HumanML3D)

| 指标 | DoubleTake | DiffCollage | **InfiniDreamer** |
|------|-----------|-------------|-------------------|
| R-precision ↑ | 0.603 | 0.605 | **0.679** |
| FID ↓ (Motion) | 1.36 | 1.07 | **0.47** |
| Diversity → | 9.33 | 9.34 | **9.58** |
| MM-Dist ↓ | 4.27 | 3.62 | **3.15** |
| FID ↓ (Transition) | 3.19 | 4.27 | **2.04** |
| Diversity → (Trans) | 8.09 | 7.47 | **8.69** |

### 消融实验 (HumanML3D)

| 配置 | R-precision ↑ | FID ↓ | Trans FID ↓ |
|------|--------------|-------|------------|
| Full model | **0.679** | **0.47** | **2.04** |
| w/o gradient masks | 0.643 | 0.64 | 2.25 |

### BABEL 数据集结果

| 指标 | TEACH | DoubleTake | DiffCollage | **InfiniDreamer** |
|------|-------|-----------|-------------|-------------------|
| R-precision ↑ | 0.461 | 0.483 | 0.487 | **0.543** |
| FID ↓ | 1.43 | 1.14 | 1.83 | **0.97** |
| Trans FID ↓ | 4.23 | 3.54 | 4.62 | **2.07** |

### 关键发现
- InfiniDreamer 在所有指标上全面超越之前的免训练方法，特别是运动 FID 从 1.07 降到 0.47（HumanML3D）
- 过渡段质量大幅提升（Trans FID 从 3.19 降到 2.04），说明 SSD 的渐进式优化比 infilling 更适合过渡生成
- 梯度掩码的消融显示其对子运动质量有重要贡献
- 学习率需要仔细调节：过高导致运动静止（motion lost），过低导致欠训练和运动扭曲
- 窗口大小 $W=120$、步长 $P=30$ 是最佳配置；$P \geq W$ 时过渡段质量急剧下降

## 亮点与洞察
- **将 SDS 从 3D 生成迁移到运动生成**是本文最大的创新点。SDS 的渐进式特性天然适合需要全局一致性的任务——在 3D 中是多视角一致，在运动中是时序一致
- **梯度掩码设计**巧妙地让子运动区域缓慢更新（保持文本对齐）、过渡区域快速更新（从随机初始化快速收敛），实现了差异化优化
- **短序列模型 + 长序列生成的解耦设计**意味着未来更好的短序列运动扩散模型可以直接插入框架中提升性能，无需重新训练

## 局限与展望
- 优化 20000 次迭代耗时较长，效率有待提升
- 仍依赖 MDM 等短序列模型的质量上限，本身不学习新的运动知识
- 几何损失（足部接触等）在某些场景可能不够充分，如复杂交互运动
- 目前仅验证了 MDM 作为先验，尚未探索 MLD 等更强的潜空间扩散模型
- 窗口大小受限于短序列模型的最大上下文（MDM 约 200 帧）

## 相关工作与启发
- **vs DoubleTake/PriorMDM**：使用 infilling 策略在边界做强修改，导致突兀过渡和运动覆盖；InfiniDreamer 的渐进式优化避免了这个问题
- **vs DiffCollage**：同为拼接方法但不做全局优化；InfiniDreamer 通过重叠窗口 + SSD 实现了全局一致性
- **vs TEACH**：自回归生成累积误差导致运动漂移；InfiniDreamer 通过全局优化规避了错误累积
- **vs FlowMDM**：FlowMDM 使用 Blended Positional Encodings 实现无缝组合但需要特殊训练；InfiniDreamer 完全免训练

## 评分
- 新颖性: ⭐⭐⭐⭐ 将 SDS 迁移到运动生成领域是巧妙的跨域创新
- 实验充分度: ⭐⭐⭐⭐ HumanML3D + BABEL 双数据集验证，完善的消融实验
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，SSD 与 SDS 的联系阐述到位
- 价值: ⭐⭐⭐⭐ 免训练长序列运动生成的实用方案，解耦短序列模型与长序列组合为未来改进留出空间

<!-- RELATED:START -->

## 相关论文

- [PINO: Person-Interaction Noise Optimization for Long-Duration and Customizable Motion Generation of Arbitrary-Sized Groups](pino_person-interaction_noise_optimization_for_long-duration_and_customizable_mo.md)
- [A Unified Framework for Motion Reasoning and Generation in Human Interaction](a_unified_framework_for_motion_reasoning_and_generation_in_human_interaction.md)
- [HPSv3: Towards Wide-Spectrum Human Preference Score](hpsv3_towards_wide-spectrum_human_preference_score.md)
- [ScoreHOI: Physically Plausible Reconstruction of Human-Object Interaction via Score-Guided Diffusion](scorehoi_physically_plausible_reconstruction_of_human-object_interaction_via_sco.md)
- [Move-in-2D: 2D-Conditioned Human Motion Generation](../../CVPR2025/image_generation/move-in-2d_2d-conditioned_human_motion_generation.md)

<!-- RELATED:END -->
