---
title: >-
  [论文解读] HUMOS: Human Motion Model Conditioned on Body Shape
description: >-
  [ECCV 2024][人体理解][人体运动生成] 提出 HUMOS，一种基于体型条件化的人体运动生成模型，通过循环一致性损失和可微分的直觉物理/动态稳定性约束，在无配对训练数据的情况下学习体型与运动之间的相关性，生成物理可信且动态稳定的人体运动。
tags:
  - ECCV 2024
  - 人体理解
  - 人体运动生成
  - 体型条件化
  - 循环一致性
  - 动态稳定性
  - 运动重定向
---

# HUMOS: Human Motion Model Conditioned on Body Shape

**会议**: ECCV 2024  
**arXiv**: [2409.03944](https://arxiv.org/abs/2409.03944)  
**代码**: [GitHub](https://github.com/CarstenEpic/humos)  
**领域**: 人体理解  
**关键词**: 人体运动生成, 体型条件化, 循环一致性, 动态稳定性, 运动重定向

## 一句话总结

提出 HUMOS，一种基于体型条件化的人体运动生成模型，通过循环一致性损失和可微分的直觉物理/动态稳定性约束，在无配对训练数据的情况下学习体型与运动之间的相关性，生成物理可信且动态稳定的人体运动。

## 研究背景与动机

**领域现状**：人体运动生成在游戏、AR/VR、机器人仿真中至关重要。现有SOTA方法（如TEMOS、MDM）已经能够根据文本或动作标签生成逼真的人体运动。

**现有痛点**：现有运动模型几乎**完全忽略了体型差异**，通常对所有训练数据做归一化处理，使用标准化的平均体型进行训练和生成。然而，不同体型（肌肉分布、四肢比例）的人在执行相同动作时，运动模式是截然不同的。

**核心矛盾**：要训练体型条件化的运动模型，需要**配对数据**（不同体型的人做相同动作），但在AMASS等现有数据集中，这样的配对数据极其稀少。

**本文目标**：如何在缺乏配对数据的情况下，训练一个能根据输入体型生成对应运动的生成模型。

**切入角度**：从无配对图像翻译（CycleGAN）中获得灵感，利用循环一致性实现自监督训练，同时引入可微分的物理约束防止平凡解。

**核心idea**：编码器将运动编码到**体型无关的隐空间**，解码器接收隐编码和目标体型参数生成新运动；循环一致性确保语义不变，物理约束确保物理可信。

## 方法详解

### 整体框架

HUMOS 是一个基于 Transformer 的条件变分自编码器（c-VAE）。输入为身份 $\mathcal{A}$ 的运动 $M_\mathcal{A}$ 及其身份特征 $\mathcal{I}_\mathcal{A} = (\beta_\mathcal{A}, \mathcal{G}_\mathcal{A})$（体型参数+性别），编码器输出体型无关的隐编码 $z_{M_\mathcal{A}}$。解码器接收 $z_{M_\mathcal{A}}$ 和随机采样的目标身份 $\mathcal{I}_\mathcal{B}$，生成重定向运动 $\hat{M}_{\mathcal{A} \to \mathcal{B}}$。模型以 SMPL mesh 表示运动，使用6D旋转表示和根关节相对旋转。

### 关键设计

1. **自监督循环一致性训练**：

    - 功能：在缺乏配对数据的情况下实现体型条件化训练
    - 核心思路：运动先被编码为体型无关的隐编码 $z_{M_\mathcal{A}} = \mathcal{E}(M_\mathcal{A}, \mathcal{I}_\mathcal{A})$，然后用目标身份解码得到 $\hat{M}_{\mathcal{A}\to\mathcal{B}} = \mathcal{D}(z_{M_\mathcal{A}}, \mathcal{I}_\mathcal{B})$。再将此运动用身份 $\mathcal{B}$ 重新编码、用原始身份 $\mathcal{A}$ 解码，得到 $\hat{M}_{\mathcal{A}\to\mathcal{A}}$，要求其与原始运动 $M_\mathcal{A}$ 一致：
    $\mathcal{L}_{\text{cycle}} = \mathcal{L}_{\text{rot}} + \mathcal{L}_{\text{pos}}$
   其中 $\mathcal{L}_{\text{rot}}$ 计算旋转矩阵的测地距离，$\mathcal{L}_{\text{pos}}$ 为根关节位置的 smooth L1 损失。
    - 设计动机：借鉴 CycleGAN 思想，在无配对数据条件下通过循环重建实现自监督学习。

2. **直觉物理（IP）约束**：

    - 功能：防止循环一致性训练中出现平凡解（直接复制源运动到目标体型）
    - 核心思路：三个可微分物理损失：
        - $\mathcal{L}_{\text{penetrate}}$：最低顶点低于地面的距离惩罚
        - $\mathcal{L}_{\text{float}}$：最低顶点高于地面的距离惩罚
        - $\mathcal{L}_{\text{slide}}$：接触地面的脚部关节的水平速度惩罚
    $\mathcal{L}_{\text{physics}} = \mathcal{L}_{\text{penetrate}} + \mathcal{L}_{\text{float}} + \mathcal{L}_{\text{slide}}$
    - 设计动机：如果网络简单地将源运动 $M_\mathcal{A}$ 复制到目标体型 $\mathcal{B}$，由于体型差异会导致地面穿透、悬浮和脚部滑动等物理不合理现象。这些物理约束迫使网络必须根据目标体型调整运动。

3. **动态稳定性项（ZMP）**：

    - 功能：确保生成的动态运动序列满足生物力学上的动态稳定性
    - 核心思路：基于零力矩点（Zero Moment Point, ZMP）概念。ZMP 是地面上地面反力水平分量力矩为零的点。当 ZMP 位于支撑基底（BoS）内时，运动被认为是动态稳定的。ZMP 计算公式为：
    $\mathcal{Z} = \mathcal{C}_m - \frac{n \times \mathcal{M}_{\mathcal{C}_m}^{gi}}{\mathcal{F}^{gi} \cdot n}$
   其中 $\mathcal{C}_m$ 为质心在地面的投影，$\mathcal{F}^{gi} = mg - ma_\mathcal{G}$ 为惯性力，$\mathcal{M}_{\mathcal{C}_m}^{gi}$ 为绕地面投影质心的力矩。动态稳定性损失为 ZMP 与压力中心（CoP）之间的距离：
    $\mathcal{L}_{\text{dyn}} = \rho(\|\mathcal{C}_P - \mathcal{Z}\|_2)$
   其中 $\rho$ 为 Geman-McClure 鲁棒惩罚函数。总体质量通过 SMPL mesh 体积估算，各顶点的质量按其所属身体部位的体积比例分配。
    - 设计动机：静态稳定性（IPMAN方法）仅适用于静止姿态，人类运动本质上是高度动态的，需要考虑加速度、角动量等动态因素。ZMP 广泛用于机器人平衡控制。

### 损失函数 / 训练策略

总损失为各项的加权和：

$$\mathcal{L} = \lambda_{\text{cycle}}\mathcal{L}_{\text{cycle}} + \lambda_{\text{physics}}\mathcal{L}_{\text{physics}} + \lambda_{\text{dyn}}\mathcal{L}_{\text{dyn}} + \lambda_{\text{KL}}\mathcal{L}_{\text{KL}} + \lambda_{\text{E}}\mathcal{L}_{\text{E}}$$

其中 $\lambda_{\text{cycle}}=1$, $\lambda_{\text{physics}}=1$, $\lambda_{\text{dyn}}=10^{-4}$, $\lambda_{\text{KL}}=10^{-5}$, $\lambda_{\text{E}}=10^{-2}$。KL 散度损失 $\mathcal{L}_{\text{KL}}$ 正则化隐空间分布，$\mathcal{L}_{\text{E}}$ 鼓励相同运动在不同身份下的隐编码一致。

训练数据来自 AMASS 数据集（480个身份），以 20fps 采样，T=200 帧。使用 AdamW 优化器，学习率 $10^{-5}$，batch size 60，训练 1300 epochs。编码器和解码器各 6 层 Transformer。

## 实验关键数据

### 主实验

| 方法 | Penetrate(cm)↓ | Float(cm)↓ | Skate(%)↓ | Dyn.Stability(%)↑ | BoSDist(cm)↓ |
|------|----------------|------------|-----------|-------------------|--------------|
| TEMOS-Simple | 6.82 | 6.55 | 27.07 | 45.85 | 16.94 |
| TEMOS-Rokoko | 4.14 | 3.85 | 20.05 | 55.92 | 16.58 |
| TEMOS-Rokoko-G | 0.75 | 4.44 | 20.05 | 55.92 | 16.58 |
| **HUMOS** | **1.23** | **1.04** | **7.37** | **71.9** | **14.62** |

### 消融实验

| 配置 | Penetrate↓ | Float↓ | Skate(%)↓ | Dyn.Stability(%)↑ | BoSDist↓ |
|------|-----------|--------|-----------|-------------------|----------|
| $\mathcal{L}_{\text{cycle}}$ only | 2.74 | 2.62 | 15.04 | 64.00 | 16.96 |
| + $\mathcal{L}_{\text{physics}}$ | 1.55 | 1.44 | 7.93 | 67.82 | 16.41 |
| + $\mathcal{L}_{\text{dyn}}$ (完整) | **1.23** | **1.04** | **7.37** | **71.9** | **14.62** |

### 关键发现

- 仅使用循环一致性相比 TEMOS-Rokoko 基线就能带来 ~33% 的穿透改善和 ~25% 的滑动改善
- 物理约束带来了脚部滑动最大的改善（~47%），说明 foot skating 是体型不匹配时最严重的问题
- 动态稳定性项使稳定帧比例从 67.82% 提升到 71.9%，同时改善了所有其他指标
- 感知研究中，HUMOS 获得 3.64/5 的评分，显著优于 TEMOS-Rokoko 的 3.25/5

## 亮点与洞察

- **巧妙避开配对数据需求**：循环一致性 + 物理约束的组合非常优雅——前者提供自监督信号，后者防止平凡解，两者缺一不可
- **可微分的动态稳定性**：将机器人领域的 ZMP 概念引入到数据驱动的运动生成中，且完全可微分，可与神经网络端到端训练
- **实用价值高**：直接实现了角色间运动重定向，无需传统的两步pipeline（生成+重定向）

## 局限与展望

- 不同体型的运动风格差异仍然较为subtle，可能受限于训练集的体型多样性
- 未处理运动中的自穿透问题
- 仅考虑体型条件化，未考虑情感状态、生理障碍等运动风格因素
- 生成结果仍存在一些运动伪影

## 相关工作与启发

- **vs TEMOS**: TEMOS 生成标准体型运动，需额外重定向步骤；HUMOS 直接生成体型适配的运动
- **vs 物理仿真方法 (RL-based)**: RL方法物理可信但计算昂贵、多样性差；HUMOS 用可微分物理项在数据驱动框架中实现物理约束
- **vs CycleGAN**: 借鉴了循环一致性思想，但创新地将其应用到运动-体型空间，并用物理约束替代对抗训练

## 评分

- 新颖性: ⭐⭐⭐⭐ 循环一致性+物理约束的组合在运动领域是新颖的，动态ZMP约束也是首次引入
- 实验充分度: ⭐⭐⭐⭐ 物理指标全面，包含感知研究，但缺少与更多最新方法的比较
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑链条清晰，公式推导完整，图示直观
- 价值: ⭐⭐⭐⭐ 为体型感知运动生成提供了可行方案，对游戏动画和虚拟人领域有实际应用价值

<!-- RELATED:START -->

## 相关论文

- [Large Motion Model for Unified Multi-Modal Motion Generation](large_motion_model_for_unified_multi-modal_motion_generation.md)
- [Modeling and Driving Human Body Soundfields through Acoustic Primitives](modeling_and_driving_human_body_soundfields_through_acoustic_primitives.md)
- [Motion Mamba: Efficient and Long Sequence Motion Generation](motion_mamba_efficient_and_long_sequence_motion_generation.md)
- [GENMO: A GENeralist Model for Human MOtion](../../ICCV2025/human_understanding/genmo_a_generalist_model_for_human_motion.md)
- [GenM3: Generative Pretrained Multi-path Motion Model for Text Conditional Human Motion Generation](../../ICCV2025/human_understanding/genm3_generative_pretrained_multi-path_motion_model_for_text_conditional_human_m.md)

<!-- RELATED:END -->
