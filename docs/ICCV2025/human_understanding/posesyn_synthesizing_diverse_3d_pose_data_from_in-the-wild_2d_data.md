---
title: >-
  [论文解读] PoseSyn: Synthesizing Diverse 3D Pose Data from In-the-Wild 2D Data
description: >-
  [ICCV 2025][人体理解][3D人体姿态估计] 提出 PoseSyn 框架，通过误差提取模块（EEM）从野外 2D 姿态数据中识别目标估计器的困难样本，再通过运动合成模块（MSM）将不准确的伪标签扩展为多样化的运动序列，最终借助人体动画模型生成带有准确 3D 标注的合成训练数据，在多个真实场景基准上将 3D 姿态估计精度提升最多 14%。
tags:
  - "ICCV 2025"
  - "人体理解"
  - "3D人体姿态估计"
  - "数据合成"
  - "运动生成"
  - "困难样本挖掘"
  - "数据增强"
---

# PoseSyn: Synthesizing Diverse 3D Pose Data from In-the-Wild 2D Data

**会议**: ICCV 2025  
**arXiv**: [2503.13025](https://arxiv.org/abs/2503.13025)  
**代码**: 无  
**领域**: 人体理解  
**关键词**: 3D人体姿态估计, 数据合成, 运动生成, 困难样本挖掘, 数据增强

## 一句话总结

提出 PoseSyn 框架，通过误差提取模块（EEM）从野外 2D 姿态数据中识别目标估计器的困难样本，再通过运动合成模块（MSM）将不准确的伪标签扩展为多样化的运动序列，最终借助人体动画模型生成带有准确 3D 标注的合成训练数据，在多个真实场景基准上将 3D 姿态估计精度提升最多 14%。

## 研究背景与动机

3D 人体姿态估计在动作识别、虚拟现实、运动分析等领域有广泛应用。然而，获取准确的 3D 姿态标注需要多相机系统或动捕设备，成本高昂且仅限于受控室内环境。现有 3D 姿态数据集（如 Human3.6M、MuCo、MPI-INF-3DHP）存在以下核心问题：

**域差距**：室内场景与真实世界之间的视觉差距（背景、光照、人物外观）导致模型在野外场景泛化不佳

**困难姿态覆盖不足**：受控环境难以涵盖真实世界中复杂、动态的姿态配置

**现有数据增强方法的不足**：
- **几何变换方法**（PoseAug、AdaptPose）：仅在关键点层面操作，无法改变图像上下文
- **合成渲染方法**（PoseGen）：使用 GAN 生成姿态 + NeRF 渲染图像，但 NeRF 渲染的人物缺乏真实背景和外观多样性
- **文本引导方法**：文本描述对复杂姿态的精确控制力不足

本文的核心洞察是：丰富的野外 2D 姿态数据可以作为桥梁，通过识别目标估计器（TPE）在哪些真实图像上表现不佳，有针对性地合成围绕这些困难姿态的训练数据。但由于困难图像没有 3D 标注，直接使用 TPE 的伪标签又不准确，因此需要通过运动合成来扩展姿态变体，使其更接近真实困难姿态。

## 方法详解

### 整体框架

PoseSyn 包含三个阶段：
1. **EEM（误差提取模块）**：利用 2D GT 标注和 TPE 的 3D 预测投影之间的误差，识别困难样本和简单样本
2. **MSM（运动合成模块）**：将困难样本的不准确伪标签扩展为运动序列，综合使用文本描述和初始姿态信息
3. **视频生成与过滤**：用人体动画模型（Champ）将运动序列和简单样本的参考图像合成为逼真的训练视频，过滤低质量样本后微调 TPE

### 关键设计

1. **误差提取模块（EEM）**:

    - 功能：从野外 2D 姿态数据集中自动识别 TPE 表现不佳的困难样本
    - 核心思路：对每个 2D 数据集样本，用 TPE 预测 3D 姿态 $\hat{J}^{\text{3D}}$，投影到 2D 得到 $\hat{J}^{\text{2D}}$，计算与 GT 2D 姿态的加权误差：
    $Err = \sum_{n=2}^{N_{\text{2D}}} \mathbf{w}_n \left|(\hat{J}^{\text{2D},n} - \hat{J}^{\text{2D},1}) - (J_{\text{GT}}^{\text{2D},n} - J_{\text{GT}}^{\text{2D},1})\right|$
      选取误差最大的 $K_C$ 个样本为困难数据 $\mathcal{D}_C$，误差最小的 $K_{NC}$ 个为非困难数据 $\mathcal{D}_{NC}$。手臂和腿部关节因运动变化性高而被赋予更大权重 $\mathbf{w}_n$。
    - 设计动机：不同于 PoseGen 从合成图像识别困难姿态，EEM 直接在真实图像上评估 TPE 性能，因此能找到真正影响实际部署的困难案例。困难数据通常包含动态复杂姿态，非困难数据主要是静态站立姿态。

2. **运动合成模块（MSM）**:

    - 功能：将困难样本的不准确伪标签 $\hat{J}_C^{\text{3D}}$ 扩展为包含多样化姿态变体的运动序列
    - 核心思路：
        - 首先用 VLM 为困难图像生成文本描述（如"person kneeling down and chopping with an axe"）
        - 将伪标签姿态 $\hat{J}_C^{\text{3D}}$ 复制 $T$ 帧形成初始运动表示 $\mathcal{MR}_{\text{init}} = F(\hat{J}_C^{\text{3D}} \otimes T)$
        - 通过 T2M-GPT 的 Motion VQ-VAE 编码器得到初始运动索引 $\mathcal{S}_{\mathcal{MR}}$
        - 在文本嵌入 $\mathbf{e}_{\text{text}}$ 和初始运动索引 $\mathcal{S}_{\mathcal{MR}}$ 的联合引导下，自回归生成运动序列：
       $$p(\mathcal{S}_C | \mathbf{e}_{\text{text}}, \mathcal{S}_{\mathcal{MR}}) = \prod_{i=0}^{|\mathcal{S}_C|} p(s^i | \mathbf{e}_{\text{text}}, \mathcal{S}_{\mathcal{MR}}, s^{<i})$$
        - 解码得到 $L$ 帧 3D 姿态的运动序列 $\mathcal{M}_C$
    - 设计动机：仅用文本生成（无 $\mathcal{MR}_{\text{init}}$）会引入歧义，无法精确定位困难姿态的几何细节；仅用伪标签又不准确。通过运动序列形式而非孤立帧，可以生成覆盖困难姿态附近的多种合理变体，提高命中真实困难姿态的概率。

3. **运动引导视频生成与训练**:

    - 功能：将运动序列转化为带有逼真外观和背景的训练图像-姿态对
    - 核心思路：使用 Champ（现成的人体动画模型）以非困难数据的图像 $I_{NC}$ 为参考外观，以运动序列 $\mathcal{M}_C$ 为驱动，生成逼真的人体动画视频。对每一帧用 TPE 预测 3D 姿态，与运动序列中的 3D 姿态做误差过滤：
    $Err_{\text{3D},l} = \sum_{n=2}^{N_{\text{3D}}} |(\hat{J}_l^{\text{3D},n} - \hat{J}_l^{\text{3D},1}) - (J_{\text{C},l}^{\text{3D},n} - J_{\text{C},l}^{\text{3D},1})|$
      误差超过阈值 $\tau$ 的帧被丢弃，保留高质量样本与原始真实数据合并训练 TPE。
    - 设计动机：不同于 PoseGen 使用 NeRF 渲染（缺乏背景和外观多样性），Champ 可以保留参考图像的真实人物外观和背景，生成更逼真的训练数据。

### 损失函数 / 训练策略

PoseSyn 本身不引入新的损失函数，仅生成增强数据用于对 TPE 进行微调。TPE 使用各自原始的训练损失（如 MPJPE loss）。关键超参数：$K_C = 500$（困难样本数），$K_{NC} = 200$（非困难参考图像数），最终合成约 27,000 个 3D 姿态数据样本。

## 实验关键数据

### 主实验

三种 TPE 在真实场景数据集上的性能对比（MPJPE↓mm / PA-MPJPE↓mm）：

| TPE | 方法 | 3DPW MPJPE↓ | EMDB MPJPE↓ | CMU_171204 MPJPE↓ | HuMMan MPJPE↓ | Mean MPJPE↓ |
|-----|------|------------|------------|-------------------|--------------|------------|
| 3DCrowdNet | Real-only | 81.7 | 115.8 | 108.8 | 98.9 | 103.2 |
| 3DCrowdNet | PoseGen | 80.0 | 113.1 | 104.0 | 94.5 | 99.7 |
| 3DCrowdNet | **PoseSyn** | **77.4** | **111.0** | **101.0** | **93.1** | **97.5** |
| HyBrik | Real-only | 88.0 | 155.4 | 117.5 | 119.7 | 121.2 |
| HyBrik | **PoseSyn** | **78.4** | **129.9** | **100.3** | **95.3** | **104.6** |
| 4DHumans | Real-only | 81.3 | 116.3 | 115.1 | 106.1 | 106.8 |
| 4DHumans | **PoseSyn** | **77.0** | **108.6** | **104.1** | **98.0** | **99.1** |

PoseSyn 在所有 TPE 和数据集上都取得最佳性能，MPJPE 提升 6-14%，PA-MPJPE 提升 5-9%。

### 消融实验

MSM 有效性分析（PA-MPJPE↓mm，EMDB 上 100 个困难样本）：

| 方法 | Mean±Std | Min | 说明 |
|------|---------|-----|------|
| (a) 伪标签 $\hat{J}^{\text{3D}}$ | 181.7 ± 0.0 | 181.7 | 单个不准确预测 |
| (b) w/o $\mathcal{MR}_{\text{init}}$ | 222.3 ± 36.4 | 151.1 (-16.8%) | 仅用文本，平均更差但最小值更好 |
| (c) **完整 MSM** | **209.3 ± 36.5** | **140.8 (-22.5%)** | 文本+初始姿态，最小误差最低 |

MSM 对 TPE 训练的影响（3DCrowdNet，Mean MPJPE↓）：

| 配置 | Mean MPJPE↓ | Mean PA-MPJPE↓ | 说明 |
|------|------------|----------------|------|
| Real-only | 103.2 | 66.2 | 基线 |
| 仅伪标签 $\hat{J}^{\text{3D}}$ | 99.6 | 65.4 | 有限提升 |
| w/o $\mathcal{MR}_{\text{init}}$ | 98.5 | 64.4 | 缺少姿态先验 |
| **完整 PoseSyn** | **97.5** | **62.9** | 两者结合效果最佳 |

### 关键发现

- **运动序列比单帧生成更有效**：通过将伪标签扩展为运动序列，MSM 能生成至少一个比原始伪标签更接近真实困难姿态的变体（Min 误差降低 22.5%）
- **初始运动表示 $\mathcal{MR}_{\text{init}}$ 至关重要**：去掉它后 Min 误差仅降低 16.8%（vs 22.5%），说明仅靠文本无法精确定位困难姿态
- **从真实图像识别困难姿态优于合成图像**：Ours-N（相同 EEM+MSM 但用 NeRF 渲染）已优于 PoseGen，但 PoseSyn（用 Champ 动画）进一步提升，最终取得最大 14% 的 MPJPE 改进
- **模型无关性**：在 3DCrowdNet、HyBrik、4DHumans 三种不同架构和规模的 TPE 上都一致有效

## 亮点与洞察

1. **困难样本驱动的数据合成**：不同于盲目增强所有数据，PoseSyn 精确定位 TPE 的薄弱环节进行有针对性的增强，这一策略在姿态估计的连续空间中尤为有效
2. **运动序列作为姿态的自然扩展**：将孤立的不准确姿态扩展为连续运动序列，利用运动先验自然地产生姿态变体，避免了直接在关键点空间做随机扰动可能产生的不合理姿态
3. **无需 3D 标注的闭环增强**：仅需 2D GT 标注即可驱动整个流程（识别困难样本→合成运动→生成图像→产出自动 3D 标注），完全避免了昂贵的 3D 标注需求
4. **图像质量的关键作用**：Champ 动画模型相比 NeRF 渲染能保留真实背景和人物外观，这对图像到 3D 姿态估计器的训练至关重要

## 局限与展望

- 依赖 VLM 生成的文本描述质量，复杂遮挡场景下文本可能不够精确
- Champ 人体动画模型可能产生人物-背景融合伪影，虽然有过滤机制但可能丢弃过多有用样本
- 目前仅在 MPII 数据集上采集困难样本，扩展到更多野外数据集可能进一步提升
- 运动合成基于 T2M-GPT，其性能上界受限于预训练运动数据的多样性

## 相关工作与启发

- EEM 的困难样本挖掘策略可推广到其他需要数据增强的回归任务（如深度估计、光流估计）
- 运动合成作为中间表示的思路可用于其他需要从 2D 信号恢复 3D 信息的任务
- Champ 等视频生成模型的进步将直接提升 PoseSyn 的合成数据质量，形成良性循环

## 评分

- 新颖性: ⭐⭐⭐⭐ 困难样本驱动+运动序列扩展的组合思路新颖，但各模块使用的是现有方法
- 实验充分度: ⭐⭐⭐⭐⭐ 三种TPE×六个数据集的全面评估，消融设计严谨
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，方法流程图直观
- 价值: ⭐⭐⭐⭐ 提供了一种无需3D标注的通用数据增强范式，对工业部署有实际价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SynFER: Towards Boosting Facial Expression Recognition with Synthetic Data](synfer_towards_boosting_facial_expression_recognition_with_synthetic_data.md)
- [\[CVPR 2026\] ReMoGen: Real-time Human Interaction-to-Reaction Generation via Modular Learning from Diverse Data](../../CVPR2026/human_understanding/remogen_real-time_human_interaction-to-reaction_generation_via_modular_learning_.md)
- [\[ICCV 2025\] HccePose(BF): Predicting Front & Back Surfaces to Construct Ultra-Dense 2D-3D Correspondences for Pose Estimation](hcceposebf_predicting_front_back_surfaces_to_construct_ultra-dense_2d-3d_corresp.md)
- [\[NeurIPS 2025\] PandaPose: 3D Human Pose Lifting from a Single Image via Propagating 2D Pose Prior to 3D Anchor Space](../../NeurIPS2025/human_understanding/pandapose_3d_human_pose_lifting_from_a_single_image_via_propagating_2d_pose_prio.md)
- [\[ICCV 2025\] Monocular Facial Appearance Capture in the Wild](monocular_facial_appearance_capture_in_the_wild.md)

</div>

<!-- RELATED:END -->
