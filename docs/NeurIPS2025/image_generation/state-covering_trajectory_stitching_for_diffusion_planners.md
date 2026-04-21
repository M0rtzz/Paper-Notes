---
title: >-
  [论文解读] State-Covering Trajectory Stitching for Diffusion Planners
description: >-
  [NeurIPS 2025][图像生成][扩散规划器] 提出 SCoTS（State-Covering Trajectory Stitching），一种无需奖励信号的轨迹增强框架，通过在时间距离保持的潜空间中迭代拼接短轨迹片段，系统性地扩展状态空间覆盖，显著提升扩散规划器在长时域、分布外任务上的泛化能力。
tags:
  - NeurIPS 2025
  - 图像生成
  - 扩散规划器
  - 轨迹拼接
  - 状态覆盖
  - 离线强化学习
  - 数据增强
---

# State-Covering Trajectory Stitching for Diffusion Planners

**会议**: NeurIPS 2025  
**arXiv**: [2506.00895](https://arxiv.org/abs/2506.00895)  
**代码**: [GitHub](https://github.com/leekwoon/scots/)  
**领域**: 扩散模型 / 轨迹规划  
**关键词**: 扩散规划器, 轨迹拼接, 状态覆盖, 离线强化学习, 数据增强

## 一句话总结

提出 SCoTS（State-Covering Trajectory Stitching），一种无需奖励信号的轨迹增强框架，通过在时间距离保持的潜空间中迭代拼接短轨迹片段，系统性地扩展状态空间覆盖，显著提升扩散规划器在长时域、分布外任务上的泛化能力。

## 研究背景与动机

扩散模型作为轨迹生成工具在离线强化学习中展现了强大潜力——它将整条轨迹看作一个高维样本进行去噪生成，天然避免了自回归模型的误差累积问题。然而，扩散规划器的性能本质上受限于训练数据的质量、多样性和覆盖范围：

**规划时域受限**：有效规划长度与训练轨迹的最大长度耦合，难以生成远超训练分布的长时域计划

**泛化能力不足**：如果数据集主要包含特定运动模式，规划器难以综合出需要不同行为组合的新任务解

**数据收集昂贵**：穷尽式收集所有场景的数据不现实

现有轨迹拼接方法依赖外部奖励进行片段选择，且难以保证拼接后轨迹的动态一致性和可行性。这促使作者设计了一种**无需奖励信号、基于状态覆盖驱动**的轨迹增强方案。

## 方法详解

### 整体框架

SCoTS 采用三阶段流程：(1) 学习时间距离保持的潜表示；(2) 基于方向性探索和新颖性的迭代拼接策略；(3) 基于扩散模型的拼接点精炼。整个过程从离线数据集 $\mathcal{D}$ 出发，生成增强数据集 $\mathcal{D}_{\text{aug}}$，然后在增强数据上训练扩散规划器。

### 关键设计

1. **时间距离保持嵌入（Temporal Distance-Preserving Embedding）**

   核心目标是将原始状态映射到潜空间 $\mathcal{Z}$，使欧氏距离近似最优时间距离。定义目标条件值函数：

   $$V(\boldsymbol{s}, \boldsymbol{g}) \coloneqq -\|\phi(\boldsymbol{s}) - \phi(\boldsymbol{g})\|_2$$

   训练采用 IQL 启发的时间差分目标：

   $$\mathcal{L}_\phi \coloneqq \mathbb{E}_{(\boldsymbol{s},\boldsymbol{a},\boldsymbol{s}',\boldsymbol{g})\sim\mathcal{D}}\left[\ell_\xi^2\left(-\mathbb{1}(\boldsymbol{s}\neq\boldsymbol{g}) - \gamma\|\bar{\phi}(\boldsymbol{s}')-\bar{\phi}(\boldsymbol{g})\|_2 + \|\phi(\boldsymbol{s})-\phi(\boldsymbol{g})\|_2\right)\right]$$

   设计动机：直接使用原始状态空间距离会忽略动态可达性，导致拼接出时间上不连贯的轨迹。学习到的潜空间虽非完美度量，但在**局部**检索可达候选片段的任务中足够可靠。

2. **方向性探索与新颖性驱动的迭代拼接**

   每条新轨迹随机采样一个初始片段和一个固定的潜空间探索方向 $\boldsymbol{z}$（单位向量）。在每次拼接迭代中：

   - 通过 Top-K 近邻检索候选片段
   - 计算**方向进展分数** $P_j = \langle \phi(\text{end}(\boldsymbol{\tau}_j)) - \phi(\boldsymbol{s}_{1,j}), \boldsymbol{z} \rangle$
   - 计算**新颖性分数** $N_j$（基于粒子估计器估计端点相对于已访问状态的熵）
   - 综合选择：$S_j = P_j + \beta N_j$

   设计动机：单纯方向引导会导致覆盖有限（$\beta=0$），过高新颖性权重会丧失方向区分度（$\beta=20$），$\beta=2$ 实现了最佳平衡。

3. **基于扩散模型的拼接精炼（Diffusion Stitcher）**

   训练一个条件扩散模型 $p_\theta^{\text{stitcher}}$，以当前轨迹末端和候选片段末端为边界条件，生成中间过渡状态：

   $$\boldsymbol{\tau}' \sim p_\theta^{\text{stitcher}}(\cdot \mid \boldsymbol{s}_1 = \text{end}(\boldsymbol{\tau}_{\text{comp}}), \boldsymbol{s}_H = \text{end}(\boldsymbol{\tau}_{\text{best}}))$$

   设计动机：拼接点处可能存在微小的动态不一致，扩散模型能平滑过渡、确保动态可行性。

### 损失函数 / 训练策略

- 嵌入网络使用 expectile 回归损失训练
- 扩散 stitcher 使用标准扩散训练目标 $\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\boldsymbol{\tau}^i, i)\|^2$
- 动作序列由逆动力学模型 $\boldsymbol{a}_t = f_\psi(\boldsymbol{s}_t, \boldsymbol{s}_{t+1})$ 推断
- 每个数据集上采样至 5M 样本

## 实验关键数据

### 主实验

在 OGBench 基准上评估，包含 PointMaze 和 AntMaze 环境的 Stitch 和 Explore 数据集。

| 环境 | 数据集类型 | 规模 | GCIQL | QRL | HIQL | GSC | CD | HD | SCoTS |
|------|-----------|------|-------|-----|------|-----|------|-----|-------|
| PointMaze-Stitch | Giant | - | 0 | 50 | 0 | 29 | 68 | 0 | **100** |
| AntMaze-Stitch | Giant | - | 0 | 0 | 2 | 20 | 65 | 0 | **87** |
| AntMaze-Explore | Large | - | 0 | 0 | 4 | 21 | 27 | 13 | **98** |
| 所有任务平均 | - | - | 12.6 | 36.5 | 36.4 | 65.3 | 77.9 | 25.4 | **96.8** |

### 消融实验 / 离线 GCRL 增强

| 算法 | 数据来源 | PointMaze-Giant-Stitch | AntMaze-Giant-Stitch | AntMaze-Large-Explore |
|------|---------|----------------------|---------------------|-----------------------|
| HIQL | Original | 0 | 2 | 4 |
| HIQL | SynthER | 0 | 0 | 12 |
| HIQL | **SCoTS** | **27** | **55** | **77** |
| CRL | Original | 0 | 0 | 0 |
| CRL | SynthER | 0 | 0 | 2 |
| CRL | **SCoTS** | **18** | **2** | **19** |

### 关键发现

1. SCoTS 在所有任务上实现近最优成功率，尤其在最大规模的 Giant 环境中优势最为显著
2. 新颖性权重 $\beta=2$ 是最佳平衡点：兼顾方向探索和状态覆盖
3. 扩散精炼步骤显著降低拼接点的 Dynamic MSE，确保动态一致性
4. SCoTS 增强数据对传统离线 GCRL 算法（GCIQL、CRL、HIQL）也带来显著提升，表明轨迹级增强优于转移级增强（SynthER）
5. SCoTS 对低级控制器的 horizon 长度不敏感，生成的子目标高度可行

## 亮点与洞察

- **无需奖励信号**的轨迹增强范式，仅依赖状态覆盖驱动，具有很强的通用性
- 将潜空间嵌入用于"局部"拼接检索而非全局度量，巧妙规避了学习完美度量的困难
- 方向探索 + 新颖性的双重评分机制设计优雅，实现了覆盖与多样性的平衡
- 端到端增强流程（嵌入→拼接→精炼→逆动力学）形成完整闭环

## 局限与展望

- 主要在 Maze 类导航环境验证，尚未扩展到高维连续控制或机器人操作任务
- 嵌入质量依赖离线数据的分布，在数据极度稀疏的情况下可能退化
- 计算开销较大：需要预训练嵌入、扩散 stitcher 和逆动力学模型
- 固定探索方向在非各向同性的状态空间中可能不是最优策略

## 相关工作与启发

- 扩散规划器（Diffuser、HD）的数据增强是一个被忽视但至关重要的方向
- 时间距离嵌入可以借鉴到其他需要可达性评估的场景
- 轨迹级数据增强的思路可以迁移到 imitation learning 和 world model 训练中

## 评分

- **新颖性**: ⭐⭐⭐⭐ 无奖励轨迹增强框架，方向+新颖性双重评分是新颖的设计
- **实验充分度**: ⭐⭐⭐⭐⭐ 多环境、多数据集类型、多基线、丰富消融
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，图示直观
- **价值**: ⭐⭐⭐⭐ 为扩散规划器的数据瓶颈提供了系统性解决方案

<!-- RELATED:START -->

## 相关论文

- [Hierarchical Koopman Diffusion: Fast Generation with Interpretable Diffusion Trajectory](hierarchical_koopman_diffusion_fast_generation_with_interpretable_diffusion_traj.md)
- [Diffuse Everything: Multimodal Diffusion Models on Arbitrary State Spaces](../../ICML2025/image_generation/diffuse_everything_multimodal_diffusion_models_on_arbitrary_state_spaces.md)
- [Long-Context State-Space Video World Models](../../ICCV2025/image_generation/long-context_state-space_video_world_models.md)
- [Multidimensional Adaptive Coefficient for Inference Trajectory Optimization in Flow and Diffusion](../../ICML2025/image_generation/multidimensional_adaptive_coefficient_for_inference_trajectory_optimization_in_f.md)
- [Learning Few-Step Diffusion Models by Trajectory Distribution Matching](../../ICCV2025/image_generation/learning_few-step_diffusion_models_by_trajectory_distribution_matching.md)

<!-- RELATED:END -->
