---
title: >-
  [论文解读] JAMUN: Bridging Smoothed Molecular Dynamics and Score-Based Learning for Conformational Ensembles
description: >-
  [NeurIPS 2025][医学图像][分子构象集成] 提出 JAMUN，一种基于 Walk-Jump Sampling 框架的分子构象集成生成方法，通过在加噪的平滑流形上执行朗之万动力学并用 SE(3) 等变去噪器跳回原始分布，实现了比传统分子动力学快一个数量级的肽段构象采样，且具备对训练外系统的迁移能力。
tags:
  - NeurIPS 2025
  - 医学图像
  - 分子构象集成
  - Walk-Jump采样
  - 去噪
  - SE(3)等变
  - 蛋白质动力学
---

# JAMUN: Bridging Smoothed Molecular Dynamics and Score-Based Learning for Conformational Ensembles

**会议**: NeurIPS 2025

**arXiv**: [2410.14621](https://arxiv.org/abs/2410.14621)

**代码**: [有](https://github.com/prescient-design/jamun)

**领域**: 分子动力学 / 生成模型

**关键词**: 分子构象集成, Walk-Jump采样, 去噪, SE(3)等变, 蛋白质动力学

## 一句话总结

提出 JAMUN，一种基于 Walk-Jump Sampling 框架的分子构象集成生成方法，通过在加噪的平滑流形上执行朗之万动力学并用 SE(3) 等变去噪器跳回原始分布，实现了比传统分子动力学快一个数量级的肽段构象采样，且具备对训练外系统的迁移能力。

## 研究背景与动机

蛋白质并非静态结构，而是以构象集成的形式存在，其动态特性对蛋白质功能理解和药物发现（如隐蔽口袋发现）至关重要。然而：

**分子动力学 (MD) 效率低下**：需要 1-2 飞秒的极短时间步长，而许多重要的蛋白质动态现象发生在毫秒尺度——Borhani & Shaw 将其比喻为"以每秒记录一次的频率追踪冰河时代长达数万年的冰川进退"

**增强采样方法的局限**：通常需要关于相关集体变量的领域知识，且未解决根本的时间步长问题

**现有ML方法缺乏迁移性**：流匹配、扩散模型等固然能生成构象，但无法推广到训练集外的系统。唯一例外是 Transferable Boltzmann Generators (TBG)，但效率受限

核心动机：开发一个既快速又具备迁移性的构象集成生成模型。

## 方法详解

### 整体框架

JAMUN 的核心是 Walk-Jump Sampling (WJS)：

1. **Noise（加噪）**：给初始构象 $x^{(0)} \sim p_X$ 添加高斯噪声
   $$y^{(0)} = x^{(0)} + \sigma \varepsilon^{(0)}, \quad \varepsilon^{(0)} \sim \mathcal{N}(0, \mathbb{I}_{N \times 3})$$

2. **Walk（行走）**：在平滑的加噪流形上通过朗之万动力学采样，使用 BAOAB 积分器求解 SDE：
   $$dy = v_y dt, \quad dv_y = \nabla_y \log p_Y(y) dt - \gamma v_y dt + M^{-1/2} \sqrt{2} dB_t$$

3. **Jump（跳跃）**：通过去噪器投射回原始分布
   $$\hat{x}^{(i)} = \hat{x}(y^{(i)}) = \mathbb{E}[X | Y = y^{(i)}]$$

关键：Walk 和 Jump 步骤通过 score function 连接：$\hat{x}(y) = y + \sigma^2 \nabla_y \log p_Y(y)$

### 关键设计

**1. 点云表示**

每个肽段表示为 N 个原子的点云 $(x, h)$，其中 $x \in \mathbb{R}^{N \times 3}$ 为3D坐标，$h$ 为原子和键信息。JAMUN 在全原子3D坐标空间中操作，无需手工特征化（如骨架扭转角）。

**2. SE(3) 等变去噪器**

采用基于 NequIP 的几何图神经网络，关键区别：
- 使用 **SE(3) 等变**而非 E(3) 等变：避免 E(3) 模型因奇偶对称性导致的对称 Ramachandran 图问题
- 无需 TBG 那样的后处理"手性检查器"
- 仅有 **8.2M 参数**

去噪器参数化遵循 Karras et al. (2022) 的设计：
$$\hat{x}_\theta(y, \sigma) = c_{\text{skip}}(\sigma) y + c_{\text{out}}(\sigma) F_\theta(c_{\text{in}}(\sigma) y, c_{\text{noise}}(\sigma))$$

**3. 固定噪声级别**

不同于扩散/流匹配需要多个噪声级别，WJS 只需在**单一固定噪声级别** $\sigma$ 下训练。$\sigma$ 的选择平衡采样效率（越大越好）和去噪难度（太大导致拓扑破坏）。

**4. 输入/输出归一化**

针对点云特性重新推导了归一化系数，基于边距离（而非像素值）进行归一化：
$$c_{\text{in}}(\sigma) = \frac{1}{\sqrt{C + 6\sigma^2}}, \quad c_{\text{skip}}(\sigma) = \frac{C}{C + 6\sigma^2}$$

其中 $C = \mathbb{E}_{(i,j)} \|x_i - x_j\|^2$ 从真实数据估计。

### 损失函数 / 训练策略

- **损失函数**：标准 L2 去噪损失
  $$\theta^* = \arg\min_\theta \mathbb{E}_{X \sim p_X, \varepsilon \sim \mathcal{N}} \|\hat{x}_\theta(Y, \sigma) - X\|^2$$
- **优化器**：Adam，学习率 0.002
- **批量大小**：42 × 6 GPU (A100)
- **噪声级别**：capped-2AA 用 $\sigma=0.4$Å，uncapped-2AA 用 $\sigma=0.6$Å
- **采样参数**：摩擦 $\gamma=0.1$，步长 $\Delta t = \sigma$，$M=1$

## 实验关键数据

### 主实验

**数据集**：
- uncapped-2AA (Timewarp)：380个双氨基酸（200训练/80验证/100测试）
- capped-2AA：带ACE和NME封端基团，更复杂更接近真实蛋白

**表1: JAMUN采样640000个构象的时间 (NVIDIA A100 GPU)**

| Capped-2AA肽段 | 总采样时间 (分钟) | 每个样本时间 (ms) |
|:---|:---:|:---:|
| ASP-TRP | 38 | 3.56 |
| GLU-THR | 27 | 2.53 |
| PHE-ALA | 28 | 2.62 |
| ASN-GLU | 29 | 2.71 |
| CYS-TRP | 34 | 3.18 |
| GLY-ASN | 22 | 2.06 |
| HIS-PRO | 30 | 2.81 |
| ILE-GLY | 22 | 2.06 |

对应 100-300 ps 的 MD 模拟时间，JAMUN 在约一小时内完成。

**Ramachandran 图定性比较**（图8-9）：
- JAMUN 准确采样了测试肽段的绝大多数低能盆地
- 相同 GPU 时间内，MD 通常仍停留在单个盆地
- 与 TBG 相比（图10），相同计算时间下 JAMUN 采样到所有状态，TBG 遗漏了部分盆地

### 消融实验

**表2: Jensen-Shannon 散度收敛分析（图11）**

| 采样方法 | 收敛特征 | 最终JSD |
|:---|:---|:---|
| JAMUN | 快速平滑收敛 | 略高于收敛MD |
| MD (等GPU时间) | 阶梯式（存在动力学陷阱）| 远未收敛 |
| MD (完全收敛) | 最终达到基准 | 参考基准 |

JAMUN 比 MD 更快收敛但不完全收敛到相同精度，部分原因是过采样了稀有/过渡区域。

**Markov State Model 分析**（附录A，图12-19）：
- JAMUN 精确捕获了特定2AA物种的亚稳态形状
- 准确采样了在简单 Ramachandran 图中不明显的细微状态

### 关键发现

1. **速度优势**：JAMUN 构象集成采样比传统 MD 快一个数量级以上
2. **迁移能力**：对训练集外的测试肽段，JAMUN 仍能准确生成构象集成
3. **SE(3) > E(3)**：SE(3) 等变网络自然处理手性，不需要后处理
4. **单噪声级别训练**：与扩散模型不同，WJS 只需一个噪声级别，简化训练
5. **物理先验的保留**：平滑 MD 数据的内在物理先验是迁移能力的关键

## 亮点与洞察

1. **优雅的物理直觉**：WJS 的核心思想是——加噪平滑了能量景观，在平滑流形上朗之万动力学可以高效穿越屏障，再去噪跳回真实分布
2. **Walk 和 Jump 的解耦**：采样通过高效遍历平滑空间的轨迹生成，而非从无信息的高斯先验每次重新开始（与扩散/流匹配模型的根本区别）
3. **首个将 WJS 应用于点云和分子动力学的工作**
4. **实用价值**：在药物发现中，更关心多样化亚稳态的采样而非精确动力学，JAMUN 在这方面表现优异

## 局限与展望

1. **未测试大蛋白**：目前仅验证了二氨基酸肽段，向更大蛋白的迁移待探索
2. **非精确采样器**：JAMUN 是 Boltzmann Emulator 而非 Generator，无法精确重加权
3. **过采样过渡区域**：在粗糙到平滑表面上的随机过程导致非物理的过渡路径
4. **可结合增强采样**：如元动力学（metadynamics）在加噪空间中应用
5. **去噪网络改进**：更好的架构可加速采样；多步去噪（à la diffusion）可锐化生成

## 相关工作与启发

- **Walk-Jump Sampling** (Saremi & Hyvärinen, 2019)：Neural Empirical Bayes 框架
- **Transferable Boltzmann Generators** (Klein & Noé, 2024)：唯一的先前可迁移方法
- **Timewarp** (Klein et al., 2024a)：提供了 uncapped-2AA 基准数据集
- **NequIP** (Batzner et al., 2022)：E(3) 等变 GNN 架构基础
- **Karras normalization** (Karras et al., 2022)：去噪器参数化设计原则
- 启发：将图像生成领域的扩散模型设计原则适配到分子点云，物理先验是实现迁移性的关键

## 评分

| 维度 | 分数 (1-5) | 说明 |
|:---|:---:|:---|
| 创新性 | 5 | WJS首次应用于点云和MD，方法优雅 |
| 技术质量 | 4 | 理论推导严谨，归一化推导完整 |
| 实验充分度 | 4 | capped/uncapped两数据集，MSM分析深入 |
| 实用性 | 4 | 药物发现中的构象采样有实际价值 |
| 写作质量 | 4 | 物理直觉阐述清晰，附录详尽 |

<!-- RELATED:START -->

## 相关论文

- [Learning Conformational Ensembles of Proteins Based on Backbone Geometry](learning_conformational_ensembles_of_proteins_based_on_backbone_geometry.md)
- [Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models](consistent_sampling_and_simulation_molecular_dynamics_with_energy-based_diffusio.md)
- [ConfRover: Simultaneous Modeling of Protein Conformation and Dynamics via Autoregression](confrover_simultaneous_modeling_of_protein_conformation_and_dynamics_via_autoreg.md)
- [Posterior Sampling by Combining Diffusion Models with Annealed Langevin Dynamics](posterior_sampling_by_combining_diffusion_models_with_annealed_langevin_dynamics.md)
- [Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design](uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)

<!-- RELATED:END -->
