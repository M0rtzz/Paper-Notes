---
title: >-
  [论文解读] Action-Minimization Meets Generative Modeling: Efficient Transition Path Sampling with the Onsager-Machlup Functional
description: >-
  [ICML2025][图像生成][Transition Path Sampling] 本文提出将预训练生成模型（扩散模型和流匹配）的 score 函数解释为随机动力学中的漂移项，通过最小化 Onsager-Machlup (OM) 作用泛函来**零样本**复用预训练模型进行分子系统的过渡路径采样 (TPS)，在丙氨酸二肽、快速折叠蛋白等系统上以远低于传统方法的计算成本获得了物理真实的过渡路径。
tags:
  - ICML2025
  - 图像生成
  - Transition Path Sampling
  - Onsager-Machlup Action
  - 扩散模型
  - Flow Matching
  - Molecular Dynamics
---

# Action-Minimization Meets Generative Modeling: Efficient Transition Path Sampling with the Onsager-Machlup Functional

**会议**: ICML2025  
**arXiv**: [2504.18506](https://arxiv.org/abs/2504.18506)  
**代码**: https://github.com/ASK-Berkeley/OM-TPS  
**领域**: 扩散模型 / 分子动力学 / 生成模型  
**关键词**: Transition Path Sampling, Onsager-Machlup Action, Diffusion Models, Flow Matching, Molecular Dynamics

## 一句话总结
本文提出将预训练生成模型（扩散模型和流匹配）的 score 函数解释为随机动力学中的漂移项，通过最小化 Onsager-Machlup (OM) 作用泛函来**零样本**复用预训练模型进行分子系统的过渡路径采样 (TPS)，在丙氨酸二肽、快速折叠蛋白等系统上以远低于传统方法的计算成本获得了物理真实的过渡路径。

## 研究背景与动机

**领域现状**：过渡路径采样 (Transition Path Sampling, TPS) 是统计力学中的核心挑战——需要找到连接能量景观上两个稳定态之间的高概率路径（如化学反应、蛋白质折叠）。传统方法包括伞形采样 (umbrella sampling)、元动力学 (metadynamics) 和射击算法 (shooting methods)。

**现有痛点**：
   - 传统方法如元动力学需要定义合适的集体变量 (CVs)，而在过渡态附近选择合适的 CV 本身就很困难
   - 射击方法 (shooting) 采样效率低、拒绝率高、需要昂贵的 MD 模拟
   - 现有基于 ML 的方法（强化学习、可微分模拟、h-变换学习）需要针对每个系统专门训练，无法利用已有的大规模原子模拟数据和预训练生成模型

**核心矛盾**：分子构象生成模型（如扩散模型）能生成独立的无偏样本，但它们训练时使用的是不相关的静态样本，无法直接用于采样**时间相关的过渡路径**。

**本文目标** 如何在不进行任何 TPS 专用训练的情况下，复用预训练生成模型来采样分子系统之间的高概率过渡路径。

**切入角度**：作者观察到扩散模型和流匹配模型的 score 函数 $s_{\theta^*} \approx \nabla \log p_{\text{data}}(\mathbf{x})$ 可以解释为随机微分方程 (SDE) 的漂移项。在这组 SDE 下，路径的概率可以通过 Onsager-Machlup 作用泛函来刻画——类似于物理学中的最小作用量原理。

**核心 idea**：将预训练生成模型诱导的随机动力学与 OM 作用泛函结合，通过梯度优化最小化 OM 作用来找到高概率过渡路径，实现生成模型的零样本 TPS 复用。

## 方法详解

### 整体框架

整个方法的 pipeline 分为三步：
1. **初始路径生成**：在生成模型的潜在空间中对两个端点做线性插值，得到初始猜测路径
2. **OM 作用优化**：固定预训练模型的 score 函数，用梯度下降最小化路径的 OM 作用泛函（即最大化路径概率）
3. **解码回构型空间**：若在潜在空间优化，则通过去噪过程解码回原子坐标空间

输入是预训练生成模型 + 两个端点构型 $\mathbf{x}^{(0)} \in \mathcal{A}$, $\mathbf{x}^{(L)} \in \mathcal{B}$；输出是连接两端点的高概率离散路径 $\mathbf{X} = \{\mathbf{x}^{(i)}\}_{i=0}^{L}$。

### 关键设计

1. **OM 作用泛函的构建与物理解释**:

    - 功能：将路径概率的负对数定义为 OM 作用泛函 $S[\mathbf{X}]$，最大化路径概率等价于最小化 $S$
    - 核心思路：离散化 OM 作用包含三个项——**项 A** $\frac{1}{2\Delta t}\|\mathbf{x}^{(i+1)} - \mathbf{x}^{(i)}\|^2$ 鼓励相邻点平滑过渡；**项 B** $\frac{\Delta t}{2\zeta^2}\|\mathbf{\Phi}(\mathbf{x}^{(i)})\|^2$ 鼓励路径经过低漂移区域（能量极值或鞍点）；**项 C** $\frac{D\Delta t}{\zeta}\nabla \cdot \mathbf{\Phi}(\mathbf{x}^{(i)})$ 鼓励路径经过凸区域（动力学更稳定的区域）
    - 设计动机：物理参数 $\Delta t$（时间步长）、$\zeta$（阻尼系数）、$D$（扩散系数）控制三项的相对贡献，提供直观的物理调控手段。对于玻尔兹曼分布数据，学到的 score 直接对应原子力场 $\mathbf{s}_{\theta^*} \propto -\nabla U(\mathbf{x}) = \mathbf{F}(\mathbf{x})$

2. **从生成模型提取 score 函数构建随机动力学**:

    - 功能：从预训练的 DDPM 或流匹配模型中提取 score 函数，构建等价的 SDE
    - 核心思路（DDPM）：通过在固定时间边际 $\tau$ 进行迭代"去噪-加噪"过程，推导出等价的 SDE：$d\mathbf{x} = \mathbf{s}_\theta(\mathbf{x}, \tau) dt + \sqrt{2} d\mathbf{W}_t$，其中 $\mathbf{s}_\theta$ 直接从去噪模型获得
    - 核心思路（Flow Matching）：证明了流匹配的速度场 $u_\theta$ 可通过解析公式转换为 score：$\mathbf{s}_\theta^{\text{FM}} = \frac{\alpha_\tau}{\dot{\sigma}_\tau \sigma_\tau \alpha_\tau - \dot{\alpha}_\tau \sigma_\tau^2}(\frac{\dot{\alpha}_\tau}{\alpha_\tau}\mathbf{x} - u_{\theta^*}(\mathbf{x}, \tau))$
    - 设计动机：这使得 OM 框架不局限于 DDPM，可扩展到流匹配等更广泛的生成模型类别

3. **潜在空间线性插值初始化**:

    - 功能：在生成模型的低噪声潜在空间 $\tau_{\text{initial}}$ 进行端点插值，生成初始路径
    - 核心思路：直接在构型空间线性插值会产生非物理路径（原子构型流形高度非凸），而在潜在空间插值生成的样本更接近数据流形
    - 设计动机：高质量的初始路径对优化结果至关重要，潜在空间的平滑性提供了更好的起点

4. **基于 Hutchinson 估计器的散度加速**:

    - 功能：使用 Hutchinson 随机迹估计器来加速 OM 作用中散度项 $\nabla \cdot \mathbf{s}_{\theta^*}$ 的计算
    - 设计动机：精确计算高维 score 的散度代价过高，随机估计器将计算复杂度降至可接受范围，使得方法可扩展到大规模蛋白质系统

### 训练策略
本方法**不需要任何训练**——核心优势在于直接复用已有预训练生成模型。优化过程仅对路径点做梯度下降，生成模型参数 $\theta^*$ 全程冻结。路径的所有中间点并行优化，天然适合多设备加速。此外引入 Truncated OM Action（忽略散度项 C）作为低扩散率情况下的简化版本。

## 实验关键数据

### 主实验：丙氨酸二肽效率对比

| 方法 | 需要 CV | 力场 / Score 评估次数 (↓) | 单路径运行时间 (↓) |
|------|---------|--------------------------|-------------------|
| MCMC 双向射击 | 否 | ≥ 1B | ≥ 100 小时 |
| 元动力学 (Metadynamics) | 是 | 1M | 10 小时 |
| **OM 优化（扩散模型，本文）** | **否** | **10K** | **50 分钟** |

OM 优化的计算效率远优于传统方法：score 评估次数相比元动力学降低了 **100 倍**，相比 MCMC 射击降低了 **100,000 倍**。

### 快速折叠蛋白实验结果（5 种蛋白平均）

| 方法 | Jensen-Shannon 散度 (↓) | 有效路径比例 (↑) | 转移负对数似然 (↓) |
|------|------------------------|-----------------|-------------------|
| MD 模拟 1μs | ~0.45 | ~0.30 | ~4.5 |
| MD 模拟 10μs | ~0.30 | ~0.55 | ~3.8 |
| MD 模拟 50μs | ~0.20 | ~0.70 | ~3.5 |
| MD 模拟 100μs | ~0.18 | ~0.80 | ~3.2 |
| **OM 优化（Diffusion）** | **~0.18** | **~0.90** | **~2.8** |
| **OM 优化（Flow Matching）** | **~0.19** | **~0.88** | **~2.9** |

OM 优化在所有三个指标上均优于或匹配 100μs 的无偏 MD 模拟，但计算成本远低。

### 消融与分析

| 实验配置 | 关键发现 |
|---------|---------|
| 扩散系数 D 变化 | D 增大 → 路径跨越更高能垒（对应更高温度下的路径行为） |
| 时间步 Δt 变化 | Δt 增大 → 路径平滑性约束减弱，允许更大"跳跃" |
| Truncated vs Full OM | 低扩散率时 Truncated 足够；高扩散率时需要 Full OM（含散度项） |
| 99% 去除过渡态数据 | 即使训练数据中过渡态稀缺，OM 优化仍能找到合理路径 |
| 零样本泛化（四肽） | 在 100 个未见四肽上，OM 优化效果接近 50-100ns MD 模拟 |

### 关键发现
- **物理参数提供直观调控**：通过调整扩散系数 D 可以控制路径跨越不同高度的能垒，这在物理上对应不同温度下的行为
- **对数据稀缺具有鲁棒性**：在去除 99% 过渡态构型后重训模型，OM 优化仍能采样合理路径，说明方法不依赖于训练集中过渡区域的完整覆盖
- **零样本泛化能力**：在四肽系统实验中，模型在未见序列上的 TPS 效果与 50-100ns MD 模拟相当，展示了预训练模型跨化学空间泛化的潜力
- **Müller-Brown 势上的反应速率估计**：估计值为 $1.3 \times 10^{-5}$，真实值为 $5.4 \times 10^{-5}$，在同一数量级内，这在反应速率估计任务中已属准确

## 亮点与洞察

- **零样本复用预训练模型的范式**：本文最核心的贡献不是某个具体技巧，而是建立了"预训练生成模型 → 提取 score → OM 作用优化 → TPS"这一完整范式。这意味着随着生成模型的进步（更大数据、更好架构），TPS 能力会自动升级——无需重新设计 TPS 方法。
- **DDPM 与流匹配的统一处理**：证明了流匹配的速度场可以解析转换为 score 函数（Eq. 17-18），使得 OM 框架对生成模型类别具有通用性。这个 velocity-to-score 的转换公式本身就有独立价值。
- **物理可解释的超参数**：不同于黑箱 ML 方法，OM 作用中的 $\Delta t$、$\zeta$、$D$ 直接对应物理量（时间步、摩擦系数、扩散系数），使得领域专家可以基于物理直觉调参。
- **路径并行优化**：由于 OM 作用是离散积分形式，整条路径的所有中间点可以并行优化，这在计算上非常高效。

## 局限与展望

- **不保证后验分布完备采样**：与传统射击方法不同，本方法不能证明采样了完整的路径后验分布。作者通过生成模型的编码/解码随机性和结合 MD/伞形采样来缓解这一问题，但理论保证仍缺失。
- **依赖预训练模型的质量**：OM 优化的效果上限由预训练生成模型的 score 估计质量决定——若模型在某些区域学习不充分，优化路径可能偏离真实路径。
- **散度项计算开销**：Full OM 作用中的散度项 $\nabla \cdot \mathbf{s}_{\theta^*}$ 即使用 Hutchinson 估计器仍有不小的计算开销，对于非常大的系统可能成为瓶颈。
- **粗粒化蛋白质系统的适用范围**：快速折叠蛋白实验使用 Cα 粗粒化表示，全原子系统的可扩展性尚待验证。
- **初始路径敏感性**：虽然潜在空间插值优于构型空间插值，但对于复杂自由能景观，初始路径选择仍可能影响优化收敛到哪条路径。

## 相关工作与启发

- **vs 传统 TPS（射击/元动力学）**：传统方法需要逐系统定义 CV 或大量 MD 模拟；本文不依赖 CV，不需要系统特定训练，计算效率高 2-5 个数量级。但传统方法有理论保证的分布完备性。
- **vs ML-TPS（强化学习/h-变换）**：Das et al. 2021, Du et al. 2024 等需要针对每个系统专门训练控制策略；本文零样本复用预训练模型，可扩展性更强，但无法像 h-变换那样保证路径分布的精确性。
- **vs Boltzmann 生成器 (Noé et al. 2019)**：Boltzmann 生成器生成独立构型样本，不考虑路径信息；本文利用相同类型的模型但加入时间相关的路径优化。
- **vs Arts et al. 2023（Score as Force Field）**：该工作将扩散模型 score 用作力场做 MD 模拟；本文进一步将 score 与 OM 作用泛函结合，避免了逐步 MD 模拟的全部计算开销。
- **启发**：OM 作用最小化作为"流形上的插值"方法，原则上可扩展到图像、视频、音频等任意数据模态——只要有预训练生成模型提供 score 函数。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将 OM 泛函与生成模型结合的视角非常原创，统一了扩散模型和流匹配
- 实验充分度: ⭐⭐⭐⭐ 四个系统逐步验证（2D→二肽→蛋白→四肽），但全原子系统验证不足
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，物理直觉与数学形式化结合出色
- 价值: ⭐⭐⭐⭐⭐ 定义了新范式——预训练生成模型零样本复用于 TPS，随着模型规模增长价值倍增

<!-- RELATED:START -->

## 相关论文

- [Multidimensional Adaptive Coefficient for Inference Trajectory Optimization in Flow and Diffusion](multidimensional_adaptive_coefficient_for_inference_trajectory_optimization_in_f.md)
- [DCTdiff: Intriguing Properties of Image Generative Modeling in the DCT Space](dctdiff_intriguing_properties_of_image_generative_modeling_in_the_dct_space.md)
- [Compositional Scene Understanding through Inverse Generative Modeling](compositional_scene_understanding_through_inverse_generative_modeling.md)
- [DDIS: When Model Knowledge Meets Diffusion Model](when_model_knowledge_meets_diffusion_model_diffusion-assisted_data-free_image_synthesis.md)
- [IMPACT: Iterative Mask-based Parallel Decoding for Text-to-Audio Generation with Diffusion Modeling](impact_iterative_mask-based_parallel_decoding_for_text-to-audio_generation_with_.md)

<!-- RELATED:END -->
