---
title: >-
  [论文解读] Renormalization Group Guided Tensor Network Structure Search
description: >-
  [AAAI 2026][人体理解][张量网络] 提出 RGTN 框架，将统计物理中的重正化群（Renormalization Group）理论引入张量网络结构搜索，通过多尺度粗粒化-扩展-压缩流程和可学习边门控实现连续拓扑演化，在光场压缩、高阶张量分解和视频补全任务上达到 SOTA 压缩率，同时比已有方法快 4–600 倍。
tags:
  - AAAI 2026
  - 人体理解
  - 张量网络
  - 结构搜索
  - 重正化群
  - 多尺度优化
  - 张量分解
---

# Renormalization Group Guided Tensor Network Structure Search

**会议**: AAAI 2026  
**arXiv**: [2512.24663](https://arxiv.org/abs/2512.24663)  
**代码**: [有](https://github.com/Applied-Machine-Learning-Lab/RGTN)  
**领域**: 人体理解  
**关键词**: 张量网络, 结构搜索, 重正化群, 多尺度优化, 张量分解

## 一句话总结

提出 RGTN 框架，将统计物理中的重正化群（Renormalization Group）理论引入张量网络结构搜索，通过多尺度粗粒化-扩展-压缩流程和可学习边门控实现连续拓扑演化，在光场压缩、高阶张量分解和视频补全任务上达到 SOTA 压缩率，同时比已有方法快 4–600 倍。

## 研究背景与动机

**领域现状**：张量网络结构搜索（TN-SS）旨在自动发现最优的张量网络拓扑和秩配置，用于高维数据的高效张量分解。现有方法包括遗传算法（TNGA）、贪心构建（TNGreedy）、局部搜索（TNLS）以及最新的统一优化方法 SVDinsTN。

**现有痛点**：(1) **单尺度优化局限**——现有方法在固定分辨率下搜索，忽略了张量数据固有的多尺度相关结构；(2) **离散搜索空间**——遗传/贪心/局部搜索方法都在组合爆炸的离散空间中操作，难以平滑演化；(3) **结构-参数优化分离**——采样-评估范式中每个候选结构都需完整优化，计算代价极高。

**核心矛盾**：遗传方法的种群大小随张量阶数指数增长，贪心算法做出不可逆的局部决策导致次优，局部搜索容易陷入局部极小值。SVDinsTN 虽通过统一优化实现 100-1000x 加速，但单尺度搜索仍易陷局部最优。

**切入角度**：从凝聚态物理的重正化群理论出发——RG 揭示物理系统在不同尺度下的行为规律。张量网络天然具有三个与 RG 对应的性质：尺度不变相关性、层次化纠缠、信息流动，可直接借用 RG 的多尺度分析框架。

**核心 idea**：将 TN-SS 转化为多尺度 RG 流上的连续优化问题，从粗到细逐步精化网络结构，同时利用节点张力和边信息流作为物理指标引导结构修改，统一结构搜索与参数优化。

## 方法详解

### 整体框架

RGTN 从最粗尺度 $s=S$ 开始，逐步向最细尺度 $s=0$ 推进。每个尺度上依次执行三个阶段：(1) **粗粒化**——对数据降采样 $\mathcal{F}_s = \mathcal{D}_s[\mathcal{F}]$ 降低维度；(2) **扩展阶段**——分裂高张力节点以增加表达能力；(3) **压缩阶段**——合并低信息流边以消除冗余。RG 变换分解为扩展算子 $R_{\text{expand}}: \mathcal{G}_v \mapsto \sum_r \mathcal{G}_u^{(r)} \otimes \mathcal{G}_v^{(r)}$ 和压缩算子 $R_{\text{compress}}: (\mathcal{G}_u, \mathcal{G}_v) \mapsto \mathcal{G}_u \times_{e_{uv}} \mathcal{G}_v$，通过尺度依赖的有效作用量统一结构和参数优化。

### 关键设计

1. **可学习边门控机制（Edge Gating）**:

    - 功能：为网络图中每条边引入连续的软开关 $g_{uv} = \sigma(w_{uv})$，实现优化过程中的动态拓扑修改
    - 核心思路：门控张量收缩 $\mathcal{C}_{uv} = g_{uv} \cdot (\mathcal{G}_u \times \mathcal{G}_v) + (1 - g_{uv}) \cdot \mathcal{I}$，当 $g_{uv} \to 0$ 时边被移除。通过温度退火 $\tau(t) = \tau_0 \exp(-t/t_0)$ 逐步从软决策过渡到硬决策
    - 设计动机：解决传统方法中结构修改是离散操作、会中断优化的问题。连续门控使拓扑演化与参数优化在同一梯度框架中进行，保持梯度流稳定

2. **物理驱动的智能提议生成**:

    - 功能：基于当前网络的物理量指标引导扩展和压缩，而非随机搜索
    - 核心思路：**节点张力** $T_v = \|\partial \mathcal{L}_{\text{data}} / \partial \mathcal{G}_v\|_F \cdot \text{degree}(v)$ 度量节点对重构误差的贡献，高张力节点优先分裂；**边信息流** $I_{uv} = g_{uv} \cdot \text{MI}(\mathcal{G}_u, \mathcal{G}_v)$ 量化连接重要性，低信息流边优先合并
    - 设计动机：避免盲目的组合搜索，利用当前优化状态的"应力分布"有针对性地修改结构，类似于物理中通过力的分析指导材料设计

3. **多尺度渐进细化（Multi-Scale Progressive Refinement）**:

    - 功能：从粗尺度（低分辨率）求解后，用上采样算子 $\mathcal{U}_s$ 将粗尺度解提升到细尺度作为初始化 $\mathcal{M}_{s-1}^{(0)} = \mathcal{U}_s[\mathcal{M}_s^*]$
    - 核心思路：粗尺度时 Lipschitz 常数 $L_s \leq L_0 \cdot 2^{-s\alpha}$ 降低，损失面更光滑，更易找到全局结构。尺度跳转产生的扰动帮助以概率 $\mathbb{P}(\text{escape}) \geq 1 - \prod_{s=1}^S \Phi(r/(2^s \sigma_0))$ 逃离局部极小
    - 设计动机：先解小问题、再解大问题的分治策略，搜索复杂度从 $\Omega(\exp(N^2))$ 降至 $\mathcal{O}(S \cdot T \cdot N^2 \cdot I^N \cdot R_{\max}^N)$，其中 $S = \mathcal{O}(\log I)$

### 损失函数 / 训练策略

多尺度损失函数包含六项：数据保真度（Frobenius 误差）+ 时间一致性 + 空间平滑性 + 对角稀疏性（$\ell_1$正则诱导秩发现）+ 边熵正则 $H(g) = -g\log g - (1-g)\log(1-g)$ 鼓励决定性门控 + 张量核范数低秩正则。各项系数随尺度变化，遵循 RG 流动方程 $d\lambda_k/ds = \beta_k(\{\lambda_j\})$。核参数学习率在细尺度更大 $\eta_{\text{cores}}(s) = \eta_0 \cdot \exp(-s/s_0)$，结构参数学习率在粗尺度更大 $\eta_{\text{struct}}(s) = \eta_0 \cdot (1 + s/s_1)$。

## 实验关键数据

### 主实验：光场数据压缩

| 方法 | Bunny CR@RE0.01 | 时间(×1000s) | Bunny CR@RE0.1 | Knights CR@RE0.01 | 时间(×1000s) |
|------|----------------|-------------|----------------|-------------------|-------------|
| TRALS | 61.2% | 13.41 | 5.38% | 74.2% | 10.42 |
| TNGA | 28.2% | 1004 | 2.27% | 39.1% | 904.5 |
| TNLS | 24.5% | 1388 | 2.18% | 27.5% | 1273 |
| SVDinsTN | 22.6% | 0.752 | 2.69% | 31.7% | 1.563 |
| **RGTN** | **22.3%** | **0.180** | **0.91%** | **29.9%** | **0.178** |

RGTN 在 RE=0.1 时比 SVDinsTN 压缩率好 2.96×（Bunny）和 1.60×（Knights），速度比 TNGA/TNLS 快超过 4500×。

### 高阶张量 & 视频补全

| 方法 | 6阶 CR@RE0.01 | 6阶时间(×1000s) | 8阶 CR@RE0.01 | 8阶时间(×1000s) |
|------|-------------|-----------------|-------------|-----------------|
| TNGreedy | 0.88% | 0.167 | 0.016% | 2.625 |
| TNGA | 0.94% | 3.825 | 0.024% | 51.40 |
| SVDinsTN | 1.13% | 0.002 | 0.016% | 0.017 |
| **RGTN** | **0.76%** | 0.006 | **0.009%** | 0.123 |

| 视频数据 | RGTN MPSNR | SVDinsTN MPSNR | RGTN 时间(s) | SVDinsTN 时间(s) |
|---------|-----------|---------------|-------------|-----------------|
| News | **32.040** | 31.643 | **135.95** | 932.42 |
| Salesman | **31.900** | 31.684 | **144.00** | 769.54 |
| Silent | 30.620 | **32.706** | **142.80** | 532.31 |

### 关键发现
- **结构发现成功率极高**：在 100 次独立测试中，4 阶和 5 阶张量的结构发现成功率达 95%-100%
- **高阶优势扩大**：8 阶时 RGTN 压缩率比 SVDinsTN 好 1.8×，且不出现 OOM（FCTNALS 在 8 阶内存溢出）
- **多尺度初始化效果**：比随机初始化在 RE=0.01 目标上快 40%，且找到更优解

## 亮点与洞察
- **物理-ML 跨界融合的典范**：RG 理论的多尺度特性天然匹配 TN-SS 中的层次化结构发现，不是简单类比而是建立了完整数学对应（固定点、临界性、普适性类），为 TN-SS 提供了全新理论框架
- **连续化离散搜索**：边门控 + 温度退火将离散拓扑搜索转为连续梯度优化，类似 NAS 中的 DARTS 思路但在物理启发下更自然
- **指数级理论加速**：搜索复杂度从 $\Omega(\exp(N^2))$ 降至 $\mathcal{O}(\log I \cdot \log(1/\epsilon))$，并给出了严格的收敛和逃离局部最小保证

## 局限与展望
- 多尺度策略中的超参数（尺度数、每尺度步数）需手动调整，未实现自适应
- 下采样-上采样算子采用固定的 2 的幂次，对非均匀结构数据可能不是最优
- 视频补全在 Silent 数据上劣于 SVDinsTN，说明对某些数据特性仍有敏感性
- 实验场景集中于压缩和补全，可考虑扩展到推荐系统、量子化学等更多落地场景

## 相关工作与启发
- **vs SVDinsTN (CVPR 2024)**：SVDinsTN 通过对角因子稀疏化实现统一优化，但局限于单尺度，易陷局部最优。RGTN 在其基础上加入多尺度 RG 流和边门控，压缩更优且能跳出局部最小
- **vs TNGA/TNLS**：传统采样-评估方法需指数级结构枚举（$\Omega(\exp(N^2))$），RGTN 通过物理驱动提议将搜索代价降至多项式级
- **vs MERA/TRG**：量子物理中 MERA 等方法用 RG 原理做固定拓扑下的态求解，RGTN 首次将 RG 用于拓扑搜索本身

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将 RG 理论完整引入 TN-SS，跨领域融合有深度
- 实验充分度: ⭐⭐⭐⭐ 涵盖结构发现、光场、高阶张量、视频四类任务，但消融不够系统
- 写作质量: ⭐⭐⭐⭐ 理论推导完整严谨，附录详尽，但物理术语门槛较高
- 价值: ⭐⭐⭐⭐ 提供了 TN-SS 新范式并取得显著加速，但领域偏小众

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] Personality-guided Public-Private Domain Disentangled Hypergraph-Former Network for Multimodal Depression Detection](personality-guided_public-private_domain_disentangled_hypergraph-former_network_.md)
- [\[ACL 2026\] Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization](../../ACL2026/human_understanding/enhancing_llm-based_search_agents_via_contribution_weighted_group_relative_polic.md)
- [\[NeurIPS 2025\] TensorRL-QAS: Reinforcement Learning with Tensor Networks for Improved Quantum Architecture Search](../../NeurIPS2025/human_understanding/tensorrl-qas_reinforcement_learning_with_tensor_networks_for_improved_quantum_ar.md)
- [\[AAAI 2026\] ParaRevSNN: A Parallel Reversible Spiking Neural Network for Efficient Training and Inference](pararevsnn_a_parallel_reversible_spiking_neural_network_for_efficient_training_a.md)
- [\[AAAI 2026\] AHAN: Asymmetric Hierarchical Attention Network for Identical Twin Face Verification](ahan_asymmetric_hierarchical_attention_network_for_identical.md)

<!-- RELATED:END -->
