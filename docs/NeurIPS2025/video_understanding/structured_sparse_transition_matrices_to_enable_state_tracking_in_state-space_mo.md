---
title: >-
  [论文解读] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models
description: >-
  [NeurIPS 2025][视频理解] 本文提出 PD-SSM，一种结构化稀疏参数化方法用于状态空间模型（SSM）的状态转移矩阵。核心思想是将转移矩阵分解为列 one-hot 矩阵 P 与复数对角矩阵 D 的乘积（A = PD），从而在保持与对角 SSM 相当的计算效率（Θ(LN)）的同时，获得与非结构化（稠密）SSM 等同的表达能力——单层即可模拟任意 N 状态有限状态自动机（FSA）。理论上证明了该参数化的 BIBO 稳定性和最优状态维度。实验在 FSA 模拟、多元时序分类、长序列基准和自然语言状态追踪任务中均表现优异。
tags:
  - NeurIPS 2025
  - 视频理解
---

# Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models

**会议**: NeurIPS 2025  
**arXiv**: [2509.22284](https://arxiv.org/abs/2509.22284)  
**代码**: https://github.com/IBM/expressive-sparse-state-space-model  
**领域**: 视频理解  

## 一句话总结

本文提出 PD-SSM，一种结构化稀疏参数化方法用于状态空间模型（SSM）的状态转移矩阵。核心思想是将转移矩阵分解为列 one-hot 矩阵 P 与复数对角矩阵 D 的乘积（A = PD），从而在保持与对角 SSM 相当的计算效率（Θ(LN)）的同时，获得与非结构化（稠密）SSM 等同的表达能力——单层即可模拟任意 N 状态有限状态自动机（FSA）。理论上证明了该参数化的 BIBO 稳定性和最优状态维度。实验在 FSA 模拟、多元时序分类、长序列基准和自然语言状态追踪任务中均表现优异。

## 研究背景与动机

1. **对角 SSM 表达能力受限**：现有高效 SSM（如 Mamba）使用对角转移矩阵，无法模拟非可解自动机（non-solvable automata），在算法状态追踪任务上表现差。
2. **稠密矩阵计算成本过高**：使用非结构化转移矩阵（SD-SSM）可模拟任意 FSA，但并行扫描复杂度为 Θ(LN³)，大规模训练不可行。
3. **现有半结构化方法不够紧凑**：DPLR（对角+低秩）方法虽增强表达力，但需要多层堆叠或指数级大的线性层才能模拟复杂自动机。
4. **计算效率与表达能力难以兼顾**：这是 SSM 领域的核心矛盾——如何在不牺牲效率的前提下提升模型的状态追踪能力。
5. **Transformer 在 FSA 任务上的理论局限**：Transformer 在自动机状态追踪中存在理论瓶颈，混合架构需要更强的 SSM 组件。
6. **对混合 Transformer-SSM 架构的需求**：大型 LLM 越来越多采用混合架构，需要高效且有表达力的 SSM 层作为互补。

### 解决思路

**本文目标**：### PD 参数化

将转移矩阵分解为 $A(u_t) = P(u_t) D(u_t)$：

- **P 矩阵**（列 one-hot 二值矩阵）：通过可训练字典 $\{M_k\}_{k=1}^K$ 的软选择 + 列 hardmax 生成。


## 方法详解

### PD 参数化

将转移矩阵分解为 $A(u_t) = P(u_t) D(u_t)$：

- **P 矩阵**（列 one-hot 二值矩阵）：通过可训练字典 $\{M_k\}_{k=1}^K$ 的软选择 + 列 hardmax 生成。输入 $u_t$ 通过 softmax 得到混合权重 $s(u_t)$，加权求和字典矩阵后对每列取 hardmax 获得稀疏 P。
- **D 矩阵**（复数对角矩阵）：由两个前馈网络分别生成幅度（sigmoid 约束到 (0,1)）和相位（映射到 (0,2π)），确保 BIBO 稳定性。

### 关键代数性质

- PD 矩阵集合在矩阵乘法下封闭（构成幺半群），因此链式矩阵乘积仍为 PD 矩阵。
- 矩阵乘法可通过 gather-scatter 在 Θ(N) 操作内完成，支持高效并行扫描。

### 梯度近似

hardmax 的梯度几乎处处为零，采用 softmax 作为反向传播的代理梯度（straight-through estimator），前向仍使用 hardmax 保持稀疏性。

### 理论保证

- **BIBO 稳定**（Proposition 1）：当 D 矩阵元素模小于 1 时系统有界。
- **完备表达**（Proposition 2）：单层 PD-SSM、N 维状态和 N×N 读出可精确表示任意 N 状态 FSA。
- **最优性**（Proposition 3）：N 状态 FSA 至少需要 N-1 维状态空间，PD-SSM 接近此下界。

## 实验关键数据

### 表1：FSA 模拟任务长度泛化准确率（%）


### 主实验

| 模型 | Cycle Nav. | Even Pairs | Mod Arith. | Parity | 平均 |
|---|---|---|---|---|---|
| Transformer | 24.4% | 90.4% | 23.6% | 52.2% | 47.7% |
| Mamba | 48.4% | 100.0 | 33.1% | 54.2% | 58.9% |
| Gated DeltaProduct | 46.3% | 100.0 | 78.4% | 98.0% | 80.7% |
| BD-SLiCE | 99.8% | 85.9% | 54.0% | 95.3% | 83.8% |
| D-DE-SLiCE | 73.3% | 84.8% | 98.4% | 83.8% | 85.1% |
| **PD-SSM** | **99.5%** | **99.7%** | **96.2%** | **99.9%** | **98.8%** |

### 表2：UEA 多元时序分类准确率（%）


### 消融实验

| 模型 | Worms | SCP1 | SCP2 | Ethanol | Heartbeat | Motor | 平均 |
|---|---|---|---|---|---|---|---|
| Log-NCDE | 85.6% | 83.1% | 53.7% | 34.4% | 75.2% | 53.7% | 64.3% |
| LRU | 87.8% | 82.6% | 51.2% | 21.5% | 78.4% | 48.4% | 61.7% |
| Mamba | 70.9% | 80.7% | 48.2% | 27.9% | 76.2% | 47.7% | 58.6% |
| LinOSS-IM | 95.0% | 87.8% | 58.2% | 29.9% | 75.8% | 60.0% | 67.8% |
| **PD-SSM** | **90.0%** | **80.9%** | **56.1%** | **34.7%** | **80.0%** | **60.0%** | **67.0%** |

### 关键发现
- 主要组件/模块贡献了最关键的性能提升


## 亮点与洞察

- **理论优雅**：PD 参数化同时实现了 BIBO 稳定、通用 FSA 模拟和最优状态维度，三个理论保证齐全。
- **效率突破**：与稠密 SSM 相比获得 71× 加速（D=5632 时），并行扫描复杂度与对角 SSM 相同为 Θ(LN)。
- **FSA 模拟质量跳跃**：在四项标准 FSA 任务上平均 98.8%，显著超越所有现有并行化 SSM。
- **非可解群表现完美**：在 A₅ 和 S₅ 群上多项配置获得 100% 准确率，理论预测与实验完美吻合。
- **混合架构验证**：在冻结 Qwen-2.5 1.5B 上附加单层 PD-SSM，可在自然语言编码的 FSA 状态追踪中成功泛化，展示了作为混合架构组件的潜力。
- **P 与 D 互补性巧妙**：P 编码任意排列、D 紧凑编码循环群，两者结合既通用又高效。

## 局限与展望

- **P 矩阵生成开销**：虽然并行扫描复杂度与对角 SSM 相同，但 P(u_t) 的生成涉及字典查询和 hardmax，实际运行时间约为对角模型的 7 倍。
- **缺乏大规模语言建模验证**：实验主要集中在合成任务和时序分类，尚未在大规模 LLM 预训练中验证效果。
- **代理梯度的理论保证有限**：hardmax→softmax 的梯度近似是启发式的，缺乏收敛性的严格理论分析。
- **字典大小 K 为超参数**：需要人为设定，不同任务最优值不同（FSA 用 K=32，LRA 用 K=6），缺乏自动选择机制。
- **LRA 上时变 SSM 整体不如时不变模型**：在 LRA 基准上，PD-SSM 虽是最佳时变模型，但仍落后于 S4/LRU 等时不变 SSM 约 10 个点。

## 相关工作与启发

- **对角 SSM**：S4 (Gu et al., 2022)、LRU (Orvieto et al., 2023)、Mamba (Gu & Dao, 2023) 等使用对角转移矩阵实现高效计算。
- **DPLR 结构 SSM**：DeltaNet (Schlag et al., 2021)、GLA (Yang et al., 2024)、RWKV-7 (Peng et al., 2025)、DeltaProduct (Siems et al., 2025) 使用对角+低秩增强表达力。
- **非结构化 SSM**：SD-SSM (Terzić et al., 2025) 使用稠密矩阵实现完备 FSA 模拟但计算成本 O(LN³)。
- **FSA 与神经网络**：Merrill & Sabharwal (2024) 证明对角 SSM 无法模拟非可解自动机；Sarrof et al. (2024) 构造复对角 SSM 模拟可解自动机。
- **混合架构**：Griffin (De et al., 2024)、Jamba (Lenz et al., 2025)、TransXSSM (Wu et al., 2025) 等结合 Transformer 与 SSM。
- **神经 CDE**：Log-NCDE (Walker et al., 2024)、LinOSS (Rusch et al., 2025) 等针对时序分析的微分方程方法。

## 评分

| 维度 | 分数 | 说明 |
|---|---|---|
| 新颖性 | 9/10 | PD 参数化是全新的矩阵结构设计，巧妙结合稀疏性与代数封闭性 |
| 理论深度 | 9/10 | 完整的稳定性、表达能力和最优性证明，理论体系非常严谨 |
| 实验充分性 | 8/10 | 覆盖合成 FSA、时序分类、LRA 和自然语言任务，但缺乏大规模 LM 实验 |
| 写作质量 | 9/10 | 行文清晰，理论与实验组织逻辑性强，图表质量高 |
| 实用价值 | 7/10 | 适用于需要精确状态追踪的场景，但大规模实用性 |
| **总分** | **8.5/10** | 兼具理论优雅和实验验证的高质量工作，对 SSM 表达力研究有重要推动 |

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] PASS: Path-Selective State Space Model for Event-Based Recognition](pass_path-selective_state_space_model_for_event-based_recognition.md)
- [\[NeurIPS 2025\] GeoDynamics: A Geometric State-Space Neural Network for Understanding Brain Dynamics on Riemannian Manifolds](geodynamics_a_geometric_state-space_neural_network_for_understanding_brain_dynam.md)
- [\[NeurIPS 2025\] DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products](deltaproduct_improving_state-tracking_in_linear_rnns_via_householder_products.md)
- [\[CVPR 2025\] GG-SSMs: Graph-Generating State Space Models](../../CVPR2025/video_understanding/gg-ssms_graph-generating_state_space_models.md)
- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)

<!-- RELATED:END -->
