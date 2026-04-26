---
title: >-
  [论文解读] Embracing Discrete Search: A Reasonable Approach to Causal Structure Learning
description: >-
  [ICLR 2026][图像生成][因果结构学习] 提出 FLOP（Fast Learning of Order and Parents），一个面向线性模型的基于得分的因果发现算法，通过快速父节点选择与迭代 Cholesky 得分更新大幅降低运行时间，使得迭代局部搜索（ILS）变得可行，在标准因果发现基准上实现近乎完美的图恢复，重新确立离散搜索在因果发现中的合理地位。
tags:
  - ICLR 2026
  - 图像生成
  - 因果结构学习
  - 离散搜索
  - 得分优化
  - DAG学习
  - 线性模型
---

# Embracing Discrete Search: A Reasonable Approach to Causal Structure Learning

**会议**: ICLR 2026  
**arXiv**: [2510.04970](https://arxiv.org/abs/2510.04970)  
**代码**: [https://github.com/CausalDisco/flopsearch](https://github.com/CausalDisco/flopsearch)  
**领域**: 因果发现 / 图模型学习  
**关键词**: 因果结构学习, 离散搜索, 得分优化, DAG学习, 线性模型

## 一句话总结

提出 FLOP（Fast Learning of Order and Parents），一个面向线性模型的基于得分的因果发现算法，通过快速父节点选择与迭代 Cholesky 得分更新大幅降低运行时间，使得迭代局部搜索（ILS）变得可行，在标准因果发现基准上实现近乎完美的图恢复，重新确立离散搜索在因果发现中的合理地位。

## 研究背景与动机

因果结构学习的核心任务是从观测数据中学习变量之间的有向无环图（DAG）。基于得分的方法为每个 DAG 分配惩罚似然得分，寻找得分最优的图。

**现有方法的格局**：

**精确算法**：保证找到得分最优图，但运行时间指数级，仅适用于约30个变量

**局部搜索**（如 GES、BOSS）：通过评估邻居图进行爬山，但容易陷入局部最优

**连续优化方法**（如 NOTEARS）：将无环性编码为光滑约束进行连续优化，但实证和理论上存在质疑，且面临收敛问题和边阈值设定困难

关键洞察：常被引用来否定离散搜索的 NP-hard 结果**不适用于**常见的因果发现设定——标准的困难构造依赖于不可观测变量，而在分布可由稀疏 DAG 表示时，离散搜索可在多项式时间内恢复目标图。

**核心动机**：在有限样本下，得分不精确导致局部搜索陷入局部最优是实践中的核心问题。FLOP 通过加速计算使得可以"完全拥抱离散搜索"——用充足的计算预算进行更彻底的探索来逃离局部最优。

## 方法详解

### 整体框架

FLOP 采用基于顺序（order-based）的 DAG 搜索策略，即搜索变量的因果顺序，然后为给定顺序选择最优父节点集。算法包含四个核心创新组件，围绕一个主题：**用计算效率换取搜索彻底性**。

### 关键设计

1. **快速父节点选择（Fast Parent Selection）**：从上一个顺序学到的父节点集重新初始化

    - 传统方法在每次顺序变化时从头计算父节点集
    - FLOP 利用前一步的父节点集作为热启动，大幅减少计算和内存开销
    - 实验证明这不会降低搜索质量，因为顺序的局部变化通常只影响少数变量的最优父节点

2. **迭代 Cholesky 得分更新（Iterative Cholesky-based Score Updates）**：加速线性高斯 BIC 得分的计算

    - 线性高斯模型的 BIC 得分涉及回归残差的计算
    - FLOP 利用 Cholesky 分解的增量更新特性：当局部搜索移动只改变一条边时，只需对 Cholesky 因子进行秩-1 更新，而非从头分解
    - 这将得分更新的成本在局部移动间**摊销**，显著降低每步计算量
    - 用 Rust 实现以确保效率，通过 Python pip 包 `flopsearch` 发布

3. **有原则的顺序初始化（Principled Order Initialization）**：相比随机初始化，减少对远距离祖先-后代关系的父节点选择失败

    - 远距离且弱依赖的祖先-后代对在有限样本下特别容易被错误处理
    - FLOP 使用基于数据统计量的初始化策略，让初始顺序更接近真实因果顺序
    - 这为后续局部搜索提供更好的起点

4. **迭代局部搜索（Iterated Local Search, ILS）**：系统性地逃离局部最优

    - 传统局部搜索（如 BOSS）找到局部最优后即停止
    - FLOP 在局部最优处进行受控扰动后重新搜索，反复迭代
    - FLOP_k 表示执行 k 次 ILS 重启，k 可作为**计算预算超参数**
    - 建立了运行时间与有限样本精度之间的直接联系：更多重启 → 更好的图

### 损失函数 / 训练策略

使用标准的 **BIC（贝叶斯信息准则）** 作为得分函数：

$$\text{BIC}(G) = -2 \log L(G | \mathcal{D}) + k \log n$$

其中 $L$ 是似然，$k$ 是参数数量，$n$ 是样本量。该得分在线性高斯模型下保证一致性——在无穷样本极限下得分最优的图恢复真实因果图（或其等价类 CPDAG）。

## 实验关键数据

### 主实验

实验在线性加性噪声模型（ANM）生成的数据上进行，考虑 Erdős-Rényi (ER) 和 Scale-Free (SF) 两种图类型。

| 方法 | ER-50-deg8 SHD ↓ | ER-50-deg8 精确恢复率 | 运行时间 |
|------|-------------------|---------------------|---------|
| PC | ~350 | 0% | ~50s |
| GES | ~250 | 0% | ~200s |
| DAGMA | ~200 | 0% | ~300s |
| BOSS (FLOP_0) | ~50 | 40% | ~30s |
| FLOP_20 | ~20 | 60% | ~100s |
| FLOP_100 | ~10 | 60% | ~400s |

在50节点、平均度8、1000样本的 ER 图上，FLOP 明显优于所有基线。

### 消融实验

| 配置 | 效果说明 |
|------|---------|
| FLOP_0（无ILS重启） | 与 BOSS 精度相当（40%精确恢复），但运行更快 |
| FLOP_20（20次重启） | 精确恢复率提升至60%，SHD 大幅下降 |
| FLOP_100（100次重启） | 与 FLOP_20 精确恢复率相当，但在困难实例上 SHD 更低 |
| 随机初始化 vs. 有原则初始化 | 有原则初始化在远距离弱依赖关系上显著减少错误 |
| 无 Cholesky 加速 | 运行时间成数倍增长，使 ILS 不可行 |

### 关键发现

1. **离散搜索的合理性**：在标准因果发现设定下，离散搜索可以近乎完美地恢复因果图结构，挑战了"NP-hard 因此不可行"的常见认知
2. **计算换精度**：FLOP_k 中的 k 参数建立了明确的**运行时间—精度权衡**，用户可根据计算预算灵活调节
3. **连续优化无优势**：DAGMA 等连续方法在相同计算预算下不如离散搜索，且存在额外的阈值、收敛等问题
4. **有限样本是核心挑战**：即使在渐近理论保证下，有限样本中搜索可能找到比真实图得分更高的图——ILS 是有效的应对策略
5. **Rust 实现使大规模可行**：高效实现使50+节点的完整离散搜索变得实际可行

## 亮点与洞察

1. **重要的认识论贡献**：不仅是算法改进，更是对因果发现方法论的重新审视。NP-hard 论点被仔细解构——标准困难构造依赖不可观测变量，不适用于常见设定
2. **工程与理论结合**：Cholesky 增量更新是经典数值线性代数技巧在因果发现中的巧妙应用，Rust 实现提供了实际可用的工具
3. **ILS 元启发式的引入**：将运筹学中的经典优化范式引入因果发现，建立了计算/精度的显式联系
4. **简洁优雅**：整个方法建立在成熟的统计基础上（BIC 得分、Cholesky 分解），没有引入神经网络或复杂优化，回归到"简单可解释"的路线

## 局限与展望

1. **仅适用于线性模型**：FLOP 的 Cholesky 加速依赖线性高斯假设，非线性因果关系场景需要新的得分函数和搜索策略
2. **因果充分性假设**：假设没有未观测的混杂因子，在实际应用中这一假设常被违反
3. **大规模扩展**：虽然50节点表现优秀，但对于数百或数千变量的基因调控网络等应用，扩展性仍需进一步提升
4. **仅支持观测数据**：未利用干预数据，而干预实验在因果发现中能提供更多信息
5. **ILS 重启次数需要人工设定**：虽然建立了运行时间—精度联系，但缺乏理论指导确定所需的最优重启次数

## 相关工作与启发

- **BOSS**（Andrews et al., 2023）：最强的基于顺序的局部搜索基线，FLOP 在此基础上加速并引入 ILS
- **GES**（Chickering, 2002）：经典的基于等价类的搜索，渐近最优但有限样本表现不佳
- **NOTEARS / DAGMA**：连续优化代表，本文实验表明其不如离散搜索
- **Partition MCMC / Order MCMC**（Kuipers et al., 2022）：贝叶斯方法通过模拟退火逃离局部最优，但计算开销更大

本文对因果发现领域的启发：计算效率的提升可以根本性地改变算法范式的可行性。当离散搜索变得足够快时，它的理论优势（无需阈值、无连续松弛、直接优化 BIC）就能充分发挥。

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Discrete Adjoint Matching](discrete_adjoint_matching.md)
- [\[NeurIPS 2025\] Non-Markovian Discrete Diffusion with Causal Language Models](../../NeurIPS2025/image_generation/non-markovian_discrete_diffusion_with_causal_language_models.md)
- [\[ICLR 2026\] Exposing Hidden Biases in Text-to-Image Models via Automated Prompt Search](exposing_hidden_biases_in_text-to-image_models_via_automated_prompt_search.md)
- [\[ICLR 2026\] Loopholing Discrete Diffusion: Deterministic Bypass of the Sampling Wall](loopholing_discrete_diffusion_deterministic_bypass_of_the_sampling_wall.md)
- [\[ICLR 2026\] JointDiff: Bridging Continuous and Discrete in Multi-Agent Trajectory Generation](jointdiff_bridging_continuous_and_discrete_in_multi-agent_trajectory_generation.md)

<!-- RELATED:END -->
