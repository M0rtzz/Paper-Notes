---
title: >-
  [论文解读] BOPO: Neural Combinatorial Optimization via Best-anchored and Objective-guided Preference Optimization
description: >-
  [ICML 2025][优化/理论][preference optimization] 将 preference optimization（偏好优化）引入神经组合优化（NCO），提出 BOPO：通过 (1) best-anchored 偏好对构建（hybrid rollout + uniform filtering + best-anchored pairing）和 (2) objective-guided 自适应缩放损失函数（$\beta = g(y_l)/g(y_w)$），在 JSP/TSP/FJSP 三类经典组合优化问题上全面超越 SOTA，无需 reward model 或参考策略。
tags:
  - "ICML 2025"
  - "优化/理论"
  - "preference optimization"
  - "neural combinatorial optimization"
  - "job-shop scheduling"
  - "TSP"
  - "DPO"
  - "SimPO"
  - "sample efficiency"
---

# BOPO: Neural Combinatorial Optimization via Best-anchored and Objective-guided Preference Optimization

**会议**: ICML 2025  
**arXiv**: [2503.07580](https://arxiv.org/abs/2503.07580)  
**代码**: [https://github.com/L-Z-7/BOPO](https://github.com/L-Z-7/BOPO)  
**领域**: 对齐RLHF  
**关键词**: preference optimization, neural combinatorial optimization, job-shop scheduling, TSP, DPO, SimPO, sample efficiency

## 一句话总结

将 preference optimization（偏好优化）引入神经组合优化（NCO），提出 BOPO：通过 (1) best-anchored 偏好对构建（hybrid rollout + uniform filtering + best-anchored pairing）和 (2) objective-guided 自适应缩放损失函数（$\beta = g(y_l)/g(y_w)$），在 JSP/TSP/FJSP 三类经典组合优化问题上全面超越 SOTA，无需 reward model 或参考策略。

## 研究背景与动机

### NCO 训练范式的演进

**监督学习（SL）**：需昂贵的（近）最优解作为标签，限制实用性

**强化学习（RL）**：主流范式（如 POMO 用 REINFORCE），但面临**稀疏 reward** 和**低样本效率**

**自标注学习（SLL）**：如 SLIM，用采样中的最优解作伪标签 + 交叉熵训练。改善了 RL，但丢弃所有非最优采样解，样本效率仍低

### 核心观察

NCO 有两个独特特点可以利用：
1. 生成模型可对同一实例生成**多个不同解**
2. 组合优化问题的目标值（如 makespan、路径长度）可**低成本精确计算**

→ 解之间存在**天然偏好关系**，可直接利用目标值构建偏好对用于训练，无需人类标注或额外 reward model。

### 与 LLM 偏好优化的对比

- DPO：需 reference model 防止策略偏离过大 → 额外计算开销
- SimPO：消除 reference model 但有额外超参数（$\beta, \gamma$）需调优
- **BOPO**：用目标值比率 $g(y_l)/g(y_w)$ 作自适应缩放因子，无额外超参数

## 方法详解

### 整体框架

BOPO 的流程（如论文 Figure 1）：

1. **Hybrid Rollout**：对每个实例生成 $B$ 个解（$B-1$ 个采样 + 1 个贪心）
2. **Uniform Filtering**：从排序后的 $B$ 个解中均匀选择 $K$ 个代表解
3. **Best-anchored Pairing**：以最优解为锚点，与其余 $K-1$ 个解配对，生成 $K-1$ 个偏好对
4. **Objective-guided Loss**：用自适应缩放的偏好优化损失训练模型

### 关键设计

#### 1. Best-anchored Preference Pair Construction

**Hybrid Rollout**：兼顾探索（$B-1$ 个采样）和利用（1 个贪心）。贪心解确保偏好对中始终有高质量候选。实验证明 hybrid rollout 效果约等于将 $B$ 翻倍。

**Uniform Filtering**：从排序解中均匀采样 $K$ 个解（$y_k = y'_{\lfloor B/K \rfloor \cdot (k-1) + 1}$），避免过拟合到相似解的聚类。对比实验中，top-$K$ filtering 效果最差——因为过滤掉差解后偏好差异不够大。

**Best-anchored Pairing**：只生成 $K-1$ 个偏好对（最优 vs 每个次优），而非 $\binom{K}{2}$ 个全排列对。理由：COP 只关注找最优解，锚定最优解的学习信号更聚焦。

#### 2. Objective-guided Preference Optimization Loss

**显式偏好**：$f^*(y, x) = -g(y)$（目标值取负，最小化问题转为偏好）

**隐式偏好**：$f_\theta(y, x) = \frac{1}{|y|} \log \pi_\theta(y|x)$（平均 log-likelihood）

**偏好分布**采用 Bradley-Terry 模型建模：

$$p_\theta(y_w \succ y_l | x) = \sigma\left(\beta(x, y_w, y_l) \cdot (f_\theta(y_w, x) - f_\theta(y_l, x))\right)$$

其中 $\beta(x, y_w, y_l) = \frac{f^*(y_l, x)}{f^*(y_w, x)} = \frac{g(y_l)}{g(y_w)}$ 是**自适应缩放因子**。

### 损失函数/训练策略

BOPO 损失函数：

$$\mathcal{L}_{\text{BOPO}}(\pi_\theta, x, y_w, y_l) = -\log \sigma\left(\underbrace{\frac{g(y_l)}{g(y_w)}}_{\text{Adaptive Scaling}} \cdot \left(\underbrace{\frac{\log \pi_\theta(y_w|x)}{|y_w|} - \frac{\log \pi_\theta(y_l|x)}{|y_l|}}_{\text{Average Log-likelihood Difference}}\right)\right)$$

**梯度分析**：

$$\nabla_\theta \mathcal{L}_{\text{BOPO}} = -\underbrace{\frac{g(y_l)}{g(y_w)}}_{\text{Adaptive Scaling}} \cdot \underbrace{(1 - \sigma(z))}_{\text{Confidence Weight}} \cdot \left(\frac{1}{|y_l|}\nabla_\theta \log \pi_\theta(y_l|x) - \frac{1}{|y_w|}\nabla_\theta \log \pi_\theta(y_w|x)\right)$$

三个关键属性：
- **Adaptive Scaling**：$g(y_l)/g(y_w) > 1$（最小化问题），差解与好解差距越大，梯度越大
- **Confidence Weight**：模型已学好的对贡献小梯度，防止过拟合
- **长度归一化**：$1/|y|$ 确保不同长度解链等贡献

训练算法用 Adam 优化器，整体流程简洁：采样 → 过滤 → 配对 → 计算 BOPO loss → 更新。

## 实验关键数据

### 主实验

**JSP (Job-shop Scheduling Problem) — TA Benchmark：**

| 方法 | 类型 | 平均 Gap (B'=512) |
|------|------|----------|
| OR-Tools | 精确求解器 | 2.0% |
| L2S 5k | RL 改进 | 7.4% |
| SLIM (GAT-MHA) | SLL-greedy | 13.4% |
| SLIM (B'=512) | SLL-sampling | 7.8% |
| **BOPO (greedy)** | **PO** | **12.9%** |
| **BOPO (B'=512)** | **PO** | **7.5%** |

BOPO 在构建式方法中全面最优。与 L2S 5k 的 7.4% 几乎持平（7.5%），但推理时间 4.8 min vs 4 小时。

**TSP — 1000 个随机实例：**

| 方法 | TSP20 Gap | TSP50 Gap | TSP100 Gap |
|------|-----------|-----------|------------|
| POMO | 0.04% | 0.21% | 0.46% |
| DABL (SL) | 0.01% | **0.04%** | 0.29% |
| SLIM | 0.22% | 1.55% | 3.22% |
| **BOPO** | **0.00%** | 0.07% | **0.12%** |

TSP100 上 BOPO 比 POMO 好 4 倍 (0.12% vs 0.46%)，且训练 epoch 更少（700 vs 2000）。

**FJSP — LA Benchmark (Sampling B'=100)：**

| 数据集 | HG | RS | **BOPO** |
|--------|-----|-----|------|
| LA(e-data) | 8.2% | 6.9% | **6.1%** |
| LA(r-data) | 5.8% | 4.7% | **4.0%** |
| LA(v-data) | 1.4% | 0.8% | **0.6%** |

### 消融实验

**偏好对构建消融（DMU benchmark，平均 Gap）：**

| 消融 | Avg Gap |
|------|---------|
| w/o Hybrid Rollout（纯 sampling） | 14.4% |
| w/o Uniform Filtering（random） | 14.0% |
| w/o Uniform Filtering（top-K） | 15.6% |
| w/o Best-anchored（full permutation） | 14.1% |
| **BOPO (完整)** | **13.3%** |

Top-K filtering 效果最差（15.6%），说明需要保留差解以创造足够偏好差异。

**损失函数消融（TA benchmark，B'=512）：**

| 损失 | TA Gap |
|------|--------|
| DPO | 7.7% |
| SimPO | 7.8% |
| BOPO (w/o scaling) | 7.6% |
| **BOPO** | **7.5%** |

DMU 上差异更显著：DPO 14.1% → SimPO 13.6% → BOPO 12.9%。SimPO 在 OOD 场景（DMU）表现差，暗示其 target margin $\gamma$ 导致过拟合。

### 关键发现

1. **训练曲线对比**（Figure 3）：BOPO 在相同训练实例数下 gap 显著低于 RL（PPO/POMO）和 SLL（SLIM），尤其数据有限时优势更大
2. **$B=256, K=16$ 是最优超参组合**：$B$ 过大收益递减但内存增长，$K$ 过大解相似性过高
3. **BOPO 的模型无关性**：成功应用于 POMO（Transformer）和 MGL（GAT+LSTM）两种不同架构，以及 INViT
4. **MGL vs GAT-MHA**（SLIM 的模型）：MGL 反向传播内存仅为 GAT-MHA 的 1/27，训练更高效

## 亮点与洞察

1. **巧妙的跨领域迁移**：将 LLM 对齐领域的 preference optimization 迁移到组合优化，思路新颖
2. **不需要 reward model 和 reference policy**：比 DPO 更简洁，比 SimPO 更少超参
3. **自适应 $\beta$ 的设计精巧**：$g(y_l)/g(y_w)$ 天然可用、无需调参、有明确物理意义——目标值比率就是最好的偏好强度度量
4. **best-anchored pairing 的简洁有效**：只用 $K-1$ 对（vs $\binom{K}{2}$），计算效率高且聚焦学最优解
5. **架构无关性**：同一训练范式可无缝集成不同 NCO 模型，设计通用性强
6. **Hybrid rollout 的隐含 curriculum**：训练初期贪心解占优提供强梯度，后期采样解追上后提供多样性

## 局限与展望

1. **仍需大量 rollout**：$B=256$ 的采样成本与 SLIM 相当，并非零成本
2. **仅验证构建式方法**：未与改进式方法（如 L2S）在同一模型上对比 BOPO 训练的效果
3. **大规模问题**：未测试 $n > 100$ 的 TSP 或 $n > 50$ 的 JSP，泛化性需进一步验证
4. **理论缺失**：缺乏 BOPO 收敛性的理论分析或对比 RL 的样本复杂度保证
5. **领域归类**：虽被归入 alignment/RLHF 领域，实际是组合优化方法，preference optimization 仅是技术工具

## 相关工作与启发

- **DPO → SimPO → BOPO 的演进路径**：DPO 需 reference model，SimPO 去掉但增超参，BOPO 用目标值自适应，是自然延伸
- **POMO**（Kwon et al. 2020）：利用解对称性的多起点 RL 基线，BOPO 的 TSP 实验基于 POMO 模型
- **SLIM**（Corsini et al. 2024）：SLL SOTA，用自标注的最优解做交叉熵。BOPO 在此基础上利用次优解信息
- **启发**：preference optimization 不限于 LLM——任何有明确目标函数的序列生成问题都可能受益。未来可探索在分子生成、蛋白质设计等领域的应用

## 评分
- 新颖性: ⭐⭐⭐⭐ (preference optimization 迁移到 NCO 是新颖组合，但各组件思路直接)
- 实验充分度: ⭐⭐⭐⭐⭐ (三类问题 + 三个 benchmark + 丰富消融 + 超参敏感性分析)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，梯度分析透彻，但论文较长)
- 价值: ⭐⭐⭐⭐ (为 NCO 建立了 preference optimization 训练范式，有广泛适用性)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Right Now, Wrong Then: Non-Stationary Direct Preference Optimization under Preference Drift](right_now_wrong_then_non-stationary_direct_preference_optimization_under_prefere.md)
- [\[NeurIPS 2025\] Probing Neural Combinatorial Optimization Models](../../NeurIPS2025/optimization/probing_neural_combinatorial_optimization_models.md)
- [\[AAAI 2026\] Pareto-Grid-Guided Large Language Models for Fast and High-Quality Heuristics Design in Multi-Objective Combinatorial Optimization](../../AAAI2026/optimization/pareto-grid-guided_large_language_models_for_fast_and_high-quality_heuristics_de.md)
- [\[NeurIPS 2025\] Robust LLM Alignment via Distributionally Robust Direct Preference Optimization](../../NeurIPS2025/optimization/robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)
- [\[NeurIPS 2025\] Rethinking Neural Combinatorial Optimization for Vehicle Routing Problems with Different Constraint Tightness Degrees](../../NeurIPS2025/optimization/rethinking_neural_combinatorial_optimization_for_vehicle_routing_problems_with_d.md)

</div>

<!-- RELATED:END -->
