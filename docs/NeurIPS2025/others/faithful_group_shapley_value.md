---
title: >-
  [论文解读] Faithful Group Shapley Value
description: >-
  [NeurIPS 2025][Data Shapley] 提出 Faithful Group Shapley Value (FGSV)，唯一满足含"忠实性"在内五条公理的组级数据估值方法，有效防御"空壳公司攻击"（通过拆分子组不当膨胀估值）…
tags:
  - "NeurIPS 2025"
  - "Data Shapley"
  - "组级数据估值"
  - "忠实性公理"
  - "Shell Company Attack"
  - "版权归属"
  - "可解释AI"
---

# Faithful Group Shapley Value

**会议**: NeurIPS 2025  
**arXiv**: [2505.19013](https://arxiv.org/abs/2505.19013)  
**代码**: [KiljaeL/Faithful_GSV](https://github.com/KiljaeL/Faithful_GSV)  
**领域**: others (数据估值 / 合作博弈论)  
**关键词**: Data Shapley, 组级数据估值, 忠实性公理, Shell Company Attack, 版权归属, 可解释AI  

## 一句话总结

提出 Faithful Group Shapley Value (FGSV)，唯一满足含"忠实性"在内五条公理的组级数据估值方法，有效防御"空壳公司攻击"（通过拆分子组不当膨胀估值），并设计了 $O(n \cdot \text{Poly}(\log n))$ 复杂度的高效近似算法。

## 研究背景与动机

### 数据估值的重要性

在现代机器学习中，量化数据的价值对于公平补偿和数据市场设计至关重要。Shapley 值作为合作博弈论中的经典概念，是唯一满足 null player、symmetry、linearity、efficiency 四条公理的估值方法，广泛用于数据估值（Data Shapley）和特征归因（SHAP）。

### 组级估值的需求

许多实际场景需要对**数据集合**而非单个数据点进行估值：

- **数据市场**：数据拥有者以整个数据集为单位贡献数据
- **版权补偿**：生成式 AI 模型训练中需对版权持有者进行组级补偿
- **可解释 AI**：组级特征重要性比个体级更鲁棒、更可解释
- **计算效率**：个体 Shapley 值在大数据上计算代价极高，组级估值更快

### 现有方法的致命缺陷——Shell Company Attack

现有的组级 Shapley 值（Group Shapley Value, GSV）采用 "Group-as-Individual"（GaI）策略，将每个组视为一个原子单元再套用标准 Shapley 公式。然而这种方法存在严重的公平性漏洞：

**Shell Company Attack（空壳公司攻击）**：恶意参与者可以把自己的数据拆分为多个小子组，从而在不改变数据内容的情况下不公平地**膨胀自己的估值**、**压缩其他组的估值**。

论文从理论上证明（Proposition 1）：当效用函数满足"审慎性条件"（即三阶差分 $\Delta_s^3 \bar{U}(s) > 0$，对应经济学中的风险规避行为和 ML 中的数据边际递减现象）时，GSV 必然满足：

$$\mathbb{E}[\text{GSV}(S_k)] < \mathbb{E}[\text{GSV}(S_k')] + \mathbb{E}[\text{GSV}(S_k'')]$$

即拆分一个组后其总估值严格增大。

## 方法详解

### 整体框架：五条公理与唯一性定理

作者提出了**忠实组级数据估值**应满足的五条公理（Definition 2）：

| 公理 | 含义 |
|------|------|
| Null Player | 若组 $S$ 的每个子集对任何联盟都无边际贡献，则 $\nu(S)=0$ |
| Symmetry | 若两组成员可一一对应且贡献等价，则估值相同 |
| Linearity | 对效用函数的线性组合保持线性 |
| Efficiency | 所有组估值之和等于总效用 $U([n])$ |
| **Faithfulness** | 一个组的估值仅取决于其自身成员的贡献，与其他组如何划分无关 |

**Theorem 1（唯一性）**：满足以上五条公理的组级估值方法是唯一的，即

$$\text{FGSV}(S_0) := \sum_{i \in S_0} \text{SV}(i)$$

其中 $\text{SV}(i)$ 是个体 Shapley 值。这意味着 FGSV 就是组内成员 Shapley 值之和。

### 关键设计：高效近似算法

直接计算 FGSV 需要组合式地计算所有个体 Shapley 值，代价极高。论文通过一系列数学洞察设计了高效近似算法（Algorithm 1）：

**Key Observation 1**：在 FGSV 的展开式中，$U(S)$ 项的系数仅取决于三元组 $(s_1, s, s_0)$，其中 $s_0 = |S_0|$，$s = |S|$，$s_1 = |S_0 \cap S|$。可将相同系数的项聚合。

**Key Observation 2**：超几何分布 $\mathbb{P}(\boldsymbol{s_1} = s_1)$ 在 $|s_1 - s \cdot s_0/n|$ 上指数衰减，即 $\mathcal{T}(s)$ 被 $\boldsymbol{s_1}$ 均值附近的值主导。

**Key Observation 3**（核心）：近似展开后 $\mathcal{T}(s) \approx s^{-1}(s_0/n)(1-s_0/n) \mu'(s_0/n)$，只需在单点求导即可高效估计。

**Theorem 2** 将上述直觉形式化：

$$\mathcal{T}(s) = \frac{n}{n-1} \alpha_0 (1-\alpha_0) \left\{ \Delta\mu(s_1^*/s; s, s_0, n) + O(s^{-(1+\upsilon)}) \right\}$$

**分阈值策略**（Algorithm 1）：
- 当 $s < \bar{s}$（小集合）：直接 Monte Carlo 估计 $\mathcal{T}(s)$
- 当 $s \geq \bar{s}$（大集合）：利用 Theorem 2 的近似公式，仅估计 $\Delta\mu$

**配对 Monte Carlo 方差缩减**：估计 $\Delta\mu$ 时不独立估计两个 $\mu$ 值再相减，而是用配对样本直接估计差值（公式 11），大幅降低方差。

### 计算复杂度（Theorem 3）

在 $O(1/s)$-deletion stable 效用函数下，Algorithm 1 达到 $(\epsilon, \delta)$-近似仅需

$$O\left(n \cdot \max\left\{1, (\alpha_0(1-\alpha_0))^2 (\log n)^3\right\}\right)$$

次效用函数评估，其中 $\alpha_0 = s_0/n$。这比先计算全部个体 Shapley 值再求和的 SOTA 方法（$O(\alpha_0 n^2 \text{Poly}(\log n))$）**少一个 $n$ 的量级**。

### 小集合效用函数的处理

对于大模型等在小数据集上效用无法良好定义的情况，论文提出注入非信息性数据点的策略：当 $|S| < B$ 时，补充 $B - |S|$ 个噪声数据点使输入集合达到阈值 $B$，从而保证效用函数在任意子集大小上都有意义。

## 实验关键数据

### 主实验：Sum-of-Unanimity 博弈中的近似精度

| 方法 | AUCC（↓更好） | ARE（↓更好） | 每迭代运行时间 |
|------|:---:|:---:|:---:|
| **FGSV（本文）** | **最低** | **最低** | **最快之一** |
| Permutation | 较高 | 较高 | 较快 |
| Group Testing | 高 | 高 | 慢（有内部优化） |
| Complementary Contribution | 中等 | 中等 | 中等 |
| One-for-All | 中等 | 中等 | 较快 |
| KernelSHAP | 高 | 高 | 慢 |
| Unbiased KernelSHAP | 高 | 高 | 慢 |
| LeverageSHAP | 中等 | 中等 | 中等 |

在 $n \in \{64, 128, 256\}$ 下，固定 20,000 次效用评估预算，FGSV 在所有问题规模上 AUCC 和 ARE 均最优。

### 消融实验 / 应用实验

**版权归属（Stable Diffusion LoRA 微调 + FlickrLogo-27）**：

| 场景 | SRS（GSV-based） | FSRS（FGSV-based） |
|------|:---:|:---:|
| 无攻击（每品牌30张单组） | 合理分配 | 合理分配 |
| Shell Company Attack（Google/Sprite 拆分为 20/10 两组） | Google/Sprite 估值膨胀，其他品牌被压缩 | **估值保持稳定，不受攻击影响** |

关键发现：在 "A logo by Vodafone" 提示词下，SRS 被攻击后 Google/Sprite 总份额竟超过 Vodafone 本身；FSRS 则始终给出合理的版权归属。

**可解释 AI（Diabetes 数据集 + Ridge 回归）**：

| 方法 | 跨分组方案的稳定性 |
|------|:---:|
| GSV | 不同分组策略下类别排名不一致（如"年轻"/"中年"/"老年"各自在至少一种方案下排名第一） |
| **FGSV** | **所有分组方案下排名一致**（"年轻"组始终最高） |

### 关键发现

1. FGSV 在固定计算预算下收敛速度和最终精度均优于 7 种基线方法
2. FGSV 有效防御 shell company attack，而 GSV 完全被攻破
3. FGSV 在不同分组方案下提供一致且可解释的组级估值

## 亮点与洞察

1. **问题发现的价值**：识别出 GSV 的 shell company attack 漏洞，这是组级数据估值中一个全新的、具有实际影响的安全威胁
2. **优雅的理论结果**：FGSV 的唯一性定理（Theorem 1）简洁有力——满足五条公理的方法恰好就是个体 Shapley 值之和，无需额外的复杂构造
3. **算法设计的数学驱动**：通过超几何分布的集中性质和 $\mu$ 函数的光滑性，将 $O(n^2)$ 复杂度降至 $O(n \cdot \text{Poly}(\log n))$，理论与实践紧密结合
4. **配对 Monte Carlo 的方差缩减**：直接估计差值而非独立估计再相减，看似简单但对精度提升显著
5. **实用性强**：版权归属和可解释 AI 两个应用场景都是真实且重要的

## 局限性

1. **仅防御 shell company attack**：对 copier attack（复制高价值数据）无直接防御能力，论文仅讨论了预处理去重等 ad-hoc 方案
2. **效用函数假设**：Assumption 2（二阶算法稳定性）虽然在 SGD 和 IF 下被验证，但对更复杂的模型（LLM、扩散模型等）是否成立尚不清楚
3. **小集合效用的处理**：注入噪声数据点的策略虽有原理支撑，但阈值 $B$ 的选择和噪声分布对结果的敏感性未充分讨论
4. **实验规模有限**：SOU 实验最大仅 $n=256$，版权归属仅 4 个品牌各 30 张图，难以评估在真实大规模场景下的表现
5. **仅支持单组估值**：每次调用 Algorithm 1 只估计一个组的 FGSV，多组同时估计时虽仍优于基线，但需组数 $o(n)$

## 相关工作与启发

- **与 Data Shapley 系列的关系**：FGSV 建立在 Data Shapley（Ghorbani & Zou, 2019）、Beta Shapley（Kwon & Zou, 2022）等工作基础上，但首次系统研究了组级场景的忠实性
- **与 KernelSHAP 的连接**：KernelSHAP 及其变体（Unbiased KernelSHAP、LeverageSHAP）作为个体级估计基线参与比较，FGSV 通过直接估计组级目标绕开了误差累积
- **版权 AI 方向**：Shapley Royalty Share（Wang et al., 2024）的 FSRS 改进版本可直接用于生成式 AI 的版权补偿
- **启发**：这种"先证明唯一性→再设计高效算法"的范式值得借鉴；shell company attack 的发现也提醒数据市场设计者关注博弈论层面的鲁棒性

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|:---:|------|
| 创新性 | ⭐⭐⭐⭐ | 新漏洞 + 新公理 + 新算法，三者有机结合 |
| 理论深度 | ⭐⭐⭐⭐⭐ | 唯一性定理、复杂度分析、稳定性验证，数学功底扎实 |
| 实验充分性 | ⭐⭐⭐ | 应用场景有说服力但规模偏小 |
| 实用价值 | ⭐⭐⭐⭐ | 对数据市场和版权归属有直接应用价值 |
| 写作质量 | ⭐⭐⭐⭐ | 结构清晰，数学推导严谨，图示直观 |
| **综合** | **⭐⭐⭐⭐** | 理论驱动的优秀工作，问题定义到解决方案一气呵成 |

## 与相关工作的对比

| 方法 | 估值层级 | 忠实性 | Shell Company Attack 防御 | 复杂度 | 核心局限 |
|------|:---:|:---:|:---:|:---:|------|
| Data Shapley (Ghorbani & Zou, 2019) | 个体 | N/A | N/A | $O(n \epsilon^{-2} \log n)$ | 无法直接用于组级场景 |
| Group Shapley / GaI (Jullum et al., 2021; Wang et al., 2024) | 组（原子化） | ✗ | ✗ | $O(K!)$（$K$ 为组数） | 受分组策略影响，可被攻击 |
| KernelSHAP / Unbiased KernelSHAP | 个体→求和 | ✓（间接） | ✓（间接） | 高（内部优化开销大） | 作为个体方法用于组估计时误差累积 |
| Beta Shapley (Kwon & Zou, 2022) | 个体 | N/A | N/A | 类似 Data Shapley | 降权小集合项，不满足 Efficiency 公理 |
| **FGSV（本文）** | **组（直接）** | **✓** | **✓** | $O(n \cdot \text{Poly}(\log n))$ | 仅防 shell company attack，不防 copier attack |

- **GSV vs FGSV 的本质区别**：GSV 将组视为原子单元套用个体 Shapley 公式，忽略了组内成员的个体信息；FGSV 定义为组内成员 Shapley 值之和，保留了个体贡献的可追溯性。这使得 FGSV 对其他组的任意重新划分具有不变性（Faithfulness 公理）。
- **与 KernelSHAP 系列的关系**：KernelSHAP 及其变体本质上是个体级 Shapley 值的近似算法。若用它们估计每个成员的 SV 再求和得到 FGSV，理论上需 $O(\alpha_0 n^2 \text{Poly}(\log n))$ 次效用评估（误差经 $\sqrt{s_0}$ 放大），比 Algorithm 1 慢一个 $n$ 量级。
- **与 Shapley Royalty Share (SRS) 的关系**：SRS 是 Wang et al. (2024) 基于 GSV 提出的版权补偿份额。本文将其改进为 FSRS（使用 FGSV 替代 GSV），在保持归一化分配的同时防御了 shell company attack。

## 启发与关联

- **"先证唯一性再设计算法"的研究范式**：本文先通过公理化证明 FGSV 是唯一解，再基于数学结构设计高效近似算法。这一范式对数据估值、机制设计等领域的后续工作具有方法论启发。
- **Shell company attack 对数据市场的警示**：随着 AI 数据市场逐步落地，GSV 的这一漏洞可能被实际利用——数据供应商可通过注册多个子公司来膨胀估值。FGSV 提供了理论上完备的防御。
- **与 Copier Attack 的互补挑战**：Shell company attack 是"拆分自己"来获利，copier attack 是"复制别人"来获利。论文指出 FGSV 不直接防御后者，但可结合去重预处理或 do-utility 设计。两类攻击的统一防御仍是开放问题。
- **超几何分布的集中性质在估值中的应用**：Key Observation 2 利用超几何分布的指数集中来减少需要估计的项数，这一技巧可推广到其他涉及子集采样的 combinatorial estimation 问题。
- **小集合效用填充策略**：对于 LLM 等在小数据上无法有效训练的模型，注入非信息性数据的策略（灵感来自 machine unlearning）值得在其他需要处理 "small coalition" 的合作博弈场景中借鉴。
- **可能的扩展方向**：(1) 动态分组场景下的 FGSV 更新；(2) 将 faithfulness 概念引入联邦学习中的贡献评估；(3) 大规模生成模型（如 LLM）训练数据的组级版权归属。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Prediction via Shapley Value Regression (ViaSHAP)](../../ICML2025/others/prediction_via_shapley_value_regression.md)
- [\[NeurIPS 2025\] FACE: Faithful Automatic Concept Extraction](face_faithful_automatic_concept_extraction.md)
- [\[NeurIPS 2025\] Equivariance by Contrast: Identifiable Equivariant Embeddings from Unlabeled Finite Group Actions](equivariance_by_contrast_identifiable_equivariant_embeddings_from_unlabeled_fini.md)
- [\[NeurIPS 2025\] A Unified Framework for Provably Efficient Algorithms to Estimate Shapley Values](a_unified_framework_for_provably_efficient_algorithms_to_estimate_shapley_values.md)
- [\[ACL 2025\] Value Residual Learning](../../ACL2025/others/value_residual_learning.md)

</div>

<!-- RELATED:END -->
