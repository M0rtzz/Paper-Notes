---
title: >-
  [论文解读] Put CASH on Bandits: A Max K-Armed Problem for Automated Machine Learning
description: >-
  [NeurIPS 2025][CASH] 针对 AutoML 中的联合算法选择和超参数优化（CASH）问题，通过数据驱动分析揭示了 HPO 奖励分布的有界左偏特性，提出了专门适配该特性的极端 Bandit 算法 MaxUCB，在理论和实验上均优于现有方法。
tags:
  - NeurIPS 2025
  - CASH
  - Max K-Armed Bandit
  - 超参数优化
  - AutoML
  - Upper Confidence Bound
---

# Put CASH on Bandits: A Max K-Armed Problem for Automated Machine Learning

**会议**: NeurIPS 2025  
**arXiv**: [2505.05226](https://arxiv.org/abs/2505.05226)  
**代码**: [GitHub](https://github.com/amirbalef/CASH_with_Bandits)  
**领域**: AutoML / 多臂老虎机  
**关键词**: CASH, Max K-Armed Bandit, 超参数优化, AutoML, Upper Confidence Bound

## 一句话总结

针对 AutoML 中的联合算法选择和超参数优化（CASH）问题，通过数据驱动分析揭示了 HPO 奖励分布的有界左偏特性，提出了专门适配该特性的极端 Bandit 算法 MaxUCB，在理论和实验上均优于现有方法。

## 研究背景与动机

CASH（Combined Algorithm Selection and Hyperparameter Optimization）是 AutoML 的核心问题：在有限预算下，同时搜索最佳模型类型（如 LightGBM、XGBoost、MLP）及其最佳超参数配置。

**现有方法的两条路线**：

**联合搜索**（combined search）：将所有模型的超参数空间合并为一个层次化空间，直接用 BO 搜索。问题：高维条件空间导致 HPO 效率低下。

**分解搜索**（two-level CASH）：上层用 Bandit 选择模型，下层对选中模型运行 HPO。问题：上层 Bandit 的目标函数定义和分布假设不明确。

**核心矛盾**：现有的 Max K-Armed Bandit（MKB）算法假设奖励是重尾分布（如 Pareto），但 CASH 中 HPO 产生的奖励分布实际上是**有界、左偏、短尾**的。这个分布假设的错配导致现有极端 Bandit 算法在 CASH 上表现不佳。

**切入角度**：首先进行系统性的数据分析，揭示 HPO 奖励分布的真实特性，然后基于正确的分布假设设计新的 Bandit 算法，并证明其遗憾上界。

## 方法详解

### 整体框架

MaxUCB 是一个两层优化框架：上层是 Max K-Armed Bandit（选择要运行 HPO 的模型类），下层是标准的 HPO 方法（如贝叶斯优化）。每次上层选定一个模型类后，下层运行一次 HPO 迭代评估一个新配置，其负损失作为 Bandit 的奖励反馈。

### 关键设计

1. **数据驱动的分布分析（Section 3）**：作者在四个 AutoML 基准上分析了 HPO 奖励分布的生存函数，发现三个关键特性：

    - **有界性**：奖励（模型性能指标）本身有界，每个 arm 有不同的最大值
    - **短尾左偏**：奖励集中在最大值附近，极端事件不是离群值（与 MKB 常用的重尾假设相反）
    - **近似平稳**：最优 arm 不随时间改变

   基于这些观察，引入 Lemma 3.3 用两个分布相关常数 $L$ 和 $U$ 来刻画有界分布在最大值附近的形状：$L\epsilon \leq G(b-\epsilon) \leq U\epsilon$。实证表明 $L$ 值多数大于 1（而重尾分布的 $L$ 接近 0），这解开了先前文献中认为极端 Bandit 不可行的理论困境。

2. **MaxUCB 算法（Algorithm 1）**：核心创新在于探索 bonus 的设计。经典 UCB 用 $\sqrt{\frac{\alpha\log t}{n}}$ 作为 bonus，MaxUCB 改为 $\left(\frac{\alpha\log t}{n}\right)^2$。

   策略更新公式为：
    $U_i = \max(r_{i,1}, \ldots, r_{i,n_i}) + \left(\frac{\alpha\log(t)}{n_i}\right)^2$

   这个更快的浓度速率来源于有界分布中最大值的特殊集中性质。不好事件的概率可写为 $P(\text{Bad}) \leq O(e^{-n\sqrt{C(n)}} + nC(n))$，取 $C(n) = 1/n^2$ 可最小化该概率。

3. **理论分析（Theorem 4.2）**：对任意次优 arm $i$，MaxUCB 的次优拉取次数满足：

    $N_i(T) \leq \frac{T^{1-2L_{i^*}\alpha\sqrt{\Delta_i}}}{1-2L_{i^*}\alpha\sqrt{\Delta_i}} + 2\alpha\sqrt{U_i T}\log(T)$

   性能受次优间隔 $\Delta_i$、最优 arm 的左尾参数 $L_{i^*}$ 和次优 arm 的右尾参数 $U_i$ 控制。当 $L_{i^*}$ 较大（最优 arm 的极端值容易被采样到）且 $\Delta_i$ 较大时，次优拉取次数迅速衰减。

   **Corollary 4.3**（总遗憾界）：选择合适的 $\alpha$ 时，$R(T) \leq \mathcal{O}\left(\frac{K\log T}{\sqrt{T}}\max_{i} b_i\right)$。

### 超参数 $\alpha$ 的作用

$\alpha$ 控制探索-利用权衡。Equation 9 给出了理论指导：$\alpha = \frac{1}{4L_{i^*}\sqrt{\min\Delta_i}}\left(1 - \frac{2\log(\log T)}{\log T}\right)$。小 $L_{i^*}$ 或小 $\Delta_i$ 需要更大的 $\alpha$ 来增加探索。实验表明 $\alpha \approx 0.5$ 是一个鲁棒的默认值。

## 实验关键数据

### 主实验：四个 AutoML 基准上的 Sign Test

| 基准 (HPO方法) | MaxUCB p-value | Quantile BayesUCB p-value | Rising Bandits p-value | MaxUCB w/t/l |
|---|---|---|---|---|
| TabRepo [RS] | **0.00000** | **0.00000** | **0.00000** | 186/4/10 |
| TabRepoRaw [SMAC] | **0.00072** | **0.00261** | 0.42777 | 24/0/6 |
| YaHPOGym [SMAC] | **0.00880** | **0.00503** | **0.00074** | 64/0/39 |
| Reshuffling [HEBO] | - | - | - | 显著优于 combined search |

MaxUCB 和 Quantile Bayes UCB 是唯二在统计上显著优于联合搜索的方法。

### 消融实验：$\alpha$ 敏感性

| $\alpha$ 范围 | 特点 | 说明 |
|---|---|---|
| 低值（~0.1） | 小预算时表现好 | 快速锁定有希望的 arm，但可能过早利用 |
| 高值（~2.0） | 充足预算时表现好 | 更充分探索，最终性能更高 |
| 中间值（~0.5） | 鲁棒的 anytime 性能 | 在所有预算水平上均表现良好 |

### 关键发现

- **两层方法 vs 联合搜索**：几乎所有 Bandit 方法在早期（T=50）都优于联合搜索，证明分解策略在预算有限时特别有效
- **MaxUCB vs 其他 Bandit**：MaxUCB 在 anytime 和最终性能上均最优。经典 UCB 和 QoMax-SDA 表现最差，因为其分布假设错配
- **Rising Bandits** 在非平稳奖励的 YaHPOGym 上有竞争力，因为它建模了非平稳性
- 极端 Bandit 和经典 UCB 假设过于宽松或错配，导致在 CASH 上效率低下

## 亮点与洞察

- **数据驱动的方法论**：先分析真实分布特性，再设计算法，避免了假设错配的问题
- **解决了开放问题**：Nishihara et al. (2016) 指出极端 Bandit 在某些分布下不可行，本文证明这些负面结论不适用于 CASH 的实际分布
- 探索 bonus 的平方形式 $(\cdot)^2$ 而非 $\sqrt{\cdot}$ 是一个简洁而深刻的发现
- $L$ 和 $U$ 常数提供了一个统一的框架来描述不同分布族在极端值附近的行为

## 局限与展望

- 方法高度针对有界左偏分布，在重尾或对称分布上可能退化
- 平稳性假设在 HPO 早期可能不成立，需要 burn-in 阶段
- $\alpha$ 的自适应调整在理论上不可行（Locatelli & Carpentier, 2018），但可利用 meta-learning 信息
- 有界极端 Bandit 的下界仍是开放问题，分布最优性未证明
- 未考虑计算成本感知的资源分配（不同模型训练时间差异大）

## 相关工作与启发

- 与 Quantile Bayes UCB 的关系：两者都针对分布的高区域，但 MaxUCB 直接优化最大值而非分位数
- 可扩展到 NAS 中的 sub-supernet 选择，因为 NAS 的奖励分布有类似特性
- AutoML 系统（如 Auto-sklearn、Auto-WEKA）可直接集成 MaxUCB 作为上层选择策略

## 评分

- 新颖性: ⭐⭐⭐⭐ 数据驱动的分布分析开创了 MKB 在 CASH 上的正确应用方式
- 实验充分度: ⭐⭐⭐⭐⭐ 四个基准、多种对比方法、敏感性分析、统计检验齐全
- 写作质量: ⭐⭐⭐⭐⭐ 数据分析→理论→算法→实验的逻辑链非常清晰
- 价值: ⭐⭐⭐⭐ 对 AutoML 实践有直接指导意义，理论分析也有独立价值

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] EvaLearn: Quantifying the Learning Capability and Efficiency of LLMs via Sequential Problem Solving](evalearn_quantifying_the_learning_capability_and_efficiency_of_llms_via_sequenti.md)
- [\[NeurIPS 2025\] Ineq-Comp: Benchmarking Human-Intuitive Compositional Reasoning in Automated Theorem Proving on Inequalities](ineq-comp_benchmarking_human-intuitive_compositional_reasoning_in_automated_theo.md)
- [\[NeurIPS 2025\] Creativity or Brute Force? Using Brainteasers as a Window into the Problem-Solving Abilities of Large Language Models](creativity_or_brute_force_using_brainteasers_as_a_window_into_the_problem-solvin.md)
- [\[CVPR 2025\] LoTUS: Large-Scale Machine Unlearning with a Taste of Uncertainty](../../CVPR2025/llm_evaluation/lotus_large-scale_machine_unlearning_with_a_taste_of_uncertainty.md)
- [\[NeurIPS 2025\] Learning Generalizable Shape Completion with SIM(3) Equivariance](learning_generalizable_shape_completion_with_sim3_equivariance.md)

<!-- RELATED:END -->
