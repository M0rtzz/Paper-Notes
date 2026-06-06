---
title: >-
  [论文解读] Position: Don't Use the CLT in LLM Evals with Fewer Than a Few Hundred Datapoints
description: >-
  [ICML 2025 (Spotlight Position Paper)][推荐系统][中心极限定理 (CLT)] 本文作为立场论文，论证了在 LLM 评估数据量少于几百个样本时，基于中心极限定理 (CLT) 的置信区间严重低估不确定性，推荐使用贝叶斯可信区间或 Wilson 得分区间作为替代方案。
tags:
  - "ICML 2025 (Spotlight Position Paper)"
  - "推荐系统"
  - "中心极限定理 (CLT)"
  - "置信区间"
  - "贝叶斯可信区间"
  - "LLM 评估"
  - "小样本统计"
  - "Wilson 得分区间"
  - "Bootstrap"
---

# Position: Don't Use the CLT in LLM Evals with Fewer Than a Few Hundred Datapoints

**会议**: ICML 2025 (Spotlight Position Paper)

**arXiv**: [2503.01747](https://arxiv.org/abs/2503.01747)

**作者**: Sam Bowyer, Laurence Aitchison, Desi R. Ivanova

**领域**: 推荐系统 / LLM 评估 / 统计推断

**关键词**: 中心极限定理 (CLT), 置信区间, 贝叶斯可信区间, LLM 评估, 小样本统计, Wilson 得分区间, Bootstrap

**代码**: [bayes_evals](https://github.com/sambowyer/bayes_evals)

---

## 一句话总结

本文作为立场论文，论证了在 LLM 评估数据量少于几百个样本时，基于中心极限定理 (CLT) 的置信区间严重低估不确定性，推荐使用贝叶斯可信区间或 Wilson 得分区间作为替代方案。

---

## 研究背景与动机

### 核心问题

当前 LLM 评估 (evals) 中的统计不确定性量化几乎全部依赖 CLT，但越来越多的高价值基准测试数据量很小：

- **FrontierMath**: 约 300 道题，部分类别少于 3 个样本
- **AIME 2024**: 仅 15 道竞赛数学题
- **SWE-Bench Verified**: 500 个样本
- **MLE-Bench**: 75 个 Kaggle 竞赛
- **LiveBench**: 每个任务平均仅 55 个样本
- **CUAD**: 510 份法律文档，造价约 200 万美元

这些专业基准规模远小于 MMLU、GSM8K 等传统大规模基准（数千到上万样本）。

### 动机

CLT 依赖 $N \to \infty$ 的渐近性质，在小样本场景下置信区间可能：

1. 当 $\hat{\theta} = 0$ 或 $\hat{\theta} = 1$ 时，标准误为零，区间退化为单点
2. 区间边界超出 $[0,1]$ 范围
3. 实际覆盖率远低于名义覆盖率（如设定 95% 但实际远低于 95%）

---

## 方法详解

### 整体框架

本文系统性地在 **5 类实验场景** 中评估 CLT 的失败模式，并对比贝叶斯与频率学派替代方案：

| 场景 | CLT 是否适用 | 推荐方法 |
|------|-------------|---------|
| 单模型 IID 问题 | 小样本下失败 | Wilson 得分区间 / Beta-Bernoulli 贝叶斯 |
| 聚类问题 (clustered) | 小样本下失败 | 贝叶斯重要性采样 (Beta-Binomial) |
| 独立模型比较 | 小样本下失败 | 贝叶斯后验采样 |
| 配对模型比较 | 小样本下失败 | 配对贝叶斯重要性采样 |
| 非均值指标 (如 $F_1$) | 不可用 | 贝叶斯 Dirichlet-Categorical |

### 关键设计

#### 1. IID 设置下的贝叶斯方法

对于 Bernoulli 数据 $y_i \sim \text{Bernoulli}(\theta)$，使用共轭先验：

$$\theta | y_{1:N} \sim \text{Beta}\left(1 + \sum_{i=1}^{N} y_i, \; 1 + \sum_{i=1}^{N}(1-y_i)\right)$$

直接从 Beta 分布分位数得到精确可信区间，无需渐近近似。

#### 2. 聚类设置下的生成模型

$$d \sim \text{Gamma}(1,1), \quad \theta \sim \text{Beta}(1,1)$$
$$\theta_t \sim \text{Beta}(d\theta, d(1-\theta)), \quad y_{i,t} \sim \text{Bernoulli}(\theta_t)$$

其中 $d$ 控制聚类间难度差异，$\theta_t$ 为每个任务/聚类的表现。积分消去 $\theta_t$ 后：

$$Y_t \sim \text{BetaBin}(N_t, d\theta, d(1-\theta))$$

使用先验作为提议分布的重要性采样 ($K=10000$) 进行推断。

#### 3. Wilson 得分区间

$$\text{CI}_{1-\alpha}(\theta) = \frac{\hat{\theta} + \frac{z_{\alpha/2}^2}{2N}}{1 + \frac{z_{\alpha/2}^2}{N}} \pm \frac{z_{\alpha/2}}{2N\left(1 + \frac{z_{\alpha/2}^2}{N}\right)} \sqrt{4N\hat{\theta}(1-\hat{\theta}) + z_{\alpha/2}^2}$$

区间中心不再是样本均值 $\hat{\theta}$，避免了零宽度和越界问题。

### 损失函数 / 评估指标

本文不涉及训练损失，而是定义了评估覆盖率的实验框架：

- 对 100 个 $\theta$ 值 × 200 个数据集 × $N \in \{3, 10, 30, 100\}$
- 对每个数据集构建 100 个名义覆盖水平的区间
- 计算 **覆盖误差** = 平均 |实际覆盖率 − 名义覆盖率|

---

## 实验关键数据

### 主实验：各场景下覆盖误差 (Coverage Error)

| 方法 | IID (N=10) | IID (N=100) | 聚类 (N=10) | 配对 (N=10) |
|------|-----------|------------|------------|------------|
| CLT | 覆盖率远低于名义值 | 接近名义值 | 覆盖率严重不足 | 覆盖率严重不足 |
| Bootstrap (K=10000) | 覆盖率低于名义值 | 接近名义值 | 覆盖率不足 | 覆盖率不足 |
| Wilson Score | 接近名义值 ✓ | 接近名义值 ✓ | N/A (仅IID) | N/A |
| Clopper-Pearson | 过度保守 (过宽) | 接近名义值 | N/A (仅IID) | N/A |
| **贝叶斯 (本文推荐)** | **接近名义值 ✓** | **接近名义值 ✓** | **接近名义值 ✓** | **接近名义值 ✓** |

### LangChain 工具使用基准实测 (N=20)

| 模型 | CLT 区间问题 | 贝叶斯区间 |
|------|-------------|----------|
| GPT-4 | 区间超出 [0,1] | 合理区间 |
| Llama-2-70B | 区间退化为零 | 合理区间 |
| Mistral-7B | 区间超出 [0,1] | 合理区间 |
| GPT-3.5 | 严重低估不确定性 | 合理区间 |

### 关键发现

1. **CLT 在 N<100 时系统性失败**：实际覆盖率远低于设定的名义覆盖率
2. **Bootstrap 同样不可靠**：即使 K=10000 次重采样，小样本下覆盖率仍然不足
3. **Wilson 得分区间**在 IID 单模型场景下表现优异，且 SciPy 直接实现
4. **贝叶斯方法**是唯一在所有场景中都能达到正确覆盖率的方法
5. **$F_1$ 等非线性指标**：CLT 完全不可用，仅贝叶斯方法有效
6. **先验不匹配鲁棒性**：贝叶斯方法在先验偏差下仍优于 CLT

---

## 亮点与洞察

1. **实用性极强**：每个场景都提供了可直接复用的 Python 代码片段（3-10 行），贝叶斯方法实现成本极低
2. **Clopper-Pearson 与贝叶斯的关系**：CP 精确区间等价于移除均匀先验的贝叶斯可信区间，解释了其过度保守性
3. **贝叶斯模型比较的优势**：可直接计算 $\mathbb{P}(\theta_A > \theta_B | \text{data})$，频率学派无法给出此概率
4. **配对 vs 非配对**：配对贝叶斯方法在先验不匹配时比非配对版本更鲁棒
5. **计算开销可忽略**：贝叶斯推断的计算成本相比构建基准和运行 LLM 评估本身微不足道

---

## 局限性

1. **立场论文**：不提出新算法，主要是方法论推荐和系统性实证验证
2. **仅考虑 Bernoulli 类型评估**：对连续值指标（如 BLEU、ROUGE）的适用性未讨论
3. **先验选择**：虽然验证了鲁棒性，但未给出针对特定场景选择先验的系统指导
4. **模拟数据为主**：真实 LLM 评估场景的验证仅有 LangChain 一个案例

---

## 相关工作与启发

- **Miller (2024)** 提出在 LLM eval 中使用 CLT + 聚类标准误，本文证明该方案在小样本下仍然失败
- **Madaan et al. (2024)** 量化评估基准中的方差，但仍使用 CLT
- **Dubey et al. (2024)** (Llama 3 报告) 承认 CLT 不适用于 $F$-score 等指标，直接放弃报告置信区间
- **启发**：在推荐系统评估中，A/B 测试数据量有限时，贝叶斯方法可提供更可靠的统计推断

---

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 3 | 方法本身不新（Beta-Bernoulli、Wilson 都是经典方法），贡献在于系统性实证 |
| 实用性 | 5 | 提供可直接复用的代码，解决了 LLM 评估中的真实痛点 |
| 实验充分性 | 5 | 5 类场景 × 多种样本量 × 多种先验，消融极为全面（39 张图） |
| 写作质量 | 5 | 结构清晰，立论有力，代码与数学推导兼备 |
| **综合** | **4.5** | 对 LLM 评估社区有重要的方法论贡献，值得广泛采纳 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Position: The Right to AI](the_right_to_ai.md)
- [\[NeurIPS 2025\] Position: Towards Bidirectional Human-AI Alignment](../../NeurIPS2025/recommender/position_towards_bidirectional_human-ai_alignment.md)
- [\[AAAI 2026\] Preference is More Than Comparisons: Rethinking Dueling Bandits with Augmented Human Feedback](../../AAAI2026/recommender/preference_is_more_than_comparisons_rethinking_dueling_bandits_with_augmented_hu.md)
- [\[ICML 2025\] RLTHF: Targeted Human Feedback for LLM Alignment](rlthf_targeted_human_feedback_for_llm_alignment.md)
- [\[ICML 2026\] Position: Neglecting the Sustainability of AI is Fuelling a Global AI Arms Race](../../ICML2026/recommender/position_neglecting_the_sustainability_of_ai_is_fuelling_a_global_ai_arms_race.md)

</div>

<!-- RELATED:END -->
