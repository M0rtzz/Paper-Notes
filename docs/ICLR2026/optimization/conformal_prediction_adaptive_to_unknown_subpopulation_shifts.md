---
title: >-
  [论文解读] Conformal Prediction Adaptive to Unknown Subpopulation Shifts
description: >-
  [ICLR 2026][优化/理论][共形预测] 针对子群体偏移（subpopulation shift）下标准 conformal prediction 失效的问题，提出三种自适应算法：利用学习的 domain classifier 加权校准数据（Algorithm 1/2）或利用嵌入相似度加权（Algor…
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "共形预测"
  - "分布偏移"
  - "subpopulation shift"
  - "不确定性量化"
  - "LLM hallucination"
---

# Conformal Prediction Adaptive to Unknown Subpopulation Shifts

**会议**: ICLR 2026  
**arXiv**: [2506.05583](https://arxiv.org/abs/2506.05583)  
**代码**: 待确认  
**领域**: LLM评测  
**关键词**: 共形预测, 分布偏移, subpopulation shift, 不确定性量化, LLM hallucination

## 一句话总结
针对子群体偏移（subpopulation shift）下标准 conformal prediction 失效的问题，提出三种自适应算法：利用学习的 domain classifier 加权校准数据（Algorithm 1/2）或利用嵌入相似度加权（Algorithm 3），在不完美甚至无 domain 标签的情况下仍能保证覆盖率，并应用于视觉分类和 LLM 幻觉检测。

## 研究背景与动机

**领域现状**：Conformal prediction (CP) 为黑盒 ML 模型提供不确定性量化，保证在可交换数据下的边际覆盖率 $\Pr(Y_{\text{test}} \in C_\alpha(X_{\text{test}})) \geq 1-\alpha$。

**现有痛点**：实际中测试环境的子群体混合比例常与校准数据不同（subpopulation shift），导致标准 CP 在某些测试环境下严重欠覆盖或过覆盖。现有解决方案要么需要已知的分布偏移（Tibshirani et al.），要么使用 worst-case 阈值（max CP，严重过覆盖），要么需要完美的 group 标签（group-conditional CP）。

**核心矛盾**：Group-conditional CP 在理论上能解决子群体偏移问题，但它需要精确的 group 成员信息。Theorem 2.1 证明不完美的 domain classifier 会导致覆盖保证严重退化——如果 classifier 准确率为 $\gamma$，覆盖可以低至 $\max(0, \gamma - \alpha)$。

**本文目标**：在 domain 标签未知或不完美的情况下，设计有理论保证的自适应 CP 算法。

**切入角度**：不要求完美 domain classifier，而是利用 multicalibration / multiaccuracy 等更弱的假设来保证覆盖率；甚至完全不需要 domain classifier，仅用嵌入相似度加权。

**核心 idea**：用不完美的 domain classifier 或嵌入相似度来自适应加权校准数据，在未知子群体偏移下保持 conformal prediction 的覆盖保证。

## 方法详解

### 整体框架

测试环境是若干子群体的混合 $\mathbb{P}_{\text{test}} = \sum_{k=1}^K \lambda_k \mathbb{P}_k$，混合比例 $\lambda_k$ 未知且与校准数据不同，标准 CP 因此会在某些环境下严重欠覆盖或过覆盖。本文的共同思路是：先从测试数据估计它所属 domain 的概率分布 $\hat{\lambda}$，再用 $\hat{\lambda}$ 加权不同 domain 的校准分数，把固定阈值变成随测试分布自适应移动的阈值。

三种算法递进放松假设：
1. **Algorithm 1**：有 domain classifier + 逐点加权（需 multicalibrated classifier）
2. **Algorithm 2**：有 domain classifier + 批量平均加权（需 multiaccurate classifier，更弱要求）
3. **Algorithm 3**：无 domain classifier，用嵌入相似度加权（无理论保证但实验有效）

### 关键设计

**1. Theorem 2.1：先证明"不完美 classifier 插进 group-conditional CP 会崩"**

这是全文的动机基石，也解释了为什么需要后面三种算法。直觉上，既然 group-conditional CP 能处理子群体偏移，那把一个学出来的 domain classifier 当作 group 成员信息插进去似乎就够了。Theorem 2.1 证明这条捷径走不通：存在某些分布，使得用条件准确率为 $\gamma$ 的 classifier 时覆盖率退化到 $\max(0, \gamma - \alpha)$。也就是说，classifier 不够准时覆盖保证不是轻微下滑，而是可能直接失效。这从根本上排除了"朴素替换"的做法，把问题逼到"如何用更弱的假设换回覆盖保证"上。

**2. Algorithm 1：用逐点 domain 概率加权校准分数，在 Bayes 最优下恢复覆盖**

针对偏移导致的欠/过覆盖，Algorithm 1 不再要求精确的 group 标签，而是对每个测试点 $X_{\text{test}}$ 用 domain classifier $c(X_{\text{test}})$ 估计它属于各 domain 的概率向量 $\hat{\lambda}$，再据此加权各 domain 的校准分数来求阈值：

$$\hat{q}_\alpha \leftarrow \min_{\hat{q}} \quad \text{s.t.} \quad \sum_{k=1}^K \frac{\hat{\lambda}_k\, m_k(\hat{q}_\alpha)}{n_k + 1} \geq 1-\alpha$$

其中 $m_k(\hat{q})$ 是 domain $k$ 中分数不超过 $\hat{q}$ 的校准样本数。这样阈值会随测试点所处的 domain 混合比例自适应移动。Theorem 3.1 证明若 $c$ 是贝叶斯最优分类器则覆盖率 $\Pr(Y_{\text{test}} \in C_\alpha(X_{\text{test}})) \geq 1-\alpha$；Theorem 3.3 进一步把条件放松到 multicalibrated classifier 仍然成立——这正是绕开 Theorem 2.1 那条死路的关键：用 multicalibration 这个比"精确 group 标签"弱、但比"裸准确率"强的假设换回保证。

**3. Algorithm 2：把逐点估计换成批量平均，只需更弱的 multiaccuracy**

Algorithm 1 要求 multicalibration，其计算和样本复杂度都不低。Algorithm 2 观察到很多场景下我们关心的是整个测试批次的覆盖率，于是用测试集上 domain 预测概率的平均值 $\hat{\lambda} = \text{mean}_{i=1}^{n_{\text{test}}} c(X_{\text{test}}^i)$ 替代逐点估计，再走同样的加权阈值计算。代价是放弃逐点自适应，换来的好处是 Theorem 3.5 只需 multiaccurate classifier 就能保证覆盖——multiaccuracy 比 multicalibration 计算和样本复杂度都更低、更容易在实践中满足。这给了实践者一个按 classifier 质量取舍的选项。

**4. Algorithm 3：彻底丢掉 domain classifier，用嵌入相似度近似 domain 归属**

前两种算法都还依赖一个能预测 domain 概率的 classifier。Algorithm 3 把这个要求也去掉：它假设语义相似的数据更可能来自相同 domain，因此直接用嵌入空间的相似度来近似 domain 归属、加权校准数据。具体做法是先按与测试点的嵌入相似度排序、保留 top $\beta$ 比例的校准数据，再用 softmax 把相似度转成权重——令 $\gamma_i = h(z(X_{\text{test}}), z(X_i'))$，权重 $m = \text{Softmax}(\{\gamma_i/\sigma\})$，最后取加权分位数作为阈值。代价是它没有 A1/A2 那样的形式化覆盖保证，但好处是不需要任何 domain 标签或 classifier，可以直接套到任意预训练模型上，实验中也最实用。

### 损失函数 / 训练策略

- Domain classifier 训练：冻结预训练模型主体，只训练最后 3 层 FC (2048→1024→512→K)，Adam + CE loss
- 训练后用 Multi-domain temperature scaling 校准
- LLM 幻觉检测：用 GPT-4o 作为正确性评估器，LLaMA-3-8B 作为生成模型

## 实验关键数据

### 主实验

ImageNet 上 100 个测试环境的覆盖率分布（26 domains, ViT, LAC score, α=0.05）：

| 方法 | 平均覆盖率 | 标准差 | 评价 |
|------|-----------|--------|------|
| 目标覆盖率 | 0.950 | - | 理想值 |
| Standard CP | ~0.95 | **高** | 部分环境严重欠覆盖 |
| Max CP | ~0.99 | 低 | 严重过覆盖 |
| Conditional Calibration | ~0.94 | 中 | 某些环境欠覆盖 |
| Algorithm 1 (A1) | ~0.95 | **低** | 紧贴目标 |
| Algorithm 2 (A2) | ~0.95 | **低** | 紧贴目标 |
| Oracle | ~0.95 | **最低** | 理想上界 |
| Algorithm 3 (A3) | ~0.95 | 低 | 无 domain info 也有效 |

### 消融实验

| 实验维度 | 结果 |
|---------|------|
| 不同 score function (HPS/APS/LAC) | 一致有效 |
| 不同模型 (ResNet50/ViT/CLIP-ViT) | 一致有效 |
| 不同偏移程度 (α'=0.1/0.5/1.0) | 偏移越大，A1-A3 vs Standard CP 差距越大 |
| LLM 幻觉检测 (LLaMA-3-8B) | A3 显著降低 recall 的标准差 |

### 关键发现
- **Standard CP 在偏移下不可靠**：100 个测试环境中有相当比例严重欠覆盖，标准差很大。
- **Max CP 过于保守**：虽然保证覆盖但严重过覆盖（~0.99 vs 目标 0.95），导致预测集过大，实用性差。
- **A1/A2 紧贴目标**：覆盖率均值和标准差都接近 oracle，说明 multicalibrated/multiaccurate classifier 假设在实践中成立。
- **A3 无需 domain 信息也有效**：仅用嵌入相似度加权即可在大部分环境下保持覆盖，是最实用的方法。
- **LLM 幻觉检测**：Standard CP 的 recall 在不同测试环境下波动大，A3 显著降低波动，更可靠。

## 亮点与洞察
- **Theorem 2.1 揭示了 group-conditional CP 的根本缺陷**：不完美 group 信息会导致覆盖保证退化，这不是小问题而是可能完全失效（覆盖率可低至 $\gamma - \alpha$）。这为本文的方法提供了强动机。
- **从 Bayes-optimal → multicalibrated → multiaccurate 的假设放松链**：优雅地逐步弱化对 domain classifier 的要求，同时保持覆盖保证。实践者可根据自身 classifier 质量选择合适的算法。
- **Algorithm 3 的无监督自适应**：不需要任何 domain 信息，仅用嵌入相似度即可实现类似效果——这使得方法可以直接应用于任何预训练模型。

## 局限与展望
- **理论未利用样本独立性**：当前理论未利用不同 domain 样本间的独立性，导致轻微偏移时有些过覆盖。
- **Algorithm 3 无理论保证**：虽然实验有效但缺乏像 A1/A2 那样的形式化覆盖保证。
- **Score function 选择无指导**：没有提供关于何种 score function 在何种场景下最优的理论或实证指导。
- **LLM 实验局限**：仅在短答题 QA 上测试幻觉检测，未涉及更复杂的生成任务。
- **单一风险控制**：扩展到同时控制多种风险（幻觉+毒性+谄媚等）是重要的未来方向。

## 相关工作与启发
- **vs Tibshirani et al. (2020)**：需要已知的 covariate likelihood ratio，在高维 ML 中不可行。本文方法仅需学习 domain classifier 或用相似度近似。
- **vs Gibbs et al. (2024, Conditional Calibration)**：也用两阶段 domain classifier 方法，但假设完美 group 信息。本文的 Theorem 2.1 证明这一假设的脆弱性，并提出更鲁棒的替代。
- **vs Max/Robust CP (Cauchois et al.)**：保证覆盖但过于保守。本文自适应地匹配实际测试分布而非 worst case。

## 评分
- 新颖性: ⭐⭐⭐⭐ multicalibration/multiaccuracy + CP 的结合新颖，Theorem 2.1 有实际意义。
- 实验充分度: ⭐⭐⭐⭐ 视觉+LLM 双领域验证，多模型多 score function 多偏移度；A3 缺理论分析。
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，假设链条逻辑自洽；Algorithm 伪代码易读。
- 价值: ⭐⭐⭐⭐ 对 CP 在实际部署中的可靠性有直接贡献，LLM 幻觉检测应用有潜力。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Conformal Prediction as Bayesian Quadrature](../../ICML2025/optimization/conformal_prediction_as_bayesian_quadrature.md)
- [\[CVPR 2026\] Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score](../../CVPR2026/optimization/semi-supervised_conformal_prediction_with_unlabeled_nonconformity_score.md)
- [\[ICML 2025\] On Temperature Scaling and Conformal Prediction of Deep Classifiers](../../ICML2025/optimization/on_temperature_scaling_and_conformal_prediction_of_deep_classifiers.md)
- [\[NeurIPS 2025\] Conformal Prediction in The Loop: A Feedback-Based Uncertainty Model for Trajectory Optimization](../../NeurIPS2025/optimization/conformal_prediction_in_the_loop_a_feedback-based_uncertainty_model_for_trajecto.md)
- [\[CVPR 2025\] Conformal Prediction for Zero-Shot Models](../../CVPR2025/optimization/conformal_prediction_for_zero-shot_models.md)

</div>

<!-- RELATED:END -->
