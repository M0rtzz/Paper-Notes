---
title: >-
  [论文解读] Rethinking Consistent Multi-Label Classification Under Inexact Supervision
description: >-
  [ICLR 2026][优化/理论][多标签分类] 提出 COMES 框架，通过一阶（Hamming loss）和二阶（Ranking loss）策略，为不精确监督下的多标签分类提供一致性风险估计器，无需估计标签生成过程或均匀分布假设。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "多标签分类"
  - "弱监督学习"
  - "部分多标签学习"
  - "互补多标签学习"
  - "风险一致性"
---

# Rethinking Consistent Multi-Label Classification Under Inexact Supervision

**会议**: ICLR 2026  
**arXiv**: [2510.04091](https://arxiv.org/abs/2510.04091)  
**领域**: 优化  
**关键词**: 多标签分类, 弱监督学习, 部分多标签学习, 互补多标签学习, 风险一致性  

## 一句话总结

提出 COMES 框架，通过一阶（Hamming loss）和二阶（Ranking loss）策略，为不精确监督下的多标签分类提供一致性风险估计器，无需估计标签生成过程或均匀分布假设。

## 研究背景与动机

多标签分类（MLC）要求每个实例关联多个相关标签，标注成本远高于单标签任务。为降低标注压力，研究者提出了两种弱监督范式：

- **部分多标签学习（PML）**：每个实例仅标注一个候选标签集，其中包含所有真正相关标签和部分无关的"假阳性"标签
- **互补多标签学习（CML）**：每个实例标注互补标签，表示实例不属于哪些类别

核心观察：PML 和 CML 在数学上是等价的——候选标签集的补集即为互补标签集。

**现有方法的局限性**：

1. 需要准确估计候选/互补标签的生成过程（即转移矩阵），但深度神经网络的过度自信问题使得估计不可靠
2. 假设均匀分布来规避估计问题，但该假设过于简化，无法处理现实中的类别不平衡
3. 许多方法独立建模不同标签，忽略标签之间的语义关联

## 方法详解

### 整体框架：COMES

COMES（COnsistent Multi-label classification under inExact Supervision）提出了一种新的数据生成过程假设和两种风险估计策略。

### 数据生成过程

假设候选标签通过逐类查询生成：如果第 $j$ 类与实例 $\boldsymbol{x}$ 无关，则以常数概率 $p_j$ 将其标注为非候选标签：

$$p(j \notin S | \boldsymbol{x}, j \notin Y) = p_j$$

**关键引理（Lemma 1）**：在上述假设下，非候选标签实例的条件密度等价于无关类的条件密度：

$$p(\boldsymbol{x} | s_j = 0) = p(\boldsymbol{x} | y_j = 0)$$

这一假设比均匀分布假设更通用，因为不同候选标签集的条件概率可以不同。

### 一阶策略：COMES-HL（基于 Hamming Loss）

将 MLC 分解为多个独立的二分类问题。通过 Theorem 1，Hamming loss 的 $\ell$-风险可等价表示为：

$$R_H^\ell(\boldsymbol{g}) = \mathbb{E}_{p(\boldsymbol{x})}\left[\frac{1}{q}\sum_{j=1}^q \ell(g_j(\boldsymbol{x}), 1)\right] + \sum_{j=1}^q \mathbb{E}_{p(\boldsymbol{x}|s_j=0)}\left[\frac{1-\pi_j}{q}(\ell(g_j(\boldsymbol{x}), 0) - \ell(g_j(\boldsymbol{x}), 1))\right]$$

通过构造无标签数据集 $\mathcal{D}_U$ 和条件数据集 $\mathcal{D}_j$，得到无偏风险估计器。为防止深度网络过拟合，使用绝对值函数包裹负项，得到修正风险估计器 $\tilde{R}_H^\ell$。

### 二阶策略：COMES-RL（基于 Ranking Loss）

考虑标签对之间的排序关系，利用对称损失函数假设 $\ell(z, \cdot) + \ell(-z, \cdot) = M$：

$$R_R^\ell(\boldsymbol{g}) = \sum_{1 \leq j < k \leq q}\left((1-\pi_j)\mathbb{E}_{p(\boldsymbol{x}|s_j=0)}[\ell(g_j - g_k, 0)] + (1-\pi_k)\mathbb{E}_{p(\boldsymbol{x}|s_k=0)}[\ell(g_j - g_k, 1)]\right)$$

使用 flooding 正则化技术缓解过拟合：$\tilde{R}_R^\ell = |\hat{R}_R^\ell - \beta| + \beta$。

### 损失函数

- COMES-HL 使用二元交叉熵作为代理损失
- COMES-RL 使用对称损失函数（如 sigmoid 损失）
- 类先验 $\pi_j$ 可通过现有的类先验估计方法从候选标签中估计

### 理论保证

| 性质 | COMES-HL | COMES-RL |
|------|----------|----------|
| 偏差有界 | $0 \leq \text{bias} \leq O(\Delta_j)$，$\Delta_j \to 0$ as $n \to \infty$ | $0 \leq \text{bias} \leq O(\Delta')$，$\Delta' \to 0$ as $n \to \infty$ |
| 估计误差收敛 | $O(\mathfrak{R}_n(\mathcal{G}) + \sqrt{\ln(1/\delta)/n})$ | $O(\mathfrak{R}_n(\mathcal{G}) + \sqrt{\ln(1/\delta)/n})$ |
| 一致性 | Bayes 最优 w.r.t. Hamming loss | Bayes 最优 w.r.t. Ranking loss |

## 实验关键数据

### 主实验：真实数据集（Ranking Loss ↓）

| 方法 | mirflickr | music_emotion | yeastBP | yeastCC | yeastMF |
|------|-----------|---------------|---------|---------|---------|
| BCE | 0.106 | 0.244 | 0.328 | 0.206 | 0.251 |
| CCMN | 0.106 | 0.224 | 0.328 | 0.210 | 0.245 |
| GDF | 0.159 | 0.278 | 0.501 | 0.504 | 0.495 |
| CTL | 0.130 | 0.266 | 0.498 | 0.467 | 0.471 |
| **COMES-HL** | **0.095** | **0.214** | **0.154** | **0.124** | 0.173 |
| **COMES-RL** | 0.106 | **0.213** | 0.166 | **0.117** | **0.151** |

### 方法对比特征

| 方法 | 无需均匀分布假设 | 无需估计生成过程 | 标签关联感知 | 支持多互补标签 |
|------|:---:|:---:|:---:|:---:|
| CCMN | ✓ | ✗ | ✓ | ✓ |
| CTL | ✗ | ✓ | ✗ | ✗ |
| GDF | ✗ | ✓ | ✗ | ✓ |
| **COMES-HL** | **✓** | **✓** | ✗ | **✓** |
| **COMES-RL** | **✓** | **✓** | **✓** | **✓** |

### 关键实验发现

1. COMES-HL 和 COMES-RL 在 6 个真实数据集上的 5 个评估指标中全面超越 SOTA 方法
2. 在 yeast 系列数据集上优势尤为明显，Ranking Loss 降低约 50%（如 yeastBP：0.154 vs 0.328）
3. COMES-RL 在标签关联性强的数据集上表现更优（如 yeastMF：0.151 vs 0.173）
4. 在不同标签生成过程（均匀/非均匀）的合成数据上均表现稳健

## 亮点与洞察

1. **理论贡献突出**：首次在不依赖转移矩阵估计和均匀分布假设的条件下，证明了不精确监督下 MLC 的一致性
2. **PML 与 CML 的统一处理**：利用两者的数学等价性，在同一框架下解决两个问题
3. **一阶与二阶策略互补**：COMES-HL 高效但忽略标签关联，COMES-RL 利用排序关系但计算成本更高，适应不同场景
4. **实用的数据生成假设**：基于"逐类查询无关性"的假设更贴合实际标注流程
5. **修正风险估计器设计精巧**：绝对值包裹和 flooding 正则化分别解决了一阶和二阶策略中的过拟合问题

## 局限性

1. 仅关注 Hamming loss 和 Ranking loss，未覆盖其他 MLC 评价指标（如 F1-measure）
2. 类先验 $\pi_j$ 的估计质量会影响最终性能，但论文未深入分析估计误差传播
3. 数据生成假设中 $p_j$ 为常数，可能无法完全描述复杂的标注行为
4. 二阶策略的计算复杂度为 $O(q^2)$，在标签空间很大时可能成为瓶颈
5. 实验中使用的数据集规模偏小，在大规模数据集上的表现有待验证

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 统一 PML/CML 并去除强假设的框架设计新颖
- **实验**: ⭐⭐⭐⭐ — 在多个数据集和指标上全面超越 SOTA，但数据集规模偏小
- **写作**: ⭐⭐⭐⭐ — 理论推导严谨完整，结构清晰
- **价值**: ⭐⭐⭐⭐ — 为弱监督 MLC 提供了坚实的理论基础和实用方法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] A Convergence Analysis of Adaptive Optimizers under Floating-Point Quantization](a_convergence_analysis_of_adaptive_optimizers_under_floating-point_quantization.md)
- [\[AAAI 2026\] Cost-Minimized Label-Flipping Poisoning Attack to LLM Alignment](../../AAAI2026/optimization/cost-minimized_label-flipping_poisoning_attack_to_llm_alignment.md)
- [\[ICLR 2026\] MT-DAO: Multi-Timescale Distributed Adaptive Optimizers with Local Updates](mt-dao_multi-timescale_distributed_adaptive_optimizers_with_local_updates.md)
- [\[AAAI 2026\] On the Learning Dynamics of Two-Layer Linear Networks with Label Noise SGD](../../AAAI2026/optimization/on_the_learning_dynamics_of_two-layer_linear_networks_with_label_noise_sgd.md)
- [\[ICML 2025\] Nearly Optimal Sample Complexity for Learning with Label Proportions](../../ICML2025/optimization/nearly_optimal_sample_complexity_for_learning_with_label_proportions.md)

</div>

<!-- RELATED:END -->
