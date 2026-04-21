---
title: >-
  [论文解读] Conditional Distribution Compression via the Kernel Conditional Mean Embedding
description: >-
  [NeurIPS 2025][distribution compression] 首次提出针对**条件分布**（而非联合分布）的压缩算法，利用核条件均值嵌入（KCME）定义新度量 AMCMD，并设计线性时间算法 ACKIP 构建保留条件分布统计特性的压缩数据集。
tags:
  - NeurIPS 2025
  - distribution compression
  - kernel methods
  - conditional mean embedding
  - RKHS
  - data compression
---

# Conditional Distribution Compression via the Kernel Conditional Mean Embedding

**会议**: NeurIPS 2025  
**arXiv**: [2504.10139](https://arxiv.org/abs/2504.10139)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: distribution compression, kernel methods, conditional mean embedding, RKHS, data compression

## 一句话总结

首次提出针对**条件分布**（而非联合分布）的压缩算法，利用核条件均值嵌入（KCME）定义新度量 AMCMD，并设计线性时间算法 ACKIP 构建保留条件分布统计特性的压缩数据集。

## 研究背景与动机

**领域现状**：现有分布压缩方法（如 Kernel Herding、Kernel Thinning）专注于无标签数据，通过最小化 MMD 构建压缩集来近似原始分布 $\mathbb{P}_X$。这些方法在减少数据量的同时保持统计保真度方面取得了显著成功。

**现有痛点**：对于有标签数据 $\{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^n$，没有任何现有方法直接针对条件分布族 $\mathbb{P}_{Y|X}$ 进行压缩。现有方法要么忽略标签信息，要么只能间接通过压缩联合分布 $\mathbb{P}_{X,Y}$ 来实现。

**核心矛盾**：核条件均值嵌入（KCME）的估计代价为 $\mathcal{O}(n^3)$（需要矩阵求逆），这使得大规模数据下的条件分布压缩在计算上不可行。直觉上，直接压缩条件分布应优于通过联合分布间接压缩，但高计算成本阻碍了实际应用。

**本文目标**：如何以线性时间 $\mathcal{O}(n)$ 代价构建一个压缩集 $\mathcal{C}$，使得压缩后的条件分布 $\tilde{\mathbb{P}}_{Y|X=\mathbf{x}} \approx \mathbb{P}_{Y|X=\mathbf{x}}$ 对几乎所有 $\mathbf{x}$ 成立。

**切入角度**：利用 tower property（塔性质）的关键观察——在条件分布压缩的目标中，$\mathbb{E}_{\mathbf{x} \sim \mathbb{P}_X}[\langle \mu_{Y|X=\mathbf{x}}, h(\mathbf{x}) \rangle] = \mathbb{E}_{(\mathbf{x},\mathbf{y}) \sim \mathbb{P}_{X,Y}}[h(\mathbf{x})(\mathbf{y})]$，从而避免显式估计 KCME，将复杂度从 $\mathcal{O}(n^3)$ 降至 $\mathcal{O}(n)$。

**核心 idea**：通过定义条件分布的度量 AMCMD 并利用塔性质简化估计，设计首个线性时间条件分布压缩算法。

## 方法详解

### 整体框架

提出四种方法用于有标签数据分布压缩，分为两大类：
- **联合分布压缩**：JKH（贪心）+ JKIP（联合优化）
- **条件分布压缩**：ACKH（贪心）+ ACKIP（联合优化）

核心流程：从原始数据集 $\mathcal{D} = \{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^n$ 构建小规模压缩集 $\mathcal{C} = \{(\tilde{\mathbf{x}}_j, \tilde{\mathbf{y}}_j)\}_{j=1}^m$（$m \ll n$），优化目标为最小化原始分布与压缩集分布之间的差异度量。

### 关键设计

#### 1. AMCMD 度量

**功能**：定义条件分布族之间的距离度量。

**核心公式**：
$$\text{AMCMD}[\mathbb{P}_{X^*}, \mathbb{P}_{Y|X}, \mathbb{P}_{Y'|X'}] = \sqrt{\mathbb{E}_{\mathbf{x} \sim \mathbb{P}_{X^*}} \left[\|\mu_{Y|X=\mathbf{x}} - \mu_{Y'|X'=\mathbf{x}}\|_{\mathcal{H}_l}^2\right]}$$

**设计动机**：(1) 引入独立的概率测度 $\mathbb{P}_{X^*}$ 使其比已有 KCD 度量更具通用性；(2) 满足度量的正定性和三角不等式（Theorem 4.1）；(3) 具有闭式估计器（Lemma 4.3）。

#### 2. 塔性质降复杂度 (Lemma 4.7)

**功能**：消除对 KCME 显式估计的需求。

**核心思路**：对任意向量值函数 $h: \mathcal{X} \to \mathcal{H}_l$：
$$\mathbb{E}_{\mathbf{x} \sim \mathbb{P}_X}[\langle \mu_{Y|X=\mathbf{x}}, h(\mathbf{x}) \rangle_{\mathcal{H}_l}] = \mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim \mathbb{P}_{X,Y}}[h(\mathbf{x})(\mathbf{y})]$$

**设计动机**：KCME 估计需 $\mathcal{O}(n^3)$（矩阵求逆），但令 $h(\mathbf{x}) = \tilde{\mu}_{Y|X=\mathbf{x}}^{\mathcal{C}}$ 后可用经验平均直接计算，降至 $\mathcal{O}(n)$。

#### 3. ACKIP 算法

**功能**：联合优化压缩集中所有点对，最小化 $\text{AMCMD}^2[\hat{\mathbb{P}}_X, \hat{\mathbb{P}}_{Y|X}, \tilde{\mathbb{P}}_{Y|X}]$。

**目标函数**：
$$\mathcal{J}^{\mathcal{D}}(\tilde{\mathbf{X}}, \tilde{\mathbf{Y}}) = \frac{1}{n}\text{Tr}(K_{\mathbf{X}\tilde{\mathbf{X}}} W_{\tilde{\mathbf{X}}\tilde{\mathbf{X}}} L_{\tilde{\mathbf{Y}}\tilde{\mathbf{Y}}} W_{\tilde{\mathbf{X}}\tilde{\mathbf{X}}} K_{\tilde{\mathbf{X}}\mathbf{X}}) - \frac{2}{n}\text{Tr}(L_{\mathbf{Y}\tilde{\mathbf{Y}}} W_{\tilde{\mathbf{X}}\tilde{\mathbf{X}}} K_{\tilde{\mathbf{X}}\mathbf{X}})$$

其中 $W_{\tilde{\mathbf{X}}\tilde{\mathbf{X}}} = (K_{\tilde{\mathbf{X}}\tilde{\mathbf{X}}} + \lambda I)^{-1}$。

**设计动机**：ACKH（贪心）由于不能回溯调整已选点且有四次方复杂度 $\mathcal{O}(m^4 + m^3 n)$，ACKIP 通过联合优化将复杂度降至 $\mathcal{O}(m^3 + m^2 n)$，快 $m$ 倍。

### 损失函数/训练策略

- 对连续响应使用高斯核，对离散响应使用指示核
- 通过梯度下降联合优化压缩集中所有点
- 初始化：从原始数据集中均匀随机采样 $m$ 个点
- 正则化参数 $\lambda$ 控制 KCME 估计的平滑程度

## 实验关键数据

### 主实验

**真实条件分布匹配 ($m=500$)**：ACKIP 在 AMCMD² 和所有测试函数的 RMSE 上优于所有方法。

**Superconductivity 数据集 ($n=10000, m=250$)**：

| 方法 | $h(y)=y$ RMSE | $h(y)=y^2$ RMSE | 排名 |
|------|-------------|--------------|------|
| ACKIP | **最低** | **最低** | 1 |
| ACKH | 次低 | 次低 | 2 |
| JKIP | 中等 | 中等 | 3 |
| JKH | 较高 | 较高 | 4 |
| Random | 最高 | 最高 | 5 |

### 消融实验

**条件 vs 联合压缩**：

| 对比 | 结论 |
|------|------|
| ACKIP vs JKIP | 直接压缩条件分布 > 通过联合分布间接压缩 |
| JKIP vs JKH | 联合优化 > 贪心选择 |
| ACKIP vs ACKH | 联合优化 > 贪心选择（且复杂度更低） |

**Imbalanced 分类数据集 ($m=250$)**：ACKIP 在 4 类中 3 类估计条件概率 $\mathbb{P}(Y=c|X)$ 的 RMSE 最低，仅用全数据 3% 即可接近全数据性能。ACKH 在 3 类中甚至不如随机采样。

### 关键发现

1. ACKIP 在所有实验（连续/离散、合成/真实数据）中一致性最优
2. 贪心方法（JKH, ACKH）受限于不能回溯修改，性能明显弱于联合优化
3. 条件分布压缩方法在条件期望估计上系统优于联合分布压缩方法
4. 压缩集大小 $m = 250$ 时 ACKIP 已可接近 $n = 10000$ 全数据的性能

## 亮点与洞察

1. **理论贡献深刻**：AMCMD 满足严格度量性质（正定性+三角不等式），有闭式估计器且一致性可证
2. **复杂度突破**：通过塔性质将 KCME 相关计算从 $\mathcal{O}(n^3)$ 降至 $\mathcal{O}(n)$，是使条件分布压缩可行的关键
3. **统一框架**：提出了联合分布与条件分布压缩的完整方法族，便于系统比较
4. **实用价值**：KCME 广泛应用于条件密度估计、贝叶斯优化、强化学习等领域，本文的压缩方法可显著扩展其适用范围

## 局限与展望

1. **无收敛保证**：ACKIP 缺乏严格的理论收敛证明，仅有经验收敛观察
2. **梯度依赖**：核梯度在图、文本等离散结构数据上不适用或难以解释
3. **核选择**：算法性能依赖核函数和超参数（带宽、正则化）的选择
4. **可扩展性**：虽然复杂度为线性，但 $\mathcal{O}(m^3)$ 项限制了压缩集大小
5. 可考虑结合 Kernel Thinning 的子集选择策略处理梯度不可用的场景

## 相关工作与启发

- **Kernel Herding** [Chen et al. 2010]：经典的无标签分布压缩方法，本文将其扩展到联合/条件分布
- **Kernel Thinning** [Dwivedi & Mackey 2021]：限制压缩集为原始数据子集的方法，可能更适合梯度不可用的场景
- **KCME** [Song et al. 2009]：核条件均值嵌入理论基础，本文通过分布压缩使其在大数据下可行

## 评分

⭐⭐⭐⭐ (4/5)

理论创新突出（AMCMD + 塔性质降复杂度），实验系统全面，但缺乏收敛保证和对高维复杂数据的验证。

<!-- RELATED:START -->

## 相关论文

- [Avoiding Leakage Poisoning: Concept Interventions Under Distribution Shifts](../../ICML2025/interpretability/avoiding_leakage_poisoning_concept_interventions_under_distribution_shifts.md)
- [SLiM: One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression](../../ICML2025/interpretability/slim_one-shot_quantization_and_sparsity_with_low-rank_approximation_for_llm_weig.md)
- [Distribution-Based Feature Attribution for Explaining the Predictions of Any Classifier](../../AAAI2026/interpretability/distribution-based_feature_attribution_for_explaining_the_predictions_of_any_cla.md)
- [VITAL: More Understandable Feature Visualization through Distribution Alignment and Relevant Information Flow](../../ICCV2025/interpretability/vital_more_understandable_feature_visualization_through_distribution_alignment_a.md)
- [Toward Real-world Text Image Forgery Localization: Structured and Interpretable Data Synthesis](toward_real-world_text_image_forgery_localization_structured_and_interpretable_d.md)

<!-- RELATED:END -->
