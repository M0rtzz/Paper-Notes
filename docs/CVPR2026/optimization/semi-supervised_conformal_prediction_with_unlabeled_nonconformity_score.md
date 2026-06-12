---
title: >-
  [论文解读] Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score
description: >-
  [CVPR 2026][优化/理论][共形预测] 提出 SemiCP 框架，通过最近邻匹配（NNM）分数将无标签数据引入 conformal prediction 的校准流程，在标注数据极少时将平均覆盖率偏差降低最多 77%，同时缩小预测集。
tags:
  - "CVPR 2026"
  - "优化/理论"
  - "共形预测"
  - "半监督学习"
  - "不确定性量化"
  - "预测集"
  - "最近邻匹配"
---

# Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score

**会议**: CVPR 2026  
**arXiv**: [2505.21147](https://arxiv.org/abs/2505.21147)  
**代码**: 有（TorchCP 库集成）  
**领域**: 优化  
**关键词**: 共形预测, 半监督学习, 不确定性量化, 预测集, 最近邻匹配

## 一句话总结

提出 SemiCP 框架，通过最近邻匹配（NNM）分数将无标签数据引入 conformal prediction 的校准流程，在标注数据极少时将平均覆盖率偏差降低最多 77%，同时缩小预测集。

## 研究背景与动机

**Conformal Prediction (CP) 的核心价值**：CP 是一种模型无关、分布自由的不确定性量化框架，能生成包含真实标签的预测集并提供有限样本覆盖保证，在医疗诊断、金融决策等高风险场景中至关重要。

**Split CP 对标注数据的依赖**：标准 Split CP 需要一个标注好的 hold-out 校准集来估计阈值，但现实中标注数据往往稀缺，导致覆盖率在不同运行中高度不稳定。

**小校准集的理论缺陷**：校准集大小 $n$ 较小时，覆盖率服从 Beta 分布，方差约为 $\alpha(1-\alpha)/(n+2)$；例如 $n=10, \alpha=0.1$ 时有 10.7% 的概率覆盖率低于 80%。

**现有方法的局限**：插值法和修改 p-value 的方法是启发式的，缺乏有限样本保证；Few-shot CP 依赖可交换任务集合，实用性受限。

**无标签数据的丰富性**：在很多场景中无标签数据量远大于标注数据，是天然可利用的资源，但此前没有工作将其用于 CP 校准。

**条件覆盖的挑战更大**：在类条件或组条件覆盖场景中，每个子组需要独立校准数据，例如 ImageNet 1000 类、每类 100 个样本就需要 $10^5$ 个标注点，远超实际可获取量。

## 方法详解

### 整体框架

SemiCP 将校准集扩展为 $\mathcal{D} = \mathcal{D}_{\text{labeled}} \cup \mathcal{D}_{\text{unlabeled}}$，其中标注数据 $n$ 个、无标签数据 $N$ 个。对标注数据使用标准不一致性分数 $s_i = S(\mathbf{x}_i, y_i)$（如 THR/APS/RAPS），对无标签数据使用专门设计的无标签分数 $\tilde{s}_i = \tilde{S}(\tilde{\mathbf{x}}_i)$。两组分数合并后计算分位数阈值：

$$\hat{\tau}_{\text{SemiCP}} = \text{Quantile}\left(\{\tilde{s}_i\}_{i=1}^N \cup \{s_i\}_{i=1}^n, \frac{\lceil(n+N+1)(1-\alpha)\rceil}{n+N}\right)$$

测试时对测试样本 $\mathbf{x}_{\text{test}}$ 构建预测集 $\mathcal{C}(\mathbf{x}_{\text{test}}) = \{y : S(\mathbf{x}_{\text{test}}, y) \le \hat{\tau}_{\text{SemiCP}}\}$。整条流水线是「标注路径直接打标准分数」与「无标签路径经 NNM 纠偏打分」两路汇合到同一个校准池、估出阈值后再构建预测集：

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    L["标注校准集（n 个，含真标签）"] --> LS["标准不一致性分数 S<br/>THR / APS / RAPS"]
    U["无标签校准集（N 个）"] --> P["伪标签 ŷ = argmax softmax"]
    P --> PS["伪分数 S(·,ŷ) 系统性偏低<br/>定义伪偏差 Δ"]
    PS --> NNM["NNM 分数：伪分数空间匹配最近标注样本<br/>借其真实偏差纠偏"]
    LS --> MERGE["SemiCP 框架：合并 n+N 个分数<br/>取分位数得阈值"]
    NNM --> MERGE
    MERGE --> C["测试样本构建预测集<br/>保留分数 ≤ 阈值的标签"]
```

### 关键设计

**1. SemiCP 框架：把无标签分数并入校准池，用覆盖保证框住「增样降方差、引偏差」的权衡**

标准 Split CP 只拿 $n$ 个标注分数估阈值，$n$ 小则覆盖率方差大、换一次 split 结果就飘。SemiCP 的做法是把 $N$ 个无标签分数和 $n$ 个标注分数倒进同一个大小 $n+N$ 的校准池再取分位数（阈值公式见整体框架）。Theorem 1 给出覆盖率下界 $1-\alpha+\epsilon_{n,N}$，Theorem 2 进一步证明平均覆盖率偏差以 $\mathcal{O}(1/\sqrt{N})$ 的速率收缩——无标签样本越多，阈值估计越稳。代价是引入了偏差项 $\epsilon_{n,N}=\frac{N}{N+n}\big(F_S(\hat{\tau})-F_{\tilde{S}}(\hat{\tau})\big)$，它由「估计分数的 CDF 与真实分数 CDF 的差异」决定。这一项把后面无标签打分的目标钉死了：只要让无标签分数的分布逼近真实分数分布，$\epsilon_{n,N}\to 0$，就能既享受降方差、又几乎不引偏差。

**2. 伪偏差：朴素伪标签分数为何系统性偏低**

最自然的无标签打分是给样本打伪标签 $\hat{y}_i = \arg\max_j f_j(\tilde{\mathbf{x}}_i)$ 再代入分数函数（naive 法）。但伪标签恒取模型最自信的类，算出的分数系统性偏低，使分位数阈值被低估、覆盖率不足——这正是框架里偏差项 $\epsilon_{n,N}$ 变大的来源。为给纠偏定个靶子，本文把这种偏差量化为真实分数与伪分数之差 $\Delta(\tilde{\mathbf{x}}_i) = S(\tilde{\mathbf{x}}_i, \tilde{y}_i) - S(\tilde{\mathbf{x}}_i, \hat{y}_i)$。但 $\Delta$ 含未知真标签 $\tilde{y}_i$，无法直接算，需要 NNM 来估计。

**3. 最近邻匹配（NNM）分数：借标注样本的真实偏差纠正无标签伪分数**

偏差本身没法直接算（无标签样本没有真标签），NNM 的做法是在伪分数空间里给每个无标签样本找伪分数最接近的标注样本 $j = \arg\min_{j \in \{1,...,n\}} |S(\tilde{\mathbf{x}}_i, \hat{y}_i) - S(\mathbf{x}_j, \hat{y}_j)|$，再用这个标注样本可观测的真实偏差去纠正无标签的伪分数：$\tilde{S}_{\text{nnm}}(\tilde{\mathbf{x}}_i) = S(\tilde{\mathbf{x}}_i, \hat{y}_i) + S(\mathbf{x}_j, y_j) - S(\mathbf{x}_j, \hat{y}_j)$。核心假设是「伪分数相近的样本有相似的伪偏差」。Theorem 3 证明随标注数 $n$ 增大，NNM 分数的 CDF 渐近收敛到真实分数的 CDF（逼近误差有界、收敛速率由 $n$ 控制）；Fig. 3 也显示 NNM 分数的经验 PDF 与真实分布高度吻合，而 naive 伪分数明显偏低。这样 $\epsilon_{n,N}$ 被压到可忽略，SemiCP 才真正享受到无标签降方差的好处。

### 损失函数/训练策略

本方法是 **training-free** 的，不需要额外训练或优化。具体策略：

- 直接利用预训练分类器的 softmax 输出，无需访问训练数据
- 兼容任何标注分数函数（THR、APS、RAPS、SAPS 等）
- 可无缝集成到条件覆盖（类条件、组条件）设置中
- 可与 Interpolation、ClusterCP 等现有方法组合使用

理论保证（Theorem 1/2 的覆盖下界与收敛速率、Theorem 3 的 NNM 分数一致性）已在关键设计 1、3 中结合各自动机展开。

## 实验关键数据

### 主实验

| 数据集 | 标注数 $n$ | 无标签数 $N$ | 方法 | CovGap ↓ | AvgSize ↓ |
|---------|-----------|-------------|------|----------|-----------|
| CIFAR-10 | 20 | 4000 | Standard | 4.8 | 1.45 |
| CIFAR-10 | 20 | 4000 | **SemiCP** | **1.1** | **1.37** |
| CIFAR-10 | 10 | 4000 | Standard | 6.4 | 1.60 |
| CIFAR-10 | 10 | 4000 | **SemiCP** | **1.1** | **1.27** |
| ImageNet | 50 | 20000 | Standard | ~3.3 | ~75 |
| ImageNet | 50 | 20000 | **SemiCP** | **~2.1** | **~70.3** |

| 设置 | 数据集 | $n_{\text{avg}}$ | 方法 | CovGap ↓ | AvgSize ↓ |
|------|--------|-----------------|------|----------|-----------|
| 类条件 | CIFAR-100 | 10 | Standard | 7.75 | 18.9 |
| 类条件 | CIFAR-100 | 10 | **SemiCP** | **6.29** | **17.0** |
| 组条件 | CIFAR-100 | 10 | Standard | 较高 | 较大 |
| 组条件 | CIFAR-100 | 10 | **SemiCP** | 显著降低 | 显著缩小 |

### 消融实验

- **无标签数据量的影响**（ImageNet, $n=50$）：随着 $N$ 从 10 增长到 20000，CovGap 持续下降、AvgSize 持续缩小。即使只有 $N=10$ 个无标签样本，CovGap 也能降低约 0.1、AvgSize 降低约 0.2。
- **与 Interpolation 组合**（CIFAR-100）：Interpolation 单独使用时 CovGap 比 Standard 更大（不稳定），但 SemiCP+Interpolation 在 $n=10$ 时将 CovGap 从 9 降到 3.9，$n>40$ 时 AvgSize 接近 Oracle。
- **与 ClusterCP 组合**：SemiCP+ClusterCP 在所有 $n_{\text{avg}}$ 下进一步降低 CovGap 和 AvgSize。
- **NNM vs. Naive**：朴素伪标签分数的 PDF 系统性偏低，NNM 分数与真实分数分布高度吻合（Fig. 3）。

### 关键发现

1. **覆盖率偏差最高降低 77%**：CIFAR-10 上 20 个标注 + 4000 无标签，CovGap 从 4.8 降到 1.1，AvgSize 同时减小 5.7%。
2. **跨架构鲁棒性**：在 10 种不同架构（ResNet/MobileNet/ConvNet/EfficientNet/ViT 等）上均一致有效，平均 CovGap 从 3.3 降到 2.1。
3. **条件覆盖提升更大**：在类条件和组条件设置中，SemiCP 的改进幅度超过边际覆盖设置。
4. **数据效率高**：极少量无标签数据（$N=10$）即可产生可观改进。

## 亮点与洞察

- **Training-free 设计**：无需额外训练或优化，直接利用预训练模型输出，实现即插即用。与并行工作 [34] 需要优化 $N \times K$ 权重矩阵形成鲜明对比。
- **NNM 的巧妙直觉**：通过在伪分数空间中做最近邻匹配来估计偏差，利用了"伪分数相近的样本有相似偏差"这一经验观察，简洁有效。
- **理论与实验闭环**：Theorem 1-3 完整刻画了覆盖保证、收敛速率和分数一致性，实验验证与理论预测高度一致。
- **极强的兼容性**：可作为插件与 THR/APS/RAPS、Interpolation、ClusterCP、条件 CP 等任意方法组合。

## 局限性

- 理论分析依赖 i.i.d. 假设，比 CP 的可交换性假设更强
- 目前仅验证了分类任务，未扩展到回归场景
- NNM 在标注数据极少（如 $n < 5$）时匹配精度可能不足
- 伪分数空间中的最近邻匹配假设（相似伪分数 → 相似偏差）在分布偏移较大时可能不成立

## 相关工作

- **Split CP 及变体**：THR [Sadinle+ 2019]、APS [Romano+ 2020]、RAPS [Angelopoulos+ 2020]、SAPS [Huang+ 2023] 设计标注分数以提高预测集效率，但均依赖充足标注校准数据。
- **小校准集处理**：Interpolation [Johansson+ 2015] 插值阈值、ClusterCP [Ding+ 2023] 类聚类共享校准、Few-shot CP [Fisch+ 2021] 元学习，但均未利用无标签数据。
- **Prediction-Powered Inference (PPI)**：[Angelopoulos+ 2023] 利用模型预测缩紧置信区间，思路相关但面向自监督推断而非 CP 校准。
- **无监督校准**：[Mazuelas 2025] 通过 IPM 最小化估计标签权重，但需优化 $N \times K$ 矩阵，计算代价高且不可扩展。
- **SemiCP 的定位**：首个将无标签数据用于估计不一致性分数并改善 CP 校准稳定性的工作，与上述方法互补。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次将半监督引入 CP 校准，NNM 分数设计简洁且有理论保证
- 实验充分度: ⭐⭐⭐⭐⭐ — 3 数据集 × 3 分数函数 × 10 架构 × 1000 次重复 × 多种 CP 变体，极为全面
- 写作质量: ⭐⭐⭐⭐ — 问题动机清晰，理论推导完整，实验呈现系统化
- 价值: ⭐⭐⭐⭐ — 实用性强，training-free 且兼容现有方法，但局限于分类场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Conformal Prediction Adaptive to Unknown Subpopulation Shifts](../../ICLR2026/optimization/conformal_prediction_adaptive_to_unknown_subpopulation_shifts.md)
- [\[ICML 2025\] Conformal Prediction as Bayesian Quadrature](../../ICML2025/optimization/conformal_prediction_as_bayesian_quadrature.md)
- [\[ICML 2025\] On Temperature Scaling and Conformal Prediction of Deep Classifiers](../../ICML2025/optimization/on_temperature_scaling_and_conformal_prediction_of_deep_classifiers.md)
- [\[NeurIPS 2025\] Conformal Prediction in The Loop: A Feedback-Based Uncertainty Model for Trajectory Optimization](../../NeurIPS2025/optimization/conformal_prediction_in_the_loop_a_feedback-based_uncertainty_model_for_trajecto.md)
- [\[CVPR 2025\] Conformal Prediction for Zero-Shot Models](../../CVPR2025/optimization/conformal_prediction_for_zero-shot_models.md)

</div>

<!-- RELATED:END -->
