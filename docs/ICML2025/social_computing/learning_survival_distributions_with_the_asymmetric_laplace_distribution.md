---
title: >-
  [论文解读] Learning Survival Distributions with the Asymmetric Laplace Distribution
description: >-
  [ICML2025][生存分析] 提出基于非对称拉普拉斯分布 (ALD) 的参数化生存分析方法，通过神经网络学习 ALD 的三个参数（位置、尺度、不对称性），实现连续、闭式的生存分布估计，在判别性和校准性上全面优于现有参数化与非参数化方法。
tags:
  - ICML2025
  - 生存分析
  - 非对称拉普拉斯分布
  - 参数化模型
  - 分位数回归
  - 最大似然估计
---

# Learning Survival Distributions with the Asymmetric Laplace Distribution

**会议**: ICML2025  
**arXiv**: [2505.03712](https://arxiv.org/abs/2505.03712)  
**代码**: 论文附有补充材料代码  
**领域**: 生存分析 / 概率建模  
**关键词**: 生存分析, 非对称拉普拉斯分布, 参数化模型, 分位数回归, 最大似然估计

## 一句话总结

提出基于非对称拉普拉斯分布 (ALD) 的参数化生存分析方法，通过神经网络学习 ALD 的三个参数（位置、尺度、不对称性），实现连续、闭式的生存分布估计，在判别性和校准性上全面优于现有参数化与非参数化方法。

## 研究背景与动机

生存分析（又称 time-to-event 分析）旨在预测给定协变量下事件发生的时间分布，广泛用于医疗、金融、工程等领域。核心挑战在于**右删失 (right-censoring)**：部分样本在观测结束时事件尚未发生。

现有方法分为三类，各有局限：

- **参数化方法**（指数分布、Weibull、对数正态）：假设固定分布形式，灵活性不足，真实数据偏离假设时性能下降
- **半参数方法**（Cox 比例风险模型、DeepSurv）：依赖比例风险假设，高删失率时可靠性降低
- **非参数方法**（DeepHit、CQRNN）：灵活但产生离散/分段常数估计，难以提取连续分布摘要（均值、中位数、分位数等），且计算开销大

本文的核心动机是：**能否找到一个既有闭式解（参数化优势）又足够灵活（非参数优势）的分布**？非对称拉普拉斯分布 (ALD) 恰好满足这一需求。

## 方法详解

### 非对称拉普拉斯分布 (ALD) 定义

ALD 由三个参数 $(\theta, \sigma, \kappa)$ 控制，分别对应位置、尺度和不对称性：

$$f_{\text{ALD}}(y;\theta,\sigma,\kappa) = \frac{\sqrt{2}}{\sigma}\frac{\kappa}{1+\kappa^2} \begin{cases} \exp\left(\frac{\sqrt{2}\kappa}{\sigma}(\theta-y)\right), & y \geq \theta \\ \exp\left(\frac{\sqrt{2}}{\sigma\kappa}(y-\theta)\right), & y < \theta \end{cases}$$

其 CDF 同样有闭式表达，使得均值、中位数、众数、方差和任意分位数均可**解析计算**。

关键性质：通过 $q = \kappa^2/(\kappa^2+1)$ 的重参数化，ALD 可自然地与分位数回归联系起来。

### 神经网络架构

模型采用共享编码器 + 三个独立预测头的结构：

- **共享编码器**：全连接层 + ReLU 激活，提取协变量特征
- **三个输出头**：分别预测 $\theta$、$\sigma$、$\kappa$，均通过 Exp 激活保证非负
- **残差连接**：增强梯度流与训练稳定性

### 最大似然学习

根据事件是否被观测，分别构建似然：

$$-\mathcal{L}_{\text{ALD}} = \sum_{n \in \mathcal{D}_O} \log f_{\text{ALD}}(y_n|\mathbf{x}_n) + \sum_{n \in \mathcal{D}_C} \log S_{\text{ALD}}(y_n|\mathbf{x}_n)$$

其中 $\mathcal{D}_O$ 为观测事件集合（$e=1$），$\mathcal{D}_C$ 为删失数据集合（$e=0$），$S_{\text{ALD}} = 1 - F_{\text{ALD}}$ 为生存函数。

**相比 CQRNN 的关键优势**：
- CQRNN 需要设定伪值 $y^* = 1.2 \max_i y_i$（对数据敏感）和近似删失分位数 $q_c$（精度受网格粒度限制）
- 本方法直接最大化生存概率，无需额外超参数
- 产生连续参数化分布，而非离散分位数点

### 与 CQRNN 的对比

CQRNN 基于 Portnoy 估计器的 pinball loss，需要预定义分位数网格 $q=\{0.1, 0.2, \ldots, 0.9\}$ 并逐一估计 $\theta_q$。本方法只需学习三个参数即可得到完整分布，避免了分位数网格选择和伪值调参的困难。

## 实验关键数据

### 数据集配置

在 14 个合成数据集 + 7 个真实数据集上评估，涵盖不同删失比例（0.20 ~ 0.80）和特征维度（1 ~ 14）。真实数据集来自肿瘤学、心脏病学等多个领域：

| 数据集 | 特征数 | 训练集 | 测试集 | 删失比例 |
|--------|--------|--------|--------|----------|
| METABRIC | 9 | 1523 | 381 | 0.42 |
| WHAS | 6 | 1310 | 328 | 0.57 |
| SUPPORT | 14 | 7098 | 1775 | 0.32 |
| GBSG | 7 | 1785 | 447 | 0.42 |
| TMBImmuno | 3 | 1328 | 332 | 0.49 |
| BreastMSK | 5 | 1467 | 367 | 0.77 |
| LGGGBM | 5 | 510 | 128 | 0.60 |

### 总体性能汇总（21 个数据集 × 9 个指标 = 189 次比较）

| 对比方法 | 显著胜出 | 显著落后 | 持平 |
|----------|----------|----------|------|
| vs. DeepHit | **113** | **22** | 54 |
| vs. LogNorm MLE | **113** | **6** | 70 |
| vs. DeepSurv | **77** | **27** | 85 |
| vs. CQRNN | **43** | **22** | 124 |

### 关键发现

- **IBS 指标**：ALD 在 21 个数据集中对 LogNorm、DeepSurv、DeepHit **全部获胜**，对 CQRNN 19 胜 1 负
- **校准性指标**：在 $S(t|\mathbf{x})$ 和 $f(t|\mathbf{x})$ 的 slope/intercept 校准上，ALD 全面领先传统参数化和非参数化方法
- **判别性指标**（C-Index）：ALD 优势主要体现在与 DeepHit 的对比中（15 胜 0 负），对 CQRNN 表现相当
- 所有实验重复 10 次，使用 Wilcoxon 签名秩检验判定显著性

## 亮点与洞察

1. **优雅的数学框架**：利用 ALD 的闭式 PDF/CDF，将生存分析回归到经典参数化模型的简洁形式，同时保持了足够的灵活性
2. **连续分布的实用价值**：与 DeepHit（离散化）和 CQRNN（分位数点）不同，ALD 输出连续分布，可直接计算均值、中位数、方差和任意分位数
3. **极少的超参数**：无需调节分位数网格、伪值等额外参数，大幅降低调参难度
4. **校准性突出**：在分布层面的校准指标上表现尤为强劲，说明 ALD 确实能更好地拟合真实生存分布的形状
5. **统一处理删失类型**：通过简单修改似然即可适配左删失等其他删失类型

## 局限与展望

1. **分布假设仍然存在**：虽然 ALD 比正态/Weibull 更灵活，但仍属于单峰分布族，面对多峰生存分布时可能力不从心
2. **ALD 支撑域包含负数**：$t < 0$ 在生存分析中无意义，作者声称实际中不太会出现，但缺乏理论保证
3. **仅在结构化数据上验证**：未涉及影像、时间序列等高维非结构化协变量
4. **MAE 指标不稳定**：在 MAE 上对 CQRNN/DeepSurv 互有胜负，说明点预测方面优势不够明显
5. **混合分布扩展**：可考虑 ALD 混合模型以处理更复杂的生存分布
6. **竞争风险场景**：当前框架仅处理单一事件，未扩展到多事件竞争风险设定

## 评分

- 新颖性: ⭐⭐⭐⭐ — ALD 用于生存分析是新颖的切入点，但 ALD 本身和分位数回归联系已被充分研究
- 实验充分度: ⭐⭐⭐⭐⭐ — 21 个数据集、9 个指标、10 次重复、统计检验，实验设计非常扎实
- 写作质量: ⭐⭐⭐⭐ — 数学推导清晰，与 CQRNN 的对比深入透彻
- 价值: ⭐⭐⭐⭐ — 为生存分析提供了一个实用且优雅的参数化新选项

<!-- RELATED:START -->

## 相关论文

- [Distribution-Aware Robust Learning from Long-Tailed Data with Noisy Labels](../../ECCV2024/social_computing/distribution-aware_robust_learning_from_long-tailed_data_with_noisy_labels.md)
- [Learning from Neighbors: Category Extrapolation for Long-Tail Learning](../../CVPR2025/social_computing/learning_from_neighbors_category_extrapolation_for_long-tail_learning.md)
- [Noise-Robustness Through Noise: A Framework Combining Asymmetric LoRA with Poisoning MoE](../../NeurIPS2025/social_computing/noise-robustness_through_noise_a_framework_combining_asymmetric_lora_with_poison.md)
- [Gradient Extrapolation for Debiased Representation Learning](../../ICCV2025/social_computing/gradient_extrapolation_for_debiased_representation_learning.md)
- [GraphKeeper: Graph Domain-Incremental Learning via Knowledge Disentanglement and Preservation](../../NeurIPS2025/social_computing/graphkeeper_graph_domain-incremental_learning_via_knowledge_disentanglement_and_.md)

<!-- RELATED:END -->
