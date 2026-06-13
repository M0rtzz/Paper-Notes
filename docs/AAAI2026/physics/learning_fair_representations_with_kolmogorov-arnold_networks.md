---
title: >-
  [论文解读] Learning Fair Representations with Kolmogorov-Arnold Networks
description: >-
  [AAAI 2026][物理/科学计算][Kolmogorov-Arnold网络] 提出将Kolmogorov-Arnold网络（KAN）引入对抗去偏框架，利用KAN的样条函数架构提供理论上的Lipschitz连续性和平滑性保证，并设计自适应 $\lambda$ 更新机制动态平衡公平性与准确率，在UCI大学录取数据集上实现了公平性指标的显著提升。
tags:
  - "AAAI 2026"
  - "物理/科学计算"
  - "Kolmogorov-Arnold网络"
  - "对抗去偏"
  - "公平性"
  - "自适应惩罚"
  - "大学录取"
---

# Learning Fair Representations with Kolmogorov-Arnold Networks

**会议**: AAAI 2026  
**arXiv**: [2511.11767](https://arxiv.org/abs/2511.11767)  
**代码**: 无  
**领域**: 公平性机器学习 / 可解释AI  
**关键词**: Kolmogorov-Arnold网络, 对抗去偏, 公平性, 自适应惩罚, 大学录取

## 一句话总结

提出将Kolmogorov-Arnold网络（KAN）引入对抗去偏框架，利用KAN的样条函数架构提供理论上的Lipschitz连续性和平滑性保证，并设计自适应 $\lambda$ 更新机制动态平衡公平性与准确率，在UCI大学录取数据集上实现了公平性指标的显著提升。

## 研究背景与动机

**领域现状**：机器学习模型在大学录取、医疗等高风险决策场景中广泛应用，但训练数据中嵌入的历史偏见常导致模型对边缘化群体产生歧视性预测。特别是在大学录取场景中，来自低收入家庭和第一代大学生群体系统性面临资源不平等。

**现有方法及痛点**：

**预处理方法**：修改输入数据分布来消除偏见，但会改变数据完整性，不适合需要全面评估的录取场景

**后处理方法**：在模型输出上调整，对模型内部表示影响有限

**过程中方法（对抗去偏）**：直接在训练过程中加入公平性约束，是最合适的选择。但现有的基于MLP的对抗去偏框架存在两个核心问题：
   - **公平性-准确率权衡的不稳定性**：Zhao等人证明了满足精确DP（Demographic Parity）的分类器误差下界为两组基准率差的一半，导致训练过程中公平与准确的博弈不稳定
   - **黑箱不可解释**：标准神经网络无法显式表达特征贡献，在录取等敏感场景中缺乏可信度

**核心矛盾**：对抗去偏框架中，公平性惩罚系数 $\lambda$ 的选取高度依赖人工调参，且在训练过程中固定 $\lambda$ 难以应对动态变化的公平性-准确率权衡。

**本文切入角度**：用KAN替代MLP作为对抗去偏框架中的分类器和对抗器。KAN基于Kolmogorov-Arnold表示定理，用可学习的B样条函数替代固定激活函数，天然具有Lipschitz连续性和 $\beta$-平滑性，使对抗训练更稳定。同时引入自适应 $\lambda$ 更新策略解决手动调参问题。

## 方法详解

### 整体框架

框架采用经典的对抗去偏结构（min-max博弈），但将分类器 $f$ 和对抗器 $g$ 都替换为KAN网络：

- **分类器 $f$**：接受输入特征 $x$，预测标签 $y$（录取/不录取）
- **对抗器 $g$**：接受分类器输出 $f(x)$，试图推断敏感属性 $z$（低收入/非低收入、第一代大学生/非第一代）
- **优化目标**：$\min_{\theta_f} \max_{\theta_g} \mathcal{L}_{\mathcal{Y}}(f(x), y) - \lambda \cdot \mathcal{L}_{\mathcal{Z}}(g(f(x)), z)$

分类器要最小化预测损失同时最大化对抗器的损失（使对抗器无法从预测中推断敏感属性），对抗器则试图最小化自己的损失。$\lambda$ 控制公平性约束的强度。

### 关键设计

1. **KAN架构替代MLP**:

    - 功能：用基于B样条的KAN取代传统的全连接网络作为分类器和对抗器
    - 核心思路：KAN将任意多变量函数表示为单变量函数的有限叠加和复合，即 $f(x_1, ..., x_n) = \sum_{q=1}^{2n+1} \Phi_q(\sum_{r=1}^n \phi_{q,r}(x_r))$，其中每个 $\phi$ 用B样条参数化
    - 设计动机：(1) 每个样条函数 $h(x)$ 是分段多项式且二阶连续可微，保证Lipschitz连续性（Lemma 1），使模型对输入扰动鲁棒；(2) 样条函数的 $\beta$-平滑性（Lemma 2）保证梯度变化有界，使对抗训练收敛稳定；(3) 样条函数的可视化和分解提供了模型可解释性

2. **自适应 $\lambda$ 更新机制**:

    - 功能：在每个训练epoch后根据当前公平性状态动态调整惩罚系数 $\lambda$
    - 核心思路：计算公平性差距 $\delta = (\tau - p\%\text{-Rule}) / \tau$（$\tau$ 为目标公平性阈值），然后更新 $\lambda \leftarrow \text{clip}(\lambda + \eta \cdot \delta, 0.1, 1.0)$
    - 设计动机：当公平性不足时增大 $\lambda$ 加强约束，当已经满足目标时减小 $\lambda$ 释放准确率空间。clip操作保证 $\lambda$ 在安全范围内，避免极端值导致训练崩溃

3. **多优化器探索**:

    - 功能：在KAN框架下比较Adam、Optimistic Adam（OAdam）和ADOPT三种优化器
    - 核心思路：OAdam利用过去梯度的外推加速min-max博弈收敛，ADOPT针对非平稳环境自适应调整学习率
    - 设计动机：KAN的高复杂性使训练对优化器敏感，不同优化器在公平性-准确率权衡上表现各异，需要经验性探索

### 损失函数 / 训练策略

训练分为两阶段交替进行：
1. 冻结分类器，训练对抗器最小化 $\mathcal{L}_{\mathcal{Z}}(g(f(x)), z)$
2. 训练分类器最小化 $\mathcal{L}_{\mathcal{Y}}(f(x), y) - \lambda \cdot \mathcal{L}_{\mathcal{Z}}(g(f(x)), z)$

KAN使用B样条阶数 $k=3$（三次样条），支持网格细化（grid refinement）逐步提升模型容量。

## 实验关键数据

### 主实验（公平性与准确率比较）

| 模型 | 优化器 | Acc | AUROC | p%-Rule(低收入) | p%-Rule(第一代) | DP Gap(低收入) | DP Gap(第一代) |
|------|--------|-----|-------|----------------|----------------|---------------|---------------|
| KAN(D₁) | Adam | 74.92 | 80.27 | 85.79 | 89.65 | 0.028 | 0.035 |
| KAN(D₁) | ADOPT | 74.58 | 81.45 | 89.21 | 90.92 | 0.047 | 0.055 |
| KAN(D₂) | OAdam | 81.69 | 86.26 | **99.25** | **99.55** | 0.026 | 0.030 |
| KAN(D₂) | ADOPT | **82.51** | **86.68** | **99.25** | **99.70** | 0.032 | 0.037 |
| MLP-B₁(D₁) | Adam | 72.55 | 72.53 | 98.32 | 99.61 | 0.009 | 0.011 |
| MLP-B₁'(D₂) | Adam | 74.87 | 71.77 | 93.97 | 97.79 | 0.010 | 0.004 |
| ExpGrad-B₂(D₁) | Adam | 72.44 | 76.12 | 95.22 | 97.59 | 0.024 | 0.012 |
| ROAD-B₃(D₁) | Adam | 74.92 | 80.95 | 83.10 | 83.91 | 0.085 | 0.080 |

在大数据集D₂上，KAN+ADOPT配置同时达到最高准确率（82.51%）和接近完美的公平性（p%-Rule 99.25%/99.70%）。

### 消融实验（样条阶数影响）

| 配置 | 关键表现 | 说明 |
|------|---------|------|
| KAN k=3 | 最稳定收敛 | 表达力与平滑性的最佳平衡点 |
| KAN k=4 | p%-Rule略有提升 | 更高阶样条能更好建模公平边界 |
| KAN k=5 | 收敛出现波动 | 存在平滑性-维度权衡，训练损失出现尖峰 |
| MLP基线 | 公平性高但准确率低 | 缺乏自适应λ策略导致过度牺牲准确率 |

### 关键发现

- KAN在大数据集D₂上优势更显著：准确率比MLP基线高约7.6%（82.51 vs 74.87），同时p%-Rule更高（99.25 vs 93.97）
- ADOPT优化器最适合KAN的样条架构，在准确率和公平性间达到最佳平衡
- 自适应 $\lambda$ 策略是KAN优于MLP基线的关键因素之一，MLP基线使用固定 $\lambda$ 往往过度牺牲准确率来换取公平性
- 训练损失中KAN出现的尖峰是KAN保持万能逼近性的固有特征（平滑性-维度权衡），不影响最终收敛

## 亮点与洞察

- 首次将KAN引入对抗去偏框架，不仅是架构替换，还提供了理论分析（Lipschitz连续性+$\beta$-平滑性）解释为什么KAN适合对抗训练
- 自适应 $\lambda$ 设计简洁实用：基于p%-Rule与目标阈值的差距线性调整，加上clip保证稳定性，避免了昂贵的超参搜索
- 在真实录取数据上验证而非合成数据，增强了实际意义

## 局限与展望

- 仅验证了二分类（录取/不录取）和二值敏感属性，多类别和交叉敏感属性的场景未探索
- 公平性定义仅考虑Demographic Parity，未涵盖Equalized Odds等条件公平定义
- KAN的训练复杂度高于MLP（B样条的系数学习），在大规模数据上的可扩展性是隐忧
- 缺少与更多近期公平性方法的比较（如FairMixup、LAFTR等）
- 实验仅在UCI录取数据上进行，需要更多领域的验证

## 相关工作与启发

- KAN的Lipschitz连续性和平滑性分析可以推广到其他需要稳定对抗训练的场景（如GAN训练、鲁棒分类）
- 自适应惩罚系数的思路类似于约束优化中的增广拉格朗日方法，但这里更简单直接
- 结合KAN的可解释性（每条边的样条函数可视化）做公平性审计是一个有前景的方向

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Catastrophic Forgetting in Kolmogorov-Arnold Networks](catastrophic_forgetting_in_kolmogorov-arnold_networks.md)
- [\[AAAI 2026\] FlashKAT: Understanding and Addressing Performance Bottlenecks in the Kolmogorov-Arnold Transformer](flashkat_understanding_and_addressing_performance_bottlenecks_in_the_kolmogorov-.md)
- [\[ICLR 2026\] Initialization Schemes for Kolmogorov-Arnold Networks: An Empirical Study](../../ICLR2026/physics/initialization_schemes_for_kolmogorov-arnold_networks_an_empirical_study.md)
- [\[CVPR 2025\] KAC: Kolmogorov-Arnold Classifier for Continual Learning](../../CVPR2025/physics/kac_kolmogorov-arnold_classifier_for_continual_learning.md)
- [\[ICLR 2026\] Empirical Stability Analysis of Kolmogorov-Arnold Networks in Hard-Constrained Recurrent Physics-Informed Discovery](../../ICLR2026/physics/empirical_stability_analysis_of_kolmogorov-arnold_networks_in_hard-constrained_r.md)

</div>

<!-- RELATED:END -->
