---
title: >-
  [论文解读] Gradient Aligned Regression via Pairwise Losses
description: >-
  [ICML2025][回归损失函数] 提出 GAR（Gradient Aligned Regression），通过在标签空间引入两个成对差异损失（误差方差 + 负Pearson相关系数）来对齐预测函数与真实函数的梯度，并利用 DRO 鲁棒聚合三个子损失，实现与传统回归损失相同的线性复杂度，同时在多个基准上超越 MAE/MSE 及对比学习方法。
tags:
  - ICML2025
  - 回归损失函数
  - 成对损失
  - 梯度对齐
  - 分布鲁棒优化
  - Pearson相关系数
---

# Gradient Aligned Regression via Pairwise Losses

**会议**: ICML2025  
**arXiv**: [2402.06104](https://arxiv.org/abs/2402.06104)  
**代码**: [GitHub](https://github.com/DixianZhu/GAR)  
**领域**: 回归 / 鲁棒性  
**关键词**: 回归损失函数, 成对损失, 梯度对齐, 分布鲁棒优化, Pearson相关系数

## 一句话总结

提出 GAR（Gradient Aligned Regression），通过在标签空间引入两个成对差异损失（误差方差 + 负Pearson相关系数）来对齐预测函数与真实函数的梯度，并利用 DRO 鲁棒聚合三个子损失，实现与传统回归损失相同的线性复杂度，同时在多个基准上超越 MAE/MSE 及对比学习方法。

## 研究背景与动机

### 传统回归的局限

传统回归损失（MAE、MSE、Huber）只关注逐样本预测误差 $\delta_{\mathbf{x}}^f = f(\mathbf{x}) - y$ 的大小，**无法捕捉样本间的关系模式**。例如两个模型误差分别为 $\{1,-1,1,-1\}$ 和 $\{1,1,1,1\}$，MAE/MSE 完全相同，但后者误差方差为零、保序性更好，且只需简单偏置修正即可达到零误差。

### 已有成对方法的不足

近期 RankSim、RNC、ConR 等方法在特征空间施加成对相似性约束：

- **计算开销大**：需要 $O(N^2)$ 的成对运算
- **信息损失**：将连续标签相似度转换为离散排名或正负对，存在不可逆的近似损失
- **缺乏理论解释**：未建立与函数梯度学习的联系

### 本文动机

直接在标签空间建模成对差异 $f(\mathbf{x}_i) - f(\mathbf{x}_j) \approx y_i - y_j$，既保留完整的标签差异信息，又能通过等价变换降到线性复杂度。

## 方法详解

### 整体框架

GAR 由三个损失组成：

$$\mathcal{L}_{\text{GAR}} = \text{DRO-Aggregate}(\mathcal{L}_c^{\text{MAE}},\; \mathcal{L}_{\text{diff}}^{\text{MSE}},\; \mathcal{L}_{\text{diffnorm}}^{p=2};\; \alpha)$$

### 损失1：常规 MAE 损失

$$\mathcal{L}_c^{\text{MAE}} = \frac{1}{N}\sum_{i=1}^{N}|f(\mathbf{x}_i) - y_i|$$

负责逐点拟合预测值与真实值。

### 损失2：成对差异损失 → 误差方差

原始定义为 $O(N^2)$ 的成对 MSE：

$$\mathcal{L}_{\text{diff}}^{\text{MSE}} = \frac{1}{N^2}\sum_{i}\sum_{j} \frac{1}{2}\big[(f(\mathbf{x}_i)-f(\mathbf{x}_j))-(y_i-y_j)\big]^2$$

**Theorem 1** 证明它等价于预测误差的方差：

$$\mathcal{L}_{\text{diff}}^{\text{MSE}} = \text{Var}(\delta_{\mathbf{x}}^f)$$

等价的线性复杂度形式为：

$$\mathcal{L}_{\text{diff}}^{\text{MSE}} = \frac{1}{N}\sum_{i=1}^{N}\big[(f(\mathbf{x}_i)-\bar{f})-(y_i-\bar{y})\big]^2$$

**直觉**：最小化误差方差意味着所有样本的误差趋于一致，增强保序性。

### 损失3：归一化成对差异 → 负 Pearson 相关

引入缩放因子后对成对差异做归一化，得到 $\ell_2$ 范数下的损失：

$$\mathcal{L}_{\text{diffnorm}}^{p=2} = 1 - \rho(f, y) = 1 - \frac{\text{Cov}(f, y)}{\sqrt{\text{Var}(f)\text{Var}(y)}}$$

**直觉**：最大化 Pearson 相关系数等同于捕捉预测函数与真实函数的"形状"（方向对齐），允许幅值不同。

### 理论洞察：梯度对齐

**Theorem 4**（核心定理）：对 $K$ 阶可微的确定性函数，成对标签差异的精确匹配等价于各阶梯度的精确匹配：

$$f(\mathbf{x}_1)-f(\mathbf{x}_2)=y_1-y_2,\;\forall (\mathbf{x}_1,\mathbf{x}_2) \iff \nabla^k f(\mathbf{x})=\nabla^k \mathcal{Y}(\mathbf{x}),\;k=1,...,K$$

证明基于中值定理和洛必达法则。这意味着成对损失**隐式地学习了真实函数的梯度场**。

### DRO 鲁棒聚合

三个子损失量纲差异大（如 MAE 可无界，Pearson 损失 $\in[0,2]$），简单算术/几何平均各有缺陷。GAR 采用基于 KL 散度的 DRO 聚合：

$$\mathcal{L}_{\text{GAR}}^{\text{KL}} = \alpha\log\Big(\frac{1}{M}\sum_{i=1}^{M}\mathcal{L}_i^{1/\alpha}\Big)$$

- $\alpha \to 0$：退化为 $\max$ 损失
- $\alpha = 1$：算术平均
- $\alpha \to +\infty$：几何平均

实验默认 $\alpha = 0.5$，兼顾对小损失的关注与数值稳定性。通过 $\mathcal{L}_{\max}$ 或 $\mathcal{L}_{\min}$ 归一化避免数值溢出。

### 复杂度

整体算法每次迭代仅需 $O(B)$（$B$ 为 batch size），与 MAE/MSE 相同，无需成对特征空间运算。

## 实验关键数据

### 合成数据集

| 数据集 | MAE (MAE↓) | MSE (MAE↓) | RNC (MAE↓) | **GAR (MAE↓)** |
|--------|-----------|-----------|-----------|-------------|
| Sine | 较差，少捕捉1-2个波峰 | 类似MAE | 中等 | **捕捉最多波峰** |
| Squared Sine | 仅捕捉最大振幅峰 | 类似MAE | 中等 | **几乎完全恢复真实模式** |

### 真实数据集（8个任务，5个表格+1个图像基准）

| 数据集 | 指标 | MAE | 最优竞品 | **GAR** | 提升 |
|--------|------|-----|---------|---------|------|
| Concrete | MAE↓ | 4.976 | 4.698(Huber) | **4.603** | 7.5%/2.0% |
| Concrete | Pearson↑ | 0.919 | 0.923(RNC) | **0.929** | 1.1%/0.6% |
| Wine | MAE↓ | 0.500 | 0.500(MAE) | **0.494** | 1.1%/1.1% |
| STS-B | Pearson↑ | 0.865 | 0.880(RNC) | **0.882** | 2.0%/0.2% |
| IMDB-WIKI | MAE↓ | 6.685 | 6.468(ConR) | **6.366** | 4.8%/1.6% |

- GAR 在 **全部8个任务** 的 MAE 指标上均优于或持平于所有基线
- 在 Pearson/Spearman 相关系数上也保持优势
- p-value 检验显示多数提升具有统计显著性

### 运行时间对比

| 方法 | 相对时间（vs MAE） |
|------|-------------------|
| MAE | 1.0× |
| RankSim | ~2.5× |
| RNC | ~1.8× |
| ConR | ~2.0× |
| **GAR** | **~1.0×** |

GAR 与 MAE 运行时间几乎相同，远快于所有在特征空间做成对运算的方法。

## 亮点与洞察

1. **优雅的等价变换**：$O(N^2)$ 成对损失被证明等价于误差方差和 Pearson 相关系数，降到 $O(N)$ 线性复杂度，理论清晰且实用
2. **梯度对齐的理论洞察**（Theorem 4）：首次建立成对标签差异学习与函数梯度匹配之间的等价关系，为方法提供深层数学直觉
3. **DRO 聚合机制**：单一超参数 $\alpha$ 统一了算术平均、几何平均和最大值之间的权衡，避免手动调节多个权重
4. **零额外计算开销**：与 MAE/MSE 同等效率，比 RankSim/RNC/ConR 快 1.8-2.5 倍
5. **广泛适用**：在表格回归和图像年龄预测等多种场景下均有效

## 局限与展望

1. **仅限干净数据**：作者明确限制了研究范围为无噪声、无异常值、无分布偏移的设置，面对脏数据的鲁棒性未验证
2. **超参数 $\alpha$ 的选择**：虽然减少到单一超参数，但最优 $\alpha$ 值仍需跨任务调参
3. **理论假设较强**：Theorem 4 要求函数 $K$ 阶可微且定义域开放，对非光滑或离散问题的适用性有限
4. **单目标回归**：仅展示了单维目标的情况，多目标回归场景的有效性待验证
5. **模型架构限制**：实验主要使用简单 FFNN 和 ResNet，在 Transformer 等大模型上的效果未探索
6. **对比学习方法的优势场景**：在特征空间的对比学习可能在表示学习方面有独特优势（如 RNC 的预训练范式），GAR 仅在标签空间操作可能难以捕捉

## 相关工作与启发

- **RankSim** (Gong et al., 2022)：标签相似度 → 排名正则化，但丢失连续信息
- **RNC** (Zha et al., 2023)：对比预训练 + 微调，效果好但计算贵
- **ConR** (Keramati et al., 2023)：对比正则化器，定义正负对方式不同
- **启发**：GAR 的思路（标签空间的简单数学等价变换）可推广到其他需要建模样本间关系的任务（如排序学习、不确定性量化）

## 评分

- 新颖性: ⭐⭐⭐⭐ （等价变换思路巧妙，梯度对齐理论新颖）
- 实验充分度: ⭐⭐⭐⭐ （8个真实任务+2合成+消融+运行时间，但缺少脏数据实验）
- 写作质量: ⭐⭐⭐⭐ （结构清晰、符号规范、定理完整，部分公式较密集）
- 价值: ⭐⭐⭐⭐ （实用性强、无额外计算开销、即插即用的损失函数提升）

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Rethinking Losses for Diffusion Bridge Samplers](../../NeurIPS2025/llm_evaluation/rethinking_losses_for_diffusion_bridge_samplers.md)
- [\[CVPR 2025\] Gradient-Guided Annealing for Domain Generalization](../../CVPR2025/llm_evaluation/gradient-guided_annealing_for_domain_generalization.md)
- [\[AAAI 2026\] Axis-Aligned Document Dewarping](../../AAAI2026/llm_evaluation/axis-aligned_document_dewarping.md)
- [\[ICML 2025\] Fully Heteroscedastic Count Regression with Deep Double Poisson Networks](fully_heteroscedastic_count_regression_with_deep_double_poisson_networks.md)
- [\[CVPR 2025\] Scene-Agnostic Pose Regression for Visual Localization](../../CVPR2025/llm_evaluation/scene-agnostic_pose_regression_for_visual_localization.md)

<!-- RELATED:END -->
