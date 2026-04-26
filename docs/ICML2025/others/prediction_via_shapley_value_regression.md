---
title: >-
  [论文解读] Prediction via Shapley Value Regression (ViaSHAP)
description: >-
  [ICML2025][Shapley值] 提出 ViaSHAP，将 Shapley 值的计算融入模型训练过程，使得推理时通过对 Shapley 值求和直接得到预测，无需后验解释器，在表格数据上达到 XGBoost 级别的预测精度，同时 Shapley 值近似质量显著优于 FastSHAP。
tags:
  - ICML2025
  - Shapley值
  - 可解释性
  - KAN
  - 特征归因
  - 表格数据
  - 自解释模型
---

# Prediction via Shapley Value Regression (ViaSHAP)

**会议**: ICML2025  
**arXiv**: [2505.04775](https://arxiv.org/abs/2505.04775)  
**代码**: [GitHub](https://github.com/amrmalkhatib/ViaSHAP)  
**领域**: 可解释预测 / Explainable ML  
**关键词**: Shapley值, 可解释性, KAN, 特征归因, 表格数据, 自解释模型

## 一句话总结

提出 ViaSHAP，将 Shapley 值的计算融入模型训练过程，使得推理时通过对 Shapley 值求和直接得到预测，无需后验解释器，在表格数据上达到 XGBoost 级别的预测精度，同时 Shapley 值近似质量显著优于 FastSHAP。

## 研究背景与动机

- **核心矛盾**：Shapley 值具有局部准确性(local accuracy)、缺失性(missingness)、一致性(consistency)等理想的解释性质，是唯一同时满足这三条公理的特征归因方案。然而传统方法(KernelSHAP, FastSHAP)都是**后验(post-hoc)**计算，推理时额外引入大量开销。
- **KernelSHAP** 需要对每个实例单独求解加权最小二乘优化问题，需采样大量联盟(coalition)才能收敛。
- **FastSHAP** 虽然训练了一个参数化解释器来摊销推理成本，但仍然需要一个已训练好的黑盒模型作为"教师"，本质上仍是后验方案。
- **研究空白**：至今没有人将 Shapley 值计算作为**预测的手段**(prediction via Shapley values)，即先算 Shapley 值、再由其求和得到预测。

## 方法详解

### 核心思想

ViaSHAP 训练一个函数 $\phi^{\mathcal{V}ia}: X \to \mathbb{R}^{n \times d}$，对输入 $x$ 输出一个 $n \times d$ 的 Shapley 值矩阵，预测结果通过列求和获得：

$$\hat{y} = \sigma\!\left(\mathbf{1}^\top \phi^{\mathcal{V}ia}(x;\theta)\right)$$

其中 $\sigma$ 为链接函数（如 sigmoid 或 softmax）。这意味着模型**先计算每个特征对每个输出维度的贡献，再汇总成预测**。

### 训练目标——双重损失

训练同时优化两个目标：

$$\mathcal{L}(\theta) = \sum_{x \in X}\sum_{j \in M}\left(\beta \cdot \mathbb{E}_{p(S)}\left[\left(\mathcal{V}ia_j^{\text{SHAP}}(x^S) - \mathcal{V}ia_j^{\text{SHAP}}(\mathbf{0}) - \mathbf{1}_S^\top \phi_j^{\mathcal{V}ia}(x;\theta)\right)^2\right] - y_j \log(\hat{y}_j)\right)$$

- **Shapley 损失** $\mathcal{L}_\phi$：对随机采样的联盟 $S$，要求用被选特征的 Shapley 值之和来复现"仅用这些特征时的模型输出"，使 Shapley 值满足加权最小二乘意义下的最优解。
- **预测损失**：标准的交叉熵(分类)或 MSE(回归)。
- 超参数 $\beta$ 控制两者的权衡，默认 $\beta=10$，每个实例采样 32 个联盟。

### 理论保证

论文证明了当 $\phi^{\mathcal{V}ia}(x;\theta^*)$ 达到全局最优时：

| 性质 | 含义 |
|------|------|
| **局部准确性** (Lemma 3.1) | $\sum_i \phi_i = f(x) - f(\mathbf{0})$，Shapley 值之和等于预测差 |
| **缺失性** (Lemma 3.2) | 对预测无影响的特征 Shapley 值为 0 |
| **一致性** (Lemma 3.3) | 特征贡献增大则 Shapley 值不减 |
| **定理 3.4** | 最优解即精确的 Shapley 值 |

### 模型实现：四种变体

| 变体 | 架构 | 备注 |
|------|------|------|
| **KAN$^{\mathcal{V}ia}$** | Kolmogorov-Arnold Network (spline) | 层结构 $n \to 64 \to 128 \to 64 \to n \times d$ |
| **KAN$_\varrho^{\mathcal{V}ia}$** | KAN + 径向基函数(RBF) | 同上结构，用 RBF 替换 spline |
| **MLP$^{\mathcal{V}ia}$** | 标准 MLP + BatchNorm + ReLU | 同维度结构 |
| **MLP$_\theta^{\mathcal{V}ia}$** | 加宽 MLP（参数量对齐 KAN） | 用于公平比较 |

此外，图像任务提供 ResNet50$^{\mathcal{V}ia}$、ResNet18$^{\mathcal{V}ia}$、U-Net$^{\mathcal{V}ia}$ 三种实现。

## 实验关键数据

### 表格数据预测性能（25 个数据集，AUC）

| 方法 | 平均排名 | 与 XGBoost 差异 |
|------|----------|----------------|
| **KAN$^{\mathcal{V}ia}$** | **最优** | 统计不显著 (Nemenyi p>0.05) |
| KAN$_\varrho^{\mathcal{V}ia}$ | 第二 | 统计不显著 |
| XGBoost | 第三 | — |
| Random Forest | 第四 | — |
| TabNet | 第五 | — |
| MLP$_\theta^{\mathcal{V}ia}$ | 第六 | 与 KAN 差异显著 |
| MLP$^{\mathcal{V}ia}$ | 第七 | 与 KAN 差异显著 |

- KAN 变体与树模型之间无统计显著差异；KAN 显著优于 MLP 变体。
- KAN$^{\mathcal{V}ia}$ 还显著优于不带 Shapley 损失的同结构 KAN 分类器，说明 Shapley 损失反而有正则化效果。

### Shapley 值近似质量

| 指标 | 最优实现 | 说明 |
|------|----------|------|
| 余弦相似度 | MLP$_\theta^{\mathcal{V}ia}$ 第一, KAN 第二 | 四种变体间 Friedman 检验无显著差异 |
| Spearman 秩相关 | **KAN$^{\mathcal{V}ia}$** 第一 | MLP$^{\mathcal{V}ia}$ 与其他差异显著 |
| vs FastSHAP | ViaSHAP **显著优于** FastSHAP | 在表格和图像数据上均成立 |

### 图像实验（CIFAR-10）

| 模型 | 测试准确率 | Shapley 值质量 |
|------|-----------|---------------|
| ResNet50$^{\mathcal{V}ia}$ | 有竞争力 | 优于 FastSHAP |
| ResNet18$^{\mathcal{V}ia}$ | 有竞争力 | 优于 FastSHAP |
| U-Net$^{\mathcal{V}ia}$ | 有竞争力 | 优于 FastSHAP |

### 消融实验

- **$\beta$ 的影响**：$\beta$ 增大可提升 Shapley 值精度且不牺牲预测性能，但过大（≥200 倍）会导致训练失败。
- **联盟采样数**：对性能和解释精度影响较小。
- **链接函数**：去掉链接函数可显著提升 Shapley 值精度，预测性能不降。
- **效率约束**：对性能和解释精度无显著影响。

## 亮点与洞察

1. **范式转换**：首次将 Shapley 值从"后验解释工具"转变为"预测机制"，做到推理即解释，零额外开销。
2. **KAN 的优势**：基于 Kolmogorov-Arnold 表示定理的 KAN 在学习 Shapley 值函数时比 MLP 更有效，即使参数量对齐仍显著领先。
3. **Shapley 损失的正则化效应**：加入 Shapley 损失后模型预测性能反而提升，表明强制学习特征贡献有类似正则化的效果。
4. **理论完备**：严格证明了最优解满足 Shapley 值的三条公理性质。
5. **架构无关**：方法可适配 KAN、MLP、ResNet、U-Net 等多种架构，通用性好。

## 局限与展望

1. **全局最优假设**：理论保证依赖全局最优，但实际训练中只能达到局部最优，Shapley 值的精确性受限。
2. **推理增加的输出维度**：模型输出从 $d$ 维扩展至 $n \times d$ 维，高维特征场景下参数量和计算量增加。
3. **未与最新表格模型对比**：仅与 XGBoost/RF/TabNet 比较，未涉及 FT-Transformer、TabPFN 等最新 SOTA。
4. **$\beta$ 的调参**：虽作者称默认值鲁棒，但消融实验显示极端值可导致训练崩溃，不同任务可能需要调参。
5. **因果解释的局限**：Shapley 值衡量的是统计贡献而非因果效应，文中承认可能误导用户。
6. **安全隐患**：实时输出 Shapley 值可能被攻击者利用于模型逆向工程或对抗攻击。

## 相关工作与启发

- **KernelSHAP** (Lundberg & Lee, 2017)：经典后验 Shapley 值近似，ViaSHAP 的理论基础。
- **FastSHAP** (Jethani et al., 2022)：预训练解释器摊销推理成本，但仍是后验方案。
- **KAN** (Liu et al., 2024)：Kolmogorov-Arnold Network，ViaSHAP 证明其在学习 Shapley 值函数时优于 MLP。
- **自解释网络** (Alvarez Melis & Jaakkola, 2018)：生成解释但不满足 Shapley 公理。
- **启发**：可将 ViaSHAP 框架迁移到更强的表格模型（如 Transformer 架构），或用于对抗鲁棒性研究。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次提出"通过 Shapley 值回归进行预测"的范式
- 实验充分度: ⭐⭐⭐⭐ — 25 个数据集 + 图像实验 + 完整消融，但缺少最新表格 SOTA 对比
- 写作质量: ⭐⭐⭐⭐ — 理论推导严谨，结构清晰
- 价值: ⭐⭐⭐⭐ — 统一预测与解释的思路有实际意义，但全局最优假设限制了理论的实用性

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Faithful Group Shapley Value](../../NeurIPS2025/others/faithful_group_shapley_value.md)
- [\[ICML 2025\] Regression for the Mean: Auto-Evaluation and Inference with Few Labels through Post-hoc Regression](regression_for_the_mean_auto-evaluation_and_inference_with_few_labels_through_po.md)
- [\[ICML 2025\] Prediction-Powered Adaptive Shrinkage Estimation](prediction-powered_adaptive_shrinkage_estimation.md)
- [\[ICML 2025\] Heavy-Tailed Linear Bandits: Huber Regression with One-Pass Update](heavy-tailed_linear_bandits_huber_regression_with_one-pass_update.md)
- [\[NeurIPS 2025\] Regression Trees Know Calculus](../../NeurIPS2025/others/regression_trees_know_calculus.md)

<!-- RELATED:END -->
