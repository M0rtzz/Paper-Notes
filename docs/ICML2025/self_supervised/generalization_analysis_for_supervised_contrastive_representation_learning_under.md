---
title: >-
  [论文解读] Generalization Analysis for Supervised Contrastive Representation Learning under Non-IID Settings
description: >-
  [ICML2025][自监督学习][对比学习] 本文首次在非独立同分布（non-IID）条件下为监督对比表征学习（CRL）建立了泛化界，利用 U-统计量分解技术处理训练元组重叠样本的依赖性问题，给出了以标记样本数 $N$ 为自变量的 excess risk 收敛速率。
tags:
  - "ICML2025"
  - "自监督学习"
  - "对比学习"
  - "泛化界"
  - "U-统计量"
  - "非独立同分布"
  - "表征学习"
---

# Generalization Analysis for Supervised Contrastive Representation Learning under Non-IID Settings

**会议**: ICML2025  
**arXiv**: [2505.04937](https://arxiv.org/abs/2505.04937)  
**代码**: 未提供  
**领域**: 自监督  
**关键词**: 对比学习, 泛化界, U-统计量, 非独立同分布, 表征学习

## 一句话总结

本文首次在非独立同分布（non-IID）条件下为监督对比表征学习（CRL）建立了泛化界，利用 U-统计量分解技术处理训练元组重叠样本的依赖性问题，给出了以标记样本数 $N$ 为自变量的 excess risk 收敛速率。

## 研究背景与动机

### 领域现状

**领域现状**：对比学习的理论理解受限**：CRL 在视觉、图、NLP 等领域取得广泛成功，但理论泛化分析局限于 i.i.d. 元组假设

### 现有痛点

**现有痛点**：实践中元组不独立**：CRL 训练时通常来自固定标记样本池 $\mathcal{S}=\{(\mathbf{x}_j,\mathbf{y}_j)\}_{j=1}^N$，通过"回收"同一数据点构造不同元组 → 元组之间数据重叠，独立性假设不成立

### 核心矛盾

**核心矛盾**：现有工作缺口**：Arora et al. (2019)、Lei et al. (2023)、Hieu et al. (2024) 均假设元组 i.i.d.

### 解决思路

**解决思路**：本文目标**：提出贴合实际的修正理论框架，在 non-i.i.d. 设定下推导 excess risk 泛化界

## 方法详解

### 修正的理论框架

从固定标记样本池 $\mathcal{S}$ 出发构造子采样元组集 $\mathcal{T}_{sub}$：
1. 按经验类别概率 $\hat{\rho}(c)=N_c^+/N$ 选择锚-正对所属类
2. 类内无放回采两个正样本
3. 非该类中无放回采 $k$ 个负样本
4. 独立重复 $M$ 次

### U-统计量公式化

对每个类 $c$ 构造类条件 U-统计量 $\mathcal{U}_N(f|c)$，加权汇总得到总体估计：

$$\mathcal{U}_N(f)=\sum_{c\in\mathcal{C}}\frac{N_c^+}{N}\mathcal{U}_N(f|c)$$

其中 $N_c=\min(\lfloor N_c^+/2\rfloor,\lfloor(N-N_c^+)/k\rfloor)$。

### 证明策略：两步分解

$$\frac{1}{2}[L_{un}(\hat{f}_{sub})-\inf_f L_{un}(f)] \leq \sup_f|\hat{\mathcal{L}}(f;\mathcal{T}_{sub})-\mathcal{U}_N(f)| + \sup_f|\mathcal{U}_N(f)-L_{un}(f)|$$

- **第一项**：子采样经验风险与 U-统计量偏差 → Rademacher 复杂度界（$\mathcal{T}_{sub}$ 中元组是 i.i.d. 采样）
- **第二项**：U-统计量与总体风险偏差 → U-统计量解耦（decoupling）+ McDiarmid 不等式

### 对比损失

标准 logistic（N-pair）损失：$\ell(\mathbf{v})=\log(1+\sum_{i=1}^k\exp(-\mathbf{v}_i))$

### 核心定理

**Theorem 5.1**（U-统计量最小化器）：以概率 $\geq 1-\delta$，

$$ER_{un}(\hat{f}_{\mathcal{U}}) \leq \mathcal{O}\left[\sum_c\rho(c)\frac{K_{\mathcal{F},c}}{\sqrt{\tilde{N}}}+\mathcal{M}\sqrt{\frac{\ln|\mathcal{C}|/\delta}{\tilde{N}}}\right]$$

其中 $\tilde{N}=N\min(\frac{\min_c\rho(c)}{2},\frac{1-\max_c\rho(c)}{k})$。

**Theorem 5.2**（Sub-sampled 最小化器）：额外增加 $\mathcal{O}(1/\sqrt{M})$ 项。

## 实验关键数据


### 主实验

| 表征函数类 | 估计器 | 泛化界（均衡类、大k） |
|---|---|---|
| 线性映射 | $\hat{f}_{\mathcal{U}}$ | $\tilde{\mathcal{O}}(\eta sab^2/\sqrt{\tilde{N}})$ |
| 神经网络 | $\hat{f}_{\mathcal{U}}$ | $\tilde{\mathcal{O}}(\mathcal{M}\mathcal{W}^{1/2}/\sqrt{\tilde{N}})$ |
| 线性映射 | $\hat{f}_{sub}$ | 增加 $1/\sqrt{M}$ 项 |
| 神经网络 | $\hat{f}_{sub}$ | 增加 $1/\sqrt{M}$ 项 |

- MNIST 实验：随着子采样元组数 $M$ 增大，$\hat{f}_{sub}$ 性能趋近 $\hat{f}_{\mathcal{U}}$
- 合成数据验证 $|\mathcal{C}|$ 和 $k$ 与样本复杂度关系
- 均衡类分布下小 $k$ 时界 $\mathcal{O}(\sqrt{|\mathcal{C}|/N})$，大 $k$ 时 $\mathcal{O}(\sqrt{k/N})$

## 亮点与洞察

1. **首次解决 CRL 非 IID 泛化分析**：利用类别级 U-统计量回避了 $(k+2)$-阶 U-统计量构造困难
2. **与先前 IID 结果一致**：在 $N=nk$ 转换下界与 Lei et al. 量级吻合
3. **实用指导**：无法遍历全部元组时增大 $M$ 即可逼近理论最优

## 局限与展望

- 小 $k$ 时界含 $\sqrt{|\mathcal{C}|}$，源于类别独立分析
- 仅考虑有监督 CRL，未扩展至自监督
- 实验仅在 MNIST 与合成数据上验证

## 相关工作与启发

- Arora et al. (2019)：CRL 理论框架奠基，i.i.d. 元组 excess risk 界
- Lei et al. (2023)：改进 $k$ 依赖至对数级
- Clémençon et al. (2008)：U-统计量解耦用于排序
- Hieu et al. (2024)：DNN 下的 CRL 泛化
- 本文思路可启发在线学习/主动学习等非 i.i.d. 采样场景的泛化分析

## 评分

⭐⭐⭐⭐ — 理论扎实，填补 CRL 非 IID 泛化分析空白，对对比学习理论社区有重要参考价值


## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] A Refined Generalization Analysis for Extreme Multi-class Supervised Contrastive Representation Learning](../../ICML2026/self_supervised/a_refined_generalization_analysis_for_extreme_multi-class_supervised_contrastive.md)
- [\[ICML 2026\] Statistical Consistency and Generalization of Contrastive Representation Learning](../../ICML2026/self_supervised/statistical_consistency_and_generalization_of_contrastive_representation_learnin.md)
- [\[ICML 2025\] Collapse-Proof Non-Contrastive Self-Supervised Learning](collapse-proof_non-contrastive_self-supervised_learning.md)
- [\[ICML 2025\] Discovering Global False Negatives On the Fly for Self-supervised Contrastive Learning](discovering_global_false_negatives_on_the_fly_for_self-supervised_contrastive_le.md)
- [\[ICML 2025\] CLARIFY: Contrastive Preference Reinforcement Learning for Untangling Ambiguous Queries](clarify_contrastive_preference_reinforcement_learning_for_untangling_ambiguous_q.md)

</div>

<!-- RELATED:END -->
