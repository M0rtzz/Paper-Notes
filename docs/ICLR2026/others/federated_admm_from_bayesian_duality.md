---
title: >-
  [论文解读] Federated ADMM from Bayesian Duality
description: >-
  [ICLR 2026][ADMM] 从变分贝叶斯(VB)视角推导出ADMM的贝叶斯对偶结构，证明经典ADMM是VB在各向同性高斯族上的特例，并导出Newton-like（二次目标一轮收敛）和Adam-like（深度异构场景+7%准确率）两个新扩展。
tags:
  - "ICLR 2026"
  - "ADMM"
  - "变分贝叶斯"
  - "自然梯度"
  - "联邦学习"
  - "贝叶斯对偶"
---

# Federated ADMM from Bayesian Duality

**会议**: ICLR 2026  
**arXiv**: [2506.13150](https://arxiv.org/abs/2506.13150)  
**代码**: [有](https://github.com/xxx)  
**领域**: 其他  
**关键词**: ADMM, 变分贝叶斯, 自然梯度, 联邦学习, 贝叶斯对偶

## 一句话总结
从变分贝叶斯(VB)视角推导出ADMM的贝叶斯对偶结构，证明经典ADMM是VB在各向同性高斯族上的特例，并导出Newton-like（二次目标一轮收敛）和Adam-like（深度异构场景+7%准确率）两个新扩展。

## 研究背景与动机

### 领域现状

**领域现状**：ADMM是联邦学习的核心算法骨架，自1970年代提出至今形式几乎未变。其鲁棒的算法结构令人好奇是否有更一般的形式化。

**现有痛点**：ADMM的加速变体（过松弛、动量、缩放范数等）只是引入额外变量但不改变算法形式。Swaroop等人发现VB和ADMM之间有逐行相似性，但未能推导出精确对应。

**核心矛盾**：ADMM的确定性优化框架难以自然扩展到异构深度学习场景。需要找到一个更一般的框架来统一和泛化ADMM。

**切入角度**：关键洞察是VB目标函数的解具有对偶结构，不仅与ADMM不动点结构类似，还自然泛化它。关键缺失环节是**自然梯度**。

**核心 idea**：用指数族分布的自然参数-期望参数对偶性建立"贝叶斯对偶"结构，ADMM是VB在各向同性高斯下的特例。

### 解决思路

**本文目标**：### 整体框架
经典ADMM：$(\theta_g^*, \theta_k^*, \mathbf{v}_k^*, \mathbf{v}_g^*)$ 的原始-对偶结构；贝叶斯ADMM：$(\mu_g^*, \mu_k^*, \eta_k^*, \lambda_g^*)$ 的期望参数-自然参数对偶结构。


## 方法详解

### 整体框架
经典ADMM：$(\theta_g^*, \theta_k^*, \mathbf{v}_k^*, \mathbf{v}_g^*)$ 的原始-对偶结构；贝叶斯ADMM：$(\mu_g^*, \mu_k^*, \eta_k^*, \lambda_g^*)$ 的期望参数-自然参数对偶结构。核心区别：梯度 $\to$ 自然梯度，参数 $\to$ 分布。

### 关键设计

1. **贝叶斯对偶结构**:

    - VB的不动点条件：$\lambda_g^* = -\sum_{k=0}^{K} \nabla \mathcal{L}_k(\mu_g^*)$
    - 引入局部分布 $q_k^*$ 和对偶变量 $\eta_k^*$，得到类似ADMM的四条件结构
    - 当 $q$ 取各向同性高斯时，自然梯度退化为普通梯度，恢复经典ADMM

2. **Newton-like扩展 (多元高斯)**:

    - 功能：$q$ 取完整协方差高斯分布
    - 核心思路：自然梯度包含Fisher信息矩阵的逆→在二次目标上等价于Newton法→一轮通信收敛
    - 设计动机：经典ADMM在二次目标上也需多轮迭代

3. **Adam-like扩展 (对角高斯, IVON-ADMM)**:

    - 功能：$q$ 取对角协方差高斯，用IVON方法高效实现
    - 核心思路：对角Fisher近似→自适应学习率效果类似Adam
    - 设计动机：完整Fisher太贵，对角近似在深度学习中更实用

### 损失函数 / 训练策略
- 客户端：最小化局部损失 + KL正则（贝叶斯版本）
- 服务器：聚合自然梯度参数（而非梯度本身）

## 实验关键数据

### 主实验
深度异构联邦学习：

| 方法 | 准确率 | 运行时间 | 说明 |
|------|--------|---------|------|
| FedADMM | 基线 | 基线 | 经典ADMM |
| FedAvg | 基线级 | 基线级 | 标准联邦 |
| IVON-ADMM | **+7%** | **相当** | Adam-like扩展 |

### 理论验证（二次目标）

| 方法 | 收敛轮数 | 说明 |
|------|---------|------|
| 经典ADMM | 多轮 | 线性收敛 |
| Newton-like ADMM | **1轮** | 一步到位 |

### 关键发现
- IVON-ADMM在深度异构场景（non-IID数据）上+7%准确率，且不增加通信和计算开销
- Newton-like变体在二次目标上确实一轮收敛，验证了理论预测
- 自然梯度是连通VB和ADMM的关键——正是Swaroop等人缺失的环节

## 亮点与洞察
- **数学之美**：经典ADMM竟然是贝叶斯方法在最简单分布族上的特例。这个关联不仅优美，还打开了用概率分布族泛化优化算法的新途径。
- **自然梯度的关键角色**：之前的工作用普通梯度无法建立精确对应，换成自然梯度后一切自然。说明信息几何在算法设计中的深层作用。
- **免费的午餐**：IVON-ADMM使用IVON的高效对角Fisher实现，不增加运行时间但显著提升异构场景性能。

## 局限与展望
- 深度学习实验规模较小（7层CNN），更大模型(LLM)上的效果未知
- 对角Fisher近似可能在某些模型上不够准确
- 需要选择指数族分布作为先验假设，分布选择的指导规则不明确
- 通信效率分析相对简单

## 相关工作与启发
- **vs FedADMM**: 经典ADMM是特例，贝叶斯ADMM提供了严格泛化
- **vs FedAvg**: 在异构场景下IVON-ADMM有显著优势
- **vs PVI (Swaroop 2025)**: 修复了PVI与ADMM之间的精确对应缺失

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 贝叶斯对偶结构原创且优美，统一了两大范式
- 实验充分度: ⭐⭐⭐ 理论验证充分但深度学习实验规模较小
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，对偶图示直观
- 价值: ⭐⭐⭐⭐ 为联邦优化提供新的理论基础和实用扩展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] A Federated Generalized Expectation-Maximization Algorithm for Mixture Models with an Unknown Number of Components](a_federated_generalized_expectation-maximization_algorithm_for_mixture_models_wi.md)
- [\[ICLR 2026\] Bayesian Influence Functions for Hessian-Free Data Attribution](bayesian_influence_functions_for_hessian-free_data_attribution.md)
- [\[NeurIPS 2025\] Rethinking PCA Through Duality](../../NeurIPS2025/others/rethinking_pca_through_duality.md)
- [\[ICML 2026\] Cost-Aware Stopping for Bayesian Optimization](../../ICML2026/others/cost-aware_stopping_for_bayesian_optimization.md)
- [\[AAAI 2026\] Bayesian Network Structural Consensus via Greedy Min-Cut Analysis](../../AAAI2026/others/bayesian_network_structural_consensus_via_greedy_min-cut_analysis.md)

</div>

<!-- RELATED:END -->
