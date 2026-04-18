---
title: >-
  [论文解读] Your Classifier Can Do More: Towards Balancing the Gaps in Classification, Robustness, and Generation
description: >-
  [CVPR 2026][对抗鲁棒性] 提出 EB-JDAT 框架，通过建模干净样本、对抗样本和生成样本的联合能量分布 $p_\theta(\mathbf{x}, \tilde{\mathbf{x}}, y)$，首次在单个模型中同时实现高分类精度、强对抗鲁棒性和具有竞争力的生成能力，在 CIFAR-10 上 AutoAttack 鲁棒性达 66.12%，超越 SOTA AT 方法超 10 个百分点。
tags:
  - CVPR 2026
  - 对抗鲁棒性
  - 能量模型
  - 联合生成判别
  - 对抗训练
  - JEM
---

# Your Classifier Can Do More: Towards Balancing the Gaps in Classification, Robustness, and Generation

**会议**: CVPR 2026  
**arXiv**: [2505.19459](https://arxiv.org/abs/2505.19459)  
**代码**: https://github.com/yujkc/EB-JDAT  
**领域**: others  
**关键词**: 对抗鲁棒性, 能量模型, 联合生成判别, 对抗训练, JEM

## 一句话总结

提出 EB-JDAT 框架，通过建模干净样本、对抗样本和生成样本的联合能量分布 $p_\theta(\mathbf{x}, \tilde{\mathbf{x}}, y)$，首次在单个模型中同时实现高分类精度、强对抗鲁棒性和具有竞争力的生成能力，在 CIFAR-10 上 AutoAttack 鲁棒性达 66.12%，超越 SOTA AT 方法超 10 个百分点。

## 研究背景与动机

**领域现状**：联合能量模型（JEM）将分类和生成统一在一个框架内，具有一定内在鲁棒性；对抗训练（AT）是提升鲁棒性最有效的方法，但牺牲干净精度且无生成能力。

**现有痛点**：JEM 鲁棒性远不如 AT；AT 会显著降低干净精度（通常掉 5-10%）；没有方法能同时兼顾分类精度、对抗鲁棒性和生成质量——存在根本性的"三难困境"。

**核心矛盾**：AT 在训练中引入对抗样本使模型偏离真实数据流形，而 JEM 未显式建模对抗分布，两者各有偏废。

**本文目标**：能否让一个模型同时具备高精度、强鲁棒性和好的生成能力？

**切入角度**：作者做了系统的**能量景观分析**——发现 AT 缩小了干净样本与对抗样本的能量差距（带来鲁棒性），JEM 缩小了干净样本与生成样本的能量差距（带来生成能力）。如果三类数据的能量分布都对齐，就可以统一两者的优势。

**核心 idea**：通过最大化干净和对抗分布的联合概率 $p_\theta(\mathbf{x}, \tilde{\mathbf{x}}, y)$，用 min-max 能量优化显式对齐三类数据的能量分布。

## 方法详解

### 整体框架

EB-JDAT 在 JEM 框架上扩展，将原来的联合分布 $p_\theta(\mathbf{x}, y)$ 扩展为三元联合分布 $p_\theta(\mathbf{x}, \tilde{\mathbf{x}}, y)$。通过贝叶斯分解将其拆为三个可优化的部分：

$$p_\theta(\mathbf{x}, \tilde{\mathbf{x}}, y) = p_\theta(y|\tilde{\mathbf{x}}, \mathbf{x}) \cdot p_\theta(\tilde{\mathbf{x}}|\mathbf{x}) \cdot p_\theta(\mathbf{x})$$

- $p_\theta(\mathbf{x})$：干净数据分布，通过 SGLD 采样
- $p_\theta(y|\tilde{\mathbf{x}}, \mathbf{x})$：鲁棒分类器（交叉熵）
- $p_\theta(\tilde{\mathbf{x}}|\mathbf{x})$：对抗分布的条件 EBM——这是本文核心创新

### 关键设计

1. **能量分布分析与对齐洞察**:

    - 功能：系统分析 AT 和 JEM 中干净/对抗/生成样本的能量分布
    - 核心发现：AT 使干净与对抗样本能量分布重叠（鲁棒性来源），JEM 使干净与生成样本能量重叠（生成能力来源）
    - 设计动机：如果三类分布全部对齐，就能统一两者优势——这是整个方法的理论基础

2. **条件对抗能量建模 $p_\theta(\tilde{\mathbf{x}}|\mathbf{x})$**:

    - 功能：用条件 EBM 显式建模对抗分布
    - 核心思路：对抗样本位于低密度（高能量）区域，通过 min-max 优化将其拉回高密度区域：
    $\min_\theta \mathbb{E}_{(\mathbf{x},y)\sim\mathcal{D}}\left[\max_{\|\tilde{\mathbf{x}}-\mathbf{x}\|\in\Omega}\left(E_\theta(\tilde{\mathbf{x}}|\mathbf{x}) - E_\theta(\mathbf{x})\right)\right]$
    - 内层最大化：沿 $-\nabla_{\mathbf{x}}\log p_\theta((\tilde{\mathbf{x}}|\mathbf{x}), y)$ 方向采样高能量对抗样本
    - 外层最小化：最小化对抗与干净样本的能量差，将对抗样本拉回低能量区域
    - 与 JEAT 的区别：JEAT 只建模 $p_\theta(\tilde{\mathbf{x}}, y)$, 忽略干净与对抗数据的内在关系；EB-JDAT 建模完整联合分布

3. **SGLD 采样与对抗采样**:

    - 功能：分别为生成分支和对抗分支提供样本
    - 生成采样：$\mathbf{x}_{t+1}^- = \mathbf{x}_t^- + \frac{c^2}{2}\frac{\partial \log p_\theta(\mathbf{x}_t^-)}{\partial \mathbf{x}_t^-} + c\epsilon$ 用于近似 $p_\theta(\mathbf{x})$
    - 对抗采样：$\tilde{\mathbf{x}}_{t+1} = \tilde{\mathbf{x}}_t - \frac{c^2}{2}\frac{\partial \log p_\theta((\tilde{\mathbf{x}}|\mathbf{x}), y)}{\partial \tilde{\mathbf{x}}_t}$（注意负号，目标是找高能量样本）

### 损失函数 / 训练策略

总梯度为三部分加权和：$h_\theta = w_1 h_1 + w_2 h_2 + w_3 h_3$

- $h_1 = \frac{\partial \log p_\theta(\mathbf{x})}{\partial \theta}$：生成梯度
- $h_2 = \frac{\partial \log p_\theta(\tilde{\mathbf{x}}|\mathbf{x})}{\partial \theta}$：对抗能量对齐梯度（核心）
- $h_3 = \frac{\partial \log p_\theta(y|\mathbf{x}, \tilde{\mathbf{x}})}{\partial \theta}$：鲁棒分类梯度

使用 WRN28-10 架构，学习率 0.01，扰动初始化 8/255，对抗采样步数 5。

## 实验关键数据

### 主实验

| 数据集 | 方法 | Clean Acc(%) | PGD-20(%) | AutoAttack(%) |
|--------|------|-------------|-----------|---------------|
| CIFAR-10 | DHAT-CFA (之前SOTA) | 84.49 | 62.38 | 54.05 |
| CIFAR-10 | LAS-AWP | 87.74 | 60.16 | 55.52 |
| CIFAR-10 | **EB-JDAT-SADAJEM** | **90.37** | **68.76** | **66.12** |
| CIFAR-100 | DHAT-CFA | 61.54 | 37.15 | 30.93 |
| CIFAR-100 | **EB-JDAT-SADAJEM** | **68.32** | **38.42** | **35.57** |
| ImageNet子集 | LAS-AT | 50.66 | 27.34 | 21.78 |
| ImageNet子集 | **EB-JDAT-JEM++** | **63.02** | **34.50** | **32.40** |

### 消融实验

| 配置 ($w_1, w_2, w_3$) | Clean(%) | AA(%) | FID | 说明 |
|------------------------|----------|-------|-----|------|
| (0, 0, 1) — 标准AT | 88.95 | 62.96 | 173.53 | 无生成能力, 41 epoch 崩溃 |
| (0, 1, 1) — 无生成 | 89.84 | 64.69 | 42.57 | $h_2$ 是关键 |
| (1, 0.5, 1) | 90.39 | 64.09 | 40.12 | 降低 $w_2$ 提升精度/FID, 稍降鲁棒 |
| **(1, 1, 1)** — 完整 | **90.37** | **64.61** | **39.67** | 最佳平衡 |

### 关键发现
- $h_2$（对抗能量对齐梯度）是最关键组件：去掉后不仅鲁棒性下降，还会导致训练崩溃（41 epoch）
- $h_1$（生成梯度）同时提升分类和生成质量
- 对抗采样步数 5 为最佳权衡，过多会导致模型崩溃
- EB-JDAT 无需额外生成数据，训练时间仅 31-66 GPU小时，远低于数据增强 AT（1438+ 小时）

## 亮点与洞察
- **能量景观分析驱动方法设计**：不是拍脑袋设计模块，而是先分析问题根源（能量分布差异），再设计对应解决方案。这种"分析→洞察→方法"的范式非常值得学习。
- **条件 EBM 建模对抗分布**：首次将对抗样本分布纳入联合能量模型的概率图，使对抗训练从"正则化"升级为"分布建模"。
- **min-max 能量优化的优雅设计**：对抗采样时往高能量方向走，训练时往低能量方向拉，与 AT 的 min-max 结构异曲同工但从能量视角统一。
- **"三难困境"的实质性突破**：在不使用额外数据的情况下，CIFAR-10 上 AA 鲁棒性超 SOTA 10+ 百分点，同时保持 90% 以上的干净精度和合理的 FID。

## 局限与展望
- 对抗采样步数过多会导致 EBM 崩溃，高分辨率大规模数据集（完整 ImageNet）的稳定性有待验证
- 生成质量相对最先进 JEM（SADAJEM）仍有差距，反映对抗训练与生成质量之间的内在张力
- 目前仅在 $\ell_\infty$ 范数下验证，其他扰动类型（$\ell_2$, spatial）未测试
- 训练需 SGLD 采样和对抗采样，仍比纯 AT 方法慢

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次从联合能量分布视角统一分类、鲁棒性和生成，理论框架优雅
- 实验充分度: ⭐⭐⭐⭐ CIFAR-10/100 和 ImageNet 子集覆盖全面，但缺少完整 ImageNet 实验
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，能量分析图表说明力强
- 价值: ⭐⭐⭐⭐⭐ 显著推进了对抗鲁棒性领域的性能前沿，方法具有普适性
