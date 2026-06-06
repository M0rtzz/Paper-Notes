---
title: >-
  [论文解读] PRESCRIBE: Predicting Single-Cell Responses with Bayesian Estimation
description: >-
  [NeurIPS 2025][计算生物][单细胞扰动预测] 提出 PRESCRIBE 框架，通过多变量深度证据回归联合建模单细胞扰动预测中的认知不确定性（模型对输入的不熟悉程度）和随机不确定性（生物系统固有的随机性），生成伪 E-distance 作为统一的不确定性代理指标…
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "单细胞扰动预测"
  - "不确定性量化"
  - "贝叶斯估计"
  - "深度证据回归"
  - "伪E-distance"
---

# PRESCRIBE: Predicting Single-Cell Responses with Bayesian Estimation

**会议**: NeurIPS 2025  
**arXiv**: [2510.07964](https://arxiv.org/abs/2510.07964)  
**代码**: [GitHub](https://github.com/Bunnybeibei/PRESCRIBE)  
**领域**: 计算生物学 / 单细胞扰动预测  
**关键词**: 单细胞扰动预测, 不确定性量化, 贝叶斯估计, 深度证据回归, 伪E-distance

## 一句话总结

提出 PRESCRIBE 框架，通过多变量深度证据回归联合建模单细胞扰动预测中的认知不确定性（模型对输入的不熟悉程度）和随机不确定性（生物系统固有的随机性），生成伪 E-distance 作为统一的不确定性代理指标，过滤不可靠预测后准确率提升 3% 以上。

## 研究背景与动机

预测基因扰动对细胞的影响是药物开发和基因治疗的关键任务。现有机器学习方法（如 GEARS、scGPT）虽然整体准确率较高，但在**特定预测**上可能犯严重错误——特别是对训练数据中未见过、且功能距离远的基因扰动。

问题的核心在于：**即使模型平均准确率很高，也不能保证每一个预测都可靠**。这在生物实验中代价极大，因为错误的预测可能导致昂贵的实验资源浪费。

预测不确定性有两个来源：

**随机不确定性（aleatoric）**: 生物系统的固有随机性——同一个基因扰动可能产生多种不同的细胞结果

**认知不确定性（epistemic）**: 模型对某个输入的不熟悉程度——如果测试扰动与训练数据中的扰动功能差异很大，模型的预测就不可靠

现有方法（如 GEARS 使用 MC Dropout）未能充分考虑配对距离中的不确定性，且其方差估计与实际预测准确度之间的相关性较差。作者从 E-distance（一种度量两个细胞群体相似性的指标）获得灵感，设计了伪 E-distance 来统一两种不确定性。

## 方法详解

### 整体框架

PRESCRIBE 由三个核心模块组成：(1) 编码器 $f_\alpha$ 将扰动映射到潜空间表示；(2) 归一化流 $f_\psi$ 估计训练数据在潜空间的密度，输出证据（evidence）；(3) 解码器 $f_\beta$ 生成后验分布的充分统计量。通过贝叶斯更新将先验与模型输出结合，得到最终的预测分布和不确定性估计。

### 关键设计

1. **多变量 Normal-Wishart 贝叶斯建模**: 假设基因表达向量 $y_i$ 服从多元高斯分布，参数上放置 Normal-Wishart 共轭先验：

$$\mathbb{P}(\mu_i | \Lambda_i) = \mathcal{N}(\mu_{0x_i}, (\kappa_i \Lambda_i)^{-1}), \quad \mathbb{P}(\Lambda_i) = \mathcal{W}(\nu_i, \Psi_i^{-1})$$

   使用下三角矩阵 $L_i$ 参数化 scale matrix $\Psi_i^{-1} = \nu_i L_i L_i^T$。这种共轭先验设计的动机是实现解析的贝叶斯更新，且在证据低（OOD 情况）时，预测自动回退到未扰动的控制状态（安全输出）。这是标准 NatPN 的多变量扩展，原始 NatPN 仅支持单变量。

2. **伪 E-distance 作为统一不确定性指标**: 定义伪 E-distance 为：

$$\tilde{E} = 2\tilde{\nu}_i^{\text{post}} - \tilde{\mathbb{H}}[\mathbb{P}(y_i | \omega_i)]$$

   其中 $\tilde{\nu}_i^{\text{post}}$ 是归一化的后验证据（度量认知不确定性），$-\tilde{\mathbb{H}}[\cdot]$ 是预测分布的负归一化熵（度量随机不确定性）。两项归一化到 $[N, 2N]$ 范围使其可比。高 $\tilde{E}$ 表示高置信度预测。关键特性：可以区分"低置信的 OOD 预测"和"高置信的无效果预测"——虽然两者输出都接近控制状态，但证据分数显著不同。

3. **归一化流估计证据**: 使用归一化流 $f_\psi$ 在潜空间估计训练数据密度，直接输出证据 $\nu_i$：

$$\nu_i = \exp(f_\psi(z_i) + \ln N_H), \quad \tilde{\nu}_i^{\text{post}} = \frac{N\nu_i}{\nu_i + \nu^{\text{prior}}} + N$$

   高密度区域（熟悉的输入）产生高证据，低密度区域（新颖的输入）产生低证据，迫使预测回退到先验。当 $\nu_i \to 0$ 时，$\tilde{\nu}_i^{\text{post}} \to N$（最低置信度）；当 $\nu_i \to \infty$ 时，$\tilde{\nu}_i^{\text{post}} \to 2N$（最高置信度）。

### 损失函数 / 训练策略

复合损失函数 $\mathcal{L} = \mathcal{L}_1 + \lambda_1 \mathcal{L}_2 + \lambda_2 \mathcal{L}_3 + \lambda_3 \mathcal{L}_4$：

- **$\mathcal{L}_1$ (期望对数似然)**: 最大化观测数据在后验预测分布下的似然，驱动准确预测
- **$\mathcal{L}_2$ (熵正则化)**: 按预测误差加权，鼓励高熵（不信息）分布
- **$\mathcal{L}_3$ (E-distance 排名损失)**: 使用 ListMLE 排名损失，使模型预测的伪 E-distance 排序与参考 E-distance 排序一致
- **$\mathcal{L}_4$ (不确定性正则化)**: 解决低证据区域的梯度消失问题

## 实验关键数据

### 主实验（扰动预测准确率）

| 模型 | Norman r↑ | Norman r^DEG↑ | Norman ACC↑ | Rep1 r↑ | K562 r↑ |
|------|-----------|---------------|-------------|---------|---------|
| GEARS | 45.30 | 63.19 | 29.09 | 48.18 | 32.57 |
| scGPT | 61.48 | 65.87 | 61.96 | 50.32 | 32.72 |
| scFoundation | 60.79 | 65.65 | 35.66 | 47.60 | 25.15 |
| PRESCRIBE | 58.38 | 64.44 | 63.24 | 59.18 | 36.20 |
| **PRESCRIBE-10%** | **64.32** | **68.61** | **64.73** | **60.28** | **38.58** |

*-X% 表示过滤掉 X% 最低置信度预测后的结果*

### 不确定性标定质量

| 方法 | Norman r^s_{perf,conf}↑ | Norman ACC_{perf,conf}↑ | Rep1 r^s↑ | K562 r^s↑ |
|------|-------------------------|-------------------------|-----------|-----------|
| GEARS-Drop | 单调性差 | - | - | - |
| GEARS-Ens | 不一致 | - | - | - |
| PRESCRIBE | **35.56** | **25.81** | **12.18** | **24.74** |

### 关键发现

1. **伪 E-distance 与真实 E-distance 正相关**: 在所有数据集上，预测的伪 E-distance 与参考 E-distance 呈正相关，且随参考计算样本数 N 增加，相关性显著增强（N=500 时 Spearman 相关达 80.00）
2. **置信度随泛化难度降低**: 在 Norman 组合扰动中，0/1/2 个未见扰动的场景下，PRESCRIBE 的置信度显著单调下降，而其他方法几乎不变或反转
3. **过滤 10% 最低置信预测后提升显著**: 在 Norman 上 r 从 58.38 提升到 64.32（+5.94），在 K562 上从 36.20 提升到 38.58（+2.38）
4. **随机过滤无效**: 随机过滤 10% 预测后准确率不升反降，说明 PRESCRIBE 确实能识别不可靠预测

## 亮点与洞察

- 伪 E-distance 统一两种不确定性的设计非常优雅：通过贝叶斯框架自然融合"模型知不知道"（证据）和"结果稳不稳定"（熵）
- 低证据时自动回退到控制状态是安全设计：宁可预测"没有效果"也不输出错误的效果
- 对生物实验有直接应用价值：实验者可以根据置信度分数决定哪些扰动值得实际验证

## 局限与展望

- 使用 PCA 降维可能丢失重要的基因表达信息，更好的降维方法值得探索
- 目前编码器设计依赖预训练基因嵌入（scGPT），不同嵌入的影响未充分研究
- 对组合扰动采用简单的加性假设，未建模扰动间的非线性交互
- 仅在有限的几个数据集上验证，未涉及多时间点或空间转录组数据

## 相关工作与启发

- 基于 Natural Posterior Network (NatPN) 的多变量扩展，解决了 NatPN 无法处理高维输出的限制
- E-distance 作为不确定性代理指标的思路可推广到其他领域（如药物反应预测、蛋白质工程）
- 启发：在生物信息学中，模型的不确定性估计可能比准确率本身更有价值，因为它指导实验资源分配

## 评分

- 新颖性: ⭐⭐⭐⭐ 伪 E-distance 和多变量 NatPN 扩展有新意，但深度证据回归框架已有先例
- 实验充分度: ⭐⭐⭐⭐ 三个数据集、多种基线、消融充分，但缺乏更多样的扰动类型
- 写作质量: ⭐⭐⭐⭐ 问题动机和方法动机清晰，但概率符号较多
- 价值: ⭐⭐⭐⭐ 为单细胞扰动预测提供了急需的不确定性量化工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Retrieval-Augmented Generation for Predicting Cellular Responses to Gene Perturbation](../../ICLR2026/computational_biology/retrieval-augmented_generation_for_predicting_cellular_responses_to_gene_perturb.md)
- [\[NeurIPS 2025\] scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery](scpilot_large_language_model_reasoning_toward_automated_single-cell_analysis_and.md)
- [\[NeurIPS 2025\] scMRDR: A Scalable and Flexible Framework for Unpaired Single-Cell Multi-Omics Data Integration](scmrdr_a_scalable_and_flexible_framework_for_unpaired_single-cell_multi-omics_da.md)
- [\[ACL 2025\] A Survey on Foundation Language Models for Single-cell Biology](../../ACL2025/computational_biology/foundation_lm_single_cell_survey.md)
- [\[NeurIPS 2025\] Is Sequence Information All You Need for Bayesian Optimization of Antibodies?](is_sequence_information_all_you_need_for_bayesian_optimization_of_antibodies.md)

</div>

<!-- RELATED:END -->
