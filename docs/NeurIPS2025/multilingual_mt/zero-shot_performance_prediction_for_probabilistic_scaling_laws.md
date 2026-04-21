---
title: >-
  [论文解读] Zero-Shot Performance Prediction for Probabilistic Scaling Laws
description: >-
  [NeurIPS 2025][scaling laws] 将 NLP 学习曲线预测建模为多任务学习问题，利用潜变量多输出高斯过程（MaGP）捕捉数据集中的双层层次结构和任务间相关性，实现学习曲线的零样本预测，并通过蒙特卡洛模拟推导概率化的 Scaling Laws。
tags:
  - NeurIPS 2025
  - scaling laws
  - learning curves
  - Gaussian processes
  - zero-shot prediction
  - hierarchical modeling
  - active learning
  - multi-task learning
---

# Zero-Shot Performance Prediction for Probabilistic Scaling Laws

**会议**: NeurIPS 2025  
**arXiv**: [2510.16743](https://arxiv.org/abs/2510.16743)  
**代码**: 论文称将公开（暂未确认链接）  
**领域**: llm_nlp  
**关键词**: scaling laws, learning curves, Gaussian processes, zero-shot prediction, hierarchical modeling, active learning, multi-task learning

## 一句话总结

将 NLP 学习曲线预测建模为多任务学习问题，利用潜变量多输出高斯过程（MaGP）捕捉数据集中的双层层次结构和任务间相关性，实现学习曲线的零样本预测，并通过蒙特卡洛模拟推导概率化的 Scaling Laws。

## 研究背景与动机

### 问题背景
推导 Scaling Laws 需要训练大量不同配置的模型，计算成本极高。如果能从少量已有学习曲线**零样本预测**未训练配置的学习曲线，就能大幅降低成本。

### 现有方法的局限

**参数化拟合方法**（power law、指数函数等）：需要对每条曲线独立拟合，无法在任务间迁移

**贝叶斯神经网络 (BNN)**：需要较多配置（Klein et al. 最小用了 256 个配置），在小数据集（30 条曲线以下）效果差

**传统 GP 方法**：假设数据内部结构扁平，未利用层次结构

**已有 Scaling Law 研究**：计算密集型的经验研究，缺乏不确定性量化

### 核心假说

NLP 学习曲线数据集具有**双层层次结构**（bi-level hierarchy），且该层次结构是**可交换的**（exchangeable）。利用这种结构 + 任务间相关性可以做零样本预测。

## 方法详解

### 整体框架

研究流程：数据建模（双层层次结构）→ MaGP 模型训练 → 零样本预测缺失曲线 → 蒙特卡洛模拟 → 概率 Scaling Law

### 层次结构的定义

三个实验场景对应不同的层次定义：

| 数据集 | 任务 $t$（第一层） | 数据实例 $d$（第二层） | 模型大小 |
|--------|-------------------|----------------------|---------|
| nanoGPT | embedding 参数数 | 层数 | $n_i$ 由两者决定 |
| 双语翻译 | 源语言 | 目标语言 | 固定模型 |
| 多语翻译 | 源语言 | 目标语言 + 模型大小 | 不同大小 M2M100 |

### 关键设计：MaGP 模型

采用 Ma et al. (2023) 提出的潜变量多输出高斯过程：

**生成过程**：

$$g(\mathbf{x}) \sim \mathcal{GP}(0, k_g(\mathbf{x}, \mathbf{x}'))$$

$$l_t^d(\mathbf{x}) \sim \mathcal{GP}(g(\mathbf{x}), k_l(\mathbf{x}, \mathbf{x}'))$$

$$y_t^d(\mathbf{x}) = l_t^d(\mathbf{x}, \mathbf{h}_t) + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2), \quad \mathbf{h}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

各组件含义：
- $g(\mathbf{x})$：**共享均值函数** — 编码所有任务的共同先验信息
- $k_g$：共享均值的协方差核
- $l_t^d(\mathbf{x})$：第 $t$ 个任务的第 $d$ 个学习曲线，以 $g(\mathbf{x})$ 为均值
- $k_l$：曲线级别的协方差核
- $\mathbf{h}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$：**潜变量** — 捕捉任务间的相关性
- $\epsilon_t$：观测噪声

**预测分布**（通过变分推断）：

$$q(\mathbf{l}^* | \mathbf{X}^*) = \int q(\mathbf{l}^* | \mathbf{X}^*, \mathbf{H}) q(\mathbf{H}) \text{d}\mathbf{H}$$

$$q(\mathbf{l}^* | \mathbf{X}^*, \mathbf{H}) = \mathcal{N}(\mathbf{l}^* | \tilde{\mathbf{m}}_*, \tilde{\mathbf{K}}_*)$$

后验为多元高斯，直接给出均值和不确定性。

### 概率化 Scaling Law

标准 Scaling Law 在 log-log 域为线性：

$$l^{\log}(c) \sim \mathcal{N}(\beta_0 + \beta_1 c, \sigma^2)$$

概率估计流程（$R$ 次蒙特卡洛）：
1. 用 $N$ 条已知曲线训练 MaGP
2. 零样本预测 $M$ 条缺失曲线
3. 在已知 + 预测曲线上拟合 compute-efficient frontier
4. 重复 $R$ 次，对 $\beta_0, \beta_1$ 取平均：

$$\hat{\beta}_0 = \frac{1}{R} \sum_{r=1}^{R} p(\beta_0^r | \mathcal{D}^*), \quad \hat{\beta}_1 = \frac{1}{R} \sum_{r=1}^{R} p(\beta_1^r | \mathcal{D}^*)$$

### 主动学习策略

从初始训练集出发，每次选择最不确定的曲线查询：

$$\text{mvar} = \frac{1}{11} \sum_{j=1}^{11} \text{var}_j, \quad \text{var} = \frac{1}{10} \sum_{i=1}^{10} (\mathbf{l}^{m_i*} - \overline{\mathbf{l}^{m_i*}})^2$$

选择 mvar 最大的曲线进行查询，重新训练模型后更新预测。

## 实验关键数据

### 主实验：nanoGPT 零样本预测

| 方法 | 划分 | MSE ↓ | MAE ↓ | MNLPD ↓ |
|------|------|-------|-------|---------|
| **MaGP** | Quad | **0.03±0.01** | **0.12±0.02** | **2.58±2.56** |
| DHGP | Quad | 0.07±0.00 | 0.20±0.01 | 3.25±0.24 |
| BNN (LC) | Quad | 10.69±0.43 | 2.82±0.03 | 596.05±23.79 |
| BNN (orig) | Quad | 13.96±0.75 | 3.06±0.05 | 778.95±41.90 |
| **MaGP** | Tri | **0.02±0.01** | **0.10±0.02** | **0.87±0.29** |
| DHGP | Tri | 0.07±0.00 | 0.19±0.00 | 1.85±0.63 |
| **MaGP** | T1 | **0.04±0.05** | **0.12±0.08** | **0.80±1.54** |
| DHGP | T1 | 0.08±0.00 | 0.18±0.00 | 0.87±0.09 |

**MaGP 在所有划分和指标上一致优于 baseline**。BNN 方法由于训练数据太少（仅 29 条曲线 × 11 点）表现极差。

### Scaling Law 预测

| 训练-测试划分 | 预测 Scaling Law | AbC ↓ | 计算成本 (PetaFLOPs) |
|-------------|-----------------|-------|---------------------|
| Quad | $(-0.043\pm0.002)c + (2.957\pm0.074)$ | 0.521±0.069 | $1.28 \times 10^5$ |
| Tri | $(-0.052\pm0.005)c + (3.352\pm0.202)$ | 0.160±0.172 | $2.06 \times 10^5$ |
| T1 | $(-0.059\pm0.006)c + (3.636\pm0.245)$ | **0.111±0.222** | $5.15 \times 10^5$ |

Ground truth: $-0.056c + 3.51$。

**发现**：
- Tri 和 T1 划分的 Scaling Law 预测非常接近 ground truth
- 训练数据越多 → AbC 越低（更准确），但不确定性也可能增加
- Quad 划分最省计算但偏差最大

### 消融实验：层次可交换性验证

| 方法 | 划分 | MSE ↓ | MAE ↓ |
|------|------|-------|-------|
| MaGP (交换) | Quad | 0.04±0.04 | 0.11±0.07 |
| DHGP (交换) | Quad | 0.11±0.00 | 0.26±0.01 |
| MaGP (交换) | T1 | **0.01±0.02** | **0.08±0.04** |
| DHGP (交换) | T1 | 0.16±0.02 | 0.34±0.02 |

交换层次后 MaGP 依然优于 baseline，甚至在 T1 上表现更好——**层次结构可交换**假说得到验证。

### 主动学习查询策略对比

四种策略在 AbC 上的表现：
- **Active Learning**：最稳定、最低不确定性，AbC 在所有查询次数上都保持最低或接近最低
- **Largest First**：AbC 接近 Active Learning 但成本更高、不确定性更大
- **Smallest First**：初期 AbC 偏高
- **Random**：不确定性最大

### 双语翻译零样本预测

在双语翻译（mBART50 + Transformer）数据集上：
- MaGP 在 BLEU 和 ChrF 指标上均取得最低 RMSE
- 朴素基线（源/目标语言曲线平均）变异性很大
- DHGP 改善了朴素基线但不确定性仍高
- 证实了双层层次结构建模的优势

### 关键发现

1. 在极小数据集（最多 30 条曲线、每条 11 个点）上，MaGP 显著优于 BNN 和 DHGP
2. 层次可交换性成立：不同的层次定义给出相似的预测性能
3. 主动学习策略能有效减少不确定性，提供最可靠的 Scaling Law 预测
4. 概率化 Scaling Law 提供了点估计 + 不确定性，比传统确定性拟合更有信息量

## 亮点与洞察

1. **问题建模角度新颖**：把 Scaling Law 推导重新框架为多任务学习 + 零样本预测问题，而非传统的参数化拟合
2. **层次结构假说有力**：在三个不同的 NLP 数据集上一致验证了双层层次结构的有效性
3. **概率化 Scaling Law 是重要贡献**：传统方法只给点估计，MaGP + MC 模拟给出分布，对决策更有价值
4. **主动学习策略**：提供了"如何选择下一个要训练的模型"的原则性方法，有实际部署价值
5. **极小数据有效**：仅 29 条学习曲线 × 11 点就能做出有意义的预测，这对资源受限的研究者非常重要
6. **模型选择直觉**：MaGP 用潜变量捕捉任务相关性，DHGP 靠层次发现聚类——当任务间有强相关时 MaGP 胜出

## 局限与展望

1. **数据集规模有限**：最大仅 30 条曲线，未在真正大规模（如 Chinchilla 规模）场景验证
2. **仅限 NLP**：虽然框架通用，但只在 NLP 任务上测试
3. **nanoGPT 是小模型**：与产业界关心的大模型（70B+）之间有 gap
4. **层次定义需要先验知识**：用户需要判断哪些因素构成层次，不同领域可能不直观
5. **计算复杂度**：GP 的 $O(n^3)$ 复杂度在曲线数或点数增大时可能成为瓶颈
6. **改进方向**：
    - 在更大规模模型（Llama-70B 级别）上验证
    - 自动发现层次结构（而非手动指定）
    - 结合 Chinchilla optimal 理论做更精细的 Scaling Law 预测
    - 扩展到 CV、Speech 等其他领域

## 相关工作与启发

- **与 Hoffmann et al. (2022, Chinchilla) 的关系**：Chinchilla 通过穷举训练推导 Scaling Law，本文用零样本预测大幅降低成本
- **与 Klein et al. (2016, BNN) 的对比**：BNN 需 256+ 配置，本文仅需 <30 条曲线
- **与 Hägele et al. 提出的替代训练技术互补**：前者从训练端降低成本，本文从预测端降低成本
- **Hensman et al. (2013, DHGP)**：层次 GP 用于基因表达数据，本文首次将类似思想引入 NLP 学习曲线
- **Ma et al. (2023, MaGP)**：提供了底层模型，本文的贡献在于将其应用于 Scaling Law 推导场景 + 发现 NLP 数据集的层次特性

**启发**：在做 Scaling Law 研究时，不必训练所有配置——通过对已训练模型之间的相关性建模，可以用远少于传统方法的计算资源推导出接近真实的 Scaling Laws。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 层次 GP 做 Scaling Law 预测是好的组合创新；MaGP 模型本身非本文贡献但应用场景新颖
- **实验充分度**: ⭐⭐⭐⭐ — 三个数据集、多种 baseline、可交换性验证、主动学习策略，很全面；但数据集规模偏小
- **写作质量**: ⭐⭐⭐⭐ — 结构严谨，三个假说逐步验证的叙事清晰；公式较多但解释充分
- **价值**: ⭐⭐⭐⭐ — 概念验证阶段，距离实际替代传统 Scaling Law 研究还需更大规模验证，但方向有价值

<!-- RELATED:START -->

## 相关论文

- [ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Finetuning, and Decoding the Curse of Multilinguality](../../ICLR2026/multilingual_mt/atlas_adaptive_transfer_scaling_laws_for_multilingual_pretraining_finetuning_and.md)
- [Machine Translation Models are Zero-Shot Detectors of Translation Direction](../../ACL2025/multilingual_mt/machine_translation_models_are_zero-shot_detectors_of_translation_direction.md)
- [Translation and Fusion Improves Zero-shot Cross-lingual Information Extraction](../../ACL2025/multilingual_mt/translation_and_fusion_improves_cross-lingual_information_extraction.md)
- [A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in LLMs](../../ACL2025/multilingual_mt/a_case_study_of_cross-lingual_zero-shot_generalization_for_classical_languages_i.md)
- [SMTPD: A New Benchmark for Temporal Prediction of Social Media Popularity](../../CVPR2025/multilingual_mt/smtpd_a_new_benchmark_for_temporal_prediction_of_social_media_popularity.md)

<!-- RELATED:END -->
