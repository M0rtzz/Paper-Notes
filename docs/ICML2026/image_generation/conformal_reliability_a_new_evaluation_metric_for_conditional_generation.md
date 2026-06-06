---
title: >-
  [论文解读] Conformal Reliability: A New Evaluation Metric for Conditional Generation
description: >-
  [ICML2026][图像生成][可靠性评估] 提出基于保形预测（Conformal Prediction）的可靠性评分 CReL，通过在隐空间构建凸预测集并优化最坏情况下的指标表现，实现对条件生成模型的不确定性感知评估，在图文互生任务上揭示了传统单输出指标无法捕捉的模型可靠性差异。
tags:
  - "ICML2026"
  - "图像生成"
  - "可靠性评估"
  - "保形预测"
  - "条件生成"
  - "最坏情况分析"
  - "不确定性量化"
---

# Conformal Reliability: A New Evaluation Metric for Conditional Generation

**会议**: ICML2026  
**arXiv**: [2605.30807](https://arxiv.org/abs/2605.30807)  
**代码**: https://ggc29.github.io/CReL/ (有)  
**领域**: 图像生成  
**关键词**: 可靠性评估, 保形预测, 条件生成, 最坏情况分析, 不确定性量化  

## 一句话总结
提出基于保形预测（Conformal Prediction）的可靠性评分 CReL，通过在隐空间构建凸预测集并优化最坏情况下的指标表现，实现对条件生成模型的不确定性感知评估，在图文互生任务上揭示了传统单输出指标无法捕捉的模型可靠性差异。

## 研究背景与动机

**领域现状**：条件生成模型（文生图、图生文等）取得了显著进展，当前主流评估指标如 CLIP Score、BERT-SIM、FID 等通常只评估单次生成输出的质量，反映的是模型的"平均水平"。

**现有痛点**：生成模型具有内在随机性——同一输入在不同采样种子下可能产生截然不同的输出。一个模型可能平均得分很高，但仍有不可忽视的概率产生灾难性失败。例如，图生文任务中模型通常会正确生成"一个人在弹吉他"，但在某些种子下可能生成"一个人拿着枪"。在安全关键场景下，单输出评估无法量化这种尾部风险。

**核心矛盾**：现有指标衡量的是"模型能多好"，而可靠性应衡量"模型最差能多差"。但直接在高维输出空间中构建预测集并优化最坏情况指标，面临维度灾难和非凸优化的双重困难。

**本文目标**：定义一个考虑不确定性的可靠性评分（reliability score），在给定置信水平 $1-\alpha$ 下量化模型的最坏情况表现，并提供高效的计算框架。

**切入角度**：将高维输出映射到低维隐空间，利用方向分位数回归（DQR）构建凸预测区域，再通过保形校准确保覆盖率保证。凸性使得最坏情况优化可用投影梯度下降求解。

**核心 idea**：在隐空间中构建满足覆盖率保证的凸预测集，将原本不可解的高维非凸可靠性优化问题转化为凸约束上的可解优化问题。

## 方法详解

### 整体框架
CReL 框架的输入是一个待评估的条件生成模型 $f$ 和用户指定的相似度指标 $\rho$，输出是置信水平 $1-\alpha$ 下的可靠性评分。整个流程分为四步：(1) 训练隐空间生成模型（LGM）将高维输出映射到低维隐空间；(2) 在隐空间中用 DQR 构建凸分位数区域；(3) 通过保形校准扩展区域以满足 $1-\alpha$ 覆盖率；(4) 在校准后的凸预测集上用投影梯度下降优化最坏情况指标。

### 关键设计

1. **隐空间保形校准**:

    - 功能：将高维输出空间中不可解的预测集构建与校准问题转化为低维隐空间中的可解问题
    - 核心思路：训练数据被拆分为三折——$\mathcal{I}_{\text{lgm}}$ 训练 VAE 编码器/解码器，$\mathcal{I}_{\text{dqr}}$ 训练 DQR 模型，$\mathcal{I}_{\text{cal}}$ 用于保形校准。编码器 $\mathcal{E}$ 将输出 $\hat{Y}$ 映射为隐变量 $Z \in \mathbb{R}^r$，DQR 对每个方向 $\mathbf{u} \in \mathbb{S}^{r-1}$ 估计 $\alpha$ 分位数半空间 $\mathbb{H}_u^+(x)$，取交集得到凸区域 $R_\mathcal{Z}(x) = \bigcap_{\mathbf{u}} \mathbb{H}_u^+(x)$。由于多方向交集会导致覆盖率低于 $1-\alpha$，计算校准集上每个样本到 $R_\mathcal{Z}$ 的距离 $E_i^+$，取 $\lceil(|\mathcal{D}_{\text{cal}}|+1)(1-\alpha)\rceil$ 分位数作为 $\gamma_{\text{cal}}$，扩展为 $S^{\gamma_{\text{cal}}}(x)$
    - 设计动机：在原始输出空间中校准需要网格离散化，计算代价随维度指数增长；隐空间中的凸区域可通过线性规划高效计算投影距离 $E_i^+$

2. **可靠性评分定义与优化**:

    - 功能：量化模型在置信水平 $1-\alpha$ 下的最坏情况表现
    - 核心思路：可靠性评分定义为 $\min_{z \in C_\mathcal{Z}(X_{n+1})} \rho(\mathcal{D}ec(z; X_{n+1}), \text{GT}_{n+1})$，即在校准预测集内找到使指标 $\rho$ 最差的输出。由于 $C_\mathcal{Z}$ 是凸紧集，投影算子可通过线性规划高效计算：先求 $y^* = \arg\min_{y_1 \in R_\mathcal{Z}(x)} \|y_1 - y\|_2$（线性规划），再沿方向平移 $\gamma_{\text{cal}}$。采用多起点（50 个）的投影梯度下降求解，实验显示标准差仅 0.00027
    - 设计动机：原始优化中 $\rho$ 和约束集 $C_\mathcal{Y}$ 都非凸，不可解；通过在隐空间重构为凸约束非凸目标问题，可利用投影梯度下降的全局收敛保证

3. **覆盖率理论保证**:

    - 功能：证明校准后的预测集满足 $\mathbb{P}(\hat{Y}_{n+1} \in C_\mathcal{Y}(X_{n+1})) \geq 1-\alpha$
    - 核心思路：基于可交换性（exchangeability）证明隐空间覆盖率 $\mathbb{P}(Z_{n+1} \in S^{\gamma_{\text{cal}}}) \geq 1-\alpha$；当 LGM 能准确恢复条件分布 $\hat{Y}|X$ 时，解码器映射保持覆盖率不降。上界为 $1-\alpha + 1/(1+|\mathcal{D}_{\text{cal}}|)$，随校准集增大趋近目标
    - 设计动机：相比 Feldman 等人在输出空间校准，隐空间校准虽可能因解码器扩展而略微保守，但换来了优化可解性这一关键优势

## 实验关键数据

### 合成数据校准结果

| 方法 | $\alpha$ | 覆盖率-$\mathcal{Z}$ | 覆盖率-$\mathcal{Y}$ | 区域面积 |
|------|----------|----------------------|----------------------|----------|
| CReL (Ours) | 0.10 | 0.8953 | 0.8915 | **232.7** |
| Feldman | 0.10 | — | 0.8940 | 234.5 |
| DQR | 0.10 | 0.8823 | 0.9145 | 287.4 |
| CReL (Ours) | 0.02 | 0.9770 | 0.9760 | **398.5** |
| DQR | 0.02 | 0.9818 | 0.9872 | 749.1 |

### 图生文任务可靠性评估（$\alpha=0.1$）

| 模型 | CLIP-SIM | CReL-CLIP | BERT-SIM | CReL-BERT |
|------|----------|-----------|----------|-----------|
| BLIP-base | 0.2330 (4th) | **0.0070 (1st)** | 0.8349 (3rd) | 0.6335 (3rd) |
| BLIP-large | 0.2453 (3rd) | −0.0074 (4th) | 0.8106 (4th) | 0.5631 (4th) |
| GIT-base | 0.2511 (2nd) | −0.0021 (2nd) | **0.8620 (2nd)** | **0.6474 (1st)** |
| GIT-large | 0.2550 (1st) | −0.0043 (3rd) | **0.8649 (1st)** | 0.6459 (2nd) |

### 关键发现
- **排名反转现象**：BLIP-base 在 CLIP-SIM 平均分排名最低（0.2330），但 CReL-CLIP 排名第一（0.0070），因为其得分分布更集中，最坏情况表现更好
- **区域面积优势**：CReL 的预测集面积（232.7）远小于 DQR（287.4）且与 Feldman（234.5）持平，表明联合校准产生更紧凑的信息集
- **可扩展性**：与 Feldman 的网格方法在高维时指数增长不同，CReL 的隐空间校准运行时间随维度线性增长
- **文生图任务**中也观察到类似反转：SD3-M 在 CLIP-SIM 排第三但 CReL-CLIP 排第一，Kandinsky-2.2 平均最高但可靠性排第三

## 亮点与洞察
- **将可靠性重新定义为最坏情况问题**：跳出传统平均指标思路，用保形预测框架量化生成模型的尾部风险，概念简洁且适用于任意用户指定指标 $\rho$。这个视角对安全关键场景（医学、自动驾驶）的生成模型评估有直接价值
- **隐空间凸化策略**：通过 LGM+DQR 的组合将非凸高维问题转化为凸约束低维优化，是一个优雅的工程-理论平衡。投影算子归结为线性规划，使整个框架可实际运行
- **模型排名反转的发现**具有实际指导意义：说明平均分高的模型未必可靠，分布集中性才是可靠性的关键特征，可迁移到任何需要评估生成一致性的场景

## 局限与展望
- LGM 需要额外训练（VAE 编码器/解码器），增加评估成本，且覆盖率保证依赖 LGM 重建质量的假设
- 当前仅在 MS-COCO 上评估图文互生任务，未涉及视频生成、3D 重建等更复杂的条件生成场景
- 保形预测提供的是边际覆盖率保证（marginal coverage），非条件覆盖率，对特定困难输入可能不够严格
- 可扩展到多对多映射场景（视频、机器人控制），但需要设计新的联合隐空间表示和校准策略

## 相关工作与启发
- Feldman et al. (2023) 的校准多输出分位数回归在输出空间校准，非凸性导致优化困难；CReL 转移到隐空间获得凸性
- 方向分位数回归 DQR (Kong & Mizera, 2012) 提供凸预测集构建基础，但在高维时过于保守
- PCP (Wang et al., 2022b) 对条件生成模型构建预测集，但其逐坐标校准可能比联合隐空间校准更保守（面积 854.24 vs 232.70）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Conf-Gen: Conformal Uncertainty Quantification for Generative Models](conf-gen_conformal_uncertainty_quantification_for_generative_models.md)
- [\[CVPR 2026\] SHOE: Semantic HOI Open-Vocabulary Evaluation Metric](../../CVPR2026/image_generation/shoe_semantic_hoi_open-vocabulary_evaluation_metric.md)
- [\[NeurIPS 2025\] Hallucination as an Upper Bound: A New Perspective on Text-to-Image Evaluation](../../NeurIPS2025/image_generation/hallucination_as_an_upper_bound_a_new_perspective_on_text-to-image_evaluation.md)
- [\[ICLR 2026\] PolyGraph Discrepancy: a classifier-based metric for graph generation](../../ICLR2026/image_generation/polygraph_discrepancy_a_classifier-based_metric_for_graph_generation.md)
- [\[ICML 2026\] AtelierEval: Agentic Evaluation of Humans & LLMs as Text-to-Image Prompters](ateliereval_agentic_evaluation_of_humans_llms_as_text-to-image_prompters.md)

</div>

<!-- RELATED:END -->
