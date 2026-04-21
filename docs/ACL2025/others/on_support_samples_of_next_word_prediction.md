---
title: >-
  [论文解读] On Support Samples of Next Word Prediction
description: >-
  [ACL 2025][数据中心解释性] 基于表示定理（representer theorem），研究语言模型下一词预测中训练样本的角色，发现两类支持样本（促进预测和抑制预测），并证明支持样本是样本的内在属性（训练前即可预测），而非支持样本对表示学习至关重要。
tags:
  - ACL 2025
  - 数据中心解释性
  - 支持样本
  - 表示定理
  - 语言模型可解释性
  - 下一词预测
---

# On Support Samples of Next Word Prediction

**会议**: ACL 2025  
**arXiv**: [2506.04047](https://arxiv.org/abs/2506.04047)  
**代码**: [github](https://github.com/liyuqian44/On-Support-Samples-of-Next-Word-Prediction)  
**领域**: 其他  
**关键词**: 数据中心解释性, 支持样本, 表示定理, 语言模型可解释性, 下一词预测

## 一句话总结

基于表示定理（representer theorem），研究语言模型下一词预测中训练样本的角色，发现两类支持样本（促进预测和抑制预测），并证明支持样本是样本的内在属性（训练前即可预测），而非支持样本对表示学习至关重要。

## 研究背景与动机

语言模型的决策可解释性是重要的研究课题。现有的解释方法主要沿两个方向：

**机制可解释性**（mechanistic interpretability）：关注神经元激活模式和电路

**数据中心可解释性**（data-centric interpretability）：追溯模型决策到训练数据

本文关注后者，核心问题是："当模型决定预测token v时，哪些训练样本贡献最大？"

现有方法中，反事实方法（influence function）需要计算Hessian矩阵，对大模型代价高昂。本文采用更高效的表示定理方法，直接将预测器参数分解为训练样本的加权组合，天然给出每个样本的重要性度量。

## 方法详解

### 整体框架

将语言模型分解为表示函数 $\phi(\mathbf{x})$ 和token预测函数 $f(\mathbf{x})$，使用表示定理将预测头参数 $\theta_v$ 表达为所有训练样本表示的加权和，权重即为样本的重要性系数。

### 关键设计

1. **表示定理与支持样本定义**：

    - 基于Crammer-Singer定理，当参数 $\theta$ 是损失函数的驻点时：
    $\theta_v = \frac{1}{2N\lambda} \sum_{i=1}^{N} (\mathbb{1}(\mathbf{y}_i = v) - p(v|\mathbf{x}_i)) \phi(\mathbf{x}_i)$
    - 系数 $\alpha_i = \mathbb{1}(\mathbf{y}_i = v) - p(v|\mathbf{x}_i)$ 衡量样本的重要性
    - 大 $|\alpha_i|$ 的样本为"支持样本"（$|α_i| ≥ τ$，$τ=0.9$），其余为非支持样本
    - 大 $\alpha_i$ 意味着样本难以学习：正确标签的预测概率小

2. **两类支持样本**：

    - **Type-1（促进型）**：$\mathbf{y}_i = v$，但模型预测置信度低（$\alpha_i > 0$）。将预测器拉向预测v
    - **Type-2（抑制型）**：$\mathbf{y}_i \neq v$，但模型错误地以高置信度预测v（$\alpha_i < 0$）。将预测器推离预测v
    - 减去实验证实：移除Type-1导致对v的预测失败；移除Type-2导致对v预测完美但影响其他token

3. **支持样本的内在性**：用简单分类器（线性/MLP）判断样本是否为支持样本。特征包括：

    - 最后一层隐藏向量
    - 所有层隐藏向量的拼接
    - 梯度特征（随机投影到4096维）
    - 在三个训练阶段（初始化、早期、最终）提取特征

4. **非支持样本的角色**：

    - 通过逐层probing分析每层的非支持样本比例
    - 发现高层包含更多非支持样本，说明它们对中间表示学习至关重要

### 损失函数 / 训练策略

- 语言模型：GPT-2架构（117M参数），12层Transformer，在wikitext-2数据集（2.37M样本）上从零训练
- 也在更大模型（345M/774M/1.5B）和更大数据集（wikitext-103，117M样本）上验证
- 二分类器：线性和MLP，训练/验证/测试=8:1:1

## 实验关键数据

### 主实验

支持样本的基本统计：

| 指标 | 数值 |
|------|------|
| 总训练样本数 | 2.37M |
| 支持样本数 | 1.29M (54%) |
| 超过半数样本对预测器重要 | - |
| 42%的token贡献了大部分支持样本 | - |
| 58%的token支持样本不足10个 | - |

减去支持样本实验（以v="hens"为例）：

| 移除集合 | 全集loss | 目标token loss |
|---------|---------|---------------|
| 无移除 | 3.28 | 0.24 |
| 移除所有支持样本 | 4.45 | 2.37 |
| 仅移除Type-1 | 3.35 | 16.73 |
| 仅移除Type-2 | 4.75 | 0.00 |
| 随机移除同量样本 | 3.75 | 0.27 |

### 消融实验

非支持样本移除实验（训练LM heads）：

| 采样方式 | 测试Loss | 说明 |
|---------|---------|------|
| 不移除 | 5.08 | 基线 |
| 硬移除（全部非支持） | 5.64 | 过拟合，新增大量支持样本 |
| 软采样（按α加权） | 5.13 | 接近基线，最优策略 |
| 随机移除同量 | 5.18 | 次优 |

非支持样本移除实验（完整模型训练）：

| 采样方式 | 测试Loss | 说明 |
|---------|---------|------|
| 不移除 | 5.08 | 基线 |
| 硬移除 | 6.57 | 严重恶化 |
| 随机移除同量 | 5.47 | 优于软采样 |
| 软采样 | 5.69 | 次于随机 |

支持样本预测准确率：

| 特征/阶段 | 初始化 (θ⁰) | 最终检查点 |
|----------|-------------|-----------|
| 梯度特征+MLP | **~80%** | ~85% |
| 最后隐藏层+MLP | ~55% | ~65% |
| 所有隐藏层+MLP | ~58% | ~66% |

### 关键发现

1. **支持样本比例惊人地高（54%）**：超过半数的训练样本对预测器参数有重大贡献，说明语言模型学到的模式远不够简洁
2. **支持样本分布高度偏斜**：少数token（如标点）几乎不需要支持样本，而语义丰富的词（动词86%是支持样本，标点仅20%）需要大量支持
3. **Type-2支持样本不可或缺**：移除Type-2后目标token loss降到0（完美预测），但其他token受损。Type-2起到"防护栏"作用，防止模型对v过度自信
4. **训练头部vs完整模型的策略不同**：训练LM heads时可以移除非支持样本（用软采样），但完整模型训练时非支持样本不可或缺
5. **支持样本在训练前即可预测**：仅用随机初始化模型的梯度特征，就能以80%准确率预测支持/非支持标签
6. **非支持样本在高层更重要**：逐层probing显示，第6层（中间层）出现非支持样本的急剧增长，暗示模型表示能力在此发生质变

## 亮点与洞察

- **理论优雅与实验验证的结合**：表示定理提供了简洁的数学框架，移除实验提供了直观的因果验证
- **重新定义"记忆"**：传统定义中"memorized"样本是高置信度正确预测的样本，但本文证明这些恰恰是非支持样本；支持样本才是真正被"记住"用于预测的
- **Token关系网络**：Type-2支持关系形成的有向网络揭示了token间的对抗关系，高频token（逗号、@、and）是主要的Type-2支持提供者
- **内在属性的发现**：一个样本是否为支持样本，在训练前就已经80%可以确定，这是一个令人惊讶的发现

## 局限与展望

1. 支持样本和非支持样本的界限并不清晰（基于阈值τ=0.9的二分类是近似的），最终检查点上分类器也无法达到100%准确率
2. 实验主要基于GPT-2（117M）的较小规模，虽然在更大模型上做了部分验证，但对现代大模型的泛化性待验证
3. 表示定理要求参数是损失函数的驻点，实际训练中可能未完全满足
4. 仅分析了预测头层面的支持关系，更深层的Transformer block的样本贡献尚未分析

## 相关工作与启发

- 与Influence Function (Koh & Liang, 2017)相比，表示定理方法计算成本更低，但适用范围更窄（仅限预测头）
- Toneva et al. (2019)的"forgetting events"和本文的支持样本有相似之处——都关注难以学习的样本
- Tirumala et al. (2022)的"memorized samples"定义恰好是本文非支持样本的子集（Claim 3）
- 数据选择/数据修剪研究通常只关注"保留困难样本"（Type-1），本文发现Type-2同样重要，提供了更完整的视角
- 可能启发新的训练数据选择策略：保留全部支持样本+适量非支持样本

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将表示定理应用于LM可解释性，发现两类支持样本及其内在属性，视角新颖
- 实验充分度: ⭐⭐⭐⭐ 实验设计严谨，多角度验证，但模型规模偏小
- 写作质量: ⭐⭐⭐⭐⭐ 定理与实验交织，逻辑紧凑，叙事引人入胜
- 价值: ⭐⭐⭐⭐⭐ 为数据中心的LM可解释性开辟了新方向，对训练数据选择有实际指导意义

<!-- RELATED:START -->

## 相关论文

- [Cautious Next Token Prediction](cautious_next_token_prediction.md)
- [Subword Models Struggle with Word Learning, but Surprisal Hides It](subword_models_struggle_with_word_learning_but_surprisal_hides_it.md)
- [MockConf: A Student Interpretation Dataset: Analysis, Word- and Span-level Alignment and Baselines](mockconf_a_student_interpretation_dataset_analysis_word-_and_span-level_alignmen.md)
- [Advancing Sequential Numerical Prediction in Autoregressive Models](advancing_sequential_numerical_prediction_in_autoregressive_models.md)
- [Generating Plausible Distractors for Multiple-Choice Questions via Student Choice Prediction](distractor_gen_multiple_choice.md)

<!-- RELATED:END -->
