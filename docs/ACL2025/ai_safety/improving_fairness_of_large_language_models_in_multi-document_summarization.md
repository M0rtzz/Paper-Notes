---
title: >-
  [论文解读] Improving Fairness of Large Language Models in Multi-document Summarization
description: >-
  [ACL 2025][AI安全][公平性] 提出 FairPO（Fair Preference Optimization），通过扰动式偏好对生成和公平感知偏好调优，同时优化多文档摘要中的摘要级和语料级公平性。
tags:
  - ACL 2025
  - AI安全
  - 公平性
  - 多文档摘要
  - 偏好优化
  - DPO
  - 覆盖率均等
---

# Improving Fairness of Large Language Models in Multi-document Summarization

**会议**: ACL 2025  
**arXiv**: [2506.07479](https://arxiv.org/abs/2506.07479)  
**代码**: [GitHub](https://github.com/leehaoyuan/coverage_fairness) (有)  
**领域**: AI安全  
**关键词**: 公平性, 多文档摘要, 偏好优化, DPO, 覆盖率均等  

## 一句话总结

提出 FairPO（Fair Preference Optimization），通过扰动式偏好对生成和公平感知偏好调优，同时优化多文档摘要中的摘要级和语料级公平性。

## 研究背景与动机

### 1. 领域现状
多文档摘要（MDS）旨在从多个文档中提炼关键信息，例如对某一产品的多条评论进行总结。每个文档通常带有社会属性（如情感倾向：正面/中性/负面），不同属性的文档可能包含有冲突的观点。

### 2. 现有痛点
- 现有方法主要关注**摘要级公平性**（单条摘要内部是否均衡覆盖不同属性的文档），忽视了**语料级公平性**（整个语料库中是否系统性地过度/不足代表某些属性）
- Zhang et al. (2023) 的提示方法依赖用户对公平性问题的先验知识，实用性有限
- Huang et al. (2024) 的策略梯度方法针对 T5 设计，难以推广到现代 LLM
- 这些方法都只关注摘要级公平性

### 3. 核心矛盾
LLM 在多文档摘要中既存在摘要级偏差（单条摘要偏向某些属性），又存在语料级偏差（系统性地过度代表某些属性），但现有方法无法同时优化两者。

### 4. 本文目标
提出一种能同时改善摘要级和语料级公平性的偏好调优方法，不依赖用户先验知识，且不影响摘要的其他质量（相关性、事实性、流畅性）。

### 5. 切入角度
将偏好调优（DPO）与公平性度量相结合，通过扰动输入文档集来产生体现公平性差异的偏好对，再通过动态加权优化语料级公平性。

### 6. 核心 idea

**通过移除部分特定属性文档的"扰动"来生成偏好对，结合动态权重分配实现同时优化两级公平性的 DPO 变体。**

## 方法详解

### 整体框架

FairPO 包含两个核心模块：
1. **扰动式偏好对生成**（Perturbation-based Preference Pair Generation）
2. **公平感知偏好调优**（Fairness-aware Preference Tuning）

### 关键设计

#### 模块一：扰动式偏好对生成

对每个文档集 $D$，FairPO 首先生成原始摘要 $S$，识别出最过度代表的属性值 $k^+$ 和最不足代表的属性值 $k^-$（基于覆盖概率差异）。然后分别从文档集中移除 $\alpha\%$ 的 $k^+$ 和 $k^-$ 属性文档，生成扰动后的摘要 $S^+$ 和 $S^-$。

在 $S, S^+, S^-$ 三者中：
- **Equal Coverage 最低**的作为 chosen summary $S_c$（最公平）
- **Equal Coverage 最高**的作为 rejected summary $S_r$（最不公平）

覆盖概率的计算通过蕴含模型估计：

$$p(d_i, s_j) = \max\{p(d_{i,l}, s_j) | d_{i,l} \in d_i\}$$

$$EC(D,S) = \frac{1}{K}\sum_{k=1}^{K}|\mathbb{E}(\{c(d_i, S) | a_i = k\})|$$

#### 模块二：公平感知偏好调优

FairPO 修改 DPO 目标函数，引入分离权重 $w_c$ 和 $w_r$：

$$\sigma(-m) \cdot \beta\left(w_r \log\frac{\pi_\theta(S_r|D)}{\pi_{ref}(S_r|D)} - w_c \log\frac{\pi_\theta(S_c|D)}{\pi_{ref}(S_c|D)}\right)$$

权重计算基于对语料级公平性的动态估计。对每个属性值 $k$，FairPO 估计过度代表 $O(k)$ 和不足代表 $U(k)$：

$$O(k) = \frac{\sum_{(D,S)\in T_k^+}|C_k(D,S)| \cdot \pi_\theta(S|D)/|S|}{\sum_{(D,S)\in T_k^+}\pi_\theta(S|D)/|S|}$$

中间权重的计算采用类 sigmoid 函数：

$$w_{c,k} = \frac{2}{1 + (O(k)/U(k))^{C_k(D,S_c)/\tau}}$$

**设计直觉**：如果某属性值在语料中系统性不足代表（$U(k)>O(k)$），而 chosen summary 恰好过度代表了它（$C_k>0$），那么该 chosen summary 有助于平衡语料级公平性，应获得更高权重。

### 训练策略
- 使用 LoRA 微调，学习率 $5e-5$，batch size 16，训练 2 个 epoch
- 扰动比例 $\alpha = 10\%$
- 温度参数 $\tau$ 根据数据集和模型在验证集上调优（范围 1-3）

## 实验关键数据

### 主实验

| 方法 | Amazon EC↓ | Amazon CP↓ | MITweet EC | MITweet CP↓ | SemEval EC↓ | SemEval CP↓ | 平均 EC↓ | 平均 CP↓ |
|------|-----------|-----------|-----------|------------|-----------|------------|---------|---------|
| Llama3.1 | 7.95 | 1.89 | 4.50 | 0.59 | 2.98 | 1.41 | 5.14 | 1.30 |
| +DPO | 7.23 | 1.27 | 4.25 | 0.47 | 2.66 | 1.09 | 4.72 | 0.94 |
| +OPTune | 6.70 | 0.62 | 4.33 | 0.51 | 2.60 | 0.95 | 4.54 | 0.69 |
| **+FairPO** | **6.87** | **0.42** | **4.24** | **0.42** | **2.49** | **0.66** | **4.53** | **0.50** |
| Gemma2 | 8.32 | 2.48 | 4.20 | 0.60 | 2.81 | 0.96 | 5.11 | 1.35 |
| **+FairPO** | **6.18** | **0.44** | **3.76** | **0.48** | **2.50** | **0.45** | **4.15** | **0.46** |

FairPO 在所有模型上取得最佳综合性能，尤其在 CP（语料级公平性）上改进显著。

### 消融实验

| 变体 | Overall EC↓ | Overall CP↓ |
|------|-----------|-----------|
| FairPO | **4.39** | **0.39** |
| w/o pert.（无扰动） | 4.54 | 0.54 |
| w/o fair.（无公平感知权重） | 4.42 | 0.64 |
| w/o rew.（无奖励裕度） | 4.42 | 0.64 |

两个模块均有贡献，移除任一都会导致性能下降。

### 关键发现
1. **人类评估**：30 对摘要中，FairPO 在 18 对中更公平（vs DPO 的 9 对），差异显著（$p<0.05$）
2. **摘要质量保持**：FairPO 在流畅性、相关性、事实性上与原始 LLM 相当，而 Prompt 方法显著损害质量
3. **三个数据集**覆盖了不同的社会属性（情感、政治意识形态、立场），文档集大小从 8 到 30 不等

## 亮点与洞察

1. **两级公平性同时优化**是本文的核心贡献，之前的工作只关注摘要级
2. **扰动式偏好对生成**非常巧妙——通过有针对性地移除文档来"放大"模型的偏差倾向，从而产生公平性差异显著的偏好对
3. **动态权重分配**的设计很精巧，能在训练过程中实时追踪语料级公平性状况并调整优化方向
4. 方法通用性强，适用于多种 LLM（Llama、Mistral、Gemma），无需修改模型架构

## 局限与展望

1. 目前只在单一领域内优化公平性，跨领域同时优化多种社会属性是更有挑战性的场景
2. 三个候选摘要中只选择两个做偏好对，如何利用全部三个摘要的信息值得探索
3. 公平性度量依赖蕴含模型，蕴含模型本身的偏差可能影响评估
4. 实验规模有限（每个数据集 1000 训练/300 测试），在更大规模上的表现待验证

## 相关工作与启发

- **DPO 偏好调优**（Rafailov et al., 2024）：FairPO 的基础，通过偏好对直接优化策略
- **OPTune**（Chen et al., 2024）：在线偏好调优，按 EC 差异加权——FairPO 比其更进一步考虑了语料级公平性
- **启发**：偏好调优的加权策略可以灵活注入不同的优化目标（公平性、安全性、多样性等），不限于质量改善

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次将偏好调优用于 MDS 公平性，扰动+动态加权的设计有新意
- **实验充分度**: ⭐⭐⭐⭐ — 3 个数据集、3 个模型、消融实验、人类评估、质量评估，较完整
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，方法推导完整，实验组织有条理
- **价值**: ⭐⭐⭐⭐ — 公平性是 MDS 的重要问题，方法通用且实用

<!-- RELATED:START -->

## 相关论文

- [The Tug of War Within: Mitigating the Fairness-Privacy Conflicts in Large Language Models](tug_of_war_fairness_privacy.md)
- [ReDial: Assessing Dialect Fairness and Robustness of Large Language Models in Reasoning Tasks](dialect_fairness_robustness.md)
- [Ensemble Watermarks for Large Language Models](ensemble_watermarks_llm.md)
- [Improved Unbiased Watermark for Large Language Models](improved_unbiased_watermark_for_large_language.md)
- [MorphMark: Flexible Adaptive Watermarking for Large Language Models](morphmark_adaptive_watermarking.md)

<!-- RELATED:END -->
