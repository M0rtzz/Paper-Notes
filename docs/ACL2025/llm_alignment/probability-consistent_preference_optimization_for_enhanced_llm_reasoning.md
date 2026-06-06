---
title: >-
  [论文解读] Probability-Consistent Preference Optimization for Enhanced LLM Reasoning
description: >-
  [ACL 2025][LLM对齐] > PCPO 在偏好对选择阶段引入 token 级概率一致性指标，选出答案正确且推理过程与错误回答最"相似"的配对进行 DPO 训练，让模型聚焦关键推理差异，在多个数学推理 benchmark 上一致超越 IRPO/ScPO。
tags:
  - "ACL 2025"
  - "LLM对齐"
---

# Probability-Consistent Preference Optimization for Enhanced LLM Reasoning

**会议**: ACL 2025  
**arXiv**: [2505.23540](https://arxiv.org/abs/2505.23540)  
**代码**: [https://github.com/YunqiaoYang/PCPO](https://github.com/YunqiaoYang/PCPO)  
**机构**: CUHK MMLab, SenseTime Research, 上海AI Lab  

## 一句话总结

> PCPO 在偏好对选择阶段引入 token 级概率一致性指标，选出答案正确且推理过程与错误回答最"相似"的配对进行 DPO 训练，让模型聚焦关键推理差异，在多个数学推理 benchmark 上一致超越 IRPO/ScPO。

## 研究背景与动机

偏好优化（DPO 及其变体）已成为提升 LLM 数学推理能力的主流方法。核心步骤是构建 chosen/rejected 偏好对：

- **IRPO**: 用 gold label 判定答案正确 → chosen，错误 → rejected
- **ScPO**: 用投票一致性（self-consistency）选择多数答案为 chosen，少数为 rejected

**问题**: 两者都只关注**最终答案**（outcome-level），完全忽略了回答内部的**推理逻辑一致性**。不同回答即使答案相同，推理路径可能完全不同；如果随机配对，模型难以学到 chosen 与 rejected 之间的关键差异。

**动机**: 能否利用 token 级的条件概率信息来衡量两个回答的"内在一致性"，从而选出最具信息量的偏好对？

## 方法详解

### 框架总览

PCPO 是一个迭代训练框架，每轮包含三步：

1. **生成 & 候选配对**: 用当前模型对每题生成 N=16 个回答，按答案正确性分为 $Y_w$（correct）和 $Y_l$（incorrect），通过 Levenshtein 距离筛选相似的候选偏好对
2. **概率一致性评分 & 偏好对选择**: 对候选对计算 token 级概率一致性加权分 $s_w$，选得分最高的配对作为训练对
3. **PCPO Loss 训练**: 用加权 DPO + NLL 损失训练下一轮模型

### 关键设计 1: Token 概率一致性评分

对一个候选偏好对 $(y_w, y_l)$，先用匹配函数（基于 SequenceMatcher 的最长公共子序列）对齐两个回答的公共 token。对每个匹配 token $y_t$，计算一致性得分：

$$c_t(y_w | y_l) = \exp(-|\log P_w(y_t|x, y_{<t}) - \log P_l(y_t|x, y_{<t})|)$$

- $c_t \in [0, 1]$，越接近 1 表示该 token 在两个回答中的条件概率越接近
- 直觉: 高一致性意味着两个回答在"犯错之前"的推理路径高度相似，差异集中在关键决策点

然后聚合为 pair-weighted score：$s_w(y_w|y_l) = \sum_t c_t(y_w|y_l) / l_{y_l}$

**选择策略**: 对每个 rejected 回答 $y_l$，选 $s_w$ 最高的 chosen 回答配对 → 保证偏好对之间推理路径最相似，差异最小且最关键。

### 关键设计 2: Levenshtein 距离预筛选

直接计算所有 $p \times q$ 候选对的概率一致性代价过高。PCPO 先用 Levenshtein 编辑距离为每个 rejected 回答选出 top-k（k=8）最相似的 chosen 回答，大幅减少计算量。实验表明 rank 1-5 覆盖了 95.4% 的最终选中对。

### 关键设计 3: PCPO 损失函数

$$\mathcal{L}_{PCPO} = \underbrace{-s_w \cdot \log\sigma\left(\beta\log\frac{M_\theta(y^+|x)}{M_t(y^+|x)} - \beta\log\frac{M_\theta(y^-|x)}{M_t(y^-|x)}\right)}_{\text{Weighted DPO Loss}} \underbrace{- \frac{\alpha \cdot s_w}{|y^+|} \log M_\theta(y^+|x)}_{\text{Weighted NLL Loss}}$$

- 用 $s_w$ 动态加权每个样本，高一致性对（更有信息量）获得更大梯度
- NLL 项防止模型偏离语言建模能力（$\alpha=1, \beta=0.5$）

## 实验结果

### 主实验: 多模型多 Benchmark 对比

| 方法 | GSM8K Pass@1 | MATH-500 Pass@1 | OlympiadBench Pass@1 | AMC23 Pass@1 |
|:---|:---:|:---:|:---:|:---:|
| Llama3-8B Seed | 71.3 | 30.8 | 8.1 | 10.0 |
| IRPO M2 | 81.1 | 30.6 | 6.7 | 0 |
| ScPO M2 | 81.6 | 32.2 | 7.9 | 5.0 |
| **PCPO M2** | **82.8** | **33.2** | **9.5** | **10.0** |
| Mathstral-7B Seed | 84.3 | 57.2 | 21.8 | 25.0 |
| IRPO M2 | 87.7 | 58.4 | 24.6 | 20.0 |
| ScPO M2 | 87.6 | 60.4 | 24.1 | 27.5 |
| **PCPO M2** | **89.0** | **61.8** | **25.2** | **32.5** |

### 泛化性: PCPO 数据 + 不同 DPO 变体

| 方法 | MATH-500 Pass@1 | OlympiadBench Pass@1 | AMC23 Pass@1 |
|:---|:---:|:---:|:---:|
| IPO M1 | 24.4 | 8.1 | 10.0 |
| **PCPO+IPO M1** | **32.2** | **9.9** | **15.0** |
| ORPO M1 | 27.0 | 8.0 | 10.0 |
| **PCPO+ORPO M1** | **29.0** | **8.6** | **10.0** |
| TDPO M1 | 29.8 | 7.7 | 5.0 |
| **PCPO+TDPO M1** | **30.4** | **8.4** | **5.0** |

PCPO 的偏好对选择策略可即插即用地提升 IPO/ORPO/TDPO/RPO 等多种 DPO 变体。

## 亮点与创新

- **首次在偏好对选择阶段引入 token 级概率一致性**，而非仅依赖 outcome-level 信号，选出推理路径最相似但结果不同的"硬"样本对
- **通用框架**: PCPO 的数据选择策略可与任意 DPO 变体组合使用，实验验证了对 5 种变体的泛化性
- **Case study 直观**: 通过具体案例展示 PCPO 能精确区分不同推理 pattern 并配对，而 outcome-only 方法完全随机

## 局限性

- **依赖 gold label**: 偏好对构建需要已知正确答案，限制了对无标注数据场景的适用性
- **额外计算开销**: token 概率计算使总训练时间增加约 15%（8.9 vs 7.7 GPU-hours/iteration）
- **仅验证数学推理**: 未在代码生成、常识推理等其他推理任务上验证泛化性
- **7B 模型为主**: 未在更大模型（70B+）上实验，在已充分优化的 Qwen2.5 系列上提升有限

## 相关工作

- **偏好优化 for 数学推理**: DPO → IRPO（迭代+答案正确性）→ ScPO（自一致性投票）→ PCPO（token 概率一致性）
- **Token-level 偏好优化**: TDPO（token 级 KL 约束）、SparsePO（稀疏 token mask）、cDPO（关键 token 识别）——这些方法在优化阶段操作 token，PCPO 在数据选择阶段利用 token 信息
- **DPO 变体**: IPO（防过拟合）、ORPO（无参考模型）、RLCD（对比蒸馏）

## 评分

⭐⭐⭐⭐ — 从 token 概率角度重新审视偏好对选择是一个新颖且直觉清晰的切入点，实验覆盖 4 个模型 × 4 个 benchmark × 5 种 DPO 变体非常充分，但方法依赖 gold label 且仅限数学推理场景。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] RPO: Retrieval Preference Optimization for Robust Retrieval-Augmented Generation](rpo_retrieval_preference_optimization_for_robust_retrieval-augmented_generation.md)
- [\[ACL 2025\] Optimal Transport-Based Token Weighting for Enhanced Preference Optimization](otpo_token_weighting.md)
- [\[ACL 2025\] ASPO: Adaptive Sentence-Level Preference Optimization for Fine-Grained Multimodal Reasoning](aspo_adaptive_sentence-level_preference_optimization_for_fine-grained_multimodal.md)
- [\[ACL 2025\] Boosting Vulnerability Detection of LLMs via Curriculum Preference Optimization with Synthetic Reasoning Data](boosting_vulnerability_detection_of_llms_via_curriculum_preference_optimization_.md)
- [\[ACL 2025\] Teaching an Old LLM Secure Coding: Localized Preference Optimization on Distilled Preferences](teaching_an_old_llm_secure_coding.md)

</div>

<!-- RELATED:END -->
