---
title: >-
  [论文解读] Enhancing Mathematical Reasoning in LLMs by Stepwise Correction
description: >-
  [ACL 2025][LLM推理][数学推理] 本文提出StepCo（Stepwise Correction），一种迭代式"验证-修正"框架：利用过程监督验证器（PSV）逐步定位LLM推理路径中的首个错误步骤并触发LLM修正，在8个数学推理基准上以GPT-4o为后端取得94.1%平均准确率，超越Best-of-10方法+2.4个点，同时减少77.8%的token消耗。
tags:
  - ACL 2025
  - LLM推理
  - 数学推理
  - 逐步修正
  - Process Supervised Verifier
  - 迭代验证
  - 错误传播
---

# Enhancing Mathematical Reasoning in LLMs by Stepwise Correction

**会议**: ACL 2025  
**arXiv**: [2410.12934](https://arxiv.org/abs/2410.12934)  
**代码**: [https://github.com/wzy6642/StepCo](https://github.com/wzy6642/StepCo)  
**领域**: LLM推理  
**关键词**: 数学推理, 逐步修正, Process Supervised Verifier, 迭代验证, 错误传播

## 一句话总结
本文提出StepCo（Stepwise Correction），一种迭代式"验证-修正"框架：利用过程监督验证器（PSV）逐步定位LLM推理路径中的首个错误步骤并触发LLM修正，在8个数学推理基准上以GPT-4o为后端取得94.1%平均准确率，超越Best-of-10方法+2.4个点，同时减少77.8%的token消耗。

## 研究背景与动机
LLM通过生成中间推理步骤（Chain-of-Thought）来解决数学问题，但推理路径中的错误具有**传播性**——一个步骤的错误会级联影响后续所有步骤。现有的Best-of-N解码方法让LLM独立生成多条推理路径并选最优，但这种"重复独立采样"往往导致**重复犯相同错误**——当正确答案不在采样集中时，Best-of-N必然失败。

作者观察到21.2%的Best-of-10错误案例中，正确推理路径根本不在采样集中，LLM倾向于在相同位置犯相同错误。核心矛盾在于：没有外部反馈，LLM无法自我纠正（Huang et al., 2024）。

本文的切入点是：与其多次独立采样，不如用外部验证器精确定位错误步骤，然后只修正该步骤及后续部分——既解决了错误传播，又大幅减少了token消耗。

## 方法详解

### 整体框架
StepCo的工作流程：
1. 给定问题，LLM生成初始推理路径 $r^{(0)}$
2. 进入迭代验证-修正循环（最多T=5轮）：
    - **验证阶段**：PSV对每个步骤评估"导向正确答案的概率"$p(s_i)$，找到第一个低于阈值θ的步骤 $s_k$
    - **修正阶段**：保留 $s_k$ 之前的正确步骤，将错误步骤及其概率作为反馈，让LLM修正 $s_k$ 及后续步骤
3. 若所有步骤概率都超过θ，接受当前路径；否则重复上述过程

### 关键设计

1. **过程监督验证器（PSV）的训练**

    - 核心思想：用自动过程标注方法构建训练数据，然后微调Llama-3-8B作为PSV
    - 数据构建过程：对每个问题和金标答案，在每个步骤使用两种demonstration——$D^+$（鼓励正确步骤）和$D^-$（鼓励探索错误/替代步骤）——扩展推理路径，形成完全二叉树
    - 每个步骤的质量分数 = 以该步骤为根的子树中，叶节点匹配正确答案的比例：$\pi(s_m) = \frac{\sum_{s_\ell \in \mathcal{L}_m^q} \mathbb{I}(s_\ell = a)}{|\mathcal{L}_m^q|}$
    - 使用MSE损失微调Llama-3-8B（LoRA）

2. **验证阶段的错误定位**

    - 对推理路径中的每个步骤，将问题+前序步骤拼接输入PSV，预测该步骤导向正确答案的概率
    - 找到第一个概率低于阈值θ（默认0.5）的步骤，将其及后续步骤标记为需修正
    - 关键约束：$k^{(t)} \geq k^{(t-1)}$，确保验证不会回退到已验证通过的步骤

3. **修正阶段的精准反馈**

    - 不是简单让LLM重新生成，而是提供具体反馈："步骤 $s_k$ 导向正确答案的概率是 $p(s_k)$，请修正步骤 $s_k$ 及后续步骤"
    - 保留正确步骤不变，只修正错误部分，避免"修了对的、破坏了好的"

### 损失函数 / 训练策略
- PSV训练使用MSE损失：$\mathcal{J} = \frac{1}{|\mathcal{D}|}\sum_{j=1}^{|\mathcal{D}|}(\text{PSV}(x_j) - y_j)^2$
- 使用LoRA微调Llama-3-8B
- StepCo本身是推理框架，不需要额外训练后端LLM

## 实验关键数据

### 主实验（8个数学推理数据集，GPT-3.5-Turbo / GPT-4o）
| 方法 | SVAMP | GSM8K | MATH500 | AQuA | 平均 |
|------|-------|-------|---------|------|------|
| Zero-Shot-CoT | 76.7/90.4 | 78.6/94.6 | 37.9/74.0 | 51.3/72.8 | 72.9/86.7 |
| Best-of-10 | 85.5/93.9 | 85.3/94.5 | 42.1/77.0 | 66.1/81.1 | 80.1/91.7 |
| CRITIC | 83.3/93.5 | 79.2/95.4 | 44.9/74.9 | 63.8/80.2 | 78.4/90.4 |
| **StepCo** | **89.7/96.0** | **87.0/96.4** | **56.9/80.4** | **72.4/84.7** | **84.7/94.1** |

### 效率对比
| 方法 | 平均准确率(GPT-4o) | Token消耗 | 相比Best-of-10 |
|------|-------------------|----------|---------------|
| Best-of-10 | 91.7% | 100% | — |
| StepCo (1轮) | 92.3% | ~22.2% | +0.6, 省78% |
| StepCo (5轮) | **94.1%** | ~22.2% | +2.4, 省77.8% |

### 消融实验
| PSV模型 | 平均准确率(GPT-3.5-Turbo) |
|---------|-------------------------|
| StepCo (DiVeRSe) | 82.6 |
| StepCo (Math-Shepherd) | 83.8 |
| StepCo (本文PSV) | **84.7** |

### 非数学推理泛化（GPT-3.5/GPT-4o）
| 方法 | HotpotQA EM | HotpotQA F1 | CSQA ACC |
|------|------------|------------|---------|
| Best-of-10 | 32.9/52.0 | 44.1/57.1 | 73.0/83.4 |
| StepCo | **35.0/53.0** | **47.4/58.7** | **74.3/84.9** |

### 关键发现
- **修正分析**：StepCo将正确改错误的比例仅5.3%（GSM8K），而Self-Correct高达14.5%，说明外部反馈质量至关重要
- **难度分析**：在MATH500最高难度级别（Level 5），StepCo仍达29.1%，显著优于所有基线
- **阈值敏感性**：θ=0.5时效果最佳，更大的θ增加迭代次数但准确率略降
- **开源模型兼容**：用Llama-3-8B做后端LLM，StepCo在MATH500上比Best-of-10高+1.4
- **21.2%的Best-of-10错误**是因为正确路径不在采样集中——这是Best-of-N方法的固有缺陷

## 亮点与洞察
- **迭代修正优于重复采样**的核心论点非常有说服力：Best-of-N重复犯同样的错，而StepCo用外部反馈打破这个循环
- **自动过程标注方法**巧妙：用正/负demonstration构建二叉树，通过叶节点比例自动计算每步质量分数，无需人工标注
- **PSV泛化到非数学任务**：虽然仅在数学数据上训练，PSV在HotpotQA和CSQA上也有效果，说明"步骤质量"的概念有一定通用性
- **77.8%的token节省**在实际部署中有巨大意义

## 局限与展望
- 仅在英语数学任务上训练PSV，多语言和更广泛的推理类型（如代码、科学推理）未验证
- 对于答案非数值/实体型的问题（如开放式生成），当前方法难以适用
- PSV基于Llama-3-8B，对于极其复杂的推理（如竞赛级别数学），PSV本身可能无法给出准确判断
- 阈值θ需要根据任务手动调整
- **研究idea**：可以将StepCo与RL结合——用PSV提供的步骤级reward信号做在线强化学习，让模型本身学会生成less-error的推理路径，而不仅在推理时修正

## 相关工作与启发
- Math-Shepherd（Wang et al., 2024）：蒙特卡洛估计构建过程监督数据，StepCo在其框架基础上更进一步做迭代修正
- CRITIC（Gou et al., 2024）：外部反馈自我纠正的代表工作，StepCo更精确到步骤级别
- OmegaPRM（Luo et al., 2024）：自动过程标注方法，与StepCo的二叉树构建思路有相似之处
- Self-Correct的失败案例（14.5%正确变错误）再次验证了Huang et al.（2024）关于"无外部反馈无法自我纠正"的结论

## 评分
- 新颖性: ⭐⭐⭐⭐ 迭代验证-修正框架结合自动过程标注，思路清晰且有效区别于Best-of-N范式
- 实验充分度: ⭐⭐⭐⭐⭐ 8个数学基准+2个非数学任务，3种LLM后端，多种消融和分析
- 写作质量: ⭐⭐⭐⭐ 问题动机阐述clear，方法描述严谨，实验分析深入
- 价值: ⭐⭐⭐⭐ 实用性强，77.8%的token节省对实际部署有直接意义

<!-- RELATED:START -->

## 相关论文

- [Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification](safe_math_reasoning.md)
- [BPP-Search: Enhancing Tree of Thought Reasoning for Mathematical Modeling Problem Solving](bpp-search_enhancing_tree_of_thought_reasoning_for_mathematical_modeling_problem.md)
- [Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models](chain-of-reasoning_towards_unified_mathematical_reasoning_in_large_language_mode.md)
- [Linguistic Generalizability of Test-Time Scaling in Mathematical Reasoning](mclm_multilingual_test_time_scaling.md)
- [ClozeMath: Improving Mathematical Reasoning in Language Models by Learning to Fill Equations](clozemath_improving_mathematical_reasoning_in_language_models_by_learning_to_fil.md)

<!-- RELATED:END -->
