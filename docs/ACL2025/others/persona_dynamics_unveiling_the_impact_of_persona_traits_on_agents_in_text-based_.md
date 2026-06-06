---
title: >-
  [论文解读] Persona Dynamics: Unveiling the Impact of Personality Traits on Agents in Text-Based Games
description: >-
  [ACL 2025][personality traits] 提出 PANDA 方法，将人类人格特质（Big Five + Dark Triad 共8种）投射到文本游戏智能体的策略学习中，通过人格分类器引导 Q 值调整，发现高开放性（Openness）人格在冒险类文本游戏中表现显著优于其他人格类型。
tags:
  - "ACL 2025"
  - "personality traits"
  - "text-based games"
  - "Big Five"
  - "Dark Triad"
  - "DRRN"
  - "强化学习"
---

# Persona Dynamics: Unveiling the Impact of Personality Traits on Agents in Text-Based Games

## 基本信息

**会议**: ACL 2025  
**arXiv**: 2504.06868  
**代码**: [pull-ups/PANDA](https://github.com/pull-ups/PANDA)  
**模型**: [mirlab/PersonalityClassifier](https://huggingface.co/mirlab/PersonalityClassifier)  
**领域**: 其他  
**关键词**: personality traits, text-based games, Big Five, Dark Triad, DRRN, reinforcement learning  

## 一句话总结

提出 PANDA 方法，将人类人格特质（Big Five + Dark Triad 共8种）投射到文本游戏智能体的策略学习中，通过人格分类器引导 Q 值调整，发现高开放性（Openness）人格在冒险类文本游戏中表现显著优于其他人格类型。

## 研究背景与动机

- **文本游戏挑战**：文本游戏（text-based games）作为 AI 的经典挑战场景，不同于 Atari/围棋等预定义动作空间的游戏，需要理解自然语言描述的环境并生成文本形式的动作，涉及复杂的自然语言理解和生成能力
- **价值对齐的拓展**：现有工作关注将通用价值体系（如伦理道德、社会规范）融入智能体行为，但尚未探索多样化的**内在人格特质**如何影响智能体的行为决策
- **核心问题**：不同人格特质如何影响智能体在交互环境中的行为和性能？是否存在某些"有利"的人格类型？
- **研究意义**：通过引入人格维度评估智能体行为，为开发更加人类中心的、行为可控的 AI 系统提供新视角

## 方法详解

### 整体框架：PANDA (Personality-Adapted Neural Decision Agents)

PANDA 的核心思想是通过两步来实现人格引导的智能体决策：

1. **训练人格分类器**：判断给定情境下的动作展现何种人格特征
2. **将人格引导整合到策略学习中**：通过调整 Q 值实现人格约束的动作选择

### 文本游戏环境建模

- 将文本游戏建模为**部分可观测马尔可夫决策过程（POMDP）**：$(S, T, A, O, R)$
- 智能体在隐含状态 $S$ 下接收文本观测 $O$，生成文本动作 $A$，根据转移函数 $T$ 改变状态
- 使用 **Jiminy Cricket** 基准测试，包含 25 个复杂文本冒险游戏，涵盖超过 1800 个位置和近 5000 个可交互对象

### 基础智能体架构：DRRN

- 使用 **Deep Reinforcement Relevance Network (DRRN)** 作为基础框架
- 神经网络训练预测 Q 值 $Q(s_t, a_t)$，即给定状态下动作的价值函数
- 通过 softmax 策略选择最大化 Q 值的动作

### 关键设计：人格引导的 Q 值调整

核心公式为调整后的 Q 值计算：

$$Q'(s_t, a_t^i) = Q(s_t, a_t^i) + \gamma \cdot C(s_t, a_t^i \mid p)$$

- $Q(s_t, a_t^i)$：原始策略网络的动作价值
- $C(s_t, a_t^i \mid p) \in \{-1, 0, 1\}$：人格分类器的输出，表示该动作在人格 $p$ 上的高/中/低效价
- $\gamma$：引导强度和方向。$\gamma > 0$ 增强匹配人格的行为，$\gamma < 0$ 抑制匹配人格的行为

动作选择策略：

$$\pi(a_t = a_t^i | s_t) = \frac{\exp(Q(s_t, a_t^i))}{\sum_{j=1}^{|A_t|} \exp(Q(s_t, a_t^j))}$$

### 人格分类器的构建

**数据集构建（12万样本）**：

1. **起始验证人格描述**：使用 BFI（Big Five Inventory）和 SD-3（Short Dark Triad）的验证问卷条目，每个特质生成 10 个实例（5 高效价 + 5 低效价），共 80 条
2. **情境种子增强**：使用 GPT-4 生成 300 个多样化情境（分为 30 个子集，每子集 10 个场景）
3. **最终规模**：$80 \times 300 \times 5 = 120,000$ 个带人格标签的样本

**涵盖的 8 种人格特质**：
- Big Five：开放性(Ope.)、尽责性(Con.)、外向性(Ext.)、宜人性(Agr.)、神经质(Neu.)
- Dark Triad：精神病态(Psy.)、自恋(Nar.)、马基雅维利主义(Mac.)

**分类器训练**：
- 使用 Flan-T5-XL（30亿参数）进行微调
- 输入：情境 + 动作 + 人格类型 → 输出：高/中/低效价
- 在验证集上达到 **98.59%** 准确率

## 实验

### 主实验设置

- **环境**：Jiminy Cricket 基准的 25 个文本冒险游戏
- **智能体配置**：16 种人格型（每种 8 个人格 × 高/低两个方向）+ 1 个无人格基线
- **评估**：每个配置取最后 50 个回合的平均分，使用 3 个随机种子

### 主实验结果（表3）

在 15 个有效游戏中的表现（剩余 10 个游戏各人格型均无法得分）：

| 指标 | High Openness | No Personality | Low Openness |
|------|--------------|----------------|--------------|
| 平均分(Avg.) | **6.5** | 5.6 | 4.4 |
| 优势游戏数(Cnt.) | **11** | - | 0 |
| 高低差(Diff.) | +2.1 | - | - |

**核心发现**：高开放性(High Openness)智能体在三项指标上全面领先，平均分提升 16%，在 11/15 个游戏中取得最佳成绩。

### 统计分析（表4）

- **Wilcoxon 符号秩检验**：高开放性 vs 无人格，$p = 0.002$；高开放性 vs 低开放性，$p = 0.000$
- **Friedman 检验**：开放性的检验统计量 $Fr = 25.2$，$p = 0.000$，远超其他人格特质
- 其他人格特质均未显示出统计显著的一致性优势

### 轨迹分析（表5）

| 指标 | No Personality | High Openness | Low Openness |
|------|----------------|---------------|--------------|
| 轨迹长度 | 45.85 | **57.04** | 39.86 |
| 访问节点数(总) | 9.49 | **10.16** | 8.32 |
| 未探索节点数 | 0.83 | **1.20** | 0.30 |

高开放性智能体具有更长的轨迹、访问更多的地点（尤其是未探索过的新地点），体现了其好奇心和探索新事物的特征。

### 消融实验

- **引导强度 $\gamma$**：测试不同 $\gamma$ 值的影响，发现适当的引导强度可有效引导行为而不过度干扰策略学习
- **分类器性能对比**：Flan-T5-XL 在 BFI 和 SD-3 验证集上达到 100% 和 96.29%，显著优于零样本 GPT-4o-mini（81.81% 和 22.22%）

## 亮点与洞察

1. **人格-游戏性能的有趣关联**：高开放性（好奇心、冒险精神）的人格特质在文本冒险游戏中具有天然优势，验证了心理学理论在 AI 行为中的适用性
2. **简洁高效的引导机制**：通过简单的 Q 值加法修正实现人格引导，不需要修改网络结构或训练流程
3. **大规模人格标注数据集**：12万条带人格标签的数据集，采用种子扩展的方法保证了多样性和覆盖度
4. **定量验证人格效应**：不仅展示了人格差异对性能的影响，还通过轨迹分析揭示了背后的行为机制（高开放性 → 更多探索 → 更高得分）

## 局限性

1. **环境局限**：仅在文本冒险游戏中验证，开放性的优势可能仅适用于探索导向的任务，未必推广到需要谨慎决策的场景
2. **LLM 时代的局限**：基础框架使用 DRRN（传统 RL 方法），未探索 LLM 作为智能体时人格引导的效果
3. **人格数据来源**：训练数据由 GPT-4 生成，可能存在偏差
4. **10 个游戏完全无法得分**：说明在过于困难的游戏中，人格引导的效果受限于基础策略的能力
5. **单一 $\gamma$ 值**：未针对不同游戏/人格动态调整引导强度

## 相关工作

- **文本游戏 AI**：DRRN (He et al., 2016)、CALM (Yao et al., 2020) 等基于 RL 的方法
- **智能体价值对齐**：Jiminy Cricket (Hendrycks et al., 2021) 伦理评估、MACHIAVELLI (Pan et al., 2023) 道德基准
- **人格计算**：Big Five (McCrae & Costa, 1987)、Dark Triad (Jones & Paulhus, 2014) 心理学量表在 NLP 中的应用

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性**：将心理学人格理论应用到游戏智能体的策略学习中，视角新颖（+1）
- **方法简洁性**：Q 值加法修正的方式简单直接，易于实现和理解（+0.5）
- **实验充分性**：25 个游戏、16 种人格配置、统计检验和轨迹分析，较为全面（+0.5）
- **实际影响力**：发现开放性人格有利于探索，为值对齐和人格化 AI 提供了参考（+0.5）
- **扣分**：未在 LLM 智能体上验证、部分游戏无效、环境较为单一（-1）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Deviation Dynamics in Cardinal Hedonic Games](../../AAAI2026/others/deviation_dynamics_in_cardinal_hedonic_games.md)
- [\[ACL 2025\] A Multi-Persona Framework for Argument Quality Assessment](a_multi-persona_framework_for_argument_quality_assessment.md)
- [\[ACL 2025\] Towards Text-Image Interleaved Retrieval](towards_text-image_interleaved_retrieval.md)
- [\[ACL 2025\] Identifying Reliable Evaluation Metrics for Scientific Text Revision](reliable_eval_metrics_scientific.md)
- [\[ACL 2025\] FRACTAL: Fine-Grained Scoring from Aggregate Text Labels](fractal_fine-grained_scoring_from_aggregate_text_labels.md)

</div>

<!-- RELATED:END -->
