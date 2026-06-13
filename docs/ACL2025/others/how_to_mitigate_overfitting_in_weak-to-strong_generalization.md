---
title: >-
  [论文解读] How to Mitigate Overfitting in Weak-to-Strong Generalization?
description: >-
  [ACL 2025][弱到强泛化] 提出两阶段训练框架解决弱到强泛化中的过拟合问题：第一阶段通过基于不确定性的过滤提高弱监督信号质量，第二阶段用已微调的强模型为被丢弃的难题重新生成答案以恢复问题质量，在 GSM8k 和 MATH 上将 PGR 从 7.19% 提升到 120.50%。 领域现状：超对齐（superalignm…
tags:
  - "ACL 2025"
  - "弱到强泛化"
  - "超级对齐"
  - "overfitting"
  - "data filtering"
  - "自一致性"
---

# How to Mitigate Overfitting in Weak-to-Strong Generalization?

**会议**: ACL 2025  
**arXiv**: [2503.04249](https://arxiv.org/abs/2503.04249)  
**代码**: 无  
**领域**: 其他  
**关键词**: 弱到强泛化, 超级对齐, overfitting, data filtering, 自一致性

## 一句话总结
提出两阶段训练框架解决弱到强泛化中的过拟合问题：第一阶段通过基于不确定性的过滤提高弱监督信号质量，第二阶段用已微调的强模型为被丢弃的难题重新生成答案以恢复问题质量，在 GSM8k 和 MATH 上将 PGR 从 7.19% 提升到 120.50%。

## 研究背景与动机

**领域现状**：超对齐（superalignment）的核心挑战是：当任务超越人类评估能力时，如何对齐超人模型？弱到强泛化（weak-to-strong generalization）探索弱监督者能否引导更强模型。

**现有痛点**：弱模型生成的标签包含噪声，强模型的强拟合能力导致其过拟合这些错误标签，性能退化严重。

**核心矛盾**：简单过滤掉错误标签虽能提高标签质量，但会同时丢弃有价值的难题样本，导致训练集难度和多样性退化（问题质量下降）——这形成了"监督信号质量"与"问题质量"之间的两难。

**本文目标** 同时提升监督信号质量和问题质量，打破过滤导致的质量-多样性权衡困境。

**切入角度**：从 Lang et al. (2024) 的扩展理论出发——弱到强泛化依赖两个机制：伪标签纠正和覆盖扩展，过度过滤虽改善前者但损害后者。

**核心 idea**：先过滤提纯监督信号，再用微调后的强模型为丢弃的难题重新标注，既保证标签正确性又恢复问题难度和多样性。

## 方法详解

### 整体框架
- **Stage I（提纯监督信号）**：弱模型对每题生成 10 个 CoT 回答 → 计算自一致性（self-consistency）→ 过滤掉低一致性样本 → 高一致性样本组成 Training Set A → 微调强模型
- **Stage II（恢复问题质量）**：用 Stage I 微调后的强模型对 Stage I 中被丢弃的题目重新生成答案 → 再次用一致性过滤 → 高置信样本组成 Training Set B → A+B 合并后重新微调原始强模型

### 关键设计

1. **基于不确定性的过滤（Uncertainty-based Filtering）**

    - 对每个问题用 CoT prompting 生成 10 个回答
    - 选择出现次数最多的答案作为最终回答
    - 计算置信度：$\text{Confidence}(\text{Ans}) = \frac{N_{Ans}}{N_{Total}} \times 100\%$
    - 设定一致性阈值（如 50%、60%、70%、80%）过滤低置信度样本
    - 实验验证：阈值越高，标签正确率越高（图3）

2. **问题退化分析**

    - **难度退化**：过滤阈值从低到高，平均难度从 3.48 降到 2.66，高难度题目（Level 4-5）比例锐减
    - **多样性退化**：某些主题（如 Counting and Probability）从 10.79% 跌至 4.31%，主题分布显著偏移

3. **Stage II：强模型重标注**

    - 利用 Stage I 微调后的强模型已超越弱教师的事实
    - 对弱模型不确定（被丢弃）的难题重新生成多样回答
    - 同样用一致性过滤确保新标签质量
    - 将高置信样本追加到训练集，增强训练数据的难度和多样性

### 损失函数 / 训练策略

- 标准 SFT 微调（数学推理的标准训练方式）
- 弱模型：Llama 3 8B Instruct / Deepseek 7B Chat
- 强模型：Llama 3 70B / Deepseek 67B Base
- 强上界（strong ceiling）：用真实标签微调的强模型
- 评估指标：$PGR = \frac{\text{weak-to-strong} - \text{weak}}{\text{strong ceiling} - \text{weak}}$
- 数据集：GSM8k（训练集）、MATH（训练集），使用与 Yang et al. (2024b) 相同的训练集

## 实验关键数据

### 主实验

**Llama 3（8B Instruct → 70B）**：

| 阶段 | GSM8k Acc | GSM8k PGR | MATH Acc | MATH PGR |
|------|-----------|-----------|----------|----------|
| Baseline | 75.20% | 7.19% | 18.2% | 36.17% |
| Stage I | 80.28% | 98.56% | 34.0% | 112.77% |
| Stage II | 81.50% | **120.50%** | 35.2% | **121.28%** |

**Deepseek（7B Chat → 67B Base）**：

| 阶段 | GSM8k Acc | GSM8k PGR | MATH Acc | MATH PGR |
|------|-----------|-----------|----------|----------|
| Baseline | 62.39% | 51.39% | 16.8% | 65.85% |
| Stage I | 71.11% | 83.33% | 21.2% | 119.51% |
| Stage II | 72.94% | **90.04%** | 21.8% | **126.83%** |

- PGR 超过 100% 意味着弱到强方法的准确率甚至超过了用真实标签训练的上界

### 消融实验

**Stage II 中过滤的必要性**（Llama 3, GSM8k）：

| Stage I 阈值 | 无 Stage II | 有过滤 Stage II | 无过滤 Stage II |
|-------------|------------|----------------|----------------|
| 50% | 78.99 | 80.89 (+1.90) | 78.31 (-0.68) |
| 60% | 80.07 | 81.50 (+1.43) | 78.84 (-1.23) |
| 70% | 80.28 | 81.19 (+0.91) | 80.28 (+0.00) |
| 80% | 80.06 | 80.74 (+0.68) | 79.59 (-0.47) |

- 不加过滤直接追加所有重新标注的样本会导致性能下降
- 验证了 Stage II 中不确定性过滤的必要性

**迭代细化探索**（Deepseek）：
- 在 Stage II 基础上再加一轮迭代（Stage Exp），MATH PGR 从 126.83% 进一步提升到 134.15%
- 说明迭代细化有继续提升的空间

### 关键发现

1. **过滤的双刃剑效应**：过滤阈值存在最优点——过低无法去噪，过高丢失难题。Stage I 性能先升后降清晰展示了这一权衡
2. **Stage II 的鲁棒性**：在所有过滤阈值下，Stage II 都能带来额外提升，且对高阈值（过度过滤）的场景恢复效果更显著
3. **难度和多样性恢复**：Stage II 的精炼数据集在难度分布和主题多样性上更接近原始数据集
4. **PGR > 100% 的含义**：弱到强泛化可以超越强上界，这可能是因为弱监督的过滤过程本身起到了数据增强和去噪的效果

## 亮点与洞察

1. **问题洞察深刻**：不仅发现"过滤提升标签质量"，更进一步发现"过度过滤损害问题质量"，这一对立关系是本文的核心贡献
2. **两阶段框架优雅**：Stage I 打基础（提纯），Stage II 补短板（恢复难题），逻辑清晰且互补
3. **自我增强循环**：微调后的强模型用来标注弱模型无法处理的题目，形成正反馈循环
4. **实验设计全面**：不同模型系列（Llama 3, Deepseek）、不同数据集（GSM8k, MATH）、不同阈值的全面网格实验

## 局限与展望

1. 仅在数学推理任务上验证，其他领域的有效性待验证
2. 最优一致性阈值因任务和数据集而异，自动确定阈值是开放问题
3. 两阶段微调的计算开销较高，尤其是 Stage II 需要对所有丢弃题目重新生成多个答案
4. 使用 instruct 版本作为弱监督者简化了实验设置，但可能不完全反映真实的人类-AI 对齐场景
5. 迭代细化的收敛性和最优迭代次数未深入研究

## 相关工作与启发

- **Burns et al. (2023)**：提出弱到强泛化的概念和 PGR 指标
- **Lang et al. (2024)**：提出弱到强泛化的两个机制（伪标签纠正 + 覆盖扩展），为本文框架提供理论基础
- **Guo & Yang (2024)**：引入过滤和置信度重加权，但未考虑问题质量退化
- **启发**：在任何涉及噪声标签学习的场景中，都应警惕"去噪"操作对数据分布的副作用——不只是清洁度，难度和多样性同样重要

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术深度 | 3 |
| 实验充分性 | 5 |
| 写作质量 | 4 |
| **总评** | **4.0** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Revisiting Weak-to-Strong Generalization: Reverse KL vs. Forward KL](revisiting_weak-to-strong_generalization_in_theory_and_practice_reverse_kl_vs_fo.md)
- [\[ACL 2025\] Synergistic Weak-Strong Collaboration by Aligning Preferences](synergistic_weak-strong_collaboration_by_aligning_preferences.md)
- [\[ACL 2025\] Well Begun is Half Done: Low-resource Preference Alignment by Weak-to-Strong Decoding](well_begun_is_half_done_low-resource_preference_alignment_by_weak-to-strong_deco.md)
- [\[ACL 2025\] Micro-Act: Mitigate Knowledge Conflict in QA via Actionable Self-Reasoning](micro_act_knowledge_conflict_reasoning.md)
- [\[ACL 2025\] FastDraft: How to Train Your Draft](fastdraft_how_to_train_your_draft.md)

</div>

<!-- RELATED:END -->
