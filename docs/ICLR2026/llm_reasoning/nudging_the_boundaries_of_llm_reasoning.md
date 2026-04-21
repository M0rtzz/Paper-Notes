---
title: >-
  [论文解读] Nudging the Boundaries of LLM Reasoning
description: >-
  [ICLR 2026][LLM推理][强化学习推理] 指出GRPO无法从模型完全无法解决的难题(pass rate=0%)中学习的根本局限，提出NuRL方法在训练时对难题注入自生成的抽象hint(不泄露答案)使其变为可学习样本，跨3个模型6个benchmark一致超越GRPO并真正提升pass@k能力上界。
tags:
  - ICLR 2026
  - LLM推理
  - 强化学习推理
  - GRPO改进
  - 自生成Hint
  - 能力上界突破
  - 近侧发展区
---

# Nudging the Boundaries of LLM Reasoning

**会议**: ICLR 2026  
**arXiv**: [2509.25666](https://arxiv.org/abs/2509.25666)  
**代码**: [GitHub](https://github.com/SalesforceAIResearch/NuRL)  
**领域**: LLM推理  
**关键词**: 强化学习推理, GRPO改进, 自生成Hint, 能力上界突破, 近侧发展区

## 一句话总结
指出GRPO无法从模型完全无法解决的难题(pass rate=0%)中学习的根本局限，提出NuRL方法在训练时对难题注入自生成的抽象hint(不泄露答案)使其变为可学习样本，跨3个模型6个benchmark一致超越GRPO并真正提升pass@k能力上界。

## 研究背景与动机
- **核心限制**: 在线RL(GRPO)对完全无法解出的难题(所有rollout都错)产生零梯度，模型无法从中学习
- **分布锐化假说**: 越来越多证据表明RL后训练主要做"分布锐化"——提高已知解法概率，而非发现新推理能力
- **pass@k不变**: 大k下的pass@k在RL训练后往往不变，说明能力上界未被突破
- **Zone of Proximal Development**: 类比Vygotsky的近侧发展区——难题在"学习区"但无指导就无法进入
- **难题的价值**: 这些"不可解"问题蕴含丰富学习信号——暴露模型的弱点
- **自给自足需求**: 需要无需外部强模型即可突破能力边界的方法

## 方法详解

### 整体框架
NuRL = 离线Hint收集 + 在线Rollout增强(两阶段训练)

### 关键设计
1. **离线Hint收集**:

    - 输入: (问题q, 正确答案a)
    - Step 1: 让模型生成"为什么答案正确"的CoT: $y = \pi_{old}(q, a; p_y)$
    - Step 2: 从CoT抽象出高层hint(核心知识线索): $h = \pi_\theta(q, a, y; p_h)$
    - 关键约束: hint必须抽象且高层，不包含具体答案或解题步骤

2. **在线Rollout增强**:

    - GRPO训练中对每个问题生成 $\mathcal{G}$ 个rollout
    - 若全部失败(pass rate=0%): 将hint拼接到问题末尾
    - 重新生成 $\mathcal{G}-1$ 个带hint的rollout + 1个不带hint的rollout(避免全部正确导致零方差)
    - 推理时不用hint——训练时的hint帮助模型内化推理模式

3. **Hint类型探索**:

    - 抽象线索(最佳) > 部分步骤 > 解释 > 直接答案(最差)
    - 核心发现: 暴露越多答案信息，性能越差——与人类学习规律一致

### 训练策略
- Stage 1: 标准GRPO训练至训练奖励和验证准确率收敛
- Stage 2: NuRL继续训练——过滤掉Stage 1中全部正确的简单题

## 实验关键数据

### 主实验

| 模型 | 方法 | MATH500 | MATH Hard | AIME | GPQA | Date | 平均 |
|------|------|:-------:|:---------:|:----:|:----:|:----:|:----:|
| Llama-3B | GRPO | 56.92 | 30.11 | 8.33 | 27.98 | 57.10 | 35.87 |
| Llama-3B | **NuRL(Self)** | **58.04** | **31.62** | **9.17** | **28.28** | **61.65** | **37.49** |
| OctoThinker-3B | GRPO | 68.81 | 41.29 | 8.33 | 23.26 | 69.85 | 42.63 |
| OctoThinker-3B | **NuRL(Self)** | **70.13** | **42.07** | **9.66** | **27.15** | **71.75** | **44.38** |

### 消融实验

| 配置 | MATH500 | GPQA | 说明 |
|------|:-------:|:----:|------|
| Hint从头训练 + 无触发器 | 53.41 | 24.84 | 最差 |
| Hint从头训练 + 仅全失败触发 | 56.06 | 27.63 | 触发器有帮助 |
| 两阶段 + 无触发器 | 53.09 | 26.62 | 两阶段也有帮助 |
| 两阶段 + 仅全失败触发(NuRL) | **58.04** | **28.28** | 最佳 |

### 关键发现
- NuRL提升pass@1024: GPQA从63.6%→69.7%，Date从86.4%→94.0%，而GRPO保持不变——能力上界被突破
- 教师hint(GPT-o4-mini)进一步提升至+3.44%，自生成hint即可有效
- NuRL+Self-Consistency提升9.4% vs GRPO+SC仅7.8%——互补性更强
- 可学习问题比例随训练从66%上升至70%，而标准GRPO维持在66%

## 亮点与洞察
- 清晰且深刻地揭示GRPO无法学习不可解问题的根本限制
- Vygotsky近侧发展区类比准确贴切，动机极具启发性
- "越抽象的hint越好"反直觉但有力——直接给答案反而最差(reward hacking)
- 自生成hint无需外部模型，避免分布偏移，自给自足
- 两阶段策略(先GRPO收敛再NuRL)简洁实用

## 局限与展望
- 改进幅度相对温和(+1-2%平均)，在强模型(Qwen3-4B)上提升有限(+0.79%)
- 自生成hint质量受限于模型本身能力——极难问题可能生成不了有用hint
- 二值判断(全失败/部分成功)决定是否注入hint，缺乏更细粒度策略
- 离线hint收集需要gold answer，限制无答案场景适用性
- 未探索hint的质量评估和动态更新机制

## 相关工作与启发
- **vs GRPO/DAPO/Dr.GRPO**: 它们改进advantage估计/KL/采样，NuRL正交地解决"不可解样本"问题
- **vs STaR**: STaR用answer-conditioned reasoning，NuRL进一步抽象为不泄露答案的hint
- **vs TBA**: TBA用多搜索节点生成多样轨迹，NuRL用hint降低问题难度

## 评分
- 新颖性: ⭐⭐⭐⭐ GRPO上界限制的insight + 自生成hint方案
- 实验充分度: ⭐⭐⭐⭐ 3模型6benchmark + 多hint类型 + pass@k分析
- 写作质量: ⭐⭐⭐⭐⭐ ZPD类比优美，动机→方法→实验逻辑流畅
- 价值: ⭐⭐⭐⭐ 解决RL推理训练的实际瓶颈，方法简洁可落地

<!-- RELATED:START -->

## 相关论文

- [On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning](on_the_design_of_kl-regularized_policy_gradient_algorithms_for_llm_reasoning.md)
- [DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning](designer_design-logic-guided_multidisciplinary_data_synthesis_for_llm_reasoning.md)
- [The Path of Least Resistance: Guiding LLM Reasoning Trajectories with Prefix Consensus](the_path_of_least_resistance_guiding_llm_reasoning_trajectories_with_prefix_cons.md)
- [LogicReward: Incentivizing LLM Reasoning via Step-Wise Logical Supervision](logicreward_incentivizing_llm_reasoning_via_step-wise_logical_supervision.md)
- [No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes](no_answer_needed_predicting_llm_answer_accuracy_from_question-only_linear_probes.md)

<!-- RELATED:END -->
