---
title: >-
  [论文解读] Dialogue Systems for Emotional Support via Value Reinforcement
description: >-
  [ACL 2025][emotional support] 提出 ES-VR，首个将人类价值观强化融入情感支持对话系统的方法，通过目标价值检测器和参考生成器（均在 Reddit 数据上训练），结合 SFT + DPO 两阶段训练，使支持者模型不仅能缓解求助者的负面情绪，还能探索和强化其积极价值观，实现更深层的内在转变。
tags:
  - ACL 2025
  - emotional support
  - value reinforcement
  - dialogue system
  - DPO
  - seeker simulator
---

# Dialogue Systems for Emotional Support via Value Reinforcement

**会议**: ACL 2025  
**arXiv**: [2501.17182](https://arxiv.org/abs/2501.17182)  
**代码**: [GitHub](https://github.com/holi-lab/ES-Value)  
**领域**: 文本生成 / 对话系统  
**关键词**: emotional support, value reinforcement, dialogue system, DPO, seeker simulator

## 一句话总结

提出 ES-VR，首个将人类价值观强化融入情感支持对话系统的方法，通过目标价值检测器和参考生成器（均在 Reddit 数据上训练），结合 SFT + DPO 两阶段训练，使支持者模型不仅能缓解求助者的负面情绪，还能探索和强化其积极价值观，实现更深层的内在转变。

## 研究背景与动机

**领域现状**: 情感支持对话系统旨在帮助求助者（seekers）缓解日常情感困难。近年 LLM 加速了此类系统的发展，许多模型聚焦于强化求助者的积极情绪。但仅关注情绪变化可能无法捕捉更深层的内在转变。

**现有痛点**: 情绪变化不等于真正的内在转变——求助者一句敷衍的"谢谢"在情感分类器中得分（0.758）甚至高于表达真正改变意愿的回复（0.583），但后者才是真正有效的支持结果。现有方法大多忽略了**人类价值观**在情感支持中的作用。

**核心矛盾**: 价值观（value）是塑造个人优先级的核心信念，在 ACT（接纳与承诺疗法）等现代心理治疗中扮演关键角色。但现有对话系统几乎没有探索将价值观强化融入情感支持。

**本文目标**: 如何训练一个对话系统，不仅提供情感安慰，还能识别并强化求助者的积极价值观，促进长期的内在变化。

**切入角度**: 利用 Reddit 的 r/offmychest 社区数据训练价值检测器和参考生成器，然后通过模拟对话 + DPO 训练支持者模型。

**核心 idea**: 用"价值观强化"替代传统的"情绪强化"作为情感支持的优化目标，通过目标价值检测 + 参考回复生成 + 基于价值奖励的 DPO 训练来实现。

## 方法详解

### 整体框架

三个核心组件：(1) **目标价值检测器**（Target Value Detector）——预测每轮应强化哪些价值观；(2) **参考生成器**（Reference Generator）——生成能促进目标价值观的参考回复；(3) **支持者模型**（Supporter Model）——结合目标价值和参考回复，选择策略并生成最终回复。

### 关键设计

1. **价值观分类体系**: 采用 Kiesel et al. (2022) 的分类法，融合 Schwartz 基本价值观理论和三个其他主要价值观列表，共 20 个价值观类别。通过分析 ESConv 数据集发现：高效果组（支持后情绪从 5 降到 1-2）的求助者在最后 4 轮中表达的积极价值观平均为 **7.9 个**，显著高于低效果组的 **6.5 个**。

2. **Reddit 数据构建**: 从 r/offmychest 收集 2019-2023 年帖子和评论，用情感强度模型和价值检测模型（Schroter et al., 2023）标注。正面评论中 OP 表达的价值观被视为成功的目标价值，之前的评论者回复被视为有效的支持话语。最终数据集超 20,000 条。

3. **目标价值检测器**: 给定对话历史 $(o_1, c_1, ..., c_{t-1}, o_t)$，预测下一轮应该强化的价值观 $v_{t+1} = \text{LM}_{\text{TVD}}(o_1, c_1, ..., c_{t-1}, o_t)$。真实标签为 $o_{t+1}$ 中概率最高的 top-3 价值观。

4. **参考生成器**: 两阶段训练——SFT 阶段学习根据对话历史和目标价值生成支持者回复 $c_t = \text{LM}_{\text{RG}}(o_1, c_1, ..., o_t; v_{t+1})$；DPO 阶段构建偏好数据（原始回复为 preferred，同一帖子下其他评论为 rejected，排除共享价值的重叠），进一步优化生成质量。

5. **支持者模型**: 在每一轮处理四个推理步骤：(a) 识别求助者问题和当前状态；(b) 分析参考回复的关键内容；(c) 决定是否采纳参考回复（生成 Yes/No 及理由）；(d) 选择情感支持策略并生成最终回复。

    - **SFT 阶段**: 用 GPT-4o-mini 作为支持者和求助者模拟器生成对话数据（33,130 条训练、2,367 条验证），蒸馏到 Llama-3-8B-Instruct。为防止模型继承 GPT 90% 不使用参考回复的偏好，在每个支持者轮次额外模拟"替代回复"。
    - **DPO 阶段**: 基于价值奖励构建偏好数据：$R(u_t^{\text{sup}}) = \sum_{k=1}^h \gamma^{k-1} N_{t+k}$，其中 $N_{t+k}$ 是目标价值在求助者后续第 $k$ 轮话语中出现的频率，$\gamma$ 为折扣因子，$h$ 为前瞻步数。当奖励差超过阈值 $T_{\text{diff}}$ 时加入偏好数据集。

6. **求助者模拟器**: 基于 GPT-4o-mini，使用 GPT-4o/4o-mini 生成 2,036 个独特人格（包含问题类型、情绪、情境），人工评估显示其自然度不低于真实人类求助者。

### 损失函数/训练策略

- SFT: 标准语言模型负对数似然
- DPO: 直接偏好优化（Rafailov et al., 2023），隐式奖励建模

## 实验关键数据

### GPT-4o-mini 消融（价值+参考的效果）

| 设置 | Sugg.↑ | Expe.↑ | Info.↑ | Overall↑ | Intensity↓ | Seeker-Value↑ | Supporter-Value↑ |
|------|--------|--------|--------|----------|------------|---------------|------------------|
| GPT 基线 | 4.03 | 2.34 | 4.11 | 4.44 | 2.19 | 0.43 | 0.36 |
| + Target values | 4.38 | 2.48 | 4.27 | 4.59 | 1.96 | 0.48 | 0.48 |
| + Reference | 4.34 | 2.54 | 4.29 | 4.61 | 1.89 | 0.47 | 0.42 |
| + Both | **4.57** | **3.11** | **4.42** | **4.72** | **1.89** | — | — |

### 主实验（与基线对比，关键指标）

| 方法 | Sugg.↑ | Expe.↑ | Info.↑ | Over.↑ | Intensity↓ | Seeker-V↑ | Supp-V↑ |
|------|--------|--------|--------|--------|------------|-----------|---------|
| GPT-4o-mini (+Both) | 4.57 | 3.11 | 4.42 | 4.72 | 1.89 | 0.49 | 0.42 |
| Llama-Psych8k | 4.75 | 2.89 | 4.63 | 4.75 | 1.53 | 0.49 | 0.62 |
| PPDPP | 4.45 | 2.49 | 4.26 | 4.54 | 1.83 | 0.44 | 0.31 |
| Emotion-DPO | 4.74 | 4.05 | 4.61 | 4.82 | 1.86 | 0.49 | 0.51 |
| **ES-VR (DPO)** | **4.80** | **4.20** | **4.65** | **4.86** | 1.81 | **0.52** | **0.56** |

ES-VR (DPO) 在 Experience（4.20 vs 3.11）和价值强化（Supporter 0.56 vs 0.42）上大幅领先 GPT-4o-mini。

### 心理治疗师评估

治疗师将 ES-VR 与 GPT 和 Emotion-DPO 进行两两对比。ES-VR 的优势在于：
- **验证求助者的挑战**（Validating challenges）
- **强调情况中的积极方面**（Emphasizing positive aspects）
- 这两项正是价值强化的核心要素

### 关键发现

- **价值强化 > 情绪强化**: ES-VR (DPO) 在 ES-Value 指标上全面优于 Emotion-DPO，证明目标从情绪转向价值观的有效性
- **Reddit 众包知识有价值**: 利用 Reddit 数据训练的参考生成器显著增强了支持效果（Experience 从 2.34 提升到 3.11）
- **DPO 阶段至关重要**: SFT 到 DPO 在 Experience 上从 3.76 跃升到 4.20，Overall 从 4.78 到 4.86
- **求助者模拟器质量高**: 人工评估显示 GPT-4o-mini 模拟器的自然度不低于真实人类求助者
- **参考回复的选择性采纳很重要**: 只在合适时使用参考回复（约 10%），而非每次都强制使用

## 亮点与洞察

- **首创性**: 首次将价值观强化明确融入情感支持对话系统，开辟了一个有心理学理论支撑的全新研究方向
- **ESConv 数据分析**提供了有力的动机证据——高效果支持中积极价值观表达显著更多
- **奖励函数设计精巧**: 使用前瞻窗口 $R = \sum \gamma^{k-1} N_{t+k}$ 评估价值强化效果，比单步奖励更能捕捉长期影响
- **模拟器验证充分**: 不仅做了自动评估，还请人工评估模拟器的自然度，增强了实验结论的可信度
- **治疗师参与评估**（持证临床心理学家）赋予了专业权威性

## 局限与展望

- 依赖 GPT-4o-mini 进行对话模拟和评估，存在循环依赖风险——GPT 生成的数据可能偏向 GPT 评估偏好
- 求助者模拟器的 2,036 个人格是否能覆盖真实世界的情感支持场景多样性存疑
- 价值检测模型（Schroter et al., 2023）的准确性直接影响整个流程，但其误差传播未充分分析
- 评估在模拟环境下进行，未部署到真实用户场景验证
- Reddit 数据的质量和代表性可能有偏——r/offmychest 社区用户群体特定

## 相关工作与启发

- **ESConv** (Liu et al., 2021) 是最核心的情感支持数据集，本文从中发现了价值观与支持效果的关联
- **PPDPP** (Deng et al., 2024) 代表了基于模拟的情感支持训练方法，ES-VR 在其基础上引入价值维度
- **ACT（接纳与承诺疗法）** 的心理学理论为本文提供了核心动机
- **DPO** (Rafailov et al., 2023) 的直接偏好优化在此场景下非常适用——价值强化的奖励天然形成偏好对
- 启发：将心理学中经过验证的治疗框架（如价值导向干预）转化为可优化的 NLP 目标，是一个有前景的跨学科方向

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次将价值观强化融入情感支持系统，方向开创性强
- **实验充分度**: ⭐⭐⭐⭐ — 多维度评估（技能/情绪/价值）、治疗师评估、模拟器验证，但缺乏真实用户实验
- **写作质量**: ⭐⭐⭐⭐ — 动机论证有力（ESConv 数据分析），方法描述清晰
- **价值**: ⭐⭐⭐⭐⭐ — 开辟了新研究方向，心理学理论支撑充分，数据和代码均已开源

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Enhancing Goal-oriented Proactive Dialogue Systems via Consistency Reflection and Correction](enhancing_goal-oriented_proactive_dialogue_systems_via_consistency_reflection_an.md)
- [\[ACL 2025\] EnSToM: Enhancing Dialogue Systems with Entropy-Scaled Steering Vectors for Topic Maintenance](enstom_enhancing_dialogue_systems_with_entropy-scaled_steering_vectors_for_topic.md)
- [\[ACL 2026\] Cognitive Policy-Driven LLM for Diagnosis and Intervention of Cognitive Distortions in Emotional Support Conversation](../../ACL2026/dialogue/cognitive_policy-driven_llm_for_diagnosis_and_intervention_of_cognitive_distorti.md)
- [\[ACL 2026\] ETHICMIND: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue](../../ACL2026/dialogue/ethicmind_a_risk-aware_framework_for_ethical-emotional_alignment_in_multi-turn_d.md)
- [\[ACL 2025\] When Harry Meets Superman: The Role of The Interlocutor in Persona-Based Dialogue Generation](when_harry_meets_superman_the_role_of_the_interlocutor_in_persona-based_dialogue.md)

<!-- RELATED:END -->
