---
title: >-
  [论文解读] Generative Value Conflicts Reveal LLM Priorities
description: >-
  [ICLR 2026][LLM/NLP][价值冲突] 提出 ConflictScope，一个自动生成价值冲突场景的 pipeline，通过开放式评估（非选择题）揭示 LLM 在冲突情境下的价值优先级排序，发现模型在开放式设置中从保护性价值（如无害性）转向个人价值（如用户自主性），且系统提示可将目标排序对齐提升 14%。
tags:
  - ICLR 2026
  - LLM/NLP
  - 价值冲突
  - LLM 对齐
  - 价值排序
  - 开放式评估
  - 可操纵性
  - AI 安全
---

# Generative Value Conflicts Reveal LLM Priorities

**会议**: ICLR 2026  
**arXiv**: [2509.25369](https://arxiv.org/abs/2509.25369)  
**代码**: [GitHub](https://github.com/andyjliu/conflictscope)  
**领域**: llm_nlp  
**关键词**: 价值冲突, LLM 对齐, 价值排序, 开放式评估, 可操纵性, AI 安全

## 一句话总结

提出 ConflictScope，一个自动生成价值冲突场景的 pipeline，通过开放式评估（非选择题）揭示 LLM 在冲突情境下的价值优先级排序，发现模型在开放式设置中从保护性价值（如无害性）转向个人价值（如用户自主性），且系统提示可将目标排序对齐提升 14%。

## 研究背景与动机

LLM 对齐研究通常追求让模型同时满足多个价值（如有用性、诚实性、无害性），但在实际部署中，这些价值**经常冲突**。了解模型如何在价值冲突中排列优先级，对预测其行为至关重要。

现有对齐数据集**很少引发价值冲突**：Buyl et al. (2025) 发现 HH-RLHF 和 PKU-SafeRLHF 中约 85% 的响应对不引发 Anthropic 宪法原则间的分歧。这使得研究特定价值对之间的冲突极为困难。

现有道德困境研究的两个**生态效度问题**：

**旁观者视角**：通常将 LLM 作为场景的第三方观察者，而非可以主动影响结果的道德主体

**选择题评估**：对评估设置的微小差异高度敏感，缺乏泛化性

核心目标：设计一个可以**自动生成特定价值对之间冲突场景**、并通过**开放式交互评估**模型行为的 pipeline。

## 方法详解

### 整体框架

ConflictScope 的流程：

1. **给定价值集合** → 自动生成冲突场景
2. **选择题评估** → 获取表达偏好（expressed preferences）
3. **开放式评估**（模拟用户交互）→ 获取显示偏好（revealed preferences）
4. **Bradley-Terry 模型拟合** → 从成对比较中推导价值排序

### 关键设计

**场景生成 Pipeline**（两阶段）：

**第一阶段**：给 Claude 3.5 Sonnet 提供两个价值描述和部署环境，生成冲突场景摘要。使用 4 种提示模板（轻微好处/强烈好处/轻微伤害/强烈伤害），缓解模型的不作为偏好并模拟现实严重程度分布。

**第二阶段**：去重（embedding 余弦相似度 ≥ 0.8 时去除），然后对每个场景生成详细描述、用户画像和两个行动选项。

**场景过滤**（6 维度 LLM-as-Judge）：
1. 场景现实性：是否可在现实中发生
2. 场景具体性：是否充分细化
3. 行动可行性：纯文本 LLM 能否执行两个动作
4. 场景不可能性：两个行动是否不可能同时执行
5. 行动价值引导性：每个价值是否推荐不同行动
6. 真正困境：是否存在明显更好的共识行动

过滤使用 GPT-4.1 作为 judge，通过人工研究验证所有维度精度高。

**开放式评估**：
- **User LLM**（GPT-4.1）扮演用户，根据场景和用户画像输出用户提示
- **Target Model** 在无场景上下文的情况下回复
- **Judge LLM**（GPT-4.1）判断模型行为更接近哪个行动

Judge 模型与人工标注的 Cohen's Kappa = 0.62（强一致性）。

**价值排序推导**：

对所有场景级成对比较拟合 **Bradley-Terry 偏好模型**：

$$P(\text{value } i \succ \text{value } j) = \frac{e^{\beta_i}}{e^{\beta_i} + e^{\beta_j}}$$

得到价值集合上的全排序。

### 可操纵性评估

定义对齐度 $a(R, R_t)$ 为模型选择与目标排序中更高价值一致选项的比例。操纵效果 = $\frac{a(R_s, R_t) - a(R_d, R_t)}{1 - a(R_d, R_t)}$。通过在系统提示中加入详细价值排序描述来操纵模型行为。

### 三个价值集合

1. **HHH**：有用性、诚实性、无害性
2. **Personal-Protective**：个人价值（自主性、真实性、创造力、赋权）vs 保护性价值（责任、无害性、合规、隐私）
3. **ModelSpec**：基于 OpenAI Model Spec 的原则子集，涵盖三个优先级层

## 实验关键数据

### 主实验

**RQ1：ConflictScope 生成的场景比现有数据集更具挑战性**

| 数据集 | 观察一致度 ↓ | Likert 差异率 ↑ |
|--------|------------|---------------|
| HH-RLHF | ~0.65 | ~0.35 |
| PKU-SafeRLHF | ~0.70 | ~0.40 |
| MoralChoice-HighAmbiguity | ~0.85 | ~0.90 |
| DailyDilemmas | ~0.85 | ~0.75 |
| **ConflictScope-HHH** | **0.786** | **0.801** |
| **ConflictScope-ModelSpec** | — | — |
| **ConflictScope-Personal-Protective** | — | — |

ConflictScope 在两个指标上**帕累托最优**：引发更多模型间分歧且模型有更强偏好。

**RQ2：表达偏好 vs 显示偏好的巨大差异**

Personal-Protective 价值集合上：
- **选择题**中保护性价值平均排名 1.7（高优先级）
- **开放式**中保护性价值平均排名 4.5（低优先级）

所有模型（除 Claude）在开放式评估中**大幅从保护性价值转向个人价值**。HHH 中同样：选择题偏好无害性 > 有用性，开放式偏好有用性 > 无害性。

**RQ3：系统提示操纵有效但有限**

| 模型 | HHH 效果 | ModelSpec 效果 | Personal-Protective 效果 |
|------|---------|--------------|------------------------|
| GPT-4o | 正向 | 正向 | 正向 |
| Claude 3.5 Sonnet | 小正向 | 小正向 | 小正向 |
| Llama 3.1 70B | 正向 | 正向 | 正向 |
| OLMo-2-32B | **0.27** | — | — |
| Claude Haiku 3.5 | **0.01** | — | — |

平均归一化效果量 **0.145**（14.5% 的未对齐案例被成功操纵），仅 1/14 模型在任何价值集上出现显著负向效果。

### 消融实验

| Pipeline 变体 | 观察一致度 ↓ | Likert 差异率 ↑ |
|--------------|------------|---------------|
| Full (ConflictScope) | **0.786** | 0.801 |
| Unfiltered | 0.824 | 0.818 |
| Single-stage | 0.898 | 0.854 |
| Direct | 0.852 | 0.830 |

过滤使一致度改善 3.8%（更多分歧）；两阶段生成比单阶段降低 7.4%（更具挑战性）。

### 关键发现

1. **选择题 ≠ 真实行为**：模型在选择题中"说"重视无害性，在开放式中"做"时却优先有用性
2. **Claude 是例外**：两种评估设置间最一致，可能与其对齐训练有关
3. **隐私和真实性更稳定**：在表达/显示偏好切换中受影响最小
4. **可操纵性变化大**：OLMo-2-32B 效果量 0.27 vs Claude Haiku 0.01
5. **领域分析**：模型在不同应用领域中的偏好总体一致（附录 H）

## 亮点与洞察

1. **揭示了 LLM 对齐的"说做不一"现象**：选择题偏好与实际行为的系统性差异是核心贡献
2. **Pipeline 设计完整**：从场景生成到过滤到评估的全自动化流程，可直接应用于任意价值集合
3. **Bradley-Terry 模型的巧妙应用**：将场景级成对比较聚合为全局排序
4. **6 维过滤设计严谨**：确保场景既是真正困境又具部署现实性
5. **14 个目标模型的广泛评估**：涵盖 OpenAI/Anthropic/Meta/Google/Allen AI 等多家

## 局限与展望

1. **单轮交互**：仅模拟单次用户-LLM 交互，多轮对话可能揭示更多
2. **LLM 模拟用户**：模拟用户是否能代表真实人类行为仍是开放问题
3. **价值排序是粗粒度的**：Bradley-Terry 假设全局线性排序，但现实中偏好可能是上下文相关的
4. **场景生成依赖 LLM**：可能继承生成模型的偏见
5. 可扩展到 agent 环境、多轮对话和更复杂的价值网络
6. 可结合 Item Response Theory 进行更高效的评估

## 相关工作与启发

- **Anthropic Constitutional AI** (Bai et al., 2022b)：ConflictScope 直接评估宪法原则间的冲突
- **DailyDilemmas** (Chiu et al., 2025a)：日常道德困境数据集，但挑战性低于 ConflictScope
- **CLASH** (Lee et al., 2025)：价值冲突数据集之一
- 启发："说做不一"现象提醒我们：现有基于选择题的对齐评估可能严重高估了保护性行为

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 开放式价值冲突评估 + 表达 vs 显示偏好分析是新颖视角
- **技术深度**: ⭐⭐⭐⭐ — Pipeline 设计完整，统计分析严谨
- **实验充分性**: ⭐⭐⭐⭐ — 14 模型 × 3 价值集 × 2 评估模式 + 消融 + 可操纵性
- **写作质量**: ⭐⭐⭐⭐⭐ — 研究问题清晰，结构优秀
- **实用价值**: ⭐⭐⭐⭐ — 为 LLM 对齐评估提供新工具和新视角
- **综合推荐**: ⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

## 相关论文

- [ConflictScope: Generative Value Conflicts Reveal LLM Priorities](quamo_quaternion_motions_for_vision-based_3d_human_kinematics_capture.md)
- [How Catastrophic is Your LLM? Certifying Risk in Conversation](how_catastrophic_is_your_llm_certifying_risk_in_conversation.md)
- [From Assumptions to Actions: Turning LLM Reasoning into Uncertainty-Aware Planning](from_assumptions_to_actions_turning_llm_reasoning_into_uncertainty-aware_plannin.md)
- [PT2-LLM: Post-Training Ternarization for Large Language Models](pt2-llm_post-training_ternarization_for_large_language_models.md)
- [The Path of Least Resistance: Guiding LLM Reasoning Trajectories for Efficient Consistency](the_path_of_least_resistance_guiding_llm_reasoning_trajectories_for_efficient_co.md)

<!-- RELATED:END -->
