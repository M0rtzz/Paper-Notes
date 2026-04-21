---
title: >-
  [论文解读] Investigating Non-Transitivity in LLM-as-a-Judge
description: >-
  [ICML 2025][LLM评估] 揭示了 LLM-as-a-Judge 框架中评判偏好的**非传递性**问题（A>B, B>C 不能推出 A>C），证明固定基线模型的排名方式不可靠，提出基于循环赛 + Bradley-Terry 模型的排名方法及高效的 Swim 锦标赛策略。
tags:
  - ICML 2025
  - LLM评估
  - 非传递性
  - Bradley-Terry模型
  - 锦标赛排名
  - 位置偏差
---

# Investigating Non-Transitivity in LLM-as-a-Judge

**会议**: ICML 2025  
**arXiv**: [2502.14074](https://arxiv.org/abs/2502.14074)  
**代码**: [yix8/llm-nontransitivity](https://github.com/yix8/llm-nontransitivity)  
**领域**: LLM/NLP  
**关键词**: LLM评估, 非传递性, Bradley-Terry模型, 锦标赛排名, 位置偏差

## 一句话总结

揭示了 LLM-as-a-Judge 框架中评判偏好的**非传递性**问题（A>B, B>C 不能推出 A>C），证明固定基线模型的排名方式不可靠，提出基于循环赛 + Bradley-Terry 模型的排名方法及高效的 Swim 锦标赛策略。

## 研究背景与动机

- **LLM-as-a-Judge 范式的普及**：以 AlpacaEval、VicunaEval、Arena-Hard 为代表的自动评估框架，采用 LLM 对目标模型与固定基线模型进行成对比较，通过胜率来排名模型。
- **隐含的传递性假设**：这些框架暗含一个关键假设——如果 Judge 偏好 A 胜过 B，B 胜过 C，则必然偏好 A 胜过 C。然而，该假设从未被系统性验证。
- **若传递性不成立**：当存在"石头-剪刀-布"式的循环偏好时，排名会高度依赖基线模型的选择，导致不同基线产生矛盾的排名结果。
- **核心问题**：LLM Judge 的偏好到底有多大程度的非传递性？这种非传递性如何影响排名的可靠性？如何设计更鲁棒的排名方法？

## 方法详解

### 整体框架

本文的方法可分为三大部分：（1）度量非传递性的指标设计；（2）基于锦标赛的排名方法；（3）高效的 Swim 匹配策略。

**评估设置**：在 AlpacaEval 数据集上，对 20 个同时出现在 AlpacaEval 和 Chatbot Arena 排行榜的模型进行评估。使用 GPT-4-Turbo 和 GPT-3.5-Turbo 作为 Judge 模型，温度设为 0。采用位置交换（position switching）策略缓解位置偏差，每种位置配置调用两次以减少 API 随机性。

### 关键设计

#### 1. 非传递性度量指标

**硬非传递性比例 (PNT)**：给定模型三元组 $(m_A, m_B, m_C)$，在指令集 $\mathcal{I}$ 上计算违反传递性的指令比例：

$$\text{PNT} = \frac{1}{|\mathcal{I}|} \sum_{I_i \in \mathcal{I}} \mathbb{1}_{\text{non-trans.}}(m_A, m_B, m_C \mid m_J, I_i)$$

**局限性**：PNT 是二值指标，无法区分"严重违反"和"轻微违反"。例如 $J(A>C)=0$ 和 $J(A>C)=0.49$ 都被同等视为非传递，但后者明显更接近传递。

**软非传递性偏差 (SNTD)**：利用 Jensen-Shannon 散度衡量观测胜率 $\phi$ 与传递性假设下的估计胜率 $\hat{\phi}$ 的差异：

$$\text{SNTD}(m_A, m_B, m_C | I_i) = \frac{1}{3} \times \mathbb{E}\left[\sum_{\text{三个配对}} \text{JSD}(\phi \| \hat{\phi})\right]$$

**估计胜率的推导**：基于 Bradley-Terry 模型，利用两对比较的观测胜率推算第三对在传递性假设下的期望胜率：

$$\hat{\phi}(o_A, o_B \mid m_J, I_i) = \frac{1}{1 + e^{-(s_{AC} - s_{BC})}}$$

其中 $s_{AB} = \ln\frac{\phi_{AB}}{1-\phi_{AB}}$ 是从观测胜率反推的质量差。

#### 2. 四种场景分类

根据模型间性能差距，将三元组分为四类：

| 场景 | 关系 | 含义 |
|------|------|------|
| LL (Lead & Lead) | $m_A \gg m_B \gg m_C$ | 三者差距均大 |
| LM (Lead & Margin) | $m_A \gg m_B \approx m_C$ | 前者领先，后两者接近 |
| ML (Margin & Lead) | $m_A \approx m_B \gg m_C$ | 前两者接近，后者落后 |
| MM (Margin & Margin) | $m_A \approx m_B \approx m_C$ | 三者水平相近 |

#### 3. 锦标赛排名方法

**循环赛 (Round-Robin)**：让所有模型两两进行成对比较，构建完整的胜率矩阵，然后使用 Bradley-Terry 模型拟合每个模型的强度系数 $\beta_i$：

$$\hat{\boldsymbol{\beta}} = \arg\max_{\boldsymbol{\beta}} \sum_i \sum_{j \neq i} \left[ W_{i,j} \cdot \ln\frac{1}{1+e^{(\beta_j - \beta_i)}} \right]$$

关键创新：使用**软标签**（连续胜率）而非硬标签（0/1），$W_{i,j} = \sum_{I_k} J(m_i \succ m_j \mid I_k)$，产生更精确的估计。最终将 BT 系数转换为 Elo 评分：$\xi_i = 400 \log_{10} \beta_i$。

**Swim 锦标赛 (Swiss-Wise Iterative Matchmaking)**：循环赛的计算复杂度为 $\mathcal{O}(nm^2)$，Swim 借鉴二分搜索和瑞士制锦标赛思想，动态调整匹配对手，将新模型加入排行榜的比较次数从 $M$ 降至 $\lceil \log_2(M) \rceil$，总复杂度降为 $\mathcal{O}(nm\log m)$。

### 损失函数 / 训练策略

本文不涉及模型训练，核心优化目标是 Bradley-Terry 模型的最大似然估计。此外采用了以下策略：

- **位置交换 (Position Switching)**：每次比较交换两个输出的位置，取均值作为最终偏好得分，缓解位置偏差
- **长度控制 (Length-Controlled)**：采用与 LC-AlpacaEval 相同的广义线性模型权重，消除冗长回答的偏差
- **多种 Prompt 策略探索**：测试了评估清单 (Checklist)、思维链 (CoT)、允许平局等策略对非传递性的影响

## 实验关键数据

### 主实验

| 方法 | Spearman (无LC) | Spearman (LC) | Kendall (无LC) | Kendall (LC) |
|------|:---:|:---:|:---:|:---:|
| AlpacaEval 2.0 | 81.4% | 95.0% | 63.2% | 82.1% |
| Round-Robin (本文) | 85.4% | **96.4%** | 68.4% | **86.3%** |
| 提升 | +4.0% | +1.4% | +5.2% | +4.2% |

与 Chatbot Arena 人类排名的相关性全面提升，证明循环赛排名更好地对齐人类偏好。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|------|------|
| GPT-4-Turbo Judge (MM场景) | PNT=8.45%, SNTD=0.143 | 模型水平相近时非传递性最高 |
| GPT-3.5-Turbo Judge (MM场景) | PNT=20.87%, SNTD=0.263 | 弱Judge非传递性显著更高 |
| GPT-4-Turbo + 位置交换 | 非传递性降低 17%-44% | 对强Judge有效 |
| GPT-3.5-Turbo + 位置交换 | 非传递性反而略增 | 对弱Judge可能反效果 |
| CoT Prompting | 位置偏差↓, 非传递性↑ | CoT缓解位置偏差但引入新的非传递性 |
| 允许平局 | 位置偏差↑, 非传递性↑ | 两者都变差 |
| Checklist Prompting | 非传递性略微↓ | 效果有限 |
| Swim vs Round-Robin | 几乎相同的相关性 | 仅需 log₂(M) 次比较 |

### 关键发现

1. **非传递性普遍存在**：在所有四种场景中，GPT-4-Turbo 和 GPT-3.5-Turbo 作为 Judge 时都表现出非传递性偏好。
2. **模型越接近，非传递性越强**：PNT 和 SNTD 在模型性能差距缩小时显著提升，在 MM 场景达到峰值。
3. **弱 Judge 更不可靠**：GPT-3.5-Turbo 的非传递性指标全面高于 GPT-4-Turbo，且跨场景几乎不变（~20% PNT），说明其评估主要由偏差驱动。
4. **基线选择高度敏感**：20 个模型中仅 20% 在所有基线下保持相同排名，任意两个排名列表平均仅 61% 的模型位置一致。
5. **非传递性由双因素共同驱动**：位置偏差是重要因素，但即使在无位置偏差的"一致性指令"中仍存在非传递性，说明 Judge 内在的推理能力也是原因。
6. **模型级硬非传递性温和**：经过位置交换后，模型级别的硬非传递性很少发生，但软非传递性仍然导致排名不一致。

## 亮点与洞察

- **问题意义重大**：首次系统性研究 LLM-as-a-Judge 中的非传递性问题，揭示了当前自动评估框架的根本缺陷。
- **SNTD 指标设计精巧**：利用 Bradley-Terry 模型构建传递性假设下的期望胜率，再用 JSD 衡量偏差，既有理论基础又具可操作性。
- **从博弈论视角解决评估问题**：将 LLM 评估类比为非传递博弈（如星际争霸、Dota 2），借鉴多智能体强化学习中的种群评估思想。
- **Swim 算法实用性强**：将新模型加入排行榜的成本从线性降至对数级，使得锦标赛方法在大规模排行榜上可行。
- **位置交换+软标签的组合**：软标签（连续胜率）比硬标签（0/1）在 BT 模型估计中更准确，位置交换对强 Judge 有效但对弱 Judge 可能适得其反。

## 局限与展望

1. **数据集覆盖有限**：仅在 AlpacaEval 数据集上验证，可能不完全代表真实的开放式任务分布。
2. **Judge 模型范围窄**：只测试了 GPT-4-Turbo 和 GPT-3.5-Turbo，未扩展到 Claude、Gemini 等其他 Judge。
3. **参考标准本身有偏差**：依赖 Chatbot Arena 人类排名作为金标准，但人类评估本身也可能存在非传递性。
4. **仅关注成对比较**：未探索逐点评分方法中的非传递性，以及逐点转成对时是否引入新的循环偏好。
5. **Bradley-Terry 模型的局限**：BT 模型假设每个模型有一个全局标量强度，无法捕捉多维能力差异，更具表达力的多维模型值得探索。

## 相关工作与启发

- **AlpacaEval / Arena-Hard**：当前最流行的固定基线比较框架，本文直接揭示了其方法论弱点。
- **Chatbot Arena**：通过真人投票进行 Elo 排名，是本文方法对齐的目标，但成本高昂。
- **Balduzzi et al. (2019) / Czarnecki et al. (2020)**：多智能体博弈中非传递性的理论刻画，发现游戏策略空间呈"旋转陀螺"分布。本文将此洞察引入 NLP 评估。
- **启发**：对于未来 LLM 评估框架设计，应避免依赖单一基线，至少采用多基线或锦标赛机制；Swim 策略可直接集成到现有排行榜降低成本。

## 评分

| 维度 | 分数 | 理由 |
|------|:---:|------|
| 新颖性 | ⭐⭐⭐⭐ | 首次系统研究 LLM Judge 非传递性，问题视角新颖 |
| 理论深度 | ⭐⭐⭐⭐ | SNTD 指标有 BT 模型理论支撑，分析透彻 |
| 实验充分性 | ⭐⭐⭐⭐ | 20模型全配对实验，多场景多Judge，消融丰富 |
| 实用价值 | ⭐⭐⭐⭐⭐ | Swim 算法可直接提升现有排行榜可靠性 |
| 写作质量 | ⭐⭐⭐⭐ | 逻辑清晰，图表丰富，动机阐述充分 |
| 综合 | ⭐⭐⭐⭐ | 对 LLM 评估社区有重要指导意义的扎实工作 |

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [Bridging Human and LLM Judgments: Understanding and Narrowing the Gap](../../NeurIPS2025/dialogue/bridging_human_and_llm_judgments_understanding_and_narrowing_the_gap.md)
- [Non-Collaborative User Simulators for Tool Agents](../../ICLR2026/dialogue/non-collaborative_user_simulators_for_tool_agents.md)
- [HyGen: Efficient LLM Serving via Elastic Online-Offline Request Co-location](../../NeurIPS2025/dialogue/hygen_efficient_llm_serving_via_elastic_online-offline_request_co-location.md)
- [SciArena: An Open Evaluation Platform for Non-Verifiable Scientific Literature-Grounded Tasks](../../NeurIPS2025/dialogue/sciarena_an_open_evaluation_platform_for_non-verifiable_scientific_literature-gr.md)
- [Position: Uncertainty Quantification Needs Reassessment for Large-language Model Agents](position_uncertainty_quantification_needs_reassessment_for_large-language_model_.md)

<!-- RELATED:END -->
