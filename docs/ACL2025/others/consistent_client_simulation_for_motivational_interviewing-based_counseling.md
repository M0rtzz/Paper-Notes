---
title: >-
  [论文解读] Consistent Client Simulation for Motivational Interviewing-based Counseling
description: >-
  [ACL 2025][客户模拟] 提出一种面向动机性访谈（MI）心理咨询的一致性客户模拟框架，通过状态转换、行动选择、信息选择和回复生成四个模块，确保模拟客户的行为与其预设的画像（动机、信念、改变计划、配合度）保持一致，在自动和专家评估中均优于基线方法。 领域现状：心理咨询师的培训传统上需要人类客户参与。为降低成本…
tags:
  - "ACL 2025"
  - "客户模拟"
  - "动机性访谈"
  - "一致性"
  - "状态跟踪"
  - "行为选择"
  - "LLM Agent"
---

# Consistent Client Simulation for Motivational Interviewing-based Counseling

**会议**: ACL 2025  
**arXiv**: [2502.02802](https://arxiv.org/abs/2502.02802)  
**代码**: 无  
**作者**: Yizhe Yang, Palakorn Achananuparp, Heyan Huang, Jing Jiang, John Pinto, Jenny Giam, Kit Phey Leng, Nicholas Gabriel Lim, Cameron Tan Shi Ern, Ee-peng Lim  
**机构**: Beijing Institute of Technology, Singapore Management University, Australian National University 等  
**领域**: 其他  
**关键词**: 客户模拟, 动机性访谈, 一致性, 状态跟踪, 行为选择, LLM Agent

## 一句话总结

提出一种面向动机性访谈（MI）心理咨询的一致性客户模拟框架，通过状态转换、行动选择、信息选择和回复生成四个模块，确保模拟客户的行为与其预设的画像（动机、信念、改变计划、配合度）保持一致，在自动和专家评估中均优于基线方法。

## 研究背景与动机

**领域现状**：心理咨询师的培训传统上需要人类客户参与。为降低成本，研究者开始用 LLM 模拟客户 agent，但已有方法主要关注咨询师 agent 的评估，忽视了客户模拟的一致性质量。

**现有方法的不足**：
   - 简单画像 prompt（Yosef et al., 2024; Wang et al., 2024a）或会话示例 prompt（Chiu et al., 2024）导致四种不一致：
     - **(a) 动机不一致**：客户同意改变的理由与预设动机不符
     - **(b) 信念不一致**：客户未遵守给定的信念
     - **(c) 计划不一致**：客户接受了画像中不存在的改变计划
     - **(d) 配合度不一致**：客户对咨询师的配合程度与设定不符
   - LLM（如 ChatGPT）天然倾向生成顺从回复（alignment 的副作用），导致模拟客户过度配合（overly-compliant），行为单一

**核心动机**：需要精细控制模拟客户在心理咨询不同阶段的状态转换和行为选择，使其画像和行为与真实客户高度一致。

## 方法详解

### 整体框架

框架包含四个核心模块，在每一轮对话中按顺序执行：

1. **状态转换模块**（State Transition）→ 确定客户下一状态
2. **行动选择模块**（Action Selection）→ 选择客户行为类型
3. **信息选择模块**（Information Selection）→ 选择要透露的画像信息
4. **回复生成模块**（Response Generation）→ 生成客户话语

### 客户画像定义

客户画像包含以下要素：

| 要素 | 说明 |
|------|------|
| 行为问题 | 客户要解决的行为问题 |
| 初始/终止状态 | 咨询前后的心理状态 |
| 人设（Persona） | 客户背景信息 |
| 动机（Motivation） | 促使改变的具体原因 |
| 信念（Beliefs） | 阻碍改变的信念 |
| 改变计划（Plans） | 客户可能同意的具体行为改变 |
| 配合度（Receptivity）| 对咨询师的配合程度，1-5 分 |

### 关键设计一：状态转换

基于跨理论模型（Transtheoretical Model），定义三个核心状态 + 终止：

- **前沉思（Precontemplation）**→ 否认问题。转入沉思的条件：咨询师提到了客户画像中的动机
- **沉思（Contemplation）**→ 认识到问题但犹豫。转入准备的条件：画像中的信念障碍被充分回应
- **准备（Preparation）**→ 开始计划改变。转入终止的条件：首选改变计划已与咨询师讨论
- **终止（Termination）**→ 结束会话

状态转换严格依赖于咨询师是否触及了客户画像中的关键内容，确保一致性。

### 关键设计二：行动选择

结合两种行动分布的融合策略：

1. **上下文感知分布**：基于当前对话上下文，用 LLM 推理合适的行动分布
2. **(状态, 配合度)感知分布**：从真实 AnnoMI 数据集中学习每个(状态, 配合度)组合下的行动分布

最终行动分布 = 两者取平均，然后在与当前状态相关的候选行动中采样。

### 关键设计三：信息选择

行动分为两类：
- **Type-1**（如 Deny, Engage, Accept）：不需要画像信息即可生成
- **Type-2**（如 Inform, Blame, Hesitate, Plan）：需要从画像中选择相关信息

信息选择模块防止客户一次性泄露过多画像信息，避免不切实际地缩短咨询会话。

### 数据标注

基于 AnnoMI 数据集（真实 MI 咨询对话），选取 86 个客户和会话：
- 使用 GPT-4 标注客户画像四要素、状态、行动和配合度
- 标注质量验证：状态 87.31%、行动 85.20%、配合度 80.32% 准确率
- 所有画像条目均确认事实准确

## 实验关键数据

### 实验设置

- LLM 骨干模型：gpt-3.5-turbo-0125
- 咨询师 agent：基于 MI 知识 prompt 的 LLM agent
- 主持人 agent：监控对话终止条件
- 每个客户画像生成 3 个会话 → 共 258 个生成会话

### 基线方法

1. **Base**：仅含行为问题的简单 prompt
2. **Example-based**（Chiu et al., 2024）：提供真实会话示例
3. **Profile-based**（Yosef et al., 2024）：提供客户画像
4. **Pro+Act-based**（Zhang et al., 2024）：画像 + 行动描述

### 画像一致性评估（自动评估，GPT-4 蕴含判定）

| 方法 | 人设↑ | 动机↑ | 信念↑ | 计划↑ | 配合度↑ |
|------|-------|-------|-------|-------|---------|
| Base | 9.01 | 16.17 | 12.15 | 9.30 | −0.31 |
| Example-based | 53.68 | 45.73 | 45.55 | 33.53 | 0.25 |
| Profile-based | 61.97 | 53.44 | 67.17 | 54.67 | 0.31 |
| Pro+Act-based | 67.09 | 55.33 | 68.60 | 57.17 | 0.33 |
| **Ours** | **70.57** | **73.37** | **71.70** | **68.51** | **0.58** |

本方法在所有五个一致性维度上均取得最优结果。

### 会话行为分析

| 方法 | 平均配合度 | 20轮动机率 | 平均动机步数 | 行动KL↓ |
|------|-----------|-----------|-------------|---------|
| Base | 4.42±0.47 | 1.00 | 6.60 | 0.39 |
| Profile-based | 4.12±0.64 | 0.96 | 9.76 | 0.15 |
| Pro+Act-based | 3.86±1.01 | 0.94 | 9.93 | 0.13 |
| **Ours** | **3.32±1.15** | **0.69** | **18.60** | **0.06** |
| 真实客户 | 3.27±1.12 | 0.48 | 27.56 | 0.00 |

本方法的行为分布（配合度、动机率、行动 KL 散度）最接近真实客户。

### 专家评估（1-5分，6个客户 × 4个画像维度 × 3位标注者）

| 方法 | 人设 | 信念 | 动机 | 计划 | 真实感 |
|------|------|------|------|------|--------|
| Profile-based | 2.61 | 2.00 | 2.61 | 1.56 | 2.38 |
| Pro+Act-based | 2.65 | 2.22 | 2.78 | 1.56 | 2.50 |
| **Ours** | **3.33** | **2.89** | **3.00** | **2.27** | **3.16** |
| 真实客户 | 4.72 | 4.67 | 4.56 | 4.61 | 4.72 |

### 配合度控制评估

| 方法 | 配合度=1 | 配合度=3 | 配合度=5 | 相关系数 |
|------|---------|---------|---------|---------|
| Pro+Act-based | 5.0 | 4.3 | 5.0 | 0.00 |
| **Ours** | **1.3** | **3.0** | **4.3** | **0.86** |

本方法能有效控制不同配合度级别的客户行为（Spearman 相关 0.86, p=0.0003），而基线几乎无法区分配合度设置。

## 亮点与洞察

1. **首次系统定义**心理咨询客户模拟的一致性维度（动机、信念、计划、配合度）
2. **状态转换设计精妙**：将 MI 的跨理论模型嵌入状态机，状态切换与画像内容直接绑定
3. **配合度可控**是重要的实用特性——训练咨询师需要面对不同难度的客户
4. **行动分布融合**（上下文感知 + 数据驱动）兼顾了对话流畅性和画像一致性
5. **发现基线方法的过度顺从问题**，揭示了 LLM alignment 在角色扮演场景中的副作用

## 局限性

1. **多步 prompt 的误差累积**：框架依赖多次 LLM prompt，对 prompt 敏感度较高
2. **仅关注客户模拟**，未涉及咨询师 agent 的优化和评估
3. **仅在 AnnoMI 数据集上实验**，且限于 MI 一种咨询方法
4. **模拟客户与真实客户仍有较大差距**（专家评分 3.16 vs 4.72）
5. 会话长度偏短——基线方法过度配合导致快速结束，本方法因严格状态控制导致咨询师提前放弃

## 相关工作

- **动机性访谈**：Miller & Rollnick (2012), Transtheoretical Model (Prochaska & Velicer, 1997)
- **客户 Agent 模拟**：PATIENT-Ψ (Wang et al., 2024b), State-Aware Patient Simulator (Liao et al., 2024)
- **LLM 对话**：Chiu et al. (2024) 计算框架评估 LLM 咨询师

## 评分

⭐⭐⭐⭐ (4/5)

在心理咨询客户模拟这一细分方向上做出了有深度的工作，一致性的多维度定义很有价值，配合度可控是亮点。评估充分（自动 + 专家），但实验限于单一数据集和单一咨询方法，且仍未弥合模拟与真实客户的差距。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Hanging in the Balance: Pivotal Moments in Crisis Counseling Conversations](hanging_in_the_balance_pivotal_moments_in_crisis_counseling_conversations.md)
- [\[ACL 2025\] GA-S3: Comprehensive Social Network Simulation with Group Agents](ga-s3_comprehensive_social_network_simulation_with_group_agents.md)
- [\[ACL 2025\] QG-SMS: Enhancing Test Item Analysis via Student Modeling and Simulation](qg-sms_enhancing_test_item_analysis_via_student_modeling_and_simulation.md)
- [\[ACL 2025\] Are Any-to-Any Models More Consistent Across Modality Transfers Than Specialists?](are_any-to-any_models_more_consistent_across_modality_transfers_than_specialists.md)
- [\[ACL 2025\] CORAL: Learning Consistent Representations across Multi-step Training with Lighter Speculative Drafter](coral_speculative_drafting.md)

</div>

<!-- RELATED:END -->
