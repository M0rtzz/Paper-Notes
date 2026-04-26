---
title: >-
  [论文解读] AndroidGen: Building an Android Language Agent under Data Scarcity
description: >-
  [LLM Agent] 提出 AndroidGen 框架，通过经验检索（ExpSearch）、反思规划（ReflectPlan）、自动校验（AutoCheck）和步骤级评判（StepCritic）四个模块，在高质量训练数据稀缺的条件下增强LLM的Android操作能力，并通过自动生成轨迹数据训练出无需人工标注的开源移动端agent。
tags:
  - LLM Agent
---

# AndroidGen: Building an Android Language Agent under Data Scarcity

| 信息 | 内容 |
|------|------|
| 会议 | ACL 2025 |
| arXiv | [2504.19298](https://arxiv.org/abs/2504.19298) |
| 代码 | [GitHub](https://github.com/THUDM/AndroidGen) |
| 领域 | LLM Agent / 移动端智能体 |
| 关键词 | Android agent, 数据稀缺, 轨迹生成, 自动评估, 开源agent |

## 一句话总结

提出 AndroidGen 框架，通过经验检索（ExpSearch）、反思规划（ReflectPlan）、自动校验（AutoCheck）和步骤级评判（StepCritic）四个模块，在高质量训练数据稀缺的条件下增强LLM的Android操作能力，并通过自动生成轨迹数据训练出无需人工标注的开源移动端agent。

## 研究背景与动机

- **问题定义**：将LLM作为agent在真实移动设备上完成用户任务（如设置闹钟、发送消息、搜索地图等）是重要但尚未充分实现的目标。核心瓶颈在于**高质量轨迹数据的稀缺**。
- **数据收集困难**：(1) **场景多样性**——不同应用差异巨大，需要广泛覆盖；(2) **复杂任务标注成本高**——多步骤任务需要精确执行和规划；(3) **数据质量控制难**——验证每步操作是否完全符合任务要求既费时又费力。
- **现有不足**：人工标注耗时耗钱，自动化方法（使用GPT-4等自动完成任务）的成功率过低（如M3A+GPT-4o在AndroidWorld仅27.7%），且缺乏有效的自动质量筛选策略。
- **核心动机**：需要一个既能提升agent性能的推理框架，又能自动生成高质量训练数据、训练开源模型的完整pipeline。

## 方法详解

### 整体框架

AndroidGen包含三个阶段：预备阶段（Preliminary）→ 任务执行（Task Execution）→ 更新（Update），核心为四个模块：

### 关键设计

1. **ExpSearch（经验检索）**：
    - 利用LLM的in-context learning能力，从历史轨迹数据库中检索最相似的已完成任务作为示例
    - 使用Contriever编码指令计算相似度，选top-1结果
    - 每次完成任务后用StepCritic评估并更新数据库，实现**迭代自我改进**和**从易到难泛化**

2. **ReflectPlan（反思规划）**：
    - 首步：分析任务和环境生成step-by-step计划
    - 后续步：反思当前进度，更新计划状态，遇到失败或循环时动态修正计划
    - 解决了传统planning对执行结果过于乐观的问题

3. **AutoCheck（自动校验）**：
    - 每步操作前主动验证有效性（元素ID是否存在、类型是否合规、滚动是否完成等）
    - 采用规则策略而非LLM自检，避免自检标准不一致导致的误报
    - 检测到问题时终止执行并在下一轮反馈

4. **StepCritic（步骤级评判）**：
    - 将任务分解为子目标，基于完整操作序列和设备最终状态进行细粒度评估
    - 每个子目标标注是否完成及对应步骤（-1表示未完成）
    - 支持轨迹增强：部分完成的轨迹可按已完成子目标截断为多条有效训练数据

### 损失函数

使用标准语言模型损失进行LoRA微调。将规划步和执行步混合训练，使模型同时具备规划和执行能力。

## 实验关键数据

### 主实验：AndroidWorld成功率

| Agent | 模型 | 平均成功率 |
|-------|------|-----------|
| SeeAct | GPT-4o | 15.9% |
| M3A | GPT-4o | 27.7% |
| AndroidGen | GLM-4-9B* | 29.2% |
| AndroidGen | Llama-3-70B* | 35.3% |
| **AndroidGen** | **GPT-4o** | **46.8%** |

### AitW基准对比

| 方法 | General | Web Shopping |
|------|---------|-------------|
| AppAgent (GPT-4o) | 16.7 | 8.3 |
| DigiRL (RL训练)* | 71.9 | 67.2 |
| AndroidGen (GLM-4-9B*) | 65.6 | 59.4 |
| AndroidGen (Llama-3-70B*) | 74.0 | 79.2 |
| **AndroidGen (GPT-4o)** | **85.4** | **81.3** |

### 消融实验

| 方法 | Easy | Medium | Hard | 平均 |
|------|------|--------|------|-----|
| Base Agent | 35.0 | 5.9 | 0.0 | 20.7 |
| +ReflectPlan | 51.7 | 14.7 | 0.0 | 32.4 |
| +AutoCheck | 53.3 | 17.6 | 0.0 | 34.2 |
| +ExpSearch | **65.0** | **32.4** | **11.8** | **46.8** |

### 关键发现

1. **AndroidGen大幅超越基线**：在AndroidWorld上GPT-4o版本达46.8%（vs SeeAct 15.9%，M3A 27.7%）
2. **ExpSearch贡献最大**：从34.2%→46.8%，是唯一能解决Hard任务的模块（11.8%）
3. **开源模型训练有效**：未经人工标注的Llama-3-70B*在AitW上超越RL训练的DigiRL（74.0% vs 71.9%）
4. **StepCritic优于二元评估**：轨迹级准确率87.9%，高于Captioner+GPT-4的84.6%，且提供细粒度子目标标签
5. **轨迹增强策略有效**：利用部分完成轨迹的截断增强，最大化数据利用率
6. **流行应用测试**：在Google Maps、YouTube、Spotify等8个真实应用上成功率65%

## 亮点与洞察

- 完整的"框架+数据+模型"闭环：从推理框架到自动数据生成到开源模型训练，形成可复现的pipeline
- StepCritic的子目标粒度评估比简单的任务成功/失败判断提供了更丰富的训练信号
- AutoCheck采用规则而非LLM自检是务实的设计选择——避免了LLM自检标准不一致导致的false positive
- ExpSearch的迭代自我改进机制实现了**无需人工干预的从易到难泛化**
- 轨迹增强算法巧妙利用了部分完成的轨迹，极大缓解了数据稀缺问题

## 局限性

- AndroidWorld任务成功率仍不到50%，Hard任务仅11.8%，距离实用仍有差距
- 依赖GPT-4o作为StepCritic评判器，存在评判偏差和API成本问题
- 环境观测基于XML accessibility tree，缺乏视觉感知能力（不用截图）
- 仅在英文环境下评测，未覆盖其他语言的移动端场景
- ExpSearch的检索质量受数据库大小限制，初始阶段可能缺乏相似任务

## 相关工作与启发

- **AI Scientist / AppAgent / Mobile-Agent**：各类agent框架的对比参照
- **DigiRL** (Bai et al., 2024)：基于RL的离线到在线训练方法，AndroidGen的训练数据方法可与之互补
- **ReAct** (Yao et al., 2022)：推理+行动范式的基础，ReflectPlan在其基础上增加了计划动态更新
- 对"自动生成训练数据→训练开源模型"这一范式有普遍启发意义

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐⭐ |
| 总体推荐 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling](metal_a_multi-agent_framework_for_chart_generation_with_test-time_scaling.md)
- [\[ACL 2025\] OS Agents: A Survey on MLLM-based Agents for General Computing Devices Use](os_agents_survey_mllm.md)
- [\[ACL 2025\] Bel Esprit: Multi-Agent Framework for Building AI Model Pipelines](bel_esprit_multi-agent_framework_for_building_ai_model_pipelines.md)
- [\[NeurIPS 2025\] Ground-Compose-Reinforce: Grounding Language in Agentic Behaviours using Limited Data](../../NeurIPS2025/llm_agent/ground-compose-reinforce_grounding_language_in_agentic_behaviours_using_limited_.md)
- [\[ICML 2025\] From Passive to Active Reasoning: Can Large Language Models Ask the Right Questions under Incomplete Information?](../../ICML2025/llm_agent/from_passive_to_active_reasoning_can_large_language_models_ask_the_right_questio.md)

<!-- RELATED:END -->
