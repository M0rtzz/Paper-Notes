---
title: >-
  [论文解读] From Conversation to Query Execution: Benchmarking User and Tool Interactions for EHR Database Agents
description: >-
  [ICLR 2026][医学图像][EHR] 提出EHR-ChatQA基准，首次评估数据库Agent在电子病历场景中的端到端交互工作流（澄清模糊查询→解决术语不匹配→生成SQL→返回答案），发现最强模型(o4-mini)的Pass@5超90%但Pass∧5(全部成功)大幅下降(差距达60%)，暴露了安全关键领域的鲁棒性缺陷。
tags:
  - ICLR 2026
  - 医学图像
  - EHR
  - 数据库Agent
  - 交互式QA
  - 查询歧义
  - 值不匹配
---

# From Conversation to Query Execution: Benchmarking User and Tool Interactions for EHR Database Agents

**会议**: ICLR 2026  
**arXiv**: [2509.23415](https://arxiv.org/abs/2509.23415)  
**代码**: [GitHub](https://github.com/glee4810/EHR-ChatQA)  
**领域**: 医学信息学/Agent  
**关键词**: EHR, 数据库Agent, 交互式QA, 查询歧义, 值不匹配

## 一句话总结
提出EHR-ChatQA基准，首次评估数据库Agent在电子病历场景中的端到端交互工作流（澄清模糊查询→解决术语不匹配→生成SQL→返回答案），发现最强模型(o4-mini)的Pass@5超90%但Pass∧5(全部成功)大幅下降(差距达60%)，暴露了安全关键领域的鲁棒性缺陷。

## 研究背景与动机

**领域现状**：LLM Agent越来越多用于与结构化数据库交互。Text-to-SQL基准（Spider、BIRD等）评估单轮自然语言→SQL翻译，但不涉及交互式场景。

**现有痛点**：(1) **查询歧义**：临床用户常提模糊问题（如"给我看最近的化验"）缺乏具体限定；(2) **值不匹配**：临床术语与数据库条目不一致（如"Lopressor" vs "metoprolol tartrate"）；(3) 现有基准不评估Agent在多轮交互中澄清intent、调用工具解决不匹配、生成正确SQL的端到端能力。

**核心矛盾**：单轮SQL生成已经不够——真实临床场景需要Agent主动提问、搜索数据库值、调用外部知识来桥接用户意图与数据库内容之间的鸿沟。

**切入角度**：构建完整的交互环境（LLM模拟用户+工具集+验证器），评估Agent从对话到查询执行的全链路。

## 方法详解

### 基准构建

1. **两种交互流**:
    - IncreQA（增量查询细化）: 用户逐步添加约束→Agent需保持上下文→线性目标
    - AdaptQA（自适应查询调整）: 用户根据中间结果调整搜索目标→Agent需处理分支

2. **工具集**: schema搜索、列搜索、值子串搜索、值相似度搜索、web搜索、SQL执行

3. **模拟用户**: Gemini-2.0-Flash(温度1.0) + 嵌套验证-反思机制确保行为一致

4. **仿真验证器**: 判断模拟用户是否偏离指令，偏离则重跑

### 关键指标
- Pass@5: 5次试验中至少1次成功（乐观评估）
- Pass∧5: 5次试验中全部成功（鲁棒性评估）
- 差距 = Pass@5 - Pass∧5: 衡量不稳定性

### 数据规模
- 366个任务实例，基于MIMIC-IV和eICU两个真实EHR数据库
- 覆盖多种查询歧义和值不匹配模式

## 实验关键数据

### 主实验
| 模型 | IncreQA Pass@5 | IncreQA Pass∧5 | AdaptQA Pass@5 | AdaptQA Pass∧5 |
|------|---------------|----------------|----------------|----------------|
| o4-mini | >90% | ~50% | 60-70% | ~20% |
| Gemini-2.5-Flash | >85% | ~45% | 55-65% | ~20% |

### Pass@5 vs Pass∧5差距
| 设置 | 最大差距 | 说明 |
|------|---------|------|
| IncreQA | >38% | 乐观vs鲁棒 |
| AdaptQA | >36% | 更不稳定 |
| 总体最大 | **~60%** | 极度不可靠 |

### 关键发现
- Pass@5高→说明Agent有能力解决任务，但Pass∧5低→不能稳定地解决
- AdaptQA比IncreQA更难——需要根据中间结果灵活调整策略
- 值不匹配是主要失败原因——Agent不能可靠地将用户术语映射到数据库条目
- LLM模拟用户偶尔偏离指令→验证器有效识别并重跑（~10%比例）

## 亮点与洞察
- **Pass@5 vs Pass∧5的洞察**：这种双指标评估揭示了关键的安全问题——在EHR领域"偶尔能做对"不够，需要"每次都做对"。60%的差距意味着同一个Agent在相同任务上有接近随机的表现。
- **值不匹配的重要性**：之前的Text-to-SQL基准忽略了这个问题，但在EHR中极其关键——"Lopressor"不等于"metoprolol tartrate"可能导致完全错误的查询结果。
- **交互流设计**：IncreQA测试上下文保持，AdaptQA测试灵活适应——两者对齐了真实临床数据访问的不同模式。

## 局限与展望
- LLM模拟用户引入了不可控的随机性
- 仅两个EHR数据库，更多样的数据源可能有不同挑战
- 工具集是预定义的，真实场景可能需要更灵活的工具调用
- 评估是离线的，在线部署的延迟和交互体验未考虑

## 相关工作与启发
- **vs EHRSQL**: EHRSQL是单轮Text-to-SQL，EHR-ChatQA是多轮交互式
- **vs Tau-Bench**: Tau-Bench测通用Agent交互，EHR-ChatQA专注EHR域的特殊挑战
- **vs MedAgentBench**: MedAgentBench的任务指令不模糊，不需要交互澄清

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个结合用户交互+工具使用+值不匹配的EHR QA基准
- 实验充分度: ⭐⭐⭐⭐ SOTA模型评估、5次重复、诊断分析
- 写作质量: ⭐⭐⭐⭐ POMDP形式化、任务设计清晰
- 价值: ⭐⭐⭐⭐⭐ 对EHR Agent部署的安全可靠性有直接警示意义

<!-- RELATED:START -->

## 相关论文

- [ReflecTool: Towards Reflection-Aware Tool-Augmented Clinical Agents](../../ACL2025/medical_imaging/reflectool_clinical_agent.md)
- [Incentives in Federated Learning with Heterogeneous Agents](incentives_in_federated_learning_with_heterogeneous_agents.md)
- [DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction](dm4ct_benchmarking_diffusion_models_for_computed_tomography_reconstruction.md)
- [Shoot First, Ask Questions Later? Building Rational Agents that Explore and Act Like People](shoot_first_ask_questions_later_building_rational_agents_that_explore_and_act_li.md)
- [Benchmarking ECG FMs: A Reality Check Across Clinical Tasks](benchmarking_ecg_fms_a_reality_check_across_clinical_tasks.md)

<!-- RELATED:END -->
