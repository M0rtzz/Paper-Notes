---
title: >-
  [论文解读] TripTailor: A Real-World Benchmark for Personalized Travel Planning
description: >-
  [ACL 2025][Travel Planning] 提出 TripTailor，一个基于真实数据的大规模旅行规划 benchmark，包含 40 个城市的 50 万+ POI 和近 4000 条真实行程，并引入可行性、合理性和个性化三维评估框架，发现最先进 LLM 生成的行程不到 10% 能达到人类水平。
tags:
  - ACL 2025
  - Travel Planning
  - 个性化行程规划
  - LLM Agent
  - benchmark
  - 真实世界评估
---

# TripTailor: A Real-World Benchmark for Personalized Travel Planning

**会议**: ACL 2025  
**arXiv**: [2508.01432](https://arxiv.org/abs/2508.01432)  
**代码**: 有 (https://github.com/swxkfm/TripTailor)  
**领域**: NLP / LLM Agent / 旅行规划  
**关键词**: Travel Planning, 个性化行程规划, LLM Agent, benchmark, 真实世界评估

## 一句话总结

提出 TripTailor，一个基于真实数据的大规模旅行规划 benchmark，包含 40 个城市的 50 万+ POI 和近 4000 条真实行程，并引入可行性、合理性和个性化三维评估框架，发现最先进 LLM 生成的行程不到 10% 能达到人类水平。

## 研究背景与动机

LLM 在旅行规划中展示了巨大潜力，但现有 benchmark 存在明显不足：

**数据不真实**：TravelPlanner 主要依赖模拟数据评估，难以反映实际条件

**规模有限**：ChinaTravel 仅覆盖 10 个城市和约 1200 POI/城市，无法捕捉真实旅行需求的复杂性

**评估维度单一**：现有框架过度关注硬约束（如预算、时间），无法评估行程的整体质量

**缺少真实对照**：没有与人类制定的旅行计划进行系统性对比

核心问题：满足所有硬约束的行程是否就是好的行程？答案是否定的——仍可能存在不合理的绕行、时间分配不当、缺乏个性化等问题。

## 方法详解

### 整体框架

TripTailor 包含三个部分：
1. **沙盒环境**：40 个中国热门旅游城市的完整信息数据库
2. **Benchmark 数据集**：近 4000 条真实旅行行程 + 配对的用户查询
3. **三维评估框架**：可行性 + 合理性 + 个性化

### 关键设计

1. **大规模沙盒环境**：

    - 40 个城市，平均每城市 12,500 个 POI
    - 28,832 条火车时刻表 + 15,110 条航班路线
    - 5,622 个精选景点（含评分、票价、地理坐标、推荐游览时长）
    - 89,224 家酒店 + 422,120 家餐厅
    - 设计动机：数量级超越现有数据集，提供更真实的评估基础

2. **真实行程构建四步法**：

    - Step I：从公开互联网收集 40 个城市的 POI 信息，用高德地图补全坐标
    - Step II：从在线旅行社收集自助游行程，选取高评分的详细行程，随机分配出发城市和日期
    - Step III：用 LLM 从行程中提取信息生成第一人称用户查询（注意 LLM 仅做重写，不填充缺失信息）
    - Step IV：质量控制，检查行程末尾异常、交通时间冲突、调度合理性等
    - 查询分两个难度：Easy (2-3天) 和 Hard (4-7天)

3. **三维评估框架**：

    - **可行性**：信息是否在沙盒内（防幻觉）、关键信息是否完整
    - **合理性**：餐厅和景点不重复、餐费合理、游览时长合理、预算控制、路线优化（用 POI 间平均距离衡量）
    - **个性化**：行程是否体现用户的兴趣偏好、美食类型、活动类型、行程强度等
    - 评估方法组合：客观指标 + LLM 评估 + 专用奖励模型

4. **路线优化指标 D_avg**：

    - 计算每日连续 POI 之间的平均地理距离，衡量路线的交通效率
    - 理想策略：将地理位置接近的 POI 安排在同一天

### 损失函数 / 训练策略

- 本文是 benchmark 工作，主要测试现有 LLM 和 agent 框架
- 提出的 Workflow 方法模拟人类旅行规划流程，将任务分解为多个步骤（如选景点→排路线→选餐厅→选酒店）
- 使用 GPT-4o-mini 作为基础 LLM

## 实验关键数据

### 主实验（各方法在Easy/Hard任务上的表现）

| 方法 | 路线距离比 | 可行性(Micro/Macro) | 合理性(Macro) | 个性化(LLM/RM) | 最终超越率 |
|------|-----------|---------------------|---------------|----------------|-----------|
| Workflow (GPT-4o-mini) Easy | **1.8** | 98.9/98.6 | 74.3 | 14.1/11.6 | 18.1% |
| Direct (GPT-4o) Easy | 3.4 | 97.7/95.5 | 28.8 | 17.8/18.4 | 10.2% |
| Direct (o1-mini) Easy | 3.6 | 91.0/83.9 | 33.3 | 29.1/9.6 | 16.1% |
| Workflow (GPT-4o-mini) Hard | **1.7** | 97.7/96.0 | **53.0** | 17.5/12.0 | 14.3% |
| Direct (GPT-4o) Hard | 3.2 | 98.9/97.7 | 16.3 | 8.0/26.4 | 4.9% |

### 不同LLM直接生成的对比（表格）

| 模型 | 可行性Pass | 合理性Pass | 最终超越率 |
|------|-----------|-----------|-----------|
| Qwen2.5-7b | 68.8% | ~5% | 0.7% |
| Qwen2.5-32b | 89.9% | ~12% | 4.7% |
| GPT-4o-mini | 89.3% | ~9% | 0.9% |
| DeepSeek-V3 | 96.5% | ~15% | 7.8% |
| GPT-4o | 98.3% | ~22% | 7.5% |

### 关键发现

1. **不到 10% 达到人类水平**：即使是最先进的 LLM (GPT-4o)，综合超越率也不到 10%，远低于人类制作的行程
2. **可行性 ≠ 合理性**：满足硬约束只保证可行性，合理性（路线优化、时间安排等）的差距更大
3. **Workflow 方法路线效率最优**：路线距离比仅 1.7-1.8，远低于直接生成方法的 3-4
4. **Hard 任务更具挑战性**：4-7天行程的各项指标全面低于 2-3 天
5. **个性化是最大短板**：LLM 评估的个性化超越率普遍低于 30%，说明 LLM 难以深入理解用户偏好
6. **CoT、ReAct、Reflexion 效果有限**：高级推理策略并未显著改善旅行规划质量

## 亮点与洞察

- **数据规模**是核心优势：50 万+ POI 和 4000 条真实行程，比现有 benchmark 高一个数量级
- **三维评估框架**比单纯的约束满足率更全面——可行性只是"及格线"，合理性和个性化才是"优秀"的标准
- 路线距离比这一指标设计简洁有效，能直观反映行程的空间规划质量
- **人类 baseline 的引入**使得评估有了明确的参照物，而非仅比较 LLM 之间的相对排名
- 发现 LLM 在长程规划（Hard）中性能急剧下降，揭示了当前 LLM 在复杂约束优化中的局限

## 局限与展望

- 仅覆盖中国 40 个城市，国际旅行场景未涉及
- 个性化评估主要依赖 LLM 评分和奖励模型，缺少真实用户的满意度反馈
- 沙盒环境的信息可能过时（如景点评分、餐厅信息会变化）
- 未考虑预算分配的优化（如何在有限预算下最大化体验）
- 多人旅行场景（需要平衡不同偏好）未被覆盖

## 相关工作与启发

- 与 TravelPlanner (Xie et al., 2024) 相比，TripTailor 使用真实数据且评估更全面
- 与 ChinaTravel (Shao et al., 2024) 相比，数据规模大一个数量级
- LLM-Modulo (Gundawar et al., 2024) 的外部验证器策略与本文的 Workflow 分解方法可互补
- 启发：旅行规划是一个天然的多目标优化问题，单纯的 LLM 推理不足以解决，需要结合搜索算法和领域知识

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 真实数据的大规模benchmark + 三维评估框架有创新
- **实验充分度**: ⭐⭐⭐⭐⭐ — 多种LLM、多种agent策略、Easy/Hard难度、客观+LLM+RM三种评估
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，数据集构建流程描述详细
- **价值**: ⭐⭐⭐⭐ — 为LLM旅行规划的评估提供了可靠的真实世界基准

<!-- RELATED:START -->

## 相关论文

- [RuleArena: A Benchmark for Rule-Guided Reasoning with LLMs in Real-World Scenarios](rulearena_rule_guided_reasoning.md)
- [AntiLeakBench: Preventing Data Contamination by Automatically Constructing Benchmarks with Updated Real-World Knowledge](antileakbench_preventing_data_contamination_by_automatically_constructing_benchm.md)
- [A Real-world Display Inverse Rendering Dataset](../../ICCV2025/llm_evaluation/a_real-world_display_inverse_rendering_dataset.md)
- [Time Travel is Cheating: Going Live with DeepFund for Real-Time Fund Investment Benchmarking](../../NeurIPS2025/llm_evaluation/time_travel_is_cheating_going_live_with_deepfund_for_real-time_fund_investment_b.md)
- [EcomScriptBench: A Multi-task Benchmark for E-commerce Script Planning via Step-wise Intention-Driven Product Association](ecomscriptbench.md)

<!-- RELATED:END -->
