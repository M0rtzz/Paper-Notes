---
title: >-
  [论文解读] Beyond Itinerary Planning: A Real-World Benchmark for Multi-Turn and Tool-Using Travel Tasks
description: >-
  [ACL 2026][旅行规划基准] 提出 TravelBench，首个融合真实用户查询、隐式用户偏好、多轮交互、不可解任务识别和10种真实工具的旅行规划基准，通过沙箱环境实现可复现评估，揭示前沿模型在不同能力维度上表现不均衡。
tags:
  - ACL 2026
  - 旅行规划基准
  - 工具使用
  - 多轮对话
  - 隐式偏好
  - 不可解任务
---

# Beyond Itinerary Planning: A Real-World Benchmark for Multi-Turn and Tool-Using Travel Tasks

**会议**: ACL 2026  
**arXiv**: [2512.22673](https://arxiv.org/abs/2512.22673)  
**代码**: [GitHub](https://github.com/small-xiangcheng/TravelBench)  
**领域**: LLM智能体/旅行规划  
**关键词**: 旅行规划基准, 工具使用, 多轮对话, 隐式偏好, 不可解任务

## 一句话总结

提出 TravelBench，首个融合真实用户查询、隐式用户偏好、多轮交互、不可解任务识别和10种真实工具的旅行规划基准，通过沙箱环境实现可复现评估，揭示前沿模型在不同能力维度上表现不均衡。

## 研究背景与动机

**领域现状**：旅行规划是评估LLM Agent多步推理、工具使用和用户交互能力的理想测试场景。现有基准（TravelPlanner、ChinaTravel等）已取得进展，但仍存在关键不足。

**现有痛点**：（1）用户偏好和约束通常是预定义的，注入指令或由模拟器逐步揭示，无法动态引出用户的隐式偏好；（2）大多基准仅覆盖行程规划，忽略POI探索、路线规划、方案比较等多样化真实旅行需求；（3）要么不支持工具使用，要么依赖合成查询，无法反映真实数据分布；（4）缺少不可解任务的评估——实际场景中Agent必须识别能力边界。

**核心矛盾**：现有基准的性能不能真实反映Agent在实际旅行规划中的表现，因为它们在任务范围、用户交互方式和评估覆盖度上与真实需求差距显著。

**本文目标**：构建一个"真正面向真实世界"的旅行规划基准，全面评估Agent的三个核心能力：独立解决问题、交互引出隐式偏好、识别能力边界。

**切入角度**：从阿里巴巴高德地图的真实用户日志中收集查询和偏好，集成10种真实旅行工具，构建约20万条缓存的工具调用轨迹。

**核心idea**：将旅行规划基准从"行程规划"扩展到覆盖POI探索、路线规划、方案比较等多领域，引入隐式偏好的多轮引出和不可解任务识别两个全新维度。

## 方法详解

### 整体框架

TravelBench 包含三个子集：500个单轮查询（Agent独立用工具解决）、500个多轮查询（需与用户交互引出隐式偏好）、100个不可解查询（需识别缺失工具或信息）。用户模拟器由LLM+用户画像驱动，Agent在集成10种真实工具的沙箱环境中执行推理。评估采用LLM-as-judge + 工具调用错误惩罚 + 元评审校准的三层协议。

### 关键设计

1. **隐式偏好与多轮交互**：

    - 功能：评估Agent主动引出用户未明确表达偏好的能力
    - 核心思路：从真实用户数据中匿名化提取偏好信息，构建用户画像（性别、家庭结构、生活方式等）。画像仅由用户模拟器持有，Agent必须通过多轮提问来获取。用3个模型×2次试验判定查询是单轮还是多轮
    - 设计动机：现有基准的偏好要么预定义注入要么由模拟器逐步揭示，不支持Agent主动探索和引出

2. **不可解任务子集**：

    - 功能：评估Agent识别自身能力边界的能力
    - 核心思路：3个模型（GPT-5.1、Qwen3-235B、Qwen-Plus）标注每个查询是否可解。三个模型一致判定不可解的作为不可解子集，分为三类原因：缺少工具支持、缺少必要上下文、无明确可执行意图。Agent需输出特殊标签[Unsolved]表示识别
    - 设计动机：实际场景中Agent必须知道何时说"我做不到"，而非强行给出错误答案

3. **可复现沙箱与工具缓存**：

    - 功能：确保评估的稳定性和可复现性
    - 核心思路：用多个模型在真实API上运行，缓存约20万条工具调用轨迹。评估时优先从缓存匹配，缓存未命中时通过嵌入检索+ICL模拟生成一致的工具响应。同时进行严格的参数验证，记录工具调用错误率
    - 设计动机：直接调用外部API结果不稳定，影响可复现性和公平比较

### 评估协议

三层评估：（1）基于规则的标准对不可解子集计算准确率；（2）LLM-as-judge对单轮（3维度）和多轮（4维度，增加user_interaction）打1-5分；（3）元评审校准过度高估的分数 + 工具调用错误率惩罚。最终分数为三个子集的平均。

## 实验关键数据

### 主实验

| 模型 | 多轮(惩罚后) | 单轮(惩罚后) | 不可解 | 总分 |
|------|------------|------------|--------|------|
| Qwen-Plus | 62.56 | 82.64 | 83.67 | **76.29** |
| GPT-5.1 | 71.31 | 73.81 | 80.00 | 75.04 |
| Kimi-K2-Th | 71.83 | 77.31 | 73.67 | 74.27 |
| DeepSeek-V3.2 | 82.80 | 83.29 | 51.33 | 72.47 |
| Qwen3-235B-It | 61.78 | 70.74 | 80.00 | 70.84 |
| DeepSeek-R1 | 35.67 | 76.93 | 83.67 | 65.42 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 评分稳定性 | std ≈ 0.01 | 3次重复运行标准差极小 |
| 离线 vs 在线评分 | <1分差异 | 缓存沙箱与真实API评估高度一致 |
| 人工验证 | 97%标签一致 | 不可解和单/多轮标签与人工判断高度吻合 |
| Judge MAE | 0.52 vs 人类间0.48 | 接近人类间分歧水平 |

### 关键发现
- 最强模型也仅约76分，说明真实旅行规划仍具挑战性
- 能力不均衡普遍存在：DeepSeek-V3.2在单/多轮最强但不可解识别差（51.33），Kimi-K2-0925不可解最强（94）但任务完成差
- 推理模型的工具调用错误率普遍低于指令遵循模型，但在不可解任务上反而倾向偏低——过强的推理让模型更"不愿放弃"
- 多轮任务的工具调用错误率高于单轮，说明多轮交互增加了工具使用的难度
- 工具惩罚对MiniMax-M2影响最大（78→67），有效区分了"看似正确但工具使用有问题"的轨迹

## 亮点与洞察
- **真正面向真实世界**：从高德真实日志出发，覆盖32个省级区域、243个城市，任务分布反映真实用户需求
- **三个核心能力的统一评估**：独立解决、交互引出偏好、识别边界——这是Agent实用化的完整能力画像
- **工具惩罚机制创新**：不仅看任务是否完成，还看工具使用过程是否可靠
- **能力不均衡的发现重要**：指出当前模型在"强推理 vs 知道放弃"之间的张力

## 局限与展望
- **仅覆盖中国旅行场景**：地理和文化范围有限，国际旅行需求未覆盖
- **用户模拟器的局限**：LLM模拟的用户交互可能与真实用户行为有偏差
- **评估依赖LLM judge**：虽然人工验证显示高一致性，但仍可能存在盲点
- 未来方向：扩展到国际旅行、引入更复杂的动态约束变化、研究Agent如何更好地平衡能力和边界识别

## 相关工作与启发
- **vs TravelPlanner**：首个旅行规划基准但已被求解器方法解决，任务过于简单
- **vs ChinaTravel**：引入真实查询和更严格约束，但不支持多轮交互和不可解任务
- **vs COMPASS**：关注软偏好优化但不使用真实查询和工具

## 评分
- 新颖性: ⭐⭐⭐⭐ 隐式偏好引出和不可解任务识别是重要的新维度，任务覆盖面远超前作
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖12+个模型，有稳定性分析、人工验证和在线/离线对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，评估协议描述详尽
- 价值: ⭐⭐⭐⭐ 为旅行规划Agent评估提供了最全面的基准，能力不均衡的发现对Agent设计有指导意义

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Wide-Horizon Thinking and Simulation-Based Evaluation for Real-World LLM Planning with Multifaceted Constraints](../../NeurIPS2025/recommender/wide-horizon_thinking_and_simulation-based_evaluation_for_real-world_llm_plannin.md)
- [\[ACL 2026\] HORIZON: A Benchmark for in-the-wild User Behaviour Modeling](horizon_a_benchmark_for_in-the-wild_user_behaviour_modeling.md)
- [\[CVPR 2025\] VisionArena: 230K Real World User-VLM Conversations with Preference Labels](../../CVPR2025/recommender/visionarena_230k_real_world_user-vlm_conversations_with_preference_labels.md)
- [\[ACL 2026\] Where and What: Reasoning Dynamic and Implicit Preferences in Situated Conversational Recommendation](where_and_what_reasoning_dynamic_and_implicit_preferences_in_situated_conversati.md)
- [\[AAAI 2026\] TraveLLaMA: A Multimodal Travel Assistant with Large-Scale Dataset and Structured Reasoning](../../AAAI2026/recommender/travellama_a_multimodal_travel_assistant_with_large-scale_dataset_and_structured.md)

<!-- RELATED:END -->
