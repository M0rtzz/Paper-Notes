---
title: >-
  [论文解读] BTL-UI: Blink-Think-Link Reasoning Model for GUI Agent
description: >-
  [NeurIPS 2025][LLM Agent][GUI agent] 提出 Blink-Think-Link（BTL）脑启发框架，将 GUI 交互分解为 Blink（快速注意力定位）、Think（认知推理决策）、Link（可执行命令生成）三个生物合理阶段，配合自动化 Blink 数据标注 pipeline 和首个基于规则的过程+结果复合奖励机制 BTL Reward，训练的 BTL-UI 在静态 GUI 理解和动态交互 benchmark 上达到 competitive 性能。
tags:
  - NeurIPS 2025
  - LLM Agent
  - GUI agent
  - Blink-Think-Link
  - cognitive-inspired
  - GRPO
  - BTL Reward
---

# BTL-UI: Blink-Think-Link Reasoning Model for GUI Agent

**会议**: NeurIPS 2025  
**arXiv**: [2509.15566](https://arxiv.org/abs/2509.15566)  
**代码**: [https://github.com/xiaomi-research/btl-ui](https://github.com/xiaomi-research/btl-ui)  
**领域**: LLM Agent / GUI 自动化  
**关键词**: GUI agent, Blink-Think-Link, cognitive-inspired, GRPO, BTL Reward

## 一句话总结

提出 Blink-Think-Link（BTL）脑启发框架，将 GUI 交互分解为 Blink（快速注意力定位）、Think（认知推理决策）、Link（可执行命令生成）三个生物合理阶段，配合自动化 Blink 数据标注 pipeline 和首个基于规则的过程+结果复合奖励机制 BTL Reward，训练的 BTL-UI 在静态 GUI 理解和动态交互 benchmark 上达到 competitive 性能。

## 研究背景与动机

AI 驱动的 GUI 交互自动化快速发展，但现有方法存在两类根本问题：

-    **SFT 方法**：依赖大规模专家标注数据，面对分布外场景泛化能力弱
-    **现有 RFT 方法**：采用 Think-Answer 结构（`<think>` + `<answer>`），但与人类 GUI 交互模式差距大，且奖励机制仅关注最终结果，缺乏对中间认知过程的引导

认知科学研究表明，人类与 GUI 交互遵循三个sequential过程：(a) 扫视阶段快速定位目标；(b) 多模态信息整合与推理；(c) 精确动作执行。现有 agent 跳过了关键的注意力定位阶段。

## 方法详解

### 整体框架

BTL 将 GUI 交互建模为 MDP，策略函数 F({z_t, u, h}) → o_t = {b_t, d_t, a_t}，输出三个阶段的结果。使用 GRPO（Group Relative Policy Optimization）进行优化。

### 关键设计

1.    **Blink 阶段（视觉注意力定位）**：

    -    功能：快速定位屏幕上与当前任务相关的 ROI 区域，输出封装在 `<blink></blink>` 标签中
    -    核心思路：模拟人类扫视眼跳（saccadic eye movements），通过两阶段 pipeline 生成 Blink 数据——(1) 解析模型提取所有 UI 元素（bbox/类型/caption）；(2) Qwen2.5-VL-32B 根据任务指令筛选 λ 个最相关元素
    -    设计动机：现有方法直接从截图生成动作，缺乏对任务相关区域的显式注意；Blink 提供了 top-down 注意力引导

2.    **Think 阶段（认知推理）**：

    -    功能：基于 Blink 定位的区域进行高级推理和决策，输出推理过程在 `<think></think>` 标签中
    -    核心思路：理解当前状态、分析任务目标、规划下一步操作
    -    设计动机：推理阶段保留了 DeepSeek-R1 风格的 chain-of-thought，但建立在 Blink 提供的聚焦信息之上

3.    **Link 阶段（动作生成）**：

    -    功能：生成可执行的 GUI 命令（点击坐标、文本输入等），输出在 `<link></link>` 标签中
    -    核心思路：动作类型 α_t + 参数 δ_t 构成完整命令
    -    设计动机：与 Think 分离确保命令的结构化和可解析性

### 损失函数 / 训练策略

**BTL Reward** 由三个组件构成：R_BTL = R_format + R_blink + R_link

-    **Dual Format Reward（R_format）**：检查输出是否满足 BTL 三阶段模板结构 + XML/JSON 内容格式，二值奖励
-    **Blink Reward（R_blink）**：使用匈牙利匹配器计算预测 ROI 与 GT ROI 的 IoU，允许预测为空（当需要滚动/返回等非交互操作时）
-    **Link Reward（R_link）**：严格二值——动作类型和参数必须**同时正确**才给奖励，防止 reward hacking

使用 GRPO 优化：生成 N 个补全 → 计算组内相对优势 A_i → 最大化策略目标（含 KL 约束）。基于 Qwen2.5-VL-3B/7B 训练。

## 实验关键数据

### 主实验（表格）

| 模型 | 方法 | ScreenSpot Avg. | ScreenSpot-V2 | ScreenSpot-Pro |
|------|------|----------------|---------------|----------------|
| GPT-4o | ZS | 18.8 | - | - |
| Qwen2.5-VL-7B | ZS | 82.0 | - | - |
| OS-Atlas-Base-7B | ZS | 82.5 | - | - |
| **BTL-UI-3B** | RFT | **competitive** | competitive | competitive |
| **BTL-UI-7B** | RFT | **competitive** | competitive | competitive |

BTL-UI 使用仅 4K 数据进行 RFT 训练，在 ScreenSpot 系列 grounding benchmark 和 AndroidControl/GUI-Odyssey 等 planning benchmark 上均达到 competitive 性能。

### 消融实验

-    **完整 BTL vs Think-Answer**：BTL 三阶段结构 > 传统 Think-Answer 两阶段
-    **BTL Reward vs 仅 Link Reward**：过程引导（Blink Reward）显著提升整体性能
-    **严格 Link Reward vs 拆分奖励**：严格二值比分离动作类型和参数的奖励更有效（避免 reward hacking）
-    **Blink 筛选元素数 λ 的影响**：λ 过小信息不足，过大增加 token 开销

### 关键发现

-    过程引导对 GUI agent 至关重要——不仅评价"做对了什么"，还评价"看对了什么"
-    Blink 阶段的注意力定位使后续推理更聚焦——减少了对不相关 UI 元素的干扰
-    严格的 Link Reward 防止模型学会"猜对动作类型但参数错误"的投机策略

## 亮点与洞察

-    **脑启发设计有实效**：Blink→Think→Link 模拟人类的注视→思考→操作，不是噱头而是切实提升了性能
-    **BTL Reward 是 GUI agent 领域首个过程+结果复合奖励**——为 RFT-based GUI agent 提供了更丰富的训练信号
-    **自动化 Blink 数据标注**解决了 GUI agent 训练数据瓶颈——不需要为注意力区域做人工标注
-    **4K 训练数据即可有效**——数据效率高

## 局限与展望

-    三阶段串行执行增加推理延迟——Blink 阶段的计算开销是否可以减少？
-    Blink 数据的自动标注依赖 Qwen2.5-VL-32B——标注模型的能力上限限制了 Blink 数据质量
-    仅在静态 grounding 和短 horizon planning 上验证——复杂多步交互任务的长 horizon 性能未充分评估
-    RFT 训练仅用 4K 数据——更大规模训练的 scaling 效果未探索

## 相关工作与启发

-    **UI-R1 / GUI-R1**：引入规则化 RL 做 GUI agent，但采用 Think-Answer 结构且仅有 outcome reward
-    **InfiGUI-R1**：Actor2Reasoner 架构桥接反应式执行和审慎推理，但缺乏显式的注意力建模
-    **UI-TARS**：结合预训练 + SFT，但不使用 RL
-    启发：BTL 的三阶段框架可迁移到其他人机交互场景（如自动驾驶的 Look→Plan→Act）

## 评分

-    新颖性: ⭐⭐⭐⭐ 脑启发三阶段框架 + BTL Reward 设计独特
-    实验充分度: ⭐⭐⭐⭐ 多个 GUI benchmark 全面评估 + 充分消融
-    写作质量: ⭐⭐⭐⭐ 认知科学动机与技术设计结合紧密
-    价值: ⭐⭐⭐⭐ GUI agent 的过程引导 RL 有开创意义

<!-- RELATED:START -->

## 相关论文

- [GUICourse: From General Vision Language Model to Versatile GUI Agent](../../ACL2025/llm_agent/guicourse_from_general_vision_language_model_to_versatile_gui_agent.md)
- [R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](rd-agent-quant_a_multi-agent_framework_for_data-centric_factors_and_model_joint_.md)
- [TrajAgent: An LLM-Agent Framework for Trajectory Modeling via Large-and-Small Model Collaboration](trajagent_an_llm-agent_framework_for_trajectory_modeling_via_large-and-small_mod.md)
- [History-Aware Reasoning for GUI Agents](../../AAAI2026/llm_agent/history-aware_reasoning_for_gui_agents.md)
- [AgentTTS: Large Language Model Agent for Test-time Compute-optimal Scaling Strategy in Complex Tasks](agenttts_large_language_model_agent_for_testtime_computeopti.md)

<!-- RELATED:END -->
