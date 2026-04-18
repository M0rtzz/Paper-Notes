---
title: >-
  [论文解读] "Are We Done Yet?": A Vision-Based Judge for Autonomous Task Completion of Computer Use Agents
description: >-
  [AAAI2026][多模态][Computer Use Agent] 提出基于 VLM 的自主任务完成评估框架，通过截图+任务描述判断 CUA 是否完成任务，并将评估反馈回传给 Agent 实现自我纠正，在 macOS 环境上达到 73% 评估准确率和 27% 的任务成功率相对提升。
tags:
  - AAAI2026
  - 多模态
  - Computer Use Agent
  - Task Completion Evaluation
  - 视觉语言
  - Autonomous Feedback
---

# "Are We Done Yet?": A Vision-Based Judge for Autonomous Task Completion of Computer Use Agents

**会议**: AAAI2026  
**arXiv**: [2511.20067](https://arxiv.org/abs/2511.20067)  
**代码**: [martasumyk/vision-based-judge](https://github.com/martasumyk/vision-based-judge)  
**领域**: multimodal_vlm  
**关键词**: Computer Use Agent, Task Completion Evaluation, Vision-Language Model, Autonomous Feedback  

## 一句话总结

提出基于 VLM 的自主任务完成评估框架，通过截图+任务描述判断 CUA 是否完成任务，并将评估反馈回传给 Agent 实现自我纠正，在 macOS 环境上达到 73% 评估准确率和 27% 的任务成功率相对提升。

## 背景与动机

Computer Use Agents (CUAs) 是一类能自主操作数字界面的 AI 系统，通过感知屏幕状态执行点击、输入等操作来完成用户目标。与基于 API 调用的 Agent 不同，CUA 以服务无关的方式直接操作 GUI，具有更强的通用性和可扩展性。

然而 CUA 面临一个关键问题：**无法可靠判断任务是否完成**。具体表现为两种失败模式：

1. **虚假完成**：Agent 宣称任务完成但实际未完成，损害用户信任
2. **完成未识别**：任务实际已完成但 Agent 未能识别，导致冗余操作和计算浪费

现有方法如 OSWorld 的脚本验证需要为每个任务手写验证脚本，扩展性极差；而 Web 环境下的评估方法依赖 HTML 等结构化表示，无法直接迁移到桌面环境。桌面界面视觉元素更多、缺乏统一的结构化表示，评估难度更大。

## 核心问题

如何在没有脚本、日志或结构化表示的情况下，仅凭屏幕截图和任务描述自动判断 CUA 是否完成了指定任务，并通过反馈机制提升 Agent 的任务成功率？

## 方法详解

### 整体框架

提出三阶段的评估-反馈流水线：

1. **任务执行（Task Execution）**：CUA 根据任务描述在 macOS 环境中执行操作，记录完整轨迹（截图、动作、推理过程）
2. **结果评估（Outcome Evaluation）**：将最终截图和任务描述输入 VLM，zero-shot 判断任务是否完成，同时生成自然语言解释
3. **反馈重试（Feedback and Retry）**：若 VLM 判定未完成，将推理结果回传 CUA，Agent 从当前状态继续尝试而非从头开始

### 数据集构建

- 覆盖 **42 个** macOS 内置应用（生产力、通信、多媒体、系统工具、开发工具等）
- 每个应用定义 **30 个**具体任务，共 **1,260 个**任务（OSWorld 仅 369 个）
- 任务难度跨度大：从简单操作（打开日历应用）到复杂多步交互（在 App Store 中筛选免费应用并打开第一个结果）
- 避免需要私人数据或外部配置的任务

### 评估模型选择

- **被评估的 CUA**：Claude Computer Use、OpenAI Operator、UI-TARS（开源）
- **评估器 VLM**：
    - 闭源：GPT-4o、Claude 3.5 Sonnet
    - 开源：LLaVA-v1.5-7B、InternVL 2-8B、Qwen2-VL-7B

### 评估方式

采用 zero-shot prompting，VLM 仅依据最终截图和任务描述进行二分类判断（done/not done），同时输出简短的自然语言理由。评估器与执行 Agent 完全独立，避免评估偏差。

## 实验关键数据

### 评估准确率（对比人工标注）

| 评估器 | OpenAI Operator | Anthropic CU | UI-TARS |
|---|---|---|---|
| GPT-4o | 0.61 | 0.69 | 0.64 |
| Claude 3.5 Sonnet | **0.69** | **0.71** | **0.73** |
| LLaVA-v1.5-7B | 0.56 | 0.61 | 0.52 |
| InternVL 2-8B | 0.62 | 0.67 | 0.61 |
| Qwen2-VL-7B | 0.68 | 0.66 | 0.70 |

- Claude 3.5 Sonnet 在闭源模型中表现最好，最高达 73%
- Qwen2-VL-7B 在开源模型中表现最好，接近闭源水平

### 反馈对成功率的提升

- 所有 VLM 反馈机制均带来可观的性能提升
- 闭源评估器最高实现 **61%** 的相对成功率提升
- 基线较弱的 Agent（如 Anthropic CU）从反馈中受益最大
- 仅需一次重试即可实现平均 **27%** 的相对成功率提升

## 亮点

1. **极简设计**：仅用最终截图+任务描述即可评估，无需脚本、日志或结构化表示，通用性强
2. **闭环反馈**：评估结果直接回传 Agent 用于自我纠正，形成评估-反馈-重试的完整闭环
3. **弱者获益更多**：基线越低的 Agent 从反馈中获得的提升越大，说明该机制有效弥补了 Agent 的自我感知缺陷
4. **数据集贡献**：1,260 个 macOS 任务，覆盖 42 个应用，规模约为 OSWorld 的 3.4 倍
5. **开源模型可用**：Qwen2-VL-7B 等 7B 开源模型表现接近闭源，降低了部署门槛

## 局限与展望

1. **仅限 macOS**：未覆盖 Windows、Linux 等主流桌面系统，通用性有待验证
2. **仅终态评估**：只看最终截图，无法判断中间步骤的正确性；未来可拓展为逐步评估
3. **准确率天花板**：最高 73% 的准确率意味着接近三成的误判，对安全敏感场景而言不够
4. **单次重试**：仅实验了一次反馈重试，多轮反馈的效果和收敛性未探索
5. **二分类局限**：只判断 done/not done，缺乏部分完成或完成质量的细粒度评估
6. **缺少消融实验**：未分析截图数量、时序信息、prompt 设计等因素对评估效果的影响

## 与相关工作的对比

| 方法 | 环境 | 评估方式 | 是否需要脚本 | 反馈闭环 |
|---|---|---|---|---|
| OSWorld | 多平台 | 手写验证脚本 | 是 | 否 |
| Pan et al. | Web/模拟 | 结构化表示+文本推理 | 否 | 是 |
| AutoEval（机器人） | 物理环境 | VLM 评估 | 否 | 否 |
| **本文** | macOS | VLM + 截图 | **否** | **是** |

与机器人领域的 AutoEval 思路相似，但适配了桌面数字环境的特殊挑战（无物理信号、界面复杂多样）。相比 Web 评估方法，本文处理的是缺乏 HTML 等结构化表示的桌面环境。

## 启发与关联

- **Agent 自我认知**：CUA 不知道自己"做完了没有"是一个被低估的问题，本文将其显式建模为一个独立的评估任务
- **评估即奖励**：VLM 评估器的输出可直接作为 RL 训练的 reward signal，替代人工标注
- **多 Agent 协作**：评估器和执行器分离的架构天然适合多 Agent 系统，未来可发展为实时监控+即时反馈的协作框架
- **与 GUI Agent 研究的交叉**：该评估框架可泛化到任何 GUI Agent 的任务完成判定，不限于 CUA

## 评分

- 新颖性: ⭐⭐⭐ — 思路直接但有效，将 VLM-as-judge 应用于 CUA 评估是自然延伸
- 实验充分度: ⭐⭐⭐ — 三个 CUA + 五个评估器覆盖较全，但缺少消融和深度分析
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，问题定义明确，技术细节充足
- 价值: ⭐⭐⭐ — 指出了一个重要问题并提供了简洁解决方案，但准确率和规模有待提升
