---
title: >-
  [论文解读] Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement
description: >-
  [ACL 2025][LLM Agent][自引用Agent] 提出 Gödel Agent，一种受 Gödel Machine 启发的自引用 Agent 框架，能通过 Python monkey patching 在运行时读取和修改自身代码（包括修改自身的修改逻辑），实现递归自改进，在 DROP/MGSM/MMLU/GPQA 上超越手工设计和元学习优化的 Agent。
tags:
  - ACL 2025
  - LLM Agent
  - 自引用Agent
  - 递归自改进
  - Monkey Patching
  - 元学习
  - Agent设计空间搜索
---

# Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement

**会议**: ACL 2025  
**arXiv**: [2410.04444](https://arxiv.org/abs/2410.04444)  
**代码**: https://github.com/Arvid-pku/Godel_Agent (有)  
**领域**: LLM Agent  
**关键词**: 自引用Agent, 递归自改进, Monkey Patching, 元学习, Agent设计空间搜索

## 一句话总结
提出 Gödel Agent，一种受 Gödel Machine 启发的自引用 Agent 框架，能通过 Python monkey patching 在运行时读取和修改自身代码（包括修改自身的修改逻辑），实现递归自改进，在 DROP/MGSM/MMLU/GPQA 上超越手工设计和元学习优化的 Agent。

## 研究背景与动机

**领域现状**：LLM Agent 系统分为两类——手工设计的固定流程 Agent（CoT、Self-Refine、LLM Debate 等）和元学习优化 Agent（Meta Agent Search、DSPy 等）。前者完全依赖人类先验，后者虽能自动优化策略但元学习算法本身是固定的。

**现有痛点**：(1) 手工设计 Agent 一经部署就无法进化；(2) 元学习 Agent 受限于固定的元算法，搜索空间仍有人为限制——例如只能优化 prompt 或只能收集 demonstration；(3) 都无法搜索完整的 Agent 设计空间。

**核心矛盾**：Agent 的"优化方式"本身也是可优化的对象。如果优化算法是固定的，就永远无法发现更好的优化算法。

**本文目标**：能否让 Agent 自主决定自己的运行逻辑、模块设计，甚至修改自身的优化方式？

**切入角度**：受 Schmidhuber 的 Gödel Machine 理论启发——一个能修改自身任意部分（包括修改逻辑本身）的系统理论上可以找到全局最优。

**核心 idea**：实现完全自引用——Agent 可以读取自己的所有代码，修改任意部分（包括决策函数、执行函数），然后递归地继续改进。

## 方法详解

### 整体框架
Gödel Agent = 初始策略 $\pi_0$ + 自引用学习算法 $I_0$。形式化为 $\pi_{t+1}, I_{t+1} = I_t(\pi_t, I_t, r_t, g)$，即优化算法 $I$ 不仅能改 $\pi$，还能改自身。用递归函数（而非循环）实现主逻辑，通过 monkey patching 在运行时动态修改代码。

### 关键设计

1. **自感知（Self-Awareness）**:

    - 功能：Agent 通过检查 Python 运行时内存（local/global variables）读取自己的所有代码
    - 核心思路：`SELF_INSPECT()` 函数提取 Agent 的所有变量、函数、类定义
    - 设计动机：自引用的前提是知道自己的当前状态，包括代码逻辑

2. **自修改（Self-Modification）**:

    - 功能：Agent 用 LLM 生成新代码并通过 monkey patching 写入运行时内存
    - 核心思路：`self_update` 动作让 LLM 分析当前策略和反馈，生成修改后的代码，动态替换运行中的函数/类
    - 设计动机：monkey patching 允许不重启程序就替换任意函数，实现"边运行边改自己"

3. **递归自改进**:

    - 功能：主函数用递归而非循环实现，每次递归深入应用最新修改
    - 核心思路：`SELF_IMPROVE` 函数递归调用自身，每层递归中 Agent 可以修改 `SELF_IMPROVE` 函数本身，下一层递归将使用新版本
    - 设计动机：递归结构使得修改逻辑的修改能在下一次调用中生效——这是"递归自改进"的关键

4. **四种初始动作**:

    - `self_inspect`: 读取自身代码
    - `interact`: 与环境交互获取反馈 $r = U(\mathcal{E}, \pi)$
    - `self_update`: 修改自身代码
    - `continue_improve`: 递归调用自改进
    - 动作集本身也可以被 Agent 修改和扩展

### 辅助设计
- **Think Before Acting**：允许 Agent 先推理再行动，提升决策质量
- **Error Handling**：代码修改出错时不终止，而是带着错误信息进入下一轮迭代
- **额外工具**：执行 Python/Bash 代码、调用 LLM API

### 损失函数 / 训练策略
- 无传统训练——Agent 在验证集上迭代自改进
- 每个任务 6 个独立 cycle，每 cycle 最多 30 迭代
- 初始策略为 CoT，所有测试用 GPT-3.5

## 实验关键数据

### 主实验

| 方法 | DROP (F1) | MGSM (Acc) | MMLU (Acc) | GPQA (Acc) |
|------|----------|-----------|-----------|-----------|
| CoT | 64.2 | 28.0 | 65.4 | 29.2 |
| CoT-SC | 64.4 | 28.2 | 65.9 | 30.5 |
| Self-Refine | 59.2 | 27.5 | 63.5 | 31.6 |
| LLM Debate | 60.6 | 39.0 | 65.6 | 31.4 |
| Meta Agent Search | 79.4 | 53.4 | 69.6 | 34.6 |
| **Gödel-base (GPT-3.5)** | **80.9** | **64.2** | **70.9** | **34.9** |
| Gödel-free (无约束) | 90.5 | 90.6 | 87.9 | 55.7 |

### 消融实验（MGSM）

| 配置 | Accuracy |
|------|---------|
| Full | 64.2 |
| w/o think | 50.8 (-13.4) |
| w/o error handling | 49.4 (-14.8) |
| w/o code running | 57.1 (-7.1) |
| w/o LLM calling | 60.4 (-3.8) |

### 关键发现
- **MGSM 上超 Meta Agent Search 11 个点**（64.2 vs 53.4），说明数学推理任务有更大的自改进空间
- **无约束模式性能炸裂**：Agent 自发请求 GPT-4o 辅助，GPQA 从 34.9→55.7
- **Error Handling 极为重要**：去掉后掉 14.8 个点——LLM 生成的代码经常有错，容错机制是持续优化的基础
- **只有 14% 的优化 trial 最终失败**：92% 的 trial 中虽有临时性能下降但最终超越初始策略
- **Game of 24 案例**：Agent 自主从 LLM 推理切换到搜索算法，达到 100% 准确率——完全突破了初始方法论

## 亮点与洞察
- **"修改修改自身的能力"**——真正的元递归：不同于元学习只优化策略，Gödel Agent 能修改优化器本身，理论上可以无限逼近最优
- **Monkey Patching 的妙用**：将"自我修改"这个抽象概念用 Python 运行时特性优雅实现，工程上简洁可行
- **无约束模式的启示**：Agent 自主决定调用更强模型是"聪明"的策略，暗示未来 Agent 需要自主管理计算资源

## 局限与展望
- 受限于当前 LLM 的代码生成能力——Agent 难以发明超越 SOTA 的全新算法（如从 CoT 出发无法超越 ToT）
- 4% 的 trial 意外终止（通常因修改了递归改进模块本身导致无法继续）
- 无约束模式的性能提升主要来自调用更强 LLM，而非算法创新
- 安全性问题：自修改 Agent 可能引入不可预测行为
- 仅在 GPT-3.5 上测试，更强基础模型上的自改进空间未知

## 相关工作与启发
- **vs Meta Agent Search**: MAS 用固定的元搜索算法优化 Agent 模块。Gödel Agent 连元搜索算法都可以自我修改，搜索空间更大
- **vs Self-Refine/Reflexion**: 它们只能修改输出，不能修改自己的推理逻辑
- **vs Gödel Machine (Schmidhuber 2003)**: 理论先驱，但原版需要形式化证明才能自修改。Gödel Agent 用 LLM 的启发式能力替代了严格证明

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个完全自引用的 LLM Agent 框架，概念突破性强
- 实验充分度: ⭐⭐⭐⭐ 4 个benchmark + 详细消融 + Game of 24 案例分析
- 写作质量: ⭐⭐⭐⭐⭐ 理论形式化清晰，与 Gödel Machine 的类比恰当
- 价值: ⭐⭐⭐⭐⭐ 为 Agent 自改进开辟了全新方向

<!-- RELATED:START -->

## 相关论文

- [SMART: Self-Aware Agent for Tool Overuse Mitigation](smart_self-aware_agent_for_tool_overuse_mitigation.md)
- [Agentic Knowledgeable Self-Awareness](agentic_knowledgeable_self-awareness.md)
- [Self-Taught Agentic Long-Context Understanding](self_taught_agentic_long_ctx.md)
- [InfiAgent: Self-Evolving Pyramid Agent Framework for Infinite Scenarios](../../ICLR2026/llm_agent/infiagent_self-evolving_pyramid_agent_framework_for_infinite_scenarios.md)
- [Bel Esprit: Multi-Agent Framework for Building AI Model Pipelines](bel_esprit_multi-agent_framework_for_building_ai_model_pipelines.md)

<!-- RELATED:END -->
