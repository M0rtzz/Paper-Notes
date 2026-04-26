---
title: >-
  [论文解读] Precise Debugging Benchmark: Is Your Model Debugging or Regenerating?
description: >-
  [ACL 2026][代码调试] 本文揭示前沿 LLM 在调试任务中的"重生成"倾向——通过引入 PDB 框架和编辑级精度/bug 级召回指标，发现 GPT-5.1-Codex 等模型虽能通过 76% 以上单元测试，但编辑精度不足 45%，且迭代和 agent 调试策略也无法显著改善精度。
tags:
  - ACL 2026
  - 代码调试
  - LLM编程
  - 精确编辑
  - 基准测试
  - 代码重生成
---

# Precise Debugging Benchmark: Is Your Model Debugging or Regenerating?

**会议**: ACL 2026  
**arXiv**: [2604.17338](https://arxiv.org/abs/2604.17338)  
**代码**: [GitHub](https://github.com)  
**领域**: 代码智能 / 调试评估  
**关键词**: 代码调试, LLM编程, 精确编辑, 基准测试, 代码重生成

## 一句话总结

本文揭示前沿 LLM 在调试任务中的"重生成"倾向——通过引入 PDB 框架和编辑级精度/bug 级召回指标，发现 GPT-5.1-Codex 等模型虽能通过 76% 以上单元测试，但编辑精度不足 45%，且迭代和 agent 调试策略也无法显著改善精度。

## 研究背景与动机

**领域现状**：LLM 在代码生成领域取得了巨大成功，能从自然语言描述合成复杂算法。然而，真实软件开发中的主要工作不是从零生成，而是调试和维护。

**现有痛点**：(1) 当给 LLM 有 bug 的代码时，模型往往重写大部分甚至全部代码来"修复"——这虽然能通过测试，但在现实代码库中代价高昂、风险大、难以 review；(2) 现有调试基准仅用单元测试通过率评估，无法区分精准修复和大规模重写——重写整个函数和修改一行 bug 得到相同分数；(3) 对于多 bug 程序，仅修复了部分 bug 的模型与完全未修复的模型得到相同的零分。

**核心矛盾**：单元测试通过率与调试精度之间存在负相关——模型越是激进地重写代码，越可能通过测试（功能正确），但编辑精度越低。现有评估体系奖励重生成行为，无法激励精准调试。

**本文目标**：(1) 设计一个能区分"精准调试"和"代码重生成"的评估框架；(2) 量化当前前沿模型距离精准调试有多远；(3) 评估迭代和 agent 调试策略是否改善精度。

**切入角度**：定义"编辑级精度"和"bug 级召回"两个新指标——精度衡量模型修改中有多少是必要的，召回衡量有多少 bug 被正确修复。通过自动注入验证过的原子 bug 并组合为多 bug 程序，构建有 ground-truth 编辑脚本的调试基准。

**核心 idea**：将调试评估从程序级（通过/不通过）下沉到编辑级（哪些修改是必要的、哪些是多余的），通过原子 bug 合成和独立性验证构建精确的评估基准。

## 方法详解

### 整体框架

PDB 框架分两阶段：**生成阶段** 从现有编程数据集出发，用 LLM 合成验证过的原子 bug 并组合为多 bug 程序；**评估阶段** 让调试系统修复 buggy 程序，用编辑级精度和 bug 级召回评估。

### 关键设计

1. **原子 Bug 合成与组合**:

    - 功能：生成有 ground-truth 编辑脚本的 buggy 程序，支持单行和多行 bug
    - 核心思路：对每个 ground-truth 程序，按 ODC（正交缺陷分类）的 5 个类别（赋值、检查、算法、构建/打包、时序），随机选择操作类型（插入/删除/替换）和可编辑行，用 LLM 注入单行 bug。通过单元测试验证 bug 有效性（必须失败）。多 bug 程序通过组合多个独立的原子 bug 构建，要求 bug 之间有最小间距（stride）且满足独立性约束
    - 设计动机：保证原子性（修复不能通过仅修改 bug 的子集来完成）和独立性（修复一个 bug 不影响其他 bug 的修复），这是精确定义编辑级精度和 bug 级召回的前提

2. **编辑级精度 (Edit-Level Precision)**:

    - 功能：衡量模型修改中有多少比例是必要的
    - 核心思路：$\text{precision}_\epsilon = \frac{1}{|\hat{E}|} \sum_{i=1}^k F_\mathcal{U}(\hat{C}_i) \cdot (|\hat{E}_i|)_\epsilon$。用 map 函数将 ground-truth 编辑与预测编辑对应，用 essential 函数搜索最小必要编辑子集，引入容差 $\epsilon$ 允许一定的编辑冗余
    - 设计动机：传统单元测试通过率无法惩罚多余修改。精度指标将评估下沉到行级别，直接衡量"这个修改是必要的吗"

3. **Bug 级召回 (Bug-Level Recall)**:

    - 功能：衡量有多少 bug 被正确修复
    - 核心思路：$\text{recall} = \frac{1}{k} \sum_{i=1}^k F_\mathcal{U}(\hat{C}_i)$。对每个 bug $i$，构建伪修正版本——保留所有其他 bug 的 ground-truth 修复，只用模型对 bug $i$ 的修改，检查是否通过单元测试
    - 设计动机：在多 bug 场景中，仅修复部分 bug 也应获得部分分数，而非全有或全无

### 损失函数 / 训练策略

PDB 不涉及模型训练。评估使用 PDB-Single-Hard（5,751 个单行 bug 样本）和 PDB-Multi（256 个多行 bug 样本），从 BigCodeBench 和 LiveCodeBench 构建。Bug 生成器池包含 GPT-5.1-Codex、Claude-4.5-Sonnet 和 Gemini-2.5-Pro。

## 实验关键数据

### 主实验

| 模型 | 精度 | 召回 | 单元测试 (%) |
|------|------|------|-------------|
| Claude-Sonnet-4.5 | **71.8** | **81.4** | 75.7 |
| Gemini-2.5-Pro | 71.4 | 83.5 | 78.1 |
| Qwen3-Coder-480B | 65.8 | 77.2 | 70.3 |
| DeepSeek-V3.2 | 48.4 | 70.0 | 71.4 |
| DeepSeek-V3.2-Thinking | 45.0 | 71.2 | **79.0** |
| GPT-5.1-Codex | 39.7 | 71.7 | 76.1 |

### 消融实验

| 分析维度 | 结果 |
|----------|------|
| 自由提示 vs 最小编辑提示 | 自由提示下所有模型精度暴跌，Gemini 下降 40 个绝对点 |
| 迭代调试（3 轮） | 提升测试通过率和召回，但精度不变或下降 |
| Agent 调试（含测试反馈） | Claude-Code 精度仍仅 50%，额外反馈反而加剧重生成 |
| Bug 数量影响 | Bug 越多，精度越低（更多多余修改），召回与数据集相关 |

### 关键发现

- **排名反转**：GPT-5.1-Codex 单元测试通过率 76.1% 排名靠前，但精度 39.7% 排末位——它是最严重的"重生成者"
- Qwen3-Coder-480B 虽然通过率较低（70.3%），但精度高达 65.8%——"弱但精准"
- 模型调试行为可分为四类：精准通过型、弱但精准型、弱但能定位型、通过导向型（重生成）
- 迭代和 agent 策略改善功能正确性但不改善精度——当前方法通过扩大修改范围来修 bug，而非精确定位
- 约 1.65% 的案例存在 bug 交互，PDB 的独立性假设在绝大多数情况下成立

## 亮点与洞察

- "调试还是重生成？"这个问题切中了当前代码 LLM 的核心痛点——揭示了单元测试评估的根本缺陷
- 编辑级精度和 bug 级召回的定义精确且有实践意义——可直接用于改进后训练流程
- GPT-5.1-Codex 精度仅 39.7% 这一发现非常震撼——最强模型反而最不精准，说明后训练流程可能在强化重生成行为

## 局限与展望

- 假设 bug 独立性在现实软件中常常不成立——交互 bug 是调试的真正难点
- 仅评估 Python，其他语言的适用性需验证
- 语义等价但形式不同的修复可能被错误惩罚
- 未探索如何改进后训练流程以提升精度——这是最有价值的后续方向

## 相关工作与启发

- **vs DebugBench**: DebugBench 从历史提交挖掘 bug，但仅用单元测试评估，无法衡量精度；PDB 引入编辑级评估填补了这一空白
- **vs SWE-bench**: SWE-bench 关注真实 repo 级别的 bug 修复，涉及更复杂的定位，但同样缺乏精度评估；两者互补
- **vs APR (自动程序修复)**: 传统 APR 关注最小修复，PDB 将这一理念引入 LLM 评估

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 提出调试评估的范式转变——从程序级到编辑级，发现极有冲击力
- 实验充分度: ⭐⭐⭐⭐⭐ 9个前沿模型、迭代/agent/多行/分类分析，手工验证指标准确性
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义精确，形式化严谨，实验分析深入
- 价值: ⭐⭐⭐⭐⭐ 直接揭示了代码 LLM 后训练的根本问题，对社区有重要启示

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] MLDebugging: Towards Benchmarking Code Debugging Across Multi-Library Scenarios](../../ACL2025/code_intelligence/mldebugging_towards_benchmarking_code_debugging_across_multi-library_scenarios.md)
- [\[ACL 2025\] Revisit Self-Debugging with Self-Generated Tests for Code Generation](../../ACL2025/code_intelligence/revisit_self-debugging_with_self-generated_tests_for_code_generation.md)
- [\[ACL 2026\] QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization](qimeng-prepair_precise_code_repair_via_edit-aware_reward_optimization.md)
- [\[ACL 2026\] River-LLM: Large Language Model Seamless Exit Based on KV Share](river-llm_large_language_model_seamless_exit_based_on_kv_share.md)
- [\[ACL 2026\] From Charts to Code: A Hierarchical Benchmark for Multimodal Models](from_charts_to_code_a_hierarchical_benchmark_for_multimodal_models.md)

<!-- RELATED:END -->
