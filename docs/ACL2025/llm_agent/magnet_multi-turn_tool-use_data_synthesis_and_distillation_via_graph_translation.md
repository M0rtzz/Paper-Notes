---
title: >-
  [论文解读] Magnet: Multi-turn Tool-use Data Synthesis and Distillation via Graph Translation
description: >-
  [ACL 2025][模型压缩][Function Calling] 提出 Magnet 框架，基于函数依赖图的随机游走和节点操作（Insert/Merge/Split）构建高质量多轮 Function Calling 训练轨迹，结合基于提示的上下文蒸馏生成正负对比轨迹进行 SFT + mDPO 训练，使 14B 模型 Magnet-14B-mDPO 在 BFCL-v3 上达到 68.01（排名第 4），在多轮场景上大幅超越教师模型 Gemini-1.5-pro-002。
tags:
  - ACL 2025
  - 模型压缩
  - Function Calling
  - 多轮对话
  - 图翻译
  - 数据合成
  - 上下文蒸馏
  - DPO
  - 偏好优化
---

# Magnet: Multi-turn Tool-use Data Synthesis and Distillation via Graph Translation

**会议**: ACL 2025  
**arXiv**: [2503.07826](https://arxiv.org/abs/2503.07826)  
**代码**: 未公开  
**作者**: Fan Yin, Zifeng Wang, I-Hung Hsu, Jun Yan, Ke Jiang, Yanfei Chen, Jindong Gu, Long T. Le, Kai-Wei Chang, Chen-Yu Lee, Hamid Palangi, Tomas Pfister
**机构**: Google, UCLA
**领域**: 模型蒸馏 / 工具使用 Agent  
**关键词**: Function Calling, 多轮对话, 图翻译, 数据合成, 上下文蒸馏, DPO, 偏好优化

## 一句话总结

提出 Magnet 框架，基于函数依赖图的随机游走和节点操作（Insert/Merge/Split）构建高质量多轮 Function Calling 训练轨迹，结合基于提示的上下文蒸馏生成正负对比轨迹进行 SFT + mDPO 训练，使 14B 模型 Magnet-14B-mDPO 在 BFCL-v3 上达到 68.01（排名第 4），在多轮场景上大幅超越教师模型 Gemini-1.5-pro-002。

## 研究背景与动机

**领域现状**：LLM Agent 需要调用外部工具（API/函数）来完成复杂任务。当前模型在单步 FC 上表现不错，但在 **多轮、多步骤**交互中仍面临挑战。

**多轮 FC 的三大挑战**：
   - **嵌套 FC（Nested FCs）**：某些轮次需要多个甚至嵌套的函数调用，但查询中并未显式提及
   - **长依赖（Long Dependency）**：某些轮次需要使用对话历史中远处的信息来组装 FC
   - **无关性（Irrelevance）**：某些轮次的功能或参数缺失，需要模型提出澄清问题

**数据瓶颈**：BFCL-v3 上最佳开源模型多轮成功率仅 ~10%，现有模型在多轮场景上的低性能导致难以收集高质量训练轨迹

**核心动机**：设计一个有原则的数据合成管道，从图的视角构建可靠的多轮 FC 训练数据

## 方法详解

### 整体流程

四阶段管道：**函数池与依赖图构建** → **节点操作增强** → **来回翻译生成查询-FC 对** → **上下文蒸馏生成正负轨迹**

### 阶段一：函数依赖图构建

1. **函数收集**：从 StableToolBench 和 BFCL-v3 收集 5,011 个可执行 API，涵盖 49 个类别
2. **局部依赖图**：将函数作为图节点，为每个节点从同类别中采样 30 个邻居候选，用 LLM 判断是否存在输入-输出依赖关系。若源节点的输出与目标节点的输入相关，则建立有向边
3. **初始 FSP 采样**：从每个节点出发进行 S=7 步随机游走，沿依赖边采样，形成初始函数签名路径 $\tilde{\phi} = (\tilde{f}_1, \tilde{f}_2, \cdots, \tilde{f}_H)$

### 阶段二：节点操作增强

设计三种图级操作来覆盖多轮 FC 的三大挑战：

**操作 1: Insert（插入）—— 解决嵌套 FC 和长依赖**
- 遍历 FSP 中每个轮次的最后一个函数签名 $\tilde{f}_{hk}$，用 LLM 检查是否有邻居函数满足嵌套调用条件
- 若存在，将嵌套函数追加到当前轮或插入到后续随机位置（模拟长依赖）
- 示例：查询"从旧金山到圣马特奥多少**公里**"→ 需调用 `get_distance()`（返回英里）+ `convert_unit()`（隐式嵌套）

**操作 2: Merge（合并）—— 创建单轮多 FC**
- 以概率 p=0.3 将相邻两轮合并为一轮，模型需理解前一函数的输出来组装后续函数
- 与 Insert 的区别：Merge 合并的函数相关但不一定嵌套

**操作 3: Split（分裂）—— 模拟缺失信息**
- 随机选择一轮，在其后插入空节点 {}，标记为 'miss params' 或 'miss func'
- 模型需识别信息缺失并提出澄清问题

**执行顺序**：先 Merge → 再 Insert → 生成增强 FSP ϕ；额外对 ϕ 使用 Split 生成带缺失信息的 FSP ϕ̂

### 阶段三：来回翻译（Back-and-Forth Translation）

迭代地将增强 FSP 转换为查询-FC 对：
- **反向翻译** $\mathcal{M}_b(f_h) = q_h$：将函数签名转换为模拟用户查询
- **正向翻译** $\mathcal{M}_f(q_h, f_h, t_{h-1}) = fc_h$：将查询转换为可执行 FC（利用上一轮输出 $t_{h-1}$）
- 逐轮迭代确保前轮输出就绪后再传递给后续轮次

### 阶段四：上下文蒸馏生成轨迹

**正轨迹生成**：
- 受上下文蒸馏（Context Distillation）启发，在教师模型（Gemini-1.5-pro-002）生成轨迹时，将 FC 引用作为 [Hint] 添加到查询后
- 确保教师模型尽可能准确地生成动作

**负轨迹生成**：
- 收集 SFT 模型在每条数据上的 10 条推理轨迹
- 用 LLM 评判每轮是否包含错误 FC
- 将错误 FC 作为误导性 [Hint] 提供给 SFT 模型重新生成负轨迹
- 形成正负轨迹对用于 mDPO 训练

### 训练损失

$$\mathcal{L}(x; \tau_w, \tau_l) = \mathcal{L}_{\text{SFT}}(x; \tau_w) + \lambda \cdot \mathcal{L}_{\text{mDPO}}(x; \tau_w, \tau_l)$$

mDPO（multi-turn DPO）将多轮中每轮的动作分别与参考策略对比，而非整体对比。

### 数据统计

| 类别 | SFT 数量 | mDPO 数量 |
|------|----------|-----------|
| 单轮 | 20,000 | 1,556 |
| 多轮 | 7,800 | 2,250 |
| 无关类 | 6,200 | 750 |
| 多轮平均轮数 | 4.71 | 5.22 |
| 多轮平均 FC 数 | 15.13 | 14.98 |

总训练集 38,556 条，仅约为 APIGen (60K)、Hammer (67.5K) 的一半。

## 实验

### BFCL-v3 主结果

| 模型 | Overall | 单轮 | 多轮 | 无关性 |
|------|---------|------|------|--------|
| watt-tool-70B | 74.31 | — | 58.75 | 76.32 |
| GPT-4o (Prompt) | 72.08 | — | 47.62 | 83.76 |
| GPT-4o (FC) | 69.58 | — | 41.00 | 83.15 |
| o1 | 66.73 | — | 28.25 | 89.62 |
| Gemini-1.5-pro-002 | 62.19 | — | 20.75 | 78.15 |
| **Magnet-14B-mDPO** | **68.01** | — | **37.88** | 84.78 |
| Magnet-14B-SFT | 66.83 | — | 33.38 | 82.59 |
| Magnet-7B-mDPO | 64.64 | — | 27.75 | 78.51 |
| Qwen2.5-Coder-14B (base) | 51.88 | — | 5.38 | 44.58 |
| Qwen2.5-Coder-7B (base) | 53.13 | — | 8.25 | 65.39 |

**关键发现**：
- Magnet-14B-mDPO 在 BFCL-v3 排名 **第 4**，超越 o1 和教师模型 Gemini-1.5-pro-002
- 多轮场景：14B 从基座的 5.38 → **37.88**（+32.5），7B 从 8.25 → 27.75（+19.5）
- 所有 Magnet 模型在多轮场景均超越教师模型 Gemini-1.5-pro-002（20.75）
- mDPO 相比纯 SFT 在多轮上提升约 2.5-4.5 个百分点

### ToolQuery 结果

| 模型 | 成功率 | 进度率 |
|------|--------|--------|
| Magnet-14B-mDPO | **73.3** | **78.7** |
| Gemini-1.5-pro-002 | 68.3 | 74.6 |
| GPT-4o | 63.3 | 80.1 |
| Magnet-7B-mDPO | 67.7 | 73.4 |
| Qwen-Coder-14B | 51.7 | 68.7 |

14B 模型在 ToolQuery 上也达到最佳成功率。

### 消融实验

| 组件 | Overall | 多轮 |
|------|---------|------|
| 初始图（无节点操作） | 58.54 | 12.75 |
| +Merge | 60.83 | 20.63 |
| +Merge+Insert | 64.39 | 29.25 |
| +全部操作（SFT） | **66.83** | **33.38** |
| -正轨迹上下文蒸馏 | 60.26 | 18.88 |
| SFT+mDPO | **68.01** | **37.88** |
| -负轨迹上下文蒸馏 | 67.35 | 36.25 |

**关键发现**：
1. 每个节点操作都带来显著提升：Merge（+7.9 多轮）→ Insert（+8.6）→ Split（+4.1）
2. 正轨迹上下文蒸馏贡献巨大：移除后多轮从 33.38 降至 18.88（-14.5）
3. 负轨迹蒸馏对 mDPO 有贡献但幅度较小（-1.63 多轮）
4. 数据合成可推广到不同基座模型和自训练场景

### 数据源对比

- APIGen+ToolAce 训练的 7B 模型多轮仅 7.13，远低于 Magnet-7B 的 26.50
- 加入无关类数据后整体提升（57.24）但多轮反而微降

## 亮点与洞察

1. **图视角的创新性**：将多轮 FC 挑战抽象为图节点操作（Insert/Merge/Split），这是一种优雅的结构化方法
2. **以少胜多**：仅 38K 训练数据（约为 APIGen/Hammer 的一半）即实现排名第 4 的性能
3. **学生超越教师**：所有 Magnet 模型在多轮场景上均超越教师模型 Gemini-1.5-pro-002，说明合成管道引入了额外监督信号
4. **正向提示蒸馏的巨大贡献**：消融显示移除正轨迹提示后多轮性能几乎减半
5. **多轮 FC 仍是前沿挑战**：即使 Magnet-14B-mDPO 多轮成功率也仅 37.88%，最强模型 watt-tool-70B 也仅 58.75%

## 局限性

1. 依赖教师模型（Gemini-1.5-pro-002）的质量——尽管学生已超越教师，但初始数据质量仍受教师制约
2. 函数池主要来源于 StableToolBench 和 BFCL-v3，API 种类和复杂度可能不完全覆盖真实场景
3. 随机游走采样可能错过某些复杂的函数组合模式
4. 负轨迹生成依赖 SFT 模型自身犯错——若 SFT 模型过弱或过强，负样本质量可能受影响
5. 仅在 Qwen2.5-Coder 系列上验证

## 相关工作

- **FC 评估基准**：BFCL-v3（综合性）、ToolBench / StableToolBench（多步骤）、ToolQuery（多轮多步骤）
- **FC 训练数据合成**：Toolformer（文本中插入 API）→ APIGen/xLAM（格式统一+自动查询生成）→ Hammer（函数遮蔽+无关函数）→ Magnet（图结构+节点操作+对比蒸馏）
- **偏好优化**：DPO → mDPO（多轮扩展）

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性**: ⭐⭐⭐⭐ — 图视角+节点操作+对比蒸馏的组合新颖
- **实验充分性**: ⭐⭐⭐⭐⭐ — 两个基准、多个模型规模、详尽消融、数据源对比
- **写作质量**: ⭐⭐⭐⭐ — 管道说明清晰，符号一致
- **实用价值**: ⭐⭐⭐⭐⭐ — 直接可部署的数据合成管道，效果超越专有模型
- **局限**: 多轮 FC 成功率绝对值仍不高，距离实用仍有差距

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Can a Single Model Master Both Multi-turn Conversations and Tool Use? CoALM: A Unified Conversational Agentic Language Model](can_a_single_model_master_both_multi-turn_conversations_and_tool_use_coalm_a_uni.md)
- [\[ACL 2025\] DICE-Bench: Evaluating the Tool-Use Capabilities of Large Language Models in Multi-Round, Multi-Party Dialogues](dice-bench_evaluating_the_tool-use_capabilities_of_large_language_models_in_mult.md)
- [\[ACL 2025\] ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use](toolhop_multi_hop_tool_use.md)
- [\[NeurIPS 2025\] T1: A Tool-Oriented Conversational Dataset for Multi-Turn Agentic Planning](../../NeurIPS2025/llm_agent/t1_a_tool-oriented_conversational_dataset_for_multi-turn_agentic_planning.md)
- [\[ACL 2026\] When Agents Look the Same: Quantifying Distillation-Induced Similarity in Tool-Use Behaviors](../../ACL2026/llm_agent/when_agents_look_the_same_quantifying_distillation-induced_similarity_in_tool-us.md)

</div>

<!-- RELATED:END -->
