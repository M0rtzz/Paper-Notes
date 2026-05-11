---
title: >-
  [论文解读] Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models
description: >-
  [ACL 2026][LLM/NLP][小语言模型] 本文提出 PA-Tool，一种无训练的工具 Schema 优化方法，利用从数据污染检测中借鉴的"尖锐度"（peakedness）信号识别模型预训练中熟悉的命名模式，通过重命名工具组件来对齐小语言模型的内化知识，在 MetaTool 和 RoTBench 上实现最高 17% 的提升，Schema 不对齐错误减少 80%。
tags:
  - ACL 2026
  - LLM/NLP
  - 小语言模型
  - 工具调用
  - Schema对齐
  - 预训练知识
  - 无训练优化
---

# Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models

**会议**: ACL 2026  
**arXiv**: [2510.07248](https://arxiv.org/abs/2510.07248)  
**代码**: [GitHub](https://github.com/holi-lab/PA-Tool)  
**领域**: LLM/NLP  
**关键词**: 小语言模型, 工具调用, Schema对齐, 预训练知识, 无训练优化

## 一句话总结

本文提出 PA-Tool，一种无训练的工具 Schema 优化方法，利用从数据污染检测中借鉴的"尖锐度"（peakedness）信号识别模型预训练中熟悉的命名模式，通过重命名工具组件来对齐小语言模型的内化知识，在 MetaTool 和 RoTBench 上实现最高 17% 的提升，Schema 不对齐错误减少 80%。

## 研究背景与动机

**领域现状**：工具增强的语言模型已成为现代 AI 系统的核心组件。随着多智能体架构的发展，越来越多的场景需要部署小语言模型（SLM，通常 ≤8B）来处理子任务，包括工具选择（识别正确 API）和参数识别（提供正确参数）。

**现有痛点**：SLM 在工具使用任务上表现远逊于大模型，一个常见的失败模式是"Schema 不对齐"（schema misalignment）：即使上下文中提供了正确的工具，模型仍会幻觉出看起来合理但不存在的工具名。这表明模型在面对不熟悉的 Schema 时会回退到预训练中内化的命名惯例。

**核心矛盾**：现有方法要么通过训练让模型适应任意 Schema（需要大量数据且可能导致灾难性遗忘），要么通过改进工具文档或交互历史来间接改善（未解决 Schema 层面的根本不匹配）。训练方法代价高且不可扩展，而免训练方法未触及命名不对齐的根源。

**本文目标**：提出一种无训练方法，通过调整工具 Schema 来匹配模型的预训练知识，而非反过来训练模型适应 Schema。

**切入角度**：从数据污染检测领域借鉴"尖锐度"（peakedness）概念——模型在预训练中频繁见过的模式会导致生成高度集中的输出分布。利用这一信号识别模型"熟悉"的命名模式。

**核心 idea**：与其训练小模型适应不熟悉的工具 Schema，不如调整 Schema 来对齐模型的预训练知识——通过生成多个候选名、计算尖锐度、选择最高尖锐度的候选来找到模型最熟悉的命名。

## 方法详解

### 整体框架

PA-Tool 对工具 Schema 中的每个组件（工具名、参数名）执行三阶段重命名过程：(1) 候选生成——让 SLM 根据组件描述多次生成候选名；(2) 尖锐度计算——衡量每个候选名周围有多少相似候选；(3) Schema 选择——选择尖锐度最高的候选作为新名称。对 Schema 中所有组件迭代执行此过程，构建原始名到预训练对齐名的映射字典。

### 关键设计

1. **候选生成（Candidate Generation）**:

    - 功能：探索模型在预训练中可能见过的多样命名模式
    - 核心思路：给定组件描述 $d$，以温度 $t \in (0,1]$ 采样 $N$ 个候选名 $\mathcal{C} = \{s_1, s_2, \ldots, s_N\}$。温度控制的采样超越单一贪心路径，揭示模型学到的多样命名模式。同时用贪心解码 ($t=0$) 生成参考名 $s_{\text{ref}}$ 用于平局打破
    - 设计动机：贪心解码只生成单一候选，限制了 Schema 空间探索（实验证明 Greedy 有时反而劣于 Base）

2. **尖锐度计算（Peakedness Computation）**:

    - 功能：识别模型预训练中频繁见过的强记忆模式
    - 核心思路：对每个候选 $s_i$，计算其尖锐度分数 $\phi(s_i) = \sum_{j \neq i} \mathbb{I}(d_{\text{edit}}(s_i, s_j) \leq \tau)$，其中 $d_{\text{edit}}$ 为字符级编辑距离，阈值 $\tau = \alpha \cdot \ell_{\max}$（$\ell_{\max}$ 为最大候选长度，$\alpha$ 控制相似度严格程度）。尖锐度高意味着模型对该命名模式生成了高度集中的分布
    - 设计动机：受 CDD 数据污染检测方法启发——训练中频繁出现的模式在多次采样中会产生集中的输出分布。长度自适应阈值确保不同长度的名称有一致的相似性标准

3. **Schema 选择与冲突解决**:

    - 功能：选出最佳预训练对齐名并处理跨工具冲突
    - 核心思路：选择尖锐度最高的候选 $s^* = \arg\max_{s_i \in \mathcal{C}} \phi(s_i)$。平局时选与参考名编辑距离最小的：$s^* = \arg\min_{s_i \in \mathcal{C}^*} d_{\text{edit}}(s_i, s_{\text{ref}})$。当不同工具因描述相似产生名称冲突时，使用优先级锁定机制解决
    - 设计动机：尖锐度最高的候选代表模型最深层内化的命名惯例。一次性 Schema 映射后可重复使用，无需每次推理重新生成

### 损失函数 / 训练策略

PA-Tool 完全无训练。仅需一次性的 Schema 映射：使用 32 个候选，温度 0.4，$\alpha = 0.2$。推理时使用温度 0 确保可复现。映射字典在工具集不变时可重复使用，无需模型修改、重训练或灾难性遗忘风险。

## 实验关键数据

### 主实验

**MetaTool 工具选择（准确率 %）**

| 模型 | 方法 | Similar | Scenario | Reliability | Multi-tool |
|------|------|------|------|------|------|
| Qwen2.5-7B | Base | 59.6 | 74.4 | 78.3 | 78.3 |
| Qwen2.5-7B | PA-Tool | 64.1 | 78.4 | **88.2** | 84.9 |
| Llama3.1-8B | Base | 61.5 | 73.9 | 53.5 | 78.7 |
| Llama3.1-8B | PA-Tool | **70.4** | **79.9** | 66.0 | **88.3** |
| Llama3.2-3B | Base | 55.0 | 58.6 | 43.6 | 79.1 |
| Llama3.2-3B | PA-Tool | 65.7 | 67.7 | 60.6 | 80.5 |

**RoTBench 工具选择与参数识别**

| 模型 | 方法 | 单轮工具选择 | 单轮参数识别 | 多轮工具选择 | 多轮参数识别 |
|------|------|------|------|------|------|
| Llama3.1-8B | Base | 58.1 | 17.1 | 42.8 | 34.3 |
| Llama3.1-8B | PA-Tool | **68.6** | 18.1 | **48.6** | **35.7** |

### 消融实验

| 配置 | 单轮工具选择 | 单轮参数识别 | 说明 |
|------|------|------|------|
| Base | 58.1 | 17.1 | 无对齐 |
| Tool-only | 62.9 | 14.3 | 仅工具名对齐 |
| Param-only | 56.2 | 17.1 | 仅参数名对齐 |
| Both (PA-Tool) | **68.6** | **18.1** | 联合对齐最佳 |

**错误类型分析（Llama3.1-8B, MetaTool）**

| 错误类型 | Base | PA-Tool | 减少 |
|------|------|------|------|
| Schema 不对齐错误 | — | — | **-80.0%** |
| 功能混淆错误 | — | — | -24.0% |
| 上下文理解错误 | — | — | -18.8% |

### 关键发现

- PA-Tool 在 Reliability 子任务上提升最大（最高 17%），该任务要求模型识别没有合适工具的情况
- Multi-tool 子任务提升达 9.6%（Llama3.1-8B: 78.7→88.3%），因为 Schema 不对齐在多工具组合选择时会累积
- PA-Tool 单独使用可在多个子任务上超越有监督微调模型，且两者结合时可进一步提升
- 尖锐度验证实验证实：随着训练轮次增加，模型的尖锐度一致上升（最高 +25.8%），支持其作为熟悉度信号的假设
- 在 API-Bank 和 τ-Bench 端到端基准上也展现了一致的改进

## 亮点与洞察

- 逆向思维极具启发性："不要让模型适应工具，而是让工具适应模型"——这一思路可推广到其他模型-接口交互场景
- 从数据污染检测到工具 Schema 优化的跨领域迁移：将 peakedness 从"检测污染"转化为"利用预训练知识"，化废为宝
- 一次性映射的设计使其部署成本极低，与微调、检索、约束解码等方法正交且可组合
- PA-Tool 使 Llama3.1-8B 在 Multi-tool 子任务上超越 Claude-Sonnet-4.5（88.3% vs 85.1%），证明 Schema 对齐可以缩小大小模型差距

## 局限与展望

- 依赖模型对组件描述的理解来生成候选名，描述质量差时可能影响效果
- 当前仅重命名工具名和参数名，未考虑 Schema 结构（如参数类型、嵌套结构）的对齐
- 在闭源模型上效果较小（因为大模型的 Schema 不对齐问题本身较轻）
- peakedness 信号在工具名非常短或非常长时可能不够稳定

## 相关工作与启发

- **vs 有监督微调 (SFT)**: SFT 需要训练数据且存在泛化问题（增加数据后 RoTBench 性能反而下降），PA-Tool 无训练且与 SFT 互补
- **vs EasyTool (描述增强)**: EasyTool 改进工具描述但不修改名称，PA-Tool 修改名称但不改描述，两者正交可组合
- **vs 约束解码**: 约束解码消除格式错误但不解决命名偏好不匹配，PA-Tool 从根源解决命名问题

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 逆向思维独特，跨领域迁移 peakedness 信号非常巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ MetaTool + RoTBench + API-Bank + τ-Bench 四个基准，错误分析、尖锐度验证、组件消融、与 SFT/免训练方法组合实验极为全面
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法直观，分析深入
- 价值: ⭐⭐⭐⭐⭐ 提供了一种零成本提升 SLM 工具使用能力的实用方案，对多智能体系统部署有重要价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] ControlLLM: Augment Language Models with Tools by Searching on Graphs](../../ECCV2024/llm_nlp/controlllm_augment_language_models_with_tools_by_searching_on_graphs.md)
- [\[ACL 2025\] PiFi: Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models](../../ACL2025/llm_nlp/plugin_finetuning_bridge.md)
- [\[ICLR 2026\] Predicting LLM Reasoning Performance with Small Proxy Models](../../ICLR2026/llm_nlp/predicting_llm_reasoning_performance_with_small_proxy_models.md)
- [\[CVPR 2026\] Perception Programs: Unlocking Visual Tool Reasoning in Language Models](../../CVPR2026/llm_nlp/perception_programs_visual_tool_reasoning.md)
- [\[NeurIPS 2025\] Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models](../../NeurIPS2025/llm_nlp/nemotron-flash_towards_latency-optimal_hybrid_small_language_models.md)

</div>

<!-- RELATED:END -->
