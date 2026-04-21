---
title: >-
  [论文解读] ControlLLM: Augment Language Models with Tools by Searching on Graphs
description: >-
  [ECCV2024][语音][tool-augmented LLM] 提出 ControlLLM 框架，通过在预构建的工具图（Tool Graph）上进行图搜索（Thoughts-on-Graph）来规划多模态工具调用，显著提升了复杂任务中工具选择和参数赋值的准确性。
tags:
  - ECCV2024
  - 语音
  - tool-augmented LLM
  - 多模态
  - graph search
  - task planning
  - Thoughts-on-Graph
---

# ControlLLM: Augment Language Models with Tools by Searching on Graphs

**会议**: ECCV2024  
**arXiv**: [2310.17796](https://arxiv.org/abs/2310.17796)  
**代码**: [OpenGVLab/ControlLLM](https://github.com/OpenGVLab/ControlLLM)  
**领域**: audio_speech  
**关键词**: tool-augmented LLM, multi-modal interaction, graph search, task planning, Thoughts-on-Graph

## 一句话总结

提出 ControlLLM 框架，通过在预构建的工具图（Tool Graph）上进行图搜索（Thoughts-on-Graph）来规划多模态工具调用，显著提升了复杂任务中工具选择和参数赋值的准确性。

## 背景与动机

大语言模型（LLM）已展现出强大的语言理解与生成能力，研究者开始探索用 LLM 作为"大脑"来调度外部工具（图像生成、音频处理、视频编辑等），实现多模态交互。代表性工作包括 HuggingGPT、Visual ChatGPT、InternGPT 等。

然而，现有方法面临三大挑战：

1. **任务分解模糊**：用户输入的自然语言指令往往含糊不清，现有方法难以准确拆解为可执行子任务
2. **工具选择与参数赋值不准确**：LLM 依赖 Chain-of-Thought（CoT）或 Tree-of-Thoughts（ToT）来做工具调用规划，容易产生幻觉（hallucination），导致选错工具或参数类型不匹配
3. **工具调度效率低**：复杂任务涉及多个工具间存在复杂拓扑依赖关系，链式或树式思维范式无法有效表达这种拓扑结构

核心观察：工具之间的输入输出存在天然的拓扑关系——一个工具的输出类型恰好可作为另一工具的输入，这种关系可以用图来建模，从而将任务规划转化为图上的路径搜索问题。

## 核心问题

如何让 LLM 在多模态工具调用场景中克服幻觉问题，准确完成工具选择、参数赋值和调度执行？

## 方法详解

ControlLLM 由三个阶段组成：任务分解 → 任务规划（ToG）→ 方案执行。

### 1. 任务分解（Task Decomposition）

利用语言模型 $\mathcal{M}$（ChatGPT 或微调后的 LLaMA）将用户请求 $r$ 分解为若干并行子任务：

$$\{s_0, \ldots, s_n\} = \mathcal{M}(r)$$

每个子任务包含结构化属性：任务描述、任务领域（domain）、输入参数（args）和输出类型（returns），以 JSON 格式输出。此阶段**不涉及**工具选择，仅做请求拆分和输入输出类型推断，为后续图搜索确定起点和终点。

### 2. 任务规划：Thoughts-on-Graph（ToG）

**工具图构建**：预先构建一个工具图 $G$，包含两类节点：

- **Resource 节点**：表示资源类型（image、mask、video、audio 等），定义为 $\langle\text{type}\rangle$
- **Tool 节点**：表示工具，定义为 $\langle\text{desc, args, ret}\rangle$，分别是功能描述、输入参数列表和返回类型

图中有两类边：

- **Tool → Resource 边**：工具连接到其输出资源类型
- **Resource → Tool 边**：资源类型连接到以其作为输入的工具

**图上搜索**：基于深度优先搜索（DFS）算法，从子任务的输入资源节点出发，沿图探索所有可行路径，直到到达期望的输出节点或超过最大长度限制（默认 $m=10$）。搜索过程中，利用语言模型对候选工具进行评分过滤，提供四种搜索策略：

| 策略 | 特点 |
|------|------|
| Greedy | 每步选评分最高的工具，快但可能找不到最优解 |
| Beam ($k=3$) | 保留评分前 $k$ 个工具，扩大搜索空间 |
| Adaptive | 动态调整 beam 大小，选超过阈值的工具，兼顾探索与效率 |
| Exhaustive | 遍历所有路径，保证最优但耗时极大 |

**后处理**：搜索完成后，由 Solution Expert 从所有候选方案中选出最优方案，Resource Expert 推断工具的剩余参数。

### 3. 方案执行（Solution Execution）

执行引擎将方案解析为 Action 序列，根据拓扑结构可并行调度多个独立子任务。支持本地部署、远程云服务或混合端点。维护一个状态内存存储所有中间结果，支持运行时自动参数修正。最后由 LLM 汇总执行结果，生成用户友好的回复。

### 语言模型选择

框架提供三种变体：

- **ControlLLM-ChatGPT**：全程使用 ChatGPT-3.5，零训练成本但性能有限
- **ControlLLM-LLaMA**：微调 LLaMA-7B 做全部模块，性能高但需 GPU 训练
- **ControlLLM-Mix**（默认）：微调 LLaMA-7B 做任务分解，其余模块用 ChatGPT，兼顾性能与成本

## 实验关键数据

### 评估指标

- **IR**（Irrelevant Tool Inclusion Rate）↓：不相关工具的引入率
- **NR**（Necessary Tool Inclusion Rate）↑：必要工具的包含率
- **HR**（Resource Hallucination Rate）↓：参数幻觉率
- **CR**（Resource Type Consistency Rate）↑：参数类型一致率
- **SE**（Solution Evaluation）↑：整体方案成功率

### 主要结果（ControlLLM-Mix vs 最强基线 HuggingGPT）

| 指标 | ControlLLM-Mix | HuggingGPT |
|------|---------------|------------|
| IR ↓ | **0.03** | 0.45 |
| NR ↑ | 0.93 | 0.64 |
| HR ↓ | **0.02** | 0.16 |
| CR ↑ | **0.98** | 0.69 |
| SE (All) ↑ | **0.93** | 0.59 |
| SE (Hard) ↑ | **0.81** | 0.33 |

在困难任务（>3 个 API）上，ControlLLM 达到 81% 成功率，最强基线仅 33%。

### 搜索策略消融

Adaptive 策略在性能和效率间取得最佳平衡：平均访问 236 个工具节点，SE=0.93；而 Exhaustive 需访问 3444 个节点才达到 SE=0.97。Greedy 最快但 SE 仅 0.49。

### 语言模型消融

加入先验知识（Prior Knowledge）后，各模型性能均显著提升。GPT-4 + PK 达到 SE=0.98，即使 LLaMA2-13B + PK 也能达到 SE=0.82。

## 亮点

1. **ToG 范式的创新性**：将工具调用规划从"让 LLM 生成方案"转变为"在预构建图上搜索方案"，从根本上规避了 LLM 的幻觉问题
2. **高可扩展性**：新增工具只需更新工具图，无需重新训练 LLM 或修改 prompt
3. **多方案输出**：图搜索天然产出多条可行路径，可为用户提供备选方案
4. **突破 token 限制**：方案搜索在图上进行，不受 LLM 上下文窗口长度限制
5. **全模态覆盖**：支持文本、图像、音频、视频的处理与交互

## 局限与展望

1. **工具输出质量不受控**：框架保证方案理论可行，但无法保证工具实际产出符合用户预期
2. **自然语言歧义**：用户意图的内在模糊性可能导致选中的"最优"方案与真实期望不一致
3. **搜索开销**：Adaptive 策略仍需访问约 236 个节点，实时交互场景下延迟较高
4. **工具图维护成本**：随工具数量增长，图的构建和维护复杂度上升
5. **评测局限**：benchmark 仅约 100 条指令，规模较小；评估依赖人工投票，可重复性有待提高

## 与相关工作的对比

| 方法 | 规划范式 | 多模态覆盖 | 多方案 | SE (Hard) |
|------|---------|-----------|--------|-----------|
| HuggingGPT | CoT | 图/视/音 | ✗ | 0.33 |
| Visual ChatGPT | CoT | 仅图像 | ✗ | 0.10 |
| InternGPT | CoT | 图/视 | ✗ | 0.00 |
| GPT4Tools | CoT/指令微调 | 仅图像 | ✗ | 0.00 |
| **ControlLLM** | **ToG（图搜索）** | **图/视/音** | **✓** | **0.81** |

核心差异在于：CoT/ToT 依赖 LLM 在运行时动态生成思维链/树，容易幻觉；ToG 在预构建的工具依赖图上搜索，避免了运行时幻觉，且能处理复杂拓扑结构的方案。

## 启发与关联

1. **图结构建模工具关系**的思路可推广到 agent 系统设计：将 agent 的能力图谱用图表达，自动搜索多 agent 协作方案
2. **任务分解与规划解耦**的设计思路值得借鉴——先用 LLM 做语义理解和分解，再用算法做确定性搜索，各取所长
3. 与 ToolLLM（DFSDT）的对比表明，将工具关系**显式建模**为图优于让 LLM **隐式推理**工具依赖
4. 可以考虑将 ToG 与 RAG 结合，用检索增强来动态扩展工具图
5. 本文的评估指标体系（IR/NR/HR/CR/SE）可作为 tool-use benchmark 的参考

## 评分
- 新颖性: ⭐⭐⭐⭐ (ToG 范式将任务规划从 LLM 生成转为图搜索，思路新颖)
- 实验充分度: ⭐⭐⭐⭐ (设计了完整指标体系，多维度消融，但 benchmark 规模偏小)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，图示直观，动机阐述充分)
- 价值: ⭐⭐⭐⭐ (对 tool-augmented LLM 领域有实质推进，对 agent 系统设计有参考价值)

<!-- RELATED:START -->

## 相关论文

- [Label-Anticipated Event Disentanglement for Audio-Visual Video Parsing](label-anticipated_event_disentanglement_for_audio-visual_video_parsing.md)
- [EDTalk: Efficient Disentanglement for Emotional Talking Head Synthesis](edtalk_efficient_disentanglement_for_emotional_talking_head_synthesis.md)
- [Latent-INR: A Flexible Framework for Implicit Representations of Videos with Discriminative Semantics](latent-inr_a_flexible_framework_for_implicit_representations_of_videos_with_disc.md)
- [CoLeaF: A Contrastive-Collaborative Learning Framework for Weakly Supervised Audio-Visual Video Parsing](coleaf_a_contrastive-collaborative_learning_framework_for_weakly_supervised_audi.md)
- [Listen to Look into the Future: Audio-Visual Egocentric Gaze Anticipation](listen_to_look_into_the_future_audio-visual_egocentric_gaze_anticipation.md)

<!-- RELATED:END -->
