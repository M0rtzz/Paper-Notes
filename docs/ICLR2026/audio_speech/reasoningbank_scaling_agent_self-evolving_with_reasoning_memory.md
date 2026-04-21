---
title: >-
  [论文解读] ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory
description: >-
  [ICLR 2026][语音][Agent记忆] 提出 ReasoningBank 记忆框架，从 Agent 自我判断的成功和失败经验中蒸馏可泛化的推理策略存入记忆库，并提出 memory-aware test-time scaling (MaTTS) 建立记忆与测试时扩展的协同效应，在 WebArena、Mind2Web 和 SWE-Bench 上一致超越基线（最高 34.2% 相对提升），同时减少 16% 交互步数。
tags:
  - ICLR 2026
  - 语音
  - Agent记忆
  - 推理策略
  - 测试时扩展
  - 自进化
  - 经验学习
---

# ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory

**会议**: ICLR 2026  
**arXiv**: [2509.25140](https://arxiv.org/abs/2509.25140)  
**代码**: [google-research/reasoning-bank](https://github.com/google-research/reasoning-bank)  
**领域**: LLM Agent / 记忆系统  
**关键词**: Agent记忆, 推理策略, 测试时扩展, 自进化, 经验学习

## 一句话总结
提出 ReasoningBank 记忆框架，从 Agent 自我判断的成功和失败经验中蒸馏可泛化的推理策略存入记忆库，并提出 memory-aware test-time scaling (MaTTS) 建立记忆与测试时扩展的协同效应，在 WebArena、Mind2Web 和 SWE-Bench 上一致超越基线（最高 34.2% 相对提升），同时减少 16% 交互步数。

## 研究背景与动机
随着 LLM Agent 在持久运行的现实角色中日益普及，它们自然会遇到连续的任务流。然而一个关键限制是它们无法从累积的交互历史中学习——每次面对新任务都从零开始，被迫丢弃有价值的洞察并重复过去的错误。

现有 Agent 记忆方法有两大缺陷：

**只存储原始轨迹或成功套路**：Synapse 存储原始轨迹作为上下文记忆，AWM 从轨迹中抽取工作流程（workflow），但都无法蒸馏更高层次的可迁移推理模式

**忽视失败经验的价值**：过度强调成功经验，导致 Agent 无法从自身的失败中学到教训

核心 idea：将成功和失败经验都蒸馏为可泛化的推理策略（而非具体操作步骤），存入结构化的记忆库；结合 test-time scaling 生成丰富的对比信号，进一步提升记忆质量。

## 方法详解

### 整体框架
ReasoningBank 是一个闭环记忆系统：Agent 接收新任务 → 从 ReasoningBank 检索相关记忆 → 指导决策执行 → 完成后从新经验中构建新记忆 → 合并回 ReasoningBank。整个过程无需真实标签，仅依赖 Agent 自我判断（LLM-as-a-judge）。

### 关键设计

1. **Memory Schema（结构化记忆单元）**: 每个记忆项包含三个部分：

    - **Title**: 简要标识核心策略或推理模式
    - **Description**: 一句话概要
    - **Content**: 蒸馏后的推理步骤、决策依据或操作洞察
   
   设计为人类可理解且机器可用，既高于原始轨迹（抽象共性），又具体到可执行（含推理步骤）。

2. **三步闭环流程**:

    - **Memory Retrieval**: 用 embedding-based 相似度搜索从 ReasoningBank 检索 top-k 相关经验的记忆项，注入 Agent 的 system prompt
    - **Memory Construction**: 任务完成后，用 LLM-as-a-judge 评判轨迹成功/失败。成功轨迹贡献"验证过的策略"，失败轨迹贡献"反事实信号和陷阱警示"。每条轨迹最多提取 3 个记忆项
    - **Memory Consolidation**: 新记忆项直接追加到 ReasoningBank（有意采用最简合并策略以隔离记忆内容的效果）

3. **MaTTS: Memory-Aware Test-Time Scaling**: 
   将 ReasoningBank 与 test-time scaling 结合，建立双向协同：
    - **Parallel Scaling**: 对同一查询生成多条轨迹，通过 self-contrast 比较不同轨迹的成败，识别一致的推理模式并过滤虚假解。提供多样化的对比信号，使记忆更可靠
    - **Sequential Scaling**: 单条轨迹内迭代式 self-refinement，中间的推理尝试、纠正和洞察也被捕获为有价值的记忆信号
   
   **关键区别**: vanilla TTS 独立处理多条轨迹再各自提取记忆（suboptimal）；MaTTS 利用冗余探索产生的内在对比信号来策划更高质量的记忆。好记忆引导扩展走向更有前景的路径，丰富的经验又锻造更强的记忆——形成正反馈循环。

### 训练策略
- 无需训练：全部基于 LLM 的上下文学习 (in-context learning)
- backbone: Gemini-2.5-flash/pro, Claude-3.7-sonnet
- 环境: BrowserGym (web browsing), Bash-only (SWE)
- ReAct 风格 Agent，默认 top-1 检索

## 实验关键数据

### 主实验 — WebArena

| 方法 | Shopping SR | Admin SR | Gitlab SR | Reddit SR | Overall SR | Steps |
|------|-----------|---------|----------|----------|------------|-------|
| No Memory (Gemini-2.5-flash) | 39.0 | 44.5 | 33.9 | 55.7 | 40.5 | 9.7 |
| Synapse | 40.6 | 45.1 | 35.6 | 59.4 | 42.1 | 9.2 |
| AWM | 44.4 | 46.7 | 37.2 | 62.3 | 44.1 | 9.0 |
| **ReasoningBank** | **49.7** | **51.1** | **40.6** | **67.0** | **48.8** | **8.3** |
| No Memory (Gemini-2.5-pro) | 45.5 | 51.1 | 35.0 | 71.7 | 46.7 | 8.8 |
| **ReasoningBank** (pro) | **51.9** | **56.6** | **44.4** | **80.2** | **53.9** | **7.4** |

### SWE-Bench-Verified

| 方法 | Resolve Rate | Steps |
|------|-------------|-------|
| No Memory (Gemini-2.5-flash) | 34.2 | 30.3 |
| ReasoningBank | **38.8** | **27.5** |
| No Memory (Gemini-2.5-pro) | 54.0 | 21.1 |
| ReasoningBank (pro) | **57.4** | **19.8** |

### MaTTS 扩展实验 (WebArena-Shopping, k=scaling factor)

| 配置 | k=1 | k=3 | k=5 |
|------|-----|-----|-----|
| MaTTS w/o memory (parallel) | 39.0 | 40.6 | 42.2 |
| MaTTS w/o aggregation (vanilla TTS) | 49.7 | 52.4 | 52.4 |
| **MaTTS (parallel)** | **49.7** | 53.5 | **55.1** |
| **MaTTS (sequential)** | 49.7 | **54.5** | 54.5 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅成功轨迹 | ReasoningBank: 46.5 SR | 仅用成功经验已优于基线 |
| 成功+失败轨迹 | ReasoningBank: 49.7 SR | 失败经验带来额外 3.2 个百分点提升 |
| Synapse+失败轨迹 | 41.7 SR | Synapse 无法有效利用失败信号 |
| AWM+失败轨迹 | 42.2 SR (反而降低) | AWM 处理失败导致性能下降 |
| 检索数量 k=1/2/3/4 | 49.7/46.0/45.5/44.4 | k=1 最优，过多记忆引入噪声 |

### 关键发现
- ReasoningBank 在 **所有数据集、所有 backbone** 上一致超越基线
- 效率提升显著：成功案例平均减少 2.1 步（26.9% 相对减少），说明记忆帮助 Agent 更快找到正确路径
- 跨领域泛化（Multi subset, Mind2Web cross-domain）优势尤为突出
- MaTTS 的协同效应：只有 ReasoningBank 能从 scaling 中受益（Pass@1 从 49.7 升至 50.8），弱记忆在 scaling 下反而退化

## 亮点与洞察
- **失败经验的价值被首次充分挖掘**: 不同于以往只利用成功轨迹的方法，ReasoningBank 证明失败中的反事实信号是更强大的泛化来源
- **涌现行为 (Emergent Behaviors)**: 记忆项会自然演化——从低级执行策略 → 自适应检查 → 组合式推理，类似 RL 中的涌现学习动态
- **记忆驱动的经验扩展作为新的 scaling 维度**: 传统 scaling 只增加计算量，MaTTS 将记忆质量和 scaling 联动，开辟了 Agent 能力提升的新途径
- **设计上的简洁性**: 整个系统不需要训练，仅靠 in-context learning + embedding retrieval + LLM judge，极易部署

## 局限与展望
- 依赖 LLM-as-a-judge 提供正确性信号，judge 本身可能出错导致记忆污染
- 记忆合并策略过于简单（直接追加），大规模部署时记忆池膨胀可能降低检索效率
- 检索仅用 embedding 相似度，缺乏推理感知的检索机制
- 未探索记忆遗忘/更新策略（过时的记忆可能干扰）
- 仅在 web browsing 和 SWE 上验证，其他 Agent 场景（如具身环境）有待探索

## 相关工作与启发
- **vs Synapse**: 存储原始轨迹作为 exemplar，记忆粒度太粗、不可迁移
- **vs AWM (Agent Workflow Memory)**: 从成功轨迹提取 workflow，但 (1) 只用成功经验 (2) 跨域迁移差（Multi subset 退化到 3.4 SR）
- **vs ExpeL**: 也利用成功/失败，但记忆以 tips 形式存储，不如 ReasoningBank 的结构化推理策略
- **启发**: Agent 的记忆系统应该像人类一样——不仅记住"怎么做成功的"，更要记住"为什么失败了"以及"抽象的决策原则"

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark](mmsu_a_massive_multi-task_spoken_language_understanding_and_reasoning_benchmark.md)
- [EmotionThinker: Prosody-Aware Reinforcement Learning for Explainable Speech Emotion Reasoning](emotionthinker_prosody-aware_reinforcement_learning_for_explainable_speech_emoti.md)
- [Stitch: Simultaneous Thinking and Talking with Chunked Reasoning for Spoken Language Models](stitch_simultaneous_thinking_and_talking_with_chunked_reasoning_for_spoken_langu.md)
- [Dynamic Parameter Memory: Temporary LoRA-Enhanced LLM for Long-Sequence Emotion Recognition in Conversation](dynamic_parameter_memory_temporary_lora-enhanced_llm_for_long-sequence_emotion_r.md)
- [The Impact of Scaling Training Data on Adversarial Robustness](../../NeurIPS2025/audio_speech/the_impact_of_scaling_training_data_on_adversarial_robustness.md)

<!-- RELATED:END -->
