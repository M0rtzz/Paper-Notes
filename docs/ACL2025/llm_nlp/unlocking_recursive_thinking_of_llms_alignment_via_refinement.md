---
title: >-
  [论文解读] Unlocking Recursive Thinking of LLMs: Alignment via Refinement
description: >-
  [ACL2025][LLM/NLP][递归思维] 提出 AvR（Alignment via Refinement）两阶段框架，通过细化感知奖励（refinement-aware reward）和差分学习，让 LLM 学会"批评→改进"的递归思维能力，仅用 10k 数据即在 AlpacaEval 2 上将 LLaMA-3-8B-Instruct 的胜率提升超 26 个百分点。
tags:
  - ACL2025
  - LLM/NLP
  - 递归思维
  - 对齐
  - 细化
  - 长链思维
  - 测试时缩放
  - DPO
---

# Unlocking Recursive Thinking of LLMs: Alignment via Refinement

**会议**: ACL2025  
**arXiv**: [2506.06009](https://arxiv.org/abs/2506.06009)  
**代码**: [Banner-Z/AvR](https://github.com/Banner-Z/AvR)  
**领域**: llm_nlp  
**关键词**: 递归思维, 对齐, 细化, 长链思维, 测试时缩放, DPO

## 一句话总结

提出 AvR（Alignment via Refinement）两阶段框架，通过细化感知奖励（refinement-aware reward）和差分学习，让 LLM 学会"批评→改进"的递归思维能力，仅用 10k 数据即在 AlpacaEval 2 上将 LLaMA-3-8B-Instruct 的胜率提升超 26 个百分点。

## 背景与动机

1. **长链思维推动测试时缩放**：OpenAI o1 系列证明 long-form CoT 能大幅提升复杂任务表现，但现有 LLM 大多缺乏自主多轮修正能力，难以在推理时迭代优化输出。
2. **传统对齐忽视过程信号**：DPO/RLHF 等方法仅对最终输出给予偏好奖励，缺少对反思、细化等中间过程的监督，导致模型无法从"哪里改进了"中学习。
3. **自我纠错困难**：已有研究表明 LLM 在缺乏外部奖励函数时很难纠正自身错误，直接提示模型 refine 反而可能降低质量（基线 LLaMA-3 refine 后胜率下降 2.6%）。
4. **o1-like 方法成本高**：基于 MCTS 或大规模 RL 的方法需要强力骨干模型、海量采样和巨大训练开销，在资源受限场景下不现实。
5. **并行采样的冗余**：传统偏好优化因无法区分生成质量差异，并行采样时容易重复同类错误。
6. **差分学习的启发**：Sutton 的差分学习思想表明，通过优化前后状态的奖励差值可更有效指导决策改进，天然适配"越改越好"的细化场景。

## 方法详解

### 整体框架

- **功能**：两阶段训练框架——Stage I 学习单步细化（多轮交互式），Stage II 将递归思维内化为自主长链 CoT。
- **为什么**：Stage I 先让模型掌握"批评→改进"的基本范式，Stage II 再通过轨迹蒸馏让模型无需外部提示即可自主递归推理，实现测试时缩放。
- **怎么做**：
  1. 将 query 和每轮 response 建模为多步 MDP，定义状态转移为上文拼接；
  2. 引入细化感知奖励 R(s_{t+1}, s_t)，要求每步改量为正且优于初始响应；
  3. 通过拒绝采样丢弃不满足条件的轨迹。

### 关键设计 1：Stage I — 单步细化优化

- **功能**：构建细化树（refinement tree），对每个 query 的初始和细化响应用 Bradley-Terry 奖励模型打分，得到批评/改进配对数据。
- **为什么**：让模型学会"每次改得更好"而非仅学好/坏偏好；通过 DPO 最大化细化前后的奖励差值，比传统 DPO 更精准。
- **怎么做**：
  1. 先用 Qwen2.5-32B 作为 corrector 为 LLaMA-3-8B 的初始输出生成批评和改进；
  2. RSFT：从细化树中选奖励最高的改进轨迹做监督微调（10k 样本）；
  3. DPO：分别对"生成""批评""改进"三类行为构造偏好对——生成步用最佳改进 vs 原始输出；批评步和改进步各自选分数最高/最低的一对。

### 关键设计 2：Stage II — 多步递归思维

- **功能**：用 Stage I 模型自动合成递归 CoT 轨迹，训练模型自主完成"生成→批评→改进→再批评→再改进→结束"全流程。
- **为什么**：Stage I 仍需显式 prompt 驱动每轮细化；Stage II 实现内化的递归思维，消除对外部指令和逐步监督的依赖。
- **怎么做**：
  1. 贪心搜索：每轮生成 x 条批评 × y 条改进（实验中 step=2），用 BT 模型选最优；
  2. 当没有任何改进优于当前最佳时停止，拼接为完整递归推理轨迹；
  3. RSFT 训练模型生成此类轨迹，模型在推理时使用 `<think>` 标记包裹中间步骤；
  4. 长度控制 DPO：采样 5 条输出，取分最高且更短的 vs 分最低且更长的做 DPO（4k 对），缓解奖励模型的长度偏好。

## 实验关键数据

### 实验 1：AlpacaEval 2 主实验

| 方法 | 数据量 | Win Rate | LC Win Rate |
|------|--------|----------|-------------|
| LLaMA-3-8B-Instruct (Seed) | - | 25.0% | 25.0% |
| DPO (传统) | 60k | 37.9% | 40.3% |
| SimPO | 60k | 40.5% | 44.7% |
| Meta-Rewarding Iter 4 | 20k | 39.5% | 39.4% |
| **AvR Stage I DPO + refine r2** | 20k | **50.8%** | 35.5% |
| **AvR Stage II RSFT** | 10k | **51.0%** | 42.5% |
| **AvR Stage II + 长度控制** | 14k | 49.0% | **51.4%** |

**发现**：AvR 仅用 10k-14k 数据超过需要 60k 数据的 SimPO 和 DPO；加入长度控制后 LC Win Rate 达到最高 51.4%，比 Seed 提升 26.4 个百分点。

### 实验 2：Arena-Hard v0.1

| 方法 | Score | 95% CI |
|------|-------|--------|
| GPT-3.5-turbo | 23.3% | (-2.2, 1.9) |
| GPT-4-0613 | 37.9% | (-2.8, 2.4) |
| SimPO | 33.8% | - |
| Meta-Rewarding Iter 4 | 29.1% | (-2.3, 2.1) |
| **AvR Stage II** | **34.5%** | (-2.5, 2.3) |

**发现**：AvR Stage II 在 Arena-Hard 上超过 GPT-3.5-Turbo，以及所有基于 LLaMA-3-8B 的基线方法，且仅用 10k SFT 数据。

### 实验 3：跨模型细化能力

AvR Stage I 模型可用于细化 GPT-4o 和 GPT-4o-mini 的输出，显著提升两者在 AlpacaEval 2 上的得分，说明习得的细化能力具有跨模型迁移性。

## 亮点

- **数据效率极高**：仅 3k 样本即可带来 20% 胜率提升，完整流程仅需 10k-14k 数据，远低于传统 RL 方法的 60k。
- **思路新颖**：将"差分学习"引入对齐训练，定义细化感知奖励实现"每步都进步"的递归思维。
- **两阶段设计优雅**：Stage I 外部驱动学基础能力，Stage II 内化为自主 CoT，渐进式地解锁递归推理。
- **跨模型迁移**：Stage I 训练的 8B 模型能改善 GPT-4o 的输出，展示细化能力作为通用技能的价值。

## 局限与展望

- **依赖外部奖励模型**：整个流程强依赖 BT 奖励模型（27B Skywork-Reward），其偏差（如长度偏好）直接影响数据合成质量。
- **仅评估开放式生成**：AlpacaEval 和 Arena-Hard 偏重对话质量，缺少数学/代码等结构化推理任务的验证。
- **Stage I 需要强模型引导**：初始阶段使用 Qwen2.5-32B 生成批评和改进数据，无法完全自举。
- **长度控制仍有损失**：长度控制 DPO 虽提升 LC Win Rate，但 Win Rate 下降 2%，两者的权衡尚需进一步研究。

## 与相关工作的对比

### vs SCoRe (Kumar et al., 2024)
SCoRe 是在线 RL 方法，需要在数学/代码基准上大量在线采样和训练来提升自我纠正能力。AvR 通过离线合成细化数据 + DPO 实现类似效果，训练成本显著更低，且面向开放式生成而非仅推理任务。

### vs Meta-Rewarding LLM (Wu et al., 2024)
Meta-Rewarding 需要 4 轮迭代训练（累计 20k 数据）才达到 39.5% 胜率。AvR 单次 Stage II 训练用 10k 数据即达 51.0% 胜率，核心区别在于 AvR 优化"改进过程"而非仅搜索更好响应。

### vs DeepSeek-R1 / o1-like 方法
o1-like 方法依赖大规模 RL + 强骨干模型（如 70B+），成本极高。AvR 在 8B 模型上用少量数据达到可比的递归思维效果，提供了一条低成本路径。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 细化感知奖励和两阶段递归思维框架是有意义的新思路
- 实验充分度: ⭐⭐⭐ — AlpacaEval 和 Arena-Hard 结果扎实但缺少推理类任务验证
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，公式推导完整，图表直观
- 价值: ⭐⭐⭐⭐ — 为小模型低成本获得递归推理能力提供了实用方案

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Condor: Enhance LLM Alignment with Knowledge-Driven Data Synthesis and Refinement](condor_enhance_llm_alignment_with_knowledge-driven_data_synthesis_and_refinement.md)
- [\[ACL 2025\] Training Language Model to Critique for Better Refinement](training_language_model_to_critique_for_better_refinement.md)
- [\[ACL 2025\] Self-Instructed Derived Prompt Generation Meets In-Context Learning: Unlocking New Potential of Black-Box LLMs](self-instructed_derived_prompt_generation_meets_in-context_learning_unlocking_ne.md)
- [\[ACL 2025\] Soundwave: Less is More for Speech-Text Alignment in LLMs](soundwave_less_is_more_for_speech-text_alignment_in_llms.md)
- [\[ACL 2025\] Nudging: Inference-time Alignment of LLMs via Guided Decoding](nudging_inference_time_alignment.md)

<!-- RELATED:END -->
