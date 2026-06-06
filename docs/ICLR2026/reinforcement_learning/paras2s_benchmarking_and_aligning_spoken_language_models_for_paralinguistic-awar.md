---
title: >-
  [论文解读] ParaS2S: Benchmarking and Aligning Spoken Language Models for Paralinguistic-Aware Speech-to-Speech Interaction
description: >-
  [ICLR 2026][强化学习][语音到语音] 提出 ParaS2S 框架——包含一个评估副语言感知（emotion/sarcasm/age/gender）的语音到语音基准 ParaS2SBench，以及一个基于 GRPO 的 RL 对齐框架 ParaS2SAlign…
tags:
  - "ICLR 2026"
  - "强化学习"
  - "语音到语音"
  - "副语言感知"
  - "benchmark"
  - "GRPO"
  - "奖励模型"
---

# ParaS2S: Benchmarking and Aligning Spoken Language Models for Paralinguistic-Aware Speech-to-Speech Interaction

**会议**: ICLR 2026  
**arXiv**: [2511.08723](https://arxiv.org/abs/2511.08723)  
**代码**: [项目页面](https://paras2sbench.github.io/)  
**领域**: 语音对话 / 强化学习  
**关键词**: 语音到语音, 副语言感知, benchmark, GRPO, 奖励模型

## 一句话总结

提出 ParaS2S 框架——包含一个评估副语言感知（emotion/sarcasm/age/gender）的语音到语音基准 ParaS2SBench，以及一个基于 GRPO 的 RL 对齐框架 ParaS2SAlign，使 S2S 模型能够在极少标注数据下习得根据说话风格调整回复的能力。

## 研究背景与动机

语音不仅传递文字内容，还携带丰富的**副语言信号**（paralinguistic cues）——情感、语调、说话者属性等，这些信号共同塑造了真实意图并引导恰当的回复。当前 S2S 模型存在几个核心问题：

**"语调聋" (Tone-deaf) 问题**：当前 S2S 模型（包括 Qwen2.5-Omni、GPT-4o Voice mode、GLM-4-Voice）在面对不同说话风格时，回复几乎没有差异。实验显示它们的评分在 3 分左右（5 分制），与忽略说话风格的 Pipeline 基线表现相当。

**缺乏评估基准**：现有基准大多关注语音到文本的理解能力（如 VoiceBench），或评估文本回复质量，没有基准直接评估 S2S 模型的输出语音在内容和风格上的恰当性。

**数据稀缺瓶颈**：构建副语言感知的 S2S 训练数据需要风格标注和表达性录制，成本极高。这是开发此类模型的主要障碍。

核心研究问题：受 DeepSeek-R1 的启发——推理能力可以通过 RL 在无 SFT 示范下涌现——**副语言感知对话能力是否也能通过 RL 在最少监督下涌现**？

## 方法详解

### 整体框架

ParaS2S 包含两大组件：

**ParaS2SBench**（基准）：
- 自动生成高质量语音测试查询
- 涵盖 4 个副语言维度：情感、讽刺、年龄、性别
- 每个查询配对两种对比说话风格
- 直接评估输入-输出语音对的适配度

**ParaS2SAlign**（对齐框架）：
- Stage 1: SFT 预热
- Stage 2: 蒸馏奖励模型
- Stage 3: GRPO 后训练

### 关键设计

1. **场景控制查询生成**：每个测试查询遵循三个设计原则：

    - **中性文本内容**：如 "I just bumped into my ex." 本身不暗示任何情感
    - **对比说话风格**：同一文本配对两种对比风格（如惊讶 vs 悲伤）
    - **副语言相关性**：说话风格确实影响恰当的回复内容

   → 核心思路：只有当模型必须通过音频信号而非文本推测说话者状态时，才能真正测试副语言感知  
   → 设计动机：防止模型仅通过文本内容"猜中"正确回复

   数据质量通过三重过滤保证：中性测试、合理性测试、副语言相关性测试。所有通过 ChatGPT 自动完成，最终由人工确认。

2. **多模型评估管线**：评估输出语音时，分别提取内容（Whisper-v3 转录）和说话风格（AudioReasoner 分析），然后用 ChatGPT 4.1 综合评分：

    $f_{\text{gpt}} = GPT(c_i, s_i, c_o, s_o, r)$

   → 核心思路：将语音评估分解为内容理解和风格理解两个子问题，再综合判断  
   → 设计动机：直接评估语音质量困难，通过文本中介桥接实现可自动化的评估

3. **三阶段对齐框架**：

   **Stage 1 - SFT 预热**：使用 10k 语音提示构建高质量示范，通过 gpt-4o-mini-tts 合成表达性回复，SFT 训练使模型获得基础副语言感知能力。这一步是必要的——基座模型完全缺乏该能力，无法在 RL 中采样到有意义的回复。

   **Stage 2 - 蒸馏奖励模型**：SFT 模型对 10k 查询各生成 32 个回复（共 320k 对），通过评估管线打分，训练一个奖励模型 $\phi$ 来近似管线评分。这解决了管线评估速度慢的问题。

   **Stage 3 - GRPO 后训练**：使用 100k 无标注语音提示，通过 GRPO 让模型从自身生成的回复中学习。组内采样 $G=8$ 个回复，用奖励模型 $\phi$ 打分，归一化后计算优势函数更新策略。

   → 核心思路：通过 SFT 建立基础能力 → 蒸馏自动评估为快速奖励模型 → RL 探索无标注空间  
   → 设计动机：最小化人工标注需求的同时最大化学习效率

4. **KL 散度约束**：GRPO 损失中包含 KL 惩罚项（$\beta = 0.2$），防止模型在学习副语言能力时遗忘原有的对话智能。该项同时作用于音频和文本两个 token 流。

   → 核心思路：在新能力与原有能力之间取得平衡  
   → 设计动机：消融实验显示 $\beta = 0$ 导致 VoiceBench 性能严重下降（灾难遗忘）

### 损失函数 / 训练策略

- **SFT**：标准 next-token prediction，同时优化音频流 $a_o$ 和文本流 $t_o$
- **奖励模型**：交叉熵损失，将 Likert 分数作为单字符预测任务
- **GRPO**：带 KL 惩罚的组相对策略优化，奖励来自蒸馏的奖励模型

训练配置：
- 基座模型：Kimi-Audio
- SFT/GRPO：8x NVIDIA H100，FSDP
- SFT 学习率 1e-5，全局批次 64
- RL 学习率 5e-4，查询批次 32，组大小 8
- 奖励模型：单 H100，LoRA 微调，学习率 1e-6

## 实验关键数据

### 主实验

**S2S 模型综合比较**（ParaS2SBench 分数，5 分制）

| 模型 | Synthetic Avg | Real Avg | Overall Avg |
|------|-------------|---------|------------|
| Whisper-GPT-TTS (Pipeline基线) | 3.022 | 3.487 | 3.176 |
| GPT-4o Voice mode | 3.284 | 3.639 | 3.403 |
| Qwen2.5 Omni | 3.248 | 3.612 | 3.369 |
| GLM-4-Voice | 3.033 | 3.037 | 3.034 |
| Kimi-Audio (基座) | 2.892 | 1.265 | 2.350 |
| **Kimi-Audio SFT** | **4.076** | **3.714** | **3.955** |
| **Kimi-Audio GRPO** | **4.441** | **4.161** | **4.382** |
| GPT-TTS (Topline) | 4.705 | 4.766 | 4.725 |

关键观察：
- 所有现有 S2S 模型的表现与 Pipeline 基线相当（~3.0-3.4），说明它们确实不具备副语言感知
- SFT 带来 68% 的相对提升，GRPO 在此基础上再提升 11%
- GRPO 模型接近 Topline（理想 TTS 系统）的表现

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| RL w/ 10h SFT 预热 | 匹配 50h SFT | RL 的标签效率极高 |
| RL w/ 20h SFT 预热 | 匹配 100h SFT | 仅需 1/5 标注即可达到相同性能 |
| KL $\beta = 0$ | VoiceBench 大幅下降 | 无 KL 约束导致灾难遗忘 |
| KL $\beta = 0.2$ | 两指标均优 | 最优平衡点 |
| 组大小 $G < 8$ | 性能显著下降 | $G=2$ 时两个样本常获相同分数 |
| 组大小 $G \geq 8$ | 无额外提升 | $G=8$ 足够提供学习信号 |
| 更多资源投入 SFT vs RM | SFT 更有利 | 10h 标注即可构建可用的奖励模型 |

### 关键发现

1. **基准与人类评分高度一致**：Pearson 相关系数均超过 0.7，模型排名与人类评估几乎一致（仅一个交换）
2. **RL 的标签效率远超 SFT**：10h SFT 预热 + RL 即可匹配 50h 纯 SFT。SFT 需要超过 10 倍的数据才能与 RL 持平
3. **合成数据到真实语音的泛化**：在合成数据上训练的 SFT 和 GRPO 在真实语音测试集（IEMOCAP、MELD）上同样有效
4. **跨域泛化**：在 IEMOCAP 上 RL 训练的模型在 MELD 上也有提升，反之亦然
5. **人类主观评估**：GRPO 模型在主观评估中也超越 SFT 7.6%，尽管普通用户对副语言失误更宽容

## 亮点与洞察

1. **发现并量化了"语调聋"问题**：首次系统性地揭示当前 SOTA S2S 模型在副语言感知上的严重不足，所有模型表现与忽略风格的 Pipeline 基线相当
2. **端到端语音级评估**：ParaS2SBench 是首个直接评估 S2S 输出语音的内容和风格恰当性的基准，而非仅评估文本回复
3. **RL 解锁副语言能力**：证明了类似 DeepSeek-R1 的 RL 范式可以在语音领域发挥作用——仅需最少的 SFT 预热，就能通过自我探索习得副语言感知
4. **极强的数据效率**：10 小时标注数据 + RL 即可超越所有现有模型，挑战了"需要大量高质量标注"的固有认知
5. **实用的自动评估管线**：将复杂的语音质量评估分解为 ASR + 风格识别 + LLM 评分的可组合管线，高效且与人类判断一致

## 局限与展望

1. **评估管线的相关系数未达 0.9**：虽然超过 0.7 且显著，但仍有改进空间，特别是在需要精细风格区分的场景
2. **依赖多个外部模型**：评估管线依赖 Whisper-v3、AudioReasoner 和 ChatGPT，这些模型本身的偏差会传播到评估结果
3. **TTS 合成数据的保真度**：gpt-4o-mini-tts 的不稳定性需要生成 10 个候选并人工筛选，合成语音的自然度仍有差距
4. **仅在 Kimi-Audio 上验证**：虽然框架声称适用于任何 LM-based S2S 模型，但实验仅在一个基座模型上进行
5. **计算成本**：Stage 2 需要生成 320k 语音回复并通过评估管线打分，计算和 API 成本不低
6. **可扩展方向**：支持更多副语言维度（如口音、语速、停顿）、多轮对话中的风格一致性、多说话者场景

## 相关工作与启发

- **DeepSeek-R1**：RL 可以在无 SFT 下涌现推理能力的开创性工作，ParaS2S 将此思想移植到语音领域
- **StyleTalk / ParalinGPT**：最早关注副语言感知对话的工作，但仅限于语音到文本，且完全依赖 SFT
- **GOAT-SLM**：唯一强调副语言的 S2S 模型，使用多阶段 SFT 管线。ParaS2S 用 RL 替代了大量标注需求
- **Align-SLM**：使用 DPO 对齐语音模型，但关注长距离语义而非副语言
- 启发：RL 在模态扩展中的力量被低估——即使是"软技能"（如情感感知）也可以通过 RL 高效学习

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次将 RL 框架应用于副语言感知 S2S，问题定义和解决方案都是新的
- 实验充分度: ⭐⭐⭐⭐⭐ — 极其全面：基准验证、模型比较、消融、数据效率、泛化性、人类评估
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，实验详尽，但部分内容有冗余
- 价值: ⭐⭐⭐⭐⭐ — 填补了 S2S 领域的重要空白，基准和方法都有长期价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](verifybench_benchmarking_reference-based_reward_systems_for_large_language_model.md)
- [\[NeurIPS 2025\] Checklists Are Better Than Reward Models For Aligning Language Models](../../NeurIPS2025/reinforcement_learning/checklists_are_better_than_reward_models_for_aligning_langua.md)
- [\[ICLR 2026\] Post-training Large Language Models for Diverse High-Quality Responses](post-training_large_language_models_for_diverse_high-quality_responses.md)
- [\[ICLR 2026\] Towards Strategic Persuasion with Language Models](towards_strategic_persuasion_with_language_models.md)
- [\[ICLR 2026\] Co-rewarding: Stable Self-supervised RL for Eliciting Reasoning in Large Language Models](co-rewarding_stable_self-supervised_rl_for_eliciting_reasoning_in_large_language.md)

</div>

<!-- RELATED:END -->
