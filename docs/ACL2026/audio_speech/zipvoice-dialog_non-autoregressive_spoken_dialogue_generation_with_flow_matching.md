---
title: >-
  [论文解读] ZipVoice-Dialog: Non-Autoregressive Spoken Dialogue Generation with Flow Matching
description: >-
  [ACL 2026][音频/语音][对话语音生成] 提出 ZipVoice-Dialog，首个基于流匹配的非自回归零样本对话语音生成模型，通过课程学习策略和说话人轮次嵌入两个简单设计，解决了流匹配直接用于对话场景时的语音不可懂和轮次混乱问题…
tags:
  - "ACL 2026"
  - "音频/语音"
  - "对话语音生成"
  - "非自回归"
  - "流匹配"
  - "说话人轮次"
  - "课程学习"
---

# ZipVoice-Dialog: Non-Autoregressive Spoken Dialogue Generation with Flow Matching

**会议**: ACL 2026  
**arXiv**: [2507.09318](https://arxiv.org/abs/2507.09318)  
**代码**: [https://github.com/k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice)  
**领域**: 图像生成  
**关键词**: 对话语音生成, 非自回归, 流匹配, 说话人轮次, 课程学习

## 一句话总结

提出 ZipVoice-Dialog，首个基于流匹配的非自回归零样本对话语音生成模型，通过课程学习策略和说话人轮次嵌入两个简单设计，解决了流匹配直接用于对话场景时的语音不可懂和轮次混乱问题，同时发布了首个大规模开源对话语音数据集 OpenDialog（6.8k 小时）。

## 研究背景与动机

**领域现状**：文本到语音（TTS）技术在单说话人独白场景已取得出色成果。然而，合成多说话人自然对话仍是重大挑战，因为对话需要准确自然的说话人轮次切换和保持不同说话人的音色区分。

**现有痛点**：当前最先进的对话语音生成方法主要依赖自回归（AR）架构（如 MoonCast、Dia），但 AR 模型存在两个固有缺陷：(1) 推理延迟高，因为需要逐步顺序生成；(2) 鲁棒性问题严重，暴露偏差导致词语重复或跳词等不稳定现象。

**核心矛盾**：流匹配作为非自回归方法在独白 TTS 中已展现出色性能，但作者的初步实验发现，直接将流匹配架构用于对话生成会导致完全不可懂的语音——模型能模仿提示音的风格和音色，但完全无法反映输入文本的内容。这是因为对话中涉及两个不同说话人的音色，使得语音-文本对齐学习变得极其困难。

**本文目标**：设计有效的方法使流匹配架构适用于多说话人对话生成，同时解决训练数据稀缺问题。

**切入角度**：作者观察到问题的根源在于多说话人音色的对齐学习难度，因此从"先学会对齐再学对话"的课程学习角度切入，并通过显式的说话人轮次嵌入提供清晰的说话人线索。

**核心 idea**：用课程学习（先独白预训练后对话微调）解决对齐问题，用可学习的说话人轮次嵌入解决轮次切换问题，使流匹配 NAR 架构在对话生成中同时获得高速度和高稳定性。

## 方法详解

### 整体框架

ZipVoice-Dialog 基于 ZipVoice（流匹配独白 TTS 模型），包含文本编码器和向量场估计器（均使用 Zipformer 骨干），以及预训练的 Vocos 声码器。输入为交错排列的多说话人文本序列和提示音频，输出为完整的对话语音。训练使用条件流匹配（CFM）目标和语音填充任务实现零样本能力。

### 关键设计

1. **独白到对话的课程学习（Monologue-to-Dialogue Curriculum Learning）**:

    - 功能：解决流匹配模型直接在对话数据上训练时的语音-文本对齐崩溃问题
    - 核心思路：分两阶段训练。**阶段一**：用 ZipVoice（在 100k 小时独白数据上预训练的模型）的权重初始化，建立稳健的语音-文本对齐基础。**阶段二**：在对话数据上微调，学习对话动态特性——多说话人上下文中的对齐适配、正确的音色分配和自然的轮次切换
    - 设计动机：直接在对话数据上从头训练会导致对齐崩溃（WER 从 ~5% 飙升到 >100%），因为两个说话人音色的存在极大增加了对齐学习难度。先在简单的独白任务上学好对齐再迁移到对话，是经典的课程学习思路

2. **说话人轮次嵌入（Speaker-Turn Embeddings）**:

    - 功能：让模型准确区分两个说话人并为每个轮次分配正确的音色
    - 核心思路：引入两个随机初始化的可学习嵌入向量，分别对应 [S1] 和 [S2] 两个说话人身份。对文本序列中的每个 token $y_i$，根据其说话人身份检索对应的嵌入 $e_{speaker(i)}$，加到文本特征上：$\widetilde{y_i} = \hat{y_i} + e_{speaker(i)}$，然后再进行时间维度上采样
    - 设计动机：对比实验表明，使用分隔符 "|" 或文本标记 [S1]/[S2] 都无法有效区分说话人（cpWER 远高于 WER），而说话人轮次嵌入将 cpWER 从 37.82/31.34 大幅降至 5.82，实现了几乎完美的轮次准确率

3. **交错文本输入与灵活提示（Interleaved Text & Flexible Prompting）**:

    - 功能：处理多说话人对话的复杂输入格式
    - 核心思路：将多轮发言按时间排序组成单一交错文本序列，前缀加说话人标识。训练时使用随机长度前缀的对话音频作为提示，推理时支持任意轮数的提示音频。模型通过流匹配目标隐式建模 token 和轮次时长，无需外部时长预测器
    - 设计动机：相比需要预定义时间戳的方案，端到端隐式建模更简洁，训练和推理都不依赖额外的时间戳预测模型

### 损失函数 / 训练策略

使用条件流匹配（CFM）损失，仅在掩码区域计算：$L_{CFM} = \mathbb{E}_{t,q(x_1),p_0(x_0)} \| (v_t(x_t, z, (1-m) \odot x_1; \theta) - (x_1 - x_0)) \odot m \|^2$。在 OpenDialog + 内部数据集（共 7.6k 小时）上微调 60k 步，总 batch 大小 4k 秒。推理使用 Euler 求解器 16 步采样。

## 实验关键数据

### 主实验

与开源对话语音生成模型对比（中英文测试集）：

| 模型 | 参数量 | RTF↓ | 中文 WER↓ | 英文 WER↓ | cpSIM↑ | UTMOS↑ |
|------|--------|------|-----------|-----------|--------|--------|
| Dia | 1.61B | 1.663 | - | 11.80 | 0.333 | 1.87 |
| MoonCast | 2.67B | 0.953 | 15.85 | 23.62 | 0.356 | 2.37 |
| ZipVoice-Dialog | **123M** | **0.063** | **3.17** | **3.25** | **0.437** | **3.07** |

ZipVoice-Dialog 以仅 123M 参数实现了全面碾压：推理速度快 15 倍以上，WER 降低 3-7 倍。

### 消融实验

| 配置 | 英文 WER↓ | 英文短 cpWER↓ | 说明 |
|------|-----------|---------------|------|
| 完整模型（课程学习+嵌入） | 3.25 | 3.27 | 最优 |
| 无课程学习 | 116.10 | 116.31 | 对齐崩溃，完全不可懂 |
| 分隔符 "|" 替代嵌入 | 5.34 | 37.82 | 轮次准确率极差 |
| 文本标记替代嵌入 | 5.57 | 31.34 | 轮次准确率差 |
| 仅 OpenDialog 数据 | 3.34 | 3.53 | 仅数据即可达到强基线 |

### 关键发现

- 课程学习是不可或缺的：没有它模型完全无法工作（WER >100%），这说明多说话人对齐是流匹配对话生成的核心瓶颈
- 说话人轮次嵌入极其简单但极其有效：两个可学习向量就将轮次错误率从 >30% 降至 <1%
- 主观评估（CMOS/SMOS）中 ZipVoice-Dialog 大幅领先 MoonCast（CMOS 差距 -1.17，SMOS 3.86 vs 2.35）
- OpenDialog 数据集单独使用即可训练出超越基线的模型，验证了数据集的高质量

## 亮点与洞察

- **极简但极效的设计哲学**令人印象深刻——仅课程学习 + 两个可学习嵌入就让流匹配架构从"完全不可用"变成"全面碾压 AR 基线"，这种以最小改动获得最大效果的思路值得借鉴
- **OpenDialog 数据集**的开源贡献非常有价值——首个大规模（6.8k 小时）开源对话语音数据集，填补了领域空白，数据构建流程（VAD→说话人日志→ASR→LLM 分类→WhisperD 精标→规则过滤）可复用
- 123M 参数模型在所有指标上碾压 1.6B-2.7B 的 AR 模型，证明了 NAR 架构在对话场景的巨大潜力

## 局限与展望

- 模型和数据规模有限，小模型在表达力上有天花板，更大模型+数据可能进一步提升
- 主观评估仅限中文，英文的主观质量未验证
- 目前限于两人对话，虽然方法可扩展到多人，但未验证
- 未探索重叠语音和回应词（backchannel）等更自然的对话现象

## 相关工作与启发

- **vs MoonCast**: MoonCast 采用 AR+NAR 混合架构（LLM→流匹配→声码器），参数量 2.67B，但 AR 部分导致严重的跳词和不稳定问题（WER 23.62%）。ZipVoice-Dialog 纯 NAR 仅 123M 参数，WER 3.25%，快 15 倍
- **vs Dia**: Dia 是纯 AR 模型直接预测音频 token，1.61B 参数但 WER 11.80%，音色相似度最低（cpSIM 0.333）。说明纯 AR 路线在对话场景的鲁棒性不足

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个将流匹配成功应用于对话语音生成的工作，但核心技巧（课程学习、嵌入）本身并不新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 完整的消融实验、主客观评估、数据集对比、基准测试建立，非常充分
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，问题动机和解决方案的逻辑链非常流畅，易于理解
- 价值: ⭐⭐⭐⭐⭐ 模型+数据集+基准三重贡献，OpenDialog 数据集对社区尤为重要

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] VoxMind: An End-to-End Agentic Spoken Dialogue System](voxmind_an_end-to-end_agentic_spoken_dialogue_system.md)
- [\[ACL 2026\] SDiaReward: Modeling and Benchmarking Spoken Dialogue Rewards with Modality and Colloquialness](sdiareward_modeling_and_benchmarking_spoken_dialogue_rewards_with_modality_and_c.md)
- [\[ACL 2025\] WavRAG: Audio-Integrated Retrieval Augmented Generation for Spoken Dialogue Models](../../ACL2025/audio_speech/wavrag_audio-integrated_retrieval_augmented_generation_for_spoken_dialogue_model.md)
- [\[ACL 2026\] Anchored Cyclic Generation: A Novel Paradigm for Long-Sequence Symbolic Music Generation](anchored_cyclic_generation_a_novel_paradigm_for_long-sequence_symbolic_music_gen.md)
- [\[ACL 2026\] Full-Duplex-Bench-v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner](full-duplex-bench-v2_a_multi-turn_evaluation_framework_for_duplex_dialogue_syste.md)

</div>

<!-- RELATED:END -->
