---
title: >-
  [论文解读] Towards Reliable Large Audio Language Model
description: >-
  [ACL 2025][语音][Large Audio Language Model] 本文首次系统研究大型音频语言模型（LALM）的可靠性问题，提出训练无关方法（IDK/MCoT/Task Agent）和训练方法（基于模型特定 IDK 数据集的 LoRA SFT），并设计 Reliability Gain Index（RGI）指标来评估可靠性提升效果，发现"知道说不知道"是可跨音频模态迁移的元能力。
tags:
  - ACL 2025
  - 语音
  - Large Audio Language Model
  - Reliability
  - IDK
  - 跨模态迁移
  - 拒答能力
---

# Towards Reliable Large Audio Language Model

**会议**: ACL 2025  
**arXiv**: [2505.19294](https://arxiv.org/abs/2505.19294)  
**代码**: 无  
**领域**: audio_speech  
**关键词**: Large Audio Language Model, Reliability, IDK, 跨模态迁移, 拒答能力

## 一句话总结

本文首次系统研究大型音频语言模型（LALM）的可靠性问题，提出训练无关方法（IDK/MCoT/Task Agent）和训练方法（基于模型特定 IDK 数据集的 LoRA SFT），并设计 Reliability Gain Index（RGI）指标来评估可靠性提升效果，发现"知道说不知道"是可跨音频模态迁移的元能力。

## 研究背景与动机

**领域现状**：大型音频语言模型（LALM）如 Qwen2-Audio 等在语音、音乐、环境音等多模态音频的理解和推理上取得了显著进展，能够处理 ASR、音频描述、情感识别等多种任务。

**现有痛点**：尽管 LALM 表现出色，但它们缺乏识别自身知识边界的能力——面对不会的问题时，模型不会主动拒答，而是会给出错误或过度自信的回答。这在医疗、自动驾驶等高风险场景中尤为危险。

**核心矛盾**：在文本 LLM 领域已有许多可靠性增强工作（如 IDK 数据集、Prudence Score 等），但音频语言模型的可靠性研究几乎空白。音频数据的特殊性——语音、音乐、环境音在结构和内容上差异巨大——使得直接迁移文本领域方法面临新挑战。

**本文目标**：(1) 如何系统地增强 LALM 的拒答能力？(2) 如何准确评估不同可靠性增强方法的有效性？(3) 可靠性意识能否在不同音频模态间迁移？

**切入角度**：作者观察到现有评估指标（Accuracy、Truthfulness、Reliability）无法区分方法在"保守性"和"谦逊性"之间的权衡效果，因此提出 RGI 新指标，从相对增益角度衡量可靠性提升。

**核心 idea**：将文本 LLM 的可靠性增强范式引入音频模态，用 training-free + training-based 双路径提升 LALM 拒答能力，并通过 RGI 指标发现可靠性意识是可跨模态迁移的"元能力"。

## 方法详解

### 整体框架

输入为音频+问题，模型需判断是否有能力正确回答：能回答则给出答案，不能则输出"I don't know"。作者探索了两大类方法：(1) 不需要额外训练的推理时增强；(2) 基于模型特定 IDK 数据集的有监督微调。

### 关键设计

1. **IDK Prompting（训练无关）**:

    - 功能：在输入问题后追加提示语，鼓励模型在不确定时主动说"I don't know"
    - 核心思路：利用模型本身的指令跟随能力，通过补充提示激活其不确定性表达
    - 设计动机：最简单的 baseline，零成本验证模型是否具备潜在拒答能力

2. **MCoT Prompting（训练无关）**:

    - 功能：多模态思维链推理，要求模型逐步分析后再回答
    - 核心思路：借鉴 Chain-of-Thought 思想，将复杂问题分解为小步骤顺序处理，让模型在推理过程中更好地评估自身置信度
    - 设计动机：逐步推理能暴露模型在中间步骤的不确定性，从而做出更可靠的最终判断

3. **Task Agent（训练无关）**:

    - 功能：三步式推理 Agent——先识别音频类型（speech/sound/music），然后生成对应内容（ASR/AAC/MC），最后结合原始音频和生成内容给出最终答案
    - 核心思路：通过中间工具调用（tool-using），将隐式推理显式化。对语音先做 ASR 获取文本内容，对环境音/音乐生成 caption，让模型基于更充分的信息做判断
    - 设计动机：音频数据在语音、音乐、环境音之间差异极大，统一处理效果不佳；根据音频类型调用不同工具可提供更有针对性的上下文信息

4. **基于 IDK 数据集的 LoRA 微调（训练方法）**:

    - 功能：构建模型特定的 IDK 数据集，然后用 LoRA 进行有监督微调
    - 核心思路：对每个问题进行 N 次推理采样，若模型 N 次全部回答正确则保留原始标签，否则将答案标记为 IDK。使用 K@N 阈值控制严格度（本文用 5@5），然后在该数据集上做 1 epoch 的 LoRA SFT
    - 设计动机：不同模型的知识边界不同，需要构建模型特定的 IDK 数据集。通过训练让模型显式学习何时应该拒答，比 prompting 更直接有效
    - 与之前方法区别：首次将文本 LLM 的 IDK 训练范式引入多模态音频领域

### 评估指标设计

本文核心贡献之一是提出 Reliability Gain Index (RGI)：

- **Accuracy** = Nc / N，正确回答比例
- **Truthfulness** = 1 - Nw / N，非错误回答比例
- **Reliability** = Rej × Acc + (1-Rej) × Tru，综合可靠性
- **相对保守性增加** ΔCon = (Nc - Ncc) / Nc，原来答对的现在被拒答的比例（越低越好）
- **相对谦逊性增加** ΔHum = (Nw - Nww) / Nw，原来答错的现在被正确拒答的比例（越高越好）
- **RGI** = log(ΔHum / ΔCon)，正值表示方法有效，值越大越好

### 损失函数 / 训练策略

- 使用 DeepSpeed + LoRA（PEFT 库）实现参数高效微调
- 基于 Qwen2-Audio-7B-Instruct 作为基座模型
- 每个模态的 IDK 数据集各训练 1 epoch
- LoRA alpha 权重是关键超参数：过小则模型学不会拒答，过大则过度保守

## 实验关键数据

### 主实验：Accuracy / Truthfulness / Reliability 比较

基于 Qwen2-Audio-7B-Instruct 在 MMAU 数据集（sound/music/speech）上的表现：

| 方法 | 训练? | Sound Acc% | Sound Rel% | Music Acc% | Music Rel% | Speech Acc% | Speech Rel% | Total Rel% |
|------|-------|-----------|-----------|-----------|-----------|------------|------------|-----------|
| Baseline | ✗ | 60.96 | 60.96 | 55.09 | 55.09 | 50.75 | 50.75 | 55.60 |
| IDK Prompting | ✗ | 58.26 | 73.03 | 54.19 | 65.19 | 43.84 | 56.18 | 64.85 |
| MCoT Prompting | ✗ | 57.96 | 67.13 | 51.50 | 67.53 | 44.74 | 57.71 | 64.29 |
| Task Agent | ✗ | 58.56 | 70.68 | 53.29 | 68.22 | 46.25 | 57.93 | 65.66 |
| LoRA SFT | ✓ | 61.71 | 70.71 | 51.35 | 66.43 | 47.90 | 59.91 | 65.68 |
| 人类 | - | 86.31 | 86.31 | 78.22 | 78.22 | 82.17 | 82.17 | 82.23 |

### RGI 指标消融对比

| 方法 | Sound ΔCon% | Sound ΔHum% | Sound RGI | Music RGI | Speech RGI | Total RGI |
|------|------------|------------|----------|----------|-----------|----------|
| IDK Prompting | 10.81 | 20.12 | 0.27 | 0.20 | 0.02 | 0.16 |
| MCoT Prompting | 11.71 | 14.41 | 0.09 | 0.25 | 0.09 | 0.15 |
| Task Agent | 9.61 | 16.52 | 0.24 | 0.27 | 0.17 | 0.23 |
| LoRA Fine-tuning | 6.91 | 15.62 | 0.36 | 0.23 | 0.19 | 0.26 |

### 关键发现

- **所有方法都能提升可靠性**：Truthfulness 和 Reliability 均有改善，但 Accuracy 有所下降，说明增强可靠性需要付出"有用性"的代价
- **训练方法优于推理方法**：LoRA SFT 在 Accuracy 损失更小的同时获得更高 RGI（0.26 vs 训练无关方法的 0.15-0.23），实现了保守性与谦逊性的更好平衡
- **Sound 和 Music 上 RGI 更高**：说明模型在这两个模态上的知识边界更清晰，Speech 上表现相对较差
- **跨模态迁移有效**：在一个模态上训练、另一个模态上测试，所有 RGI > 0，验证了可靠性意识的跨模态可迁移性
- **LoRA alpha 非单调效应**：很小的 alpha 即可学到高 RGI，过大反而导致过度保守（RGI < 0），说明可靠性意识是一种容易获取的能力
- **IDK 比例变化小**：从 1@5 的 50.2% 到 5@5 的 63.5%，相比文本 LLM 变化更小，说明 LALM 响应稳定性较高

## 亮点与洞察

- **"元能力"发现**：可靠性意识（知道何时说"不知道"）可以在 sound/music/speech 之间迁移。这说明拒答能力不依赖于特定模态的内容理解，而是模型层面的通用能力，这对构建统一多模态可靠系统具有重要启发
- **RGI 指标设计巧妙**：RGI = log(ΔHum/ΔCon) 将传统指标无法区分的"好拒答"和"坏拒答"分开度量——一个方法可能 Reliability 很高但其实只是全部拒答（过度保守），RGI 能有效检测这种问题
- **Task Agent 的模态感知设计**：先识别音频类型再调用对应工具（ASR/AAC/MC），这种 pipeline 思路可迁移到其他多模态任务，例如在视觉语言模型中先判断图像类型再选择处理策略

## 局限与展望

- **只支持简单拒答**：模型只能说"I don't know"，不能给出拒答原因或主动追问用户获取更多信息，交互性不足
- **仅在 Qwen2-Audio 上验证**：虽然附录测试了其他模型，但主实验只用了一个模型，结论的普适性有待验证
- **评估仅限 MMAU**：该数据集是多选题形式，未验证在开放式问答场景下可靠性增强方法是否有效
- **IDK 数据集构建成本高**：5@5 阈值需要对每个问题推理 5 次，计算成本较大
- **跨模态迁移仅限音频内部**：sound/music/speech 虽然差异大但同属音频，是否能迁移到视频、图像等更远模态有待探索

## 相关工作与启发

- **vs 文本 LLM 可靠性（Cheng et al., 2024）**：文本领域首先提出 IDK 数据集和 Knowledge Quadrants 概念，本文将其扩展到多模态音频设置，发现音频模型的 IDK 比例变化相对较小（50.2%→63.5%），说明 LALM 的响应稳定性比文本 LLM 更高
- **vs Xu et al. (2024a) 可靠性评估**：Xu 提出了加权 Reliability 指标，但该指标无法区分"好的拒答"和"过度保守"，本文的 RGI 通过对比谦逊性增益和保守性增益弥补了此缺陷
- **vs Qwen2-Audio**：作为当前最强开源 LALM 之一，其在 MMAU 上的可靠性仍然很低（Rel = 55.6%），说明即使性能强大的模型也需要额外的可靠性增强

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统研究 LALM 可靠性，RGI 指标设计合理，"元能力"发现有启发性
- 实验充分度: ⭐⭐⭐⭐ 覆盖三个音频模态、四种方法、跨模态实验和超参数分析，但只在一个数据集上验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，问题定义明确，公式推导完整，图表丰富
- 价值: ⭐⭐⭐⭐ 为 LALM 可靠性研究开辟了新方向，RGI 指标和跨模态迁移发现具有实用价值

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Benchmarking Open-ended Audio Dialogue Understanding for Large Audio-Language Models](audio_dialogue_benchmark.md)
- [\[ACL 2025\] Investigating and Enhancing Vision-Audio Capability in Omnimodal Large Language Models](investigating_and_enhancing_vision-audio_capability_in_omnimodal_large_language_.md)
- [\[AAAI 2026\] AHAMask: Reliable Task Specification for Large Audio Language Models without Instructions](../../AAAI2026/audio_speech/ahamask_reliable_task_specification_for_large_audio_language.md)
- [\[ACL 2025\] Contextual Biasing with the Knowledgeable External Language Model for End-to-End Speech Recognition](contextual_biasing_with_the_knowledgeable_external_language_model_for_end-to-end.md)
- [\[ACL 2025\] Who Can Withstand Chat-Audio Attacks? An Evaluation Benchmark for Large Audio-Language Models](who_can_withstand_chat-audio_attacks_an_evaluation_benchmark_for_large_audio-lan.md)

<!-- RELATED:END -->
