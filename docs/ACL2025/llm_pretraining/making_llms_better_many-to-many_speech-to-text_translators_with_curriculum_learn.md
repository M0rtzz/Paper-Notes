---
title: >-
  [论文解读] Making LLMs Better Many-to-Many Speech-to-Text Translators with Curriculum Learning
description: >-
  [ACL 2025][预训练][语音到文本翻译] 提出 LLM-SRT，将语音到文本翻译（S2TT）任务转化为语音识别与翻译联合任务（SRT），并通过三阶段课程学习策略（ASR→SMT→SRT）有效利用 LLM 的机器翻译能力…
tags:
  - "ACL 2025"
  - "预训练"
  - "语音到文本翻译"
  - "课程学习"
  - "MLLM"
  - "多语言"
  - "低资源"
---

# Making LLMs Better Many-to-Many Speech-to-Text Translators with Curriculum Learning

**会议**: ACL 2025  
**arXiv**: [2409.19510](https://arxiv.org/abs/2409.19510)  
**代码**: [有](https://github.com/yxduir/LLM-SRT)  
**领域**: LLM预训练  
**关键词**: 语音到文本翻译, 课程学习, MLLM, 多语言, 低资源

## 一句话总结

提出 LLM-SRT，将语音到文本翻译（S2TT）任务转化为语音识别与翻译联合任务（SRT），并通过三阶段课程学习策略（ASR→SMT→SRT）有效利用 LLM 的机器翻译能力，在低资源场景（每种语言不到 10 小时数据）下实现 15×14 语言对的 SOTA 多对多语音翻译性能。

## 研究背景与动机

### 问题现状

语音到文本翻译（S2TT）传统上依赖级联系统（ASR + MT），存在错误传播问题。近年多模态大语言模型（MLLM）在简化架构和减少错误传播方面展现优势，但面临两个关键挑战：

**数据稀缺**：现有 S2TT 数据集以英语为中心（如 MuST-C），支持多对多翻译的数据集（如 FLEURS）每种语言仅约 10 小时

**能力迁移问题**：LLM 已具备强大的多语言机器翻译能力，但如何在有限数据下将这种能力迁移到 S2TT 任务尚未解决

### 核心洞察

LLM 已经具备"翻译"能力（MT），缺的是"听"的能力（Speech ↔ Text 对齐）。如果能通过少量数据教会 MLLM "听懂"语音并连接到已有的翻译能力，就能实现低资源下的多对多 S2TT。

### 创新观点

将 S2TT 任务重新定义为 SRT（Speech Recognition and Translation）任务：输入语音，**同时输出转录和翻译**。这种设计使 MLLM 能在推理时先产出转录再利用 LLM 内在的 MT 能力翻译，结合了级联和端到端两种方法的优势。

## 方法详解

### 整体框架

LLM-SRT 的架构由三部分组成：
- **语音编码器**：冻结的 Whisper-large-v3，提取语音特征
- **语音适配器**：Q-Former + MLP，压缩语音特征维度并对齐到 LLM 隐空间
- **LLM**：Qwen 系列（3B/7B/32B），冻结或 LoRA 微调

### 关键设计

1. **三阶段课程学习（Curriculum Learning）**

   **第一阶段：ASR（语音识别）**  
    - 目标：多模态对齐，让模型学会"听"
    - 输入：语音 + 语言标签指令（如 `<|eng|>`）
    - 输出：转录文本
    - 训练所有目标语言，使用 Common Voice 数据集
    - 仅训练语音适配器

   **第二阶段：SMT（语音辅助机器翻译）**  
    - 目标：激活 LLM 的跨语言翻译能力
    - 输入：语音 + 转录文本 + 翻译指令（如 `Will it rain tomorrow?<|eng|><|deu|>`）
    - 输出：翻译文本
    - 建立 MT 与 S2TT 的连接桥梁
    - 从 ASR checkpoint 继续，仅训练适配器

   **第三阶段：SRT（语音识别与翻译）**  
    - 目标：最终激活端到端 S2TT 能力
    - 输入：语音 + 任务指令（如 `<|eng|><|deu|>`）
    - 输出：转录 + 翻译（如 `Will it rain tomorrow?<|eng|><|deu|>Regnet es morgen?`）
    - 从 SMT checkpoint 继续，可选择解冻 LLM（LoRA）

2. **极简指令设计**

    - ASR：`<|eng|>` 表示"识别英语"
    - SMT：`转录文本<|源语言|><|目标语言|>` 表示"将此翻译"
    - SRT：`<|源语言|><|目标语言|>` 表示"先识别再翻译"
    - 语言标签同时出现在指令和生成答案中，自然分隔转录和翻译内容
    - 设计动机：减少指令 token 长度，提高效率

3. **语音适配器优化**

    - Q-Former 将变长语音特征压缩为固定长度 80 个 query（$\mathbf{Q'} \in \mathbb{R}^{n_q \times D_q}$）
    - MLP 将 Q-Former 输出映射到 LLM 隐空间维度（$\mathbf{E^X} \in \mathbb{R}^{n_q \times d_{LLM}}$）
    - 相比 Qwen2-Audio 的变长特征，固定长度大幅减少 LLM 输入 token 数
    - 推理速度提升约 3 倍，且支持更大 batch size

### 训练策略

- 三阶段渐进式训练，每阶段从上一阶段 checkpoint 继续
- 语音编码器始终冻结
- 阶段 1-2：仅训练适配器
- 阶段 3：可选额外解冻 LLM（LoRA）
- 使用 bf16、DDP，lr=1e-4，AdamW 优化器
- 4 卡 A100：3B/7B 模型约 3 天，32B 模型约 7 天

## 实验关键数据

### 主实验 — FLEURS 6×12 方向 BLEU

| 模型 | S2TT 数据量 | Eng | Deu | Fra | Jpn | Zho | 平均 |
|------|-----------|-----|-----|-----|-----|-----|------|
| Whisper+Qwen2.5-32B (级联) | - | 29.9 | 26.0 | 24.6 | 17.3 | 18.5 | 23.3 |
| SeamlessM4T-V2 (2.3B) | 351K小时 | 33.1 | 20.5 | 19.6 | 13.2 | 15.2 | 20.2 |
| Qwen2-Audio (7B) | 内部 | 22.6 | 20.1 | 20.6 | 4.0 | 13.7 | 16.0 |
| Baseline-3B | 52小时 | 11.8 | 9.0 | 9.5 | 5.2 | 6.2 | 8.6 |
| **LLM-SRT-3B** | **52小时** | **27.2** | **22.6** | **22.0** | **14.3** | **16.5** | **20.6** |
| **LLM-SRT-32B** | **52小时** | **32.5** | **26.8** | **26.1** | **17.5** | **19.2** | **24.6** |

仅用 52 小时数据的 LLM-SRT-3B（BLEU 20.6）就超越了用 351K 小时训练的 SeamlessM4T-V2（20.2）。

### 消融实验 — 课程学习各阶段的影响（CoVoST-2）

| 设置 | Deu | Jpn | Zho | 平均 |
|------|-----|-----|-----|------|
| LLM-SRT-7B（完整） | 28.7 | 41.6 | 47.1 | 39.1 |
| w/o ASR | 26.4 (-2.3) | 38.6 (-3.0) | 45.5 (-1.6) | 36.8 (-2.3) |
| w/o SMT | 27.6 (-1.1) | 39.7 (-1.9) | 46.5 (-0.6) | 38.0 (-1.1) |
| w/o SRT | 25.6 (-3.1) | 36.7 (-4.9) | 40.4 (-6.7) | 34.2 (-4.9) |

去除 SRT 阶段影响最大（-4.9），验证了将 MT 能力延伸到 S2TT 的关键作用。

### 推理速度对比

| 模型 | 策略 | Batch | 处理1000条用时 |
|------|------|-------|-------------|
| Qwen2-Audio | Greedy | 4 | 59s |
| Qwen2-Audio | Greedy | 8 | OOM |
| LLM-SRT-7B | Greedy | 4 | 74s |
| LLM-SRT-7B | Greedy | 64 | 19s |
| LLM-SRT-7B | Beam 5 | 12 | 56s |

LLM-SRT 在大 batch 下推理速度约为 Qwen2-Audio 的 **3倍**，且不会 OOM。

### 关键发现

1. **课程学习效果显著**：Baseline-3B 直接微调仅得 8.6 BLEU，LLM-SRT-3B 通过三阶段策略提升到 20.6（+140%）
2. **遵循 LLM 缩放律**：3B→7B→32B 性能稳步提升（20.6→21.4→24.6）
3. **S2TT 性能与 MT 强相关**：210 个翻译方向上 S2TT 与 MT 的 BLEU 呈强正相关
4. **数据量扩展有效**：从 52 小时到 430 小时，3B 模型 34.3→36.6，7B 模型 35.0→39.1
5. **SRT 任务不损害 ASR**：SRT 甚至略微提升了 ASR 性能（WER 11.1→10.9）
6. **SMT 任务验证**：给定正确转录时 SMT 可达很高 BLEU（如 Zho 55.6），证明 LLM 的 MT 能力被有效激活

## 亮点与洞察

- **任务重定义的巧妙性**：将 S2TT 转化为 SRT，让模型"先听后翻"，自然利用 LLM 的 MT 能力
- **极致的数据效率**：52 小时多语言数据（每语言 <10h）即可超越 35 万小时训练的 SeamlessM4T-V2
- **端到端优于级联**：使用相同 LLM 时，LLM-SRT 的端到端方式优于 Whisper+LLM 级联方式
- **Q-Former 压缩的实际价值**：固定 80 query 不仅提升推理速度 3x，还解决了 Qwen2-Audio batch>4 即 OOM 的问题
- **首个 32B 规模的多对多 S2TT MLLM**

## 局限与展望

1. **英语翻译方向较弱**：FLEURS 每语言仅 ~10h，英语数据不足导致 Eng→X 不如 SeamlessM4T-V2
2. **语音编码器冻结**：未探索解冻 Whisper 的潜在收益
3. **Q-Former 的信息瓶颈**：固定 80 个 query 是否会损失关键语音信息值得进一步研究
4. **评估指标局限**：仅用 BLEU，未使用 COMET 等更鲁棒的翻译评估指标
5. **流式推理**：当前方法需要完整音频才能处理，不支持实时流式翻译

## 相关工作与启发

- 与 Qwen2-Audio 的关系：LLM-SRT 采用相似架构但通过课程学习策略大幅提升了性能
- 与 SeamlessM4T 的比较：后者依赖海量数据（351K 小时），LLM-SRT 展示了 LLM 内在 MT 能力的数据效率优势
- 启发：课程学习的"能力迁移"思路（MT→S2TT）可能适用于其他跨模态任务（如图像描述→视频描述）

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | ⭐⭐⭐⭐ | SRT 任务定义和三阶段课程学习策略新颖有效 |
| 实验充分度 | ⭐⭐⭐⭐⭐ | 多数据集、多模型规模、多方向、消融完整 |
| 写作质量 | ⭐⭐⭐⭐ | 结构清晰，表格丰富 |
| 实用价值 | ⭐⭐⭐⭐⭐ | 低资源多语言翻译有重大应用价值 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] An Effective Incorporating Heterogeneous Knowledge Curriculum Learning for Sequence Labeling](dual_stage_curriculum_learning_sequence_labeling.md)
- [\[ACL 2025\] Pre-Training Curriculum for Multi-Token Prediction in Language Models](pre-training_curriculum_for_multi-token_prediction_in_language_models.md)
- [\[ACL 2025\] InSerter: Speech Instruction Following with Unsupervised Interleaved Pre-training](inserter_speech_instruction.md)
- [\[ACL 2025\] Splintering Nonconcatenative Languages for Better Tokenization](splintering_nonconcatenative_languages_for_better_tokenization.md)
- [\[ACL 2025\] LEANCODE: Understanding Models Better for Code Simplification of Pre-trained Large Language Models](leancode_understanding_models_better_for_code_simplification_of_pre-trained_larg.md)

</div>

<!-- RELATED:END -->
