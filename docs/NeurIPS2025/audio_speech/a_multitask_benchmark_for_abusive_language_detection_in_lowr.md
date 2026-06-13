---
title: >-
  [论文解读] A Multi-Task Benchmark for Abusive Language Detection in Low-Resource Settings
description: >-
  [NeurIPS 2025][音频/语音][辱骂检测] 提出 TiALD（Tigrinya Abusive Language Detection），首个面向 Tigrinya 低资源语言的大规模多任务基准数据集，包含 13,717 条 YouTube 评论的辱骂/情感/主题三任务联合标注，同时发现小型微调模型（TiRoBERTa, 125M）在所有任务上全面超越 GPT-4o 和 Claude Sonnet 3.7 等前沿 LLM。
tags:
  - "NeurIPS 2025"
  - "音频/语音"
  - "辱骂检测"
  - "低资源语言"
  - "多任务学习"
  - "Tigrinya"
  - "基准数据集"
---

# A Multi-Task Benchmark for Abusive Language Detection in Low-Resource Settings

**会议**: NeurIPS 2025  
**arXiv**: [2505.12116](https://arxiv.org/abs/2505.12116)  
**代码**: [https://github.com/fgaim/TiALD](https://github.com/fgaim/TiALD)  
**领域**: 音频语音  
**关键词**: 辱骂检测、低资源语言、多任务学习、Tigrinya、基准数据集

## 一句话总结

提出 TiALD（Tigrinya Abusive Language Detection），首个面向 Tigrinya 低资源语言的大规模多任务基准数据集，包含 13,717 条 YouTube 评论的辱骂/情感/主题三任务联合标注，同时发现小型微调模型（TiRoBERTa, 125M）在所有任务上全面超越 GPT-4o 和 Claude Sonnet 3.7 等前沿 LLM。

## 研究背景与动机

**领域现状**：内容审核研究在英语等高资源语言上取得了显著进展，自动辱骂检测已相当成熟。但全球大多数语言仍处于"计算资源荒漠"，缺乏标注数据、工具和模型，使得数百万用户暴露在未经审核的网络敌意中。

**现有痛点**：Tigrinya 是一种约有 1000 万使用者的语言（主要在 Eritrea 和 Ethiopia），计算资源极度匮乏——缺少标注数据集、缺少评估基准、缺少训练好的模型。没有基准就无法衡量进展，无法推动研究。更棘手的是，约 64% 的 Tigrinya 社交媒体内容使用**拉丁音译**而非原生 Ge'ez 文字，现有工具几乎不覆盖这种书写方式。

**核心矛盾**：低资源语言的辱骂检测面临"没数据→没模型→没保护→没动力建数据"的恶性循环。简单的关键词采样会导致词汇同质化，随机采样则辱骂样本占比极低（仅 14.3%），都不适合构建高质量训练集。

**本文目标** (1) 构建首个覆盖 Tigrinya 两种书写系统的多任务标注基准；(2) 设计适合低资源环境的数据采样策略；(3) 评估微调模型 vs LLM 在低资源场景下的表现差异。

**切入角度**：从数据构建出发，通过迭代式种子词扩展策略解决采样偏差问题，通过多任务联合标注提供更丰富的上下文信号。

**核心 idea**：用迭代语义聚类采样 + 多任务联合标注，为低资源语言构建首个高质量辱骂检测基准。

## 方法详解

### 整体框架

TiALD 的构建流程：(1) 从 51 个 YouTube 频道收集 410 万条 Tigrinya 评论；(2) 用迭代种子词扩展策略从中选取 20K 条代表性评论；(3) 9 名母语标注员对 13,717 条评论进行辱骂/情感/主题三任务标注；(4) 900 条测试集通过三人标注 + 专家裁决获得金标签。基线实验覆盖单任务微调、多任务联合学习和 LLM zero/few-shot 三种范式。

### 关键设计

1. **迭代种子词扩展采样策略**:

    - 功能：从大规模未标注语料中高效选取辱骂内容丰富且词汇多样的标注候选集
    - 核心思路：在 410 万条评论上训练 word2vec（CBOW, 300 维），从 61 个种子词出发进行三阶段迭代扩展——第一阶段每个种子词取 50 个最近邻并过滤形态变体，第二阶段每个新词取 25 个近邻，第三阶段每个词取 10 个近邻。最终得到 8,728 个多样化术语，从中选取 15K 条覆盖评论 + 5K 随机控制组
    - 设计动机：纯关键词搜索的 type-to-token ratio 仅 0.18，随机采样仅 0.13，迭代扩展达到 0.28。辱骂样本占比从随机采样的 14.3% 提升到 65.2%

2. **双文字系统覆盖**:

    - 功能：同时覆盖原生 Ge'ez 文字和拉丁音译文字
    - 核心思路：标注数据按 70% Ge'ez + 30% Latin/混合比例分配，使用 GeezSwitch 库进行文字系统识别和过滤
    - 设计动机：64% 的 Tigrinya 社交媒体内容使用非标准拉丁音译，只覆盖 Ge'ez 文字的模型实际上遗漏了大部分内容

3. **多任务联合学习框架**:

    - 功能：用共享编码器同时学习辱骂检测（二分类）、情感分析（四分类）和主题分类（五分类）
    - 核心思路：共享 Transformer 编码器输出 $\mathbf{h} = \text{Encoder}(x)$，单个线性头映射到 $L=2+4+5=11$ 个标签的 logit，每个标签用 sigmoid + 0.5 阈值做二值预测，用平均 BCE 损失训练
    - 设计动机：辱骂性、情感和主题之间存在互补语言特征（如政治话题常伴随负面情感和辱骂内容），联合学习可以通过参数共享捕获这些关联

### 评估指标

引入 TiALD Score = 三个任务 macro F1 的平均值，作为基准级别的综合评估指标。

## 实验关键数据

### 主实验（微调模型 vs LLM）

| 模型 | 类型 | 辱骂 F1↑ | 情感 F1↑ | 主题 F1↑ | TiALD Score↑ |
|------|------|----------|----------|----------|-------------|
| TiRoBERTa（单任务） | 微调 125M | **86.67** | 52.82 | 54.23 | 64.57 |
| TiRoBERTa（多任务） | 微调 125M | 86.11 | **53.41** | **54.91** | **64.81** |
| AfroXLMR-76L | 微调 560M | 85.20 | 54.94 | 51.42 | 63.86 |
| GPT-4o（few-shot） | LLM | 72.06 | 21.88 | 27.56 | 40.50 |
| Claude 3.7（few-shot） | LLM | 79.31 | 23.39 | 27.92 | 43.54 |
| Gemma-3 4B（few-shot） | LLM | 58.37 | 30.46 | 39.49 | 42.78 |

### 消融：视频上下文的影响

| 设置 | GPT-4o 辱骂 F1 | Claude 3.7 辱骂 F1 | 说明 |
|------|---------------|-------------------|------|
| 纯评论（zero-shot） | 71.05 | 59.20 | 基线 |
| + 视频标题 | 75.59 | 67.64 | +4.5 / +8.4 |
| + 标题 + 视频描述 | 74.70 | 72.02 | Claude 提升显著 |

### 关键发现

- **小模型碾压大模型**：125M 的 TiRoBERTa 在辱骂检测上比 GPT-4o 高 14.6 个百分点，在 TiALD Score 上高 24 个百分点。低资源语言下微调仍然是王道
- **LLM 在多类别任务上崩溃**：LLM 在二分类辱骂检测上尚可（71-79%），但在四分类情感和五分类主题上严重退化（最高仅 30% 和 39%），说明 LLM 的低资源多类别分类能力根本不足
- **多任务学习一致性提升**：对几乎所有模型，联合训练都带来提升，TiELECTRA-small 提升最大（+1.76 TiALD Score）
- **LLaMA-3.2 3B 分类偏差翻转**：zero-shot 时将 68% 评论判为辱骂，few-shot 时反转为 77% 判为非辱骂，暴露了小 LLM 在分布外语言上的严重不稳定性
- **视频上下文对 LLM 有效**：添加视频标题和描述后 Claude 的 zero-shot 辱骂检测 F1 提升 12.8 个百分点

## 亮点与洞察

- **迭代种子词扩展策略巧妙**：三阶段迭代 + 形态去重实现了高覆盖、高多样性的采样，type-to-token ratio 比随机采样高 2 倍。这个方法可以直接迁移到任何低资源语言的标注前数据筛选
- **双文字系统的现实考量**：大多数低资源 NLP 工作忽略了社交媒体上的非标准书写实践，TiALD 是少数正视这一问题的工作。这个经验对其他使用多种文字系统的语言（如印地语的 Devanagari/Latin）同样适用
- **小模型 > 大模型的定量证据**：在低资源场景下提供了 LLM 不如微调小模型的清晰证据链，有助于指导实际部署决策

## 局限与展望

- 数据仅来自 YouTube 评论，未覆盖 Twitter/Facebook/Telegram 等其他平台，泛化性存疑
- 标注员仅 9 人，主题分类的 IAA 偏低，说明标注指南对主题的界定还不够清晰
- 多任务框架目前使用等权损失，per-task 或 per-class 加权可能缓解类别不平衡问题
- 未探索数据增强（如回译、跨文字系统转换）对低资源性能的影响
- VLM 生成的视频描述质量未充分验证，引入的噪声可能影响分析结论

## 相关工作与启发

- **vs 现有低资源 NLP 基准**：已有的非洲语言 NLP 工作主要关注 NER 和机器翻译，TiALD 是首个聚焦辱骂检测的多任务基准
- **vs 高资源辱骂检测**：英语辱骂检测已经达到 90%+ F1，但方法直接迁移到 Tigrinya 效果很差（XLM-R 仅 81%），说明语言特异性仍然重要
- **vs LLM-as-judge 范式**：本文结果表明大模型在低资源语言上的 zero/few-shot 能力远不如预期，对"LLM 替代一切"的叙事提供了重要反例

## 评分

- 新颖性: ⭐⭐⭐ 数据集类工作，核心贡献在于填补空白而非方法创新
- 实验充分度: ⭐⭐⭐⭐ 覆盖单任务/多任务/LLM 三种范式，消融分析细致
- 写作质量: ⭐⭐⭐⭐ 排版清晰，动机阐述充分，数据构建过程透明
- 价值: ⭐⭐⭐⭐ 对低资源 NLP 社区有重要参考价值，迭代采样策略可广泛复用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] GigaSpeech 2: An Evolving, Large-Scale and Multi-domain ASR Corpus for Low-Resource Languages](../../ACL2025/audio_speech/gigaspeech2_low_resource_asr.md)
- [\[ICLR 2026\] MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark](../../ICLR2026/audio_speech/mmsu_a_massive_multi-task_spoken_language_understanding_and_reasoning_benchmark.md)
- [\[ACL 2026\] Multimodal In-Context Learning for ASR of Low-Resource Languages](../../ACL2026/audio_speech/multimodal_in-context_learning_for_asr_of_low-resource_languages.md)
- [\[ACL 2026\] HCFD: A Benchmark for Audio Deepfake Detection in Healthcare](../../ACL2026/audio_speech/hcfd_a_benchmark_for_audio_deepfake_detection_in_healthcare.md)
- [\[NeurIPS 2025\] Multi-head Temporal Latent Attention](multi-head_temporal_latent_attention.md)

</div>

<!-- RELATED:END -->
