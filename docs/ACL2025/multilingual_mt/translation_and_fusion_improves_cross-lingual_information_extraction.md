---
title: >-
  [论文解读] Translation and Fusion Improves Zero-shot Cross-lingual Information Extraction
description: >-
  [ACL 2025][跨语言迁移] 提出 TransFusion 框架，通过在推理时先将低资源语言文本翻译为英语、在英语上进行信息抽取标注、再用融合模型将英语标注与原文结合来生成最终预测，在50种语言的零样本跨语言IE任务上显著优于基线（MasakhaNER2平均F1从47.9提升到62.4）。
tags:
  - ACL 2025
  - 跨语言迁移
  - 信息抽取
  - 机器翻译
  - 低资源语言
  - 标注融合
---

# Translation and Fusion Improves Zero-shot Cross-lingual Information Extraction

**会议**: ACL 2025  
**arXiv**: [2305.13582](https://arxiv.org/abs/2305.13582)  
**代码**: https://github.com/edchengg/gollie-transfusion  
**领域**: NLP理解  
**关键词**: 跨语言迁移, 信息抽取, 机器翻译, 低资源语言, 标注融合

## 一句话总结

提出 TransFusion 框架，通过在推理时先将低资源语言文本翻译为英语、在英语上进行信息抽取标注、再用融合模型将英语标注与原文结合来生成最终预测，在50种语言的零样本跨语言IE任务上显著优于基线（MasakhaNER2平均F1从47.9提升到62.4）。

## 研究背景与动机

**领域现状**：大语言模型结合指令微调（如GoLLIE）已在信息抽取任务上展现出强大的零样本泛化能力，能够根据标注指南对未见过的数据集进行IE。然而，这些模型通常以英语为中心进行预训练，对低资源语言的处理能力严重不足。

**现有痛点**：即使是GPT-4这样的顶级模型，在非洲低资源语言上的NER性能也从英语的80 F1骤降至55 F1。传统的跨语言迁移方法（如translate-train和translate-test）虽然有一定效果，但在信息抽取这类span级别标注任务中，翻译后的文本与原始标注之间存在对齐困难，不能简单地套用已有方案。

**核心矛盾**：低资源语言缺乏标注数据进行微调，也缺乏足够的无标注文本用于预训练。而机器翻译模型（如NLLB-200）虽然支持200种语言的翻译，但翻译质量参差不齐，简单的translate-test或translate-train无法充分利用翻译信息。

**本文目标** 如何让以英语为中心的IE模型，在不需要目标语言标注数据的情况下，利用外部MT系统显著提升低资源语言上的信息抽取性能。

**切入角度**：作者观察到，虽然直接在翻译后的英语文本上做IE可以获得较好的英语预测，但这些预测可能因为翻译噪声而存在误差。如果让模型学会"融合"英语版预测和原始低资源语言文本的信息，就能两头获益。

**核心 idea**：训练模型在推理时先翻译→再标注英语版→最后融合英语标注与原文，通过三步自回归推理链实现跨语言IE的显著提升。

## 方法详解

### 整体框架

TransFusion 是一个三步推理框架：(1) Translate：用NLLB-200将低资源语言测试数据翻译为英语；(2) Annotate：用英语IE模型对翻译后文本进行标注；(3) Fuse：融合模型结合英语标注结果和原始低资源语言文本，生成最终预测。整个流程在decoder-only LLM中被实现为一个单次自回归解码过程。

### 关键设计

1. **TransFusion推理链（Autoregressive Annotate-and-Fuse）**:

    - 功能：将翻译、标注、融合三步统一到一次自回归生成中
    - 核心思路：在GoLLIE的Python代码表示格式基础上，将TransFusion指令嵌入prompt中。模型接收"标注指南+目标语言文本+英语翻译+TransFusion指令"作为输入，先自回归生成英语翻译的IE标注 $\tilde{y}_{src}^{trans}$，再基于这些标注生成目标语言的最终预测 $y_{tgt}$
    - 设计动机：统一为一次解码避免了pipeline中的错误累积，且让融合模型能直接看到英语标注的上下文来校正目标语言预测

2. **跨语言训练数据构建（EasyProject标注投影）**:

    - 功能：自动生成TransFusion训练所需的双语平行IE数据
    - 核心思路：利用EasyProject的mark-then-translate方法，将英语IE训练数据翻译到36种目标语言，同时投影span级别标注。构建混合数据集 $\mathcal{D}_{mix} = \{x_{src}, y_{src}, x_{tgt}^{trans}, y_{tgt}^{trans}\}$，其中英语数据19,109条、翻译数据仅891条，总共约20,000条
    - 设计动机：只需极少量翻译数据（每语言每任务仅8个样本），配合大量英语数据，即可实现高效的跨语言迁移，同时保持英语性能和对未见标签schema的泛化能力

3. **Encoder-only模型的两步TransFusion**:

    - 功能：将TransFusion框架扩展到encoder-only架构（如AfroXLM-R）
    - 核心思路：由于encoder模型不能生成文本，采用两步pipeline：先用英语微调的模型标注翻译文本（用XML标签标注span），再将标注后的英语翻译与原文拼接输入融合模型，用分隔符"||"分开，仅对目标语言部分计算分类损失
    - 设计动机：验证TransFusion框架不依赖于特定模型架构，在encoder-only模型上也能达到SOTA

### 损失函数 / 训练策略

对于decoder-only模型（GoLLIE-TF），基于GoLLIE-7B使用QLoRA继续微调，LoRA rank=128，alpha=16，学习率1e-4，batch size=16，采用cosine调度器。在2块NVIDIA A40 GPU上训练约6小时。推理时使用greedy decoding。

融合训练损失为条件语言建模损失：$\mathcal{L}_{fusion}(\theta, \mathcal{D}_{mix}) = \sum \mathcal{L}(P(y | x_{tgt}^{trans}, x_{src}, y_{src}; \theta_{fusion}), y_{tgt}^{trans})$，仅在TransFusion指令之后的token上计算next-token prediction loss。

## 实验关键数据

### 主实验

| 数据集 | 指标 | GoLLIE-TF | GoLLIE-7B | GPT-4 | 提升(vs GoLLIE) |
|--------|------|-----------|-----------|-------|------|
| MasakhaNER2 (20语言) | F1 | 62.4 | 47.9 | 54.2 | +14.5 |
| UNER (13语言) | F1 | 77.8 | 73.6 | 69.0 | +4.2 |
| ACE05 NER (en/ar/zh) | F1 | 61.5 | 58.7 | 41.6 | +2.8 |
| MultiCoNER2 (12语言, unseen) | F1 | 34.5 | 22.2 | 46.1 | +12.2 |
| Massive (15低资源语言, unseen) | F1 | 19.0 | 5.8 | 33.3 | +13.1 |
| 所有数据集平均 | F1 | 45.7 | 40.2 | 36.6 | +5.5 |

### 消融实验

| 配置 | MasakhaNER2 F1 | 说明 |
|------|---------|------|
| GoLLIE-TF (完整) | 62.4 | 包含annotate+fuse |
| w/o annotate | 55.7 (-6.7) | 直接从未标注英语翻译生成，验证英语标注的关键作用 |
| AfroXLM-R (TransFusion) | 72.1 | Encoder-only模型也有效 |
| AfroXLM-R (Trans-train) | 65.8 | TransFusion优于简单translate-train |
| AfroXLM-R (基线) | 58.8 | 无翻译增强 |

### 关键发现

- TransFusion在低资源语言上的提升远大于高资源语言：MasakhaNER2（低资源）提升14.5 F1，UNER（混合）仅提升4.2 F1
- 对翻译质量有一定鲁棒性：使用不同大小的NLLB模型（600M/1.3B/3.3B）时，性能差异不大，但更强的翻译模型仍带来轻微提升
- 错误分析显示31个错误中22个来自英语预测阶段、12个来自融合阶段，说明继续提升英语IE模型是最大的改进方向
- TransFusion也能以prompting方式应用于GPT-4，在MasakhaNER2上F1从53.4提升到62

## 亮点与洞察

- **翻译+标注+融合的三步统一解码**设计非常巧妙，避免了多阶段pipeline的误差传播，同时利用了自回归模型的上下文建模能力。这种"先在简单版本上做，再融合校正"的范式可以推广到其他跨模态/跨领域任务
- 仅需极少量翻译数据（每语言每任务8个样本）就能实现显著提升，说明TransFusion学到的是一种通用的"翻译→标注→融合"推理模式，而非记忆特定语言的映射
- 框架的架构无关性很强：在decoder-only（GoLLIE）、encoder-only（AfroXLM-R）和proprietary model（GPT-4）上都有效

## 局限与展望

- 依赖外部MT系统的可用性，对于MT系统不支持的极低资源语言可能无效
- 推理时增加了翻译步骤的额外延迟和成本
- 错误分析表明71%的错误源自英语预测阶段，未来应考虑如何让融合模型更好地纠正上游错误
- 当前仅在IE任务上验证，可扩展到其他NLP任务如关系抽取、事件检测等更复杂的场景

## 相关工作与启发

- **vs Translate-Train**: TransFusion不仅在翻译数据上训练，还学习了推理时的annotate-fuse推理链，MasakhaNER2上高出Trans-Train近10个F1点
- **vs GPT-4 零样本**: GoLLIE-TF (7B) 在seen label schema上超越GPT-4（61.8 vs 33.7），在低资源语言NER上同样更优，说明小模型通过专门训练可以超过大模型的零样本能力
- **vs Codec**: 在AfroXLM-R上的TransFusion (72.1 F1) 超越了之前SOTA Codec (70.1 F1)，后者使用受约束解码进行翻译模型标签投影

## 评分

- 新颖性: ⭐⭐⭐⭐ TransFusion的三步推理链设计新颖，但翻译辅助跨语言迁移本身是已有思路
- 实验充分度: ⭐⭐⭐⭐⭐ 50种语言、12个数据集、多种架构验证，非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，实验设置详尽，但方法部分公式较多影响可读性
- 价值: ⭐⭐⭐⭐ 对低资源语言IE的实际应用价值很高，框架通用性强
