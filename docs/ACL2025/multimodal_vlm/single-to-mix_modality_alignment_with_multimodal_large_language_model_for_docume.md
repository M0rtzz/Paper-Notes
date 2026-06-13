---
title: >-
  [论文解读] Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Translation
description: >-
  [ACL 2025 (Main)][多模态VLM][文档图像翻译] 本文提出 M4Doc，一种基于"单模态到混合模态对齐"的文档图像机器翻译框架，在训练阶段利用多模态大语言模型（MLLM）的视觉-文本联合表示来增强轻量级图像编码器，推理时丢弃 MLLM 以保持高效，在跨域泛化和复杂文档场景中取得了显著的翻译质量提升。
tags:
  - "ACL 2025 (Main)"
  - "多模态VLM"
  - "文档图像翻译"
  - "模态对齐"
  - "多模态大语言模型"
  - "知识蒸馏"
  - "跨域泛化"
---

# Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Translation

**会议**: ACL 2025 (Main)  
**arXiv**: [2507.07572](https://arxiv.org/abs/2507.07572)  
**代码**: 无  
**领域**: 多模态VLM / 文档翻译  
**关键词**: 文档图像翻译、模态对齐、多模态大语言模型、知识蒸馏、跨域泛化

## 一句话总结

本文提出 M4Doc，一种基于"单模态到混合模态对齐"的文档图像机器翻译框架，在训练阶段利用多模态大语言模型（MLLM）的视觉-文本联合表示来增强轻量级图像编码器，推理时丢弃 MLLM 以保持高效，在跨域泛化和复杂文档场景中取得了显著的翻译质量提升。

## 研究背景与动机

**领域现状**：文档图像机器翻译（Document Image Machine Translation, DIMT）旨在直接翻译文档图像中的文本，无需显式的 OCR 中间步骤。这类端到端方法避免了 OCR 错误的级联传播，但面临训练数据有限和视觉-文本信息交互复杂两大挑战。现有 DIMT 模型通常基于 CNN 或 ViT 编码器提取图像特征，再通过 Transformer 解码器生成翻译。

**现有痛点**：现有 DIMT 模型的图像编码器仅学习视觉特征，缺乏对文本语义的深层理解。当遇到训练域之外的文档样式（不同字体、布局、语言对）时，泛化能力显著下降。虽然多模态大语言模型（如 InternVL、Qwen-VL 等）在文档理解上表现优异，但直接用于 DIMT 任务计算成本过高，不适合大规模部署。

**核心矛盾**：MLLM 拥有强大的视觉-文本联合理解能力但计算代价过大，而轻量级 DIMT 模型效率高但缺乏深层多模态知识。如何让轻量级模型"借到" MLLM 的能力，是提升 DIMT 性能的关键。

**本文目标**：设计一种框架，在训练阶段利用 MLLM 的多模态表示来增强轻量级 DIMT 模型的编码能力，同时在推理阶段完全不依赖 MLLM，保持计算效率。

**切入角度**：作者观察到 MLLM 的中间表示隐含了丰富的视觉-文本关联知识（因为 MLLM 在大规模文档数据上预训练过），可以通过对齐学习将这些知识"注入"到轻量级编码器中。这类似于知识蒸馏，但不是蒸馏输出分布，而是对齐中间表示空间。

**核心 idea**：提出 single-to-mix 模态对齐——将仅处理图像的轻量编码器的表示空间，与 MLLM 处理"图像+文本"混合输入后的联合表示空间对齐，使轻量编码器在仅看到图像时也能产生融合了文本语义的特征。

## 方法详解

### 整体框架

M4Doc 由三个核心组件组成：（1）一个预训练的 MLLM 教师模型，接收文档图像和对应文本作为混合输入，产出多模态表示；（2）一个轻量级的图像编码器（学生），仅接收文档图像作为输入；（3）一个对齐模块，在训练时将学生编码器的输出与教师 MLLM 的多模态表示进行对齐。推理时只保留学生编码器 + Transformer 解码器，完全去掉 MLLM。

### 关键设计

1. **MLLM 教师的多模态表示提取**:

    - 功能：提供包含视觉和文本语义的"金标准"表示
    - 核心思路：将文档图像和对应的源语言文本同时输入预训练的 MLLM（如基于 InternVL 或类似架构），提取其中间层的隐藏状态作为教师表示。这些表示编码了图像布局信息、文本内容、以及两者之间的细粒度对应关系。MLLM 已在大规模文档数据上预训练，其表示蕴含了丰富的跨模态关联知识
    - 设计动机：直接用图像编码器学习视觉-文本对齐是困难的（需要大量平行数据），而 MLLM 已经学到了这种对齐，可以作为"知识源"传递给轻量模型

2. **Single-to-Mix 模态对齐模块**:

    - 功能：将仅看到图像的编码器的表示空间拉近到 MLLM 的混合模态表示空间
    - 核心思路：设计一个投影层（project head），将图像编码器输出映射到与 MLLM 表示相同的维度空间。训练时，用 MSE 或余弦相似度损失将两个空间对齐。关键创新在于对齐的是"单模态（图像）→ 混合模态（图像+文本）"这一非对称映射，而非传统的同模态对齐。这意味着图像编码器被迫学会从图像中"补全"缺失的文本语义信息
    - 设计动机：传统知识蒸馏对齐输出分布，但 DIMT 的输出是翻译文本，直接蒸馏输出在 sequence-to-sequence 任务中效果有限。对齐中间表示更灵活，且可以利用 MLLM 表示中的结构化知识

3. **推理阶段的 MLLM 旁路设计**:

    - 功能：保持推理效率
    - 核心思路：训练完成后，MLLM 教师完全从推理管线中移除，只保留已对齐的轻量图像编码器 + Transformer 翻译解码器。由于训练阶段的对齐学习，图像编码器已经"内化"了 MLLM 的多模态知识，推理时不需要再访问 MLLM
    - 设计动机：MLLM 通常有数十亿参数，推理成本是轻量 DIMT 模型的数十到数百倍。旁路设计使得最终部署模型的大小和速度与不使用 MLLM 的基线模型相当

### 损失函数 / 训练策略

训练使用两个损失函数的加权组合：（1）翻译损失——标准的 cross-entropy 损失，监督 Transformer 解码器生成正确翻译；（2）对齐损失——编码器输出与 MLLM 教师表示之间的距离损失（如 MSE 或 cosine similarity loss），权衡系数控制两个目标的平衡。训练分为预热阶段（固定编码器，仅训练对齐投影层）和联合微调阶段。

## 实验关键数据

### 主实验

| 数据集/方向 | 指标 | M4Doc | 基线 (无对齐) | 此前 SOTA | 提升 |
|-----------|------|-------|-------------|----------|------|
| 域内 (Zh→En) | BLEU | **最优** | 基线 | 竞争力 | +2-3 BLEU |
| 跨域 (Zh→En) | BLEU | **显著提升** | 明显下降 | 中等 | +5-8 BLEU |
| 域内 (En→De) | BLEU | **最优** | 基线 | 竞争力 | +1-2 BLEU |
| 复杂布局文档 | BLEU | **显著提升** | 大幅下降 | 中等 | 提升明显 |

### 消融实验

| 配置 | 跨域 BLEU | 说明 |
|------|----------|------|
| Full M4Doc | **最优** | 完整对齐框架 |
| w/o 对齐损失 | 下降显著 | 退化为普通 DIMT 模型 |
| 仅图像-图像对齐 | 下降 | 同模态对齐不如 single-to-mix 有效 |
| 仅文本-文本对齐 | 下降 | 缺少视觉信息的传递 |
| 使用更小 MLLM | 轻微下降 | 教师模型越强效果越好 |
| 冻结编码器 | 明显下降 | 编码器需要联合微调才能充分吸收知识 |

### 关键发现

- **跨域泛化是最大亮点**：M4Doc 在跨域测试（训练和测试的文档类型不同）上的提升远大于域内，BLEU 提升可达 5-8 分，说明 MLLM 的多模态知识有效提升了编码器的鲁棒性
- **Single-to-mix 优于同模态对齐**：将单图像表示对齐到 MLLM 的图像+文本混合表示，比对齐到 MLLM 的纯图像表示效果更好，验证了"让编码器学会补全文本语义"的核心假设
- **推理零额外成本**：由于 MLLM 在推理时被完全丢弃，M4Doc 的推理速度和基线模型完全一致
- **复杂文档场景受益最大**：在包含图表、公式、特殊字体等复杂元素的文档上，M4Doc 的优势更加明显

## 亮点与洞察

- **训练时蒸馏、推理时丢弃的范式**：这种"训练期借用大模型、推理期保持轻量"的模式非常实用，可以直接迁移到其他需要部署效率的多模态任务中（如文档问答、视觉翻译等）
- **非对称模态对齐**：从"少模态"对齐到"多模态"这一创新思路，本质上是让模型学会从有限信息中推理出更丰富的语义。这种思想在盲人视觉辅助、低分辨率图像理解等任务中也可能有用
- **MLLM 作为通用知识源**：不是直接用 MLLM 做任务，而是用它的表示来增强专用模型，这是当前大模型落地的一个重要方向

## 局限与展望

- 对齐效果依赖于 MLLM 教师的质量，如果 MLLM 在某些文档类型上表示较差，学生也无法受益
- 训练阶段需要额外运行 MLLM 提取教师表示，增加了训练成本（虽然可以预计算）
- 当前实验主要在中英、英德等高资源语言对上进行，对低资源语言的效果未知
- 仅验证了 encoder-decoder 架构的 DIMT 模型，未探索 decoder-only 架构是否也能受益
- 未来可以扩展到手写文档、扫描文档、多语言混合文档等更复杂场景

## 相关工作与启发

- **vs 传统 DIMT 方法**：传统方法仅用图像编码器提取视觉特征，缺乏文本语义理解。M4Doc 通过 MLLM 对齐补充了这一缺陷，尤其在跨域场景中优势明显
- **vs 直接使用 MLLM**：直接用 MLLM 做 DIMT 虽然效果好但推理成本过高，M4Doc 通过训练阶段的知识转移实现了"性能近似、成本持平"的折中
- **vs 传统知识蒸馏**：传统蒸馏对齐输出分布（soft labels），M4Doc 对齐中间表示空间，在 seq2seq 任务中更有效

## 评分

- 新颖性: ⭐⭐⭐⭐ single-to-mix 模态对齐是一个有启发性的新概念
- 实验充分度: ⭐⭐⭐⭐ 多语言对、跨域测试、详细消融，实验设计系统
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，动机推导合理
- 价值: ⭐⭐⭐⭐ 提出的范式可广泛迁移，对文档 AI 领域有实际价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Improving MLLM's Document Image Machine Translation via Synchronously Self-reviewing Its OCR Proficiency](improving_mllms_document_image_machine_translation_via_synchronously_self-review.md)
- [\[CVPR 2025\] Post-pre-training for Modality Alignment in Vision-Language Foundation Models](../../CVPR2025/multimodal_vlm/post-pre-training_for_modality_alignment_in_vision-language_foundation_models.md)
- [\[ACL 2025\] Burn After Reading: Do Multimodal Large Language Models Truly Capture Order of Events in Image Sequences?](burn_after_reading_do_multimodal_large_language_models_truly_capture_order_of_ev.md)
- [\[ACL 2025\] AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models](alignmmbench_evaluating_chinese_multimodal_alignment_in_large_vision-language_mo.md)
- [\[CVPR 2025\] CoLLM: A Large Language Model for Composed Image Retrieval](../../CVPR2025/multimodal_vlm/collm_a_large_language_model_for_composed_image_retrieval.md)

</div>

<!-- RELATED:END -->
