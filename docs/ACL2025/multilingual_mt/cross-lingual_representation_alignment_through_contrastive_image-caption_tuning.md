---
title: >-
  [论文解读] Cross-Lingual Representation Alignment Through Contrastive Image-Caption Tuning
description: >-
  [ACL 2025][多语言/翻译][跨语言对齐] 探索了一种无需平行语料的跨语言表示对齐方法——通过多语言图像-文本描述的对比学习（类 CLIP），让不同语言的文本表示在共享视觉空间中隐式对齐，并证明即使是编码器预训练中未见过的语言（如 Quechua）也能通过这种方式被纳入对齐体系。 编码器语言模型仍是提取文本语义信息的…
tags:
  - "ACL 2025"
  - "多语言/翻译"
  - "跨语言对齐"
  - "对比学习"
  - "图像-文本"
  - "低资源语言"
  - "多语言表示"
---

# Cross-Lingual Representation Alignment Through Contrastive Image-Caption Tuning

**会议**: ACL 2025  
**arXiv**: [2505.13628](https://arxiv.org/abs/2505.13628)  
**代码**: 有 (github.com/nkrasner/cl-clip-align)  
**领域**: 多语言翻译  
**关键词**: 跨语言对齐, 对比学习, 图像-文本, 低资源语言, 多语言表示

## 一句话总结

探索了一种无需平行语料的跨语言表示对齐方法——通过多语言图像-文本描述的对比学习（类 CLIP），让不同语言的文本表示在共享视觉空间中隐式对齐，并证明即使是编码器预训练中未见过的语言（如 Quechua）也能通过这种方式被纳入对齐体系。

## 研究背景与动机

编码器语言模型仍是提取文本语义信息的主流方法，但多语言编码器（如 XLM-R）的内部表示往往是"离散"的——语言 A 的句子表示与其翻译在语言 B 中的表示可能毫不相似。这种不对齐主要源于预训练数据的语言间不平衡和领域失配。

传统的跨语言对齐方法几乎都依赖平行语料（bitext），而平行语料的获取对低资源语言既昂贵又耗时。相比之下，**描述一张图片远比翻译一段文字容易**——任何语言的说话者都能为图像写标题。语言记录工作也经常产生配有单语音频或文字的多媒体材料。

因此核心假设是：**图像可以作为跨语言的"桥梁"模态——如果多语言文本都与相同图像对齐，文本之间也会隐式对齐**。

## 方法详解

### 整体框架

将文本编码器（XLM-Roberta-Large）与视觉编码器（ViT-Base）通过对比学习连接。给定图像-文本描述对，用标准的 CLIP 风格对比损失训练：

$$S = E_c \cdot E_i^\top \ast t$$
$$L(E_i, E_c) = \text{CrossEntropy}(S, I)$$

其中 $E_c$ 和 $E_i$ 分别是文本和图像的表示，$t$ 是可学习的温度参数，$I$ 是单位矩阵。

### 关键设计

1. **四种数据配置的对比实验**:

    - **Eng-only**: 纯英语 MS-COCO 图像-标题对（118k）
    - **Eng-Pivot**: 英语标题 + 轮换的西/日/印语翻译标题（文本-文本对齐，类似 LaBSE）
    - **Multilingual**: 每张图像轮换配英/西/日/印一种语言的标题（文本-图像对齐）
    - **Multilingual+Quechua**: 在 Multilingual 基础上加入 Quechua（编码器预训练中未见的语言）

2. **编码器架构处理**: 由于 XLM-R 和 ViT 的隐藏维度不同，在各自输出后加线性层映射到 512 维匹配空间。前半个 epoch 仅训练线性层（warm up），之后解冻编码器进行全参数微调。

3. **对预训练中未见语言的处理**: XLM-R 的预训练不包含 Quechua 等拉丁美洲原住民语言。通过将 Quechua 图像标题加入训练数据的轮换中，探索能否将未见语言整合进对齐表示空间。

### 损失函数 / 训练策略

- 标准对比学习损失（cross-entropy + 温度缩放）
- 使用 MS-COCO 数据集（118k English Image-Caption pairs）
- 通过 Google Translate 生成西/日/印/Quechua 翻译
- 线性层 warm-up → 全参数微调

## 实验关键数据

### 实验1: 双文检索准确率（Flores-200 数据集）

| 编码器 | 全部(203语言) | XLM-R 训练语言(92) | 未见语言(111) | Quechua |
|--------|-------------|-------------------|-------------|---------|
| XLM-R (原始) | 0.5 | 0.6 | 0.4 | 0.5 |
| Eng-Only | 18.3 | 27.5 | 10.7 | 7.2 |
| Eng-Pivot (文本-文本) | **62.2** | **92.6** | **37.1** | 13.1 |
| Multilingual (文本-图像) | 55.7 | 82.2 | 33.7 | 18.0 |
| Multilingual+Quechua | 50.4 | 76.6 | 28.6 | **29.2** |

### 实验3: XNLI 跨语言NLI准确率（选取部分语言）

| 编码器 | en | es | hi | de | zh | ar | 平均(12语言) |
|--------|----|----|----|----|----|----|-------------|
| XLM-R | 50 | 44 | 44 | 43 | 44 | 45 | 43.8 |
| Eng-Only | 53 | 50 | 46 | 49 | 48 | 47 | 48.0 |
| Eng-Pivot | **67** | **65** | **60** | **64** | **62** | **61** | **61.8** |
| Multilingual | 55 | 52 | 51 | 52 | 51 | 51 | 51.3 |
| +Quechua | 56 | 53 | 51 | 53 | 51 | 51 | 51.6 |

### 关键发现

1. **文本-图像对齐可以导致文本-文本对齐**: Multilingual 模型虽然仅训练文本与图像的对齐，但在双文检索任务上达到 55.7% 的准确率，远超原始 XLM-R 的 0.5%，虽不及直接的文本-文本对齐（62.2%），但已相当可观。

2. **未见语言可以被纳入**: 加入 Quechua 标题后，Quechua 的双文检索准确率从 18.0% 大幅提升至 29.2%，且其他语言的性能基本不受影响（整体下降主要因为其他语言数据量减少以适配数据规模可比性）。

3. **下游任务质量不降反升**: 在 XNLI 跨语言 NLI 任务上，文本-图像对齐后的编码器表现优于原始 XLM-R（51.3 vs 43.8），说明图像对齐不会覆写对下游任务有用的文本特征，反而有所改善。

4. **加入 Quechua 甚至提升其他语言的 NLI**: Multilingual+Quechua 的英语 NLI 分数（56）高于 Multilingual（55），在几乎所有语言上都持平或更好。更广泛的语言覆盖似乎对 NLI 任务有益。

5. **英语单语训练的有限效果**: Eng-Only 虽将检索准确率从 0.5% 提升到 18.3%，但远不如多语言训练，说明跨语言对齐需要多语言信号。

## 亮点与洞察

- **极简但有效**: 整个方法仅需图像-标题对（单语标题即可），不需要任何平行语料。对低资源语言来说，标注图像标题远比获取翻译语料容易。
- **图像作为语义桥梁**: 这一假设在实验中得到验证——不同语言对相同图像的描述使文本表示在共享视觉空间中自然对齐。虽不及直接的文本-文本对齐，但作为低资源语言的 bootstrapping 手段非常实用。
- **t-SNE 可视化直观**: 从 XLM-R 原始的明显语言聚类，到多语言图像对齐后翻译句对靠近，视觉化结果清晰展示了对齐效果。

## 局限与展望

- 多语言图像-文本对齐虽然可用，但与平行语料方法仍有差距（55.7 vs 62.2 双文检索准确率）
- 翻译由 Google Translate 生成，引入了翻译质量的偏差（尤其是 Quechua 的翻译质量可能较差）
- 加入 Quechua 后整体性能小幅下降，未能明确区分是数据量减少还是新语言引入导致的影响
- 在更多模型架构（如 mBERT、更大的 XLM-R）上的泛化性未验证
- 仅测试了 NLI 一个下游任务，更广泛的任务覆盖（如 NER、情感分析）可以更好验证通用性
- 可考虑用真人标注的非英语标题替代机器翻译

## 相关工作与启发

- 与 LaBSE（文本-文本对比对齐）的直接对比展示了图像作为桥梁的可行性和差距
- Muraoka et al. (2023) 在 NLU 任务输入中引入图像表示来改善跨语言迁移，为本文假设提供了佐证
- 对于语言记录领域——很多濒危语言有配图的文本材料但无平行翻译——这种方法可作为构建 NLP 工具的第一步
- 可结合更强的视觉编码器（如 SigLIP、EVA-CLIP）或更大的图像-标题数据集来缩小与平行语料方法的差距

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 新颖性 | 3.5 |
| 实验充分度 | 3.5 |
| 写作质量 | 4 |
| 价值 | 3.5 |

研究思路清晰、假设新颖（图像作为跨语言桥梁），但方法本身（CLIP 风格对比学习）较为标准。实验设置合理但规模有限（仅 118k 数据、4-5 种语言、1 个下游任务）。作为短论文（ACL Findings），质量不错。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs](mid_layer_crosslingual_alignment.md)
- [\[ACL 2025\] Statement-Tuning Enables Efficient Cross-lingual Generalization in Encoder-only Models](statement-tuning_enables_efficient_cross-lingual_generalization_in_encoder-only_.md)
- [\[ACL 2025\] Modular Sentence Encoders: Separating Language Specialization from Cross-Lingual Alignment](modular_sentence_encoders.md)
- [\[ACL 2025\] CC-Tuning: A Cross-Lingual Connection Mechanism for Improving Joint Multilingual Supervised Fine-Tuning](cc-tuning_a_cross-lingual_connection_mechanism_for_improving_joint_multilingual_.md)
- [\[ACL 2025\] Cross-Lingual Transfer of Cultural Knowledge: An Asymmetric Phenomenon](cross-lingual_transfer_of_cultural_knowledge_an_asymmetric_phenomenon.md)

</div>

<!-- RELATED:END -->
