---
title: >-
  [论文解读] MarkushGrapher-2: End-to-end Multimodal Recognition of Chemical Structures
description: >-
  [CVPR 2026][多模态][化学结构识别] MarkushGrapher-2 提出了一个端到端多模态化学结构识别模型，通过专用化学 OCR 模块联合编码图像、文本和布局信息，结合两阶段训练策略（先适配 OCSR 特征再融合多模态编码），在 Markush 结构识别上大幅超越现有方法（M2S 准确率 56% vs 38%），同时保持分子结构识别的竞争力。
tags:
  - CVPR 2026
  - 多模态
  - 化学结构识别
  - Markush结构
  - 多模态编码
  - 专利文档分析
  - OCR
---

# MarkushGrapher-2: End-to-end Multimodal Recognition of Chemical Structures

**会议**: CVPR 2026  
**arXiv**: [2603.28550](https://arxiv.org/abs/2603.28550)  
**代码**: https://github.com/DS4SD/MarkushGrapher  
**领域**: 多模态VLM / 文档理解  
**关键词**: 化学结构识别、Markush结构、多模态编码、专利文档分析、OCR

## 一句话总结

MarkushGrapher-2 提出了一个端到端多模态化学结构识别模型，通过专用化学 OCR 模块联合编码图像、文本和布局信息，结合两阶段训练策略（先适配 OCSR 特征再融合多模态编码），在 Markush 结构识别上大幅超越现有方法（M2S 准确率 56% vs 38%），同时保持分子结构识别的竞争力。

## 研究背景与动机

1. **领域现状**：从文档中自动提取化学结构是大规模化学文献分析的基础。现有方法分别处理图像中的分子结构（OCSR）或文本中的化学命名实体，但对于结合图像和文本的多模态描述——Markush 结构——仍然处理不力。

2. **现有痛点**：Markush 结构在专利分析中至关重要（用于先行技术搜索、自由运营评估等），但当前仅被 MARPAT 和 DWPIM 两个人工标注的专有数据库收录。前代 MarkushGrapher-1 需要预标注 OCR 输出作为输入（无法端到端处理），且视觉识别精度有提升空间。通用 VLM（GPT-5、DeepSeek-OCR）在 Markush 识别上表现很差（M2S 上 GPT-5 仅 3%）。

3. **核心矛盾**：Markush 结构的图像风格在不同专利局和出版年份间差异巨大，文本描述缺乏标准化且包含条件/递归描述，同时缺乏大规模真实世界训练数据。

4. **本文目标** 构建统一的端到端模型，同时识别标准分子和多模态 Markush 结构。

5. **切入角度**：利用双编码器（OCSR 视觉编码器 + VTL 多模态编码器）互补融合，配合专用化学 OCR 模块和两阶段训练策略。

6. **核心 idea**：双编码管线融合视觉结构特征和多模态文本-布局特征，端到端识别化学 Markush 结构。

## 方法详解

### 整体框架

输入一张化学结构图像，输出 CXSMILES 表示（Markush 骨架的图形描述）和取代基表（可能替代可变基团的分子片段）。整体为编码器-解码器架构：

1. **管线 1**：图像 → 视觉编码器（MolScribe 的 Swin-B ViT，冻结）→ MLP 投影器 → 视觉嵌入 e1
2. **管线 2**：图像 → ChemicalOCR → 文本+边界框 → VTL 编码器（T5-base）→ 多模态嵌入 e2
3. **融合**：e1 与 e2 拼接 → 文本解码器 → 自回归生成 CXSMILES + 取代基表

### 关键设计

1. **ChemicalOCR 模块**:

    - 功能：从化学结构图像中提取字符级文本和边界框，实现端到端处理
    - 核心思路：基于 Smoldocling（256M 参数轻量 VLM）微调。先在 235k 合成化学结构上预训练（自动 OCR 标注），再在 7k 手工标注的 IP5 专利文档化学结构上微调。提取的文本和边界框为 VTL 编码器提供文本和布局模态信息。
    - 设计动机：现有 OCR 模型（PaddleOCR、EasyOCR）在化学图像上表现极差（F1仅7.7/10.2 vs ChemicalOCR的87.2），常见问题包括将化学键误识为减号/等号、无法处理化学缩写。化学 OCR 是准确识别 Markush 特征（括号、指标等）的关键。

2. **双编码器融合（OCSR + VTL）**:

    - 功能：互补捕获视觉结构特征和多模态文本-布局特征
    - 核心思路：OCSR 视觉编码器（Swin-B ViT 来自 MolScribe）擅长分子骨架识别但无法处理 Markush 特征；VTL 编码器（T5-base，UDOP 融合范式）将空间重合的视觉和文本 token 对齐融合，擅长 Markush 特征但弱于分子结构。两个投影后拼接送入文本解码器。消融实验验证：Pipeline 1 单独在 USPTO SMILES 上达 89.1%但 M2S 仅 8%，Pipeline 2 在 M2S 达 39%但 USPTO 仅 46%，融合模型兼顾两者。
    - 设计动机：Markush 结构同时包含视觉信息（分子骨架）和文本信息（可变基团定义），需要互补编码才能完整识别。

3. **两阶段训练策略**:

    - 功能：在不破坏预训练 OCSR 特征的前提下有效融合两个编码器
    - 核心思路：Phase 1（适配）：冻结视觉编码器，训练投影器和文本解码器做标准 SMILES 预测（243k 真实样本，3 epochs），让解码器适配 OCSR 特征空间。Phase 2（融合）：冻结视觉编码器和投影器，引入 OCR 和 VTL 编码器，端到端训练 VTL 编码器和文本解码器做 CXSMILES + 取代基表预测（235k 合成 + 145k 真实，2 epochs）。
    - 设计动机：直接单阶段训练（Fusion only）的 M2S 准确率 44%，两阶段训练提升到 50%（+6%）。冻结 OCSR 编码器保护原始视觉特征，让 VTL 编码器专注于学习 Markush 特征的缺失信息。

### 损失函数 / 训练策略

模型整体使用标准的自回归交叉熵损失。Phase 1 训练 SMILES 预测，Phase 2 训练 CXSMILES + 取代基表预测。总模型 831M 参数，其中 744M 可训练。训练在 NVIDIA A100 GPU 上进行。

## 实验关键数据

### 主实验

| 方法 | M2S (CXSMILES A) | USPTO-M A | WildMol-M A | IP5-M A |
|------|-------------------|-----------|-------------|---------|
| MolParser-Base (图像) | 39 | 30 | 38.1 | 47.7 |
| MolScribe (图像) | 21 | 7 | 28.1 | 22.3 |
| GPT-5 (多模态) | 3 | — | — | — |
| DeepSeek-OCR | 0 | 0 | 1.9 | 0.0 |
| MarkushGrapher-1 | 38 | 32 | — | — |
| **MarkushGrapher-2** | **56** | **55** | **48.0** | **53.7** |

### 消融实验

| 配置 | M2S A | M2S A_InChIKey | USPTO-M A | IP5-M A |
|------|-------|----------------|-----------|---------|
| 无 OCR 输入 | 4 | 39 | 3 | 15.4 |
| 有 OCR 输入 | 56 | 80 | 55 | 53.7 |
| 单阶段训练 (Fusion only) | 44 | 53 | — | — |
| 两阶段训练 (Adapt + Fusion) | 50 | 68 | — | — |

### 关键发现

- OCR 模块是最关键组件：没有 OCR 的话 M2S 准确率从 56% 暴跌到 4%，因为括号和索引等文本信息对 Markush 特征预测至关重要
- ChemicalOCR 大幅超越通用 OCR：在 IP5-M 上 F1=86.5 vs PaddleOCR 的 1.9 和 EasyOCR 的 18.4
- 通用 VLM 在 Markush 识别上完全失败：GPT-5 仅 3%，DeepSeek-OCR 为 0%
- 两阶段训练比单阶段在 M2S 骨架准确率上提升 15%（53%→68%）
- 在标准分子识别（OCSR）上也保持竞争力：UOB 上达 96.6%（最佳），WildMol 上 68.4%

## 亮点与洞察

- **双编码互补设计**：分别利用视觉编码器的分子骨架识别能力和 VTL 编码器的多模态融合能力，是一种通用的多模态架构设计模式。可迁移到其他需要同时处理结构化视觉和文本信息的任务（如表格理解、电路图分析）。
- **USPTO-MOL-M 数据生成管线**：从 USPTO MOL 文件自动提取真实 Markush 训练数据，解决了标注数据稀缺问题。这种利用已有结构化数据自动生成训练样本的思路值得借鉴。
- **领域特化 OCR**：通用 OCR 在化学图像上完全不可用，但仅需 7k 手工标注+235k 合成数据即可训练出高精度的领域 OCR。说明领域适配在 OCR 中依然很重要。

## 局限与展望

- **整体准确率仍不高**：M2S 上 56%、IP5-M 上 53.7%，离实用化还有距离，特别是取代基表预测（M2S 表准确率仅 22%）
- **训练数据仍以合成为主**：235k 合成 + 145k 真实，合成数据的分布可能与真实专利文档有差距
- **OCR 错误级联**：OCR 模块的错误会直接影响下游 Markush 识别，级联误差可能在复杂结构中放大
- **仅支持 2D 结构**：未处理 3D 分子构象信息
- **推理效率未讨论**：831M 参数模型的推理速度是否满足大规模专利扫描的需求

## 相关工作与启发

- **vs MarkushGrapher-1**：前代需要预标注 OCR 输入，本文实现端到端处理；准确率从 38% 提升到 56%
- **vs MolParser**：MolParser 仅处理图像模态且只支持有限的 Markush 特征，本文联合处理图文且更全面
- **vs GPT-5/DeepSeek-OCR**：通用 VLM 在此任务上完全失败，说明化学结构识别仍需领域特化方法

## 评分

- 新颖性: ⭐⭐⭐⭐ 双编码融合+两阶段训练+专用OCR的组合设计新颖，但各组件技术本身不算全新
- 实验充分度: ⭐⭐⭐⭐⭐ 多个基准、多种基线对比、详尽消融实验，还发布了新基准IP5-M
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，对化学背景解释充分，但部分内容较冗长
- 价值: ⭐⭐⭐⭐ 填补了端到端Markush识别的空白，对化学信息学和专利分析有重要实用价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] MarkushGrapher: Joint Visual and Textual Recognition of Markush Structures](../../CVPR2025/multimodal_vlm/markushgrapher_joint_visual_and_textual_recognition_of_markush_structures.md)
- [\[ICCV 2025\] MolParser: End-to-end Visual Recognition of Molecule Structures in the Wild](../../ICCV2025/multimodal_vlm/molparser_end-to-end_visual_recognition_of_molecule_structures_in_the_wild.md)
- [\[ICLR 2026\] WebDS: An End-to-End Benchmark for Web-based Data Science](../../ICLR2026/multimodal_vlm/webds_an_end-to-end_benchmark_for_web-based_data_science.md)
- [\[AAAI 2026\] SpeakerLM: End-to-End Versatile Speaker Diarization and Recognition with Multimodal Large Language Models](../../AAAI2026/multimodal_vlm/speakerlm_end-to-end_versatile_speaker_diarization_and_recognition_with_multimod.md)
- [\[CVPR 2026\] Multimodal OCR: Parse Anything from Documents](multimodal_ocr_parse_anything_from_documents.md)

<!-- RELATED:END -->
