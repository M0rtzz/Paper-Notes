---
title: >-
  [论文解读] READoc: A Unified Benchmark for Realistic Document Structured Extraction
description: >-
  [ACL 2025][文档结构化提取] READoc 提出了首个将文档结构化提取（DSE）定义为端到端 PDF 到 Markdown 转换的统一基准，包含 3,576 篇来自 arXiv/GitHub/Zenodo 的真实文档和三模块评估套件（标准化+分割+评分），首次揭示了当前 DSE 系统与真实场景需求之间的差距。
tags:
  - ACL 2025
  - 文档结构化提取
  - 统一基准
  - PDF转Markdown
  - 评估套件
  - 视觉语言模型
---

# READoc: A Unified Benchmark for Realistic Document Structured Extraction

**会议**: ACL 2025  
**arXiv**: [2409.05137](https://arxiv.org/abs/2409.05137)  
**代码**: [有](https://github.com/icip-cas/READoc)  
**领域**: 文档 AI / 多模态  
**关键词**: 文档结构化提取, 统一基准, PDF转Markdown, 评估套件, 视觉语言模型

## 一句话总结

READoc 提出了首个将文档结构化提取（DSE）定义为端到端 PDF 到 Markdown 转换的统一基准，包含 3,576 篇来自 arXiv/GitHub/Zenodo 的真实文档和三模块评估套件（标准化+分割+评分），首次揭示了当前 DSE 系统与真实场景需求之间的差距。

## 研究背景与动机

### 问题背景

文档结构化提取（DSE）是将原始文档转换为机器可读结构化文本的任务，对构建知识库、高质量语料库和 RAG 系统至关重要。然而，现有评估存在两个致命问题：

**碎片化**：现有基准将 DSE 拆分为孤立的子任务——文档版面分析（PubLayNet）、OCR（Robust Reading）、表格识别（PubTabNet）、公式转换（Im2Latex-100K）、目录提取（HRDoc）、阅读顺序检测（ReadingBank）——数据来源不同、格式不一致，无法统一评估

**局部化**：现有研究聚焦于单页内的局部区域（版面块或表格），忽略了真实文档的多页性、层级标题和长距离依赖

### 核心动机

需要一个统一基准来量化现有工作与真实 DSE 目标之间的差距，促进更全面、更实用的解决方案。

## 方法详解

### 整体框架

READoc 包含两大部分：

1. **READoc 数据集**：3,576 个 PDF-Markdown 配对，来自三个异构来源
2. **评估 S³uite**：标准化（Standardization）→ 分割（Segmentation）→ 评分（Scoring）

### 关键设计

1. **任务定义——PDF到Markdown范式**：

    - **输入**：完整的原始 PDF 文档（最具通用性且本质无结构化）
    - **输出**：Markdown 文本（轻量级标记语言，支持标题、列表等结构元素，并通过 LaTeX 语法支持表格和公式）
    - 输出可被直接分块、索引或被 LLM 消费

2. **数据集构建**：

   | 子集 | 来源 | 文档数 | 平均页数 | 特点 |
   |------|------|-------|---------|------|
   | READoc-arXiv | arXiv 预印本 | 1,009 | 11.67 | 复杂学术结构（公式/表格），多栏布局 |
   | READoc-GitHub | GitHub README | 1,224 | 6.54 | 仅基本元素，标题风格多样化 |
   | READoc-Zenodo | Zenodo 开放仓库 | 1,343 | 14.93 | 文档更长（>30页），27种语言 |

    - **arXiv**：LaTeX → HTML（LaTeXML）→ Markdown（改进的 Nougat 流程）
    - **GitHub**：原始 Markdown → PDF（Pandoc + Eisvogel 模板）
    - **Zenodo**：DOCX/HTML → Markdown（Markitdown/Pandoc）→ PDF

3. **评估 S³uite 三模块**：

   **标准化模块**：统一不同系统输出的格式差异
    - 对齐公式边界（`$$`、`\begin{equation}`、`\[`）
    - 统一 Markdown 标题风格
    - 对齐 Markdown 表格与 LaTeX 表格格式
    - 移除图片和链接语法

   **分割模块**：将 Markdown 文本分为四种语义单元
    - 标题（不同层级）
    - 公式（行内和独立）
    - 表格
    - 纯文本（基本文本+粗体/斜体/列表等简单格式）

   **评分模块**：包含两个子模块
    - **语义单元评估**：文本提取（拼接 EDS + 词汇 F1）、标题检测（拼接 EDS + 树编辑距离 TEDS）、公式转换（行内/独立 EDS）、表格识别（拼接 EDS + 最优二部匹配 TEDS）
    - **阅读顺序检测**：基于语义单元边界分块，计算 Kendall's Tau 距离相似度（块级和 token 级）

## 实验关键数据

### READoc-arXiv 主实验（Table 3，Average 指标）

| 系统类型 | 方法 | Text(C) | Head(T) | Formula(E) | Formula(I) | Table(T) | RO(B) | **Avg** |
|---------|------|---------|---------|-----------|-----------|---------|-------|---------|
| 基线 | PyMuPDF4LLM | 74.27 | 20.77 | 0.07 | 0.02 | 15.83 | 89.09 | 40.55 |
| 流水线 | **MinerU** | 91.22 | 41.97 | 62.77 | 70.76 | 52.85 | 97.90 | **73.07** |
| 流水线 | Marker | 82.71 | 39.39 | 3.47 | 48.74 | 72.36 | 97.74 | 64.98 |
| 专家模型 | Nougat-base | 92.29 | 88.50 | 76.19 | 79.47 | 52.30 | 98.41 | **81.42** |
| VLM | InternVL-Chat-V1.5 | 68.44 | 13.57 | 33.13 | 24.37 | 34.35 | 91.31 | 47.83 |
| VLM | GPT-4o-mini | 84.37 | 18.65 | 42.23 | 41.67 | 39.85 | 96.35 | 57.98 |

### 跨子集泛化性（Nougat-base Average）

| 子集 | Average |
|------|---------|
| READoc-arXiv | **81.42** |
| READoc-GitHub | 74.12 |
| READoc-Zenodo | 49.94 |

### 消融/精细分析

| 分析维度 | 发现 |
|---------|------|
| 单栏 vs 多栏 | 所有系统在多栏文档上均有性能下降；InternVL 下降最大(-7.13) |
| 文档长度影响 | VLM 随长度增加性能下降；流水线工具和专家模型保持稳定 |
| 文档深度影响 | 流水线工具和专家模型随深度增加性能下降；VLM 保持稳定 |
| 单页 vs 多页范式 | 多页范式显著增强全局 ToC 构建（+28 Tree），但损害公式/表格局部能力 |
| 效率 | Marker 最快(24s/doc)，InternVL 最慢(1182s/doc) |

### 关键发现

1. **流水线工具面临复杂工程挑战**：Docling 不识别公式，Marker 不识别行内公式，Pix2Text 会崩溃
2. **专家模型泛化能力差**：Nougat 在 arXiv 上 81.42 但在 GitHub 上降至 74.12，在 Zenodo（多语言）上暴跌至 49.94
3. **VLM 在复杂学术文档上普遍落后**：最好的开源 VLM（InternVL）仅 47.83，远低于流水线工具
4. **全局 ToC 构建是重大挑战**：所有单页系统在层级标题树上表现远低于拼接评测
5. **阅读顺序检测相对容易**：基线工具 Tesseract 即可达 96.70 KTDS
6. **不同系统呈现截然不同的短板**：VLM 在深度上稳定但随长度下降，流水线工具反之

## 亮点与洞察

- **统一范式的价值**：PDF-to-Markdown 范式首次将 6 个子任务统一到一个端到端框架中，揭示了子任务之间的交互和系统级瓶颈
- **三来源数据设计精妙**：arXiv（复杂结构）、GitHub（简单布局复杂标题）、Zenodo（多语言长文档）互补覆盖不同挑战
- **S³uite 设计周到**：标准化模块消除格式差异、分割模块量化多维能力、评分模块兼顾局部+全局
- **首次揭示多页挑战**：单页图像到标记语言的范式存在本质局限，多页处理是未来方向
- **对 RAG 系统直接有用**：Markdown 输出可直接用于 LLM 消费、分块和索引

## 局限与展望

- 自动构建的 PDF-Markdown 配对存在噪声，难以完全消除
- 标准化模块无法覆盖所有格式差异场景
- 未包含手写文档、扫描文档等更复杂类型
- 未来：引入更多格式统一模块、探索更高效的多页建模范式
- 未来：构建包含 OCR 质量更差的现实文档的子集

## 相关工作与启发

- 继承并统一了 PubLayNet、PubTabNet、Im2Latex-100K 等经典 DSE 子任务基准
- 与 OmniDocBench（Ouyang et al., 2024）相比，READoc 处理多页文档和端到端流程
- Nougat（Blecher et al., 2023）定义了单页图像到 Markdown 范式，READoc 将其扩展到完整多页文档
- 对 MinerU、Marker 等当前热门工具提供了客观对比评估，对文档处理工具选型有参考价值

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个统一、端到端、多页的 DSE 基准，填补了重要空白
- **实验充分度**: ⭐⭐⭐⭐⭐ 14 种系统、3 个子集、多维能力评测、精细分析（布局/长度/深度/效率），极其全面
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，图表丰富，对比分析透彻
- **价值**: ⭐⭐⭐⭐⭐ 对文档 AI 社区和 RAG 工程实践有重大推动价值，填补了评估真空

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] StructFlowBench: A Structured Flow Benchmark for Multi-turn Instruction Following](structflowbench_a_structured_flow_benchmark_for_multi-turn_instruction_following.md)
- [\[ACL 2025\] TUMLU: A Unified and Native Language Understanding Benchmark for Turkic Languages](tumlu_a_unified_and_native_language_understanding_benchmark_for_turkic_languages.md)
- [\[ACL 2025\] KITAB-Bench: A Comprehensive Multi-Domain Benchmark for Arabic OCR and Document Understanding](kitab-bench_a_comprehensive_multi-domain_benchmark_for_arabic_ocr_and_document_u.md)
- [\[ACL 2025\] MDBench: A Synthetic Multi-Document Reasoning Benchmark Generated with Knowledge Guidance](mdbench_a_synthetic_multi-document_reasoning_benchmark_generated_with_knowledge_.md)
- [\[ACL 2025\] RealHiTBench: A Comprehensive Realistic Hierarchical Table Benchmark for Evaluating LLM-Based Table Analysis](realhitbench_a_comprehensive_realistic_hierarchical_table_benchmark_for_evaluati.md)

<!-- RELATED:END -->
