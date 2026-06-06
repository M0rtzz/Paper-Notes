---
title: >-
  [论文解读] Docling Technical Report
description: >-
  [ECCV 2024][PDF解析] Docling 是一个开源的 PDF 文档转换工具，集成了基于 DocLayNet 的布局分析模型和 TableFormer 表格结构识别模型，可在普通硬件上高效地将 PDF 转换为结构化的 JSON 或 Markdown 格式。
tags:
  - "ECCV 2024"
  - "PDF解析"
  - "文档布局分析"
  - "表格结构识别"
  - "开源工具"
  - "DocLayNet"
---

# Docling Technical Report

**会议**: ECCV 2024  
**arXiv**: [2408.09869](https://arxiv.org/abs/2408.09869)  
**代码**: [GitHub](https://github.com/DS4SD/docling)  
**领域**: 文档AI / PDF转换  
**关键词**: PDF解析, 文档布局分析, 表格结构识别, 开源工具, DocLayNet

## 一句话总结

Docling 是一个开源的 PDF 文档转换工具，集成了基于 DocLayNet 的布局分析模型和 TableFormer 表格结构识别模型，可在普通硬件上高效地将 PDF 转换为结构化的 JSON 或 Markdown 格式。

## 研究背景与动机

**领域现状**：PDF 文档因格式多样、标准化程度低、以打印优化为目标丢失了大部分结构信息，长期以来难以高效转换为机器可处理格式。随着 LLM 和 RAG 应用的兴起，高质量 PDF 内容提取的需求日益迫切。

**现有痛点**：(1) 市面上强大的文档理解方案多为商业软件或云服务；(2) 现有开源工具（如 pymupdf 受限于许可证、pypdfium 存在质量问题）在功能和质量上与商业方案差距显著；(3) 多模态视觉语言模型虽可处理文档，但速度慢、成本高，不适合大规模批处理。

**核心矛盾**：开源社区缺乏一个许可友好、功能完整、质量可靠的 PDF 文档转换工具。

**本文目标**：提供一个 MIT 许可的开源 PDF 转换库，集成 SOTA 的布局分析和表格结构识别能力，可完全本地运行。

**切入角度**：不追求端到端的多模态方法，而是组合专业化 AI 模型（布局检测 + 表格识别）与传统 PDF 解析，在效率和质量间取得最佳平衡。

**核心 idea**：用模块化流水线架构串联 PDF 解析、布局分析、表格识别和后处理，提供易于扩展的开源框架。

## 方法详解

### 整体框架

Docling 实现线性处理流水线：PDF 后端 → 页面渲染 + 文本提取 → 布局分析模型 → 表格结构识别模型 → 后处理组装 → 输出 JSON/Markdown。每个阶段可独立配置和替换。

### 关键设计

1. **自研 PDF 后端（docling-parse）**:

    - 功能：从 PDF 提取文本内容及其坐标，渲染页面图像
    - 核心思路：基于底层 qpdf 库自研解析器，解决了 pymupdf 的 AGPL 许可限制和 pypdfium/PyPDF 的质量问题（如跨列文本合并）。同时提供 pypdfium 作为备选后端
    - 设计动机：PDF 解析是整个流水线的基础，需要高质量的文本坐标提取和可靠的页面渲染

2. **基于 RT-DETR 的布局分析模型**:

    - 功能：检测页面中的段落、标题、列表、图表、表格等元素的边界框和类别
    - 核心思路：架构基于 RT-DETR 目标检测器，在 DocLayNet（大规模人工标注文档布局数据集）及私有数据集上重训练。输入 72dpi 页面图像，单 CPU 亚秒级推理。预测边界框经过去重和置信度筛选后与 PDF 文本 token 交叉匹配，分组为语义单元
    - 设计动机：文档布局理解是结构化转换的核心，RT-DETR 在速度和精度上提供了良好平衡

3. **TableFormer 表格结构识别**:

    - 功能：从表格图像恢复逻辑行列结构、单元格合并和表头层级
    - 核心思路：Vision Transformer 架构，使用自定义结构 token 语言描述表格结构。可处理无边框、空单元格、行列合并、不规则缩进等复杂情况。预测结构与 PDF 文本单元格匹配，避免重新 OCR
    - 设计动机：表格是文档中信息密度最高但结构最复杂的元素，TableFormer 在处理边界情况上显著优于传统方法

### 损失函数 / 训练策略

技术报告未详述训练细节，布局分析和 TableFormer 的训练方法分别在各自的独立论文中描述。推理依赖 ONNX Runtime（布局）和 PyTorch（表格）。

## 实验关键数据

### 主实验

| 配置 | TTS (225页) | 页/秒 | 峰值内存 |
|------|-----------|------|---------|
| M3 Max, 4线程, native | 177s | 1.27 | 6.20 GB |
| M3 Max, 16线程, native | 167s | 1.34 | 6.20 GB |
| Xeon E5, 4线程, native | 375s | 0.60 | 6.16 GB |
| M3 Max, 4线程, pypdfium | 103s | 2.18 | 2.56 GB |

### 消融实验

| 组件 | 耗时占比 | 说明 |
|------|---------|------|
| PDF 解析 | ~10% | 基础文本提取 |
| 布局分析 | ~30% | 亚秒/页 |
| 表格识别 | ~50% | 2-6秒/表格，主要瓶颈 |
| 后处理 | ~10% | 阅读顺序、元数据 |

### 关键发现

- 表格识别是主要性能瓶颈（每表 2-6 秒），复杂表格更慢
- pypdfium 后端速度快约 2x 但转换质量（尤其表格）较差
- 线程数从 4 增到 16 仅提升约 6%（M3 Max），说明主要受限于单线程模型推理
- OCR 模式（EasyOCR）大幅增加延迟（30+秒/页），目前仅适用于扫描件

## 亮点与洞察

- **工程完整性**：从 PDF 解析到结构化输出的完整流水线，自带两个 SOTA AI 模型，MIT 许可，pip 一行安装。对 RAG 应用的落地价值极高
- **模块化设计**：通过 BaseModelPipeline 基类和 Callable 接口设计，第三方可以轻松替换或添加模型。这种架构设计值得学习
- **务实的技术选择**：不追求端到端多模态，而是组合专业模型+传统解析，在商品硬件上实现合理的速度和质量平衡

## 局限与展望

- GPU 加速尚未完善，目前主要依赖 CPU 推理
- OCR 支持（EasyOCR）速度较慢且精度有限
- 缺少公式识别、代码块识别等专门模型
- 不支持 DOCX、HTML 等非 PDF 格式的输入
- 表格识别在极复杂表格上仍有改进空间
- 计划增加图形分类器、公式识别器、代码识别器等

## 相关工作与启发

- **vs 商业方案（Azure Document Intelligence等）**: 功能和质量仍有差距，但完全本地运行、MIT 许可、零成本是核心优势
- **vs 多模态 VLM**: Docling 速度快 10-100x，资源占用低，适合批量处理；VLM 适合需要深度理解的少量文档
- **vs pymupdf**: AGPL 许可限制商业使用；Docling MIT 许可无此问题

## 评分

- 新颖性: ⭐⭐⭐ 技术报告性质，模型/方法来自先前工作，贡献在于工程整合
- 实验充分度: ⭐⭐⭐ 给出了性能基准但缺乏与竞品的准确率对比
- 写作质量: ⭐⭐⭐⭐ 技术报告格式规范，信息完整
- 价值: ⭐⭐⭐⭐⭐ 填补了开源文档转换工具的空白，对 RAG 生态贡献巨大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Dropout Mixture Low-Rank Adaptation for Visual Parameters-Efficient Fine-Tuning](dropout_mixture_low-rank_adaptation_for_visual_parameters-efficient_fine-tuning.md)
- [\[ECCV 2024\] Interleaving One-Class and Weakly-Supervised Models with Adaptive Thresholding for Unsupervised Video Anomaly Detection](interleaving_one-class_and_weakly-supervised_models_with_adaptive_thresholding_f.md)
- [\[ECCV 2024\] Teaching Tailored to Talent: Adverse Weather Restoration via Prompt Pool and Depth-Anything Constraint](teaching_tailored_to_talent_adverse_weather_restoration_via_prompt_pool_and_dept.md)
- [\[ECCV 2024\] Learning Anomalies with Normality Prior for Unsupervised Video Anomaly Detection](learning_anomalies_with_normality_prior_for_unsupervised_video_anomaly_detection.md)
- [\[ECCV 2024\] MemBN: Robust Test-Time Adaptation via Batch Norm with Statistics Memory](membn_robust_test-time_adaptation_via_batch_norm_with_statistics_memory.md)

</div>

<!-- RELATED:END -->
