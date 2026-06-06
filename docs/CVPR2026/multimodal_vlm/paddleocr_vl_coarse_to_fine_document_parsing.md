---
title: >-
  [论文解读] PaddleOCR-VL: Boosting Document Parsing Efficiency and Performance with Coarse-to-Fine Visual Processing
description: >-
  [CVPR 2026][多模态VLM][文档解析] PaddleOCR-VL 提出粗到细的文档解析架构：粗阶段用轻量级有效区域聚焦模块(VRFM)定位文档中的有效视觉区域并预测阅读顺序，细阶段用紧凑的0.9B视觉语言模型对裁剪区域进行精细识别，在最少视觉token和参数下实现文档解析SOTA。
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "文档解析"
  - "视觉语言模型"
  - "粗到细处理"
  - "视觉冗余消除"
  - "OCR"
---

# PaddleOCR-VL: Boosting Document Parsing Efficiency and Performance with Coarse-to-Fine Visual Processing

**会议**: CVPR 2026  
**arXiv**: [2603.24326](https://arxiv.org/abs/2603.24326)  
**代码**: [https://github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)  
**领域**: 多模态VLM  
**关键词**: 文档解析, 视觉语言模型, 粗到细处理, 视觉冗余消除, OCR

## 一句话总结

PaddleOCR-VL 提出粗到细的文档解析架构：粗阶段用轻量级有效区域聚焦模块(VRFM)定位文档中的有效视觉区域并预测阅读顺序，细阶段用紧凑的0.9B视觉语言模型对裁剪区域进行精细识别，在最少视觉token和参数下实现文档解析SOTA。

## 研究背景与动机

1. **领域现状**：文档解析方法分为三类——流水线方法（拼接专家组件）、通用VLM（端到端但重）、专用VLM（统一架构但效率低）。高分辨率输入对文档解析至关重要，但导致视觉token数量二次增长。
2. **现有痛点**：通用VLM在手写或复杂文档上频繁产生幻觉和识别错误；专用VLM（如MinerU2-VLM）参数量大、解码序列长导致延迟高；统一压缩视觉token的方法（如DeepSeek-OCR）会损害细粒度布局精度。
3. **核心矛盾**：文档图像中有效视觉区域高度不均匀——PPT文档有效区域仅占39%，信息密集型文档约60%。大量背景/装饰区域浪费了计算资源。
4. **本文目标**：在保持高分辨率精度的同时消除视觉冗余，实现高精度+高效率。
5. **切入角度**：观察到有效视觉区域的稀疏性，用检测器定位有效区域后仅对这些区域做精细识别。
6. **核心 idea**：解耦布局分析与元素识别——轻量检测器做粗粒度定位+阅读顺序预测，紧凑VLM对裁剪区域做细粒度识别，避免处理整张大图。

## 方法详解

### 整体框架

PaddleOCR-VL 分两个阶段：粗阶段(VRFM)接收完整文档图像，输出有效区域的位置、类别和阅读顺序；细阶段(PaddleOCR-VL-0.9B)接收裁剪的有效区域，输出精细识别结果（文本、公式、表格等）。最终按阅读顺序重组为结构化文档。

### 关键设计

1. **有效区域聚焦模块 (VRFM)**:
    - 功能：高效定位文档中的有效视觉元素并预测阅读顺序
    - 核心思路：基于 RT-DETR 检测器进行布局元素检测和分类，生成区域级表示。在此基础上扩展指针网络(Pointer Network)建模检测区域间的成对关系，预测 $N \times N$ 矩阵编码相对阅读顺序。整体轻量级，联合完成区域定位、类别预测和阅读顺序估计。
    - 设计动机：用任务特定检测器做布局分析比生成式VLM更高效且坐标更准确，指针网络适合序列排序任务

2. **PaddleOCR-VL-0.9B**:
    - 功能：对裁剪的有效区域进行精细元素识别
    - 核心思路：设计紧凑的0.9B参数视觉语言模型。仅处理VRFM裁剪出的有效区域（而非整页），大幅减少视觉token数量。支持文本、公式、表格、图表等多种元素的识别，覆盖109种语言。
    - 设计动机：解耦后的识别模块只需处理小区域图像，可用更小模型达到更好效果

3. **高质量数据流水线**:
    - 功能：构建大规模、多样化的训练数据
    - 核心思路：从公开来源和合成数据收集超过3000万个广泛分布的样本。数据多样性覆盖各类文档类型、语言和复杂度，成为模型高性能的关键因素之一。
    - 设计动机：数据质量和多样性对VLM性能的影响不亚于模型架构

### 损失函数 / 训练策略

- VRFM：标准目标检测损失 + 指针网络排序损失
- PaddleOCR-VL-0.9B：自回归生成损失
- 两个模块独立优化，各自专注各自子任务
- 超3000万样本的大规模训练数据

## 实验关键数据

### 主实验

| 方法 | 参数量 | 视觉Token数 | OmniDocBench v1.5 整体 |
|------|--------|-------------|----------------------|
| MinerU2-VLM | 大 | 多 | 次优 |
| Dolphin | 大 | 多 | 次优 |
| DeepSeek-OCR | 中 | 中(压缩) | 次优 |
| **PaddleOCR-VL** | **最少(0.9B)** | **最少** | **SOTA** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 端到端VLM | 基线 | 处理整页，token多 |
| 粗阶段(VRFM) | 高效定位 | 过滤39-60%冗余区域 |
| + 细阶段(VL-0.9B) | SOTA | 精细识别裁剪区域 |
| 无指针网络 | 阅读顺序差 | 验证排序模块必要性 |

### 关键发现

- PaddleOCR-VL 在文本、公式、表格、阅读顺序四个关键指标上均达到SOTA
- 参数量和视觉token数均为最少，推理延迟和吞吐量显著优于竞品
- 高质量数据流水线是性能的关键因素之一
- 在手写和历史文档等挑战性内容上表现出强鲁棒性
- 支持109种语言的多语言文档解析

## 亮点与洞察

- **文档视觉冗余的统计分析**提供了有说服力的动机：PPT文档仅39%有效区域，直接证明了选择性处理的必要性
- **解耦设计**允许各模块独立优化是实用优势——可以单独升级检测器或识别模型
- **0.9B参数+最少token达到SOTA**证明了"聪明地选择在哪里投入计算"比"用更大模型处理所有内容"更有效

## 局限与展望

- 两阶段流水线引入级联误差——VRFM的检测错误会传导到识别阶段
- 密集排布页面上VRFM的定位精度可能受限
- 阅读顺序预测在极复杂布局（多栏混排+浮动元素）下可能不准确
- 仅在文档解析场景验证，未扩展到更广泛的VLM应用

## 相关工作与启发

- **vs MinerU2.5/Dolphin**: 统一端到端VLM，但参数大、效率低；PaddleOCR-VL通过粗到细解耦实现更高效率
- **vs DeepSeek-OCR**: 统一压缩视觉token，但会损害布局精度；PaddleOCR-VL选择性丢弃无效区域而非均匀压缩
- **vs 流水线方法**: 传统流水线使用多个独立专家模型，复杂且易误差累积；PaddleOCR-VL仅两个模块，更简洁

## 评分

- 新颖性: ⭐⭐⭐⭐ 粗到细解耦+有效区域聚焦的思路清晰有效
- 实验充分度: ⭐⭐⭐⭐⭐ 多基准全面验证，公私数据集覆盖广
- 写作质量: ⭐⭐⭐⭐ 动机分析有数据支撑
- 价值: ⭐⭐⭐⭐⭐ 开源代码+模型，实际可用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Efficient Document Parsing via Parallel Token Prediction](efficient_document_parsing_via_parallel_token_prediction.md)
- [\[CVPR 2026\] Towards Real-World Document Parsing via Realistic Scene Synthesis and Document-Aware Training](towards_real-world_document_parsing_via_realistic_scene_synthesis_and_document-a.md)
- [\[CVPR 2026\] Multimodal OCR: Parse Anything from Documents](multimodal_ocr_parse_anything_from_documents.md)
- [\[CVPR 2026\] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](docseeker_long_document_understanding.md)
- [\[CVPR 2026\] VL-RouterBench: A Benchmark for Vision-Language Model Routing](vl-routerbench_a_benchmark_for_vision-language_model_routing.md)

</div>

<!-- RELATED:END -->
