---
title: >-
  [论文解读] A Survey on MLLM-based Visually Rich Document Understanding: Methods, Challenges, and Emerging Trends
description: >-
  [ACL 2026][多模态VLM][视觉丰富文档理解] 系统综述基于多模态大语言模型（MLLM）的视觉丰富文档理解（VRDU），从特征表示/融合和训练范式两个维度梳理OCR-based和OCR-free方法，并讨论数据稀缺、多页文档、多语言支持、RAG和智能体等新兴方向。
tags:
  - "ACL 2026"
  - "多模态VLM"
  - "视觉丰富文档理解"
  - "多模态大语言模型"
  - "OCR"
  - "文档信息抽取"
  - "检索增强生成"
---

# A Survey on MLLM-based Visually Rich Document Understanding: Methods, Challenges, and Emerging Trends

**会议**: ACL 2026  
**arXiv**: [2507.09861](https://arxiv.org/abs/2507.09861)  
**代码**: 无  
**领域**: Document Understanding / Multimodal LLM  
**关键词**: 视觉丰富文档理解, 多模态大语言模型, OCR-free, 文档信息抽取, 检索增强生成

## 一句话总结

系统综述基于多模态大语言模型（MLLM）的视觉丰富文档理解（VRDU），从特征表示/融合和训练范式两个维度梳理OCR-based和OCR-free方法，并讨论数据稀缺、多页文档、多语言支持、RAG和智能体等新兴方向。

## 研究背景与动机

**领域现状**：视觉丰富文档理解（VRDU）旨在从包含复杂视觉、文本和布局元素的文档中自动提取和理解信息，在金融、医疗、教育等领域有广泛应用。随着MLLM的快速发展，该领域正经历从传统方法到MLLM-based方法的范式转变。

**现有痛点**：(1) 早期方法依赖OCR管道，误差会级联传播；(2) 文档的多模态性（文本、视觉、布局）增加了特征融合的复杂性；(3) 标注数据稀缺制约了监督学习方法；(4) 多页、多语言文档处理仍是难题。

**核心矛盾**：MLLM在通用视觉-语言任务上表现出色，但文档理解有其特殊性——需要理解精确的布局关系、表格结构和印刷/手写文本，通用MLLM难以直接胜任。

**本文目标**：提供一个全面的MLLM-based VRDU综述，覆盖方法分类、训练策略、挑战和未来方向，为研究者提供系统性路线图。

**切入角度**：从两个核心维度组织——(1) 文本、视觉、布局特征的表示与融合技术；(2) 预训练、指令微调和训练策略。

**核心 idea**：MLLM-based VRDU正从OCR-dependent向OCR-free演进，同时从单页静态理解向多页动态交互（RAG、智能体）扩展。

## 方法详解

### 整体框架

综述将MLLM-based VRDU方法分为两大类：OCR-Dependent（需要外部OCR输出作为文本输入）和OCR-Free（端到端从文档图像直接理解），并在每类中按特征融合方式和LLM骨干进行细分。

### 关键设计

1. **OCR-Dependent方法**:

    - 功能：利用外部OCR引擎提供的文本和布局信息，结合LLM进行文档理解
    - 核心思路：典型代表包括DocLLM（使用交叉注意力融合文本和布局）、ICL-D3IE（利用GPT-3的in-context learning处理文档IE）、LayoutLLM（结合LayoutLMv3的布局编码和Vicuna的生成能力）
    - 设计动机：OCR提供精确的文本内容，但引入了OCR误差传播和管道复杂性的trade-off

2. **OCR-Free方法**:

    - 功能：直接从文档图像端到端地理解文档，无需外部OCR
    - 核心思路：代表方法如mPLUG-DocOwl系列（基于视觉编码器直接感知文档图像）、TextMonkey（使用sliding window处理高分辨率文档）、InternVL-based方法（动态分辨率处理）
    - 设计动机：消除OCR误差传播，实现真正端到端，但需要视觉编码器能精确识别文档中的细粒度文本

3. **训练范式分类**:

    - 功能：系统化MLLM的训练流程
    - 核心思路：三阶段范式——预训练（PT）学习文档表示基础，指令微调（IT）对齐任务指令理解，下游微调（FT）适配特定任务。不同方法采用不同的阶段组合
    - 设计动机：理解训练策略的选择对最终性能的影响，帮助研究者做出informed的设计决策

### 损失函数 / 训练策略

综述覆盖的方法使用多种训练策略：标准自回归语言建模损失、对比学习（如CLIP-style）、文本-布局对齐损失等。预训练通常使用大规模文档-文本对，指令微调使用结构化QA格式。

## 实验关键数据

### 主要模型对比

| 方法 | 类型 | 任务 | 模态 | LLM骨干 | 多页 |
|------|------|------|------|---------|------|
| DocLLM | OCR-Dep | KIE, QA, DC | T, L | Custom | 单页 |
| LayoutLLM | OCR-Dep | KIE, QA | T, V, L | Vicuna-7B | 单页 |
| mPLUG-DocOwl | OCR-Free | QA | V | mPLUG-Owl | 单页 |
| TextMonkey | OCR-Free | QA | V | Qwen-VL | 单页 |
| InternVL-Doc | OCR-Free | QA, KIE | V | InternVL | 多页 |
| DocThinker | OCR-Free | QA, KIE | T, V | Qwen2.5-VL | 单页 |

### 挑战与趋势

| 挑战 | 当前状态 | 未来方向 |
|------|---------|---------|
| 数据稀缺 | 合成数据+迁移学习 | 自监督预训练+少样本学习 |
| 多页文档 | 少数方法支持 | 动态页面选择+检索增强 |
| 多语言 | 英语为主 | 多语言预训练+跨语言迁移 |
| RAG集成 | 初步探索 | 文档检索+生成pipeline |
| 智能体框架 | 新兴方向 | 多工具协作的文档理解agent |

### 关键发现
- OCR-Free方法正快速追赶OCR-Dependent方法，尤其在高分辨率视觉编码器的支持下
- 多页文档理解是当前最大瓶颈，大多数方法仍仅支持单页
- RAG和智能体框架的引入为文档理解提供了从"理解"到"应用"的新路径

## 亮点与洞察
- 综述的分类维度设计清晰：从OCR依赖性×特征融合×训练范式三个维度构建了完整的方法空间
- 模型总结表格非常实用，涵盖LLM骨干、视觉编码器、训练阶段、多页支持、prompt格式等关键信息
- 对新兴方向（RAG、智能体）的前瞻性讨论为后续研究指明方向

## 局限与展望
- 综述截至2025年中的方法，未来MLLM的快速发展可能很快使部分内容过时
- 缺少统一benchmark上的定量对比，难以直接比较不同方法的性能
- 对计算成本和效率的讨论不够深入
- 未来方向：(1) 统一的多任务多页文档理解框架；(2) 可信赖的文档理解（幻觉控制）；(3) 文档理解与知识图谱的结合

## 相关工作与启发
- **vs 传统文档理解综述**: 聚焦MLLM时代的新方法，涵盖更多OCR-free和生成式方法
- **vs 通用MLLM综述**: 深入文档理解的特殊需求（布局感知、表格理解、高分辨率）
- **vs 文档AI应用综述**: 更注重技术方法分类而非应用场景

## 评分
- 新颖性: ⭐⭐⭐ 综述类文章，重在系统性而非原创性
- 实验充分度: ⭐⭐⭐ 全面的方法覆盖，但缺少统一定量对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，分类维度合理，表格信息密度高
- 价值: ⭐⭐⭐⭐ 对VRDU领域的研究者有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding](slideagent_hierarchical_agentic_framework_for_multi-page_visual_document_underst.md)
- [\[ACL 2026\] A Survey of Multimodal Mathematical Reasoning: From Perception, Alignment to Reasoning](a_survey_of_multimodal_mathematical_reasoning_from_perception_alignment_to_reaso.md)
- [\[CVPR 2025\] Relation-Rich Visual Document Generator for Visual Information Extraction](../../CVPR2025/multimodal_vlm/relation-rich_visual_document_generator_for_visual_information_extraction.md)
- [\[AAAI 2026\] Exo2Ego: Exocentric Knowledge Guided MLLM for Egocentric Video Understanding](../../AAAI2026/multimodal_vlm/exo2ego_exocentric_knowledge_guided_mllm_for_egocentric_vide.md)
- [\[ICLR 2026\] Index-Preserving Lightweight Token Pruning for Efficient Document Understanding](../../ICLR2026/multimodal_vlm/index-preserving_lightweight_token_pruning_for_efficient_document_understanding_.md)

</div>

<!-- RELATED:END -->
