---
title: >-
  [论文解读] SciPostGen: Bridging the Gap between Scientific Papers and Poster Layouts
description: >-
  [CVPR 2026][多模态VLM][海报布局生成] 构建了包含 18,097 个论文-海报对的大规模数据集 SciPostGen，分析发现论文结构与海报布局元素数量存在中等相关性，并提出检索增强海报布局生成框架，通过对比学习检索与论文匹配的布局模板来指导 LLM 生成海报布局。
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "海报布局生成"
  - "科学论文"
  - "检索增强生成"
  - "对比学习"
  - "文档布局分析"
---

# SciPostGen: Bridging the Gap between Scientific Papers and Poster Layouts

**会议**: CVPR 2026  
**arXiv**: [2511.22490](https://arxiv.org/abs/2511.22490)  
**代码**: [https://omron-sinicx.github.io/paper2layout/](https://omron-sinicx.github.io/paper2layout/)  
**领域**: 多模态VLM / 文档理解  
**关键词**: 海报布局生成, 科学论文, 检索增强生成, 对比学习, 文档布局分析

## 一句话总结

构建了包含 18,097 个论文-海报对的大规模数据集 SciPostGen，分析发现论文结构与海报布局元素数量存在中等相关性，并提出检索增强海报布局生成框架，通过对比学习检索与论文匹配的布局模板来指导 LLM 生成海报布局。

## 研究背景与动机

科学论文数量持续增长（arXiv 月投稿量从 2015 年约 8000 篇增长到 2025 年超 2 万篇），海报是高效传达研究成果的重要媒介。自动从论文生成海报需解决两个问题：内容摘要（放什么）和布局生成（怎么排）。

现有工作主要聚焦内容摘要，布局要么用固定模板要么用基于论文结构的规则生成。然而，布局设计对信息传达效果有重要影响，值得数据驱动地学习论文到布局的映射关系。

核心瓶颈是**缺乏大规模配对数据集**。现有海报生成数据集仅包含几百个论文-海报对，不足以支持数据驱动的方法。SciPostGen 通过结合自动标注和人工校正，将规模扩大到 18,097 对，同时提供论文（OCR 文本、图表包围框）和海报（8 类布局元素标注）的细粒度标注。

分析发现论文结构与海报布局存在可利用的相关性：论文文本量越多，海报中图元素越少（Spearman $\rho < -0.40$）；论文图表数量与海报图元素正相关。这启发了检索增强的布局生成策略——检索结构相似的论文的海报布局作为生成参考。

## 方法详解

### 整体框架

系统由两个模块组成：(1) 布局检索器——基于对比学习训练的论文编码器和布局编码器，将论文页面图像和海报布局图像映射到共享嵌入空间，推理时检索 top-3 最相似布局；(2) 布局生成器——基于 LLM（Llama-3.1-8B-Instruct），接收检索到的布局和论文结构信息作为输入，输出最终布局（类别+归一化包围框）。支持自动和半自动两种模式。

### 关键设计

1. **对比学习布局检索器**:

    - 功能：检索与给定论文结构匹配的海报布局
    - 核心思路：论文编码器将多页论文 PDF 渲染为图像序列，每页经 DiT（文档图像 Transformer）提取 patch 特征，两级注意力池化（页内→页间）聚合为论文嵌入 $x^p$。布局编码器对布局渲染图像做类似处理得到 $x^l$。用 InfoNCE 对比损失训练，配对的论文-布局为正样本，batch 内其他为负样本。推理时用余弦相似度从训练集检索 top-3 布局
    - 设计动机：论文结构（文本量、图表数）与海报布局存在中等相关性，直接用图像编码可隐式捕获这些结构特征。检索多个布局而非单一模板，适应海报设计的多样性

2. **LLM 布局生成器**:

    - 功能：整合检索结果和论文结构约束，生成最终布局
    - 核心思路：将论文结构（章节数、图表数量和宽高比）和检索到的布局以文本序列形式输入 LLM，指示模型生成布局序列 $L = \{(c_i, b_i)\}$。在半自动模式下还接收用户指定的部分布局约束（如预放置的两个最大元素），模型需在约束内补全剩余元素
    - 设计动机：LLM 具有灵活整合非结构化输入的能力，比 GAN/Transformer/Diffusion 等专用布局模型更容易融合检索结果、论文结构和用户约束等异构信息

3. **半自动约束机制**:

    - 功能：模拟实际工作流，创作者放置主要元素后系统补全剩余布局
    - 核心思路：取 gold layout 中面积最大的两个元素作为约束条件输入，系统生成剩余元素。这模拟了"人先定大框架、AI 补充细节"的协作模式
    - 设计动机：完全自动生成难以满足个性化需求，半自动模式在实用性和自动化之间取得平衡

### 损失函数 / 训练策略

检索器用 InfoNCE 对比损失训练：$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(s_{ii})}{\sum_{j=1}^{N}\exp(s_{ij})}$，其中 $s_{ij}$ 为论文和布局嵌入的余弦相似度。生成器基于 Llama-3.1-8B-Instruct 微调，使用 SciPostGen 训练集的 silver layout 标注。

## 实验关键数据

### 主实验

**布局检索性能（论文→布局检索）**

| 方法 | Recall@1 | Recall@3 | Recall@5 |
|------|----------|----------|----------|
| Random | 0.05 | 0.15 | 0.25 |
| 仅论文编码器 | 4.83 | 12.12 | 18.11 |
| 完整检索器 | **8.20** | **19.87** | **28.37** |

**布局生成质量（FID / mIoU / Alignment）**

| 配置 | FID ↓ | mIoU ↑ | Overlap ↓ |
|------|-------|--------|-----------|
| 无检索 | 基线 | 基线 | 基线 |
| + 检索增强 | 改善 | 改善 | 减少 |
| + 检索 + 约束（半自动） | 最优 | 最优 | 最低 |

### 消融实验

| 配置 | 检索 Recall@3 | 生成 FID | 说明 |
|------|-------------|----------|------|
| 仅图像编码（DiT） | 19.87 | - | 基础检索性能 |
| 仅布局标注编码 | 更低 | - | 图像编码优于结构化标注 |
| 无检索直接生成 | - | 更高 | 无参考布局质量差 |
| 检索 top-1 | - | 中等 | 单模板多样性不足 |
| 检索 top-3 | - | 最低 | 多模板提供更好指导 |

### 关键发现

- 论文结构与海报布局的 Spearman 相关性为中等水平（|ρ| 约 0.40-0.50），说明结构信息有用但不足以完全决定布局
- 图像编码比直接用布局标注作为输入效果更好——图像隐式保留了空间关系
- 半自动模式下约束的加入显著提升了布局与真实布局的一致性
- silver（自动标注）和 gold（人工校正）布局的 mAP@0.50:0.95 为 0.53，属中等一致性

## 亮点与洞察

- **数据集构建方法论值得借鉴**：自动标注（Azure Document Intelligence + Nougat OCR）+ 人工校正验证/测试集，兼顾规模和质量。在标注资源有限时的实用策略
- **"论文结构→海报布局"这一研究问题本身有新意**：之前工作聚焦内容摘要，本文首次系统研究结构到布局的映射关系，定量分析了两者的相关性
- **检索增强策略**：通过检索相似论文的布局作为"参考设计"来指导生成，比从零生成更可控且多样性更好

## 局限与展望

- 仅生成布局（包围框），不生成实际海报内容（文字、图片），离端到端海报生成仍有距离
- 数据集限于计算机科学会议（CVPR/ICLR/ICML/NeurIPS），其他学科海报风格可能不同
- 检索 Recall@1 仅 8.2%，说明论文到布局的映射关系仍较弱，可能需要更丰富的论文表示
- 未评估生成布局的主观质量（如可读性、美观度），仅用数值指标衡量

## 相关工作与启发

- **vs PosterLayout [56]**: 提供了细粒度布局标注但仅数百对，SciPostGen 规模大 30 倍
- **vs 基于规则的布局**: 如 [42] 用预定义规则从论文结构推导布局，缺乏灵活性和多样性
- **vs 通用布局生成**: 广告/网页布局生成方法无法利用论文结构信息，本文引入论文作为条件

## 评分

- 新颖性: ⭐⭐⭐⭐ 研究问题新颖（论文→海报布局），数据集有价值，但方法本身是标准的检索增强+LLM 组合
- 实验充分度: ⭐⭐⭐ 缺乏用户研究和主观评估，检索和生成的定量指标不够全面
- 写作质量: ⭐⭐⭐⭐ 数据集构建和分析部分清晰，整体结构合理
- 价值: ⭐⭐⭐⭐ 数据集对社区有价值，框架为自动化学术海报生成奠定基础

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Responses Fall Short of Understanding: Revealing the Gap between Internal Representations and Responses in VDU](responses_fall_short_of_understanding_gap_between_internal_representations_and_responses_in_vdu.md)
- [\[AAAI 2026\] Bridging the Copyright Gap: Do Large Vision-Language Models Recognize and Respect Copyrighted Content?](../../AAAI2026/multimodal_vlm/bridging_the_copyright_gap_do_large_vision-language_models_r.md)
- [\[AAAI 2026\] Remember Me: Bridging the Long-Range Gap in LVLMs with Three-Step Inference-Only Decay Resilience Strategies](../../AAAI2026/multimodal_vlm/remember_me_bridging_the_long-range_gap_in_lvlms_with_three-step_inference-only_.md)
- [\[ACL 2025\] Can Multimodal Foundation Models Understand Schematic Diagrams? An Empirical Study on Information-Seeking QA over Scientific Papers](../../ACL2025/multimodal_vlm/can_multimodal_foundation_models_understand_schematic_diagrams_an_empirical_stud.md)
- [\[ICML 2026\] R$^3$L: Reasoning 3D Layouts from Relative Spatial Relations](../../ICML2026/multimodal_vlm/r3l_reasoning_3d_layouts_from_relative_spatial_relations.md)

</div>

<!-- RELATED:END -->
