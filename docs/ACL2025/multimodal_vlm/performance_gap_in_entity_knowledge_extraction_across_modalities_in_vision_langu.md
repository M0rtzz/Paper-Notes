---
title: >-
  [论文解读] Performance Gap in Entity Knowledge Extraction Across Modalities in Vision Language Models
description: >-
  [ACL 2025][多模态][VLM] 系统性地揭示了视觉语言模型（VLM）在视觉 vs 文本表征下实体知识提取的显著性能差距（最高达 18%），通过机制可解释性工具发现图像 token 的关键信息流发生在模型中间层很深处，导致后续事实推理的层数不足。
tags:
  - ACL 2025
  - 多模态
  - VLM
  - 实体知识提取
  - 跨模态差距
  - 机制可解释性
  - POPVQA
---

# Performance Gap in Entity Knowledge Extraction Across Modalities in Vision Language Models

**会议**: ACL 2025  
**arXiv**: [2412.14133](https://arxiv.org/abs/2412.14133)  
**代码**: 有（POPVQA 数据集及实验代码）  
**领域**: Multimodal / Vision-Language Models  
**关键词**: VLM, 实体知识提取, 跨模态差距, 机制可解释性, POPVQA

## 一句话总结

系统性地揭示了视觉语言模型（VLM）在视觉 vs 文本表征下实体知识提取的显著性能差距（最高达 18%），通过机制可解释性工具发现图像 token 的关键信息流发生在模型中间层很深处，导致后续事实推理的层数不足。

## 研究背景与动机

VLM 在图像理解任务上表现优异，但一个关键问题被忽略了：**当实体以图像而非文本呈现时，模型能否同样好地利用其内部知识？**

场景：对于实体 Robin Williams——
- 文本表征：直接问 "Who is Robin Williams's spouse?"
- 视觉表征：给一张 Robin Williams 的图片，问 "Who is the spouse of the subject in this image?"

后者本质上是一个**两跳推理任务**：第一跳识别图像中的实体（identification），第二跳从内部知识中提取相关事实（extraction）。作者假设，VLM 会把大量计算预算花在第一跳的实体识别上，导致留给第二跳事实推理的层数不够。

现有数据集（如 InfoSeek、VeQuAE）不包含图中实体的显式标注，难以分离识别和提取两个步骤。因此作者构建了专用数据集 POPVQA。

## 方法详解

### 整体框架

本文不是一个新模型，而是一个**诊断性研究**，包含：
1. POPVQA 数据集构建
2. 多模型基准对比实验
3. 激活修补（activation patching）实验
4. 前向激活冻结实验
5. 注意力剔除（attention knockout）实验

### 关键设计

1. **POPVQA 数据集**：做什么→构建一个允许分离实体识别和问答的视觉事实问答数据集；核心思路→包含 15,395 个实体-图像对（名人 63.6%、地标 18%、绘画 14.6%、品牌 3.8%），每个实体至少 2 个基于 Wikidata 三元组生成的事实问题；设计动机→通过先让模型识别图像中实体来确认其"认识"该实体，再评估知识提取能力，从而将识别失败和知识提取失败解耦。图像统一缩放为 336×336，基于 Wikipedia 页面浏览量筛选高人气实体。

2. **跨修补（Cross Patching）实验**：做什么→在不同实体的两次前向传播间，将注入实体的图像隐状态替换到原始实体的前向传播中；核心思路→在第 $\ell$ 层将注入实体的 $\mathbf{h}_{1,v}^\ell \ldots \mathbf{h}_{n,v}^\ell$ 替换原始实体的对应位置，观察模型预测从注入实体切换到原始实体的**临界层**；设计动机→找到模型完成图像表征处理的分界点，即图像 token 信息传播到 query/generation token 位置的关键层。

3. **前向激活冻结实验**：做什么→从某源层开始到第 20 层，将图像 token 的隐状态"冻结"为源层的值；核心思路→设 $(\mathbf{h}_{1,v}^\ell \ldots \mathbf{h}_{n,v}^\ell) = (\mathbf{h}_{1,v}^{source} \ldots \mathbf{h}_{n,v}^{source})$，对 $\ell$ 从 source 到 20；设计动机→检验实体是否能在早期层就被识别——如果冻结从早期层开始仍能正确识别，说明视觉编码器已提供足够信号。

4. **注意力剔除实验**（附录）：做什么→在注意力 mask 中阻断文本/生成 token 对图像 token 的注意力；核心思路→自顶向下和自底向上两个方向逐层剔除，观察哪些层的阻断最致命；设计动机→验证跨模态信息流发生在哪些层。

### 损失函数 / 训练策略

本文为分析性工作，不涉及模型训练。

## 实验关键数据

### 主实验（表格）

**跨模态性能差距（POPVQA 数据集）**

| 模型 | 识别实体数 | 图像准确率 | 文本准确率 | 差距 |
|------|-----------|-----------|-----------|------|
| LLaVA7B | 925 | 0.276 | 0.453 | **0.177** |
| LM-CLIP | 844 | 0.216 | 0.378 | 0.163 |
| LM-SigLIP | 660 | 0.201 | 0.377 | 0.176 |
| LLaVA34B | 1,286 | 0.534 | 0.656 | **0.121** |
| Qwen2-VL | 3,143 | 0.433 | 0.476 | **0.043** |

### 消融实验（表格）

**早期 vs 晚期识别实体的 QA 准确率对比**

| 模型 | 分组 | 图像准确率 | 文本准确率 | 差距 |
|------|------|-----------|-----------|------|
| LM-CLIP | 早期识别 | 0.260 | 0.442 | 0.182 |
| LM-CLIP | 晚期识别 | 0.192 | 0.347 | 0.154 |
| LM-SigLIP | 早期识别 | 0.234 | 0.417 | 0.183 |
| LM-SigLIP | 晚期识别 | 0.168 | 0.338 | 0.170 |

### 关键发现

1. **普遍存在的性能差距**：所有测试模型都存在视觉 vs 文本表征的准确率下降，最高达 17.7%（LLaVA7B）。
2. **模型深度有帮助**：LLaVA34B 的差距（12.1%）明显小于 7B 版本（17.7%），更深的模型有更多层用于第二跳推理。
3. **训练方案更重要**：Qwen2-VL 差距仅 4.3%，得益于训练时解冻视觉 backbone 参数，实现更好的模态对齐。
4. **关键信息流发生在中间层之后**：LLaVA7B 的临界点在第 20 层（共 32 层），LM-SigLIP 在第 17 层，LM-CLIP 在第 24 层。在这之前修补图像 token 都会导致模型预测注入实体。
5. **早期识别不等于更好的QA**：即使某些实体可以在前5层就被识别（因视觉编码器已提供足够信号），模型行为与文本准确率的提升是相关的，表明这是因为模型对该实体知识更丰富而非利用了早期识别优势。差距反而更大，说明模型未能有效利用早期识别的优势。
6. **常见错误**：模型在被问及实体的配偶/父母/兄弟时，经常输出实体本身的名字——表明信息传播不充分。

## 亮点与洞察

- **两跳推理的精准建模**：将 VLM 的视觉实体 QA 分解为 identification + extraction 两步，提供了清晰的分析框架。
- **跨实体修补方法新颖**：通过将一个实体的图像隐状态注入另一个实体的前向传播来精确定位信息传播时机。
- **反直觉发现**：与认知心理学中"图像优越效应"（humans perform better with images）相矛盾，VLM 表现了"文本优越"。
- **模型不能 efficient 利用层**：即使视觉编码器已经完成了实体识别（早期层已包含足够信息），LLM 部分仍需到中间层才开始利用这些信息。

## 局限与展望

1. **机制分析仅限 LLaVA 架构**：其简单架构利于干预实验，但其他架构（如 Qwen2-VL）的有利结果未做深入分析。
2. **图像中的混淆因素**：运动员穿国家队球衣可能被误识别，图像中的视觉线索可能引导或误导答案。
3. **实体类型有限**：主要是名人、地标、品牌和画作，缺少对更多类型实体和不同种族的覆盖。
4. **未提出解决方案**：只诊断了问题，没有提出缓解差距的方法（作者自己也承认）。
5. **可能的方向**：早期融合模型（early-fusion）可能缓解该问题，通过创建视觉和文本的共享嵌入空间。

## 相关工作与启发

- **机制可解释性**：延续了 Geva et al. 的 attention knockout 和 activation patching 方法论
- **VLM 可解释性**：与 Neo et al. 和 Kaduri et al. 的同期工作互补，后者发现 query token 在中间层存储高层图像描述
- **多模态知识冲突**：Zhu et al. 提出检测和缓解参数化知识冲突的方法，本文提供了更底层的机制解释
- **数据集设计启发**：POPVQA 展示了如何通过精心设计数据集来解耦复合能力评测

## 评分

- **新颖性**: ★★★★☆ — 跨模态知识提取差距的系统挖掘和两跳推理视角很有价值
- **实验充分度**: ★★★☆☆ — 分析深入但仅限于 LLaVA 架构的机制实验，且样本量有限
- **写作质量**: ★★★★☆ — 逻辑清晰、图表直观，实验设计环环相扣
- **价值**: ★★★★☆ — 揭示了 VLM 的根本性信息处理瓶颈，对模型设计有重要启示

<!-- RELATED:START -->

## 相关论文

- [Vision-Language Models Struggle to Align Entities across Modalities](vision-language_models_struggle_to_align_entities_across_modalities.md)
- [Self-Supervised Spatial Correspondence Across Modalities](../../CVPR2025/multimodal_vlm/self-supervised_spatial_correspondence_across_modalities.md)
- [A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks](../../CVPR2025/multimodal_vlm/a_closed-form_solution_for_debiasing_vision-language_models_with_utility_guarant.md)
- [Weaving Context Across Images: Improving Vision-Language Models through Focus-Centric Visual Chains](weaving_context_across_images_improving_vision-language_models_through_focus-cen.md)
- [Insight Over Sight: Exploring the Vision-Knowledge Conflicts in Multimodal LLMs](conflictvis_vision_knowledge_conflict.md)

<!-- RELATED:END -->
