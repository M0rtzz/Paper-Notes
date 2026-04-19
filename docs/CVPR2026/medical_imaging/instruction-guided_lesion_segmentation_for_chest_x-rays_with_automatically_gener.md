---
title: >-
  [论文解读] Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset
description: >-
  [CVPR 2026][医学图像][胸部X光] 提出指令引导的胸部X光病变分割任务（ILS），构建了首个大规模自动生成的指令-回答数据集MIMIC-ILS（1.1M样本、192K图像、91K mask），并训练ROSALIA模型实现gIoU 71.2%和空目标准确率91.8%，远超现有通用和医学分割模型。
tags:
  - CVPR 2026
  - 医学图像
  - 胸部X光
  - 病变分割
  - 指令引导
  - 自动数据集构建
  - 视觉语言模型
---

# Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset

**会议**: CVPR 2026  
**arXiv**: [2511.15186](https://arxiv.org/abs/2511.15186)  
**代码**: [GitHub](https://github.com/checkoneee/ROSALIA)  
**领域**: 医学图像  
**关键词**: 胸部X光, 病变分割, 指令引导, 自动数据集构建, 视觉语言模型

## 一句话总结

提出指令引导的胸部X光病变分割任务（ILS），构建了首个大规模自动生成的指令-回答数据集MIMIC-ILS（1.1M样本、192K图像、91K mask），并训练ROSALIA模型实现gIoU 71.2%和空目标准确率91.8%，远超现有通用和医学分割模型。

## 研究背景与动机

胸部X光（CXR）是最常见的医学影像检查之一，病变定位和边界识别是放射科医生的核心工作，但这一过程劳动密集且需要高度临床专业知识。

现有CXR病变分割面临两大瓶颈：
1. **标注规模有限**：现有数据集（VinDr-CXR 15K图像、SIIM-ACR 13K图像）依赖专家手动标注，规模受限且多数仅提供bounding box或单一病变类型mask
2. **用户输入门槛高**：已有文本引导分割方法要求用户提供专家级的详细描述（如"双侧肺部感染，两个感染区域…"），非专业用户无法使用

核心矛盾：如何在无人工标注的情况下大规模生成高质量的病变mask和指令-回答对，且支持简单易用的用户指令？

本文切入角度：利用MIMIC-CXR中现有的图像-报告配对数据，通过多模态自动化pipeline从影像和报告中提取空间信息和文本信息，生成全自动标注的大规模ILS数据集。

## 方法详解

### 整体框架

系统分为两大阶段：
1. **定位病变mask生成**：从CXR图像和放射报告中自动生成病变分割mask
2. **指令-回答对生成**：基于上一步提取的信息构建多样化的指令-回答训练样本

模型ROSALIA基于LISA架构，整合VLM（LLaVA）和SAM进行端到端训练。

### 关键设计

1. **多模态自动mask生成管线**:
    - 功能：从原始CXR图像（无标注）自动生成高质量病变分割mask
    - 核心思路：四步级联流程——
      1. **报告结构化**：用LLM将放射报告中的异常描述转化为六元组（实体、句子索引、存在性、确定性、位置、病变类型），并将位置映射到标准解剖标签
      2. **空间信息提取**：并行使用三个视觉模型——RadEdit（扩散模型生成异常图 $\mathcal{A}$，通过生成"无病变"图像与原图做差）、CXAS（解剖分割模型提供解剖mask $\{\mathcal{M}_i\}$）、YOLO（病变检测得到bounding box $\{\mathcal{B}_j\}$）
      3. **mask生成**：通过四条件过滤（解剖重叠c1 + 检测置信度c2 + 异常信号比c3 + 最小尺寸c4）筛选高质量候选框，提取与筛选框相交的连通区域并精炼
      4. **位置验证**：确认生成的mask是否成功定位到报告描述的位置，同时标记空白位置用于负样本生成
    - 设计动机：通过多模态交叉验证确保mask质量——文本信息提供"在哪里"、视觉异常图提供"哪里不正常"、检测模型提供"界限"。四条件过滤有效排除假阳性

2. **指令-回答对生成系统**:
    - 功能：基于定位信息自动构建多样化的训练样本
    - 核心思路：支持三种指令类型——
        - **基础指令**：指定病变类型+位置（如"分割右肺的肺炎"），仅当mask成功定位时生成
        - **全局指令**：仅指定病变类型（如"分割不透明影"），仅当定位位置与报告位置完全一致时生成
        - **病变推理指令**：要求模型预测opacity的具体类型，将肺炎/肺不张/水肿替换为"opacity"
    - 负样本生成策略：利用报告中未提及或显式否定的病变类型、以及空白位置替换正样本位置
    - 设计动机：三种指令类型覆盖从具体到通用的不同用户需求。动态生成策略确保只产生有效的指令-回答对，避免不一致

3. **ROSALIA模型架构**:
    - 功能：基于用户指令生成病变分割mask和文本描述
    - 核心思路：基于LISA-7B架构，VLM（LLaVA）处理图像和指令输入，生成特殊[SEG] token和文本描述。[SEG] token的隐藏嵌入传递给SAM-H的mask解码器生成最终分割mask
    - 训练策略：LoRA微调VLM（rank=128, alpha=256），全微调mask解码器。15个epoch，AdamW，batch size 256，正负样本1:1

### 损失函数 / 训练策略

$$\mathcal{L} = \lambda_{txt}\mathcal{L}_{txt} + \lambda_{bce}\mathcal{L}_{bce} + \lambda_{dice}\mathcal{L}_{dice}$$

- $\mathcal{L}_{txt}$：自回归交叉熵损失（文本生成），$\lambda_{txt}=0.5$
- $\mathcal{L}_{bce}$：二元交叉熵损失（分割），$\lambda_{bce}=5$
- $\mathcal{L}_{dice}$：DICE损失（分割，仅对正样本计算），$\lambda_{dice}=1$

## 实验关键数据

### 主实验

| 模型 | gIoU | cIoU | N-Acc. | 说明 |
|------|------|------|--------|------|
| LISA-7B | 8.3% | 12.8% | 0.7% | 通用域 |
| LISA-13B | 8.9% | 12.2% | 0.0% | 通用域 |
| Text4Seg | 6.1% | 10.3% | 20.6% | 通用域 |
| BiomedParse | 23.8% | 18.5% | 0.6% | 医学域 |
| RecLMIS | 22.4% | 19.5% | 0.0% | 医学域 |
| **ROSALIA (Ours)** | **71.2%** | **75.6%** | **91.8%** | MIMIC-ILS训练 |

### 各病变类型性能

| 病变类型 | gIoU | cIoU | N-Acc. |
|---------|------|------|--------|
| 心脏肥大 | 89.0% | 89.0% | 85.8% |
| 肺炎 | 57.2% | 60.4% | 97.1% |
| 肺不张 | 60.2% | 58.7% | 91.7% |
| 不透明影 | 60.5% | 64.2% | 85.0% |
| 实变 | 61.9% | 65.6% | 91.2% |
| 水肿 | 64.8% | 66.6% | 92.2% |
| 胸腔积液 | 60.3% | 59.6% | 90.4% |

### 消融实验 — 数据集质量评估

| 专家 | 总接受率 | 正样本接受率 | 负样本接受率 |
|------|---------|-------------|-------------|
| 专家A | 96.1% | 95.6% | 96.5% |
| 专家B | 97.2% | 96.0% | 98.3% |
| 专家C | 98.7% | 99.8% | 97.8% |
| 专家D | 97.6% | 96.9% | 98.2% |
| **总体** | **96.4%** | 90.1% | 97.7% |

### 关键发现

- 现有通用/医学域分割模型在ILS任务上系统性失败，gIoU低于24%且几乎无法处理空目标场景（N-Acc接近0）
- 全自动生成的数据集经4位放射肿瘤科专家评估，整体接受率高达96.4%
- 文本回答准确率94.4%，其中基础指令96.8%最高，病变推理指令84.8%有提升空间
- 心脏肥大分割最好（gIoU 89.0%），因为使用心脏mask作为标注；肺炎等局部病变稍低

## 亮点与洞察

- 全自动数据集构建pipeline是核心贡献，通过多模态交叉验证实现了媲美人工标注的质量
- ILS任务定义具有临床实用性：支持简单指令而非专家级描述，且支持空目标检测（"没有发现病变"）
- 数据集规模（1.1M样本）是现有CXR分割数据集的10-100倍
- RadEdit用于生成异常图的方法很巧妙——用扩散模型生成"正常"图像，通过差分定位异常区域

## 局限与展望

- 自动标注质量：正样本接受率90.1%低于负样本的97.7%，正样本标注的精度仍需提升
- 仅覆盖7种主要病变类型，CXR中还有更多细粒度异常
- 病变推理任务（opacity→具体类型）准确率75.1%相对较低
- pipeline依赖RadEdit、CXAS、YOLO三个预训练模型，任一模型失效会影响整体质量
- 仅在MIMIC-CXR上验证，不同机构的CXR风格差异可能影响泛化

## 相关工作与启发

- **vs BiomedParse**: 虽然是医学域模型但仅支持类标签prompt，无法处理指令级输入和空目标检测
- **vs RecLMIS**: 需要用户提供专家级描述（"双侧肺部感染…"），使用门槛高
- **vs LISA**: ROSALIA基于LISA架构但在MIMIC-ILS上微调后性能从8.3%跃升至71.2%，证明了任务特定数据的重要性

## 评分

- 新颖性: ⭐⭐⭐⭐ 全自动数据集构建pipeline和ILS任务定义均有创新
- 实验充分度: ⭐⭐⭐⭐ 多种基线对比、分病变类型评估、专家质量验证，但缺少跨数据集泛化实验
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，pipeline描述详细，图表丰富
- 价值: ⭐⭐⭐⭐⭐ 数据集规模和公开的代码/数据对社区有很高的实用价值

<!-- RELATED:START -->

## 相关论文

- [CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays](../../NeurIPS2025/medical_imaging/cxreasonbench_a_benchmark_for_evaluating_structured_diagnostic_reasoning_in_ches.md)
- [AFD-INSTRUCTION: A Comprehensive Antibody Instruction Dataset with Functional Annotations for LLM-Based Understanding and Design](../../ICLR2026/medical_imaging/afd-instruction_a_comprehensive_antibody_instruction_dataset_with_functional_ann.md)
- [CheX: Interactive Localization and Region Description in Chest X-rays](../../ECCV2024/medical_imaging/chex_interactive_localization_and_region_description_in_chest_x-rays.md)
- [GEMeX: A Large-Scale, Groundable, and Explainable Medical VQA Benchmark for Chest X-ray Diagnosis](../../ICCV2025/medical_imaging/gemex_a_large-scale_groundable_and_explainable_medical_vqa_benchmark_for_chest_x.md)
- [Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation](../../AAAI2026/medical_imaging/small_but_mighty_dynamic_wavelet_expert-guided_fine-tuning_of_large-scale_models.md)

<!-- RELATED:END -->
