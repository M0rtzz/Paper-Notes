---
title: >-
  [论文解读] EndoBench: A Comprehensive Evaluation of Multi-Modal Large Language Models for Endoscopy Analysis
description: >-
  [NeurIPS 2025][医学图像][内窥镜] 提出 EndoBench，首个覆盖 4 种内窥镜场景、12 项临床任务、5 级视觉提示粒度的综合 MLLM 评估基准，包含 6,832 个经临床验证的 VQA 对，对 23 个 MLLM 的评估显示商用模型整体领先但仍落后人类专家。
tags:
  - NeurIPS 2025
  - 医学图像
  - 内窥镜
  - MLLM评估
  - benchmark
  - VQA
  - 临床工作流
---

# EndoBench: A Comprehensive Evaluation of Multi-Modal Large Language Models for Endoscopy Analysis

**会议**: NeurIPS 2025  
**arXiv**: [2505.23601](https://arxiv.org/abs/2505.23601)  
**代码**: 已公开（benchmark + 代码）  
**领域**: 医学多模态 / 内窥镜分析  
**关键词**: 内窥镜, MLLM评估, benchmark, VQA, 临床工作流

## 一句话总结

提出 EndoBench，首个覆盖 4 种内窥镜场景、12 项临床任务、5 级视觉提示粒度的综合 MLLM 评估基准，包含 6,832 个经临床验证的 VQA 对，对 23 个 MLLM 的评估显示商用模型整体领先但仍落后人类专家。

## 研究背景与动机

**内窥镜 AI 需求**：胃肠和泌尿系统疾病是全球重大健康挑战，内窥镜是关键诊疗工具，但日益增长的检查需求需要 AI 辅助提升准确性和效率

**现有基准局限**：(a) 场景覆盖窄——Surgical-VQA 仅评估手术场景，ColonINST 仅覆盖结肠镜；(b) 任务维度少——Kvasir-VQA 仅关注基础识别任务；(c) 缺乏对多维能力的全面评估

**临床实际需求**：真实临床中，医生需要跨场景（胃镜/肠镜/胶囊内镜/手术内镜）工作，需要从解剖识别到手术干预的多层次能力，现有基准无法衡量 MLLM 是否具备这种全面能力

**核心矛盾**：MLLM 在内窥镜领域快速发展，但评估体系碎片化、不全面，无法准确反映临床实用性

## 方法详解

### 基准构建流程

EndoBench 的构建分为三个步骤：

**Step 1: 数据收集**
- 收集 20 个公开 + 1 个私有内窥镜数据集
- 涵盖诊断性影像（胃镜/肠镜/胶囊内镜）和治疗性影像（手术内镜）
- 所有数据经过隐私脱敏处理

**Step 2: QA 标准化**
- 与专业医生合作，设计 4 大类 12 项专业任务
- 每个任务创建 5-8 个不同问题模板
- 空间理解任务统一坐标格式为 $[x_1, y_1, x_2, y_2]$
- 生成合理干扰项：对视觉提示创建维度相似但重叠度低于阈值的替代框
- 产出 446,535 个图文对，构成 EndoVQA-Instruct 数据集（当前最大内窥镜指令调优集）

**Step 3: 数据筛选**
- 平衡实体分布 → DINO-V2 提取视觉嵌入 → K-center 聚类选取代表性样本
- GPT-4o-mini 改写问题以评估语言表达适应性
- 两位专业医生进行临床审核，剔除低质量样本
- 最终得到 6,832 个临床验证的 VQA 对

### 多场景覆盖

| 场景 | 缩写 | 样本数 | 占比 |
|------|------|--------|------|
| 胃镜 | GS | 583 | 8.53% |
| 肠镜 | CS | 1,848 | 27.05% |
| 胶囊内镜 | CE | 1,678 | 24.56% |
| 手术内镜 | SE | 2,723 | 39.86% |

### 多维能力评估框架

模仿临床工作流程设计的 4 大类 12 项任务：

1. **解剖结构识别**：器官识别、解剖标志识别
2. **病变分析与分级**：病变量化、病变类型识别、病变严重度分级
3. **空间定位与区域理解**：视觉定位、区域选择、区域识别
4. **手术工作流与操作分析**：术前评估、宏观阶段识别、微观操作分析、器械管理

### 视觉提示粒度

5 级粒度设计：图像级、框级、轮廓级、多框、多轮廓——从 segmentation mask 获取轮廓和框，用于评估细粒度感知能力。

## 实验关键数据

### 评估配置

- **模型总数**：23 个（13 开源通用 + 5 医学专用 + 5 商用）
- **开源模型**：LLaVA, LLaVA-Next, CogVLM, ShareGPT-4v, Qwen2.5-VL (3B-72B), Janus-Pro, InternVL2.5, QvQ
- **商用模型**：GPT-4o, Claude-3.7-Sonnet, Gemini-2.5-Pro, Deepseek-V3(视觉), Grok-3
- **医学模型**：MedDr, LLaVA-Med, HuatuoGPT-Vision, ColonGPT
- **人类基线**：2 名认证临床医生，每子任务 5-15 个样本，平均得分作为参考

### 主要实验结果

**整体准确率比较**（按类别）：

| 模型类型 | 解剖识别 | 病变分析 | 空间定位 | 手术操作 | 整体 |
|----------|----------|----------|----------|----------|------|
| 商用模型（最佳） | 最高 | 最高 | 最高 | 最高 | **领先** |
| 开源通用（最佳） | 次高 | 中等 | 中等 | 中等 | 第二 |
| 医学专用 | 中等 | 较高 | 较低 | 中等 | 第三 |
| 人类临床医生 | — | — | — | — | **最高** |

### 关键发现

1. **商用模型整体领先**：商用 MLLM 在所有类别上优于开源和医学专用模型，但仍落后于人类专家
2. **医学微调有效**：医学领域 SFT 显著提升任务特定准确率，证明领域适配的价值
3. **提示敏感性**：模型性能对提示格式和任务复杂度敏感，不同提示粒度下表现差异大
4. **空间理解薄弱**：所有模型在空间定位和区域理解任务上表现最差，是当前 MLLM 的主要短板
5. **规模效应**：更大参数量通常带来更好性能（如 Qwen2.5-VL-72B >> 3B）

### 与现有基准对比

| 基准 | 样本量 | 场景 | 任务数 | 粒度级别 | 数据源 |
|------|--------|------|--------|----------|--------|
| OmniMedVQA | 1,877 | - | 3 | 1 | 3 公开 |
| GMAI-MMBench | 3,749 | - | 7 | 4 | 16 公开 |
| Kvasir-VQA | 6,500 | GS,CS | 6 | 1 | 2 公开 |
| Surgical-VQA | 54K | SG | 5 | 1 | 2 公开 |
| ColonINST | 300K+ | CS | 4 | 1 | 19 公开 |
| **EndoBench** | **6,832** | **GS,CS,CE,SE** | **12** | **5** | **20+1** |

## 亮点

- ⭐⭐⭐⭐ **全面性**：首个覆盖 4 种内窥镜场景 × 12 项任务 × 5 级粒度的综合基准
- ⭐⭐⭐⭐ **临床对齐**：评估框架严格对齐临床工作流，从解剖识别到手术干预逐层推进
- ⭐⭐⭐ **人类基线**：引入认证临床医生作为参考标准，提供 MLLM 与人类能力的直接对比
- ⭐⭐⭐ **大规模训练集**：附带 446K EndoVQA-Instruct 指令调优数据集，可直接用于模型训练
- ⭐⭐⭐ **严格质控**：DINO-V2 聚类 + GPT-4o-mini 改写 + 医生审核的三重质控管线

## 局限与展望

1. **评估局限**：仅使用精确匹配（exact match）作为评估指标，对开放式生成评估能力有限
2. **场景偏斜**：手术内镜样本占比近 40%，场景分布不均衡
3. **静态评估**：未考虑视频序列理解，实际内窥镜检查是连续视频流
4. **语言单一**：仅支持英文问答，限制了非英语环境的适用性
5. **缺乏生成任务**：以选择题形式评估为主，未覆盖报告生成、对话等实际临床交互场景

## 总体评价 ⭐⭐⭐⭐

对内窥镜 MLLM 评估来说是重要贡献，全面性和临床对齐性是最大亮点。基准设计系统严谨，23 模型大规模评估提供了有价值的社区参考。但评估方式（选择题+精确匹配）和缺乏视频理解评估是明显局限。适合作为内窥镜 MLLM 研究的标准评测平台。

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

## 相关论文

- [Position: Thematic Analysis of Unstructured Clinical Transcripts with Large Language Models](position_thematic_analysis_of_unstructured_clinical_transcripts_with_large_langu.md)
- [3D-RAD: A Comprehensive 3D Radiology Med-VQA Dataset with Multi-Temporal Analysis and Diverse Diagnostic Tasks](3drad_a_comprehensive_3d_radiology_medvqa_dataset_with_multi.md)
- [Towards Omni-RAG: Comprehensive Retrieval-Augmented Generation for Large Language Models in Medical Applications](../../ACL2025/medical_imaging/omni_rag_medical.md)
- [Mol-LLaMA: Towards General Understanding of Molecules in Large Molecular Language Models](mol-llama_towards_general_understanding_of_molecules_in_large_molecular_language.md)
- [FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models](fgbench_a_dataset_and_benchmark_for_molecular_property_reasoning_at_functional_g.md)

<!-- RELATED:END -->
