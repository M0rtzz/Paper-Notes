---
title: >-
  [论文解读] Mind the (Data) Gap: Evaluating Vision Systems in Small Data Applications
description: >-
  [NeurIPS 2025][医学图像][小数据评估] 在 NeWT 生态分类基准上系统比较了 MLLMs（如 Gemini、Qwen2.5-VL）和视觉编码器+SVM 在"小数据区间"（10~1000 标注样本）的表现，发现 MLLMs 在 10-30 个样本后即触顶，而视觉方法持续近对数增长，呼吁社区重视小数据评估。
tags:
  - NeurIPS 2025
  - 医学图像
  - 小数据评估
  - 多模态大语言模型
  - 视觉编码器
  - SVM
  - 预训练策略
---

# Mind the (Data) Gap: Evaluating Vision Systems in Small Data Applications

**会议**: NeurIPS 2025  
**arXiv**: [2504.06486](https://arxiv.org/abs/2504.06486)  
**代码**: 暂无  
**领域**: 医学图像 / 计算机视觉评估  
**关键词**: 小数据评估, 多模态大语言模型, 视觉编码器, SVM, 预训练策略

## 一句话总结

在 NeWT 生态分类基准上系统比较了 MLLMs（如 Gemini、Qwen2.5-VL）和视觉编码器+SVM 在"小数据区间"（10~1000 标注样本）的表现，发现 MLLMs 在 10-30 个样本后即触顶，而视觉方法持续近对数增长，呼吁社区重视小数据评估。

## 研究背景与动机

**领域现状**：当前 AI 研究的评估体系严重倾斜——要么是零样本/少样本（0~5 例），要么是大规模数据集（>10K 例）。作者通过手动收集近年视觉和语言研究中使用的评估任务（覆盖 CLIP、DINOv2、Gemini、Phi-4 等方法），发现在 10 到 1000 个训练样本这一区间的评估任务几乎为零，形成了明显的"数据缺口"。

**现有痛点**：这个缺口恰好对应着大量现实应用——生态监测需要生物学家标注物种、医学诊断依赖专家标注、工业质检需要领域知识。这些场景通常只能获得数十到数千个标注样本，既不属于零样本也不是大规模数据。当前为零样本优化的 MLLMs 是否在这些场景中真的好用？没有人系统验证过。

**核心矛盾**：MLLMs 的 few-shot prompting 机制本质上是将标注样本塞入上下文窗口作为示例，但上下文注意力的信息利用效率与特征空间上的显式分类器（如 SVM）有根本差异。随着可用标注样本从几个增长到几百个，两种范式的缩放行为可能完全不同。

**本文目标** 首次在小数据区间（10~1000 标注样本）系统对比 MLLMs 和视觉编码器方法，揭示它们各自的缩放特性。

**切入角度**：选择 NeWT（Natural World Tasks）作为测试平台——该基准包含 164 个生态二分类任务，每个任务仅有 200~400 个标注样本，天然处于小数据区间。

**核心 idea**：用 NeWT 基准覆盖从 0 到全量的训练子集，对比 MLLMs 和视觉+SVM 在小数据区间的缩放表现。

## 方法详解

### 整体框架

本文是一项实证评估研究而非提出新模型。实验设计的核心是：(1) 在 NeWT 的 164 个二分类任务上，(2) 按近对数间隔设定 0/1/3/10/30/100/300/全量 八个训练规模，(3) 对比 MLLMs（通过 few-shot prompting 利用标注样本）和视觉编码器（提取冻结特征后训练 SVM）两类方法的表现随数据量的变化趋势。

### 关键设计

1. **近对数间隔的数据规模采样**:

    - 功能：构造 0, 1, 3, 10, 30, 100, 300, 全量 八个训练子集
    - 核心思路：均匀采样标注样本并保证每个类别至少一个样本。在每个规模下独立评估所有方法
    - 设计动机：对数间隔能覆盖多个数量级，区分"几个样本"和"几百个样本"之间的行为变化，这正是现有评估忽略的范围

2. **MLLMs 评估协议**:

    - 功能：统一评测 Gemini Flash 2.0、Gemini Flash 1.5 8B、Qwen2-VL 7B、Qwen2.5-VL 72B
    - 核心思路：将标注样本作为 few-shot 示例放入提示中，模型响应通过确定性正则表达式解析为分类结果。若响应中包含多个物种名，取第一个提到的
    - 设计动机：这是 MLLMs 在小数据下利用标注信息的唯一标准方式——将样本"展示"给模型

3. **视觉编码器+SVM 方案**:

    - 功能：对比 DINOv2（ViT-g/14）、CLIP（ViT-L/14）、SigLIP（ViT-SO400M/14）等视觉编码器
    - 核心思路：冻结预训练编码器提取图像特征，用 scikit-learn 的交叉验证网格搜索调优 SVM 超参数进行二分类
    - 设计动机：SVM 对小样本场景天然适合，且与 NeWT 原始评估方法一致，保证对比的公平性

### 损失函数 / 训练策略

MLLMs 无需训练，视觉编码器冻结，仅训练 SVM 分类超平面。所有评估使用 1000 次 bootstrap 重采样计算 95% 置信区间。

## 实验关键数据

### 主实验：数据缩放行为

| 方法类别 | 3 个样本 | 10 个样本 | 30 个样本 | 100 个样本 | 300 个样本 | 趋势 |
|---------|---------|----------|----------|----------|----------|------|
| Gemini Flash 2.0 | ~67% | ~68% | ~70% | ~70% | ~70% | 10-30样本后平台期 |
| Qwen2.5-VL 72B | ~64% | ~65% | ~68% | ~68% | ~68% | 类似平台期 |
| DINOv2 ViT-g+SVM | ~55% | ~63% | ~71% | ~77% | ~81% | 持续近对数增长 |
| SigLIP SO400M+SVM | ~53% | ~60% | ~70% | ~76% | ~80% | 持续近对数增长 |

关键交叉点：在 ~10 个样本时，DINOv2+SVM 超越所有 MLLMs，此后差距持续扩大。

### 消融实验：模型规模 vs 数据规模

| 配置对比 | 结论 |
|---------|------|
| SigLIP 从 45 GFLOPs 扩到 700+ GFLOPs（模型增大 10×） | 准确率提升有限 |
| 标注样本从 10 增到 100（数据增加 10×） | 准确率提升显著超过模型扩大 |
| DINOv2 vs CLIP/SigLIP 在 Species/Age 任务 | DINOv2 显著更优（纯视觉预训练擅长细粒度判别） |
| CLIP/SigLIP vs DINOv2 在 Gestalt/Behavior 任务 | 语言监督预训练显著更优（需要语义推理能力） |

### 预训练策略对比（30 个样本，ViT-L）

| 任务簇 | DINOv2 | CLIP | SigLIP | 哪种预训练更优？ |
|--------|--------|------|--------|--------------|
| Species（物种识别） | 最高 | 中等 | 中等 | 纯视觉 |
| Age（年龄判断） | 最高 | 较低 | 较低 | 纯视觉 |
| Gestalt（整体感知） | 较低 | 最高 | 高 | 语言监督 |
| Behavior（行为识别） | 较低 | 高 | 最高 | 语言监督 |
| Context（上下文） | 较低 | 高 | 高 | 语言监督 |
| Counting（计数） | 接近 | 接近 | 接近 | 无明显差异 |
| Health（健康状态） | 接近 | 接近 | 接近 | 无明显差异 |

### 关键发现

- MLLMs 通过 few-shot prompting 在 10-30 个样本后即达到性能天花板，无法从更多数据中持续受益
- 视觉编码器+SVM 在 10~300 样本区间展现近对数的持续增长，且没有饱和迹象
- 10 倍数据量增长带来的准确率提升稳定优于 10 倍模型计算量增长——挑战"越大越好"的主流范式
- DINOv2 的自监督预训练在细粒度视觉判别上有独特优势，CLIP/SigLIP 的语言监督在语义推理任务上领先，这种差异在所有训练集大小下保持一致

## 亮点与洞察

- **揭示评估盲区**：通过手动统计近年论文的训练集大小，清晰证明了 10~1000 样本区间的"数据缺口"，这种数据驱动的论证方式比单纯的观点陈述更有说服力
- **反直觉发现的实用价值**：对于实际部署场景，这个结论意味着——当你有几百个标注样本时，用一个中等大小的视觉编码器+SVM 可能比调用最先进的 MLLM API 效果更好，且成本更低
- **预训练范式的互补性**：纯视觉预训练和语言监督预训练在不同任务类型上的系统性差异，为实际模型选择提供了实证指导——需要细粒度形态判别选 DINOv2，需要语义理解选 CLIP/SigLIP

## 局限与展望

- 仅使用 NeWT 生态学基准验证，虽然作者声称结论可推广到医学/工业等领域，但缺乏直接实验
- MLLMs 仅通过 few-shot prompting 利用标注数据，未测试参数高效微调（如 LoRA）是否能改变缩放行为
- 视觉方法固定使用冻结编码器+SVM，未探索线性探测、k-NN 或轻量微调等替代方案
- 未考虑数据增强对小数据方法的影响，这在实际应用中通常是标准做法
- 所有视觉编码器均为通用域预训练，未测试领域专用基础模型（如 BioCLIP）的表现

## 相关工作与启发

- **vs BioCLIP**：BioCLIP 是为生物领域定制的视觉基础模型，作者之前的工作。本文没有使用 BioCLIP 而是用通用编码器，可能是为了保证结论的通用性
- **vs Many-shot ICL (Jiang et al.)**: 该工作在多模态基础模型中探索了 many-shot in-context learning，但未系统覆盖 10-1000 样本区间
- **方法论迁移**：这套评估框架可以直接应用到医学影像（如皮肤病分类、病理切片）等小数据领域，用以指导方法选择

## 评分

- 新颖性: ⭐⭐⭐ 评估性工作无新模型，但视角新颖、问题定义精准
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型×多规模×多任务类型的全面交叉对比，统计分析严谨
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，图表信息密度高，Fig.1 的双图设计一目了然
- 价值: ⭐⭐⭐⭐ 对实际 AI 部署中的方法选择有直接指导意义，倡导的评估实践值得推广

<!-- RELATED:START -->

## 相关论文

- [Toward a Vision-Language Foundation Model for Medical Data: Multimodal Dataset and Benchmarks for Vietnamese PET/CT Report Generation](toward_a_vision-language_foundation_model_for_medical_data_multimodal_dataset_an.md)
- [Steering Generative Models with Experimental Data for Protein Fitness Optimization](steering_generative_models_with_experimental_data_for_protein_fitness_optimizati.md)
- [EDBench: Large-Scale Electron Density Data for Molecular Modeling](edbench_large-scale_electron_density_data_for_molecular_modeling.md)
- [Multiscale Guidance of Protein Structure Prediction with Heterogeneous Cryo-EM Data](multiscale_guidance_of_protein_structure_prediction_with_heterogeneous_cryo-em_d.md)
- [CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays](cxreasonbench_a_benchmark_for_evaluating_structured_diagnostic_reasoning_in_ches.md)

<!-- RELATED:END -->
