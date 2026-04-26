---
title: >-
  [论文解读] Any Information Is Just Worth One Single Screenshot: Unifying Search With Visualized Information Retrieval
description: >-
  [ACL 2025][可视化信息检索] 本文正式定义了可视化信息检索（Vis-IR）范式——将多模态信息统一渲染为截图（Screenshot）进行检索，构建了包含1300万截图的VIRA数据集、UniSE检索模型家族和MVRB基准测试，为统一搜索引擎奠定基础。
tags:
  - ACL 2025
  - 可视化信息检索
  - 截图表示
  - 多模态检索
  - 统一搜索
  - 跨模态嵌入
---

# Any Information Is Just Worth One Single Screenshot: Unifying Search With Visualized Information Retrieval

**会议**: ACL 2025  
**arXiv**: [2502.11431](https://arxiv.org/abs/2502.11431)  
**领域**: 多模态VLM / 信息检索  
**关键词**: 可视化信息检索, 截图表示, 多模态检索, 统一搜索, 跨模态嵌入

## 一句话总结
本文正式定义了可视化信息检索（Vis-IR）范式——将多模态信息统一渲染为截图（Screenshot）进行检索，构建了包含1300万截图的VIRA数据集、UniSE检索模型家族和MVRB基准测试，为统一搜索引擎奠定基础。

## 研究背景与动机

**领域现状**：信息检索领域在文本检索和多模态检索上取得了巨大进展。随着视觉语言模型的发展，越来越多的场景中信息以视觉形式呈现——网页截图包含文字、图片、表格和图表的混合内容。用户开始使用"圈选搜索"（如Google Circle to Search）等新交互方式。

**现有痛点**：（1）现有检索系统要么处理纯文本，要么处理图像，难以统一处理包含混合模态的复杂文档。（2）将网页/PDF渲染为截图后的检索（如ColPali）只是初步尝试，缺乏系统化的问题定义、大规模数据集和全面的基准测试。（3）截图中的文字、图表、布局等信息需要被协同理解，而不是分别处理。

**核心矛盾**：不同类型的信息（文字、图片、表格、代码）有各自的最佳表示方式，但用户在真实搜索中经常面对混合内容。统一用视觉形式（即截图）来表示所有信息，虽然概念上简洁，但带来了如何让模型理解截图中丰富语义的技术挑战。

**本文目标**：正式定义Vis-IR问题，构建完整的数据-模型-评估生态，推动这一新范式的发展。

**切入角度**：一切信息都可以渲染为一张截图——文本是截图，网页是截图，论文是截图，商品页面也是截图。将"截图"作为统一的信息载体，所有检索任务都变成了截图之间或查询与截图之间的匹配。

**核心 idea**：通过大规模截图-标注数据集训练通用截图嵌入模型，实现"截图搜截图"、"文本搜截图"、"条件搜索"等多种检索模式的统一支持。

## 方法详解

### 整体框架
三大贡献：（1）VIRA：包含1300万截图和2000万数据样本的大规模训练数据集；（2）UniSE：基于CLIP和MLLM两种架构的通用截图嵌入模型；（3）MVRB：覆盖多种任务和域的可视化检索基准。

### 关键设计

1. **VIRA数据集构建**:

    - 功能：提供大规模、多样化的截图检索训练数据
    - 核心思路：从七类来源（新闻网站、电商平台、Wikipedia、GitHub、ArXiv论文、PDF文档、图表）收集1300万截图。为每张截图生成细粒度caption（通过元数据提取或OCR工具），然后用LLM生成两种问答数据：q2s（查询→截图）元组和sq2s（截图+条件查询→目标截图）三元组。还通过文本/视觉相似性挖掘hard negatives
    - 设计动机：截图来源的多样性确保模型泛化能力，caption的精细度确保训练信号质量，hard negatives提升检索模型的区分能力

2. **UniSE模型家族**:

    - 功能：将截图和文本查询映射到统一的嵌入空间
    - 核心思路：提供两种架构选择。UniSE-CLIP基于CLIP，截图走视觉编码器、文本走文本编码器，组合查询通过嵌入相加实现，效率高但表达力有限。UniSE-MLLM基于Qwen2-VL-2B，将截图视觉token和查询文本token通过多模态LLM编码，使用[EOS] token的输出作为嵌入，表达力更强但计算成本更高
    - 设计动机：两种架构面向不同的使用场景——CLIP版适合大规模在线检索，MLLM版适合需要深度理解的场景

3. **两阶段训练流程**:

    - 功能：分阶段逐步提升模型能力
    - 核心思路：第一阶段用截图-caption对进行对比学习预训练，让模型理解截图的细粒度语义。第二阶段在q2s和sq2s数据上微调，使模型学习检索相关的匹配能力。两阶段都使用双向对比损失 $\mathcal{L} = \mathcal{L}_{con}(e_s, e_c) + \mathcal{L}_{con}(e_c, e_s)$
    - 设计动机：先"看懂"截图，再学习"检索匹配"，渐进式训练比直接在检索数据上训练更稳定

### 损失函数 / 训练策略
双向InfoNCE对比损失，温度参数 $\tau$ 可学习。使用batch内负样本和挖掘的hard negatives。训练基于DeepSpeed ZeRO-2，UniSE-CLIP在64个GPU上训练，UniSE-MLLM在32个GPU上训练。

## 实验关键数据

### 主实验

| 模型 | MVRB平均 (nDCG@10) | q2s任务 | s2s任务 | sq2s任务 |
|------|-------------------|---------|---------|----------|
| CLIP-Large (zero-shot) | 38.2 | 42.1 | 31.5 | 35.8 |
| ColPali | 45.6 | 51.3 | 38.2 | 41.7 |
| E5-V | 43.1 | 48.5 | 36.8 | 39.4 |
| UniSE-CLIP | 58.7 | 63.2 | 52.4 | 55.1 |
| **UniSE-MLLM** | **62.3** | **67.5** | **55.8** | **58.9** |

### 消融实验

| 配置 | MVRB平均 | 说明 |
|------|---------|------|
| UniSE-MLLM (Full) | 62.3 | 完整模型 |
| 仅Stage 1训练 | 52.8 | 预训练但未在检索数据上微调 |
| 无Hard Negatives | 58.5 | 缺少困难样本区分能力下降 |
| 仅新闻域数据训练 | 54.2 | 单域训练泛化性差 |
| 无sq2s数据 | 59.1 | 条件检索场景降几个点 |
| CLIP架构 vs MLLM | 58.7 vs 62.3 | MLLM更强但速度慢3× |

### 关键发现
- 现有多模态检索器在Vis-IR任务上表现严重不足（CLIP仅38.2%），说明截图检索是一个远未解决的问题
- UniSE-MLLM比UniSE-CLIP高3.6个点，在条件检索（sq2s）上优势更大，因为组合查询需要深层语义理解
- 数据多样性至关重要——仅在单个域训练的模型泛化性差
- Hard negatives贡献约4个点的提升，对检索模型至关重要

## 亮点与洞察
- "一切信息都是截图"的统一范式非常优雅，有望成为下一代搜索引擎的核心思路。这与Google Circle to Search的产品方向完美对齐。
- VIRA数据集的构建方法值得学习：自动化的caption生成 + LLM辅助的QA数据构造 + hard negative挖掘，形成了可扩展的数据飞轮。
- 两种架构选择（CLIP vs MLLM）的设计体现了工程思维——让用户根据延迟和准确度需求自行选择。

## 局限与展望
- 截图作为统一表示会丢失文本文档中的精确文字信息，OCR质量成为瓶颈
- MVRB基准目前以英语为主，多语言截图检索是重要方向
- 视频内容尚未被纳入Vis-IR框架
- 截图分辨率和长文档的多页截图表示是实际部署中的挑战

## 相关工作与启发
- **vs ColPali**: ColPali仅处理文档截图，Vis-IR将范围扩展到所有类型的截图（网页、商品、图表等）
- **vs DSE (Document Screenshot Embedding)**: DSE聚焦Wiki文档，VIRA的数据多样性和规模远超前作
- **vs CLIP/SigLIP**: 通用视觉-语言模型在截图这种信息密集的视觉内容上表现欠佳，需要专门训练

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 正式定义Vis-IR范式，构建完整的数据-模型-评估体系，具有开创性
- 实验充分度: ⭐⭐⭐⭐⭐ 1300万数据、两种模型架构、全面的基准测试
- 写作质量: ⭐⭐⭐⭐ 结构清晰，三大贡献层次分明
- 价值: ⭐⭐⭐⭐⭐ 为下一代搜索引擎提供了技术基础，产业应用前景广阔

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] AIR-Bench: Automated Heterogeneous Information Retrieval Benchmark](air-bench_automated_heterogeneous_information_retrieval_benchmark.md)
- [\[ACL 2025\] CoIR: A Comprehensive Benchmark for Code Information Retrieval Models](coir_a_comprehensive_benchmark_for_code_information_retrieval_models.md)
- [\[ACL 2025\] Atomic LLM: A Fine-Grained Information Retrieval Evaluation Benchmark for Language Models](atomic_llm_a_fine-grained_information_retrieval_evaluation_benchmark_for_languag.md)
- [\[ACL 2025\] Logical Consistency is Vital: Neural-Symbolic Information Retrieval for Negative-Constraint Queries](logical_consistency_is_vital_neural-symbolic_information_retrieval_for_negative-.md)
- [\[ACL 2025\] HoH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on Retrieval-Augmented Generation](hoh_a_dynamic_benchmark_for_evaluating_the_impact_of_outdated_information_on_ret.md)

<!-- RELATED:END -->
