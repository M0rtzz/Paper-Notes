---
title: >-
  [论文解读] SMMILE: An Expert-Driven Benchmark for Multimodal Medical In-Context Learning
description: >-
  [NeurIPS 2025][医学图像][多模态上下文学习] 提出 SMMILE——首个由 11 位医学专家驱动的多模态医学上下文学习（ICL）基准，包含 111 道问题（517 个图文问答三元组）覆盖 6 个医学专科和 13 种成像模态…
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "多模态上下文学习"
  - "医学基准测试"
  - "多模态大语言模型"
  - "上下文示例质量"
  - "近因偏差"
---

# SMMILE: An Expert-Driven Benchmark for Multimodal Medical In-Context Learning

**会议**: NeurIPS 2025  
**arXiv**: [2506.21355](https://arxiv.org/abs/2506.21355)  
**代码**: [项目主页](https://smmile-benchmark.github.io)  
**领域**: 医学图像 / 多模态学习  
**关键词**: 多模态上下文学习, 医学基准测试, 多模态大语言模型, 上下文示例质量, 近因偏差

## 一句话总结

提出 SMMILE——首个由 11 位医学专家驱动的多模态医学上下文学习（ICL）基准，包含 111 道问题（517 个图文问答三元组）覆盖 6 个医学专科和 13 种成像模态，系统性揭示了当前 MLLM 在医学多模态 ICL 上的严重不足以及上下文示例质量和顺序对性能的关键影响。

## 研究背景与动机

上下文学习（ICL）是大语言模型的核心能力——在推理时通过上下文中的少量示例即可适应新任务，无需参数更新。这种能力在医学领域尤为重要：临床医生日常工作就是基于有限的先例经验来处理专业化任务。

然而，当前存在三个关键研究空白：

**评测空白**：虽然多模态 ICL 在通用领域已有初步研究，但医学领域尚无系统的多模态 ICL 评测基准

**示例质量问题**：现有的医学 few-shot 评估通常随机选取示例，而非精心设计的任务演示，这可能部分解释了为什么 ICL 改善有限

**能力认知不足**：MLLM 在医学 VQA 上取得了进展，但其从多模态上下文中学习新任务的能力几乎未知

SMMILE 的核心切入点是：由临床专家精心设计每个问题的上下文示例（而非随机检索），从而能准确评估 MLLM 的真正 ICL 能力。如果即便是专家精心设计的示例也不能有效帮助模型学习，那么问题确实出在模型本身。

## 方法详解

### 整体框架

SMMILE 不是一个模型方法论文，而是一个评测基准（benchmark），其核心贡献在于数据集构建和评测分析。

### 关键设计

#### 1. 专家驱动的数据收集流程

- **专家招募**：21 位临床专家中 11 位成功提交数据，包括 9 位医生（平均 6.4 年临床经验）和 2 位医学生
- **Web 界面引导**：设计了分步式工作流——(1) 阅读详细指南 → (2) 初始化问题 → (3) 创建面板（可添加/删除/重排上下文示例和查询）→ (4) 提交验证
- **质量控制**：三步手动质量审查——两位作者独立检查 → 分类标注 → 拼写语法校正（修改了 15 处语法和 6 处拼写）

**数据规模**：111 道问题，每题平均 3.65 个上下文示例（范围 2-19），共 517 个图文问答三元组，覆盖 6 个医学专科（放射、病理等）和 13 种成像模态（X 光、CT、MRI 等）。

#### 2. 评测设计

**两种评测任务**：
- **开放式生成**：MLLM 接收查询图文并生成自由文本回答
- **封闭式选择**：从上下文示例集中选择答案

**三种评价指标**：
- Exact Match (EM)：完全匹配答案
- LLM-as-a-Judge：使用 Llama3.3 70B 评判正确性
- 人类专家评估：5 位临床专家独立评判，评估者间一致率 ≥98.2%

**SMMILE++**：通过排列上下文示例顺序生成的增强数据集，包含 1038 道问题（每题最多 24 种排列），用于研究示例顺序的影响。

#### 3. 上下文示例分析框架

- **质量分析**：构建 Random-Noise 和 Targeted-Noise 两个扰动版本，评估加入不相关示例的影响
- **顺序分析**：控制相关示例在列表中的位置（首位 vs 末位），评估近因偏差

### 评测协议

评测 15 个 MLLM，包括：
- 开源通用模型：LLaVA 系列、Qwen2.5-VL 系列、Llama-3.2-Vision-90B
- 开源医学模型：LLaVA-Med、MedGemma、MedVLM-R1  
- 闭源模型：GPT-4o、Claude 3.7 Sonnet
- 基线：Random、Majority、Text-Only (Llama3.3 70B)

所有模型统一最大生成 512 tokens，使用 bootstrap 重采样（1000 次迭代）估计采样变异性。

## 实验关键数据

### 主实验（SMMILE 上 15 个模型）

| 模型 | 0-shot (Judge)↑ | ICL (Judge)↑ | ICL (EM)↑ | ICL (MCQA)↑ |
|------|----------------|-------------|----------|------------|
| GPT-4o | 32.56 | **49.88** | 8.94 | **58.85** |
| Claude 3.7 Sonnet | **37.18** | 36.17 | 2.63 | 42.01 |
| Qwen2.5-VL-72B | 29.90 | 42.59 | 15.71 | 54.71 |
| Qwen2.5-VL-32B | 25.27 | 41.79 | **31.84** | 49.97 |
| Llama-3.2-Vision-90B | 31.84 | 40.66 | 30.53 | 30.30 |
| MedGemma-4B | 27.73 | 36.86 | 12.14 | 40.67 |
| LLaVA-Med-7B | 21.65 | 10.19 | 0.00 | 0.00 |
| Random 基线 | — | 27.86 | 23.16 | 36.30 |

### 上下文示例质量分析

| 模型 | SMMILE (正常)↑ | Random-Noise↑ | Targeted-Noise↑ | 性能降幅 |
|------|---------------|--------------|-----------------|---------|
| Qwen2.5-VL-32B | **41.79** | 39.60 | 39.10 | -5.2% / -6.4% |
| Qwen2.5-VL-3B | **33.58** | 30.40 | 30.37 | -9.5% / -9.6% |
| LLaVA-Med-7B | **10.19** | 4.88 | 1.88 | -52.1% / -81.6% |
| 平均 | **24.92** | 22.65 | 22.55 | -9.1% / -9.5% |

仅加入一个不相关示例即可导致平均 9.5% 的性能下降。

### 关键发现

1. **ICL 改善有限且高度不均**：15 个模型中 7 个 ICL 表现甚至不如 Random 基线（随机从示例中选答案），ICL 平均仅提升 8%
2. **医学专用模型未见优势**：LLaVA-Med 在 ICL 设置下性能反而暴跌（21.65% → 10.19%），MedGemma 与同规模通用模型差异不大
3. **近因偏差严重**：将最相关示例放在列表末尾可带来最高 71% 的性能提升，放在开头反而下降最多 47%
4. **数值推理全面失败**：所有模型在数值型答案上正确率为 0%
5. **自动指标偏乐观**：LLM-as-a-Judge 一致性在 ICL 设置下与专家评分仅中等相关（r=0.72），且倾向高估
6. **MRI 和插图模态表现为零**：所有模型在 MRI 和 illustration 模态上完全失败

## 亮点与洞察

- **专家驱动而非规模驱动**：11 位临床专家精心设计每道题的上下文示例，确保示例确实是有效的任务演示，使评测结果更能反映模型真实 ICL 能力
- **揭示假象**：之前 few-shot 评测中 ICL 改善不大可能不是因为 ICL 无用，而是因为随机选取的示例本身就无效；但即便专家精心设计的示例也未能显著帮助大多数模型
- **近因偏差的发现**：对实际部署 MLLM 有直接指导意义——简单地将最相关示例放在末尾即可显著提升性能
- **自动/人工评价差异**：提醒社区不能完全依赖 LLM-as-a-Judge，特别是在 ICL 场景下

## 局限与展望

- 数据集规模相对较小（111 道题），可能不足以覆盖所有医学场景
- 仅包含图像模态，未来可扩展到视频、音频等医学数据
- 专家数量有限（11 位），可能存在专科覆盖不均
- SMMILE++ 仅通过排列顺序增强，未引入新的医学内容
- 缺少针对发现的具体改进方法

## 相关工作与启发

- Flamingo 等多模态 ICL 模型在通用领域的成功激发了医学领域的探索
- SMMILE 的近因偏差发现与 NLP 中的 "lost-in-the-middle" 现象相呼应
- 人工专家评估与自动指标的差异表明，医学 AI 评测需要更多领域专家参与

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个医学多模态 ICL 基准，发现了近因偏差和示例质量敏感性
- 实验充分度: ⭐⭐⭐⭐⭐ 15 个模型 + 人工/自动双重评估 + 多维细粒度分析 + 扰动实验
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，数据可视化丰富，分析深入
- 价值: ⭐⭐⭐⭐ 对 MLLM 社区有重要警示作用，但本身不提供解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] CSTRL: Context-Driven Sequential Transfer Learning for Abstractive Radiology Report Summarization](../../ACL2025/medical_imaging/cstrl_context-driven_sequential_transfer_learning_for_abstractive_radiology_repo.md)
- [\[NeurIPS 2025\] Meta-Learning an In-Context Transformer Model of Human Higher Visual Cortex](meta-learning_an_in-context_transformer_model_of_human_higher_visual_cortex.md)
- [\[CVPR 2025\] Show and Segment: Universal Medical Image Segmentation via In-Context Learning](../../CVPR2025/medical_imaging/show_and_segment_universal_medical_image_segmentation_via_in-context_learning.md)
- [\[ACL 2026\] PCoA: A New Benchmark for Medical Aspect-Based Summarization With Phrase-Level Context Attribution](../../ACL2026/medical_imaging/pcoa_a_new_benchmark_for_medical_aspect-based_summarization_with_phrase-level_co.md)
- [\[ECCV 2024\] Improving Medical Multi-modal Contrastive Learning with Expert Annotations](../../ECCV2024/medical_imaging/improving_medical_multi-modal_contrastive_learning_with_expert_annotations.md)

</div>

<!-- RELATED:END -->
