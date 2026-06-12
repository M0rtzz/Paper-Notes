---
title: >-
  [论文解读] ChartCap: Mitigating Hallucination of Dense Chart Captioning
description: >-
  [ICCV 2025][幻觉检测][图表理解] 构建了包含56.5万张真实图表-描述对的大规模数据集ChartCap，通过类型特定的描述模式排除无关信息、强调结构与关键洞察，并提出无参考的Visual Consistency Score评估指标，有效减少VLM在图表描述中的幻觉问题。
tags:
  - "ICCV 2025"
  - "幻觉检测"
  - "图表理解"
  - "视觉语言模型"
  - "幻觉缓解"
  - "图表描述"
  - "数据集构建"
---

# ChartCap: Mitigating Hallucination of Dense Chart Captioning

**会议**: ICCV 2025  
**arXiv**: [2508.03164](https://arxiv.org/abs/2508.03164)  
**代码**: [https://junyoung-00.github.io/ChartCap/](https://junyoung-00.github.io/ChartCap/)  
**领域**: 幻觉检测  
**关键词**: 图表理解, 视觉语言模型, 幻觉缓解, 图表描述, 数据集构建

## 一句话总结

构建了包含56.5万张真实图表-描述对的大规模数据集ChartCap，通过类型特定的描述模式排除无关信息、强调结构与关键洞察，并提出无参考的Visual Consistency Score评估指标，有效减少VLM在图表描述中的幻觉问题。

## 研究背景与动机

图表描述（Chart Captioning）是评估VLM理解图表能力的核心任务。理想的描述应满足两个条件：（1）不包含从图表中无法推断的错误信息；（2）包含图表结构描述和关键洞察（如最大最小值、趋势模式等）。

然而，现有真实图表数据集存在两个严重问题：

**无关信息混入**：图表通常嵌入在论文/文档中，原始标题基于上下文撰写（如引用参数名、描述缺失的误差条等），这些信息无法仅从图表图片推断。这给模型造成了"不可能任务"——期望模型预测图中不存在的信息，最终导致幻觉

**结构和洞察信息不足**：原始标题常省略关键信息（如数据趋势、极值），因为人类作者假设读者能自行从图表推断这些内容。缺乏类型特定的描述模式使得不同图表类型的关键信息被忽略

## 方法详解

### 整体框架

ChartCap的构建包含四个阶段的自动化流水线和一个基于循环一致性的人工验证环节：
1. 过滤非图表图像 → 2. 分类图表类型并提取标题 → 3. 提取类型特定的结构信息和关键洞察 → 4. 转化为连贯的句子级描述

### 关键设计

#### 1. 类型特定的描述模式（Caption Schema）
- **功能**：为9种图表类型（折线图、柱状图、饼图、直方图、散点图、面积图、气泡图、等值线图、树状图）定义结构描述和关键洞察的标准化模式
- **核心思路**：基于数据可视化领域的研究——Munzner的《Visualization Analysis and Design》提供设计框架，VLAT测试蓝图确定每种图表类型的认知任务。例如，散点图强调聚类和分布，折线图强调时间趋势和变化
- **设计动机**：通过标准化减少描述质量的模糊性，使模型学习到每种图表类型应关注的关键信息

#### 2. 四阶段数据生成流水线
- **阶段1-过滤**：使用InternVL2.5-8B过滤非数据驱动图表（如示意图、插图），从310万图像中保留120万张，精确度100%
- **阶段2-分类与提取**：使用GPT-4o获取图表类型和标题，保留属于9种预定义类型的57.7万张图表，准确率99%
- **阶段3-信息提取**：GPT-4o处理粗粒度任务（整体趋势），Claude 3.5 Sonnet处理细粒度任务（精确数值）。准确率94%
- **阶段4-生成描述**：GPT-4o-mini将半结构化数据转换为句子级描述，准确率100%

#### 3. 基于循环一致性的人工验证
- **功能**：利用人类视觉感知的毫秒级速度，通过对比原始图表和从描述重建的图表来验证质量
- **核心思路**：使用Claude 3.5 Sonnet将描述转换为Python代码重建图表，人工对比原图和重建图。相比直接人工比对图-文对，速度提升约24倍，F1保持95%
- **设计动机**：人类比较两张图的速度远快于仔细阅读长描述文本。通过视觉对比同时保障准确性和信息完整性

#### 4. Visual Consistency Score (VCS)
- **功能**：提出无参考的图表描述质量评估指标
- **核心思路**：将描述 $C_i$ 通过LLM转换为Matplotlib代码 $G_i$，执行生成重建图表 $\hat{I}_i$，使用视觉编码器计算与原图 $I_i$ 的余弦相似度：

$$\text{VCS} = \frac{1}{N} \sum_{i=1}^{N} \text{Sim}(I_i, \hat{I}_i)$$

辅以OCRScore衡量文本元素保留度：

$$\text{OCRScore} = 2 \cdot \frac{P \times R}{P + R}$$

其中 $P = \frac{\sum|\mathcal{T}_i \cap \hat{\mathcal{T}}_i|}{\sum|\hat{\mathcal{T}}_i|}$，$R = \frac{\sum|\mathcal{T}_i \cap \hat{\mathcal{T}}_i|}{\sum|\mathcal{T}_i|}$

- **设计动机**：现有指标（BLEU、BERTScore）无法捕捉深层语义质量且高度依赖参考描述质量；图表的独特特性使其可从代码中间表示确定性地生成，使"反向验证"成为可能

### 损失函数 / 训练策略

使用LoRA微调InternVL2.5-8B和Phi3.5-vision-4B，训练集509K样本。采用标准的监督微调范式，输入为图表图像 + 指令"Please provide a detailed caption for the chart."

## 实验关键数据

### 主实验

ChartCap测试集上各模型表现：

| 模型 | sacreBLEU | ROUGE-L | BERTScore | VCS (Large) | OCRScore |
|------|-----------|---------|-----------|-------------|----------|
| Claude 3.5 Sonnet | 5.35 | 0.2265 | 0.6606 | 0.8834 | 0.4868 |
| InternVL2.5-78B | 8.15 | 0.2510 | 0.6642 | 0.8841 | 0.4677 |
| Phi3.5-4B | 8.41 | 0.2466 | 0.6626 | 0.8433 | 0.4875 |
| **Phi3.5-4B_ChartCap** | **23.82** | **0.3900** | **0.7427** | **0.8933** | **0.5179** |
| **InternVL2.5-8B_ChartCap** | **19.47** | **0.3393** | **0.7238** | **0.8913** | **0.5089** |

### 消融实验 / 数据集对比

各真实图表数据集的VCS和信息量对比：

| 数据集 | VCS (Large) | VCS (So400M) | OCRScore | 平均词数 |
|--------|-------------|--------------|----------|---------|
| ArxivCap | 0.7561 | 0.7421 | 0.1781 | 43.7 |
| ChartSumm | 0.8940 | 0.9008 | 0.2635 | 45.4 |
| Chart-to-Text | 0.6925 | 0.7089 | 0.0951 | 62.2 |
| **ChartCap** | **0.8983** | **0.9089** | **0.5424** | **231.1** |

在VisText测试集上的零样本迁移：

| 模型 | VCS (Large) | VCS (So400M) | OCRScore |
|------|-------------|--------------|----------|
| 人工标注 | 0.9172 | 0.9151 | 0.3407 |
| Claude 3.5 Sonnet | 0.8970 | 0.9008 | 0.3286 |
| **Phi3.5-4B_ChartCap** | **0.9443** | **0.9382** | 0.3414 |

### 关键发现

1. 仅4B参数的Phi3.5微调模型在所有指标上超越78B的InternVL2.5和商用Claude 3.5 Sonnet，说明**高质量数据比模型规模更重要**
2. ChartCap微调模型生成的描述甚至被人类评估者判定优于人写的标注描述（VisText数据集上）
3. 使用原始描述（包含无关信息）或ChartSumm微调的模型性能反而下降，证实无关信息确实导致幻觉
4. VCS与人类判断的一致性显著高于BERTScore等传统指标

## 亮点与洞察

- **根因分析到位**：明确指出幻觉的来源不是模型能力不足，而是训练数据中包含无法从图表推断的无关信息
- **循环一致性验证**：巧妙利用"描述→代码→重建图表→视觉对比"的循环，将耗时的文本审核转化为快速的视觉比较
- **VCS指标**：利用图表可从代码生成的独特性质，首次实现真正的无参考图表描述评估
- **反直觉发现**：小模型+好数据 > 大模型，ChartCap微调的4B模型超越人工标注

## 局限与展望

- 描述模式仅覆盖9种图表类型（基于VLAT），未包括热力图、桑基图等
- 依赖商用API（GPT-4o、Claude 3.5 Sonnet）构建数据集，成本较高且不可完全复现
- VCS评估本身依赖LLM生成代码的能力，对代码生成失败的情况需要重试机制
- 未探索在ChartQA等问答任务上的迁移效果

## 相关工作与启发

- **ArxivCap**和**SciCap**等从论文收集的数据集验证了"原始标题包含无关信息"的问题
- **CLIPScore**在长描述和精确图表理解场景下不可靠，VCS提供了更优解
- 该工作的方法论——"定义schema → 自动生成 → 循环一致性验证"——可推广到其他结构化视觉内容的描述任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 从数据质量角度解决幻觉问题，VCS指标设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 参考指标+人工评估+VCS三维度全面验证，跨数据集泛化测试
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，数据集构建过程详尽透明
- 价值: ⭐⭐⭐⭐ 56.5万高质量数据集和VCS指标对图表理解社区有重要推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Mitigating Object Hallucinations via Sentence-Level Early Intervention](mitigating_object_hallucinations_via_sentence-level_early_intervention.md)
- [\[ACL 2025\] Monitoring Decoding: Mitigating Hallucination via Evaluating the Factuality of Partial Response during Generation](../../ACL2025/hallucination/monitoring_decoding_mitigating_hallucination_via_evaluating_the_factuality_of_pa.md)
- [\[NeurIPS 2025\] Mitigating Hallucination Through Theory-Consistent Symmetric Multimodal Preference Optimization](../../NeurIPS2025/hallucination/mitigating_hallucination_through_theory-consistent_symmetric_multimodal_preferen.md)
- [\[ACL 2025\] Activation Steering Decoding: Mitigating Hallucination in Large Vision-Language Models through Bidirectional Hidden State Intervention](../../ACL2025/hallucination/activation_steering_decoding_mitigating_hallucination_in_large_vision-language_m.md)
- [\[ACL 2026\] Generating Effective CoT Traces for Mitigating Causal Hallucination](../../ACL2026/hallucination/generating_effective_cot_traces_for_mitigating_causal_hallucination.md)

</div>

<!-- RELATED:END -->
