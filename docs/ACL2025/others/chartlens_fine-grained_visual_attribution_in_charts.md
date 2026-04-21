---
title: >-
  [论文解读] ChartLens: Fine-Grained Visual Attribution in Charts
description: >-
  [ACL 2025][图表归因] 提出图表的事后细粒度视觉归因（Post-Hoc Fine-grained Visual Attribution）任务，设计 ChartLens 算法利用分割技术标记图表元素并通过 Set-of-Marks 提示多模态 LLM 进行精确归因，同时构建 ChartVA-Eval 基准，在三类图表上取得 26-66% 的 F1 提升。
tags:
  - ACL 2025
  - 图表归因
  - 视觉定位
  - 多模态LLM
  - 幻觉检测
  - Set-of-Marks
---

# ChartLens: Fine-Grained Visual Attribution in Charts

**会议**: ACL 2025  
**arXiv**: [2505.19360](https://arxiv.org/abs/2505.19360)  
**代码**: 有  
**领域**: NLP / 多模态理解  
**关键词**: 图表归因, 视觉定位, 多模态LLM, 幻觉检测, Set-of-Marks

## 一句话总结

提出图表的事后细粒度视觉归因（Post-Hoc Fine-grained Visual Attribution）任务，设计 ChartLens 算法利用分割技术标记图表元素并通过 Set-of-Marks 提示多模态 LLM 进行精确归因，同时构建 ChartVA-Eval 基准，在三类图表上取得 26-66% 的 F1 提升。

## 研究背景与动机

- 多模态大语言模型（MLLM）在图表理解任务上取得进展，但**幻觉问题**严重：生成的文本与图表视觉数据不一致
- **归因（Attribution）**是缓解幻觉的关键策略——使模型回答可追溯到具体视觉证据
- 已有 LLM 归因方法主要面向文本（引用检索、后检索问答），**视觉归因**尤其是**图表视觉归因**几乎空白
- 图表的特殊挑战：
    - 包含精确数量、趋势、比例等复杂关系
    - 视觉元素（柱状、饼扇、折线）高度结构化且可能重叠
    - 需要理解图表类型、数据编码、轴/图例/颜色等上下文
- 可靠的图表归因对金融分析、政策制定、科研等关键领域至关重要

## 方法详解

### 整体框架

ChartLens 两阶段流水线：
1. **Mark Generation**（标记生成）：通过分割技术检测图表元素（柱/饼/线），为每个元素生成标记
2. **Attribution with MLLMs**（归因推理）：将标记后的图表输入 MLLM，通过 Set-of-Marks 提示进行验证和归因

### 关键设计

#### 1. Mark Generation —— 分割与标记

**柱状图分割**：
- 输入图像通过 Otsu 阈值二值化（RGB 和 HSV 双空间）
- 暗背景自动反转，提取前景轮廓
- 基于唯一像素值分解轮廓，隔离单个柱元素
- 通过 solidity 和面积阈值过滤噪声轮廓
- **关键改进**：使用 SAM（Segment Anything Model）精化分割——从检测到的元素中采样 n 个点作为 SAM 的 prompt，生成精确 mask
- SAM 自然抑制网格线等背景特征（低 IoU mask）

**饼图分割**：
- 识别最大轮廓作为饼图边界
- 计算最小外接圆近似图表区域
- 沿径向展开为线性表示，检测完整边缘识别扇区边界
- 映射回原始圆形区域得到各扇区分割

**折线图分割**：
- 使用 **LineFormer**（Transformer架构）专门处理折线的细长结构、重叠轨迹和交叉线
- 检测候选线后按水平范围等分为小段，作为细粒度标记
- Transformer 全局上下文有助于区分紧密间隔或交叉的线

#### 2. Set-of-Marks 提示归因

- 将分割后的元素叠加字母/数字标签到图表图像上
- 构造结构化提示，包含两个目标：
    - **验证（Validation）**：QA 对是否与图表信息一致
    - **归因（Attribution）**：识别并引用支持答案的具体标记元素
- 使用 few-shot 文本示例 + Chain-of-Thought 推理引导 MLLM 逐步验证和归因

### 问题形式化

给定图表图像 c 和关联响应 v，目标是产生归因集合 A_{c,v} = {a_1, a_2, ..., a_n}，其中每个 a_i 对应图表中支持响应的一个视觉区域（柱/线段/扇区）。归因集合需满足：
1. **相关性（Relevance）**：每个区域直接相关于响应
2. **完整性（Completeness）**：覆盖所有必要视觉证据
3. **精确性（Precision）**：排除无关部分

## 实验关键数据

### ChartVA-Eval 基准统计

| 数据集 | 查询数 | 图表数 | 柱状图 | 饼图 | 折线图 | 来源 |
|-------|-------|-------|-------|------|-------|------|
| ChartVA-AITQA | 301 | 301 | 203 | 0 | 98 | 合成 |
| ChartVA-PlotQA | 595 | 581 | 396 | 0 | 199 | 合成 |
| ChartVA-ChartQA | 348 | 266 | 121 | 109 | 118 | 真实世界 |

### 主实验：柱状图归因 (P / R / F1)

| 方法 | AITQA F1 | PlotQA F1 | ChartQA F1 |
|------|---------|----------|-----------|
| Zero-shot GPT-4o | 22.77 | 3.30 | 7.75 |
| KOSMOS-2 | 0.51 | 1.01 | 3.13 |
| LISA | 1.62 | 0.34 | 1.01 |
| **ChartLens** | **69.28** | **34.65** | **64.14** |

### 折线图归因 (Detection % / Chart Area %)

| 方法 | AITQA Det.↑ | AITQA Area↓ | PlotQA Det.↑ | PlotQA Area↓ | ChartQA Det.↑ | ChartQA Area↓ |
|------|-----------|-----------|------------|------------|-------------|-------------|
| Zero-shot GPT-4o | 18.28 | 1.94 | 6.79 | 8.63 | 3.39 | 1.15 |
| KOSMOS-2 | 74.19 | 46.03 | 38.83 | 27.06 | 87.29 | 41.49 |
| LISA | 94.62 | 63.18 | 50.21 | 40.92 | 50.21 | 40.92 |
| **ChartLens** | **59.14** | **1.25** | **51.84** | **9.98** | **77.8** | **5.34** |

LISA/KOSMOS-2 高检测率是因为覆盖了大面积图表区域（40-63%），ChartLens 面积仅 1-10%，精确度远胜。

### 饼图归因 (ChartQA)

| 方法 | P | R | F1 |
|------|---|---|-----|
| Zero-shot GPT-4o | 8.94 | 5.99 | 7.17 |
| KOSMOS-2 | 20.18 | 8.24 | 11.70 |
| LISA | 1.32 | 13.86 | 2.41 |
| **ChartLens** | **53.33** | **44.57** | **48.56** |

### 消融实验

- **分割模块的模块化**：SAM 可替换为更先进的分割模型以进一步提升
- **MLLM 选择**：ChartLens 基于 ChatGPT-4o，可灵活替换底层多模态模型
- **SoM 提示的有效性**：直接让 GPT-4o zero-shot 预测边界框效果极差（F1 仅 3-23%），验证了 SoM 提示+分割的必要性

### 关键发现

1. **ChartLens 在所有图表类型上大幅领先**：柱状图 +46-66% F1，饼图 +37% F1
2. **现有视觉定位模型不适合图表归因**：KOSMOS-2 和 LISA 在图表元素精确定位上表现极差
3. **GPT-4o 的零样本边界框预测不可靠**：缺乏精确的视觉定位能力
4. **精确度 vs 覆盖率权衡**：LISA/KOSMOS-2 通过覆盖大面积获得高召回但精确性极低
5. **SoM 提示显著提升归因质量**：标记化使 MLLM 能明确引用具体视觉元素

## 亮点与洞察

- 首次系统定义图表细粒度视觉归因任务，填补多模态归因的重要空白
- 分割+标记+MLLM 的模块化设计：每个组件可独立升级，具有良好扩展性
- ChartVA-Eval 基准涵盖合成+真实世界图表，来源包括 SEC 财报、世界银行、GTD 等权威数据
- 提供了一种即插即用的后处理归因机制，可搭配任何 MLLM 使用，不影响原回答生成

## 局限与展望

- 分割精度是瓶颈：分割不准确会导致归因不完整
- 仅处理视觉图表元素（柱/线/扇区），**未涉及文本元素**（标题、标签、图例的归因）
- 依赖 GPT-4o 作为归因推理引擎，成本较高
- 未在更复杂图表类型上测试（如散点图、热力图、组合图表）
- 归因标注部分依赖自动化+人工验证，大规模标准化标注仍有挑战

## 相关工作与启发

- **文本归因**（Direct/Post-retrieval/Post-hoc）→ 本文将其扩展到视觉模态
- **MATSA** 提出表格归因任务，ChartLens 将其进一步扩展到图表视觉元素
- **SAM + LineFormer** 的组合策略为其他结构化视觉理解任务提供参考
- 启发：可将类似归因框架应用于信息图（infographics）、地图、流程图等其他结构化视觉数据

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ⭐⭐⭐⭐ 首次定义图表视觉归因任务 |
| 实验充分度 | ⭐⭐⭐⭐ 三类图表+四种基线+完整基准 |
| 实用价值 | ⭐⭐⭐⭐ 模块化可即插即用 |
| 写作质量 | ⭐⭐⭐⭐ 问题定义清晰、实验详尽 |
| 总体推荐 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [FRACTAL: Fine-Grained Scoring from Aggregate Text Labels](fractal_fine-grained_scoring_from_aggregate_text_labels.md)
- [Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)
- [DEMO: Reframing Dialogue Interaction with Fine-grained Element Modeling](demo_reframing_dialogue_interaction_with_fine-grained_element_modeling.md)
- [Guidelines for Fine-grained Sentence-level Arabic Readability Annotation](guidelines_for_fine-grained_sentence-level_arabic_readability_annotation.md)
- [Barec: A Large and Balanced Corpus for Fine-grained Arabic Readability Assessment](a_large_and_balanced_corpus_for_fine-grained_arabic_readability_assessment.md)

<!-- RELATED:END -->
