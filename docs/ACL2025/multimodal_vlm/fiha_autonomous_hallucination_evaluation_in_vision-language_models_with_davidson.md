---
title: >-
  [论文解读] FIHA: Autonomous Fine-grained Hallucination Evaluation in Vision-Language Models with Davidson Scene Graphs
description: >-
  [ACL 2025][多模态][幻觉评估] 本文提出 FIHA，一个无需 LLM 和人工标注的自动化细粒度幻觉评估框架，通过从图像和描述中提取实体、属性和关系生成 Q&A 对，并引入 Davidson 场景图（DSG）建模问题间的依赖关系，构建了 FIHA-v1 基准，全面评估了主流大视觉语言模型的幻觉水平。
tags:
  - ACL 2025
  - 多模态
  - 幻觉评估
  - 大视觉语言模型
  - Davidson场景图
  - 无LLM评估
  - 细粒度评估
---

# FIHA: Autonomous Fine-grained Hallucination Evaluation in Vision-Language Models with Davidson Scene Graphs

**会议**: ACL 2025  
**arXiv**: [2409.13612](https://arxiv.org/abs/2409.13612)  
**代码**: https://github.com/confidentzzzs/FIHA (有)  
**领域**: multimodal_vlm  
**关键词**: 幻觉评估, 大视觉语言模型, Davidson场景图, 无LLM评估, 细粒度评估

## 一句话总结
本文提出 FIHA，一个无需 LLM 和人工标注的自动化细粒度幻觉评估框架，通过从图像和描述中提取实体、属性和关系生成 Q&A 对，并引入 Davidson 场景图（DSG）建模问题间的依赖关系，构建了 FIHA-v1 基准，全面评估了主流大视觉语言模型的幻觉水平。

## 研究背景与动机

大视觉语言模型（LVLMs）如 LLaVA、MiniGPT-4 在视觉理解方面展现了强大能力，但普遍存在幻觉问题——模型可能描述图像中不存在的物体、错误的属性或关系。准确评估模型的幻觉程度对于提升模型可信度至关重要。

**现有痛点**：当前幻觉评估方法面临两个核心挑战：

**忽视问题间的依赖关系**：现有方法独立评估每个问题，但实际上问题之间存在逻辑依赖。例如，如果模型连"图中是否有自行车"都回答错误，那么"自行车是什么颜色"这类依赖性问题的回答就没有意义。忽视这种依赖会导致评估结果不可靠——弱模型可能因为猜对了依赖性问题而获得虚高分数。

**依赖昂贵的人工标注或 LLM**：大多数现有基准需要人工标注（如 AMBER）或依赖 LLM 生成 Q&A 对（如 Hal-Eval），成本高且可扩展性差。

**核心矛盾**：需要一个既全面覆盖多种幻觉类型（物体、属性、关系）、同时成本低且可靠的评估框架。

**切入角度**：利用成熟的视觉工具（目标检测、关系提取）自动生成 Q&A 对，避免对 LLM 和人工标注的依赖；引入 DSG 建模问题依赖结构，提升评估可靠性。

## 方法详解

### 整体框架
FIHA 提供两条并行的 Q&A 生成路径：基于图像、基于描述文字。两条路径分别提取实体信息，生成多样化的问题，最终通过 DSG 组织问题间的依赖关系后送入模型评估。

### 关键设计

1. **基于描述的信息提取**：

    - 如无现成描述，使用 BLIP-2 为图像生成描述（选择小模型以减少幻觉）
    - 用 SpaCy 词性标注提取物体及其属性（颜色、数量、大小等），得到 $G^C_{O,A} = \{o_1:A_1, \ldots, o_n:A_n\}$
    - 用 Stanford CoreNLP 提取物体间关系 $G^C_R = \{R_1(o^1_{R_1}, o^2_{R_1}), \ldots\}$
    - 设计动机：描述文字提供了人类视角的信息总结，提取方法成熟可靠

2. **基于图像的信息提取**：

    - 用 Grounding DINO 进行目标检测，提取物体和属性（颜色、大小、形状）
    - 用 RelTR 生成稀疏场景图，提取物体间的空间和动作关系
    - 设计动机：图像包含比描述更丰富的细节信息，两条路径互补覆盖

3. **多类型 Q&A 对生成**：

    - **Yes-No 问题**：检查物体存在性（"图中是否有{obj}？"）和关系（"{obj1}附近是否有{obj2}？"）
    - **Wh-问题**：使用 what/who/which/where/how many 等疑问词，要求自由文本回答（不超过三个词）
    - **否定问题**：将真实物体/属性/关系替换为不存在的对象，检测模型是否产生虚假确认
    - 设计动机：多类型问题比仅用 Yes-No 问题提供更全面的幻觉评估

4. **Davidson 场景图（DSG）依赖建模**：

    - 将 Q&A 对组织为树状结构，物体存在性问题为根节点，相关属性和关系问题为叶节点
    - 评估时先判断根节点：如果物体存在性回答错误，直接判定所有叶节点问题为幻觉
    - 设计动机：避免弱模型通过猜对依赖性问题获得虚高分数，使评估更严格可靠

## 实验关键数据

### 主实验 — MSCOCO 数据集（图像生成的Q&A）
| 模型 | Accuracy | Precision | Recall | F1 | F1(Gen) |
|------|----------|-----------|--------|-----|---------|
| mPLUG-Owl | 42.1 | 70.2 | 61.4 | 43.7 | 15.2 |
| MiniGPT-4 | 23.5 | 27.5 | 22.2 | 22.1 | 21.6 |
| LLaVA-1.5-7B | 77.8 | 77.0 | 65.9 | 67.7 | 21.4 |
| LLaVA-1.5-13B | 78.9 | 80.9 | 66.4 | 68.3 | 20.9 |
| InstructBLIP | 84.7 | 83.3 | 78.6 | 80.4 | 21.8 |
| GPT-4V | **87.2** | **81.4** | **86.3** | **85.5** | **25.2** |

### 消融实验 — DSG 引入后的性能变化
| 模型 | Acc.降幅 | F1降幅 | 说明 |
|------|---------|--------|------|
| GPT-4V | 6.0% | 9.9%→8.4% | 强模型受 DSG 影响最小，显示更强的上下文推理 |
| LLaVA-1.5-13B | 2.7% | 3.6% | 较少的级联错误 |
| mPLUG-Owl | 29.6% | 28.7% | 弱模型根节点错误频繁传播 |
| MiniGPT-4 | 62.6% | 61.2% | 最弱模型暴露出大量基础幻觉 |

### 细粒度结果（MSCOCO Caption-based）
| 幻觉类型 | GPT-4V F1 | InstructBLIP F1 | 说明 |
|----------|-----------|-----------------|------|
| 物体存在性 | 88.6 | 84.2 | 模型普遍表现较好 |
| 属性识别 | 79.8 | 55.6 | 颜色/大小等属性判断难度显著上升 |
| 关系判断 | 58.3 | 52.1 | 最具挑战性，涉及多物体交互 |

### 关键发现
- GPT-4V 在所有维度上表现最优，InstructBLIP 紧随其后
- 所有模型在关系幻觉上表现最差（GPT-4V 的 F1 仅 58.3%），因为关系涉及多个物体
- 属性幻觉是中等难度，模模型在雾化图像上性能显著下降
- 模型参数量越大表现越好（LLaVA-1.5-13B 优于 7B）
- FIHA 生成的 Q&A 对人工验证准确率达 96-98.2%

## 亮点与洞察
- 首个同时做到 LLM-free 和 Annotation-free 的 LVLM 幻觉评估框架，成本极低可大规模部署
- DSG 依赖建模是一个简单但有效的创新，暴露了弱模型被传统评估高估的问题
- 同时支持判别式和生成式两种题型评估，覆盖面比此前任何工作都广
- 在雾化图像上的测试展示了评估框架对噪声图像的鲁棒性分析

## 局限与展望
- 信息提取依赖 Grounding DINO 和 RelTR 等工具的准确性，工具本身的错误会传播到评估中
- 目前仅测试了 7 个模型，缺少更新的模型如 LLaVA-Next、Qwen-VL 等
- Wh-问题使用 BERTScore 评估，对自由文本回答的匹配可能不够精确
- 否定问题的构造基于随机替换，可能不够困难（模型容易通过排除法作答）
- 未考虑更复杂的幻觉类型，如时间关系、因果推理等

## 相关工作与启发
- 与 POPE（仅检测物体幻觉）和 AMBER（需要人工标注）相比，FIHA 覆盖更全面且成本更低
- DSG 的引入受 Cho et al. (2023) 启发，将其从文本生成评估迁移到幻觉评估领域
- 两条信息提取路径（图像+描述）的互补设计思路可推广到其他视觉评估任务
- 提示了一个方向：利用现有成熟工具组合实现复杂评估任务，而非依赖端到端的大模型

## 评分
- 新颖性: ⭐⭐⭐⭐ DSG 依赖建模和无 LLM 评估是亮点，但信息提取方法本身不够新颖
- 实验充分度: ⭐⭐⭐⭐ 多数据集、多模型、细粒度分析和可靠性验证都很到位，但模型覆盖可更广
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，但部分符号定义略显冗余
- 价值: ⭐⭐⭐⭐ 提供了实用的低成本幻觉评估工具，对 LVLM 开发者有直接价值

<!-- RELATED:START -->

## 相关论文

- [ReefKnot: A Comprehensive Benchmark for Relation Hallucination Evaluation, Analysis and Mitigation in Multimodal Large Language Models](reefknot_a_comprehensive_benchmark_for_relation_hallucination_evaluation_analysi.md)
- [Zina: Multimodal Fine-grained Hallucination Detection and Editing](../../CVPR2026/multimodal_vlm/zina_multimodal_fine-grained_hallucination_detection_and_editing.md)
- [Do Vision-Language Models Have Internal World Models? Towards an Atomic Evaluation](do_vision-language_models_have_internal_world_models_towards_an_atomic_evaluatio.md)
- [Unveiling Chain of Step Reasoning for Vision-Language Models with Fine-grained Rewards](../../NeurIPS2025/multimodal_vlm/unveiling_chain_of_step_reasoning_for_visionlanguage_models.md)
- [SPHERE: Unveiling Spatial Blind Spots in Vision-Language Models Through Hierarchical Evaluation](sphere_unveiling_spatial_blind_spots_in.md)

<!-- RELATED:END -->
