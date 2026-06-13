---
title: >-
  [论文解读] ODE: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models
description: >-
  [CVPR 2025][幻觉检测][幻觉评估] 本文提出 ODE（Open-set Dynamic Evaluation）协议，通过图结构建模现实世界物体概念及其分布关联，从中动态提取概念组合并生成合成测试图像，实现了开放集、持续更新的多模态幻觉评估，有效避免了现有静态基准可能存在的数据污染问题。 领域现状：多模态大模型（M…
tags:
  - "CVPR 2025"
  - "幻觉检测"
  - "幻觉评估"
  - "开放集评估"
  - "数据污染"
  - "动态测试"
  - "图文生成"
---

# ODE: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models

**会议**: CVPR 2025  
**arXiv**: [2409.09318](https://arxiv.org/abs/2409.09318)  
**代码**: [https://github.com/Iridescent-y/ODE](https://github.com/Iridescent-y/ODE)  
**领域**: 幻觉检测  
**关键词**: 幻觉评估, 开放集评估, 数据污染, 动态测试, 图文生成

## 一句话总结

本文提出 ODE（Open-set Dynamic Evaluation）协议，通过图结构建模现实世界物体概念及其分布关联，从中动态提取概念组合并生成合成测试图像，实现了开放集、持续更新的多模态幻觉评估，有效避免了现有静态基准可能存在的数据污染问题。

## 研究背景与动机

**领域现状**：多模态大模型（MLLM）的幻觉问题已引起广泛关注，社区提出了一系列评估基准：CHAIR 测量描述中的物体准确率、POPE 评估物体存在性判别、AMBER 从存在性/属性/关系三个维度评估、HallusionBench 关注视觉常识推理。这些基准推动了幻觉研究的快速发展。

**现有痛点**：现有基准几乎全部是静态的——使用固定的测试数据（如 COCO2014 子集），分布有限。随着模型训练数据规模不断扩大，测试数据与训练数据重叠的风险日益增大。作者发现了一个关键证据：在相同语义分布下，模型在 COCO2014 图像上的表现明显优于最新互联网图像（后者更不可能被训练过），暗示正确回答可能源于数据污染而非真正的理解。

**核心矛盾**：静态基准无法区分模型是"真正理解了视觉内容"还是"记住了训练中见过的测试样本"。在 LLM 领域，数据污染已被广泛讨论（GPT-4、LLaMA 报告都有提及），但多模态领域尚无针对性的解决方案。

**本文目标** (1) 如何生成开放集的、模型未见过的测试样本来评估幻觉；(2) 如何在不同分布水平上系统化地测试模型的鲁棒性；(3) 如何利用动态评估数据反哺模型优化。

**切入角度**：如果测试数据是全新生成的（合成图像+动态概念组合），模型就不可能在训练中见过，从而消除数据污染。关键创新是用图结构建模概念之间的共现关系，按不同频率分布标准选择概念组合。

**核心 idea**：用图结构建模物体概念关联，动态生成不同分布层次的合成测试样本，实现开放集幻觉评估。

## 方法详解

### 整体框架

ODE 协议包含四个步骤：(1) 图结构建模——将现实世界物体概念、属性及其共现关系构建为加权图 $G=(V, A, E, W)$；(2) 语义场景构建——按四种分布标准从图中选取概念对并赋予属性；(3) 图像生成与过滤——用文本到图像模型生成测试图片并质控；(4) 查询模板设计——自动生成针对存在性和属性幻觉的评估问题。

### 关键设计

1. **图结构概念建模**:

    - 功能：将现实世界场景抽象为可操作的图结构
    - 核心思路：从 AMBER 基准提取 337 个物体类别作为节点 $V$，按场景功能分为环境级（如 grass）和实体级（如 frisbee）。每个节点附带属性节点 $A$（状态、动作、数量）。边权 $W$ 由两个概念在数据集中的共现频率确定，反映语义关联强度。概念进一步区分为实体-环境和实体-实体两种共现模式
    - 设计动机：图结构不仅能表示概念间的关联强度，还能方便地按不同分布标准提取概念组合，支持动态更新和领域扩展

2. **四级分布选择标准**:

    - 功能：在不同语义分布水平上系统测试模型的幻觉表现
    - 核心思路：(1) **Standard**——选择共现频率最高的概念对 $(V_i, V_j) \in \arg\max c_{i,j}$，测试模型对高频组合的理解；(2) **Long-tail**——选择中等共现频率的对 $\epsilon < c_{k,l} < \delta$，测试长尾分布下的表现；(3) **Random**——均匀随机选择 $(V_i, V_j) \sim \text{Uniform}(V \times V)$，属性也随机选取，测试鲁棒性；(4) **Fictional**——选择没有共现记录的对 $c_{k,l} = 0$，测试对全新概念组合的推理能力
    - 设计动机：模型在不同分布频率下的表现可能截然不同——高频可能靠记忆，低频/虚构可能暴露真实理解能力

3. **合成图像生成与质控**:

    - 功能：生成模型未见过的高质量测试图像
    - 核心思路：用 FLUX.1-dev 或 Stable Diffusion 1.5 根据文本描述（如"a picture of a black running dog and a yellow frisbee"）生成图像。为每个测试用例设不同随机种子生成多张，用开放词汇目标检测模型过滤——若目标实体的检测置信度低于 0.65 则丢弃。最终保留高质量样本，检测到的所有概念作为 ground truth
    - 设计动机：合成图像从源头消除数据污染可能性。CLIP 特征分析表明合成图像和自然图像在特征空间高度相似，验证了替代可行性

### 损失函数 / 训练策略

ODE 本身是评估协议，不涉及训练。但作者展示了 ODE 生成的数据可用于模型微调——对 ODE 识别出的错误样本进行针对性微调可有效减少幻觉。

## 实验关键数据

### 主实验（ODE vs AMBER 静态基准对比）

| 模型 | AMBER-Exist F1 | ODE-Standard Exist F1 | AMBER-Attr F1 | ODE-Standard Attr F1 |
|------|---------------|----------------------|--------------|---------------------|
| LLaVA-1.5 | 83.0 | 70.7 | 64.8 | 44.8 |
| CogVLM | 34.5 | 41.5 | 29.7 | 50.8 |
| InstructBLIP | 80.5 | 67.4 | 71.4 | 36.6 |
| MiniGPT-4 | 98.4 | 64.3 | 56.6 | 19.0 |

### 不同图像生成模型的影响

| 模型 | ODE-SD Exist Acc | ODE-Flux Exist Acc | Δ |
|------|----------------|--------------------|---|
| LLaVA-1.5 | 94.3 | 51.3 | +43.0 |
| CogVLM | 92.8 | 41.4 | +51.4 |
| MiniGPT-4 | 66.7 | 67.1 | -0.4 |

### 关键发现

- 多数模型在 ODE 生成的样本上表现明显低于静态基准（如 MiniGPT-4 的存在性 F1 从 AMBER 的 98.4% 降到 ODE-Standard 的 64.3%），强烈暗示静态基准中存在数据污染
- Random 和 Fictional 分布下幻觉率明显上升，尤其在属性识别任务中，说明模型高度依赖训练中学到的共现模式
- 不同图像生成模型（FLUX vs SD1.5）得到的合成图像质量差异导致评估结果差异巨大（CogVLM 差 51.4 个点），引入了新的不可控变量
- 生成任务中模型对高频概念表现尚可，但判别任务中高频概念反而可能因过度记忆而不稳定

## 亮点与洞察

- **用合成图像消除数据污染**：核心洞察简单而深刻——如果测试图都是新生成的，模型不可能在训练中见过。CLIP 特征对比验证了合成与自然图像的相似性，为后续工作提供了方法论基础
- **四级分布标准的设计**：从 Standard 到 Fictional，提供了从"模型熟悉"到"模型陌生"的连续谱系评估。这揭示了模型在不同认知难度下的表现差异，比单一分布评估信息量大得多
- **评估-优化闭环**：ODE 不仅用于评估，其生成的数据还可以直接用于微调模型减少幻觉，实现了评估和改进的闭环

## 局限与展望

- 合成图像质量受生图模型限制，FLUX vs SD1.5 导致评估结果差异极大（某些模型差 40+ 个点），这引入了新的不可控变量
- 目前仅支持两个物体的组合场景，无法评估更复杂的多物体场景
- 概念图只包含 337 个类别（来自 AMBER），覆盖范围有限
- 物体检测过滤（置信度阈值 0.65）可能过于保守或不够准确，影响 ground truth 质量
- 没有评估关系型幻觉（物体之间的空间关系、交互关系），仅覆盖存在性和属性幻觉

## 相关工作与启发

- **vs POPE**: POPE 基于固定的 COCO 图像评估存在性幻觉，受数据污染影响；ODE 动态生成图像避免了这个问题，且揭示了 POPE 可能高估了模型能力
- **vs AMBER**: AMBER 提供了多维度幻觉评估框架，但仍是静态的；ODE 可以视为 AMBER 的动态化扩展
- **vs DyVal (LLM 动态评估)**: DyVal 通过有向无环图动态合成数学推理样本，但限于特定算法。ODE 将动态评估思路扩展到多模态领域，通过概念图+图像生成实现跨模态动态测试
- 数据污染是一个被低估的问题——社区在报告模型性能时是否需要常规性地加入 OOD 评估？

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次系统性地解决多模态幻觉评估中的数据污染问题，思路新颖
- 实验充分度: ⭐⭐⭐⭐ 多模型、多分布、多任务全面评估，但合成图像质量差异是遗留问题
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法流程明确，但部分细节（如过滤阈值选择）不够充分
- 价值: ⭐⭐⭐⭐⭐ 提出了可持续更新的评估范式，对社区有长远影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] ReefKnot: A Comprehensive Benchmark for Relation Hallucination Evaluation, Analysis and Mitigation in Multimodal Large Language Models](../../ACL2025/hallucination/reefknot_a_comprehensive_benchmark_for_relation_hallucination_evaluation_analysi.md)
- [\[NeurIPS 2025\] Seeing is Believing? Mitigating OCR Hallucinations in Multimodal Large Language Models](../../NeurIPS2025/hallucination/seeing_is_believing_mitigating_ocr_hallucinations_in_multimodal_large_language_m.md)
- [\[AAAI 2026\] Beyond Hallucinations: A Composite Score for Measuring Reliability in Open-Source Large Language Models](../../AAAI2026/hallucination/beyond_hallucinations_a_composite_score_for_measuring_reliability_in_open-source.md)
- [\[CVPR 2025\] HalLoc: Token-Level Localization of Hallucinations for Vision Language Models](halloc_token-level_localization_of_hallucinations_for_vision_language_models.md)
- [\[ACL 2025\] FIHA: Autonomous Fine-grained Hallucination Evaluation in Vision-Language Models with Davidson Scene Graphs](../../ACL2025/hallucination/fiha_autonomous_hallucination_evaluation_in_vision-language_models_with_davidson.md)

</div>

<!-- RELATED:END -->
