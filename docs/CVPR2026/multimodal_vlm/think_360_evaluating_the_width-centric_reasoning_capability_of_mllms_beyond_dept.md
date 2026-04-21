---
title: >-
  [论文解读] Think360: Evaluating the Width-centric Reasoning Capability of MLLMs Beyond Depth
description: >-
  [CVPR 2026][多模态][多模态推理] 本文提出 Think360，一个聚焦于"推理宽度"（即模型在多路径搜索、多约束剪枝、回溯试错等方面的能力）的多模态基准，包含 1200+ 高质量样本，并设计细粒度 Tree-of-Thought 评估协议，揭示当前 MLLM 在宽度方向推理上的显著短板。
tags:
  - CVPR 2026
  - 多模态
  - 多模态推理
  - 推理宽度
  - 思维树评估
  - 基准测试
  - 大语言模型
---

# Think360: Evaluating the Width-centric Reasoning Capability of MLLMs Beyond Depth

**会议**: CVPR 2026  
**arXiv**: [2603.22689](https://arxiv.org/abs/2603.22689)  
**代码**: [Think360](https://github.com/Think360-Benchmark/Think360)  
**领域**: 多模态VLM / LLM推理  
**关键词**: 多模态推理, 推理宽度, 思维树评估, 基准测试, 大语言模型

## 一句话总结

本文提出 Think360，一个聚焦于"推理宽度"（即模型在多路径搜索、多约束剪枝、回溯试错等方面的能力）的多模态基准，包含 1200+ 高质量样本，并设计细粒度 Tree-of-Thought 评估协议，揭示当前 MLLM 在宽度方向推理上的显著短板。

## 研究背景与动机

1. **领域现状**：近年来大型推理模型（LRM）在 test-time scaling 和长链推理方面取得显著进展。现有基准如 MathVista、MathVerse、OlympiadBench 等不断推高难度和任务覆盖面，从 K-12 到研究生级别、从文本到多模态输入。

2. **现有痛点**：几乎所有已有评测基准都隐含地只衡量"推理深度"（reasoning depth），即模型沿单一推理链条逐步推导的能力。然而，人类解决问题时很少仅靠线性推演，更多是在解空间中多方向搜索、分支回溯、试错剪枝，最终整合部分发现形成答案。

3. **核心矛盾**：推理深度和推理宽度是两个正交维度。现有基准将二者混为一谈，导致无法区分模型到底是"想得深"还是"搜得广"。缺乏对宽度维度的系统评测，使得模型的真实推理能力被片面评估。

4. **本文目标** 构建一个专门评估推理宽度的多模态基准，包括：(a) 系统化地定义推理宽度的认知能力维度，(b) 设计合理的评测协议来同时量化深度和宽度，(c) 全面评测主流 MLLM 的宽度推理能力。

5. **切入角度**：作者类比神经网络架构中的"宽度"设计（shortcut connection、dropout、金字塔特征、梯度反传）与推理过程中的策略（剪枝、分而治之、试错、回溯），建立了深度/宽度在架构与推理之间的对应关系。

6. **核心 idea**：通过构建聚焦宽度推理的 1200+ 多模态基准 Think360 和 Tree-of-Thought 评估协议，系统揭示 MLLM 在探索式推理方面的不足。

## 方法详解

### 整体框架

Think360 是一个评测基准而非模型方法。整体构建流程分为三个阶段：(1) 多源原始数据收集 → (2) 粗到细质量过滤 → (3) 标注与重写。评测方面采用 pass@1 准确率 + Tree-of-Thought 深度/宽度得分 + 推理时间/token消耗。

### 关键设计

1. **推理宽度的形式化定义**

    - 功能：将推理宽度与推理深度明确区分为两个正交维度
    - 核心思路：推理深度衡量沿单一推理链条的逐步延伸能力；推理宽度聚焦于系统化地试错搜索（trial-and-error）、多约束剪枝（branch-and-bound）、分治策略（divide-and-conquer）、假设检验（hypothesize-and-test）以及感知理解（perceive-and-comprehend）五种认知能力。这五种能力对应了不同类型的"横向"搜索策略，类比于神经网络中 dropout 对应剪枝、shortcut 对应回溯等。
    - 设计动机：已有基准几乎没有专门量化宽度推理的，导致模型被认为"会推理"但实际上只是"能沿固定路径走得远"，缺乏对多路径搜索能力的系统评估。

2. **多源数据构建与质量过滤**

    - 功能：构建 1225 道高质量多模态推理题
    - 核心思路：数据来自四类来源——数学/逻辑竞赛题、教材例题、已有基准（MathVision、DynaMath、MME-Reasoning 等）、在线益智游戏/IQ 测试。过滤采用两阶段策略：粗筛用关键词匹配（如 maximum/minimum、possible ways）+ GPT-4o 作为评判；细筛由人工进行二次质量和多样性检查。对证明题进行改写使答案可验证，对游戏题设计可枚举的问答格式。
    - 设计动机：直接从已有基准抽取的宽度推理题占比极低（如 MathVista 仅 2.7%，OlympiadBench 仅 1.7%），需要专门收集和改造。同时不同来源的数据格式差异大，必须统一为可客观验证的形式。

3. **细粒度分类体系**

    - 功能：从多个维度对题目进行分类，支持细粒度分析
    - 核心思路：采用四个分类轴——答案类型（选择题 16.9%、自由作答 83.1%）、难度分级（Easy/Basic/Medium/Hard/Olympiad 五级，近似正态分布）、认知能力（5 种非互斥类别）、题型（6 种非互斥类别）。非互斥分类允许一道题同时标注多种认知能力。
    - 设计动机：互斥分类无法反映推理宽度题目通常需要同时调用多种能力的特点；非互斥分类通过频率统计和弦图可视化揭示不同能力间的共现关系。

4. **Tree-of-Thought 评估协议（ToT-Eval）**

    - 功能：超越传统的 pass@1 准确率，从深度和宽度两个维度量化模型推理过程
    - 核心思路：分为两步——(a) 树构建：给定问题和模型完整回复，由 GPT-4o 提取关键推理步骤并组织为层次树结构，其中深度表示顺序推理依赖（父子关系），宽度表示并行探索的替代方案（同层兄弟节点）。(b) 深度/宽度评分：由 GPT-4o 判定每个节点的正确性（逻辑是否合理、事实是否准确）。深度得分 = 最长正确推理链深度，宽度得分 = 有效的并行推理分支数。
    - 设计动机：传统 outcome-based 评测只看最终答案对错，无法区分模型是"一步到位"还是经过了充分的探索和验证。ToT-Eval 能够同时量化模型在探索广度和推理深度上的表现，更精准地反映宽度推理能力。

### 损失函数 / 训练策略

本文为 benchmark 评测工作，不涉及模型训练。评估方面设定温度 0.7，每题重复 3 次取均值以减少方差。所有模型配置为支持的最大输出长度。同时测试了有/无 Chain-of-Thought 提示的影响。

## 实验关键数据

### 主实验

评测涵盖 12 个主要模型系列（GPT、Gemini、Claude、Grok、Doubao、QwenVL、InternVL、LLaVA、Llama、GLM-V、MiMo、Kimi），共 30+ 模型。

| 模型 | 总体准确率 | 推理时间(s) | Token消耗 | Trial-and-Error | Branch-and-Bound |
|------|-----------|------------|----------|-----------------|------------------|
| Gemini-2.5-pro | **46.0%** | 160.19 | 17270 | 38.5% | 51.8% |
| o3 | 42.3% | 261.59 | 6326 | 35.5% | 48.0% |
| o4-mini | 42.1% | 84.61 | 6736 | 34.3% | 48.0% |
| Gemini-2.5-flash-thinking | 38.3% | 107.33 | 21273 | 31.1% | 43.4% |
| o1 | 36.8% | 186.81 | 6537 | 29.6% | 40.6% |
| Claude-3.7-Sonnet-Thinking | 35.5% | 295.94 | 13819 | 29.4% | 38.8% |
| MiMo-VL-RL (7B) | 28.3% | 334.21 | 7381 | 24.9% | 27.9% |
| GPT-4o | 16.0% | 13.28 | 309 | 15.3% | 16.8% |
| LLaVA-Onevision (7B) | 8.3% | 36.58 | 648 | 5.8% | 10.0% |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| CoT prompting (GPT-4o) | +0.4% 准确率 | CoT 提示带来微小提升，推理时间翻倍 |
| Perceive-and-Comprehend 子集 | 高于总体均值 | 感知理解型任务模型表现相对好 |
| Trial-and-Error 子集 | 低于总体均值 | 试错搜索型任务是模型短板 |
| Divide-and-Conquer 子集 | 低于总体均值 | 分而治之任务同样困难 |
| Text-Only vs Image+Text | 详见附录 | 多模态输入的影响分析 |

### 关键发现

- **Gemini-2.5-pro 以 46.0% 准确率排名第一**，其 thinking token 平均 17270 个，约为 o3/o4-mini 的 3 倍，但推理时间反而更短（160s vs o3 的 262s），说明该模型的推理效率更高。
- **综合性价比最优为 o4-mini**：准确率 42.1% 与 o3 相当，但推理时间仅 85s（o3 的 1/3）。
- **所有模型在 40% 以下挣扎**：仅 3 个模型突破 40% 门槛，说明宽度推理对当前 MLLM 仍是严峻挑战。
- **感知理解 vs. 试错搜索的分化**：各模型在 Perceive-and-Comprehend 子集上表现普遍高于平均，但在 Trial-and-Error 和 Divide-and-Conquer 子集上显著低于平均，表明当前 MLLM 更擅长结构化感知而非探索式推理。
- **开源模型差距明显**：最佳开源模型 MiMo-VL-RL (7B) 准确率 28.3%，与闭源领先者差距约 18 个百分点。

## 亮点与洞察

- **推理宽度的概念化**：将推理宽度与深度明确分离，并建立了与神经网络架构设计的精彩类比（dropout↔剪枝、shortcut↔回溯、金字塔↔分治等），概念清晰且有启发性。
- **ToT-Eval 评估协议**：不仅评最终答案，还分析推理过程的树结构，量化深度和宽度两个维度，比传统 pass@1 提供了更丰富的诊断信息。这一评估方式可以迁移到任何需要评价推理质量的场景。
- **1200+ 题目的精细构建流程**：从竞赛题到益智游戏，多源数据经过关键词匹配 + LLM-as-Judge + 人工审核的三级过滤，保证了题目质量和宽度推理的针对性。证明题和游戏题的改造方法值得借鉴。

## 局限与展望

- **评估依赖 GPT-4o/GPT-4o-mini**：树构建和节点正确性判断都依赖 GPT-4o，引入了评估器自身的偏差，且评估成本较高。
- **数据集规模有限**：1225 道题相对主流推理基准（如 MathVista 5000+ 题）偏少，各认知能力子集的样本量可能不足以支持稳健的统计结论。
- **缺乏过程奖励/过程监督的评估**：虽然提出了 ToT-Eval，但未将其应用于训练（如 process-based reward），未能验证对模型改进的指导作用。
- **可扩展性**：如何自动化生成更多高质量宽度推理题目，避免人工标注瓶颈，是实际推广的关键问题。

## 相关工作与启发

- **vs MathVista/MathVerse**: 这些基准覆盖多模态数学推理，但宽度推理题目占比极低（<3%）。Think360 专注宽度维度，二者互补。
- **vs CLEVR/GQA**: 早期组合视觉推理基准着重语义理解，Think360 强调更高层次的搜索和规划策略。
- **vs OlympiadBench**: 竞赛级别难度基准侧重长链推理（深度），Think360 在同等难度下聚焦多路径搜索（宽度）。
- **启发**：该 benchmark 暴露了当前 MLLM 的一个系统性不足——缺乏有效的探索和回溯能力。这提示 RL-based 训练（如 o1/o3）可能需要更多地鼓励模型在推理过程中进行多分支搜索，而非仅仅拉长链条。

## 评分

- 新颖性: ⭐⭐⭐⭐ 推理宽度作为独立维度的系统化评测是新颖视角，但 benchmark 类工作本身创新有限
- 实验充分度: ⭐⭐⭐⭐⭐ 30+ 模型全面评测，分维度分难度分析详尽
- 写作质量: ⭐⭐⭐⭐ 概念阐述清晰，类比贴切，但部分表格过于密集影响阅读
- 价值: ⭐⭐⭐⭐ 揭示了 MLLM 推理能力的盲区，对后续模型设计和训练策略有指导意义

<!-- RELATED:START -->

## 相关论文

- [Beyond Recognition: Evaluating Visual Perspective Taking in Vision Language Models](beyond_recognition_evaluating_visual_perspective_taking_in_vision_language_model.md)
- [EgoMind: Activating Spatial Cognition through Linguistic Reasoning in MLLMs](egomind_activating_spatial_cognition_through_linguistic_reasoning_in_mllms.md)
- [Beyond Static Artifacts: A Forensic Benchmark for Video Deepfake Reasoning in Vision Language Models](beyond_static_artifacts_a_forensic_benchmark_for_video_deepfake_reasoning_in_vis.md)
- [Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models](../../ICLR2026/multimodal_vlm/vision-r1_incentivizing_reasoning_capability_in_multimodal_large_language_models.md)
- [SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models](spatialqa_a_benchmark_for_evaluating_spatial_logical_reasoning_in_vision-languag.md)

<!-- RELATED:END -->
