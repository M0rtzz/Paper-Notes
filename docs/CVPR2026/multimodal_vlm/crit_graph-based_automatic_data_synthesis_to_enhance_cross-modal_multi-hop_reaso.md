---
title: >-
  [论文解读] CRIT: Graph-Based Automatic Data Synthesis to Enhance Cross-Modal Multi-Hop Reasoning
description: >-
  [CVPR 2026][多模态][跨模态] 提出基于图结构的自动数据生成 pipeline，构建了 CRIT 数据集与 benchmark，用于训练和评测 VLM 在交错图文内容上的跨模态多跳推理能力，训练后的模型在 SPIQA 等多个基准上取得显著提升。
tags:
  - CVPR 2026
  - 多模态
  - 跨模态
  - Multi-Hop Reasoning
  - Data Synthesis
  - Graph-Based Pipeline
  - VLM Benchmark
---

# CRIT: Graph-Based Automatic Data Synthesis to Enhance Cross-Modal Multi-Hop Reasoning

**会议**: CVPR 2026  
**arXiv**: [2604.01634](https://arxiv.org/abs/2604.01634)  
**代码**: 无  
**领域**: Multimodal VLM  
**关键词**: Cross-Modal Reasoning, Multi-Hop Reasoning, Data Synthesis, Graph-Based Pipeline, VLM Benchmark

## 一句话总结

提出基于图结构的自动数据生成 pipeline，构建了 CRIT 数据集与 benchmark，用于训练和评测 VLM 在交错图文内容上的跨模态多跳推理能力，训练后的模型在 SPIQA 等多个基准上取得显著提升。

## 研究背景与动机

现实世界推理常需跨模态整合信息：例如阅读 DIY 教程时需不断在文字指令和配图之间交叉参照。然而，现有多模态基准存在严重缺陷：

**评估端**：大多数 benchmark 仅涉及单图或一组图片，答案往往可从单一模态推断，无法测试真正的跨模态推理

**训练端**：虽然大量交错图文数据用于预训练，但其中真正需要互补跨模态推理的数据极少

**模型端**：即使 SOTA 模型（GPT-4o）在需要 CoT 推理时，也经常产生与视觉/文本证据脱节的幻觉

直接用 VLM 生成复杂推理数据会引入循环偏差（用同类模型生成和评测）和幻觉问题。本文通过图结构作为中间表示，**全程仅需 LLM（无需 VLM）**即可生成问答对，避免了上述问题。

## 方法详解

### 整体框架

三阶段自动数据生成 pipeline：
1. **多模态内容图构建**：从带场景图标注的图像出发，构建统一的图结构表示
2. **文本上下文生成**：基于子图生成互补性文本描述
3. **问答对生成**：采样跨模态子图链，生成需要多跳推理的 QA 对

### 关键设计

1. **多模态内容图 (Multimodal Content Graph)**: 有向图 $G=(\mathcal{V}, \mathcal{E})$，节点代表实体（视觉对象或文本实体），边代表关系。核心操作：

    - 随机采样 1-6 张标注了场景图的图像
    - 规则过滤：仅保留可通过属性或关系唯一标识的实体，避免歧义
    - LLM 增强：为每个图像节点生成新的文本实体和关系，作为跨图像的桥接节点

2. **文本上下文生成**: 为每张图像提取关联子图，排除图像节点的属性和跨图关系（这些留给模型在推理时从图像中获取），让 LLM 以多种叙事风格（故事、日记、纪录片等）生成互补性文本。关键约束：文本只描述增强的文本节点及其与图像节点的连接，不泄露需要从图像推理的信息。

3. **QA 生成与多层过滤**: 

    - 采样包含 1-5 条边的跨模态子图链，终端节点必须来自图像
    - LLM 根据序列化的子图 JSON 和目标答案生成问题，约束中间实体不可在问题中直接提及
    - 同时生成 CoT 推理链
    - 三层过滤：(a) 剔除问题中显式提及中间实体的样本 (b) 用 3 个不同 LLM 检验单模态即可回答的问题 (c) 修剪过长 CoT

4. **扩展到视频和科学论文**: 

    - 视频：利用密集字幕数据集，选择与字幕高 CLIP 相似度的帧，LLM 转换字幕为场景图
    - 科学论文：将段落/图表/表格转化为统一图结构，标记视觉实体后从文本中移除对应描述

### 损失函数 / 训练策略

- 使用 LoRA 对 Qwen2.5-VL-7B 和 Idefics2-8B 进行 SFT
- 每个训练样本同时包含直接回答和 CoT 两种格式
- 数据生成 LLM：Qwen3-30B-A3B-Instruct-2507
- 过滤 LLM：Qwen3-30B + Gemma-3-27b-it + Mistral-Small-3.2-24B

## 实验关键数据

### 主实验

CRIT Benchmark 结果（CoT 评测，EM/F1）：

| 模型 | NI-EM | NI-F1 | VF-EM | VF-F1 | SP-EM | SP-F1 |
|------|-------|-------|-------|-------|-------|-------|
| GPT-4o | 35.1 | 37.7 | 32.0 | 38.9 | 8.4 | 14.0 |
| Qwen2.5-VL-7B | 28.3 | 29.1 | 24.0 | 27.8 | 6.8 | 9.6 |
| Qwen2.5-VL-72B | 38.0 | 39.4 | 30.1 | 33.9 | 9.4 | 12.3 |
| **Qwen2.5-VL_CRIT** | **58.6** | **59.5** | **38.8** | **42.2** | **15.9** | **22.5** |
| **Idefics2_CRIT** | **54.1** | **54.9** | **31.2** | **33.9** | **12.3** | **20.2** |

训练后的 7B 模型大幅超越 GPT-4o 和 72B 模型。

跨基准迁移效果（Idefics2 + Mantis-Instruct + CRIT vs. Mantis-Instruct only）：

| 基准 | 指标 | +CRIT | 仅 Mantis | 提升 |
|------|------|-------|-----------|------|
| SPIQA | METEOR | 10.53 | 3.60 | +192% |
| SPIQA | CIDEr | 67.93 | 23.83 | +185% |
| VEGA | ROUGE-L | 35.1 | 29.5 | +19% |
| MMQA | EM | 30.0 | 27.3 | +10% |
| FCMR | F1 | 50.5 | 44.9 | +12% |

### 消融实验

| 配置 | NI-EM | VF-EM | SP-EM | 说明 |
|------|-------|-------|-------|------|
| No Fine-tuning | 28.3 | 24.0 | 6.8 | 基线 |
| CRIT (84k) | 58.6 | 38.8 | 15.9 | 标准训练集 |
| CRIT Augmented (210k) | 62.6 | 45.6 | 16.7 | 扩展训练集，视频域提升最大 |

使用模型生成标注的扩展数据能进一步提升性能，且科学论文域也受益于自然图像/视频域的数据扩展（跨域迁移）。

### 关键发现

- **SOTA 模型在跨模态多跳推理上表现很差**：GPT-4o 在自然图像域仅 35.1% EM，科学论文域仅 8.4%
- **错误分析**（75 个 GPT-4o 错误样本）：55% 为证据定位错误（模型找错了图片或文本段落），视觉感知错误是文本理解错误的 4 倍
- **训练后不损害通用能力**：加入 CRIT 后在 MME、SeedBench 等通用基准上保持甚至提升性能

## 亮点与洞察

1. **图结构作为中间表示的设计极为精巧**：通过子图采样可编程式地保证多跳、跨模态约束，比直接 prompt VLM 生成数据质量高得多
2. **全程无需 VLM，仅用 LLM**：避免了用 VLM 生成 VLM 评测数据的循环偏差问题
3. **单模态过滤设计巧妙**：用 3 个不同 LLM 分别验证文本和视觉模态，确保问题确实需要跨模态推理
4. **管道高度可扩展**：从标注图像扩展到视频帧和科学论文，仅需适配图结构构建阶段

## 局限与展望

- 科学论文域表现仍然较低（15.9% EM），长文本 + 复杂图表的精确跨模态对齐仍是挑战
- 图结构构建依赖已有的场景图标注（GQA）或密集字幕标注（ActivityNet），完全无标注场景的适用性有待验证
- 当前仅评测了手动验证的 1,446 个测试样本，规模相对有限
- 未探索 CoT 推理链质量对训练效果的影响

## 相关工作与启发

- 图结构中间表示 → LLM 生成 QA 的范式可推广到其他需要结构化推理的数据合成任务
- "互补性"约束（排除图像属性和跨图关系从文本中泄露）是保证跨模态推理质量的关键
- 错误分析揭示"证据定位"是当前 VLM 最大瓶颈，而非推理能力本身

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 图结构数据生成 pipeline 设计精巧，解决了数据合成中的循环偏差问题
- 实验充分度: ⭐⭐⭐⭐ — 多模型对比 + 多基准迁移 + 数据扩展 + 错误分析
- 写作质量: ⭐⭐⭐⭐ — 三阶段 pipeline 描述清晰，Fig.2 的流程图信息量大
- 价值: ⭐⭐⭐⭐⭐ — 开创性地定义并解决了跨模态多跳推理的数据和评测瓶颈

<!-- RELATED:START -->

## 相关论文

- [Multi-Crit: Benchmarking Multimodal Judges on Pluralistic Criteria-Following](multi-crit_benchmarking_multimodal_judges_on_pluralistic_criteria-following.md)
- [VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion](videofusion_a_spatiotemporal_collaborative_network.md)
- [ImageBindDC: Compressing Multi-modal Data with ImageBind-based Condensation](../../AAAI2026/multimodal_vlm/imagebinddc_compressing_multi-modal_data_with_imagebind-based_condensation.md)
- [InEx: Hallucination Mitigation via Introspection and Cross-Modal Multi-Agent Collaboration](../../AAAI2026/multimodal_vlm/inex_hallucination_mitigation_via_introspection_and_cross-mo.md)
- [DynamicGTR: Leveraging Graph Topology Representation Preferences to Boost VLM Capabilities on Graph QAs](dynamicgtr_leveraging_graph_topology_representation_preferences_to_boost_vlm_cap.md)

<!-- RELATED:END -->
