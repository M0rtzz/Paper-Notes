---
title: >-
  [论文解读] Perception Programs: Unlocking Visual Tool Reasoning in Language Models
description: >-
  [CVPR 2026][LLM/NLP][感知程序] 提出 Perception Programs (P2)，一种训练免费、模型无关的方法，将视觉工具（深度、光流、对应等）的原始输出转换为紧凑的语言原生结构化摘要，使 MLLM 能直接"阅读"视觉模态而非从密集像素推断…
tags:
  - "CVPR 2026"
  - "LLM/NLP"
  - "感知程序"
  - "视觉工具"
  - "语言原生表示"
  - "训练免费"
  - "多模态推理"
---

# Perception Programs: Unlocking Visual Tool Reasoning in Language Models

**会议**: CVPR 2026  
**arXiv**: [2604.12896](https://arxiv.org/abs/2604.12896)  
**代码**: [https://github.com/AISmartPerception/perception-programs](https://github.com/AISmartPerception/perception-programs)  
**领域**: LLM/NLP  
**关键词**: 感知程序, 视觉工具, 语言原生表示, 训练免费, 多模态推理

## 一句话总结

提出 Perception Programs (P2)，一种训练免费、模型无关的方法，将视觉工具（深度、光流、对应等）的原始输出转换为紧凑的语言原生结构化摘要，使 MLLM 能直接"阅读"视觉模态而非从密集像素推断，在 BLINK 6 个任务上平均提升 19.66%。

## 研究背景与动机

**领域现状**：MLLM 越来越多地与视觉工具（深度估计、光流、视觉对应等）配合使用来增强视觉推理。

**现有痛点**：尽管视觉工具提供了准确的感知信号，MLLM 常常无法充分利用。原始工具输出是密集的像素级表示，与 LLM 的语言原生推理能力不匹配。实验表明 GPT-5 Mini 甚至无法从深度图恢复正确的深度排序（Kendall τ 快速趋近零）。

**核心矛盾**：瓶颈不在于更多的工具调用或更大的 MLLM，而在于视觉工具输出的表示方式。密集数值 token 与语言推理基底的根本性不匹配。

**本文目标**：将工具输出从密集像素级表示转换为语言原生的结构化摘要。

**切入角度**：人类对视觉信息的线索提取方式因数据类型而异（深度关注远近、光流关注方向等）。将关键信息转换为文本减轻了模型处理像素细节的负担。

**核心 idea**：P2 标准化了工具传达的内容（what）、空间位置（where）和部分间关系（how），使任何 MLLM 都能直接解析和推理。

## 方法详解

### 整体框架

给定视觉工具的原始输出，P2 将像素域划分为有限的基元集合（patches/points），为每个基元提取结构化项 $I_p = (p, c_p, r_p, b_p)$（标识符、归一化坐标、模态读数、可选标签），并生成稀疏的符号关系三元组 $\mathcal{T}$。整个摘要序列化为 YAML 格式的文本块，直接作为 MLLM 输入。

### 关键设计

1. **统一项模式 (Unified Item Schema)**:

    - 功能：跨模态的标准化表示
    - 核心思路：所有模态共享相同的项结构 $(p, c_p, r_p, b_p)$：基元标识、归一化到 [0,1000]² 的空间坐标、从模态数据提取的读数、可选语义标签。模态间唯一不同的是读数 $r_p$ 的构造方式和是否包含关系
    - 设计动机：统一模式使方法可泛化到深度、光流、对应、检测等多种模态

2. **模态特定读数构造**:

    - 功能：为每种视觉模态提取关键信息
    - 核心思路：深度：每个网格单元存储最小和最大深度值 $r_p = [\min D, \max D]$，并生成邻域间的关系三元组（如"更近于"、"更远于"）。光流：编码运动方向和幅度。对应：编码匹配点位置和置信度。检测：编码物体类别和边界框
    - 设计动机：每种模态的关键信息不同，需要专门的提取方式

3. **训练免费、模型无关部署**:

    - 功能：即插即用到任何 MLLM
    - 核心思路：P2 不需要参数更新、架构修改或额外工具调用。同一工具输出在标准工具使用管线中转换为 P2 后直接被 MLLM 消费。在推理时仅增加文本处理的微小开销
    - 设计动机：避免训练成本和模型修改，保持最大的灵活性

### 损失函数 / 训练策略

P2 不涉及任何训练。它是一个纯推理时的表示转换模块。

## 实验关键数据

### 主实验

| 模型 | 任务 | 基线 | +原始工具 | +P2 |
|------|------|------|---------|-----|
| GPT-5 Mini | 多视角推理 | 41.4% | 52.8% | **86.5%** |
| GPT-5 Mini | 相对深度 | 52.4% | 61.2% | **81.5%** |
| GPT-5 Mini | 视觉对应 | 38.7% | 45.3% | **72.1%** |
| InternVL3.5-4B | 6任务平均 | 42.1% | 48.5% | **70.3%** |
| Qwen3VL-4B | 6任务平均 | 43.5% | 49.2% | **71.8%** |

### 消融实验

| 配置 | BLINK 6任务平均 | 说明 |
|------|---------------|------|
| 完整 P2 | 86.5% | 项+关系 |
| 仅项 (无关系) | 78.2% | 无邻域关系 |
| 粗网格 (4×4) | 82.1% | 分辨率降低 |
| 细网格 (12×12) | 85.8% | 更高分辨率 |
| 原始工具输出 | 52.8% | 像素级表示 |

### 关键发现

- P2 在多视角推理上将 GPT-5 Mini 的准确率从 41.4% 提升到 86.5%（+45 个百分点），效果惊人
- 即使在 4B 级小模型上也有 21-25% 的绝对提升
- P2 可增强现有的 agent 工具使用方法：在深度和定位任务上额外提升 18.28%

## 亮点与洞察

- 核心洞察深刻：视觉推理的瓶颈不在工具准确性，而在表示方式。MLLM 能"读"文本但不能有效"看"密集数值
- P2 的设计体现了"让机器做机器擅长的事"的原则：让视觉工具提取感知信号，让 LLM 做语言推理
- 训练免费+模型无关使其具有极高的实用价值

## 局限与展望

- 网格划分的粒度需要根据任务调整
- 对于需要精确像素级信息的任务（如精细分割边界），P2 的空间离散化可能损失信息
- 未评估在视频时间维度上的扩展
- 可探索自适应粒度和动态关系生成

## 相关工作与启发

- **vs VisProg/ViperGPT**: 这些方法生成调用工具的程序，但仍在像素级操作工具输出；P2 改变的是工具输出的表示
- **vs Aurora/Mirage**: 这些方法用训练来改善工具使用，P2 不需要训练即可获得更大提升

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ "表示方式才是瓶颈"的洞察改变了问题定义
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型多任务的全面验证，效果惊人
- 写作质量: ⭐⭐⭐⭐⭐ 动机、分析和实验都很清晰
- 价值: ⭐⭐⭐⭐⭐ 对 MLLM 工具使用范式有重要启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Unlocking the Potential of Diffusion Language Models through Template Infilling](../../ACL2026/llm_nlp/unlocking_the_potential_of_diffusion_language_models_through_template_infilling.md)
- [\[ACL 2025\] ToolSpectrum: Towards Personalized Tool Utilization for Large Language Models](../../ACL2025/llm_nlp/toolspectrum_towards_personalized_tool_utilization_for_large_language_models.md)
- [\[ACL 2026\] Hot-Start from Pixels: Low-Resolution Visual Tokens for Chinese Language Modeling](../../ACL2026/llm_nlp/hot-start_from_pixels_low-resolution_visual_tokens_for_chinese_language_modeling.md)
- [\[ICLR 2026\] How Far Are LLMs from Professional Poker Players? Revisiting Game-Theoretic Reasoning with Agentic Tool Use](../../ICLR2026/llm_nlp/how_far_are_llms_from_professional_poker_players_revisiting_game-theoretic_reaso.md)
- [\[CVPR 2025\] Test-Time Visual In-Context Tuning](../../CVPR2025/llm_nlp/test-time_visual_in-context_tuning.md)

</div>

<!-- RELATED:END -->
