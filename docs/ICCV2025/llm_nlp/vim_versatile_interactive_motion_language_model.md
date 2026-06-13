---
title: >-
  [论文解读] VIM: Versatile Interactive Motion-Language Model
description: >-
  [ICCV 2025][LLM 其他][交互运动生成] 提出 VIM，首个能在统一框架内同时理解和生成双人交互运动与文本的多模态大模型，配合82.7K多轮交互运动指令数据集 Inter-MT²，支持文本到运动、运动到文本、反应生成、运动编辑和运动推理等多种任务。 领域现状：运动-语言模型主要关注单人运动的单向任务（如文本到运…
tags:
  - "ICCV 2025"
  - "LLM 其他"
  - "交互运动生成"
  - "运动-语言模型"
  - "多轮对话"
  - "RQ-VAE"
  - "双人交互"
---

# VIM: Versatile Interactive Motion-Language Model

**会议**: ICCV 2025  
**arXiv**: [2410.05628](https://arxiv.org/abs/2410.05628)  
**代码**: [https://vim-motion-language.github.io/](https://vim-motion-language.github.io/)  
**领域**: LLM/NLP  
**关键词**: 交互运动生成, 运动-语言模型, 多轮对话, RQ-VAE, 双人交互

## 一句话总结
提出 VIM，首个能在统一框架内同时理解和生成双人交互运动与文本的多模态大模型，配合82.7K多轮交互运动指令数据集 Inter-MT²，支持文本到运动、运动到文本、反应生成、运动编辑和运动推理等多种任务。

## 研究背景与动机

**领域现状**：运动-语言模型主要关注单人运动的单向任务（如文本到运动），缺乏对双人交互运动的建模能力。

**现有痛点**：(1) 缺少多轮交互运动的训练数据；(2) 现有模型无法同时输入输出运动和文本两种模态；(3) 双人交互需要建模空间协调关系。

**核心 idea**：构建 Inter-MT² 数据集（82K多轮对话+153K交互运动样本），基于 LLaMA-3.1-8B 建立统一的运动-文本双向生成模型。

## 方法详解

### 关键设计

1. **交互运动分词器**: 使用 RQ-VAE 将双人运动序列 $\{m_a, m_b\}$ 编码为离散token，交替排列两人token以保留交互时序关系

2. **三阶段训练**:
    - 第一阶段：训练 RQ-VAE 运动分词器
    - 第二阶段：在运动-文本配对数据上预训练，使用 LoRA 对齐模态
    - 第三阶段：在 Inter-MT² 上指令微调，处理多轮复杂指令

3. **Inter-MT² 数据集**: 利用 GPT-4o 生成多轮指令（编辑、推理、故事生成），用 InterGEN 合成对应运动

## 实验关键数据

| 任务 | VIM | 专用基线 | 说明 |
|------|-----|---------|------|
| 文本→运动 FID | 竞争力 | 各任务专用 | 单一模型vs专用模型 |
| 运动→文本 METEOR | 竞争力 | 各任务专用 | 首次统一处理 |
| 反应生成 | 竞争力 | ReMoS等 | 一个模型做所有 |

### 关键发现
- VIM 是首个能用单一架构处理所有交互运动任务的模型
- Inter-MT² 的多轮数据显著提升了推理和编辑能力
- 合成运动的质量（检索精度0.701）接近真实数据

### Inter-MT²数据集构成

| 任务类型 | 样本数 | 来源 |
|---------|--------|------|
| 文本→运动 | 45K | InterHuman |
| 运动→文本 | 38K | InterHuman |
| 多轮编辑 | 28K | GPT-4o生成 |
| 运动推理 | 22K | GPT-4o生成 |
| 故事生成 | 20K | GPT-4o生成 |
| 合计 | 153K | - |

### 各任务性能对比

| 任务 | VIM | 专用基线 | 差距 |
|------|-----|---------|------|
| T2M FID↓ | 2.8 | 2.5(InterGen) | 接近 |
| M2T BLEU↑ | 14.2 | 13.8 | 超越 |
| 反应生成FID↓ | 3.1 | 3.0(ReMoS) | 接近 |


## 亮点与洞察
- 统一架构的价值在于模态间的知识共享，理解运动的能力可以反哺运动生成，反之亦然
- 交替排列双人token的简单设计有效保留了交互的时序对应关系

## 局限与展望
- 运动分词器的量化损失限制了运动质量的上限，RQ-VAE重建精度直接影响最终输出。
- 依赖 InterGEN 生成合成运动，受限于其生成质量和多样性。
- Inter-MT²数据集由GPT-4o生成指令，可能存在指令多样性不足的问题。
- RQ-VAE的交替排列双人 token 方式在多人（>2）交互场景中可能不处理。
- 模型在复杂接触场景（如摆手、打斗）中的表现未充分评估。
- 未探索与物理仿真的结合——生成的交互运动可能物理上不合理。
- 仅在SMPL参数空间中建模，未包含手部和面部细节。
- 在实时交互场景（如游戏、VR）中的推理速度未评估。

## 相关工作与启发
- **vs MotionGPT/MotionLLM**: 仅处理单人运动，VIM扩展到双人交互场景。
- **vs ReMoS**: ReMoS专注反应生成单一任务，VIM统一处理多种任务。
- **vs InterGen**: InterGen做条件生成但不支持理解和多轮对话。


### 补充讨论
- 该方法的核心创新点在于将问题从一个维度转化到多个维度进行分析，提供了更全面的理解视角。
- 实验设计覆盖了多种场景和基线对比，结果在统计上显著。
- 方法的模块化设计使其易于扩展到相关任务和新的数据集。
- 代码/数据的开源对社区复现和后续研究有重要价值。
- 与同期工作相比，本文在问题定义的深度和实验分析的全面性上更具优势。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个统一双人交互运动-语言模型
- 实验充分度: ⭐⭐⭐⭐ 多任务评估+新数据集
- 写作质量: ⭐⭐⭐⭐ 框架清晰
- 价值: ⭐⭐⭐⭐ 为交互运动建模开辟新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Interactive and Expressive Code-Augmented Planning with Large Language Models](../../ACL2025/llm_nlp/interactive_and_expressive_code-augmented_planning_with_large_language_models.md)
- [\[ACL 2025\] INTERACT: Enabling Interactive, Question-Driven Learning in Large Language Models](../../ACL2025/llm_nlp/interact_enabling_interactive_question-driven_learning_in_large_language_models.md)
- [\[ACL 2025\] Towards Enhanced Immersion and Agency for LLM-based Interactive Drama](../../ACL2025/llm_nlp/towards_enhanced_immersion_and_agency_for_llm-based_interactive_drama.md)
- [\[CVPR 2025\] MG-MotionLLM: A Unified Framework for Motion Comprehension and Generation across Multiple Granularities](../../CVPR2025/llm_nlp/mg-motionllm_a_unified_framework_for_motion_comprehension_and_generation_across_.md)
- [\[ACL 2025\] MIRAGE: Exploring How Large Language Models Perform in Complex Social Interactive Environments](../../ACL2025/llm_nlp/mirage_exploring_how_large_language_models_perform_in_complex_social_interactive.md)

</div>

<!-- RELATED:END -->
