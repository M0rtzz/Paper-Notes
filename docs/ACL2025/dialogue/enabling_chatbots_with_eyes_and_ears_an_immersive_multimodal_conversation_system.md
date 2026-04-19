---
title: >-
  [论文解读] Enabling Chatbots with Eyes and Ears: An Immersive Multimodal Conversation System
description: >-
  [ACL 2025] 本文提出赋予聊天机器人"眼睛和耳朵"的沉浸式多模态对话系统，构建了融合视觉与听觉的多会话多方对话数据集 M3C，并设计了包含对话模块和多模态记忆检索模块的对话模型，实现了多说话者共享视听体验的动态长期对话。
tags:
  - ACL 2025
---

# Enabling Chatbots with Eyes and Ears: An Immersive Multimodal Conversation System

**会议**: ACL 2025  
**arXiv**: [2506.00421](https://arxiv.org/abs/2506.00421)  
**代码**: 无  
**领域**: 多模态VLM  

## 一句话总结

本文提出赋予聊天机器人"眼睛和耳朵"的沉浸式多模态对话系统，构建了融合视觉与听觉的多会话多方对话数据集 M3C，并设计了包含对话模块和多模态记忆检索模块的对话模型，实现了多说话者共享视听体验的动态长期对话。

## 背景与动机

1. **现有多模态对话重"眼"轻"耳"**：当前研究主要聚焦图像相关对话（视觉对话、图像指令等），给聊天机器人装上了"眼睛"，但听觉（"耳朵"）方面严重不足，缺乏同时整合视觉和听觉的方案。

2. **静态交互限制了对话自然度**：现有范式中聊天机器人接收共享图像后回答问题，属于"讨论模态"而非"自然融入模态"的静态交互模式，无法捕捉真实人类沟通的动态实时特性。

3. **多方+多会话场景探索不足**：虽然多模态已在多方对话和多会话对话中有所探索，但受限于特定任务约束，难以在动态自然对话中无缝整合。

4. **缺乏共享时空体验的数据集**：现有数据集（如 PhotoChat、DialogCC）中说话者并非在同一空间同时体验视听输入，不符合真实世界多人共处场景。

5. **长期记忆机制的重要性**：真实对话中人们会回忆先前的共同经历，但现有模型缺乏有效的多模态记忆存储与检索机制来支持跨会话的连贯对话。

6. **模型自主多方交互的挑战**：要实现无需人类干预的多 agent 自主对话，模型需要判断何时轮到自己发言，这在多方对话中是一个未被充分解决的关键问题。

## 方法详解

### 数据集：M3C (Multimodal Multi-Session Multi-Party Conversation)

**数据规模与结构**：
- 54K 对话 episode（34K 训/8K 验/12K 测），共 2.5M 轮对话
- 每个 episode 包含 4 位说话者跨 3 个连续会话
- 每个会话中主说话者与 2 位不同搭档交互
- 每个会话包含 2 个多模态输入（视觉或听觉）
- 所有说话者共享同一时空环境中的视听体验

**数据构建流程**：
1. **模态结构化**：使用 COCO 图像（24K 张）和 AudioCaps/Clotho 音频（73K 条）作为种子，通过 GPT-4o mini 优化图像标注
2. **场景准备**：生成说话者信息、会话搭档、模态输入和时间间隔，基于 K-means 聚类（K=30）按位置标签分组相似模态
3. **对话与记忆生成**：逐会话生成对话，从主说话者视角创建记忆摘要，通过记忆链接连接相关元素
4. **质量过滤**：通过机器验证问题排除时空一致性不合格的 episode

### 模型架构

**基座模型**：Qwen2-VL-2B-Instruct，通过 CLAP + 线性适配器扩展音频理解能力。

**对话模块 (Dialogue Module)**：
- 负责对话生成、记忆生成和记忆链接三项任务
- 会话进行中基于历史对话和多模态输入生成回复
- 会话结束后构建整合了多模态感知的记忆单元
- 通过结构化记忆链接将新记忆与语义/感知相关的旧记忆显式关联

**检索模块 (Retrieval Module)**：
- 基于当前对话上下文从多模态记忆库中检索相关记忆
- 将整个会话（对话+感知模态）联合嵌入共享表示空间
- 使用余弦相似度度量记忆相关性：sim(c, m_i) = cos(E_c(c), E_m(m_i))
- 选取 Top-1 最相关记忆用于增强对话

**训练策略**：分阶段微调——先在视觉语言任务上微调（音频作为文本标注），再接入线性适配器处理原始音频。模型支持 model-to-model 对话，自主管理轮次。

## 实验结果

### 人工评估与机器评估（Table 2）

| 评估维度 | 人工评分 | 机器评分(o3-mini) |
|---------|---------|-----------------|
| **数据集质量** | | |
| 连贯性与一致性 | 4.81 | 4.99 |
| 可记忆性 | 4.63 | 4.99 |
| 模态对齐度 | 4.21 | 4.26 |
| 模态参与度 | 4.36 | 4.57 |
| 数据集总体 | 4.50 | 4.70 |
| **模型表现** | | |
| 自然度 | 4.34 | 4.68 |
| 沉浸感 | 4.14 | 4.56 |
| 可记忆性 | 4.35 | 4.46 |
| 模型总体 | 4.28 | 4.57 |

### 检索模块性能（Table 4）

| 模型 | 图像 R@1 | 图像 MRR | 音频 R@1 | 音频 MRR |
|------|---------|---------|---------|---------|
| Qwen2-VL-2B | 66.77 | 77.56 | - | - |
| LLaMA-3.2-11B-Vision | 72.41 | 78.90 | - | - |
| Qwen2-Audio-7B | - | - | 69.94 | 80.72 |
| **本文模型** | **92.99** | **95.06** | **92.83** | **94.78** |

**跨数据集对比（Table 3）**：GPT-4o-mini 和 Claude-3.5-Sonnet 在 M3C vs 其他数据集的"沉浸自然度"选择中，M3C 分别获得 81% 和 99% 的选择率。

**多方对话能力（Table 5）**：下一说话者预测准确率——本文模型 85.2% vs Qwen2-VL 基线 10.3%。

### 消融实验

- **音频标注 vs 原始音频**：装备音频适配器的模型能与听觉体验直接对齐，而基于标注的模型过度依赖标注文本内容
- **多模态记忆**：有检索器时模型能引用先前会话的相关细节（如"上次捡的贝壳"），无检索器则生成泛化回复
- **超越三会话**：虽然仅在三会话数据上训练，但通过独立记忆机制可支持更多会话的长期对话

## 亮点与创新

- **首个视听融合的多方多会话对话数据集**：M3C 是首个让所有说话者在共享时空中同步体验图像和音频的开放域对话数据集
- **多模态记忆检索**：通过结构化记忆链接和跨模态检索实现跨会话的连贯对话，记忆在存储时（而非检索时）建立关联
- **自主多方对话**：模型能自主判断轮次、决定模态出现时机，实现无人干预的多 agent 对话
- **检索性能大幅领先**：在图像和音频检索上 R@1 均超 92%，远超对比模型

## 局限性

- 音频数据集标注未像图像标注一样进行优化，可能影响音频沉浸质量
- 基座模型受限于 2B 参数的 VLM，缺乏原生视觉-音频-语言联合理解能力的大模型
- 数据集由 GPT-4o mini 生成而非真人对话，可能存在机器生成偏差
- 模型规模较小（2B），在更大模型上的表现和泛化性未验证

## 相关工作对比

| 对比方向 | 本文优势 |
|---------|---------|
| **PhotoChat / DialogCC / Stark** | 这些数据集仅支持图像模态且多为单方/单会话；M3C 同时覆盖图像+音频，支持多方多会话，GPT-4o-mini 评测中 81-99% 选择率证明沉浸度碾压 |
| **Audio Dialogues / MELD** | Audio Dialogues 仅支持音频 QA 任务；MELD 虽包含视听但为情感分析导向的短对话；M3C 是开放域长期多轮对话，视听协同且有共享体验设计 |
| **MiSC (Jang et al., 2024)** | MiSC 支持多会话多方但无模态输入；M3C 在此基础上扩展为多方+每会话多搭档+视听模态，大幅提升交互复杂度和真实感 |

## 评分

- ⭐⭐⭐⭐ 新颖性：视听融合的多方多会话设定有明确创新，多模态记忆检索设计合理
- ⭐⭐⭐⭐ 实验充分度：人工+机器+跨数据集多维评估，含消融和定量分析
- ⭐⭐⭐⭐ 实用价值：为构建更自然的多模态对话系统提供了数据集和建模范式
- ⭐⭐⭐⭐ 写作质量：结构清晰，案例丰富，数据集对比表格信息量大

<!-- RELATED:START -->

## 相关论文

- [Evolutionary Multimodal Reasoning via Hierarchical Semantic Representation for Intent Recognition](../../CVPR2026/dialogue/evolutionary_multimodal_reasoning_via_hierarchical_semantic_representation_for_i.md)
- [BI-MDRG: Bridging Image History in Multimodal Dialogue Response Generation](../../ECCV2024/dialogue/bi-mdrg_bridging_image_history_in_multimodal_dialogue_response_generation.md)
- [Dialogue Systems for Emotional Support via Value Reinforcement](dialogue_systems_for_emotional_support_via_value_reinforcement.md)
- [UniConv: Unifying Retrieval and Response Generation for Large Language Models in Conversations](uniconv_retrieval_response_gen.md)
- [Wizard of Shopping: Target-Oriented E-commerce Dialogue Generation with Decision Tree Branching](wizard_of_shopping_target-oriented_e-commerce_dialogue_generation_with_decision_.md)

<!-- RELATED:END -->
