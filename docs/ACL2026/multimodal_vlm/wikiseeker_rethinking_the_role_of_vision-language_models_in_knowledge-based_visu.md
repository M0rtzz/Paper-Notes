---
title: >-
  [论文解读] WikiSeeker: Rethinking the Role of Vision-Language Models in Knowledge-Based Visual Question Answering
description: >-
  [ACL 2026][多模态][知识型VQA] 提出 WikiSeeker，重新定义 VLM 在多模态 RAG 中的角色——从单纯的答案生成器转变为两个专门化智能体（Refiner 用 RL 训练重写查询、Inspector 验证检索上下文是否可靠），在 EVQA、InfoSeek、M2KR 三个基准上实现 SOTA。
tags:
  - ACL 2026
  - 多模态
  - 知识型VQA
  - 多模态RAG
  - 查询重写
  - 强化学习
  - 检索增强生成
---

# WikiSeeker: Rethinking the Role of Vision-Language Models in Knowledge-Based Visual Question Answering

**会议**: ACL 2026  
**arXiv**: [2604.05818](https://arxiv.org/abs/2604.05818)  
**代码**: https://github.com/zhuyjan/WikiSeeker (有)  
**领域**: 多模态VLM  
**关键词**: 知识型VQA, 多模态RAG, 查询重写, 强化学习, 检索增强生成

## 一句话总结

提出 WikiSeeker，重新定义 VLM 在多模态 RAG 中的角色——从单纯的答案生成器转变为两个专门化智能体（Refiner 用 RL 训练重写查询、Inspector 验证检索上下文是否可靠），在 EVQA、InfoSeek、M2KR 三个基准上实现 SOTA。

## 研究背景与动机

**领域现状**：多模态检索增强生成（RAG）是知识型视觉问答（KB-VQA）的主流范式——检索外部知识库中的相关文档，与输入查询拼接后送入生成模型产出答案。

**现有痛点**：(1) **纯视觉检索**：大多数方法仅使用查询图片作为检索键，忽略了用户文本查询中的语义信息，当视觉内容模糊时检索效果差；(2) **VLM 角色错位**：VLM 通常仅被用作最终的答案生成器，但实验表明 VLM 在从检索上下文中提取答案方面反而不如纯文本 LLM——图像 token 在答案提取阶段往往是噪声而非有用信号。

**核心矛盾**：VLM 的视觉理解能力在检索和验证阶段有价值（理解图中是什么实体、判断检索结果是否匹配），但在答案提取阶段反而是负担（视觉 token 干扰文本阅读理解）。

**本文目标**：重新设计 VLM 在多模态 RAG 中的角色定位，充分利用 VLM 的视觉理解能力来改善检索和验证，而将答案提取交给更擅长的纯文本 LLM。

**切入角度**：作者通过实验发现，当检索上下文中正确信息占比增加时，纯文本 LLM 的 VQA 性能反而超过带图像输入的 VLM（如 Ratio=1.0 时 Qwen 93.45% vs QwenVL(I+T) 88.46%）。

**核心 idea**：将 VLM 重定位为 Refiner（用视觉线索重写查询提升检索）和 Inspector（验证检索上下文可靠性并路由决策），答案生成交给纯文本 LLM。

## 方法详解

### 整体框架

WikiSeeker 包含三个阶段：(1) **检索**：VLM Refiner 扩展原始问题，多模态检索器（视觉+文本嵌入拼接）从知识库中检索候选文档；(2) **重排**：多模态重排器筛选最相关段落；(3) **生成**：VLM Inspector 评估检索上下文是否充分——通过则路由到纯文本 LLM 生成答案，不通过则 VLM 用内部知识直接回答。

### 关键设计

1. **VLM 作为 Refiner（基于 RL 的查询重写）**:

    - 功能：利用视觉线索重写和扩展用户的简短查询，生成更具信息量的检索查询
    - 核心思路：使用 Qwen2.5-VL-3B-Instruct 作为 Refiner，通过 GRPO（Group Relative Policy Optimization）进行强化学习训练。模型先生成 CoT 推理（`<think>` 标签），再输出重写后的查询（`<answer>` 标签）。奖励函数由两部分组成：(1) 格式奖励——检查输出是否符合 XML 格式要求；(2) 检索奖励——用重写后的查询进行检索，根据正确实体的命中排名给予离散奖励（top-5 得 +4，top-200 内递减，未命中 -2.5）
    - 设计动机：KB-VQA 中用户查询通常简短且抽象，直接用于检索噪声大。RL 训练让 Refiner 自主发现最优查询重写策略，无需昂贵的人工标注查询对

2. **多模态密集检索（加权拼接策略）**:

    - 功能：同时利用视觉和文本信息进行检索
    - 核心思路：知识库构建为 <图像, 段落> 对，用 EVA-CLIP-8B 编码视觉、Qwen3-Embedding-0.6B 编码文本，拼接为统一向量。检索时用加权拼接：$\mathbf{v}_q = \text{Concat}[\alpha \cdot \Phi_{vis}(I_q), (1-\alpha) \cdot \Phi_{text}(T_q)]$，超参数 $\alpha$ 控制视觉和文本特征的相对重要性
    - 设计动机：纯视觉检索忽略文本语义，拼接策略让两种模态同时参与检索，$\alpha$ 提供灵活的模态平衡控制

3. **VLM 作为 Inspector（解耦生成策略）**:

    - 功能：验证检索上下文的可靠性并动态路由答案生成
    - 核心思路：Inspector（VLM）接收图像、问题和重排后的段落，输出判断 $s \in \{\text{PASS}, \text{FAIL}\}$ 和内部知识答案 $A_{internal}$。PASS 则将重写查询+检索上下文送到纯文本 LLM（如 LLaMA/Qwen）生成答案；FAIL 则使用 VLM 的内部知识答案
    - 设计动机：实验证明 VLM 在利用检索上下文生成答案时不如纯文本 LLM（视觉 token 是噪声），但 VLM 的视觉理解能力使其擅长判断检索结果是否与图像一致。解耦策略让各组件做最擅长的事

### 损失函数 / 训练策略

Refiner 用 GRPO 训练，总奖励 $r_i = r_{retrieval}(o_i) + r_{format}(o_i)$。检索奖励基于命中排名的离散映射（top-5: +4, top-200: +0.1, miss: -2.5），格式奖励检查 XML 标签正确性（+1/-4）。训练集每个基准 7000 样本，按命中排名分层采样。

## 实验关键数据

### 主实验

EVQA 和 InfoSeek 检索结果（R@1）：

| 方法 | EVQA R@1 | EVQA R@20 | InfoSeek R@1 | InfoSeek R@20 |
|------|---------|----------|-------------|--------------|
| EchoSight | 36.5 | 48.8 | 53.2 | 77.9 |
| OMGM | 42.8 | 58.7 | 64.0 | 84.8 |
| WikiSeeker (w/o Refiner) | 28.0 | 43.4 | 53.5 | 78.5 |
| WikiSeeker (w. Refiner) | **44.1** | **62.3** | **67.0** | **87.7** |

Refiner 将 EVQA R@1 从 28.0 提升到 44.1（+57.5%），超越所有基线。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| w/o Refiner | R@1 28.0 (EVQA) | 基础多模态检索 |
| w. Refiner | R@1 44.1 (EVQA) | 查询重写大幅提升检索 |
| VLM 生成 vs LLM 生成 | 88.46% vs 93.45% (Ratio=1.0) | 有可靠上下文时 LLM 更优 |
| w/o Inspector | 下降 | 不可靠上下文时 LLM 会被误导 |

### 关键发现

- VLM 在答案生成阶段确实不如纯文本 LLM：当检索上下文中正确信息占比增加时（Ratio=0.3→1.0），LLM 的优势越来越明显
- RL 训练的 Refiner 效果远超 SFT：RL 让模型自动学会如何重写查询以最大化检索命中率
- Inspector 的路由策略在不可靠检索场景尤其重要——VLM 的内部知识在 FAIL 路径上补偿了检索失败
- M2KR 多任务基准上也取得 SOTA，证明方法的通用性

## 亮点与洞察

- **"VLM 在答案提取时不如 LLM"**这个实证发现非常重要且反直觉——原因是视觉 token 在已经检索到正确文本上下文后变成了噪声。这启示我们在 RAG 系统中应该"用对的模型做对的事"
- **用 RL 训练查询重写**是一个优雅的自监督方案——以检索命中排名作为奖励信号，无需人工标注重写查询对。GRPO 的组内相对优势估计避免了训练 critic 模型的额外开销
- Inspector 的双路径设计实现了检索增强和参数知识的优雅融合——不是简单地"总是用检索"或"总是用内部知识"，而是根据可靠性动态选择

## 局限与展望

- Inspector 的 PASS/FAIL 判断是硬决策，可能存在边界情况的误判
- Refiner 使用较小的 VLM（3B），更大模型可能产出更好的查询重写
- 知识库构建依赖 LLM 对长段落的摘要，摘要质量影响检索效果
- 仅在百科知识型 VQA 上验证，对常识推理型 VQA 的效果未知

## 相关工作与启发

- **vs EchoSight/OMGM**: 它们用 VLM 做答案生成+纯视觉检索。WikiSeeker 将 VLM 重定位为 Refiner+Inspector，答案生成交给 LLM，检索升级为多模态。在 EVQA R@1 上超 OMGM 1.3 个百分点
- **vs ReflectiVA**: ReflectiVA 引入反思机制判断是否需要外部知识，但仍用 VLM 生成答案。WikiSeeker 的解耦策略更根本地解决了 VLM 在答案提取阶段的噪声问题

## 评分

- 新颖性: ⭐⭐⭐⭐ VLM 角色重定位的洞察有价值，RL 训练 Refiner 的方案优雅，但整体框架是已有技术的巧妙组合
- 实验充分度: ⭐⭐⭐⭐⭐ 三个基准、多个消融、VLM vs LLM 的系统对比实验充分
- 写作质量: ⭐⭐⭐⭐ 动机和方法描述清晰，Table 2 的实验设计有说服力
- 价值: ⭐⭐⭐⭐ 对多模态 RAG 系统的 VLM 角色设计有直接指导意义

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] MAGIC-VQA: Multimodal and Grounded Inference with Commonsense Knowledge for Visual Question Answering](../../ACL2025/multimodal_vlm/magic-vqa_multimodal_and_grounded_inference_with_commonsense_knowledge_for_visua.md)
- [\[ICCV 2025\] ReasonVQA: A Multi-hop Reasoning Benchmark with Structural Knowledge for Visual Question Answering](../../ICCV2025/multimodal_vlm/reasonvqa_a_multi-hop_reasoning_benchmark_with_structural_knowledge_for_visual_q.md)
- [\[ACL 2026\] Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring](rethinking_jailbreak_detection_of_large_vision_language_models_with_representati.md)
- [\[ICLR 2026\] Meta-Adaptive Prompt Distillation for Few-Shot Visual Question Answering](../../ICLR2026/multimodal_vlm/meta-adaptive_prompt_distillation_for_few-shot_visual_question_answering.md)
- [\[AAAI 2026\] MacVQA: Adaptive Memory Allocation and Global Noise Filtering for Continual Visual Question Answering](../../AAAI2026/multimodal_vlm/macvqa_adaptive_memory_allocation_and_global_noise_filtering_for_continual_visua.md)

<!-- RELATED:END -->
