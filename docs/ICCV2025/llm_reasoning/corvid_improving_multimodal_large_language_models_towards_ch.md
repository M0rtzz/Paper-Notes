---
title: >-
  [论文解读] Corvid: Improving Multimodal Large Language Models Towards Chain-of-Thought Reasoning
description: >-
  [ICCV 2025][LLM推理][MLLM] 提出Corvid MLLM，通过混合视觉编码器(SigLIP+ConvNeXt)+GateMixer门控连接器+MCoT-Instruct-287K高质量CoT数据+推理时自验证策略，三阶段训练实现超越同规模o1-like模型的CoT推理能力。
tags:
  - ICCV 2025
  - LLM推理
  - MLLM
  - CoT推理
  - 混合视觉编码器
  - 推理时缩放
  - 自验证
---

# Corvid: Improving Multimodal Large Language Models Towards Chain-of-Thought Reasoning

**会议**: ICCV 2025  
**arXiv**: [2507.07424](https://arxiv.org/abs/2507.07424)  
**代码**: [Project Page](https://mm-vl.github.io/corvid)  
**领域**: 多模态推理 / CoT推理  
**关键词**: MLLM, CoT推理, 混合视觉编码器, 推理时缩放, 自验证  

## 一句话总结

提出Corvid MLLM，通过混合视觉编码器(SigLIP+ConvNeXt)+GateMixer门控连接器+MCoT-Instruct-287K高质量CoT数据+推理时自验证策略，三阶段训练实现超越同规模o1-like模型的CoT推理能力。

## 研究背景与动机

- **领域现状**：MLLM在感知任务上表现优秀，但在需要深层推理的复杂任务（数学、科学问题）上仍有不足。
- **现有痛点**：(1)高质量多模态CoT数据缺乏——人工简短、AI生成有噪声；(2)视觉表示不充分导致推理基于错误证据；(3)推理时过度/不足推理的平衡问题。
- **核心矛盾**：简单任务不需要CoT反而可能被CoT幻觉误导 vs 复杂任务必须深度推理。
- **本文目标**：增强MLLM的CoT推理能力，同时解决数据、架构和推理策略三个层面的问题。
- **切入角度**：数据侧标准化多源CoT数据，架构侧增强视觉表示和跨模态对齐，推理侧自适应选择推理深度。
- **核心 idea**：高质量CoT数据 + 充分视觉表示 + 自适应推理深度 = 强CoT推理能力。

## 方法详解

### 整体框架

SigLIP ViT + ConvNeXt-XXL双编码器→GateMixer门控融合→Llama3-8B LLM→三阶段训练（对齐预训练→CoT增强微调→纯CoT指令调优）。

### 关键设计

**设计1：GateMixer门控连接器**
- **功能**：融合ViT语义特征和CNN空间特征为LLM友好的token。
- **核心思路**：两路线性映射到统一空间后，用sigmoid门控机制逐元素混合：$h = (1-\alpha) \odot h_v + \alpha \odot h_c$，再加可学习prefix token。
- **设计动机**：保持视觉token数量不变的同时实现双编码器特征的自适应融合。

**设计2：MCoT-Instruct-287K高质量CoT数据**
- **功能**：从多源公开推理数据集中标准化提取高质量CoT指令数据。
- **核心思路**：分别对人工和AI生成的CoT进行GPT辅助标准化，统一格式、去噪去重。覆盖数学、科学、逻辑等多类推理。
- **设计动机**：人工CoT准确但简短，AI CoT详细但有噪声，分别处理取长补短。

**设计3：推理时自验证策略**
- **功能**：自适应决定是否使用CoT推理。
- **核心思路**：同时生成直接回答和CoT回答，用跨模态表示相似度S和模型置信度C的加权平均来选择最终答案。C定义为归一化困惑度。
- **设计动机**：简单任务用直接回答避免过度推理引入的上下文丢失和幻觉。

### 损失函数/训练策略

三阶段：Stage1冻结编码器+LLM训练GateMixer（对比正则化+文本生成损失）；Stage2联合训练GateMixer+LLM；Stage3纯CoT指令调优。

## 实验关键数据

### 主实验

| 模型 | MMStar | MathVista | AI2D | MMBench | HallusionBench | 平均 |
|------|--------|-----------|------|---------|----------------|------|
| LLaVA-o1 | 58.1 | 56.1 | 78.8 | 75.6 | 48.2 | 63.1 |
| LlamaV-o1 | 59.5 | 54.4 | 81.2 | 79.9 | 63.5 | 67.3 |
| **Corvid-o1-8B** | **65.2** | **72.0** | **85.0** | **82.9** | 54.0 | **67.5** |

### 消融实验

| 组件 | MathVista |
|------|-----------|
| 无混合编码器（仅SigLIP） | 65.3 |
| 无GateMixer（拼接） | 67.8 |
| 无自验证 | 69.5 |
| 完整Corvid | **72.0** |

### 关键发现

1. MathVista上72.0%的成绩大幅领先同规模o1-like模型，说明视觉表示质量对数学推理至关重要。
2. 自验证策略在简单benchmark上提升更显著，验证了过度推理的危害。
3. 三阶段训练中每个阶段都有不可替代的贡献。

## 亮点与洞察

1. GateMixer用简洁的门控机制解决双编码器融合问题，比拼接或交叉注意力更高效。
2. 自验证策略不需要外部验证器，模型自身判断推理深度很优雅。
3. MCoT-Instruct数据的标准化流程可复用于其他推理数据构建。

## 局限与展望

1. 自验证的阈值选择需要调优，可能对不同任务不通用。
2. 仅在8B规模验证，更大规模效果未知。
3. ConvNeXt-XXL较大，推理开销不可忽略。

## 相关工作与启发

- LLaVA-o1和Mulberry-o1通过beam search选择推理路径，Corvid的自验证更轻量。
- 启发：好的CoT数据质量可能比推理算法更重要，数据工程值得投入。

## 评分

| 维度 | 评分 |
|------|------|
| 创新性 | ★★★★☆ |
| 实用性 | ★★★★☆ |
| 实验充分性 | ★★★★★ |
| 写作清晰度 | ★★★★☆ |

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing](../../NeurIPS2025/llm_reasoning/thinksound_chain-of-thought_reasoning_in_multimodal_large_language_models_for_au.md)
- [\[ICCV 2025\] Unsupervised Visual Chain-of-Thought Reasoning via Preference Optimization](unsupervised_visual_chain-of-thought_reasoning_via_preference_optimization.md)
- [\[ACL 2025\] MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification](../../ACL2025/llm_reasoning/mm-verify_enhancing_multimodal_reasoning_with_chain-of-thought_verification.md)
- [\[ACL 2025\] Improve Vision Language Model Chain-of-thought Reasoning](../../ACL2025/llm_reasoning/improve_vlm_cot_reasoning.md)
- [\[ACL 2025\] Can Large Language Models Detect Errors in Long Chain-of-Thought Reasoning?](../../ACL2025/llm_reasoning/can_large_language_models_detect_errors_in_long_chain-of-thought_reasoning.md)

<!-- RELATED:END -->
