---
title: >-
  [论文解读] Mitigate Position Bias in LLMs via Scaling a Single Hidden States Channel
description: >-
  [ACL 2025][LLM/NLP][position bias] 发现 LLM 隐状态中存在编码绝对位置信息的特定通道（positional hidden states），通过缩放这单一通道即可缓解"lost in the middle"位置偏差，在多文档 QA 基准上提升高达 15.2%，且不影响模型其他能力。
tags:
  - ACL 2025
  - LLM/NLP
  - position bias
  - lost in the middle
  - hidden states
  - positional channel
  - 注意力机制
---

# Mitigate Position Bias in LLMs via Scaling a Single Hidden States Channel

**会议**: ACL 2025  
**arXiv**: [2406.02536](https://arxiv.org/abs/2406.02536)  
**代码**: https://aka.ms/PositionalHidden  
**领域**: LLM/NLP  
**关键词**: position bias, lost in the middle, hidden states, positional channel, attention

## 一句话总结
发现 LLM 隐状态中存在编码绝对位置信息的特定通道（positional hidden states），通过缩放这单一通道即可缓解"lost in the middle"位置偏差，在多文档 QA 基准上提升高达 15.2%，且不影响模型其他能力。

## 研究背景与动机

**领域现状**：长上下文 LLM 仍存在"lost in the middle"偏差——中间位置的关键信息被忽略。

**现有痛点**：现有缓解方法从数据分布（FILM）或位置编码（Ms-PoE）入手，但忽略了隐状态中也携带位置信息。

**核心矛盾**：因果注意力掩码会在隐状态中注入位置信息，这是位置偏差的另一来源——但此前无人研究。

**本文目标** 找到隐状态中编码位置信息的具体通道，并通过缩放它来缓解偏差。

**切入角度**：发现 hidden states 的某些通道值与 token 位置呈单调关系，定义为"positional channels"。

**核心 idea**：只修改隐状态的一个通道就能显著缓解位置偏差——这是 LLM 位置偏差的一个此前未知的来源。

## 方法详解

### 整体框架
发现注意力权重中的 U 形偏差 -> 证明隐状态中的位置信息也导致偏差 -> 识别 positional channels（值与位置单调相关）-> 设计缩放算法 -> 只影响最后 token 的注意力查询。

### 关键设计

1. **Positional Channel 识别**

    - 启发式搜索：沿 hidden size 轴找值与位置呈单调+平滑关系的通道
    - 用校准数据集选择最优通道
    - 设计动机：精准定位位置信息的载体

2. **缩放方法**

    - 对选定通道的值进行缩放（减小其幅度）
    - 只影响最后 token 对其他 token 的注意力计算
    - 设计动机：消除位置偏差同时避免干扰模型其他功能

3. **注意力修改算法**

    - 缩放后的隐状态只在最后 token 的查询中使用
    - 其他 token 之间的注意力不受影响
    - 设计动机：最后 token 的注意力决定了检索哪些信息用于回答

## 实验关键数据

### 主实验 — NQ 多文档 QA

| 方法 | 准确率 | 提升 |
|------|--------|------|
| 原始模型 | ~50% | 基线 |
| FILM (SFT) | ~58% | +8% |
| Ms-PoE | ~55% | +5% |
| **Scale Positional HS** | **~65%** | **+15.2%** |

### 跨模型泛化

| 模型 | 提升 |
|------|------|
| LLaMA-2-7B | +12% |
| Mistral-7B | +15% |
| Gemma-7B | +10% |
| Qwen-7B | +8% |
| MPT-7B (Alibi) | +6% |

### 副作用评估

| 评估 | 结果 |
|------|------|
| 时间排序任务 | 无显著下降 |
| MMLU | 无显著下降 |

### 关键发现
- **只改一个通道就提升 15.2%**——极其轻量的干预
- **跨模型泛化**：RoPE/Alibi 等不同位置编码的模型都有效
- **不需要训练**：纯推理时的干预
- **位置偏差有两个来源**：位置编码 + 隐状态中的位置信息（新发现）
- **positional channels 可视化**证实其值与绝对位置高度线性相关

## 亮点与洞察
- **发现 positional hidden states**是论文最重要的贡献——证明了因果注意力掩码在隐状态中注入了位置信息
- **单通道干预**的极简方法论令人印象深刻——修改 4096 维中的 1 维就有 15% 提升
- **对"lost in the middle"问题提供了新的理解视角和解决路径**

## 局限与展望
- 最优通道需要通过搜索找到，不同模型可能不同
- 仅在 7B 规模模型上验证
- 改进方向：更大模型验证、动态通道选择

## 相关工作与启发
- **vs FILM (An et al.)**：FILM 需要 SFT 数据，本文无需训练
- **vs Ms-PoE (Zhang et al.)**：Ms-PoE 修改位置编码，本文修改隐状态——互补方向
- **vs CogSteer**：CogSteer 发现中间层重要，本文发现特定通道重要——不同粒度的发现

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次发现 positional hidden states 是位置偏差的来源
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型×多任务×多位置编码类型
- 写作质量: ⭐⭐⭐⭐⭐ 可视化和分析清晰
- 价值: ⭐⭐⭐⭐⭐ 对长上下文 LLM 有直接实用改进

<!-- RELATED:START -->

## 相关论文

- [Veracity Bias and Beyond: Uncovering LLMs' Hidden Beliefs in Problem-Solving Reasoning](veracity_bias_llm_hidden_beliefs.md)
- [On the Role of Hidden States of Modern Hopfield Network in Transformer](../../NeurIPS2025/llm_nlp/on_the_role_of_hidden_states_of_modern_hopfield_network_in_transformer.md)
- [Computation Mechanism Behind LLM Position Generalization](computation_mechanism_behind_llm_position_generalization.md)
- [AutoGUI: Scaling GUI Grounding with Automatic Functionality Annotations from LLMs](autogui_scaling_gui_grounding_with_automatic.md)
- [Attention Speaks Volumes: Localizing and Mitigating Bias in Language Models](attention_speaks_volumes_localizing_and_mitigating_bias_in_language_models.md)

<!-- RELATED:END -->
