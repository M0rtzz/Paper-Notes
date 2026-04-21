---
title: >-
  [论文解读] Detecting Referring Expressions in Visually Grounded Dialogue with Autoregressive Language Models
description: >-
  [ACL 2025][LLM/NLP][mention detection] 本文将视觉对话中的指称表达检测建模为自回归 token 预测任务，通过对 Llama 3.1-8B 进行参数高效微调 (QLoRA)，证明仅使用文本上下文即可有效检测视觉对话中的 mention span，在 AGOS 和 PhotoBook 数据集上 F1 达 0.90 和 0.94。
tags:
  - ACL 2025
  - LLM/NLP
  - mention detection
  - referring expressions
  - visually grounded dialogue
  - 自回归语言模型
  - 参数高效微调
---

# Detecting Referring Expressions in Visually Grounded Dialogue with Autoregressive Language Models

**会议**: ACL 2025  
**arXiv**: [2506.21294](https://arxiv.org/abs/2506.21294)  
**代码**: [GitHub](https://github.com/willemsenbram/mention-detection-vgd)  
**领域**: 对话理解 / 视觉-语言  
**关键词**: mention detection, referring expressions, visually grounded dialogue, 自回归语言模型, 参数高效微调  

## 一句话总结

本文将视觉对话中的指称表达检测建模为自回归 token 预测任务，通过对 Llama 3.1-8B 进行参数高效微调 (QLoRA)，证明仅使用文本上下文即可有效检测视觉对话中的 mention span，在 AGOS 和 PhotoBook 数据集上 F1 达 0.90 和 0.94。

## 研究背景与动机

- **核心问题**: 在视觉情境对话中，说话者经常用词语或短语引用视觉场景中的物体（即指称表达 / referring expressions）。有效检测这些 mention 是后续指代消解和视觉定位的前提。
- **传统方法局限**: 早期基于规则 + 依存分析的方法需要大量特征工程；BERT 类编码器模型虽有效但框架为序列标注，不够灵活。目前基于自回归 LLM 的生成式信息抽取尚未应用于视觉情境对话中的 mention 检测。
- **关键研究兴趣**: 纯文本上下文能在多大程度上支撑本质上是多模态的任务？对话历史对 mention 检测性能的影响如何？

## 方法详解

### 整体框架

将 mention 检测建模为**生成式复述任务**：给定当前话语和对话历史，模型自回归地生成当前话语的副本，但在 mention span 的起止位置插入边界标记 `>>` 和 `<<`。例如输入 "I have a dog" → 输出 "I have >>a dog<<"。

### 关键设计

1. **对话历史条件化生成**: 生成目标 $u_i' = f(u_i, H)$，其中 $H = (u_{i-h}, ..., u_{i-1})$ 是可配置长度的历史消息。通过实验比较不同上下文窗口大小 (0, 3, 7, 19 条历史消息) 的影响。

2. **参数高效微调**: 使用 Llama 3.1-8B 的 QLoRA (4-bit 量化 + LoRA) 微调，在 AGOS（15 段对话，1486 个 mention）和 PhotoBook（50 段对话，2111 个 mention）两个小规模数据集上训练。

3. **span 边界标记设计**: 在 tokenizer 词表中添加特殊的 mention 起止标记，使模型在生成过程中自然地分割 mention span，无需额外的 CRF 或序列标注层。

### 评估方案

交叉验证评估数据集内性能 + 跨数据集迁移测试（AGOS 训练、PB 测试，反之亦然）+ 与 NP 提取基线和 BERT 序列标注基线对比。

## 实验

### 主实验结果 (数据集内交叉验证)

| 模型 | 上下文窗口 | AGOS F1 | PB-GOLD F1 |
|------|:---:|:---:|:---:|
| Llama 3.1-8B | 0 | .863 | .930 |
| Llama 3.1-8B | 3 | .892 | .930 |
| Llama 3.1-8B | 7 | .900 | .937 |
| Llama 3.1-8B | 19 | **.902** | **.940** |
| NP 基线 | - | 较低 | 较低 |

### 跨数据集迁移

| 训练集 → 测试集 | F1 |
|------|:---:|
| AGOS → PB | 性能下降但仍合理 |
| PB → AGOS | 迁移效果有限 |

跨数据集迁移存在挑战，因为两个数据集的 mention 分布特征不同（AGOS 中 17.94% 消息含 >1 个 mention，PB 仅 1.95%）。

### 消融：对话历史的影响

| 历史窗口 | AGOS F1 变化 | PB F1 变化 |
|------|:---:|:---:|
| 0 → 3 | +0.029 | +0.000 |
| 3 → 7 | +0.008 | +0.007 |
| 7 → 19 | +0.002 | +0.003 |

### 关键发现

- 仅使用文本上下文即可达到较高的 mention 检测性能 (F1 > 0.90)，凸显了语言上下文的信息量
- 对话历史一致性地提升 AGOS 性能，但对 PB 帮助较小——因 PB 的 mention 更多是独立描述性表达
- 小数据集 + 参数高效微调 + 中等规模 LLM 的组合已经足够有效
- 跨数据集迁移存在gap，说明任务导向对话的指称语言具有领域特异性
- 作为纯文本方法，在需要视觉信息才能判断指称性的边界案例上存在根本局限

## 亮点

- 将 mention 检测优雅地转化为"带标注的复述"生成任务，无需序列标注架构
- 首次将自回归 LLM + 生成式信息抽取应用于视觉情境对话的 mention 检测
- 清晰地分析了文本 vs 多模态方法的能力边界，讨论坦诚

## 局限性

- 纯文本方法无法处理需要视觉信息的歧义 mention（如"那个"是否指代图片中的物体）
- 数据集规模小（15 和 50 段对话），结论的泛化性需更多验证
- 仅测试 Llama 3.1-8B，未与其他规模/架构的 LLM 对比
- 仅检测 mention span，不涉及指代消解或视觉定位
- 任务为粗粒度 mention 检测，未区分 mention 类型

## 相关工作

- **Mention Detection**: Lee et al. (2013); Devlin et al. (2019) — 从规则到 BERT 的演进
- **生成式信息抽取**: Cao et al. (2021); Zhang et al. (2025) — 将结构化预测建模为自回归生成
- **视觉情境对话**: PhotoBook (Haber et al., 2019); AGOS (Willemsen et al., 2022) — 引导参与者对图像进行指称的对话任务
- **参数高效微调**: LoRA (Hu et al., 2022); QLoRA (Dettmers et al., 2023) — 低资源下微调大模型

## 评分

| 维度 | 分数 |
|------|:---:|
| 创新性 | ★★★☆☆ |
| 实用性 | ★★★☆☆ |
| 实验充分度 | ★★★★☆ |
| 写作质量 | ★★★★☆ |
| 总评 | ★★★☆☆ |

<!-- RELATED:START -->

## 相关论文

- [Beyond Dialogue: A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model](beyond_dialogue_a_profile-dialogue_alignment_framework_towards_general_role-play.md)
- [Deontological Keyword Bias: The Impact of Modal Expressions on Normative Judgments of Language Models](deontological_keyword_bias.md)
- [Towards Geo-Culturally Grounded LLM Generations](geocultural_grounded_llm.md)
- [B-score: Detecting biases in large language models using response history](../../ICML2025/llm_nlp/b-score_detecting_biases_in_large_language_models_using_response_history.md)
- [LLM-Powered Test Case Generation for Detecting Bugs in Plausible Programs](llm_test_case_gen_bugs.md)

<!-- RELATED:END -->
