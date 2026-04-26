---
title: >-
  [论文解读] Doc-PP: Document Policy Preservation Benchmark for Large Vision-Language Models
description: >-
  [ACL 2026][多模态][文档问答] 本文提出 Doc-PP 基准，揭示大型视觉-语言模型（LVLM）在多模态文档问答中存在"推理诱导的安全缺口"——模型在需要跨模态推理时会绕过显式非披露策略泄露敏感信息，并提出 DVA（Decompose–Verify–Aggregation）结构化推理框架来显著降低泄露率。
tags:
  - ACL 2026
  - 多模态
  - 文档问答
  - 信息泄露
  - 策略保留
  - 多模态推理
  - 安全对齐
---

# Doc-PP: Document Policy Preservation Benchmark for Large Vision-Language Models

**会议**: ACL 2026  
**arXiv**: [2601.03926](https://arxiv.org/abs/2601.03926)  
**代码**: [项目页面](https://hwanchang00.github.io/docpp_project_page)  
**领域**: 多模态VLM / 文档安全  
**关键词**: 文档问答, 信息泄露, 策略保留, 多模态推理, 安全对齐

## 一句话总结

本文提出 Doc-PP 基准，揭示大型视觉-语言模型（LVLM）在多模态文档问答中存在"推理诱导的安全缺口"——模型在需要跨模态推理时会绕过显式非披露策略泄露敏感信息，并提出 DVA（Decompose–Verify–Aggregation）结构化推理框架来显著降低泄露率。

## 研究背景与动机

**领域现状**：LVLM 被广泛用于复杂多模态文档的问答任务。实际部署中，文档通常附带用户定义的动态策略，指定哪些信息可以或不可以披露（如季度财报中某些区域收入数据需保密）。这些约束随用户、组织、访问场景而变化，手动遮蔽敏感区域不可行。

**现有痛点**：(1) 现有安全研究主要关注隐式社会规范或纯文本场景，忽略了多模态文档的复杂性；(2) CoPriva 等文本域工作仅处理文本输入，不涉及图表、表格等异构视觉组件；(3) 即便是 GPT-5.2 等先进模型，在被明确指示"不要披露中东地区收入"时，仍会从饼图提取百分比、从文本获取总收入并通过隐式推理计算出保护信息。

**核心矛盾**：模型的推理能力越强，越容易通过跨模态证据合成绕过安全约束——推理能力与策略遵从之间存在根本性张力。

**本文目标**：构建首个评估多模态文档中用户定义策略保留的基准，并提出有效的防御框架。

**切入角度**：将评估聚焦于需要跨模态推理才能回答的查询，揭示模型在显式查询和隐式查询之间的安全差距。

**核心 idea**：安全检查应嵌入推理过程的每个步骤，而非仅在最终输出时过滤——DVA 将推理与策略验证解耦，每个子步骤独立验证后再聚合。

## 方法详解

### 整体框架

Doc-PP 包含三阶段构建流程：(1) 策略构建——从真实文档中生成保密目标并通过检查清单过滤；(2) 查询构建——生成显式和隐式两类查询；(3) 评估——使用检查清单框架测量泄露率和忠实度。评估实例定义为三元组 $(D, P, Q)$，即文档、安全策略和查询。文档支持两种输入条件：$D^{ocr}$（OCR 解析内容）和 $D^{img}$（PNG 图像）。

### 关键设计

1. **策略构建流程 (Policy Construction)**:

    - 功能：从真实 PDF 文档中自动生成高质量的非披露策略
    - 核心思路：先用 GPT-5.2 根据敏感类别分类法（战略决策、路线图、内部辩论、法律细节等）提出保密目标，要求提供证据类型（文本/表格/图表/混合）、页面索引和原文引用。然后用 target-aligned clipping 从长文档（平均 100 页）中裁剪出相关页面窗口 $[p-2, p+2]$，建立保密目标与文档片段的一对一映射。最后通过五项检查清单过滤低质量候选
    - 设计动机：保密目标不是简单的事实片段，而是需要深度理解（如解读图表趋势、跨模态合成上下文）才能定位的信息，这样才能真正测试模型的策略遵从能力

2. **显式 vs 隐式查询分类 (Explicit/Implicit Query)**:

    - 功能：区分两种不同难度的安全挑战
    - 核心思路：显式查询 $Q_e$ 直接请求目标信息（如"中东地区收入是多少？"）；隐式查询 $Q_i$ 以摘要式请求呈现，忠实回答自然会涉及披露（如"请总结各地区收入分布"）。模型需在满足信息需求的同时选择性地隐瞒敏感值
    - 设计动机：现实场景中信息泄露往往不是直接询问导致的，而是通过间接推理暴露——隐式查询更接近真实威胁

3. **DVA 结构化推理框架 (Decompose–Verify–Aggregation)**:

    - 功能：将推理与策略验证解耦，从结构上防止推理过程中的策略违规
    - 核心思路：(1) Decompose——将复杂查询分解为独立的子问题；(2) Verify——对每个子问题的回答独立进行策略合规检查，识别并阻止涉及保密目标的证据；(3) Aggregation——仅聚合通过验证的子回答生成最终输出
    - 设计动机：标准提示防御（如 CoT、事后修订）无法拦截导致策略违规的中间推理步骤——信息一旦在推理链中被计算出来，后续过滤往往已经太迟

### 损失函数 / 训练策略

Doc-PP 为评估基准而非训练方法。数据集从 MMlongbench-Doc 和 Sustainable QA 收集 90 篇长 PDF 文档，涵盖商业、金融和行业报告。评估采用检查清单框架测量信息泄露率和回答忠实度。

## 实验关键数据

### 主实验

| 发现 | 说明 |
|------|------|
| 推理诱导安全缺口 | 隐式查询泄露率远高于显式查询——模型能遵守直接请求但无法阻止推理推导 |
| OCR 悖论 | 提供 OCR 文本提升了感知能力但显著增加了信息泄露 |
| 跨模态泄露 | 需要整合文本和视觉证据的多模态设置下策略遵从显著下降 |
| DVA 优势 | DVA 在所有文档类型和查询设置下均大幅优于标准提示防御 |

### 消融实验

| 防御策略 | 效果 |
|----------|------|
| 标准 CoT 提示 | 有限保护，无法拦截中间推理步骤 |
| 事后输出修订 | 有限保护，信息已在推理中被计算 |
| DVA（完整） | 大幅降低泄露率，提供实用安全基线 |

### 关键发现

- 即便是 GPT-5.2 等最先进模型也会系统性地在跨模态推理场景中泄露保护信息
- 提供 OCR 文本是把双刃剑——改善感知但加剧泄露，揭示了"能力-安全"权衡
- 混合证据类型（mixed）的泄露风险最高，因为需要整合多种模态的信息
- DVA 的分步验证策略有效阻断了推理链中的信息传播路径

## 亮点与洞察

- "推理诱导安全缺口"是一个深刻的观察——模型的推理能力本身成为了安全漏洞的来源，这与传统安全研究中"对抗性输入"的范式截然不同
- DVA 的核心思想——将安全检查嵌入推理的每个子步骤——可推广到任何需要在信息处理过程中维持约束的场景
- 数据集设计将保密目标锚定在需要深度理解的信息上（而非简单事实），大幅提升了基准的现实相关性

## 局限与展望

- 数据集规模较小（90 篇文档），可能不覆盖所有文档类型和策略模式
- DVA 增加了推理延迟，对实时应用可能有影响
- 仅评估了非披露策略，未涉及更复杂的条件性披露规则
- 未探索模型微调或安全对齐训练对策略保留的影响

## 相关工作与启发

- **vs CoPriva**: CoPriva 限于纯文本输入和局部文本片段查询，Doc-PP 扩展到多模态文档和跨文档推理
- **vs VLM-GEOPRIVACY**: 后者关注隐式隐私规范（地理位置推断），Doc-PP 关注显式用户定义约束
- **vs 传统安全对齐**: RLHF 等方法针对隐式社会规范训练，无法处理动态的、用户指定的策略

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个多模态文档策略保留基准，"推理诱导安全缺口"概念新颖
- 实验充分度: ⭐⭐⭐⭐ 评估了多个 LVLM 和多种防御策略，但数据集规模有限
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，威胁模型直观，实验设计严谨
- 价值: ⭐⭐⭐⭐⭐ 揭示了 LVLM 部署中一个被忽视但极重要的安全问题

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] MMErroR: A Benchmark for Erroneous Reasoning in Vision-Language Models](mmerror_a_benchmark_for_erroneous_reasoning_in_vision-language_models.md)
- [\[ACL 2026\] MedLayBench-V: A Large-Scale Benchmark for Expert-Lay Semantic Alignment in Medical Vision Language Models](medlaybench-v_a_large-scale_benchmark_for_expert-lay_semantic_alignment_in_medic.md)
- [\[CVPR 2026\] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](../../CVPR2026/multimodal_vlm/continual_learning_with_visionlanguage_models_via.md)
- [\[ICLR 2026\] PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models](../../ICLR2026/multimodal_vlm/ppe_positional_preservation_embedding_for_token_compression_in_multimodal_large_.md)
- [\[ACL 2026\] Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring](rethinking_jailbreak_detection_of_large_vision_language_models_with_representati.md)

<!-- RELATED:END -->
