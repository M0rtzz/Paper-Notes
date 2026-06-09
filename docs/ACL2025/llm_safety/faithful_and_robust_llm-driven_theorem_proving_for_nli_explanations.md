---
title: >-
  [论文解读] Faithful and Robust LLM-Driven Theorem Proving for NLI Explanations
description: >-
  [ACL 2025][LLM安全][自然语言推理] 本文研究 LLM 与定理证明器（TP）的交互架构，提出四种策略来缓解自动形式化中的语义信息损失、语法错误、证明构造不足和反馈解读困难等问题…
tags:
  - "ACL 2025"
  - "LLM安全"
  - "自然语言推理"
  - "定理证明"
  - "自动形式化"
  - "解释生成"
  - "忠实性"
---

# Faithful and Robust LLM-Driven Theorem Proving for NLI Explanations

**会议**: ACL 2025  
**arXiv**: [2505.24264](https://arxiv.org/abs/2505.24264)  
**代码**: 无  
**领域**: LLM安全  
**关键词**: 自然语言推理、定理证明、自动形式化、解释生成、忠实性

## 一句话总结

本文研究 LLM 与定理证明器（TP）的交互架构，提出四种策略来缓解自动形式化中的语义信息损失、语法错误、证明构造不足和反馈解读困难等问题，在 e-SNLI、QASC 和 WorldTree 三个数据集上分别实现了形式化精度 +18.46%/+34.2%/+39.77% 和解释质量 +29.5%/+51.5%/+41.25% 的显著提升。

## 研究背景与动机

**领域现状**：自然语言推理（NLI）要求模型判断前提和假设之间的逻辑关系（蕴含/矛盾/中立），并生成解释说明推理过程。近年来，LLM 与定理证明器的结合已被用于验证和改进 NLI 解释的逻辑有效性。

**现有痛点**：LLM-TP（定理证明器）混合架构面临多重挑战：（1）将自然语言转化为机器可验证的形式化表示（autoformalisation）时存在语义信息损失，LLM 难以精确捕捉关键逻辑结构；（2）LLM 生成的形式化表示常包含语法错误；（3）LLM 在形式化验证框架中构造严格证明的能力不足；（4）LLM 难以有效解读 TP 返回的反馈信息进行迭代改进。

**核心矛盾**：形式化验证需要精确的逻辑表示，但自然语言本质上是模糊和多义的。LLM 虽然擅长理解自然语言，但在生成严格的形式化表示时容易丢失语义细节或引入逻辑错误，导致忠实性（faithfulness）和鲁棒性（robustness）双重不足。

**本文目标**：针对 LLM-TP 架构的四个痛点，分别提出针对性策略，提升 NLI 解释的自动形式化质量和解释精炼效果。

**切入角度**：作者从混合架构的交互流程入手，分析了 LLM → TP（形式化）和 TP → LLM（反馈）两个方向信息流中的瓶颈，逐一设计干预策略。

**核心 idea**：通过语义保持提示、语法错误自动修复、逻辑表达式引导的证明草图生成、以及增强的反馈解读机制，系统性地改善 LLM-TP 协作的忠实性和鲁棒性。

## 方法详解

### 整体框架

pipeline 由 LLM 和 TP 交替协作完成：输入 NLI 三元组（前提、假设、标签），LLM 先为前提和假设中的解释语句生成形式化的逻辑表示（一阶逻辑/命题逻辑），TP 试图基于这些形式化表示构造证明以验证解释的逻辑有效性。验证失败时，TP 的反馈传回给 LLM 进行迭代修正。最终输出经过验证的忠实 NLI 解释。

### 关键设计

1. **语义保持的自动形式化策略（Semantic-Preserving Autoformalisation）**:

    - 功能：缓解自然语言到形式化表示转换过程中的语义损失
    - 核心思路：在 LLM 进行自动形式化时，引入额外的语义约束提示。具体包括：（a）要求 LLM 显式列出原始自然语言中的关键语义成分，然后逐一检查形式化表示是否覆盖了这些成分；（b）在形式化表示旁注释其对应的自然语言含义，形成"双语对照"以便 LLM 自检。通过这种方式，自动形式化的精度在三个数据集上分别提升了 +18.46%、+34.2%、+39.77%
    - 设计动机：信息损失是因为 LLM 在翻译时倾向于简化或遗漏细节，显式的语义检查清单迫使 LLM 关注完整性

2. **语法错误高效修复机制（Efficient Syntax Error Correction）**:

    - 功能：快速识别并修正 LLM 生成的形式化表示中的语法错误
    - 核心思路：TP 在解析形式化表示时会报告具体的语法错误位置和类型。在将这些错误反馈给 LLM 时，附加结构化的错误分类信息（如"括号不匹配"、"未定义谓词"、"参数类型错误"等），帮助 LLM 精准定位和修复。同时维护一个常见语法错误的修复模板库，对高频错误进行模式匹配自动修复，减少 LLM 调用次数
    - 设计动机：语法错误是最表面但最消耗迭代次数的问题类型，自动修复可以大幅减少无意义的迭代轮次

3. **逻辑表达式引导的证明草图生成（Logic-Guided Proof Sketch Generation）**:

    - 功能：利用逻辑表达式的结构信息引导 LLM 生成更结构化的证明草图
    - 核心思路：在要求 LLM 生成证明之前，先让 LLM 分析已有形式化表示的逻辑结构（如蕴含关系链、条件分支等），然后基于这些结构信息生成一个证明"骨架"（proof sketch），再由 LLM 填充具体的证明步骤。这种从结构到细节的分步生成方式比直接生成完整证明更可靠
    - 设计动机：LLM 直接生成完整的形式化证明时容易迷失在细节中，先有骨架再填充内容是人类数学家常用的策略

### 损失函数 / 训练策略

本文是纯推理时方法，不涉及模型微调。改进全部体现在提示设计和 LLM-TP 交互协议上。关键的训练策略改进在于迭代精炼循环——增强 LLM 解读 TP 反馈的能力，使每次迭代的修正更有效，从而大幅减少达到成功验证所需的迭代轮次。

## 实验关键数据

### 主实验

在三个 NLI 解释数据集上的性能对比：

| 数据集 | 指标 | 本文方法 | 之前SOTA | 提升 |
|--------|------|---------|---------|------|
| e-SNLI | 形式化精度 | 显著提升 | 基线 | +18.46% |
| QASC | 形式化精度 | 显著提升 | 基线 | +34.2% |
| WorldTree | 形式化精度 | 显著提升 | 基线 | +39.77% |
| e-SNLI | 解释精炼质量 | 显著提升 | 基线 | +29.5% |
| QASC | 解释精炼质量 | 显著提升 | 基线 | +51.5% |
| WorldTree | 解释精炼质量 | 显著提升 | 基线 | +41.25% |

### 消融实验

| 配置 | 形式化精度变化 | 说明 |
|------|--------------|------|
| 完整方法 | 最优 | 四种策略全部启用 |
| w/o 语义保持策略 | 明显下降 | 形式化质量大幅退化 |
| w/o 语法修复 | 下降 | 迭代次数增加，效率降低 |
| w/o 证明草图引导 | 下降 | 证明构造成功率降低 |
| w/o 反馈增强解读 | 下降 | 迭代修正质量变差 |

### 关键发现

- 四种策略的贡献各不相同：语义保持策略对形式化精度提升最大，证明草图引导对验证成功率贡献最大
- 在不同 LLM（GPT-4、GPT-3.5 等）上的提升幅度不同，说明策略的效果与 LLM 的基础逻辑推理能力相关
- 特定的架构干预可以"戏剧性地"减少成功验证所需的迭代轮次，显著提升效率
- WorldTree 数据集上的提升最大（形式化 +39.77%、精炼 +41.25%），可能因为该数据集的解释链更长更复杂，策略的优势更明显

## 亮点与洞察

- **系统化的故障分析**非常有价值——将 LLM-TP 架构的问题拆解为四个独立但相关的维度（语义损失、语法错误、证明构造、反馈解读），每个维度有针对性的解决方案
- **从结构到细节的证明草图策略**体现了对人类推理过程的理解——好的证明不是一步写出来的，而是先有整体思路再逐步完善。这个思路可以推广到 LLM 的其他形式化推理场景
- 数值结果非常强——三个数据集上的提升幅度都很大，尤其是解释精炼质量的提升（+29.5% 到 +51.5%），说明策略确实解决了关键瓶颈

## 局限与展望

- 方法依赖于特定的定理证明器（可能是 Isabelle 或 Coq），不同 TP 的反馈格式不同，适配成本存在
- 形式化精度虽然大幅提升，但绝对值可能仍不够高，自然语言到形式逻辑的鸿沟仍然很大
- 论文未开源代码，可复现性有待验证
- 未来可以探索用 RL 来训练 LLM 的形式化能力，将本文的提示策略作为奖励信号

## 相关工作与启发

- **vs NaturalProver**: 早期的 NLI 证明工作直接让 LLM 生成证明，本文增加了 TP 验证环节和迭代修正
- **vs Autoformalisation 系列 (如 ProofNet)**: 这些工作聚焦数学定理的形式化，本文关注 NLI 解释，领域不同但技术相通
- **vs LLM+Tool 框架 (如 ToolFormer, ReAct)**: 本文的 LLM-TP 交互可以看作一种特殊的 LLM+Tool 范式，但针对形式化验证做了深度优化

## 评分

- 新颖性: ⭐⭐⭐⭐ 针对 LLM-TP 交互的系统化改进，四种策略设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集 × 多个 LLM × 详细消融，提升幅度显著且可信
- 写作质量: ⭐⭐⭐⭐ 问题分析深入，策略描述清晰
- 价值: ⭐⭐⭐⭐ 对 LLM 形式化推理的实际应用有直接指导价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] When Backdoors Speak: Understanding LLM Backdoor Attacks Through Model-Generated Explanations](when_backdoors_speak_understanding_llm_backdoor_attacks_through_model-generated_.md)
- [\[ICML 2025\] Robust Multi-bit Text Watermark with LLM-based Paraphrasers](../../ICML2025/llm_safety/robust_multi-bit_text_watermark_with_llm-based_paraphrasers.md)
- [\[ACL 2025\] Robust Data Watermarking in Language Models by Injecting Fictitious Knowledge](robust_data_watermarking_in_language_models_by_injecting_fictitious_knowledge.md)
- [\[ACL 2025\] Core: Robust Factual Precision with Informative Sub-Claim Identification](core_robust_factual_precision_with_informative_sub-claim_identification.md)
- [\[ACL 2025\] Towards Context-Robust LLMs: A Gated Representation Fine-tuning Approach](towards_context-robust_llms_a_gated_representation_fine-tuning_approach.md)

</div>

<!-- RELATED:END -->
