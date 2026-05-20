---
title: >-
  [论文解读] When Helpers Become Hazards: A Benchmark for Analyzing Multimodal LLM-Powered Safety in Daily Life
description: >-
  [ACL 2026][LLM安全][多模态安全] 提出 SaLAD 基准，包含 2013 个真实图文样本覆盖 10 类日常场景，评估多模态大模型在日常辅助中识别隐性安全风险并提供安全警告的能力，揭示即使最强模型在不安全查询上准确率也仅 57.2%。
tags:
  - "ACL 2026"
  - "LLM安全"
  - "多模态安全"
  - "基准测试"
  - "日常生活安全"
  - "安全警告评估"
  - "MLLM对齐"
---

# When Helpers Become Hazards: A Benchmark for Analyzing Multimodal LLM-Powered Safety in Daily Life

**会议**: ACL 2026  
**arXiv**: [2601.04043](https://arxiv.org/abs/2601.04043)  
**代码**: [GitHub](https://github.com/xinyuelou/SaLAD)  
**领域**: 多模态VLM  
**关键词**: 多模态安全, 基准测试, 日常生活安全, 安全警告评估, MLLM对齐

## 一句话总结
提出 SaLAD 基准，包含 2013 个真实图文样本覆盖 10 类日常场景，评估多模态大模型在日常辅助中识别隐性安全风险并提供安全警告的能力，揭示即使最强模型在不安全查询上准确率也仅 57.2%。

## 研究背景与动机

**领域现状** 多模态大语言模型（MLLM）已成为人类生活中不可或缺的助手，能够辅助用户解决问题和提供指导。然而，当用户将 MLLM 的回复作为行为参考时，不准确或有偏差的内容可能导致不安全的决策。

**现有痛点** 现有多模态安全基准主要聚焦于显式的恶意行为（如越狱攻击），但在真实场景中，用户通常并非故意引导模型生成不安全内容，而是在医疗、交通、营养等领域寻求常规信息时被误导。已有隐性安全基准（如 SIUO、MSSBench）存在样本量少、场景不切实际、类别不完整等问题。

**核心矛盾** 传统安全评估以"拒绝回答"为标准，但在日常辅助场景中，简单拒绝并不能保护用户——模型需要识别出隐藏风险并提供建设性的安全警告。现有安全对齐方法在传统基准上效果显著，但在 SaLAD 这种隐性安全场景中几乎无效。

**本文目标** 构建一个贴近真实日常生活的多模态安全基准，全面评估 MLLM 在辅助人类日常活动中识别隐性安全风险的能力。

**切入角度** 从真实用户交互中挖掘查询，配以真实图片（非 AI 生成），确保安全风险不能仅从文本推断（需要跨模态推理），并引入"安全警告"评估框架取代简单的"拒绝/通过"二分法。

**核心 idea** 安全评估应从"模型是否拒绝回答"转向"模型是否识别风险并提供有意义的安全警告"，这更符合日常辅助场景的实际需求。

## 方法详解

### 整体框架
SaLAD 的构建遵循三步流水线：(1) 从 100K 真实用户查询中挖掘潜在不安全行为；(2) 为每个文本查询检索匹配的真实图片，构建图文对；(3) 为不安全样本编写安全警告，为安全样本编写正常建议。评估采用基于安全警告的 GPT-4o 自动评判。

### 关键设计

1. **数据构建——隐性风险挖掘**:
    - 功能：从真实用户交互中识别日常场景中的潜在不安全行为
    - 核心思路：收集 100K 真实用户查询作为候选池，使用 K-Means + Sentence-BERT 聚类为 10 个类别（家居、运动、学习、办公等），由 LLM 过滤相似查询并生成潜在不安全行为，人工标注员依据权威安全手册验证
    - 设计动机：确保场景真实可信，避免不切实际的极端案例（如"用雨伞飞"），同时保证覆盖面广

2. **无视觉安全信息泄露（No VSIL）设计**:
    - 功能：确保安全风险不能仅从文本推断，必须结合图文跨模态推理
    - 核心思路：文本查询中不披露图像特定信息，模型必须整合视觉和文本两种模态才能正确响应
    - 设计动机：已有基准中很多安全风险可以仅从文本推断出来（VSIL 问题），这使得评估变成了纯文本安全而非多模态安全能力

3. **安全警告评估框架**:
    - 功能：以安全警告质量而非拒绝率作为评估标准
    - 核心思路：对不安全样本，正确回复应明确识别风险并提供解释；对安全样本，正确回复应正常回答而非过度敏感地拒绝。使用 GPT-4o 作为评判器，与人工评估的一致性达到 93.85%
    - 设计动机：日常辅助场景中简单拒绝无法保护用户，模型应像有经验的助手一样指出"为什么危险"和"如何安全地做"

### 损失函数 / 训练策略
本文是基准构建工作，无训练过程。数据由 6 名人工标注员交叉验证，每个样本由其他 5 人校验，确保满足两个标准：(1) 风险不能仅从文本推断；(2) 图文结合后安全警告清晰连贯。

## 实验关键数据

### 主实验

| 模型 | 安全集准确率 | 不安全集准确率 | 总体准确率 |
|------|------------|--------------|----------|
| Claude3.7-Sonnet | 99.58 | **57.20** | **77.05** |
| Gemini2.5-Flash | 99.68 | 55.05 | 75.96 |
| GPT-4o | 99.79 | 53.83 | 75.36 |
| LLaVA-OneVision | 99.89 | 37.10 | 66.52 |
| Qwen2.5-VL-7B | 98.41 | 31.59 | 62.89 |
| Deepseek-VL2-Tiny | 89.08 | 10.93 | 47.54 |

### 消融实验

| 配置 | 安全集 | 不安全集 | 总体 | 说明 |
|------|--------|---------|------|------|
| Qwen2.5-VL-7B Vanilla | 100.00 | 33.00 | 66.50 | 基线 |
| w/o image | 98.50 | 23.50 | 61.00 | 无图时不安全检测大幅下降 |
| w/ image caption | 100.00 | 27.50 | 63.75 | 图片描述不能替代原图 |
| w/ Safety Prompt | 100.00 | 41.50 | 70.75 | 安全提示有限提升 |
| + VLGuard | 94.50 | 43.50 | 69.00 | 对齐方法效果有限 |
| + SPA-VL | 100.00 | 35.00 | 67.50 | 仅提升 1% |

### 关键发现
- 即使最强闭源模型（Claude3.7）在不安全查询上准确率也仅 57.2%，开源模型平均仅 30.65%
- 去掉图片后不安全检测下降约 10%，证实数据集的跨模态设计有效
- 图片描述（caption）不能替代原图，因为隐性安全风险藏在细粒度视觉细节中
- 现有安全对齐方法（VLGuard、MIS、SPA-VL）效果有限：VLGuard 会增加安全集的拒绝率，MIS 虽不拒绝但无法识别风险
- 多选题测试表明模型拥有 80%+ 的安全知识，但无法在多模态场景中正确应用

## 亮点与洞察
- "安全警告"评估框架比传统的拒绝率评估更有实际意义，推动安全评估从"能否拒绝"到"能否保护用户"的范式转变
- 无视觉安全信息泄露（No VSIL）设计确保了基准真正测试的是跨模态安全理解
- 发现了"知识-应用鸿沟"：模型具备安全知识但无法在视觉场景中应用
- 安全和过敏两个子集的平衡设计，避免模型通过一刀切拒绝来"刷分"

## 局限与展望
- 数据集规模为 2013 个样本，虽足以揭示漏洞但覆盖面有限
- 评估依赖 LLM-as-a-judge，虽然与人工评估一致性高但存在固有局限
- 仅覆盖英文场景，不同文化背景下的安全标准可能不同
- 未来需要开发更细粒度、更具泛化能力的多模态安全对齐策略

## 相关工作与启发
- 与 VLSBench 的发现形成对比：后者认为"用图片描述比用图片更安全"，但 SaLAD 证明在隐性安全场景中图片描述无法替代原图
- 对安全对齐领域的警示：在传统基准上有效的方法在隐性安全场景中几乎失效
- 为未来的多模态安全对齐研究提供了更贴近实际的评测平台

## 评分
- 新颖性: ⭐⭐⭐⭐ 隐性安全+安全警告评估是有意义的新视角
- 实验充分度: ⭐⭐⭐⭐⭐ 18个模型、多种安全对齐方法、详细的模态分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机充分
- 价值: ⭐⭐⭐⭐ 对多模态安全研究有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Robust Multimodal Safety via Conditional Decoding](robust_multimodal_safety_via_conditional_decoding.md)
- [\[ACL 2026\] SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models](safetyalfred_evaluating_safety-conscious_planning_of_multimodal_large_language_m.md)
- [\[ACL 2026\] MUSE: A Run-Centric Platform for Multimodal Unified Safety Evaluation of Large Language Models](muse_a_run-centric_platform_for_multimodal_unified_safety_evaluation_of_large_la.md)
- [\[ACL 2026\] When Models Outthink Their Safety: Unveiling and Mitigating Self-Jailbreak in Large Reasoning Models](when_models_outthink_their_safety_unveiling_and_mitigating_self-jailbreak_in_lar.md)
- [\[ACL 2026\] Into the Gray Zone: Domain Contexts Can Blur LLM Safety Boundaries](into_the_gray_zone_domain_contexts_can_blur_llm_safety_boundaries.md)

</div>

<!-- RELATED:END -->
