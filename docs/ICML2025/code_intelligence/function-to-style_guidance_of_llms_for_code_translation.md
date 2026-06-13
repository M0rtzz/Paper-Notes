---
title: >-
  [论文解读] Function-to-Style Guidance of LLMs for Code Translation
description: >-
  [ICML 2025][代码智能][code translation] 提出 F2STrans，通过功能学习（正确性）和风格学习（可读性）两阶段渐进式微调 LLM，使 Qwen-1.5B 在 20 种代码翻译场景中平均超越 prompt 增强的 Qwen-32B 和 GPT-4。 领域现状 领域现状：领域现状: LLM 在代…
tags:
  - "ICML 2025"
  - "代码智能"
  - "code translation"
  - "LLM fine-tuning"
  - "functional learning"
  - "style learning"
  - "readability"
---

# Function-to-Style Guidance of LLMs for Code Translation

**会议**: ICML 2025  
**arXiv**: [2507.11083](https://arxiv.org/abs/2507.11083)  
**代码**: 有（论文提及模型和 benchmark 已公开）  
**领域**: 代码智能  
**关键词**: code translation, LLM fine-tuning, functional learning, style learning, readability

## 一句话总结
提出 F2STrans，通过功能学习（正确性）和风格学习（可读性）两阶段渐进式微调 LLM，使 Qwen-1.5B 在 20 种代码翻译场景中平均超越 prompt 增强的 Qwen-32B 和 GPT-4。

## 研究背景与动机

### 领域现状

**领域现状**：领域现状**: LLM 在代码翻译（Java→Python 等）上取得进展，但翻译结果的正确性和可读性仍是挑战。

**现有痛点**: 大多数方法关注功能正确性，但代码风格不自然、可读性差。缺乏同时评估功能和风格的 benchmark。

**核心矛盾**: 功能正确性和代码风格是两个不同维度的目标，直接翻译可能功能正确但风格生硬。

**本文解决什么**: 设计渐进式框架，先确保功能正确，再优化代码风格。

**切入角度**: 从在线编程平台挖掘高质量代码对进行功能学习，再用正负风格样本进行风格学习。

**核心 idea**: 将代码翻译优化解耦为"先功能后风格"两步。

### 解决思路

**本文目标**：### 整体框架
Stage 1 - Functional Learning: 用高质量源-目标代码对微调，优化翻译正确性。


## 方法详解

### 整体框架
Stage 1 - Functional Learning: 用高质量源-目标代码对微调，优化翻译正确性。
Stage 2 - Style Learning: 用正/负风格样本进一步微调，引导模型输出更自然的代码风格。

### 关键设计
1. **高质量代码对挖掘**: 从 LeetCode/Codeforces 中对同一题目的不同语言提交配对，筛选功能等价且质量高的代码对。设计动机：真实多语言代码比合成数据更自然，功能正确性有测试用例保证。

2. **风格学习**: 引入对比学习思想，使用正例（符合目标语言惯用法）和负例（功能正确但风格不自然）。模型学习在保持功能不变的前提下向更自然的风格靠拢。

3. **新 benchmark**: 包含最新源代码、大量测试用例和人工标注 ground-truth，支持 20 种翻译场景（5 语言两两翻译），同时评估功能和风格。

### 损失函数 / 训练策略
- 功能学习阶段：标准 SFT loss
- 风格学习阶段：结合正负样本的对比/偏好学习损失

## 实验关键数据

### 主实验（20 种翻译场景）

| 模型 | 功能正确率 | 风格评分 | 综合 |
|------|----------|---------|------|
| F2STrans (Qwen-1.5B) | 最佳 | 最佳 | SOTA |
| Qwen-32B + prompt | 较高 | 中等 | 不及 1.5B微调 |
| GPT-4 + prompt | 较高 | 中等 | 不及 1.5B微调 |

### 消融实验

| 配置 | 功能正确率 | 风格评分 | 说明 |
|------|----------|---------|------|
| 完整 F2STrans | 最高 | 最高 | 两阶段都有 |
| 仅功能学习 | 较高 | 中等 | 正确但不自然 |
| 仅风格学习 | 较低 | 较高 | 风格好但可能错误 |
| 直接联合训练 | 中等 | 中等 | 不如分阶段 |

### 关键发现
- 1.5B 微调超越 32B 和 GPT-4 的 prompt 方法
- 解耦训练优于联合训练
- 在线编程平台是高质量翻译数据的宝库

## 亮点与洞察
- "先功能后风格"可推广到其他代码生成任务
- 小模型精细微调超越大模型 prompt 再次被验证

## 局限与展望
- 仅覆盖 5 种编程语言
- 风格评估部分依赖人工标注
- 未探讨语义等价性的形式化验证

## 相关工作与启发
- 代码质量不仅是"能跑"，风格和可读性同样重要
- 数据源选择对翻译质量影响很大

## 评分
- 新颖性: ⭐⭐⭐⭐ 功能-风格解耦训练范式新颖
- 实验充分度: ⭐⭐⭐⭐ 20 种场景、新 benchmark
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰
- 价值: ⭐⭐⭐⭐ 对代码翻译工具有直接应用价值

---

## 补充思考

### 与领域发展趋势的关系
本文的研究方向与当前 AI 研究的几个大趋势密切相关：(1) 对 LLM 内部机制的深入理解需求日益增长；(2) 模型效率和可访问性的重要性不断提升；(3) AI 安全和可靠性成为核心关注点。从方法论角度看，本文代表了一种从"黑盒使用"到"白盒理解"的研究范式转变。

### 对未来研究的具体建议
1. 可以将本文的核心思路与其他模态（视觉、语音）结合
2. 考虑在更大规模的模型和数据上验证结论的普适性
3. 探索与强化学习和在线学习结合的可能性
4. 开发自动化的评估和优化工具链


---

## 补充思考

### 与领域发展趋势的关系
本文的研究方向与当前 AI 研究的几个大趋势密切相关：模型能力评估与可靠性保证、参数高效微调与模型压缩、以及 AI 安全与对齐。从方法论角度看，本文代表了对 LLM 深层机制的探索，有助于推动从经验驱动到理论驱动的研究范式转变。

### 对未来研究的具体建议
1. 可以将核心思路与其他模态（视觉、语音、多模态）结合，验证方法的跨模态通用性
2. 在更大规模模型（70B+）和更新的架构（Mixture-of-Experts 等）上验证结论
3. 探索与强化学习、在线学习结合的可能性，实现动态适应
4. 开发自动化评估和优化工具，降低方法的使用门槛
5. 考虑与 LLM alignment 研究的交叉，探索安全性和性能的协同优化

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Bootstrapping Code Translation with Weighted Multilanguage Exploration](../../ACL2026/code_intelligence/bootstrapping_code_translation_with_weighted_multilanguage_exploration.md)
- [\[ACL 2025\] Rethinking Repetition Problems of LLMs in Code Generation](../../ACL2025/code_intelligence/rethinking_repetition_problems_of_llms_in_code_generation.md)
- [\[ICML 2026\] Poison with Style: A Practical Poisoning Attack on Code Large Language Models](../../ICML2026/code_intelligence/poison_with_style_a_practical_poisoning_attack_on_code_large_language_models.md)
- [\[ACL 2025\] TeXpert: A Multi-Level Benchmark for Evaluating LaTeX Code Generation by LLMs](../../ACL2025/code_intelligence/texpert_a_multi-level_benchmark_for_evaluating_latex_code_generation_by_llms.md)
- [\[ICML 2026\] MatchFixAgent: Language-Agnostic Autonomous Repository-Level Code Translation Validation and Repair](../../ICML2026/code_intelligence/matchfixagent_language-agnostic_autonomous_repository-level_code_translation_val.md)

</div>

<!-- RELATED:END -->
