---
title: >-
  [论文解读] Are They Lovers or Friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues
description: >-
  [ACL 2026][LLM评测][社会关系推理] 本文提出 SCRIPTS 基准，包含 1.1K 英语和韩语电影对话，通过三层概率标签（HIGHLY LIKELY / LESS LIKELY / UNLIKELY）评估 9 个 LLM 的社会关系推理能力…
tags:
  - "ACL 2026"
  - "LLM评测"
  - "社会关系推理"
  - "大语言模型评测"
  - "跨语言分析"
  - "电影对话"
  - "文化依赖性"
---

# Are They Lovers or Friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues

**会议**: ACL 2026  
**arXiv**: [2510.19028](https://arxiv.org/abs/2510.19028)  
**代码**: [https://github.com/rladmstn1714/SCRIPTS](https://github.com/rladmstn1714/SCRIPTS)  
**领域**: LLM评测  
**关键词**: 社会关系推理, 大语言模型评测, 跨语言分析, 电影对话, 文化依赖性

## 一句话总结

本文提出 SCRIPTS 基准，包含 1.1K 英语和韩语电影对话，通过三层概率标签（HIGHLY LIKELY / LESS LIKELY / UNLIKELY）评估 9 个 LLM 的社会关系推理能力，发现模型在英语上准确率仅 75-80%、韩语 58-69%，且 CoT 和思维模型对社会推理几乎无帮助。

## 研究背景与动机

**领域现状**：随着 LLM 代理在多方交互场景中越来越普及（如 ChatGPT 群聊），LLM 需要正确识别对话参与者之间的社会关系（如恋人、朋友、亲子）。错误的关系推断可能导致不当回复和隐私泄露等安全风险。

**现有痛点**：先前评估 LLM 社会关系推理的研究存在简化设置——(1) 多采用多选分类，限制了推理的细粒度；(2) 关系类型分类学有限；(3) 多聚焦于简单的双人对话；(4) 单一正确标签无法捕捉社会关系的固有不确定性。

**核心矛盾**：社会关系推理本质上是模糊和上下文依赖的——同一句"你从来不听我说"在情侣间可能是严肃抱怨，在朋友间可能是玩笑——但现有评测框架无法捕捉这种模糊性。

**本文目标**：构建一个支持不确定性感知评估的跨语言社会关系推理基准，全面评估当前 LLM 在英语和韩语中的社会推理能力和失败模式。

**切入角度**：使用电影脚本作为接近真实对话的数据源，引入三层概率标签方案，包含三人对话场景，并从跨语言角度分析文化特异性。

**核心 idea**：社会关系推理不能用单一正确标签评估，需要区分"高度可能"、"不太可能"和"不可能"的关系推断，以细粒度地衡量模型的社会智能。

## 方法详解

### 整体框架

SCRIPTS 从美国和韩国电影脚本中提取多轮对话，由母语标注者进行三阶段标注：(1) 标记 UNLIKELY 关系（多数投票，≥2/3 标注者同意）；(2) 开放式标注 HIGHLY LIKELY 关系（取并集，允许多标签）；(3) 推导 LESS LIKELY 关系（预定义集合中排除前两类的剩余部分）。评测时，每个模型对每段对话运行 5 次取多数回答，计算 HIGHLY LIKELY 准确率和 UNLIKELY 错误率。

### 关键设计

1. **三层概率标签方案 (Probabilistic Labeling)**:

    - 功能：捕捉社会关系推理中的固有模糊性
    - 核心思路：HIGHLY LIKELY（对话强烈支持的关系，开放式标注取并集）、LESS LIKELY（可能但不突出的关系）、UNLIKELY（与对话明确矛盾的关系，≥2/3 标注者同意）。平均每段对话有 3.67 个 HIGHLY LIKELY 和 20.79 个 UNLIKELY 关系
    - 设计动机：传统单标签评测无法区分"荒谬的推断"和"虽不最优但合理的推断"。三层方案允许对无意义预测施加惩罚（UNLIKELY 指标），同时奖励识别最突出关系的能力（HIGHLY LIKELY 指标）

2. **跨语言双语数据集设计**:

    - 功能：评估 LLM 在不同文化语境下的社会推理能力
    - 核心思路：580 段英语对话（28 部美国电影）+ 567 段韩语对话（32 部韩国电影），包含 41.8% 的三人对话以增加推理复杂度。每段对话平均约 10 轮，共 230 种（英语）和 617 种（韩语）独特的 HIGHLY LIKELY 关系类型
    - 设计动机：社会关系推理高度依赖语言和文化背景——韩语的语法化敬语系统（尊卑关系、动词形态变化）编码了英语所没有的关系信息，使跨语言评估成为理解社会推理文化依赖性的必要条件

3. **对话级标注策略 (Dialogue-Level Labeling)**:

    - 功能：避免静态角色标签的误导性
    - 核心思路：不使用电影元数据中的全局角色标签，而是为每段具体对话独立标注关系。对比发现 19% 的电影级标签在特定对话中不相关，且平均每段对话包含超过 3 个 HIGHLY LIKELY 关系
    - 设计动机：社会关系是动态的——说话者可在不同语境中切换角色并同时持有多种关系。全局标签无法反映这种上下文依赖的复杂性

### 评估设置

使用开放式生成而非固定选项分类。提示包含 27 种示例关系类型作为参考，但允许模型自由生成。每个模型运行 5 次取多数投票，使用 GPT-4o 作为评估者（人工验证准确率 92%）。

## 实验关键数据

### 主实验

**多语言模型性能对比（5 次运行多数投票）**

| 模型 | 英语 HIGHLY LIKELY ↑ | 英语 UNLIKELY ↓ | 韩语 HIGHLY LIKELY ↑ | 韩语 UNLIKELY ↓ |
|------|---------------------|----------------|---------------------|----------------|
| GPT-4o | 0.767 | 0.116 | 0.642 | 0.215 |
| o3 (thinking) | 0.807 | 0.086 | 0.742 | 0.152 |
| Gemini-2.5-Flash | 0.759 | 0.154 | 0.582 | 0.318 |
| Qwen-3-14B | 0.623 | 0.164 | 0.455 | 0.444 |
| Llama-3.1-8B | 0.413 | 0.319 | 0.321 | — |
| A.X-4.0-Light (韩语专用) | 0.589 | — | 0.467 | — |

### 消融实验

**Thinking 模式消融（单次运行）**

| 模型 | Thinking | 英语 HL ↑ | 英语 UL ↓ | 韩语 HL ↑ | 韩语 UL ↓ |
|------|---------|----------|----------|----------|----------|
| Gemini-2.5-Flash | ✗ | 0.759 | 0.154 | 0.582 | 0.318 |
| Gemini-2.5-Flash | ✓ | 0.776 | 0.138 | 0.538 | 0.239 |
| Qwen-3-14B | ✗ | 0.623 | 0.164 | 0.455 | 0.444 |
| Qwen-3-14B | ✓ | 0.673 | 0.107 | 0.467 | 0.443 |

**辅助社会信息的影响（GPT-4o，人类标注标签）**

提供辅助社会信息（年龄/性别、关系维度等）可减少 UNLIKELY 预测比例，但不一致提升 HIGHLY LIKELY 准确率。模型自主推断的辅助信息因不够准确（年龄/性别 <60%，关系维度 <75%）而无法帮助。

### 关键发现

- 所有模型在韩语上表现显著低于英语（HL 差距 7-19%p），UNLIKELY 比例增加 7-17%p
- CoT 提示对社会推理无一致性帮助，偶尔还会放大社会偏见（如 Llama 在韩语上 UNLIKELY 增加 3.1%p）
- 韩语专用模型（A.X-4.0-Light）在两种语言上均排名第一，但韩语 HL 仍仅 0.467
- 思维模式的效果在统计上不显著（bootstrap test, p > 0.05）
- 辅助社会信息帮助模型避免不合理推断，但当前 LLM 自主推断这些信息的能力不足

## 亮点与洞察

- 四种失败模式分析极具洞察力：(1) 混淆称谓词与指称词——模型将"that's my Dad"中对第三方的指称误解为对听话者的称呼；(2) 无法聚合多线索——检测到多个线索但错误地权重分配；(3) 无法识别非典型关系——当父母和子女像同龄人一样交谈时预测为"兄弟姐妹"；(4) 不理解韩语/文化特征（占韩语错误的 46%）——误解敬语和亲属称谓
- GPT-4o 的失败模式分布在英语和韩语间截然不同：英语以"无法聚合多线索"为主（36.7%），韩语以"文化特征理解失败"为主（46%）
- 开放式生成 + 概率标签的评测范式可推广到其他需要不确定性感知的 NLP 任务

## 局限与展望

- 仅覆盖英语和韩语两种语言，可能无法推广到其他文化
- 电影脚本可能不完全反映真实对话
- CoT 痕迹可能是事后合理化而非真实推理过程
- 未来应扩展到更多语言、更真实的对话来源（如隐私保护的真实对话、人-AI 对话日志）

## 相关工作与启发

- **vs DDRel (Jia et al., 2021)**: DDRel 使用多选分类和有限关系类型，SCRIPTS 采用开放式生成和三层概率标签，关系类型更丰富（230+/617+ 独特类型）
- **vs PRIDE (Tigunova et al., 2021)**: PRIDE 从电影摘要而非对话中收集标注，无法反映对话中实际呈现的关系
- **vs Jurgens et al. (2023)**: 使用单轮话语而非多轮对话，缺少韩语等非英语语言

## 评分

- 新颖性: ⭐⭐⭐⭐ 三层概率标签和跨语言社会推理评测是有意义的创新
- 实验充分度: ⭐⭐⭐⭐ 9 个模型、双语、CoT 消融、辅助信息消融、失败模式分析，覆盖全面
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，案例分析生动，失败模式归纳精彩
- 价值: ⭐⭐⭐⭐ 揭示了 LLM 社会推理的重要盲点，对安全部署有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Evaluating Reasoning Models for Queries with Presuppositions](evaluating_reasoning_models_for_queries_with_presuppositions.md)
- [\[ACL 2026\] Personalized Benchmarking: Evaluating LLMs by Individual Preferences](personalized_benchmarking_evaluating_llms_by_individual_preferences.md)
- [\[AAAI 2026\] Where Norms and References Collide: Evaluating LLMs on Normative Reasoning](../../AAAI2026/llm_evaluation/where_norms_and_references_collide_evaluating_llms_on_normative_reasoning.md)
- [\[ACL 2026\] Evaluating Legal Reasoning Traces with Legal Issue Tree Rubrics](evaluating_legal_reasoning_traces_with_legal_issue_tree_rubrics.md)
- [\[ACL 2026\] Can LLMs Act as Historians? Evaluating Historical Research Capabilities of LLMs via the Chinese Imperial Examination](can_llms_act_as_historians_evaluating_historical_research_capabilities_of_llms_v.md)

</div>

<!-- RELATED:END -->
