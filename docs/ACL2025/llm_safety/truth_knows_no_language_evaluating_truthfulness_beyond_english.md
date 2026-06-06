---
title: >-
  [论文解读] Truth Knows No Language: Evaluating Truthfulness Beyond English
description: >-
  [ACL 2025][LLM安全][真实性评估] 构建首个专业翻译的多语言 TruthfulQA 基准（巴斯克语、加泰罗尼亚语、加利西亚语、西班牙语），发现 LLM 的跨语言真实性差异小于预期，且 LLM-as-a-Judge 比多选题指标更贴合人类判断。
tags:
  - "ACL 2025"
  - "LLM安全"
  - "真实性评估"
  - "TruthfulQA"
  - "多语言"
  - "LLM-as-a-Judge"
  - "低资源语言"
---

# Truth Knows No Language: Evaluating Truthfulness Beyond English

**会议**: ACL 2025  
**arXiv**: [2502.09387](https://arxiv.org/abs/2502.09387)  
**代码**: [github.com/hitz-zentroa/truthfulqa-multi](https://github.com/hitz-zentroa/truthfulqa-multi)  
**领域**: LLM安全  
**关键词**: 真实性评估, TruthfulQA, 多语言, LLM-as-a-Judge, 低资源语言

## 一句话总结

构建首个专业翻译的多语言 TruthfulQA 基准（巴斯克语、加泰罗尼亚语、加利西亚语、西班牙语），发现 LLM 的跨语言真实性差异小于预期，且 LLM-as-a-Judge 比多选题指标更贴合人类判断。

## 研究背景与动机

TruthfulQA 是评估 LLM 真实性的标准基准，其核心思想是测试模型是否会模仿人类的错误信念和误解。然而该基准存在明显局限：

**仅支持英语**：尽管一些开发者已进行机器翻译，但缺乏专业翻译版本和系统的跨语言评估

**评估方法争议**：标准的多选题指标（MC2）是否足以衡量真实性存疑,尤其在非英语场景

**文化偏向**：TruthfulQA 有强烈的英语/美国文化背景，许多问题涉及美国法律、英语谚语等

这项工作的动机是：LLM 在不同语言中是否同样真实？如果英语中能避免某个误解，在巴斯克语中是否也能做到？

目标语言的选取也有考量：
- **巴斯克语**：黏着语孤立语，LLM 预训练数据极少
- **加泰罗尼亚语、加利西亚语**：低资源罗曼语系语言
- **西班牙语**：资源相对丰富的对照组

## 方法详解

### 整体框架

工作包含三个主要组件：

1. **专业翻译数据集**：将 TruthfulQA 的817个问题翻译为4种目标语言
2. **三种评估方法**：人工评估、MC2 多选题指标、LLM-as-a-Judge
3. **机器翻译替代实验**：验证机器翻译能否替代专业翻译

### 关键设计

**翻译策略选择**：
团队面临两种选择：（1）本地化适配目标文化，（2）保留原文化语境。最终选择保留原文化语境以保持跨语言的完全平行性。具体翻译指南包括：
- **谚语和误引**：采用直译策略（如"每天一个苹果"直接翻译）
- **缩写词误解**：保留英语原词并在问题中标注"in English"
- **虚构命名实体**：使用已有翻译（如电影角色名），无翻译时从英语借用
- 所有翻译由目标语言母语者的专业翻译人员完成

**人工评估设计**：
- 评估400个回答（4个模型 × 100个问题），涵盖真实性和信息量
- 采用二值标签（真实/不真实、有信息/无信息），而非原论文的标量评分
- 新增针对指令微调模型的附加指南：长回答中额外信息需由评估者核实
- 使用50个重叠标注计算评估者间信度

**LLM-as-a-Judge 训练**：
- 基础模型：Llama 2 7B（已有微调版）、Gemma 2 9B、Llama 3.1 8B
- 训练数据：英语原始数据 vs. 包含机器翻译的全语言数据
- 最佳配置：Gemma 2 9B instruct + 全语言翻译数据
- 信息量评判模型单独训练，但仅对基座模型有效（因指令模型几乎没有无信息回答）

**MC2 指标**：
- 衡量真实答案的总似然度占所有参考答案的归一化比例
- 使用 LM Evaluation Harness，6-shot 设置
- 指令模型使用多轮对话格式

### 损失函数 / 训练策略

LLM-as-a-Judge 的训练使用学习率 0.01，训练 5 个 epoch。核心是对 (问题, 回答) 对进行真实性二分类。

## 实验关键数据

### 主实验

**人工评估结果**（100个样本/模型/语言的真实性百分比）：

| 模型 | 英语 | 西语 | 加泰 | 加利 | 巴斯克 |
|------|-----|-----|-----|-----|-------|
| Gemma-2-27b-it | 73% | 73% | 71% | 72% | 62% |
| Llama-3-70B-IT | 67% | 70% | 62% | 58% | 48% |
| Llama-3-8B-IT | 67% | 61% | 63% | 51% | 34% |
| Llama-3-70B（base）| 36% | 58% | 58% | 60% | 54% |

关键观察：
- 指令模型通常英语最好，巴斯克语最差，但差异小于预期
- 基座 Llama-3-70B 在英语上真实性最低（36%），但在其他语言上更高——因为英语时更"有信息"但容易模仿错误信念
- 信息量方面，基座模型在非英语中经常产生无信息回答

**Judge-LLM 与 MC2 的人类判断相关性**（Cohen Kappa）：

| 方法 | 英语 | 西语 | 加泰 | 加利 | 巴斯克 |
|------|-----|-----|-----|-----|-------|
| MC2 | ~0.3 | ~0.2 | ~0.2 | ~0.2 | ~0.1 |
| Judge-LLM（最佳）| 0.74 | 0.70 | 0.75 | 0.72 | 0.60 |
| 人工间一致性 | ~0.75 | ~0.72 | ~0.70 | ~0.70 | ~0.65 |

**关键结果**：Judge-LLM 与人类判断的一致性远高于 MC2，且接近人类评估者之间的一致性。

**Judge-LLM 完整评估**（12个模型 × 5语言）：
- Gemma-2-27b-it 在所有语言上均表现最佳（平均约 61%）
- 指令模型平均约 57%，基座模型平均约 46%
- 语言间差异：英语（~50-58%）> 西语 ≈ 加利 > 加泰 > 巴斯克

### 消融实验

**机器翻译 vs. 专业翻译**：
- 使用 Claude 3.5 Sonnet 自动翻译 TruthfulQA
- 以专业翻译为参考，MT 翻译质量高（巴斯克语稍低，因黏着语特性）
- 使用 MT 版本训练的 Judge-LLM 与专业翻译版本效果相当
- 结论：机器翻译是扩展真实性基准到更多语言的可行替代方案

**通用 vs. 语境依赖问题**：
- 将 TruthfulQA 问题分为"通用知识"（如变色龙变色原因）和"语境/时间依赖"（如美国法律）
- 通用知识问题在所有语言上表现更一致
- 语境依赖问题的跨语言差异更大，更适合评估多语言真实性

### 关键发现

1. LLM 的跨语言真实性差异比预期小得多——即使在巴斯克语（最低资源）中，性能下降也有限
2. MC2 作为真实性评估的唯一指标是不充分的，LLM-as-a-Judge 是更可靠的替代
3. 信息量是真实性评估中的关键因素——基座模型常产生无信息回答，若不考虑信息量会扭曲评估结果
4. 更大的 LLM 通常比同家族小模型更真实，与 Lin et al. (2022) 的早期发现相反
5. 定性分析显示英语回答在推理深度上仍显著领先

## 亮点与洞察

1. **首个专业翻译**：避免了机器翻译可能引入的系统性偏差，为后续研究提供可靠基线
2. **评估方法的深入比较**：系统对比人工评估、MC2、Judge-LLM，用 Cohen Kappa 量化相关性
3. **基座模型的反直觉现象**：英语真实性低于其他语言，因为模型在英语中更"自信"地模仿错误信念
4. **实用发现**：机器翻译可作为专业翻译的可行替代，极大降低了多语言基准构建成本
5. **文化与时间依赖的维度**：强调真实性评估应区分通用知识和语境依赖知识

## 局限与展望

1. TruthfulQA 本身是高度英美中心的——即使专业翻译也无法改变问题的文化背景
2. 仅覆盖4种目标语言（西班牙语、加泰罗尼亚语、加利西亚语、巴斯克语），需扩展到更多语言家族
3. 信息量评估仅对基座模型有效，指令模型缺乏无信息回答使训练不足
4. 人工评估的样本量有限（100题 × 4模型 × 5语言 = 2000个评判）
5. 未探索不同语言的"文化特定误解"——不同文化可能有不同的错误信念

## 相关工作与启发

- **TruthfulQA (Lin et al., 2022)**：原始基准，使用 GPT-3 作为 Judge，本文用更强的多语言 Judge 替代
- **Aula-Blasco et al. (2025)**：多语言真实性基准，区分通用 vs 语境依赖问题——本文沿用该分类并提供实证支持
- **HuggingFace OpenLLM Leaderboard**：广泛使用 MC2 指标——本文证明 MC2 与人类判断相关性低
- 启发：跨语言评估不仅关乎翻译质量，更涉及文化适配和评估方法论的根本思考

## 评分

- **创新性**：⭐⭐⭐ — 核心贡献是高质量翻译和系统评估，方法论创新有限
- **实用性**：⭐⭐⭐⭐ — 为多语言真实性评估提供了重要基准和方法指导
- **实验充分性**：⭐⭐⭐⭐⭐ — 12个模型 × 5语言 × 3评估方法，加人工评估和 IAA 分析
- **写作质量**：⭐⭐⭐⭐ — 结构清晰，翻译指南和评估方法论阐述详尽

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Beyond Facts: Evaluating Intent Hallucination in Large Language Models](intent_hallucination_eval.md)
- [\[NeurIPS 2025\] Trans-EnV: A Framework for Evaluating the Linguistic Robustness of LLMs Against English Varieties](../../NeurIPS2025/llm_safety/trans-env_a_framework_for_evaluating_the_linguistic_robustness_of_llms_against_e.md)
- [\[ACL 2025\] ComparisonQA: Evaluating Factuality Robustness of LLMs Through Knowledge Frequency Control and Uncertainty](comparisonqa_evaluating_factuality_robustness_of_llms_through_knowledge_frequenc.md)
- [\[ACL 2025\] Monitoring Decoding: Mitigating Hallucination via Evaluating the Factuality of Partial Response during Generation](monitoring_decoding_mitigating_hallucination_via_evaluating_the_factuality_of_pa.md)
- [\[ICLR 2026\] No Caption, No Problem: Caption-Free Membership Inference via Model-Fitted Embeddings](../../ICLR2026/llm_safety/no_caption_no_problem_caption-free_membership_inference_via_model-fitted_embeddi.md)

</div>

<!-- RELATED:END -->
