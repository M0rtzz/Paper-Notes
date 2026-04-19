---
title: >-
  AAAI2026 NLP理解方向 2篇论文解读
description: >-
  2篇AAAI2026 NLP理解方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📖 NLP理解

**🤖 AAAI2026** · **2** 篇论文解读

**[Language Models and Logic Programs for Trustworthy Tax Reasoning](language_models_and_logic_programs_for_trustworthy_tax_reasoning.md)**

:   将税法推理重新定义为语义解析任务，让LLM将法规文本和纳税案例翻译为Prolog逻辑程序，由符号求解器执行计算，通过金标准法规+智能检索案例示例+自一致性检查，在SARA数据集上实现86/100的正确率，并将预计部署成本降至15.78美元/人（低于美国人均报税成本的6%）。

**[Understanding Syllogistic Reasoning in LLMs from Formal and Natural Language Perspectives](understanding_syllogistic_reasoning_in_llms_from_formal_and_natural_language_per.md)**

:   系统评估14个LLM在160个三段论上的推理表现，通过双维度ground truth框架（句法有效性+NLU可信度）揭示顶级模型在形式逻辑上接近完美(99.6%)但自然语言可信度判断仅为随机水平(~52%)——与人类推理模式恰好相反；12/14模型存在显著信念偏差，且few-shot提示反而降低形式推理性能。
