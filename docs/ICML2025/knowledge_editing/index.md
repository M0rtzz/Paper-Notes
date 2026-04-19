---
title: >-
  ICML2025 知识编辑方向 2篇论文解读
description: >-
  2篇ICML2025 知识编辑方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✏️ 知识编辑

**🧪 ICML2025** · **2** 篇论文解读

**[Representation Shattering in Transformers: A Synthetic Study with Knowledge Editing](representation_shattering_in_transformers_a_synthetic_study_with_knowledge_editi.md)**

:   通过在环形结构知识图谱上训练Transformer的合成实验，发现知识编辑（KE）会"粉碎"模型内部学到的几何表示流形，且粉碎程度与编辑距离正相关（$r^2=0.905$），从而提出"表示粉碎"（representation shattering）作为KE损害模型能力的机制性假说，并在Llama 3和Mamba上验证了该现象的普遍性。

**[WikiBigEdit: Understanding the Limits of Lifelong Knowledge Editing in LLMs](wikibigedit_understanding_the_limits_of_lifelong_knowledge_editing_in_llms.md)**

:   本文提出 WikiBigEdit，一个包含 50 万+ 真实 Wikidata 知识编辑的大规模终身知识编辑基准，揭示了现有知识编辑方法在实际规模下的严重局限性——检索增强和持续微调+模型合并等通用方法反而表现更优。
