---
title: >-
  ACL2026 图学习方向8篇论文解读
description: >-
  8篇ACL2026的图学习方向论文解读，涵盖强化学习、LLM、图神经网络、信息抽取等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🕸️ 图学习

**💬 ACL2026** · **8** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (8)](../../CVPR2026/graph_learning/index.md) · [🔬 ICLR2026 (21)](../../ICLR2026/graph_learning/index.md) · [🤖 AAAI2026 (38)](../../AAAI2026/graph_learning/index.md) · [🧠 NeurIPS2025 (52)](../../NeurIPS2025/graph_learning/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/graph_learning/index.md) · [🧪 ICML2025 (31)](../../ICML2025/graph_learning/index.md)

**[AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning](agentgl_towards_agentic_graph_learning_with_llms_via_reinforcement_learning.md)**

:   提出 AgentGL，首个基于强化学习的智能体图学习（AGL）框架，让 LLM 智能体通过图原生搜索工具自主导航文本属性图（TAG），在节点分类和链接预测任务上分别实现最高 17.5% 和 28.4% 的绝对准确率提升。

**[ARK: Answer-Centric Retriever Tuning via KG-augmented Curriculum Learning](ark_answer-centric_retriever_tuning_via_kg-augmented_curriculum_learning.md)**

:   提出ARK框架，通过三维答案充分性评分（Forward+Backward+Retriever对齐）筛选正样本，利用LLM构建的知识图谱生成渐进难度的困难负样本进行课程对比学习，在10个数据集上平均提升14.5% F1。

**[AutoPKG: An Automated Framework for Dynamic E-commerce Product-Attribute Knowledge Graph Construction](autopkg_an_automated_framework_for_dynamic_e-commerce_product-attribute_knowledg.md)**

:   提出 AutoPKG，一个多智能体 LLM 框架，从多模态电商商品内容自动构建 Product-Attribute 知识图谱（PKG），通过类型归纳 Agent、属性键发现 Agent、属性值提取 Agent 和集中式 KGD 决策 Agent 实现动态本体的持续演化和规范化，在 Lazada 数据集上取得 0.953 WKE（类型）和 0.724 WKE（属性键），线上 A/B 测试推荐 GMV 提升 7.89%。

**[Comparing Human and Large Language Model Interpretation of Implicit Information](comparing_human_and_large_language_model_interpretation_of_implicit_information.md)**

:   本文提出隐含信息提取（IIE）任务和基于 LLM 的三阶段提取管道（信息提取→推理验证→时序分析），构建结构化知识图谱来表示文本的隐含含义，并通过众包人类判断对比发现 LLM 在社交丰富语境中比人类更保守，但在短事实语境中人类更保守。

**[From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context](from_nodes_to_narratives_explaining_graph_neural_networks_with_llms_and_graph_co.md)**

:   本文提出 Gspell，一个轻量级后验解释框架，通过将 GNN 节点嵌入投影到 LLM 嵌入空间并构建混合提示（软提示+文本），使 LLM 能够直接推理 GNN 内部表示并生成自然语言解释和解释子图，在文本属性图（TAG）上实现了忠实性与可解释性的良好平衡。

**[Graph-Based Alternatives to LLMs for Human Simulation](graph-based_alternatives_to_llms_for_human_simulation.md)**

:   本文提出 GEMS（Graph-basEd Models for Human Simulation），将封闭式人类行为模拟任务建模为异构图上的链接预测问题，在三个数据集和三种评估设定下匹配或超越强 LLM 基线方法，同时参数量减少 3 个数量级。

**[LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs](llms_underperform_graph-based_parsers_on_supervised_relation_extraction_for_comp.md)**

:   本文在六个关系抽取数据集上对比四个 LLM（7B-70B）和一个轻量级图解析器（124M参数），发现当文档的关系图平均边数超过约 18 条时，图解析器持续且显著优于 LLM，在最复杂的 ERFGC 数据集上 F1 差距达 13.2 个点，揭示了 LLM 在复杂语言图结构抽取上的根本局限。

**[Which bird does not have wings: Negative-constrained KGQA with Schema-guided Semantic Matching and Self-directed Refinement](which_bird_does_not_have_wings_negative-constrained_kgqa_with_schema-guided_sema.md)**

:   本文提出了否定约束知识图谱问答（NEST KGQA）新任务和 NestKGQA 数据集，设计了 Python 格式逻辑形式 PyLF 来清晰表达否定约束，并提出 CUCKOO 框架通过约束感知草稿生成、Schema 引导语义匹配和自导向细化三个模块，在 few-shot 设置下实现了多约束问题的高效精确回答。
