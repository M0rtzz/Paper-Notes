---
title: >-
  ICML2026 信息检索/RAG方向2篇论文解读
description: >-
  2篇ICML2026的信息检索/RAG 方向论文解读，涵盖 RAG、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "信息检索/RAG"
  - "论文解读"
  - "论文笔记"
  - "RAG"
  - "多模态"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔍 信息检索/RAG

**🧪 ICML2026** · **2** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (60)](../../ACL2026/information_retrieval/index.md) · [📷 CVPR2026 (7)](../../CVPR2026/information_retrieval/index.md) · [🔬 ICLR2026 (33)](../../ICLR2026/information_retrieval/index.md) · [🤖 AAAI2026 (28)](../../AAAI2026/information_retrieval/index.md) · [🧠 NeurIPS2025 (31)](../../NeurIPS2025/information_retrieval/index.md) · [📹 ICCV2025 (8)](../../ICCV2025/information_retrieval/index.md)

**[Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation](hierarchical_abstract_tree_for_cross-document_retrieval-augmented_generation.md)**

:   Ψ-RAG 用"合并—坍缩"式的层次聚类替换 RAPTOR 的 k-means 来构建跨文档抽象树，并配上一个具备多轮重写能力的检索回答 Agent 与稀疏 BM25 混合索引，让 Tree-RAG 第一次能在语料级、跨文档多跳问答上追平甚至超越 Graph-RAG，平均 F1 比 RAPTOR 高 25.9%、比 HippoRAG 2 高 7.4%。

**[Very Efficient Listwise Multimodal Reranking for Long Documents](very_efficient_listwise_multimodal_reranking_for_long_documents.md)**

:   ZipRerank 同时砍掉 VLM 列表式重排的两大瓶颈——「视觉 token 序列过长」和「自回归解码逐 token 输出排名」——用 query-aware token 剪枝 + 单 logit 排序在 MMDocIR 上把 LLM 推理延迟降一个数量级，同时匹配或超越当前 SOTA 的 MM-R5。
