---
title: >-
  ICML2026 多语言/翻译方向2篇论文解读
description: >-
  2篇ICML2026的多语言/翻译方向论文解读，收录 ML-Embed、Optimizing Language Models for等。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICML2026"
  - "多语言/翻译"
  - "论文解读"
  - "论文笔记"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🌐 多语言/翻译

**🧪 ICML2026** · **2** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (52)](../../ACL2026/multilingual_mt/index.md) · [📷 CVPR2026 (2)](../../CVPR2026/multilingual_mt/index.md) · [🔬 ICLR2026 (5)](../../ICLR2026/multilingual_mt/index.md) · [🤖 AAAI2026 (11)](../../AAAI2026/multilingual_mt/index.md) · [🧠 NeurIPS2025 (14)](../../NeurIPS2025/multilingual_mt/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/multilingual_mt/index.md)

**[ML-Embed: Inclusive and Efficient Embeddings for a Multilingual World](ml-embed_inclusive_and_efficient_embeddings_for_a_multilingual_world.md)**

:   ML-Embed 把 Matryoshka 思想从一维 (representation 维度) 扩展到**三维** —— 在 embedding 参数 (MEL)、模型深度 (MLL)、表征维度 (MRL) 上**全栈嵌套训练**, 同时构建 282 种自然语言 + 40 种编程语言、5000 万样本的多语训练集, 推出 140M-8B 一族开源模型, 在 17 个 MTEB benchmark 上 9 个拿第一, 波兰语 +22.89, 越南语 +6.88.

**[Optimizing Language Models for Crosslingual Knowledge Consistency](optimizing_language_models_for_crosslingual_knowledge_consistency.md)**

:   本文针对多语言 LLM 在不同语言间回答同一问题却给出冲突答案的问题，设计了一个**用"另一种语言下回答的对数似然"作为 reward 的 RL 目标**，证明其最优策略呈 product-of-experts 形式并在 $\gamma_1\gamma_2=\beta^2$ 时保证跨语言偏好一致；据此推导出无需 reward model、无需 online 采样的 **DCO（Direct Consistency Optimization）** 算法，在 9 个 LLM、3 个多语言 QA 基准、26 种语言上同时提升跨语言一致性（RankC）与回答准确率。
