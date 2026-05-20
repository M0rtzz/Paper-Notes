---
title: >-
  ICML2026 知识编辑方向2篇论文解读
description: >-
  2篇ICML2026的知识编辑方向论文解读，涵盖 LLM、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "知识编辑"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "多模态"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✏️ 知识编辑

**🧪 ICML2026** · **2** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (7)](../../ACL2026/knowledge_editing/index.md) · [📷 CVPR2026 (2)](../../CVPR2026/knowledge_editing/index.md) · [🔬 ICLR2026 (8)](../../ICLR2026/knowledge_editing/index.md) · [🤖 AAAI2026 (5)](../../AAAI2026/knowledge_editing/index.md) · [🧠 NeurIPS2025 (6)](../../NeurIPS2025/knowledge_editing/index.md) · [🧪 ICML2025 (2)](../../ICML2025/knowledge_editing/index.md)

**[CrispEdit: Low-Curvature Projections for Scalable Non-Destructive LLM Editing](crispedit_low-curvature_projections_for_scalable_non-destructive_llm_editing.md)**

:   把 LLM 编辑写成"最小化编辑损失 s.t. 能力损失不变"的约束优化, 用 Bregman 散度等价转化为 Gauss-Newton Hessian 的低曲率子空间投影, 再借 K-FAC + 一个无需显式构造投影矩阵的 Kronecker 特征基技巧, 让 3000 条编辑在 A40 上 6 分钟跑完, 同时把 LLaMA-3-8B 的 MMLU/IFEval/ARC-C/TruthfulQA/GSM8K 平均掉点压到 < 1%, 显著优于 AlphaEdit / MEMIT / 微调。

**[KORE: Enhancing Knowledge Injection for Large Multimodal Models via Knowledge-Oriented Controls](kore_enhancing_knowledge_injection_for_large_multimodal_models_via_knowledge-ori.md)**

:   KORE 通过两阶段"知识导向控制"为 LMM 注入新知识 — 一边把单条事实自动扩成结构化的多轮对话+指令任务（提升泛化），一边用先前知识的协方差矩阵零空间初始化 LoRA 适配器（最小化对旧能力的干扰），在 LLaVA-v1.5 / Qwen2.5-VL 上同时实现强适配和强保留。
