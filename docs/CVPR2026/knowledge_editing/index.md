---
title: >-
  CVPR2026 知识编辑方向2篇论文解读
description: >-
  2篇CVPR2026的知识编辑方向论文解读，涵盖个性化生成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "知识编辑"
  - "论文解读"
  - "论文笔记"
  - "个性化生成"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✏️ 知识编辑

**📷 CVPR2026** · **2** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (8)](../../ICML2026/knowledge_editing/index.md) · [💬 ACL2026 (9)](../../ACL2026/knowledge_editing/index.md) · [🔬 ICLR2026 (8)](../../ICLR2026/knowledge_editing/index.md) · [🤖 AAAI2026 (5)](../../AAAI2026/knowledge_editing/index.md) · [🧠 NeurIPS2025 (6)](../../NeurIPS2025/knowledge_editing/index.md) · [🧪 ICML2025 (2)](../../ICML2025/knowledge_editing/index.md)

**[Attribution-Guided Model Rectification of Unreliable Neural Network Behaviors](attribution-guided_model_rectification_of_unreliable_neural_network_behaviors.md)**

:   提出归因引导的动态模型纠正框架，将rank-one model editing从领域适配重定位为行为纠正，通过Integrated Gradients量化各层可编辑性自动定位嫌疑层，仅需1个清洁样本即可修复后门攻击、虚假相关和特征泄漏三类不可靠行为。

**[MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization](mokus_leveraging_crossmodal_knowledge_transfer_for.md)**

:   提出"知识感知概念定制"新任务，发现LLM文本编码器中的知识编辑可以自然迁移到视觉生成模态（跨模态知识迁移），基于此提出MoKus框架：先用LoRA微调将稀有token绑定为视觉概念的锚表征，再通过知识编辑技术将多条自然语言知识高效映射到锚表征上，每条知识更新仅需约7秒。
