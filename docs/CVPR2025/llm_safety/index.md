---
title: >-
  CVPR2025 LLM 安全方向3篇论文解读
description: >-
  3篇CVPR2025的 LLM 安全方向论文解读，涵盖持续学习、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2025"
  - "LLM 安全"
  - "论文解读"
  - "论文笔记"
  - "持续学习"
  - "对抗鲁棒"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔒 LLM 安全

**📷 CVPR2025** · **3** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (21)](../../ACL2026/llm_safety/index.md) · [📷 CVPR2026 (16)](../../CVPR2026/llm_safety/index.md) · [🔬 ICLR2026 (36)](../../ICLR2026/llm_safety/index.md) · [🤖 AAAI2026 (29)](../../AAAI2026/llm_safety/index.md) · [🧠 NeurIPS2025 (59)](../../NeurIPS2025/llm_safety/index.md) · [📹 ICCV2025 (8)](../../ICCV2025/llm_safety/index.md)

🔥 **高频主题：** 持续学习 ×2

**[Low-Rank Adaptation in Multilinear Operator Networks for Security-Preserving Incremental Learning](low-rank_adaptation_in_multilinear_operator_networks_for_security-preserving_inc.md)**

:   针对全同态加密（Leveled FHE）场景下多线性算子网络的灾难性遗忘问题，提出了一种结合低秩适应（LoRA）和梯度投影记忆（GPM）机制的增量学习方法，在保障数据安全的前提下实现持续学习。

**[Neural Gate: Mitigating Privacy Risks in LVLMs via Neuron-Level Gradient Gating](neural_gate_mitigating_privacy_risks_in_lvlms_via_neuron-level_gradient_gating.md)**

:   Neural Gate 发现 LVLM 中隐私相关神经元具有强跨样本不一致性——仅约 10% 的神经元一致性编码隐私信号。基于此发现，提出神经元级梯度门控编辑：仅对强一致性隐私神经元施加梯度更新，在 MiniGPT 上将 Safety EtA 从 0.48 提升至 0.89，同时 Utility 保持不降。

**[Order-Robust Class Incremental Learning: Graph-Driven Dynamic Similarity Grouping](order-robust_class_incremental_learning_graph-driven_dynamic_similarity_grouping.md)**

:   提出 GDDSG，用图着色理论将类按相似度分组——同组内类别尽量不相似（减少干扰），每组独立用 NCM 分类器+LoRA 适配器学习，在 CIFAR-100 10-step 上达到 94.00% 准确率和仅 0.78% 遗忘率（前 SOTA RanPAC 90.50%/3.49%）。
