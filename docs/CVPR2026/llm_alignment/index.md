---
title: >-
  CVPR2026 对齐/RLHF论文汇总 · 2篇论文解读
description: >-
  2篇CVPR2026的对齐 / RLHF 方向论文解读，涵盖多模态、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "对齐 / RLHF"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "对抗鲁棒"
item_list:
  - u: "bases_of_steerable_kernels_for_equivariant_cnns_from_2d_rotations_to_the_lorentz/"
    t: "Bases of Steerable Kernels for Equivariant CNNs: From 2D Rotations to the Lorentz Group"
  - u: "principled_steering_via_null-space_projection_for_jailbreak_defense_in_vision-la/"
    t: "Principled Steering via Null-space Projection for Jailbreak Defense in Vision-Language Models"
item_total: 2
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚖️ 对齐 / RLHF

**📷 CVPR2026** · **2** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (26)](../../ICML2026/llm_alignment/index.md) · [💬 ACL2026 (31)](../../ACL2026/llm_alignment/index.md) · [🔬 ICLR2026 (40)](../../ICLR2026/llm_alignment/index.md) · [🤖 AAAI2026 (17)](../../AAAI2026/llm_alignment/index.md) · [🧠 NeurIPS2025 (36)](../../NeurIPS2025/llm_alignment/index.md) · [📹 ICCV2025 (2)](../../ICCV2025/llm_alignment/index.md)

**[Bases of Steerable Kernels for Equivariant CNNs: From 2D Rotations to the Lorentz Group](bases_of_steerable_kernels_for_equivariant_cnns_from_2d_rotations_to_the_lorentz.md)**

:   提出一种绕过 Clebsch-Gordan 系数的方法来求解等变CNN中的可转向核（steerable kernel）约束，通过在稳定子群上求解简单的不变性条件再"转向（steer）"到任意点，为 SO(2) 到 Lorentz 群等不同对称群给出了显式的核基底。

**[Principled Steering via Null-space Projection for Jailbreak Defense in Vision-Language Models](principled_steering_via_null-space_projection_for_jailbreak_defense_in_vision-la.md)**

:   提出 NullSteer，一种基于零空间投影的激活转向防御框架，通过将转向操作限制在良性激活的零空间中，在不损害模型通用能力的前提下有效抵御视觉越狱攻击。
