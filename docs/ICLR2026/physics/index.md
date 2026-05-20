---
title: >-
  ICLR2026 物理学方向2篇论文解读
description: >-
  2篇ICLR2026的物理学方向论文解读，收录 Feedback-driven Recurrent Quan、Sublinear Time Quantum Algorit等。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICLR2026"
  - "物理学"
  - "论文解读"
  - "论文笔记"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚛️ 物理学

**🔬 ICLR2026** · **2** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (1)](../../ICML2026/physics/index.md) · [📷 CVPR2026 (1)](../../CVPR2026/physics/index.md) · [🤖 AAAI2026 (2)](../../AAAI2026/physics/index.md) · [🧠 NeurIPS2025 (19)](../../NeurIPS2025/physics/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/physics/index.md) · [🧪 ICML2025 (6)](../../ICML2025/physics/index.md)

**[Feedback-driven Recurrent Quantum Neural Network Universality](feedback-driven_recurrent_quantum_neural_network_universality.md)**

:   本文首次为基于反馈的循环量子神经网络 (RQNN) 建立了定量逼近误差界和普适性证明，表明 RQNN 可在 qubit 数仅以 $\lceil\log_2(\varepsilon^{-1})\rceil$ 对数增长的条件下，以线性读出层逼近任意 fading memory 滤波器，且不受维度灾难影响。

**[Sublinear Time Quantum Algorithm for Attention Approximation](sublinear_time_quantum_algorithm_for_attention_approximation.md)**

:   提出首个对序列长度 $n$ 具有**亚线性**时间复杂度的量子数据结构，用于近似 Transformer 注意力矩阵的行查询，预处理时间 $\widetilde{O}(\epsilon^{-1} n^{0.5} \cdot \text{poly}(d, s_\lambda, \alpha))$，每次行查询 $\widetilde{O}(s_\lambda^2 + s_\lambda d)$，相对经典算法实现了关于 $n$ 的二次加速。
