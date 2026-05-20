---
title: >-
  ICML2026 物理学方向1篇论文解读
description: >-
  1篇ICML2026的物理学方向论文解读，收录 Neural QAOA$^2$等。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "物理学"
  - "论文解读"
  - "论文笔记"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚛️ 物理学

**🧪 ICML2026** · **1** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (1)](../../CVPR2026/physics/index.md) · [🔬 ICLR2026 (2)](../../ICLR2026/physics/index.md) · [🤖 AAAI2026 (2)](../../AAAI2026/physics/index.md) · [🧠 NeurIPS2025 (19)](../../NeurIPS2025/physics/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/physics/index.md) · [🧪 ICML2025 (6)](../../ICML2025/physics/index.md)

**[Neural QAOA$^2$: Differentiable Joint Graph Partitioning and Parameter Initialization for Quantum Combinatorial Optimization](neural_qaoa2_differentiable_joint_graph_partitioning_and_parameter_initializatio.md)**

:   用一个生成-评估神经网络（GEN）一次性地把 QAOA² 的"图分割 + 量子电路参数初始化"两件事联合可微化：评估器学一个高保真的 quantum performance surrogate，生成器在它的梯度引导下吐出离散分区 + 参数初值，配合直通估计器 + 正交补头让端到端可训练；在 183 个 QUBO/Ising/MaxCut 实例（21-1000 变量）上超越启发式 baseline，101 个实例排第一。
