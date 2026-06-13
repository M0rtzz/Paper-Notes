---
title: >-
  CVPR2026 多智能体论文汇总 · 1篇论文解读
description: >-
  1篇CVPR2026的多智能体方向论文解读，涵盖 Agent、少样本学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "多智能体"
  - "论文解读"
  - "论文笔记"
  - "Agent"
  - "少样本学习"
item_list:
  - u: "motor-bench_a_real-world_dataset_and_multi-agent_framework_for_zero-shot_human_m/"
    t: "MOTOR-Bench: A Real-world Dataset and Multi-agent Framework for Zero-shot Human Mental State Understanding"
item_total: 1
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👥 多智能体

**📷 CVPR2026** · **1** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (15)](../../ICML2026/multi_agent/index.md) · [💬 ACL2026 (38)](../../ACL2026/multi_agent/index.md) · [🔬 ICLR2026 (15)](../../ICLR2026/multi_agent/index.md) · [🤖 AAAI2026 (26)](../../AAAI2026/multi_agent/index.md) · [🧠 NeurIPS2025 (17)](../../NeurIPS2025/multi_agent/index.md) · [🧪 ICML2025 (7)](../../ICML2025/multi_agent/index.md)

**[MOTOR-Bench: A Real-world Dataset and Multi-agent Framework for Zero-shot Human Mental State Understanding](motor-bench_a_real-world_dataset_and_multi-agent_framework_for_zero-shot_human_m.md)**

:   针对「从可观察行为推断深层心理状态」缺少结构化标注这一空白，本文构建了真实课堂协作学习场景的多模态数据集 MOTOR-dataset（1,440 段视频，行为/认知/情绪三维标注），并提出基于自我调节学习理论(SRL)的推理型多智能体框架 MOTOR-MAS——三个专职 agent 按「行为→认知→情绪」顺序级联推理，把前一阶段的预测当锚点喂给后一阶段，零样本下 Macro-F1 达 42.77，比最强单模型基线高 15.93 分。
