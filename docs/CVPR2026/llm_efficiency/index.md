---
title: >-
  CVPR2026 LLM效率论文汇总 · 2篇论文解读
description: >-
  2篇CVPR2026的 LLM 效率方向论文解读，收录 Gated KalmaNet、Generalizable Video Quality As等。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "CVPR2026"
  - "LLM 效率"
  - "论文解读"
  - "论文笔记"
item_list:
  - u: "gated_kalmanet_a_fading_memory_layer_through_test-time_ridge_regression/"
    t: "Gated KalmaNet: A Fading Memory Layer Through Test-Time Ridge Regression"
  - u: "generalizable_video_quality_assessment_via_weak-to-strong_learning/"
    t: "Generalizable Video Quality Assessment via Weak-to-Strong Learning"
item_total: 2
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ LLM 效率

**📷 CVPR2026** · **2** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (32)](../../ICML2026/llm_efficiency/index.md) · [💬 ACL2026 (22)](../../ACL2026/llm_efficiency/index.md) · [🔬 ICLR2026 (20)](../../ICLR2026/llm_efficiency/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/llm_efficiency/index.md) · [🧠 NeurIPS2025 (34)](../../NeurIPS2025/llm_efficiency/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/llm_efficiency/index.md)

**[Gated KalmaNet: A Fading Memory Layer Through Test-Time Ridge Regression](gated_kalmanet_a_fading_memory_layer_through_test-time_ridge_regression.md)**

:   把线性状态空间模型（SSM）的状态更新重新解释成"对全部历史做一次测试时岭回归"，用卡尔曼滤波的精确增益替代现有 SSM 的一步梯度近似，并通过自适应正则 + Chebyshev 迭代解决低精度数值不稳与并行训练两大障碍，在短/长上下文及 ImageNet 上都超过 Mamba2、Gated DeltaNet 等线性 SSM。

**[Generalizable Video Quality Assessment via Weak-to-Strong Learning](generalizable_video_quality_assessment_via_weak-to-strong_learning.md)**

:   不依赖任何人工打分标签，用现成 VQA 模型当"弱老师"去监督一个高容量多模态大模型"强学生"，再把学生回收成下一轮老师做迭代，最终在域内持平、在 OOD 上大幅超越所有老师，把 VQA 的 OOD 整体 SRCC 从 0.59 推到 0.745。
