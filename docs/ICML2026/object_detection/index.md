---
title: >-
  ICML2026 目标检测方向1篇论文解读
description: >-
  1篇ICML2026的目标检测方向论文解读，收录 Smoothing Slot Attention Itera等。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "目标检测"
  - "论文解读"
  - "论文笔记"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎯 目标检测

**🧪 ICML2026** · **1** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (36)](../../CVPR2026/object_detection/index.md) · [🔬 ICLR2026 (9)](../../ICLR2026/object_detection/index.md) · [🤖 AAAI2026 (15)](../../AAAI2026/object_detection/index.md) · [🧠 NeurIPS2025 (17)](../../NeurIPS2025/object_detection/index.md) · [📹 ICCV2025 (26)](../../ICCV2025/object_detection/index.md) · [🧪 ICML2025 (8)](../../ICML2025/object_detection/index.md)

**[Smoothing Slot Attention Iterations and Recurrences](smoothing_slot_attention_iterations_and_recurrences.md)**

:   针对 Slot Attention 在图像与视频对象中心学习中"冷启动查询信息不足"和"首帧/非首帧聚合变换被强行统一"两个长期被忽视的痛点，作者提出 SmoothSA：用一个自蒸馏的小预热模块给查询注入样本信息，同时让首帧跑完整三次迭代、非首帧只跑一次，从而在图像和视频两个 OCL 基准上同时刷新 SOTA。
