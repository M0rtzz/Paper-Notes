---
title: >-
  ICML2026 自动驾驶方向1篇论文解读
description: >-
  1篇ICML2026的自动驾驶方向论文解读，涵盖自动驾驶等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "自动驾驶"
  - "论文解读"
  - "论文笔记"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🚗 自动驾驶

**🧪 ICML2026** · **1** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (88)](../../CVPR2026/autonomous_driving/index.md) · [🔬 ICLR2026 (18)](../../ICLR2026/autonomous_driving/index.md) · [🤖 AAAI2026 (58)](../../AAAI2026/autonomous_driving/index.md) · [🧠 NeurIPS2025 (49)](../../NeurIPS2025/autonomous_driving/index.md) · [📹 ICCV2025 (93)](../../ICCV2025/autonomous_driving/index.md) · [🧪 ICML2025 (11)](../../ICML2025/autonomous_driving/index.md)

**[DeepSight: Long-Horizon World Modeling via Latent States Prediction for End-to-End Autonomous Driving](deepsight_long-horizon_world_modeling_via_latent_states_prediction_for_end-to-en.md)**

:   DeepSight 把"未来世界预测"从显式像素重建（codebook 单帧）换成在 BEV 空间对 DINOv3 语义特征做**多帧并行隐式预测**，再叠加一个按需触发的 Adaptive Chain-of-Thought，让 Qwen2.5-VL-3B 在 Bench2Drive 闭环上 Driving Score 86.23 (+7.39)、Success Rate 71.36% (+13.63)，且只多 ~4% 推理延迟。
