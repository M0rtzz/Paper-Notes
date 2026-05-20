---
title: >-
  ICML2026 人体理解方向1篇论文解读
description: >-
  1篇ICML2026的人体理解方向论文解读，收录 MotionGRPO等。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "人体理解"
  - "论文解读"
  - "论文笔记"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧑 人体理解

**🧪 ICML2026** · **1** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (1)](../../ACL2026/human_understanding/index.md) · [📷 CVPR2026 (53)](../../CVPR2026/human_understanding/index.md) · [🔬 ICLR2026 (7)](../../ICLR2026/human_understanding/index.md) · [🤖 AAAI2026 (15)](../../AAAI2026/human_understanding/index.md) · [🧠 NeurIPS2025 (18)](../../NeurIPS2025/human_understanding/index.md) · [📹 ICCV2025 (38)](../../ICCV2025/human_understanding/index.md)

**[MotionGRPO: Overcoming Low Intra-Group Diversity in GRPO-Based Egocentric Motion Recovery](motiongrpo_overcoming_low_intra-group_diversity_in_grpo-based_egocentric_motion_.md)**

:   MotionGRPO 把 head-mounted 设备的第一人称全身动作恢复转化为扩散采样上的 MDP，用 GRPO 配合"轨迹条件感知模型 + 4 个 joint-level 子奖励"的混合奖励做后训练；同时识别出"输入条件太强、组内样本几乎一样导致 advantage 方差消失"这一致命瓶颈，并用 Perlin 噪声注入条件来恢复组内多样性，在 AMASS/RICH 上把 MPJPE 从 EgoAllo 的 124.985 mm 降到 114.207 mm。
