---
title: >-
  ICML2026 代码智能方向2篇论文解读
description: >-
  2篇ICML2026的代码智能方向论文解读，涵盖强化学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "代码智能"
  - "论文解读"
  - "论文笔记"
  - "强化学习"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💻 代码智能

**🧪 ICML2026** · **2** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (48)](../../ACL2026/code_intelligence/index.md) · [📷 CVPR2026 (2)](../../CVPR2026/code_intelligence/index.md) · [🔬 ICLR2026 (20)](../../ICLR2026/code_intelligence/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/code_intelligence/index.md) · [🧠 NeurIPS2025 (22)](../../NeurIPS2025/code_intelligence/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/code_intelligence/index.md)

**[BoostAPR: Boosting Automated Program Repair via Execution-Grounded Reinforcement Learning with Dual Reward Models](boostapr_boosting_automated_program_repair_via_execution-grounded_reinforcement_.md)**

:   BoostAPR 给"用 RL 训 program-repair 模型"造了一套三阶段流水线——execution-verified SFT → 训序列级 + 行级双重 reward → PPO 时用行级模型把序列奖励重新分配到关键 edit lines；在 Qwen2.5-Coder-32B 上把 SWE-bench Verified 从 17.8% 推到 40.7% (+22.9pp)，跨语言迁移到 Defects4J 取 24.8%。

**[HE-SNR: Uncovering Latent Logic via Entropy for Guiding Mid-Training on SWE-bench](he-snr_uncovering_latent_logic_via_entropy_for_guiding_mid-training_on_swe-bench.md)**

:   在 SWE-bench 上传统 PPL 既受"长上下文税"干扰又无法预测 SFT 后的智能体能力，本文提出"熵压缩假说"和 HE-SNR 指标，只在 Top-10 熵大于 $(\ln 3 + \ln 4)/2$ 的"高熵决策点"上算信号噪声比，与下游 SWE-bench 得分的 Pearson 相关达 0.96，Kendall 一致性 0.98。
