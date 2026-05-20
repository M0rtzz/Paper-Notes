---
title: >-
  ICML2026 对齐 / RLHF方向7篇论文解读
description: >-
  7篇ICML2026的对齐 / RLHF 方向论文解读，涵盖对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "对齐 / RLHF"
  - "论文解读"
  - "论文笔记"
  - "对齐/RLHF"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚖️ 对齐 / RLHF

**🧪 ICML2026** · **7** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (26)](../../ACL2026/llm_alignment/index.md) · [📷 CVPR2026 (10)](../../CVPR2026/llm_alignment/index.md) · [🔬 ICLR2026 (41)](../../ICLR2026/llm_alignment/index.md) · [🤖 AAAI2026 (19)](../../AAAI2026/llm_alignment/index.md) · [🧠 NeurIPS2025 (50)](../../NeurIPS2025/llm_alignment/index.md) · [📹 ICCV2025 (2)](../../ICCV2025/llm_alignment/index.md)

🔥 **高频主题：** 对齐/RLHF ×6

**[BLOCK-EM: Preventing Emergent Misalignment via Latent Blocking](block-em_preventing_emergent_misalignment_via_latent_blocking.md)**

:   BLOCK-EM 用 SAE 找到一小撮"因果地控制 emergent misalignment"的内部 latent，然后在窄域 SFT 时加一个 one-sided 正则，禁止模型把这些 latent 朝"失对齐方向"放大——在 6 个 fine-tuning 域上把 emergent misalignment 平均砍掉 93%，同时几乎不损伤 in-domain 任务表现。

**[$f$-Divergence Regularized RLHF: Two Tales of Sampling and Unified Analyses](f-divergence_regularized_rlhf_two_tales_of_sampling_and_unified_analyses.md)**

:   本文给在线 RLHF 在**通用 $f$-divergence 正则**下首次建立 $O(\log T)$ regret 和 $O(1/T)$ 次优 gap 上界，提出两套采样策略：(1) 基于 optimism in face of uncertainty 加 bonus 项；(2) 一个新颖的 **"derivative-as-uncertainty"** 视角——把 $f'$ 当作不确定性信号，从而设计 derivative-based 采样而无需在每轮显式估计 confidence bound。

**[Pareto-Guided Optimal Transport for Multi-Reward Alignment](pareto-guided_optimal_transport_for_multi-reward_alignment.md)**

:   PG-OT 把「多奖励文生图对齐」从「加权全局求和」改成「为每个 prompt 单独构造 Pareto 前沿、用 Sinkhorn 最优传输把被支配样本传到前沿」，并引入 Joint Domination Rate / Joint Collapse Rate 两个新指标暴露平均值掩盖的奖励 hacking，在 Parti-Prompts 上 JDR₂ 47.98% 比强基线提升 11%，人评胜率近 80%。

**[Reward Modeling from Natural Language Human Feedback](reward_modeling_from_natural_language_human_feedback.md)**

:   本文指出在二元偏好奖励上训练的 generative reward model (GRM) 严重存在"猜对偏好但 critique 错误"的 outcome-process 不一致（20-30%、最高 44%），并提出 RM-NLHF：把模型 critique 与人工 critique 的核心论点相似度作为额外过程奖励，并用 MetaRM 自动预测过程奖励、在线随策略更新，从而在多个 benchmark 上稳定超过 outcome-only GRPO 训练的 SOTA GRM。

**[The Realignment Problem: When Right becomes Wrong in LLMs](the_realignment_problem_when_right_becomes_wrong_in_llms.md)**

:   本文把"模型部署后政策变了怎么办"形式化为 Realignment 问题,提出 TRACE 框架:用更强的 proxy 模型把已有 preference pair 三分类 (Invert / Punish / Retain) 后用混合 IPO+NPO+KL 目标做手术式再对齐,无需新一轮人工标注就能跟上政策漂移。

**[Toward Stable Value Alignment: Introducing Independent Modules for Consistent Value Guidance](toward_stable_value_alignment_introducing_independent_modules_for_consistent_val.md)**

:   本文提出 SVGT，把价值对齐从"嵌入 backbone 参数/激活"改为"挂一个独立的价值模块"，先在隔离的 value space 里持续判断当前 hidden state 的安全方向，再用一组可学习的 Bridge Token 作为注意力锚点显式引导生成轨迹，在四种 backbone 上把有害分数普遍降低 70% 以上且几乎不损失流畅度。

**[TUR-DPO: Topology- and Uncertainty-Aware Direct Preference Optimization](tur-dpo_topology-_and_uncertainty-aware_direct_preference_optimization.md)**

:   TUR-DPO 在 DPO 的偏好 logit 上同时叠加一个"语义+拓扑结构"塑形奖励差和一个"按每对样本不确定性"动态降权的实例权重，让模型在保持 RL-free 训练简洁性的同时，显式奖励推理过程的结构合理性并削弱脆弱偏好对的影响，从而在 GSM8K / MATH / BBH / QA 等推理类任务上系统超过 DPO 与 IPO，并在多数任务上追平 PPO。
