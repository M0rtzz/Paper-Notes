---
title: >-
  ICML2026 因果推理方向3篇论文解读
description: >-
  3篇ICML2026的因果推理方向论文解读，收录 Causal Fine-Tuning under Laten、Controllable Generative Sandbo、The (Marginal) Value of a Search Ad等。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICML2026"
  - "因果推理"
  - "论文解读"
  - "论文笔记"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔗 因果推理

**🧪 ICML2026** · **3** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (6)](../../ACL2026/causal_inference/index.md) · [📷 CVPR2026 (3)](../../CVPR2026/causal_inference/index.md) · [🔬 ICLR2026 (18)](../../ICLR2026/causal_inference/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/causal_inference/index.md) · [🧠 NeurIPS2025 (21)](../../NeurIPS2025/causal_inference/index.md) · [📹 ICCV2025 (2)](../../ICCV2025/causal_inference/index.md)

**[Causal Fine-Tuning under Latent Confounded Shift](causal_fine-tuning_under_latent_confounded_shift.md)**

:   本文提出 Causal Fine-Tuning (CFT)：在标准 BERT 微调里嵌入一个 SCM 启发的"高级稳定特征 $C$ + 低级混杂敏感特征 $\Phi$"分解，并用 front-door 风格的 do-calculus 调整公式做预测，在文本伪相关注入攻击下显著优于 SFT/SWA/WISE 等单域泛化基线。

**[Controllable Generative Sandbox for Causal Inference](controllable_generative_sandbox_for_causal_inference.md)**

:   本文提出 CausalMix：一个变分生成框架，把数据类型特定的 multi-head decoder + Bayesian Gaussian 混合潜在 prior 与三类可独立调控的因果"旋钮"（overlap $\alpha(X)$、CATE 函数 $\tau(X)$、未观测混杂 $\kappa(X,T)$）联合优化，从而在保持真实数据分布 fidelity 的前提下让用户自由设计 counterfactual benchmark，在 mCRPC（前列腺癌）真实病例上验证 CausalMix 既能高保真复现 mixed-type 表格，又能稳定地按需注入 overlap / confounding / 异质效应，用作 CATE 估计器的可控 stress test。

**[The (Marginal) Value of a Search Ad: An Online Causal Framework for Repeated Second-price Auctions](the_marginal_value_of_a_search_ad_an_online_causal_framework_for_repeated_second.md)**

:   本文把搜索广告的真实价值建模为"赢拍 vs 输拍"的 treatment effect，在重复二价拍卖（SPA）binary 反馈下设计了一个利用支付规则的在线因果学习算法，得到 $\widetilde\Theta(\sqrt{dT})$ 的极小极大最优 regret，比同设定下的一价拍卖严格更易学。
