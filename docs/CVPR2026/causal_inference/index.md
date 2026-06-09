---
title: >-
  CVPR2026 因果推理方向3篇论文解读
description: >-
  3篇CVPR2026的因果推理方向论文解读，涵盖扩散模型等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "因果推理"
  - "论文解读"
  - "论文笔记"
  - "扩散模型"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔗 因果推理

**📷 CVPR2026** · **3** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (16)](../../ICML2026/causal_inference/index.md) · [💬 ACL2026 (7)](../../ACL2026/causal_inference/index.md) · [🔬 ICLR2026 (18)](../../ICLR2026/causal_inference/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/causal_inference/index.md) · [🧠 NeurIPS2025 (20)](../../NeurIPS2025/causal_inference/index.md) · [📹 ICCV2025 (2)](../../ICCV2025/causal_inference/index.md)

🔥 **高频主题：** 扩散模型 ×2

**[Fighting Hallucinations with Counterfactuals: Diffusion-Guided Perturbations for LVLM Hallucination Suppression](fighting_hallucinations_with_counterfactuals_diffusion-guided_perturbations_for_.md)**

:   提出 CIPHER，一种无需训练的测试时幻觉抑制方法：离线阶段用扩散模型生成反事实图像构建 OHC-25K 数据集，通过 SVD 提取视觉幻觉子空间；推理阶段将隐状态投影到该子空间的正交补空间，在不修改模型参数、不增加推理开销的前提下显著降低 LVLM 的视觉幻觉。

**[MaskDiME: Adaptive Masked Diffusion for Precise and Efficient Visual Counterfactual Explanations](maskdime_adaptive_masked_diffusion_for_precise_and_efficient_visual_counterfactu.md)**

:   提出 MaskDiME，一个免训练的扩散框架，通过自适应双掩码机制将全局分类器引导转化为决策驱动的局部编辑，实现精确高效的视觉反事实解释，推理速度比 DiME 快 30 倍以上，GPU 内存仅为 ACE/RCSB 的十分之一。

**[Retrieving Counterfactuals Improves Visual In-Context Learning](retrieving_counterfactuals_improves_visual_in-context_learning.md)**

:   提出 CIRCLES 框架，通过属性引导的 composed image retrieval 检索反事实示例，构建因果+相关性双通道 in-context demonstration，显著提升 VLM 的细粒度视觉推理能力。
