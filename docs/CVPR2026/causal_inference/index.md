---
title: >-
  CVPR2026 因果推理方向 4篇论文解读
description: >-
  4篇CVPR2026 因果推理方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔗 因果推理

**📷 CVPR2026** · **4** 篇论文解读

**[Fighting Hallucinations with Counterfactuals: Diffusion-Guided Perturbations for LVLM Hallucination Suppression](cipher_counterfactual_diffusion_hallucination_sup.md)**

:   提出 CIPHER，一种免训练的 test-time 幻觉抑制方法——通过扩散模型生成语义篡改但结构保持的反事实图像，将其与原图在 LVLM 隐层中的表示差异做 SVD 分解提取幻觉子空间，推理时将隐状态投影到该子空间的正交补空间，首次从视觉模态入手定位和消除 LVLM 幻觉。

**[Fighting Hallucinations with Counterfactuals: Diffusion-Guided Perturbations for LVLM Hallucination Suppression](fighting_hallucinations_with_counterfactuals_diffusion-guided_perturbations_for_.md)**

:   提出 CIPHER，一种无需训练的测试时幻觉抑制方法：离线阶段用扩散模型生成反事实图像构建 OHC-25K 数据集，通过 SVD 提取视觉幻觉子空间；推理阶段将隐状态投影到该子空间的正交补空间，在不修改模型参数、不增加推理开销的前提下显著降低 LVLM 的视觉幻觉。

**[MaskDiME: Adaptive Masked Diffusion for Precise and Efficient Visual Counterfactual Explanations](maskdime_adaptive_masked_diffusion_for_precise_and_efficient_visual_counterfactu.md)**

:   提出 MaskDiME，一个免训练的扩散框架，通过自适应双掩码机制将全局分类器引导转化为决策驱动的局部编辑，实现精确高效的视觉反事实解释，推理速度比 DiME 快 30 倍以上，GPU 内存仅为 ACE/RCSB 的十分之一。

**[Retrieving Counterfactuals Improves Visual In-Context Learning](retrieving_counterfactuals_improves_visual_in-context_learning.md)**

:   提出 CIRCLES 框架，通过属性引导的 composed image retrieval 检索反事实示例，构建因果+相关性双通道 in-context demonstration，显著提升 VLM 的细粒度视觉推理能力。
