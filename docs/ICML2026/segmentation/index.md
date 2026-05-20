---
title: >-
  ICML2026 语义分割方向4篇论文解读
description: >-
  4篇ICML2026的语义分割方向论文解读，涵盖语义分割、语音、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "语义分割"
  - "论文解读"
  - "论文笔记"
  - "语音"
  - "对抗鲁棒"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✂️ 语义分割

**🧪 ICML2026** · **4** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (1)](../../ACL2026/segmentation/index.md) · [📷 CVPR2026 (83)](../../CVPR2026/segmentation/index.md) · [🔬 ICLR2026 (11)](../../ICLR2026/segmentation/index.md) · [🤖 AAAI2026 (31)](../../AAAI2026/segmentation/index.md) · [🧠 NeurIPS2025 (48)](../../NeurIPS2025/segmentation/index.md) · [📹 ICCV2025 (73)](../../ICCV2025/segmentation/index.md)

🔥 **高频主题：** 语义分割 ×2

**[LightAVSeg: Lightweight Audio-Visual Segmentation](lightavseg_lightweight_audio-visual_segmentation.md)**

:   LightAVSeg 通过解耦 "语义筛选 (what)" 和 "空间定位 (where)"，用全局通道调制替换 $\mathcal{O}(N^2)$ 的跨模态注意力，让 AVS 模型在 20.5M 参数下达到 50.4 mIoU (MS3)，并在 Snapdragon 8 Elite 上做到 163.4 ms 的端侧延迟，比 AVSegFormer-R50 快约 $8\times$。

**[Segment Anything with Robust Uncertainty-Accuracy Correlation](segment_anything_with_robust_uncertainty-accuracy_correlation.md)**

:   针对 SAM 系列只输出 mask-level 单一置信度、在域漂移下出现"Mask-level Confidence Confusion"的问题，本文给 SAM2 接上 Weibull 双粒度贝叶斯 mask decoder 做像素级 epistemic 估计，并配以受人类视觉启发的 style + deformation 协同对抗扰动 + 校准损失，让 uncertainty 在 23 个 zero-shot 目标域始终与误差对齐，平均 J&F 达 79.87 同时不确定性图变得显著可信。

**[SEMIR: Semantic Minor-Induced Representation Learning on Graphs for Visual Segmentation](semir_semantic_minor-induced_representation_learning_on_graphs_for_visual_segmen.md)**

:   SEMIR 把体素栅格当作母图 $G$，通过参数化的边收缩 / 节点删除 / 边删除把它压成一张「边界对齐」的图 minor $H$（节点数从 $\sim10^7$ 降到 $\sim10^3$），用 5–20 张少样本黑盒优化 $\Theta$ 最大化边界 Dice，再在 minor 上用 GNN 做超节点分类，最后通过 minor 与体素之间的双射 exact lifting 回到原栅格——在 BraTS / KiTS / LiTS 三大肿瘤分割任务的少数类 Dice 上稳定超过 nnU-Net，且仅需 16GB T4 GPU。

**[UGround: Towards Unified Visual Grounding with Unrolled Transformers](uground_towards_unified_visual_grounding_with_unrolled_transformers.md)**

:   UGround 把 LMM-based 视觉定位从"用最后一层 $\langle\text{SEG}\rangle$ token 当 prompt"的范式翻转为"用动态选中的中间层相似度图当 prompt"，通过强化学习策略 SSC 让 $\langle\text{SEG}\rangle$ 滑过所有 transformer 层、把相似度图同时当作 SAM 的软 logit mask 和反向监督信号，首次在单一框架内统一了 RES / RS / FP-RES / gRES / Multi-RS 五种视觉定位任务，并在 ReasonSeg test 上 cIoU +9.0%、gRefCOCO val N-acc +12.1%。
