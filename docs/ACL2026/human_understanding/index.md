---
title: >-
  ACL2026 人体理解论文汇总 · 3篇论文解读
description: >-
  3篇ACL2026的人体理解方向论文解读，涵盖重识别、联邦学习、扩散模型等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2026"
  - "人体理解"
  - "论文解读"
  - "论文笔记"
  - "重识别"
  - "联邦学习"
  - "扩散模型"
item_list:
  - u: "co-evo_co-evolving_semantic_anchoring_and_style_diversification_for_federated_dg/"
    t: "CO-EVO: Co-evolving Semantic Anchoring and Style Diversification for Federated DG-ReID"
  - u: "hybrid_autoregressive-diffusion_model_for_real-time_sign_language_production/"
    t: "Hybrid Autoregressive-Diffusion Model for Real-Time Sign Language Production"
  - u: "segment_embed_and_align_a_universal_recipe_for_aligning_subtitles_to_signing/"
    t: "Segment, Embed, and Align: A Universal Recipe for Aligning Subtitles to Signing"
item_total: 3
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧑 人体理解

**💬 ACL2026** · **3** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (4)](../../ICML2026/human_understanding/index.md) · [📷 CVPR2026 (60)](../../CVPR2026/human_understanding/index.md) · [🔬 ICLR2026 (8)](../../ICLR2026/human_understanding/index.md) · [🤖 AAAI2026 (19)](../../AAAI2026/human_understanding/index.md) · [🧠 NeurIPS2025 (20)](../../NeurIPS2025/human_understanding/index.md) · [📹 ICCV2025 (40)](../../ICCV2025/human_understanding/index.md)

**[CO-EVO: Co-evolving Semantic Anchoring and Style Diversification for Federated DG-ReID](co-evo_co-evolving_semantic_anchoring_and_style_diversification_for_federated_dg.md)**

:   CO-EVO 针对联邦域泛化行人重识别（FedDG-ReID）中的"语义-风格冲突"，提出 CSA（相机不变语义锚定）学习冻结的身份级文本原型作为"引力中心"+ GSD（全局风格多样化）用轻量 GCSB（全局相机风格库）合成真实跨域扰动，二者耦合优化在 Market-1501/MSMT17/CUHK03 leave-one-out 上 ViT mAP 平均比 SOTA 提升 14 个点（34.1→48.1）。

**[Hybrid Autoregressive-Diffusion Model for Real-Time Sign Language Production](hybrid_autoregressive-diffusion_model_for_real-time_sign_language_production.md)**

:   这篇论文提出 HybridSign，把自回归逐帧生成和 flow-based diffusion 细化结合起来，并加入三专家多尺度姿态表示与 confidence-aware causal attention，在 PHOENIX14T 和 How2Sign 上取得更好的手语生成质量-延迟折中。

**[Segment, Embed, and Align: A Universal Recipe for Aligning Subtitles to Signing](segment_embed_and_align_a_universal_recipe_for_aligning_subtitles_to_signing.md)**

:   SEA 将连续手语视频的字幕对齐拆成 sign segmentation、text-sign embedding 和 episode-level dynamic programming 三步，在 BOBSL、How2Sign、WMT-SLT SRF、SwissSLi 四个数据集上取得 SOTA F1@0.50，并能在 CPU 上高效处理长视频。
