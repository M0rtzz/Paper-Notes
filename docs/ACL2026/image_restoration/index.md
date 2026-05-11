---
title: >-
  ACL2026 图像恢复方向5篇论文解读
description: >-
  5篇ACL2026的图像恢复方向论文解读，涵盖扩散模型、LLM、RAG、强化学习、图像恢复等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🖼️ 图像恢复

**💬 ACL2026** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (39)](../../CVPR2026/image_restoration/index.md) · [🔬 ICLR2026 (15)](../../ICLR2026/image_restoration/index.md) · [🤖 AAAI2026 (13)](../../AAAI2026/image_restoration/index.md) · [🧠 NeurIPS2025 (26)](../../NeurIPS2025/image_restoration/index.md) · [📹 ICCV2025 (29)](../../ICCV2025/image_restoration/index.md) · [🧪 ICML2025 (5)](../../ICML2025/image_restoration/index.md)

🔥 **高频主题：** 扩散模型 ×3 · LLM ×2

**[CreditDecoding: Accelerating Parallel Decoding in Diffusion Large Language Models with Trace Credit](creditdecoding_accelerating_parallel_decoding_in_diffusion_large_language_models.md)**

:   本文提出 CreditDecoding，一种无需训练的并行解码加速方法，通过累积 token 级历史证据（轨迹信用）来增强正确但置信度不足的 token，在 LLaDA-8B-Instruct 上实现最高 5.48 倍加速且准确率提升 0.48。

**[Diffusion-CAM: Faithful Visual Explanations for dMLLMs](diffusion-cam_faithful_visual_explanations_for_dmllms.md)**

:   提出 Diffusion-CAM，首个专为扩散式多模态大语言模型（dMLLM）设计的可解释性方法，通过在去噪轨迹中提取结构有效的中间表征并配合四个后处理模块（自适应核去噪、分布感知置信门控、上下文背景衰减、单实例因果去偏），在 COCO Caption 和 GranDf 上显著超越自回归 CAM 基线。

**[Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation](learning_to_extract_rational_evidence_via_reinforcement_learning_for_retrieval-a.md)**

:   提出 EviOmni，通过"先推理再提取"的范式学习从检索文档中提取理性证据：将证据推理和证据提取整合为统一轨迹，用知识 token 掩码避免信息泄露，通过 GRPO 以可验证奖励优化，在 5 个基准上以极高压缩比（~38x）取得优于全文检索的准确率。

**[Lost in Diffusion: Uncovering Hallucination Patterns and Failure Modes in Diffusion Large Language Models](lost_in_diffusion_uncovering_hallucination_patterns_and_failure_modes_in_diffusi.md)**

:   首次系统性地对比扩散大语言模型（dLLM）与自回归（AR）对应模型的幻觉模式，揭示当前 dLLM 幻觉倾向更高，并识别出三种扩散特有的失败模式：过早终止、不完全去噪和上下文入侵。

**[Purging the Gray Zone: Latent-Geometric Denoising for Precise Knowledge Boundary Awareness](purging_the_gray_zone_latent-geometric_denoising_for_precise_knowledge_boundary_.md)**

:   本文提出 GeoDe 框架，通过在 LLM 隐空间中训练线性探针构建真值超平面，利用样本到超平面的几何距离作为置信度信号来筛选高质量弃权微调数据，有效消除决策边界附近的"灰色地带"噪声，显著提升模型的真实性和可靠性。
