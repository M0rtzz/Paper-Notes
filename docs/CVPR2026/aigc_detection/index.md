---
title: >-
  CVPR2026 AIGC检测论文汇总 · 4篇论文解读
description: >-
  4篇CVPR2026的 AIGC 检测方向论文解读，涵盖机器人、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "AIGC 检测"
  - "论文解读"
  - "论文笔记"
  - "机器人"
  - "推理"
item_list:
  - u: "fine-grained_image_aesthetic_assessment_learning_discriminative_scores_from_rela/"
    t: "Fine-grained Image Aesthetic Assessment: Learning Discriminative Scores from Relative Ranks"
  - u: "frame_forensic_routing_and_adaptive_multi-path_evidence_fusion_for_image_manipul/"
    t: "FRAME: Forensic Routing and Adaptive Multi-path Evidence Fusion for Image Manipulation Detection"
  - u: "quality-aware_calibration_for_ai-generated_image_detection_in_the_wild/"
    t: "Quality-Aware Calibration for AI-Generated Image Detection in the Wild"
  - u: "realign_generalizable_image_forgery_detection_via_reasoning-aligned_representati/"
    t: "ReAlign: Generalizable Image Forgery Detection via Reasoning-Aligned Representation"
item_total: 4
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔎 AIGC 检测

**📷 CVPR2026** · **4** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (7)](../../ICML2026/aigc_detection/index.md) · [💬 ACL2026 (16)](../../ACL2026/aigc_detection/index.md) · [🔬 ICLR2026 (6)](../../ICLR2026/aigc_detection/index.md) · [🤖 AAAI2026 (2)](../../AAAI2026/aigc_detection/index.md) · [🧠 NeurIPS2025 (9)](../../NeurIPS2025/aigc_detection/index.md) · [💬 ACL2025 (15)](../../ACL2025/aigc_detection/index.md)

**[Fine-grained Image Aesthetic Assessment: Learning Discriminative Scores from Relative Ranks](fine-grained_image_aesthetic_assessment_learning_discriminative_scores_from_rela.md)**

:   定义"细粒度图像美学评估"新任务，构建含32,217张图像/10,028个系列的FGAesthetics基准，提出FGAesQ模型：通过差异保留Tokenization（DiffToken）+ 对比文本辅助对齐（CTAlign）+ 排序感知回归（RankReg）从相对排序中学习判别性审美评分，在细粒度场景准确率0.779的同时保持粗粒度SRCC 0.770。

**[FRAME: Forensic Routing and Adaptive Multi-path Evidence Fusion for Image Manipulation Detection](frame_forensic_routing_and_adaptive_multi-path_evidence_fusion_for_image_manipul.md)**

:   FRAME 把一堆传统取证算法（ELA、DCT、噪声、CFA、copy-move 等）组织成一个"取证 supernet"，对每张待检图像用 GNN 预测器挑出最合适的若干条"分析路径"并把它们的证据图融合，从而避免"单一检测器不通用 + 固定融合稀释信号"的老问题，在多个跨域测试集上同时把检测 AUC 和像素级定位刷到优于固定组合和端到端深度模型。

**[Quality-Aware Calibration for AI-Generated Image Detection in the Wild](quality-aware_calibration_for_ai-generated_image_detection_in_the_wild.md)**

:   针对同一张图在网络传播中产生的多个画质各异的"近重复版本"，本文提出 QuAD：用无参考 IQA 估计每个版本的画质，再用画质作条件对取证检测器的 logit 做高斯校准并加权融合，让低画质版本少说话、高画质版本多说话，平均把六个 SOTA 检测器的平衡准确率提升约 8 个百分点。

**[ReAlign: Generalizable Image Forgery Detection via Reasoning-Aligned Representation](realign_generalizable_image_forgery_detection_via_reasoning-aligned_representati.md)**

:   ReAlign 先用 GRPO 训出一个会"讲理由"的多模态大模型 AIGI-R1，再把它生成的推理文本作为"桥梁"，通过对比学习把推理文本空间蒸馏进一个轻量 CLIP 检测器，让小模型同时继承大模型的跨域泛化和语义错误敏感性，推理时只用图像编码器即可，在 AIGCDetectBenchmark / AIGI-Holmes / 自建 UltraSynth-10k 上都拿到 SOTA（mAcc 96.14% / 99.44% / 97.09%）。
