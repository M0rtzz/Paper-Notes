---
title: >-
  ICML2026 AIGC 检测方向4篇论文解读
description: >-
  4篇ICML2026的 AIGC 检测方向论文解读，涵盖 LLM、对抗鲁棒、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "AIGC 检测"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "对抗鲁棒"
  - "多模态"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔎 AIGC 检测

**🧪 ICML2026** · **4** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (11)](../../ACL2026/aigc_detection/index.md) · [📷 CVPR2026 (1)](../../CVPR2026/aigc_detection/index.md) · [🔬 ICLR2026 (6)](../../ICLR2026/aigc_detection/index.md) · [🤖 AAAI2026 (3)](../../AAAI2026/aigc_detection/index.md) · [🧠 NeurIPS2025 (8)](../../NeurIPS2025/aigc_detection/index.md) · [💬 ACL2025 (15)](../../ACL2025/aigc_detection/index.md)

**[Black-Box Detection of LLM-Generated Text Using Generalized Jensen-Shannon Divergence](black-box_detection_of_llm-generated_text_using_generalized_jensen-shannon_diver.md)**

:   SurpMark 把"AI 文本检测"重构成似然无关假设检验：用代理 LM 算 token surprisal 后 k-means 离散成 k 个状态，估计一阶 Markov 转移矩阵，再用广义 Jensen-Shannon 散度（GJS）和预先建好的"人写 / 机写"参考转移矩阵比较，单次前向就给出黑盒、无需重训、无需 per-instance 重采样的判别分数。

**[DGS-Net: Distillation-Guided Gradient Surgery for CLIP Fine-Tuning in AI-Generated Image Detection](dgs-net_distillation-guided_gradient_surgery_for_clip_fine-tuning_in_ai-generate.md)**

:   论文针对"CLIP 微调到 AI 生成图像检测时灾难性遗忘破坏可迁移先验"的问题，提出 DGS-Net：把分类损失的梯度按坐标拆成有害正分量 $g^+$ 与有益负分量 $g^-$，让训练网络的图像梯度先正交投影到冻结 CLIP **文本梯度有害方向**的补空间（Orthogonal Suppression，剔除任务无关语义），再额外对齐到冻结 CLIP **图像梯度有益方向**（Prior Alignment，保住预训练先验），从而在 50 个生成模型上的平均检测精度比 SOTA 高 6.6%。

**[Feature-Augmented Transformers for Robust AI-Text Detection Across Domains and Generators](feature-augmented_transformers_for_robust_ai-text_detection_across_domains_and_g.md)**

:   本文在「单阈值固定协议」下系统暴露 AI 文本检测器在跨数据集/跨生成器 shift 下的脆弱性，并提出把可学注意力加权的手工语言特征与 transformer [CLS] 表征融合，配合 DeBERTa-v3 backbone，在 M4 多域多生成器基准上达到 85.9% balanced accuracy，比强 zero-shot 基线（Fast-DetectGPT、RADAR、Log-Rank）高最多 +7.22。

**[PRPO: Paragraph-level Policy Optimization for Vision-Language Deepfake Detection](prpo_paragraph-level_policy_optimization_for_vision-language_deepfake_detection.md)**

:   作者用一个 115k 带推理标注的 DF-R5 数据集 + 把 CLIP ViT 换成 ConvNeXT 的 DX-LLaVA 架构，并提出 PRPO —— 段落级别 GRPO 变体，每段以 CLIP-文本-图像相似度（VCR）+ 推理-结论多数票一致性（PCR）为 reward，把跨域 deepfake 检测 F1 从 SOTA 75.26% 推到 89.91%，推理质量从 4.2/5 提到 4.55/5。
