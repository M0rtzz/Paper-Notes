---
title: >-
  ACL2025 图像恢复方向 2篇论文解读
description: >-
  2篇ACL2025 图像恢复方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🖼️ 图像恢复

**💬 ACL2025** · **2** 篇论文解读

**[DiffuseDef: Improved Robustness to Adversarial Attacks via Iterative Denoising](diffusedef_adversarial_defense.md)**

:   DiffuseDef 在编码器与分类器之间插入一个扩散去噪层，训练时学习预测隐状态噪声，推理时对隐表示加噪→迭代去噪→集成平均，以即插即用的方式大幅提升文本分类模型在黑盒和白盒对抗攻击下的鲁棒性。

**[PreP-OCR: A Complete Pipeline for Document Image Restoration and Enhanced OCR Accuracy](prep-ocr_a_complete_pipeline_for_document_image_restoration_and_enhanced_ocr_acc.md)**

:   提出 PreP-OCR 两阶段流水线：先用合成退化数据训练的 ResShift 模型修复历史文档图像（多方向 patch 提取+中值融合），再用 ByT5 做 OCR 后语义纠错，在 13,831 页真实历史文档上降低 CER 63.9-70.3%。
