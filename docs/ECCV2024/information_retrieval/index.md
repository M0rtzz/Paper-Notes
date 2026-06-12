---
title: >-
  ECCV2024 信息检索/RAG论文汇总 · 4篇论文解读
description: >-
  4篇ECCV2024的信息检索/RAG 方向论文解读，涵盖布局/合成、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2024"
  - "信息检索/RAG"
  - "论文解读"
  - "论文笔记"
  - "布局/合成"
  - "LLM"
item_list:
  - u: "grounding_language_models_for_visual_entity_recognition/"
    t: "Grounding Language Models for Visual Entity Recognition"
  - u: "multi-label_cluster_discrimination_for_visual_representation_learning/"
    t: "Multi-Label Cluster Discrimination for Visual Representation Learning"
  - u: "onerestore_a_universal_restoration_framework_for_composite_degradation/"
    t: "OneRestore: A Universal Restoration Framework for Composite Degradation"
  - u: "towards_open-ended_visual_recognition_with_large_language_models/"
    t: "Towards Open-Ended Visual Recognition with Large Language Model"
item_total: 4
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔍 信息检索/RAG

**🎞️ ECCV2024** · **4** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (22)](../../ICML2026/information_retrieval/index.md) · [💬 ACL2026 (73)](../../ACL2026/information_retrieval/index.md) · [📷 CVPR2026 (9)](../../CVPR2026/information_retrieval/index.md) · [🔬 ICLR2026 (31)](../../ICLR2026/information_retrieval/index.md) · [🤖 AAAI2026 (26)](../../AAAI2026/information_retrieval/index.md) · [🧠 NeurIPS2025 (30)](../../NeurIPS2025/information_retrieval/index.md)

**[Grounding Language Models for Visual Entity Recognition](grounding_language_models_for_visual_entity_recognition.md)**

:   提出 AutoVER——首个将多模态大语言模型（MLLM）应用于大规模视觉实体识别的方法，通过将检索能力集成到 MLLM 内部，结合对比训练和前缀树约束解码，在 Oven-Wiki 基准上大幅超越 PaLI-17B 等先前方法。

**[Multi-Label Cluster Discrimination for Visual Representation Learning](multi-label_cluster_discrimination_for_visual_representation_learning.md)**

:   提出多标签聚类判别方法 MLCD，通过为每张图像分配多个聚类伪标签并设计消歧多标签分类损失，在 LAION-400M 上预训练的 ViT 在 linear probe、zero-shot 分类和检索任务上全面超越 OpenCLIP、FLIP 和 UNICOM。

**[OneRestore: A Universal Restoration Framework for Composite Degradation](onerestore_a_universal_restoration_framework_for_composite_degradation.md)**

:   提出 OneRestore，一种基于 Transformer 的通用图像复原框架，通过场景描述符引导的交叉注意力机制和复合退化复原损失，能在单一模型中自适应地处理低光照、雾、雨、雪及其任意组合的复合退化场景，并支持文本/视觉双模式的可控复原。

**[Towards Open-Ended Visual Recognition with Large Language Model](towards_open-ended_visual_recognition_with_large_language_models.md)**

:   提出 OmniScient Model (OSM)——一个基于冻结 CLIP-ViT + 可训练 MaskQ-Former + 冻结 LLM (Vicuna-7B) 的生成式 mask 分类器，将视觉识别从"从预定义词表中选择类别"转变为"直接生成类别名称"，消除了训练和测试时对预定义词表的依赖，在 COCO 全景分割上超越 DaTaSeg +4.3 PQ。
