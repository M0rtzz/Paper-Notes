---
title: >-
  ACL2026 目标检测方向5篇论文解读
description: >-
  5篇ACL2026的目标检测方向论文解读，涵盖多模态、信息抽取、模型压缩、LLM、扩散模型、RAG等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎯 目标检测

**💬 ACL2026** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (39)](../../CVPR2026/object_detection/index.md) · [🔬 ICLR2026 (9)](../../ICLR2026/object_detection/index.md) · [🤖 AAAI2026 (17)](../../AAAI2026/object_detection/index.md) · [🧠 NeurIPS2025 (17)](../../NeurIPS2025/object_detection/index.md) · [📹 ICCV2025 (28)](../../ICCV2025/object_detection/index.md) · [🧪 ICML2025 (8)](../../ICML2025/object_detection/index.md)

**[E2E-GMNER: End-to-End Generative Grounded Multimodal Named Entity Recognition](e2e-gmner_end-to-end_generative_grounded_multimodal_named_entity_recognition.md)**

:   提出E2E-GMNER，首个将实体识别、语义分类、视觉定位和隐式知识推理统一在单一多模态大语言模型中的端到端GMNER框架，通过CoT推理自适应判断视觉/知识线索的可用性，并引入高斯风险感知框扰动（GRBP）提升生成式框预测的鲁棒性。

**[Evaluating Memory Capability in Continuous Lifelog Scenario](evaluating_memory_capability_in_continuous_lifelog_scenario.md)**

:   本文提出LifeDialBench，一个评估连续生活日志场景下记忆能力的基准（含7天真实数据的EgoMem和1年模拟的LifeMem），引入在线评估协议确保时间因果性，反直觉地发现简单RAG基线一致优于复杂记忆系统。

**[Evolutionary Negative Module Pruning for Better LoRA Merging](evolutionary_negative_module_pruning_for_better_lora_merging.md)**

:   提出 ENMP 方法，通过进化搜索策略发现并剪除 LoRA 合并中降低性能的"负面模块"，作为即插即用的增强手段，在 NLP 和视觉领域全面提升现有合并算法的效果。

**[GigaCheck: Detecting LLM-generated Content via Object-Centric Span Localization](gigacheck_detecting_llm-generated_content_via_object-centric_span_localization.md)**

:   提出 GigaCheck，一个双策略框架：文档级使用微调 LLM 进行分类，片段级创新地将 AI 生成文本片段视为"目标"，用 DETR-like 架构实现端到端的字符级定位。

**[Retrievals Can Be Detrimental: Unveiling the Backdoor Vulnerability of Retrieval-Augmented Diffusion Models](retrievals_can_be_detrimental_unveiling_the_backdoor_vulnerability_of_retrieval-.md)**

:   提出 BadRDM，首个针对检索增强扩散模型（RDM）的后门攻击框架，通过恶意对比学习微调检索器建立触发词到毒性代理图像的捷径，在类条件和 T2I 两种任务中分别达到 90.9% 和 96.4% 攻击成功率，同时保持良性生成质量。
