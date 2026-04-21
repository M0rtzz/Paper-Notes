---
title: >-
  CVPR2025 信息检索/RAG方向 16篇论文解读
description: >-
  16篇CVPR2025 信息检索/RAG方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔍 信息检索/RAG

**📷 CVPR2025** · **16** 篇论文解读

**[Advancing Myopia To Holism: Fully Contrastive Language-Image Pre-training](advancing_myopia_to_holism_fully_contrastive_language-image_pre-training.md)**

:   将CLIP从传统的一对一(image, text)对比学习升级为多对多(multi-image-embeddings, multi-texts)对比学习范式，通过VLM生成多视角多层次的描述文本、多分支视觉编码器输出多种视觉embedding，实现更全面的视觉语言对齐，在检索/分类/密集任务上大幅超越baseline。

**[ChatHuman: Chatting about 3D Humans with Tools](chathuman_chatting_about_3d_humans_with_tools.md)**

:   提出 ChatHuman，一个基于 LLM 的语言驱动系统，通过自动选择和集成专门的 3D 人体分析工具（3D 姿态估计、形状恢复、接触检测、人物交互分析、情感识别等），利用学术论文作为工具使用说明和 RAG（检索增强生成）创建 in-context 示例以管理新工具，在工具选择准确率和整体 3D 人体任务性能上超越现有 LLM 模型。

**[DocoPilot: Improving Multimodal Models for Document-Level Understanding](docopilot_improving_multimodal_models_for_document-level_understanding.md)**

:   本文构建了 Doc-750K——一个包含 758K 问答对和 3.1M 图像的高质量文档级多模态数据集，并基于此训练原生文档理解模型 Docopilot，在 MM-NIAH 上超越 InternVL2-8B 达 19.9 个百分点，无需 RAG 即可高效处理多页文档。

**[EZSR: Event-based Zero-Shot Recognition](ezsr_event-based_zero-shot_recognition.md)**

:   提出 EZSR，通过将事件相机数据与 CLIP 的 RGB 嵌入空间对齐，实现基于事件数据的零样本物体识别。

**[Few-Shot Recognition via Stage-Wise Retrieval-Augmented Finetuning](few-shot_recognition_via_stage-wise_retrieval-augmented_finetuning.md)**

:   本文首次将检索增强学习（RAL）扩展到少样本识别（FSR），揭示了检索数据的分布不平衡和域差距两大挑战，提出两阶段方法 SWAT（先在混合数据上微调视觉编码器、再在少量标注数据上重训分类器），在 9 个基准上以 >6% 的优势超越所有先前方法。

**[GENIUS: A Generative Framework for Universal Multimodal Search](genius_a_generative_framework_for_universal_multimodal_search.md)**

:   首个通用生成式多模态检索框架，通过模态解耦的语义量化将多模态数据编码为离散 ID，用自回归解码器直接从查询生成目标 ID，在 Flickr30K 文本→图像检索上超越先前生成式方法 25+ 个点，存储开销比 CLIP 降低 99%。

**[GOAL: Global-Local Object Alignment Learning](goal_global-local_object_alignment_learning.md)**

:   提出GOAL方法，通过局部图-句匹配（LISM）和Token相似性学习（TSL）两个模块增强CLIP对长文本描述的理解能力，在全局对齐的基础上引入局部语义对齐，大幅提升图文检索性能。

**[Joint Vision-Language Social Bias Removal for CLIP](joint_vision-language_social_bias_removal_for_clip.md)**

:   本文揭示了现有CLIP去偏方法因图文偏差分布不一致导致的"过度去偏"问题，提出先对齐图文偏差再联合移除的双模态去偏框架，在多个骨干网络上显著提升ABLE综合指标，实现了偏差消除与V-L对齐能力的良好平衡。

**[LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant](lamra_large_multimodal_model_as_your_advanced_retrieval_assistant.md)**

:   将生成式大语言模型（LMM）改造为通用多模态检索器+重排器，通过两阶段训练（语言预训练+多模态指令微调）和联合逐点/列表重排训练，仅插入轻量LoRA模块即可在16种检索任务上显著超越双编码器方法，且在10个未见数据集上展现强泛化能力。

**[LotusFilter: Fast Diverse Nearest Neighbor Search via a Learned Cutoff Table](lotusfilter_fast_diverse_nearest_neighbor_search_via_a_learned_cutoff_table.md)**

:   提出LotusFilter，通过离线预计算每个向量的邻近关系构建截断表(cutoff table)，在线阶段用贪心集合删除实现多样化过滤，将传统 $O(DS^2)$ 的多样化搜索降至 $O(T+S+KL)$，过滤仅需0.02ms/query，内存仅为传统方法的1/40。

**[NeighborRetr: Balancing Hub Centrality in Cross-Modal Retrieval](neighborretr_balancing_hub_centrality_in_cross-modal_retrieval.md)**

:   提出 NeighborRetr，通过三重机制解决跨模态检索中的 Hubness 问题（少数样本垄断近邻）：中心性加权损失（降低 hub 样本的训练权重）、邻域调整损失（区分好/坏 hub）和均匀正则化（确保每个样本被公平检索），在 MSR-VTT 文本→视频 R@1 达 49.5%（+0.9% SOTA）。

**[Preserving Clusters in Prompt Learning for Unsupervised Domain Adaptation](preserving_clusters_in_prompt_learning_for_unsupervised_domain_adaptation.md)**

:   提出 CRPL 框架，通过源域增强的伪标签和基于最优传输的聚类保持策略，改进 CLIP 在无监督域适应（UDA）中的 prompt 学习，使得目标域 prompt 的文本嵌入能更好地覆盖视觉嵌入的聚类结构。

**[RANGE: Retrieval Augmented Neural Fields for Multi-Resolution Geo-Embeddings](range_retrieval_augmented_neural_fields_for_multi-resolution_geo-embeddings.md)**

:   提出RANGE，通过检索增强策略将高分辨率视觉信息近似注入地理位置嵌入，解决了对比学习（如SatCLIP）丢弃模态特有信息的问题，在分类任务上提升高达13.1%，回归任务上提升0.145 $R^2$。

**[Towards Smart Point-and-Shoot Photography](towards_smart_point-and-shoot_photography.md)**

:   提出智能"傻瓜相机"摄影系统：先用 CLIP 文本嵌入的构图质量评估器（CCQA）判断当前构图质量，再用专家混合（MoE）相机姿态调整模型（CPAM）预测偏航/俯仰调整角度，在 PCARD 数据集（320K 图像，从 4K 全景图生成）上实现 79.3% AUC 的调整建议和 0.613 IoU 的调整精度。

**[VDocRAG: Retrieval-Augmented Generation over Visually-Rich Documents](vdocrag_retrieval-augmented_generation_over_visually-rich_documents.md)**

:   构建首个直接以文档图像（而非解析文本）为输入的 RAG 框架，用 LVLM 作为双编码器检索器 + 两种自监督预训练任务（对比+生成）实现文档图像检索，在 ChartQA 上比文本 RAG 高 24 个点。

**[VladVA: Discriminative Fine-tuning of LVLMs](vladva_discriminative_fine-tuning_of_lvlms.md)**

:   提出VladVA框架，通过混合短/长caption数据策略、对比损失+自回归损失的联合训练、以及soft prompting+LoRA的参数高效适配，将生成式LVLM（LLaVA）转化为强判别式模型，在图文检索和组合性理解基准上大幅超越CLIP类模型和18B EVA-CLIP。
