---
title: >-
  CVPR2025 LLM/NLP方向 31篇论文解读
description: >-
  31篇CVPR2025 LLM/NLP方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💬 LLM/NLP

**📷 CVPR2025** · **31** 篇论文解读

**[Attribute-formed Class-specific Concept Space: Endowing Language Bottleneck Model with Better Generalization](attribute-formed_class-specific_concept_space_endowing_language_bottleneck_model.md)**

:   针对语言瓶颈模型（LBM）中所有概念混合在一起导致的虚假线索推理和零样本泛化差的问题，提出属性构成的类特定概念空间，将概念按属性维度为每个类别组织独立空间。

**[Breaking the Low-Rank Dilemma of Linear Attention](breaking_the_low-rank_dilemma_of_linear_attention.md)**

:   从理论上揭示线性注意力性能不及 Softmax 注意力的根本原因是输出特征的低秩问题，提出秩增强线性注意力（RALA），通过增强 KV 缓存秩和输出特征秩两种互补策略，在保持线性复杂度的同时追平甚至超越 Softmax 注意力的表现。

**[Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior](bridging_the_vision-brain_gap_with_an_uncertainty-aware_blur_prior.md)**

:   提出不确定性感知的模糊先验，为从脑信号（fMRI）重建视觉刺激提供物理合理的图像退化模型，缓解大脑编码过程中高频信息丢失对重建质量的影响。

**[Building Vision Models upon Heat Conduction](building_vision_models_upon_heat_conduction.md)**

:   提出 vHeat 视觉 backbone，将图像 patch 建模为热源，利用物理热传导方程通过 DCT/IDCT 变换实现 $O(N^{1.5})$ 复杂度的信息传播，在 ImageNet-1K 上以 3 倍吞吐量和 80% 更少 GPU 显存达到 84.0% top-1 准确率。

**[Chat-based Person Retrieval via Dialogue-Refined Cross-Modal Alignment](chat-based_person_retrieval_via_dialogue-refined_cross-modal_alignment.md)**

:   本文提出基于对话的行人检索（ChatPR）新范式，构建了首个对话-图像配对数据集ChatPedes，并设计了DiaNA框架通过自适应属性精炼器实现对话与图像间的细粒度跨模态对齐，显著优于传统单句文本检索方法。

**[Classifier-to-Bias: Toward Unsupervised Automatic Bias Detection for Visual Classifiers](classifier-to-bias_toward_unsupervised_automatic_bias_detection_for_visual_class.md)**

:   本文提出Classifier-to-Bias（C2B），首个无需任何标注数据的视觉分类器偏差自动发现框架，仅依靠分类任务的文本描述，利用LLM生成偏差候选并通过检索增强的生成-验证流程来识别预训练模型中的系统性偏差。

**[ComRoPE: Scalable and Robust Rotary Position Embedding Parameterized by Trainable Commuting Angle Matrices](comrope_scalable_and_robust_rotary_position_embedding_parameterized_by_trainable.md)**

:   ComRoPE将RoPE从固定的2D旋转矩阵推广到SO(n)群的更大子群，证明交换性是保持相对位置鲁棒性的充要条件，提出AP和LD两种可训练参数化方案，在ImageNet分类（+1.6%）、COCO检测（+0.2 AP）上均优于LieRE。

**[Dora: Sampling and Benchmarking for 3D Shape Variational Auto-Encoders](dora_sampling_and_benchmarking_for_3d_shape_variational_auto-encoders.md)**

:   提出 Dora-VAE，通过 Sharp Edge Sampling (SES) 关注几何锐边区域、Dual Cross-Attention 分别处理均匀和显著采样点，以仅 1,280 个 latent codes（8× 小于 XCube-VAE 的 10,000+）实现更优的 3D 形状重建质量，同时建立了新的 Dora-Bench 评测基准。

**[Empowering LLMs to Understand and Generate Complex Vector Graphics](empowering_llms_to_understand_and_generate_complex_vector_graphics.md)**

:   本文提出LLM4SVG，首个支持任意LLM进行SVG理解与生成的统一框架，通过可学习语义token精确编码SVG组件属性，配合模块化架构和580K条SVG指令微调数据（SVGX-SFT），显著超越GPT-4等基线在复杂矢量图形生成上的表现。

**[Exposure-slot: Exposure-centric Representations Learning with Slot-in-Slot Attention](exposure-slot_exposure-centric_representations_learning_with_slot-in-slot_attent.md)**

:   本文提出Exposure-slot框架，将Slot Attention算法扩展为层次化的slot-in-slot结构，通过可学习的曝光prompt引导特征聚类，实现以曝光为中心的区域感知表征学习，在欠曝/过曝图像矫正任务上取得SOTA性能。

**[Guiding Human-Object Interactions with Rich Geometry and Relations](guiding_human-object_interactions_with_rich_geometry_and_relations.md)**

:   本文提出ROG框架，通过在物体网格上采样富含几何信息的关键点构建交互距离场（IDF），并利用基于扩散的关系模型在推理时引导运动生成模型产生关系感知且语义对齐的人物-物体交互动作，在FullBodyManipulation数据集上显著超越SOTA。

**[Imagine and Seek: Improving Composed Image Retrieval with an Imagined Proxy](imagine_and_seek_improving_composed_image_retrieval_with_an_imagined_proxy.md)**

:   提出IP-CIR方法，通过大语言模型生成"想象中的目标图像描述"作为代理，将组合图像检索(CIR)转化为标准图像检索问题，在CIRR和FashionIQ等基准上达到零样本SOTA。

**[Improving Autoregressive Visual Generation with Cluster-Oriented Token Prediction](improving_autoregressive_visual_generation_with_cluster-oriented_token_predictio.md)**

:   本文深入分析LLM框架下视觉embedding空间的特性，发现视觉token间的相关性有助于实现更稳定的生成，据此提出IAR方法，通过码本重排（Codebook Rearrangement）和簇导向token预测（Cluster-Oriented Token Prediction）提升自回归视觉生成的效率和质量。

**[L-SWAG: Layer-Sample Wise Activation with Gradients Information for Zero-Shot NAS on Vision Transformers](l-swag_layer-sample_wise_activation_with_gradients_information_for_zero-shot_nas.md)**

:   本文提出L-SWAG（Layer-Sample Wise Activation with Gradients），一种新型通用零代价代理，通过结合层级和样本级的激活值与梯度信息来评估网络架构质量，首次将零代价NAS系统性地扩展到Vision Transformer搜索空间，并在Autoformer搜索空间的6个任务上建立了新的benchmark。

**[Learning Textual Prompts for Open-World Semi-Supervised Learning](learning_textual_prompts_for_open-world_semi-supervised_learning.md)**

:   本文提出了一种针对开放世界半监督学习（OWSSL）的新方法，通过全局-局部文本提示学习策略增强图文对齐效果，并设计前向-反向策略降低无标签样本中图文匹配的噪声，在多个细粒度数据集上显著超越SOTA。

**[Let Samples Speak: Mitigating Spurious Correlation by Exploiting the Clusterness](let_samples_speak_mitigating_spurious_correlation_by_exploiting_the_clusterness_.md)**

:   提出NSF方法，通过利用样本在特征空间中的聚类特性自动识别依赖虚假特征的样本组，无需组标注即可训练出对虚假相关性鲁棒的分类器，最差组准确度显著超越ERM基线。

**[Lost in Translation, Found in Context: Sign Language Translation with Contextual Cues](lost_in_translation_found_in_context_sign_language_translation_with_contextual_c.md)**

:   通过引入背景视频描述、历史翻译和伪词汇表三种上下文线索，结合Llama3-8B的LoRA微调，实现了连续手语到文本的精确翻译，在BOBSL数据集上相比SOTA提升40%以上。

**[Making Old Film Great Again: Degradation-aware State Space Model for Old Film Restoration](making_old_film_great_again_degradation-aware_state_space_model_for_old_film_res.md)**

:   本文提出MambaOFR框架，针对老电影特有的复合退化问题，设计退化感知prompt引导Mamba模型动态调整修复模式，配合光流引导的掩码变形对齐模块防止结构缺陷传播，并引入首个包含合成与真实数据的老电影修复benchmark数据集。

**[PosterO: Structuring Layout Trees to Enable Language Models in Generalized Content-Aware Layout Generation](postero_structuring_layout_trees_to_enable_language_models_in_generalized_conten.md)**

:   本文提出PosterO，一种以布局为中心的海报生成方法，将数据集中的布局结构化为SVG语言的层次化树表示，通过通用形状表示、设计意图向量化和层次节点描述三大机制，使LLM能够通过in-context learning在推理时生成多样化的内容感知布局。

**[Rethinking Spiking Self-Attention Mechanism: Implementing a-XNOR Similarity Calculation in Spiking Transformers](rethinking_spiking_self-attention_mechanism_implementing_a-xnor_similarity_calcu.md)**

:   本文深入分析了点积在脉冲查询-键对中因大量"非脉冲事件"导致相似度度量失效的根本原因，提出专为脉冲序列设计的a-XNOR相似度度量，将非脉冲对的相关性重定义为特定值a，在多种脉冲Transformer架构和数据集上显著提升性能。

**[RoadSocial: A Diverse VideoQA Dataset and Benchmark for Road Event Understanding from Social Video Narratives](roadsocial_a_diverse_videoqa_dataset_and_benchmark_for_road_event_understanding_.md)**

:   本文提出RoadSocial，一个来源于社交媒体的大规模多样化VideoQA数据集（13.2K视频、260K问答对），覆盖全球多地域多视角的道路事件场景，通过半自动标注框架和12类QA任务系统性评测了18种Video LLM的道路事件理解能力。

**[Robust Message Embedding via Attention Flow-Based Steganography](robust_message_embedding_via_attention_flow-based_steganography.md)**

:   本文提出RMSteg（Robust Message Steganography）框架，首次将Transformer注意力机制集成到归一化流网络中（AttnFlow），配合可逆QR码转换和可逆Token融合模块，实现了高质量、高容量且鲁棒的消息-图像隐写，隐写图像即使经过打印-拍照等极端扭曲仍可准确解码。

**[SATA: Spatial Autocorrelation Token Analysis for Enhancing the Robustness of Vision Transformers](sata_spatial_autocorrelation_token_analysis_for_enhancing_the_robustness_of_visi.md)**

:   本文提出SATA（Spatial Autocorrelation Token Analysis），一种免训练的ViT鲁棒性增强方法，通过空间自相关分析将token按空间关联模式分组，利用分组信息重新加权token表示，提升ViT在分布偏移和对抗攻击下的鲁棒性，且不影响干净样本性能。

**[ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Model](scamo_exploring_the_scaling_law_in_autoregressive_motion_generation_model.md)**

:   首次在人类动作生成领域系统验证缩放律，提出包含Motion FSQ-VAE（解决codebook collapse）、260小时MotionUnion数据集和文本前缀自回归Transformer的可扩展系统ScaMo，发现归一化测试损失与FLOPs的对数律以及词汇参数/模型参数/数据量与FLOPs的幂律关系，并在$1\times 10^{18}$FLOPs预算下成功预测最优配置。

**[SEC-Prompt: SEmantic Complementary Prompting for Few-Shot Class-Incremental Learning](sec-promptsemantic_complementary_prompting_for_few-shot_class-incremental_learni.md)**

:   提出 SEC-Prompt（SEmantic Complementary Prompt）框架，学习两组语义互补的提示——判别性提示（D-Prompt）和非判别性提示（ND-Prompt），通过自适应查询机制协同工作，分别强化类间区分和促进新类泛化，在三个基准数据集上取得 SOTA 性能。

**[SoftShadow: Leveraging Soft Masks for Penumbra-Aware Shadow Removal](softshadow_leveraging_soft_masks_for_penumbra-aware_shadow_removal.md)**

:   提出SoftShadow框架，用连续灰度软掩码替代传统二值硬掩码来表示阴影区域，通过SAM+LoRA预测软掩码并引入半影形成约束损失联合训练检测与去阴影网络，在SRD/ISTD+/LRSS/UIUC四个数据集上达到SOTA且无需外部掩码输入。

**[Spiking Transformer with Spatial-Temporal Attention](spiking_transformer_with_spatial-temporal_attention.md)**

:   将空间-时间注意力机制融入脉冲Transformer架构，通过时空解耦的注意力设计和脉冲驱动的自注意机制，在保持SNN能效优势的同时缩小与ANN的性能差距，在多个视觉基准上达到SNN SOTA。

**[STAA-SNN: Spatial-Temporal Attention Aggregator for Spiking Neural Networks](staa-snn_spatial-temporal_attention_aggregator_for_spiking_neural_networks.md)**

:   通过在SNN中集成全局上下文自注意(GC)、位置编码(PE)、步骤注意(SA)和时间步随机退出(TSRD)四大模块，STAA-SNN在CIFAR-10/100和ImageNet上达到97.14%/82.05%/70.40%的SNN SOTA性能。

**[Test-Time Visual In-Context Tuning](test-time_visual_in-context_tuning.md)**

:   首次系统研究VICL模型在分布偏移下的鲁棒性问题，提出VICT方法利用循环一致性自监督信号在测试时进行单样本微调，在6个视觉任务和15种图像破坏下显著改进Painter等VICL模型。

**[The Change You Want To Detect: Semantic Change Detection In Earth Observation With Hybrid Data Generation](the_change_you_want_to_detect_semantic_change_detection_in_earth_observation_wit.md)**

:   本文提出HySCDG（Hybrid Semantic Change Detection Data Generation），一种混合数据生成流水线，结合真实超高分辨率（VHR）遥感影像和图像inpainting技术生成大规模语义变化检测训练数据，在简洁的架构设计下实现了强大的时间和空间泛化能力。

**[TIDE: Training Locally Interpretable Domain Generalization Models Enables Test-time Correction](tide_training_locally_interpretable_domain_generalization_models_enables_test-ti.md)**

:   本文提出TIDE，一种针对单源域泛化的新型训练方案，利用扩散模型和LLM自动生成类别级概念标注（如"鸟类=尖嘴+翅膀+爪子"），通过概念显著性对齐损失训练模型关注域不变的局部概念而非全局背景特征，使模型在测试时能通过概念显著图自动矫正域偏移导致的错误预测。
