---
title: >-
  ICML2025 语义分割方向 20篇论文解读
description: >-
  20篇ICML2025 语义分割方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✂️ 语义分割

**🧪 ICML2025** · **20** 篇论文解读

**[ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation](actionpiece_contextually_tokenizing_action_sequences_for_generative_recommendati.md)**

:   提出 ActionPiece，首个上下文感知的动作序列分词器，将用户行为序列建模为"特征集合的序列"，通过类 BPE 的合并策略在集合内部和相邻集合之间发现高频特征模式，使同一动作在不同上下文中被分词为不同 token，显著提升生成式推荐性能。

**[Adapter Naturally Serves as Decoupler for Cross-Domain Few-Shot Semantic Segmentation](adapter_naturally_serves_as_decoupler_for_cross-domain_few-shot_semantic_segment.md)**

:   本文发现 adapter 天然具有领域信息解耦能力（基于结构而非损失），据此提出 Domain Feature Navigator (DFN) 作为结构化领域解耦器，配合 SAM-SVN 防止源域过拟合，在跨域少样本语义分割 (CD-FSS) 上以 1-shot 平均 63.99% / 5-shot 平均 69.77% MIoU 显著超越 SOTA。

**[Alberta Wells Dataset: Pinpointing Oil and Gas Wells from Satellite Imagery](alberta_wells_dataset_pinpointing_oil_and_gas_wells_from_satellite_imagery.md)**

:   提出首个大规模油气井检测基准数据集 Alberta Wells Dataset（213k+ 井位、188k+ 卫星图像 patch），将废弃/暂停/活跃油气井的定位问题建模为二值分割和目标检测任务，并评估了多种 CNN 和 Transformer 基线模型。

**[Aligning Spoken Dialogue Models from User Interactions](aligning_spoken_dialogue_models_from_user_interactions.md)**

:   首次为全双工语音对话模型（Moshi）设计完整的偏好对齐框架，从15万+条真实用户语音对话中自动构建内容+时序两类偏好对，通过仅在文本token上做DPO-LN对齐，QA平均提升3.1%、安全性提升6.9%，并通过人类评估确认多轮对话质量的改善。

**[Balanced Learning for Domain Adaptive Semantic Segmentation](balanced_learning_for_domain_adaptive_semantic_segmentation.md)**

:   提出 BLDA——通过分析网络预测的 logit 分布来直接量化类别偏差程度，用共享锚点分布对齐各类 logit 分布实现后处理校准，同时在自训练中用 GMM 在线估计并修正 logit 生成无偏伪标签，在 GTA→CS 和 SYN→CS 两个基准上为多种基线方法带来一致提升。

**[ConText: Driving In-context Learning for Text Removal and Segmentation](context_driving_in-context_learning_for_text_removal_and_segmentation.md)**

:   首次将视觉上下文学习（V-ICL）范式应用于OCR任务，提出任务链式提示（task-chaining prompting）、上下文感知聚合（CAA）和自提示策略（self-prompting）三项关键设计，在文本去除和分割任务上大幅超越现有V-ICL通用模型和专用模型，分别取得 +4.50 PSNR 和 +3.34% fgIoU 的提升。

**[Dual form Complementary Masking for Domain-Adaptive Image Segmentation](dual_form_complementary_masking_for_domain-adaptive_image_segmentation.md)**

:   提出 MaskTwins 框架，将掩码重建理论化为稀疏信号重建问题，证明互补掩码对（dual form complementary masks）在提取域无关特征方面具有理论优势，并在端到端训练中通过互补掩码一致性约束实现域自适应分割。

**[Efficient and Robust Semantic Image Communication via Stable Cascade](efficient_and_robust_semantic_image_communication_via_stable_cascade.md)**

:   基于 Stable Cascade 架构构建语义图像通信框架，利用 EfficientNet-V2 提取极紧凑图像嵌入（仅占原始大小 0.29%）作为 LDM 条件，通过噪声鲁棒微调使系统在低 SNR 信道下仍能忠实重建图像，同时实现 3-16 倍推理加速。

**[FeatSharp: Your Vision Model Features, Sharper](featsharp_your_vision_model_features_sharper.md)**

:   提出 FeatSharp，通过将 FeatUp 的联合双边上采样（JBU）与图像瓦片（tiling）特征进行注意力融合，以极低成本将低分辨率视觉编码器的特征图连贯地上采样到高分辨率，同时捕获原始分辨率下丢失的细粒度细节。

**[InfoSAM: Fine-Tuning the Segment Anything Model from An Information-Theoretic Perspective](infosam_fine-tuning_the_segment_anything_model_from_an_information-theoretic_per.md)**

:   提出 InfoSAM，从信息论角度为 SAM 的参数高效微调（PEFT）设计了基于 Rényi 互信息的关系压缩与蒸馏框架，通过压缩伪不变信息、保留域不变关系来提升微调效果。

**[IT³: Idempotent Test-Time Training](it3_idempotent_test-time_training.md)**

:   提出 IT³，一种基于幂等性（idempotence）的测试时训练方法，通过最小化网络递归应用的输出差异来实现无需辅助任务、无需额外数据的通用测试时适应。

**[MorphTok: Morphologically Grounded Tokenization for Indian Languages](morphtok_morphologically_grounded_tokenization_for_indian_languages.md)**

:   提出 MorphTok 框架，通过形态学感知的预分词步骤（查找表/语言模型）和约束 BPE（CBPE）算法处理印度语言中的依存元音问题，在机器翻译和语言建模任务上提升下游性能，并引入人类评估指标 EvalTok。

**[QMamba: On First Exploration of Vision Mamba for Image Quality Assessment](qmamba_on_first_exploration_of_vision_mamba_for_image_quality_assessment.md)**

:   首次将 Vision Mamba（状态空间模型）引入图像质量评估（IQA），提出 QMamba 框架和 StylePrompt 轻量微调策略，在合成/真实/AIGC 多种 IQA 任务上以更低计算成本超越 CNN 和 Transformer 基线。

**[Self-Disentanglement and Re-Composition for Cross-Domain Few-Shot Segmentation](self-disentanglement_and_re-composition_for_cross-domain_few-shot_segmentation.md)**

:   本文发现跨域少样本分割（CD-FSS）中基于距离比较的方法存在特征纠缠问题，其根源在于ViT各层输出在距离计算时的等权交叉匹配，进而提出通过自解耦（Self-Disentanglement）和重组合（Re-Composition）的方式，学习ViT组件间的比较权重来解决该问题。

**[Separating Knowledge and Perception with Procedural Data](separating_knowledge_and_perception_with_procedural_data.md)**

:   仅用程序化生成数据（非真实图像）训练视觉表征模型，再通过 visual memory（KNN 检索数据库）注入真实世界知识，在分类和分割任务上逼近真实数据训练的性能，同时实现对所有真实数据的完全可控（隐私保护、高效遗忘）。

**[Sounding that Object: Interactive Object-Aware Image to Audio Generation](sounding_that_object_interactive_object-aware_image_to_audio_generation.md)**

:   提出一种交互式对象感知音频生成模型，通过多模态点积注意力在训练时学习图像区域与声音的关联，在测试时用 SAM 分割掩码替代注意力权重，允许用户通过点击选择图像中的视觉对象来生成对应的声音。

**[SpikeVideoFormer: An Efficient Spike-Driven Video Transformer with Hamming Attention and $\mathcal{O}(T)$ Complexity](spikevideoformer_an_efficient_spike-driven_video_transformer_with_hamming_attent.md)**

:   提出 SpikeVideoFormer，首个面向视频任务的脉冲驱动 Transformer，通过 Hamming 注意力替代点积注意力实现 spike 特征相似性的准确度量，结合联合时空注意力保持 $\mathcal{O}(T)$ 线性时间复杂度，在三个视频任务上达到 SNN SOTA，同时效率比 ANN 高 5-16 倍。

**[SToFM: a Multi-scale Foundation Model for Spatial Transcriptomics](stofm_a_multi-scale_foundation_model_for_spatial_transcriptomics.md)**

:   提出 SToFM，首个多尺度空间转录组学基础模型，通过基因尺度域适应、微观尺度子切片划分和宏观尺度虚拟细胞注入，结合 SE(2) Transformer 和 88M 细胞的大规模预训练语料库，在组织区域语义分割和细胞类型标注等任务上显著超越现有方法。

**[unMORE: Unsupervised Multi-Object Segmentation via Center-Boundary Reasoning](unmore_unsupervised_multi-object_segmentation_via_center-boundary_reasoning.md)**

:   提出 unMORE，通过学习三层物体中心表征（存在性/中心场/边界距离场）并设计无网络的多目标推理模块，实现无监督多目标分割，在 COCO 等 6 个数据集上大幅超越所有无监督方法。

**[Using Multiple Input Modalities Can Improve Data-Efficiency and O.O.D. Generalization for ML with Satellite Imagery](using_multiple_input_modalities_can_improve_data-efficiency_and_ood_generalizati.md)**

:   系统研究在卫星遥感 ML 任务中融合光学影像与额外地理数据层（DEM、土地覆盖图、温度、风速等）的效果，发现多模态输入显著提升模型性能，且收益在标注数据有限和地理分布外场景中最大；意外地，硬编码融合策略优于学习型融合策略。
