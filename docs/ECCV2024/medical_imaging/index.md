---
title: >-
  ECCV2024 医学图像方向 23篇论文解读
description: >-
  23篇ECCV2024 医学图像方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🏥 医学图像

**🎞️ ECCV2024** · 共 **23** 篇

**[Adaptive Correspondence Scoring For Unsupervised Medical Ima](adaptive_correspondence_scoring_for_unsupervised_medical_ima.md)**

:   针对医学图像无监督配准中噪声、遮挡等干扰因素导致的虚假重建误差问题，提出了一个自适应对应关系评分框架（AdaCS），通过学习像素级的对应置信度图来重新加权误差残差，以即插即用方式一致提升三种主流配准架构在三个数据集上的性能。

**[Alternate Diverse Teaching For Semi-Supervised Medical Image Segmentation](alternate_diverse_teaching_for_semi-supervised_medical_image_segmentation.md)**

:   提出 AD-MT（Alternate Diverse Mean Teacher），通过随机周期性交替更新两个教师模型 + 基于熵的冲突调和策略，在半监督医学分割中解决 confirmation bias 问题，在 ACDC/LA/Pancreas 上全面超越 SOTA。

**[Architecture-Agnostic Untrained Network Priors For Image Reconstruction With Fre](architecture-agnostic_untrained_network_priors_for_image_reconstruction_with_fre.md)**

:   提出三种与架构无关的频率正则化技术（带宽受限输入、带宽可控上采样、Lipschitz 正则化卷积层），统一解决 untrained network prior 的架构敏感性、过拟合和运行效率问题，在 MRI 重建任务中显著缩小不同架构间的性能差距。

**[Cardiacnet Learning To Reconstruct Abnormalities For Cardiac Disease Assessment ](cardiacnet_learning_to_reconstruct_abnormalities_for_cardiac_disease_assessment_.md)**

:   提出基于重建的心脏疾病评估框架 CardiacNet，通过 Consistency Deformation Codebook (CDC) 和 Consistency Deformation Discriminator (CDD) 学习正常与异常心脏超声视频之间的结构和运动差异，在射血分数预测、肺动脉高压和房间隔缺损分类三个任务上达到 SOTA。

**[Chameleon A Data-Efficient Generalist For Dense Visual Prediction In The Wild](chameleon_a_data-efficient_generalist_for_dense_visual_prediction_in_the_wild.md)**

:   提出 Chameleon，一个基于 meta-learning 和 token matching 的数据高效视觉通才模型，仅需几十张标注图像即可适应全新的密集预测任务（包括医学图像、视频、3D 等），在六个下游基准上显著超越现有通才方法。

**[Chex Interactive Localization And Region Description In Chest X-Rays](chex_interactive_localization_and_region_description_in_chest_x-rays.md)**

:   提出ChEX——一个同时支持文本提示和边界框查询的交互式胸部X光解释模型，通过DETR风格的prompt检测器和多任务联合训练，在9个胸部X光任务上与SOTA竞争，同时提供独特的定位可解释性和用户交互能力。

**[Co-Synthesis Of Histopathology Nuclei Image-Label Pairs Using A Context-Conditio](co-synthesis_of_histopathology_nuclei_image-label_pairs_using_a_context-conditio.md)**

:   提出一种上下文条件化的联合扩散模型，能够同时合成组织病理学细胞核图像、语义标签和距离图，通过点图（centroid layout）和文本提示两种条件实现对合成过程的精确控制，并生成高质量的实例级标签用于下游核分割和分类任务。

**[Gtp-4O Modality-Prompted Heterogeneous Graph Learning For Omni-Modal Biomedical ](gtp-4o_modality-prompted_heterogeneous_graph_learning_for_omni-modal_biomedical_.md)**

:   提出基于异构图的全模态生物医学表征学习框架 GTP-4o，通过异构图嵌入显式建模跨模态关系，利用图提示机制补全缺失模态，并设计知识引导的层次化跨模态聚合，在胶质瘤分级和生存预测任务上取得SOTA。

**[Gtp4O Modalityprompted Heterogeneous Graph Learning For](gtp4o_modalityprompted_heterogeneous_graph_learning_for.md)**

:   提出 GTP-4o，一种基于模态提示的异构图学习框架，通过异构图嵌入、图提示补全缺失模态、知识引导的层级聚合，实现基因组学-病理图像-细胞图-文本等多种临床模态的统一表示学习。

**[I-Medsam Implicit Medical Image Segmentation With Segment Anything](i-medsam_implicit_medical_image_segmentation_with_segment_anything.md)**

:   提出 I-MedSAM，将 SAM 的强泛化能力与隐式神经表示（INR）的连续空间预测优势结合，通过频率适配器增强边界高频信息、不确定性引导采样精细化分割，仅用 1.6M 可训练参数即超越现有离散和隐式方法。

**[Improving Medical Multi-Modal Contrastive Learning With Expert Annotations](improving_medical_multi-modal_contrastive_learning_with_expert_annotations.md)**

:   提出 eCLIP，通过整合放射科专家的眼动注视热力图作为额外监督信号，结合 mixup 增强和课程学习策略，在不修改 CLIP 核心架构的前提下增强医学多模态对比学习的表征质量。

**[Improving Medical Multimodal Contrastive Learning With Exper](improving_medical_multimodal_contrastive_learning_with_exper.md)**

:   提出eCLIP，通过引入放射科医生的眼动热力图（eye-gaze heatmap）作为专家标注，利用热力图处理器和mixup增强策略扩充高质量正样本对，有效缓解医学CLIP中的"模态间隙"问题，在零样本推理、线性探测、跨模态检索和RAG报告生成等任务上取得一致性提升。

**[Is User Feedback Always Informative Retrieval Latent Defending For Semi-Supervis](is_user_feedback_always_informative_retrieval_latent_defending_for_semi-supervis.md)**

:   发现用户反馈在域适应中并非总是有益——偏向纠正错误预测的"负偏反馈"(NBF)会导致现有半监督域适应方法性能下降，并提出 Retrieval Latent Defending 方法，通过在每个 mini-batch 中加入伪标签防御样本来平衡监督信号。

**[Ophnet A Large-Scale Video Benchmark For Ophthalmic Surgical Workflow Understand](ophnet_a_large-scale_video_benchmark_for_ophthalmic_surgical_workflow_understand.md)**

:   构建了OphNet——目前最大规模的眼科手术视频基准数据集（2278个视频、285小时、66种手术类型、102种手术阶段、150种精细操作），支持手术类型识别、阶段识别、时序定位和阶段预测四大任务，其规模约为现有最大手术工作流分析基准的20倍。

**[Pathology-Knowledge Enhanced Multi-Instance Prompt Learning For Few-Shot Whole S](pathology-knowledge_enhanced_multi-instance_prompt_learning_for_few-shot_whole_s.md)**

:   提出 PEMP 框架，将病理学先验知识（视觉样例 + 文本描述）融入 patch 级和 slide 级的 prompt 中，结合 CLIP 进行多实例 prompt learning，在少样本弱监督 WSI 分类任务上平均超越 SOTA 方法 4%。

**[Pathologyknowledge Enhanced Multiinstance Prompt Learni](pathologyknowledge_enhanced_multiinstance_prompt_learni.md)**

:   提出 PEMP——病理知识增强的多实例提示学习框架，将视觉和文本病理先验（典型 patch/slide 示例 + 语言描述）注入 CLIP 的提示中，在 patch 和 slide 两个层级进行对比学习，显著提升少样本全切片图像（WSI）分类性能。

**[Radedit Stress-Testing Biomedical Vision Models Via Diffusion Image Editing](radedit_stress-testing_biomedical_vision_models_via_diffusion_image_editing.md)**

:   提出 RadEdit，一种基于扩散模型的医学图像编辑方法，通过引入 edit mask 和 keep mask 的双重掩码机制，打破数据中的虚假关联（spurious correlations），生成高质量的合成测试集来压力测试（stress-test）生物医学视觉模型对数据集偏移的鲁棒性。

**[Radiative Gaussian Splatting For Efficient X-Ray Novel View Synthesis](radiative_gaussian_splatting_for_efficient_x-ray_novel_view_synthesis.md)**

:   提出 X-Gaussian，首个将 3D 高斯泼溅（3DGS）应用于 X 射线新视角合成的框架，通过设计辐射高斯点云模型（替代球谐函数）和角度位姿长方体均匀初始化策略（替代 SfM），在性能上超越 SOTA NeRF 方法 6.5 dB 的同时，实现 73× 推理加速和仅 15% 的训练时间。

**[Textttnephi Neural Deformation Fields For Approximately Diff](textttnephi_neural_deformation_fields_for_approximately_diff.md)**

:   NePhi用隐式神经网络（SIREN）替代传统的体素化形变场来表示配准变换，通过编码器预测latent code + 可选的测试时优化实现快速且近似微分同胚的医学图像配准，在多分辨率设置下与SOTA精度相当但内存降低5倍。

**[Tip Tabular-Image Pre-Training For Multimodal Classification With Incomplete Dat](tip_tabular-image_pre-training_for_multimodal_classification_with_incomplete_dat.md)**

:   提出 TIP 框架，通过掩码表格重建、图像-表格匹配和对比学习三种自监督任务，对表格数据和图像进行联合预训练，学习对不完整表格数据鲁棒的多模态表征，用于下游分类任务。

**[Tip Tabularimage Pretraining For Multimodal Classification W](tip_tabularimage_pretraining_for_multimodal_classification_w.md)**

:   提出TIP框架，通过掩码表格重建、图像-表格匹配和对比学习三个自监督任务，在表格数据不完整的条件下学习鲁棒的多模态表示，在自然图像和医学图像分类任务上超越现有方法。

**[Topology-Preserving Downsampling Of Binary Images](topology-preserving_downsampling_of_binary_images.md)**

:   提出首个基于离散优化（整数规划）的拓扑保持二值图像下采样方法，通过将下采样像素的黑白决策编码为布尔变量、拓扑保持作为硬约束、与原图相似度作为目标函数来求解，保证下采样结果具有与原图完全相同的Betti数（连通分量数和孔洞数），同时保持与传统方法竞争性的像素级相似度。

**[Unleashing The Power Of Prompt-Driven Nucleus Instance Segmentation](unleashing_the_power_of_prompt-driven_nucleus_instance_segmentation.md)**

:   提出 PromptNucSeg 框架，通过训练一个 prompter 自动生成细胞核中心点 prompt，并微调 SAM 进行逐核分割，同时引入相邻核作为 negative prompt 解决重叠核分割问题，无需复杂后处理即在三个 benchmark 上达到 SOTA。
