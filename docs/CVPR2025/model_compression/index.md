---
title: >-
  CVPR2025 模型压缩方向 11篇论文解读
description: >-
  11篇CVPR2025 模型压缩方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📦 模型压缩

**📷 CVPR2025** · 共 **11** 篇

**[Adapter Merging With Centroid Prototype Mapping For Scalable Class-Incremental L](adapter_merging_with_centroid_prototype_mapping_for_scalable_class-incremental_l.md)**

:   提出ACMap框架，通过将每个任务独立训练的adapter增量平均合并为单一adapter（保持O(1)推理复杂度），结合centroid prototype mapping对齐旧任务原型在新子空间中的表示，在5个基准上实现与SOTA EASE相当的精度同时推理速度快39倍。

**[Alternating Gradient Flow Utility A Unified Metric For Structural Pruning And Dy](alternating_gradient_flow_utility_a_unified_metric_for_structural_pruning_and_dy.md)**

:   提出基于交替梯度流(AGF)的统一效用度量，将特征空间总变差作为结构化剪枝指标，并结合置信度级联路由实现离线拓扑构建与在线动态推理的解耦，在ImageNet-1K极端压缩下避免传统指标导致的结构崩溃，在ImageNet-100动态推理中以0.92x计算代价匹配全模型精度。

**[An Fpga Implementation Of Displacement Vector Search For Intra Pattern Copy In J](an_fpga_implementation_of_displacement_vector_search_for_intra_pattern_copy_in_j.md)**

:   首次提出JPEG XS帧内模式复制(IPC)中位移向量(DV)搜索模块的FPGA架构实现，采用四级流水线设计和优化的存储组织方式，在Xilinx Artix-7上实现38.3 Mpixels/s吞吐量和277 mW功耗，为IPC实际硬件部署和ASIC转化奠定基础。

**[Arche Autoregressive Residual Compression With Hyperprior And Excitation](arche_autoregressive_residual_compression_with_hyperprior_and_excitation.md)**

:   提出ARCHE端到端学习型图像压缩框架，在统一概率架构中整合分层Hyperprior、掩码空间自回归上下文、通道条件化和SE激励通道重校准，无需Transformer或循环组件，在Kodak上相对Ballé基线BD-Rate降低约48%，相对VVC Intra降低约5.6%，仅95M参数和222ms解码时间。

**[Autossvh Exploring Automated Frame Sampling For Efficient Self-Supervised Video H](autossvh_exploring_automated_frame_sampling_for_efficient_self-supervised_video_h.md)**

:   提出AutoSSVH方法，通过对抗式自动帧采样网络（Grade-Net）选择最具挑战性的帧子集作为训练信号，并设计P2Set（Point-to-Set）哈希对比学习范式，实现了高效的自监督视频哈希检索，在UCF101和HMDB51上大幅超越现有方法。

**[Bhvit Binarized Hybrid Vision Transformer](bhvit_binarized_hybrid_vision_transformer.md)**

:   针对 ViT 二值化性能严重下降的问题，提出专为二值化设计的混合 ViT 架构 BHViT，包含多尺度分组空洞卷积 token mixer、量化分解注意力矩阵二值化、shift 增强的 MLP 和正则化损失，在 ImageNet-1K 上达到 1-bit 二值化模型的 SOTA 性能。

**[Charm The Missing Piece In Vit Fine-Tuning For Image Aesthetic Assessment](charm_the_missing_piece_in_vit_fine-tuning_for_image_aesthetic_assessment.md)**

**[Cl-Lora Continual Low-Rank Adaptation For Rehearsal-Free Class-Incremental Learn](cl-lora_continual_low-rank_adaptation_for_rehearsal-free_class-incremental_learn.md)**

:   提出 CL-LoRA，设计双适配器架构（任务共享 + 任务特定 LoRA），结合知识蒸馏与梯度重分配以及可学习块级权重，在仅 0.3% 可训练参数下实现 SOTA 持续学习性能。

**[Coa Towards Real Image Dehazing Via Compression-And-Adaptation](coa_towards_real_image_dehazing_via_compression-and-adaptation.md)**

:   提出压缩-适应（CoA）框架实现实际图像去雾：先在合成数据上训练大模型，然后压缩+适应到真实域，平衡性能和部署效率

**[Geochemad Benchmarking Unsupervised Geochemical Anomaly Detection For Mineral Ex](geochemad_benchmarking_unsupervised_geochemical_anomaly_detection_for_mineral_ex.md)**

:   提出 GeoChemAD 开源基准数据集（8 个子集，覆盖多区域/多采样源/多目标元素）和 GeoChemFormer 框架，通过空间上下文自监督预训练和元素依赖建模实现无监督地球化学异常检测，在所有子集上取得最优 AUC。

**[Hiap A Multi-Granular Stochastic Auto-Pruning Framework For Vision Transformers](hiap_a_multi-granular_stochastic_auto-pruning_framework_for_vision_transformers.md)**

:   HiAP 提出了一种多粒度自动剪枝框架，通过在宏观（attention heads、FFN blocks）和微观（intra-head dimensions、FFN neurons）两级部署可学习 Gumbel-Sigmoid 门控，在单阶段端到端训练中自动发现最优子网络，无需手工重要性排序或后处理阈值。
