---
title: >-
  CVPR2026 计算生物方向11篇论文解读
description: >-
  11篇CVPR2026的计算生物方向论文解读，涵盖生物分子、医学影像、自监督学习、压缩/编码、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "计算生物"
  - "论文解读"
  - "论文笔记"
  - "生物分子"
  - "医学影像"
  - "自监督学习"
  - "压缩/编码"
  - "多模态"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧬 计算生物

**📷 CVPR2026** · **11** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (35)](../../ICML2026/computational_biology/index.md) · [💬 ACL2026 (5)](../../ACL2026/computational_biology/index.md) · [🔬 ICLR2026 (38)](../../ICLR2026/computational_biology/index.md) · [🤖 AAAI2026 (20)](../../AAAI2026/computational_biology/index.md) · [🧠 NeurIPS2025 (72)](../../NeurIPS2025/computational_biology/index.md) · [📹 ICCV2025 (4)](../../ICCV2025/computational_biology/index.md)

🔥 **高频主题：** 生物分子 ×3

**[HINGE: Adapting a Pre-trained Single-Cell Foundation Model to Spatial Gene Expression Generation from Histology Images](adapting_a_pre-trained_single-cell_foundation_model_to_spatial_gene_expression_g.md)**

:   提出HINGE框架，首次将预训练的表达空间单细胞基础模型(sc-FM, CellFM)改装为组织学图像条件的空间基因表达生成器，通过恒等初始化的SoftAdaLN调制轻量注入视觉上下文、表达空间掩码扩散过程对齐预训练目标、warm-start课程稳定训练，在三个ST数据集上达SOTA并保持优越的基因共表达一致性。

**[CARE: A Molecular-Guided Foundation Model with Adaptive Region Modeling for Whole Slide Image Analysis](care_a_molecular-guided_foundation_model_with_adaptive_region_modeling_for_whole.md)**

:   提出 CARE，一种病理学 slide-level 基础模型，通过自适应区域生成器（ARG）将 WSI 划分为形态学相关的不规则区域（类似 NLP 中的词级 token），并结合 RNA/蛋白质表达谱的跨模态对齐进行两阶段预训练，仅用主流模型约 1/10 的数据即在 33 个下游任务上取得最优平均性能。

**[Cell-Type Prototype-Informed Neural Network for Gene Expression Estimation from Pathology Images](cell-type_prototype-informed_neural_network_for_gene_expression_estimation_from_.md)**

:   提出 CPNN，利用公开单细胞 RNA-seq 数据构建细胞类型原型（cell-type prototype），将 slide/patch 级基因表达建模为原型的加权组合，在基因表达估计任务上取得 SOTA 并提供可解释性。

**[Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference](cross-slice_knowledge_transfer_via_masked_multi-modal_heterogeneous_graph_contra.md)**

:   提出 SpaHGC，一种基于多模态异构图的框架，通过构建目标切片内、跨切片和参考切片内三种子图，结合 masked graph 对比学习和跨节点双注意力机制，实现从 H&E 病理图像预测空间基因表达，在七个数据集上 PCC 指标提升 7.3%-27.1%。

**[CryoHype: Reconstructing a Thousand Cryo-EM Structures with Transformer-Based Hypernetworks](cryohype_reconstructing_a_thousand_cryo-em_structures_with_transformer-based_hyp.md)**

:   提出 CryoHype，一种基于 Transformer 超网络的冷冻电镜重建方法，通过动态调整隐式神经表示（INR）的权重来减少参数共享，首次实现了从无标签冷冻电镜图像中同时重建 1000 种不同蛋白质结构。

**[cryoSENSE: Compressive Sensing Enables High-throughput Microscopy with Sparse and Generative Priors on the Protein Cryo-EM Image Manifold](cryosense_compressive_sensing_enables_high-throughput_microscopy_with_sparse_and.md)**

:   提出 cryoSENSE，首个冷冻电镜压缩成像的计算框架，证明蛋白质 cryo-EM 图像在稀疏先验（DCT/小波/TV）和生成先验（扩散模型）下均可从欠采样测量中高保真重建，在保持 3D 分辨率的同时实现最高 2.5× 通量提升。

**[Hyperbolic Busemann Neural Networks](hyperbolic_busemann_neural_networks.md)**

:   利用 Busemann 函数将多类逻辑回归（MLR）和全连接层（FC）内蕴地提升到双曲空间，提出 BMLR 和 BFC 两个统一组件，在 Poincaré 球和 Lorentz 模型上同时适用，且在图像分类、基因组序列、节点分类、链接预测四类任务上均优于已有双曲层。

**[Multimodal Protein Language Models for Enzyme Kinetic Parameters: From Substrate Recognition to Conformational Adaptation](multimodal_protein_language_models_for_enzyme_kinetic_parameters_from_substrate_.md)**

:   提出**ERBA(Enzyme-Reaction Bridging Adapter)**，将酶动力学参数预测重新建模为**分阶段多模态条件生成问题**——先通过MRCA注入底物信息捕获底物识别特异性，再通过G-MoE整合活性位点3D结构捕获构象适应，配合ESDA分布对齐保持PLM语义先验。

**[Sampling-Aware 3D Spatial Analysis in Multiplexed Imaging](sampling-aware_3d_spatial_analysis_in_multiplexed_imaging.md)**

:   本文系统研究了多重成像中采样几何（2D切片 vs 3D序列切片）对空间统计量恢复精度的影响，并提出了一种几何感知的稀疏3D重建模块，在有限的成像预算下实现可靠的深度感知空间分析。

**[SHREC: A Spectral Embedding-Based Approach for Ab-Initio Reconstruction of Helical Molecules](shrec_a_spectral_embedding-based_approach_for_ab-initio_reconstruction_of_helica.md)**

:   提出 SHREC 算法，通过谱嵌入（spectral embedding）从冷冻电镜 2D 投影图像中直接恢复螺旋分子片段的投影角度，无需预先知道螺旋对称参数（rise/twist），实现了真正的 ab-initio 螺旋结构重建。

**[Stronger Normalization-Free Transformers](stronger_normalization-free_transformers.md)**

:   通过系统分析逐点函数替代归一化层所需的四个关键属性（零中心性、有界性、中心敏感性、单调性），在大规模搜索中发现 $\text{Derf}(x) = \text{erf}(\alpha x + s)$ 是最优的归一化层替代函数，在视觉识别、图像生成、语音表示和DNA序列建模等多个领域持续超越LayerNorm和DyT，且性能增益主要来自更强的泛化而非拟合能力。
