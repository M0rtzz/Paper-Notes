---
title: >-
  ICML2025 3D 视觉方向12篇论文解读
description: >-
  12篇ICML2025的 3D 视觉方向论文解读，涵盖 3D 高斯渲染、语义分割、图像编辑、点云、新视角合成、三维重建等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICML2025"
  - "3D 视觉"
  - "论文解读"
  - "论文笔记"
  - "3D 高斯渲染"
  - "语义分割"
  - "图像编辑"
  - "点云"
  - "新视角合成"
  - "三维重建"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧊 3D 视觉

**🧪 ICML2025** · **12** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (24)](../../ICML2026/3d_vision/index.md) · [💬 ACL2026 (1)](../../ACL2026/3d_vision/index.md) · [📷 CVPR2026 (236)](../../CVPR2026/3d_vision/index.md) · [🔬 ICLR2026 (62)](../../ICLR2026/3d_vision/index.md) · [🤖 AAAI2026 (76)](../../AAAI2026/3d_vision/index.md) · [🧠 NeurIPS2025 (112)](../../NeurIPS2025/3d_vision/index.md)

🔥 **高频主题：** 3D 高斯渲染 ×3 · 语义分割 ×2

**[EvoMesh: Adaptive Physical Simulation with Hierarchical Graph Evolutions](evomesh_adaptive_physical_simulation_with_hierarchical_graph_evolutions.md)**

:   EvoMesh 提出一种全可微的层次图演化框架，通过各向异性消息传递（AMP）和基于 Gumbel-Softmax 的可微节点选择（DiffSELECT），根据物理输入自适应构建随时间演化的多尺度图层次结构，在五个物理仿真基准上平均超越固定层次方法约 20%。

**[FlowDrag: 3D-aware Drag-based Image Editing with Mesh-guided Deformation Vector Flow Fields](flowdrag_3d-aware_drag-based_image_editing_with_mesh-guided_deformation_vector_f.md)**

:   提出 FlowDrag，从图像构建 3D 网格后利用渐进式 SR-ARAP 变形生成连续 2D 向量流场，将全局几何先验注入扩散模型的 motion supervision 过程，在 DragBench（MD=22.88）和新提出的 VFD-Bench（PSNR=18.55, 1-LPIPS=0.82, MD=28.23）上全面领先。

**[FreeMesh: Boosting Mesh Generation with Coordinates Merging](freemesh_boosting_mesh_generation_with_coordinates_merging.md)**

:   提出 Per-Token-Mesh-Entropy（PTME）度量来免训练评估网格tokenizer质量，并引入从NLP借鉴的 Rearrange & Merge Coordinates（RMC）坐标合并技术，在 MeshXL/MeshAnythingV2/EdgeRunner 三种tokenizer上实现最高21.2%的压缩率、显著增加可生成面片数和几何细节保留。

**[GAPrompt: Geometry-Aware Point Cloud Prompt for 3D Vision Model](gaprompt_geometry-aware_point_cloud_prompt_for_3d_vision_model.md)**

:   提出 GAPrompt，针对预训练 3D 视觉模型的几何感知 PEFT 方法，通过可学习点云提示 (Point Prompt)、点偏移提示器 (Point Shift Prompter) 和提示传播 (Prompt Propagation) 三个模块协同利用点云几何信息，仅训练 2.19% 参数即可匹配甚至超越全量微调。

**[High Dynamic Range Novel View Synthesis with Single Exposure](high_dynamic_range_novel_view_synthesis_with_single_exposure.md)**

:   首次提出仅使用单曝光LDR图像进行HDR新视角合成（HDR-NVS）的问题设定，并设计了一个基于相机成像原理的元算法框架Mono-HDR-3D，通过LDR→HDR颜色转换器（L2H-CC）和HDR→LDR闭环转换器（H2L-CC）实现无HDR监督下的HDR场景建模。

**[PhysicsNeRF: Physics-Guided 3D Reconstruction from Sparse Views](physicsnerf_physics-guided_3d_reconstruction_from_sparse_views.md)**

:   PhysicsNeRF 提出了一个基于物理先验的稀疏视角 NeRF 框架，通过深度排序、跨视角一致性、稀疏性正则和渐进训练四种互补约束，在仅 8 个视角下实现 21.4 dB 的 PSNR，并对稀疏视角下过拟合的本质进行了深入的理论分析。

**[Probabilistic Interactive 3D Segmentation with Hierarchical Neural Processes](probabilistic_interactive_3d_segmentation_with_hierarchical_neural_processes.md)**

:   NPISeg3D提出了首个基于层次化神经过程（Hierarchical Neural Processes）的概率交互式3D分割框架，通过场景级和物体级双层潜变量结构以及概率原型调制器，在少量点击下实现了优于AGILE3D的分割精度，同时提供可靠的不确定性估计。

**[ReferSplat: Referring Segmentation in 3D Gaussian Splatting](refersplat_referring_segmentation_in_3d_gaussian_splatting.md)**

:   ReferSplat 提出了 Referring 3D Gaussian Splatting Segmentation（R3DGS）新任务，通过构建 3D Gaussian Referring Fields、位置感知跨模态交互模块和 Gaussian-Text 对比学习，实现了基于自然语言描述在 3DGS 场景中分割目标物体（包括遮挡/不可见物体），在新构建的 Ref-LERF 数据集和开放词汇分割基准上取得 SOTA。

**[SE(3)-Equivariant Diffusion Policy in Spherical Fourier Space](se3-equivariant_diffusion_policy_in_spherical_fourier_space.md)**

:   提出在球面 Fourier 空间中构建 SE(3) 等变扩散策略，利用球谐函数的等变性质使策略对输入场景的刚体变换保持等变，在机器人操作任务上实现更好的空间泛化。

**[LaGa: Tackling View-Dependent Semantics in 3D Language Gaussian Splatting](tackling_view-dependent_semantics_in_3d_language_gaussian_splatting.md)**

:   提出LaGa方法，通过3D场景分解建立跨视角语义连接、用自适应聚类+双因子重加权构建视角聚合语义表示，解决3D语言高斯中被忽视的视角依赖语义问题，在LERF-OVS上3D mIoU达64.0%（+18.7%）。

**[Thickness-aware E(3)-Equivariant 3D Mesh Neural Networks](thickness-aware_e3-equivariant_3d_mesh_neural_networks.md)**

:   提出 T-EMNN，通过引入厚度感知的消息传递机制和基于 PCA 的数据驱动坐标系，在保持表面网格计算效率的同时建模对立面之间的厚度交互，实现 E(3)-等变/不变的节点级 3D 形变预测。

**[VTGaussian-SLAM: RGBD SLAM for Large Scale Scenes with Splatting View-Tied 3D Gaussians](vtgaussian-slam_rgbd_slam_for_large_scale_scenes_with_splatting_view-tied_3d_gau.md)**

:   提出视图绑定3D高斯（View-Tied 3D Gaussians），将高斯绑定到深度像素上并简化为球形，大幅节省存储开销，配合仅优化最近视图相关高斯的tracking/mapping策略，实现面向大规模场景的可扩展RGBD SLAM系统。
