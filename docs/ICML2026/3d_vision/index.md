---
title: >-
  ICML2026 3D 视觉方向8篇论文解读
description: >-
  8篇ICML2026的 3D 视觉方向论文解读，涵盖布局/合成、推理、三维重建、人脸/视线、点云、形状补全等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "3D 视觉"
  - "论文解读"
  - "论文笔记"
  - "布局/合成"
  - "推理"
  - "三维重建"
  - "人脸/视线"
  - "点云"
  - "形状补全"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧊 3D 视觉

**🧪 ICML2026** · **8** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (230)](../../CVPR2026/3d_vision/index.md) · [🔬 ICLR2026 (63)](../../ICLR2026/3d_vision/index.md) · [🤖 AAAI2026 (76)](../../AAAI2026/3d_vision/index.md) · [🧠 NeurIPS2025 (112)](../../NeurIPS2025/3d_vision/index.md) · [📹 ICCV2025 (265)](../../ICCV2025/3d_vision/index.md) · [🧪 ICML2025 (12)](../../ICML2025/3d_vision/index.md)

🔥 **高频主题：** 布局/合成 ×2

**[FSI2P: A Hierarchical Focus–Sweep Registration Network with Dynamically Allocated Depth](fs-i2pa_hierarchical_focus-sweep_registration_network_with_dynamically_allocated.md)**

:   本文把人类“先扫一眼再逐块细看”的观察过程抽象为 Focus-Sweep 两阶段范式，用 Mamba 替换 Transformer 做图像-点云交互，并用强化学习动态决定每个尺度上的交互层数，在 RGB-D Scenes V2 和 7-Scenes 上拿到 I2P 配准的 SOTA。

**[LabBuilder: Protocol-Grounded 3D Layout Generation for Interactable and Safe Laboratory](labbuilder_protocol-grounded_3d_layout_generation_for_interactable_and_safe_labo.md)**

:   LabBuilder 把自由文本的实验描述编译成"资产-化学协议"，再用层级化生成 + 几何/化学多目标优化 + 导航修复，产出既视觉合理、又能让机器人真正跑通实验流程的 3D 化学实验室布局。

**[Pair2Scene: Learning Local Object Relations for Procedural Scene Generation](pair2scene_learning_local_object_relations_for_procedural_scene_generation.md)**

:   Pair2Scene 把 3D 室内场景生成从「直接拟合全局联合分布」改成「学习一对一的局部物体关系（支撑 + 功能）然后按场景层级树递归装配」，配合点云几何编码、Mixture-of-Logistics 概率头和碰撞感知拒绝采样，在仅用 3D-Front 数据训练时即可生成对象数从约 4 跃升到约 14 的复杂场景，FID 和用户研究均优于 ATISS、DiffuScene、LayoutVLM 等基线。

**[PhysForge: Generating Physics-Grounded 3D Assets for Interactive Virtual World](physforge_generating_physics-grounded_3d_assets_for_interactive_virtual_world.md)**

:   把"造可交互 3D 物体"重新理解成"先做物理规划、再做物理生成"的两阶段问题——VLM 充当物理建筑师生成包含层级关系、材料、运动学约束的 "Hierarchical Physical Blueprint"，扩散模型再用 KineVoxel Injection 把铰接参数和几何 voxel 协同去噪，配合 150k 资产、四层标注的 PhysDB 数据集，首次实现单视图到"可在物理引擎里抓握、推动、铰接"的 3D 资产生成。

**[PhysHanDI: Physics-Based Reconstruction of Hand-Deformable Object Interactions](physhandi_physics-based_reconstruction_of_hand-deformable_object_interactions.md)**

:   本文提出 PhysHanDI，把 MANO 手模型和 Spring-Mass 软体模型耦合起来，用稠密手网格驱动可变形物体的物理仿真，并反向利用物体仿真去精化手的重建，在稀疏视角 RGB-D 视频上同时拿到了手和软物的稠密 3D 重建 SOTA。

**[R$^3$L: Reasoning 3D Layouts from Relative Spatial Relations](r3l_reasoning_3d_layouts_from_relative_spatial_relations.md)**

:   R³L 把 MLLM 多跳"相对空间关系"推理的两类系统性误差（语义漂移与度量漂移）归因于"反复发生的参考系变换"，并通过不变性空间分解（缩短关系链）、一致性空间想象（imagine-and-revise 循环消除冲突）与支持性空间优化（全局-局部位姿重参数化）三个模块，让 GPT-5 生成的开放词汇 3D 场景在 9 类场景下的碰撞率与越界率都接近 0、语义指标显著反超 LayoutVLM/Holodeck/LayoutGPT。

**[Revisiting Photometric Ambiguity for Accurate Gaussian-Splatting Surface Reconstruction](revisiting_photometric_ambiguity_for_accurate_gaussian-splatting_surface_reconst.md)**

:   AmbiSuR 把 Gaussian Splatting 的两类内生光度歧义（基元边缘外溢、像素混合欠约束）显式建模并用截断 + 射线-颜色一致性消歧，再借高阶球谐系数作"自指示器"找出歧义高风险基元并做无定形局部先验正则，在 DTU 上把平均 Chamfer 距离降到 0.46，超过此前最优 GeoSVR (0.47)。

**[SplAttN: Bridging 2D and 3D with Gaussian Soft Splatting and Attention for Point Cloud Completion](splattn_bridging_2d_and_3d_with_gaussian_soft_splatting_and_attention_for_point_.md)**

:   本文指出多模态点云补全里"硬投影把 3D 点直接打到 2D 网格"会让支持集 Lebesgue 测度为零、梯度被 Dirac delta 截断（称为 Cross-Modal Entropy Collapse），用可微 Gaussian Soft Splatting 把硬投影换成连续密度估计，搭配 EdgeConv 局部 + Transformer 全局的混合编码器和全局-局部解码器，在 PCN/ShapeNet-55/34 拿到 SOTA，并用 KITTI 上的 counter-factual 评估证明 baseline 实际是退化的"单模态模板检索器"。
