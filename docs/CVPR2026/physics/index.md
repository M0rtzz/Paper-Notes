---
title: >-
  CVPR2026 物理/科学计算论文汇总 · 5篇论文解读
description: >-
  5篇CVPR2026的物理/科学计算方向论文解读，涵盖知识蒸馏、持续学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "物理/科学计算"
  - "论文解读"
  - "论文笔记"
  - "知识蒸馏"
  - "持续学习"
item_list:
  - u: "continuous_exposure-time_modeling_for_realistic_atmospheric_turbulence_synthesis/"
    t: "Continuous Exposure-Time Modeling for Realistic Atmospheric Turbulence Synthesis"
  - u: "high-quality_and_efficient_turbulence_mitigation_with_events/"
    t: "EHETM: High-Quality and Efficient Turbulence Mitigation with Events"
  - u: "nestor_a_nested_moe-based_neural_operator_for_large-scale_pde_pre-training/"
    t: "NESTOR: A Nested MOE-based Neural Operator for Large-Scale PDE Pre-Training"
  - u: "physskin_real-time_and_generalizable_physics-based_animation_via_self-supervised/"
    t: "PhysSkin: Real-Time and Generalizable Physics-Based Skin Simulation"
  - u: "qkd_quantum_gated_incremental_learning/"
    t: "QKD: Quantum-Gated Task-interaction Knowledge Distillation for Class-Incremental Learning"
item_total: 5
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚛️ 物理/科学计算

**📷 CVPR2026** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (26)](../../ICML2026/physics/index.md) · [🔬 ICLR2026 (16)](../../ICLR2026/physics/index.md) · [🤖 AAAI2026 (14)](../../AAAI2026/physics/index.md) · [🧠 NeurIPS2025 (55)](../../NeurIPS2025/physics/index.md) · [📹 ICCV2025 (2)](../../ICCV2025/physics/index.md) · [🧪 ICML2025 (20)](../../ICML2025/physics/index.md)

**[Continuous Exposure-Time Modeling for Realistic Atmospheric Turbulence Synthesis](continuous_exposure-time_modeling_for_realistic_atmospheric_turbulence_synthesis.md)**

:   提出曝光时间依赖的调制传递函数（ET-MTF），将曝光时间建模为连续变量，构建了大规模合成湍流数据集 ET-Turb（5083视频、200万帧），显著提升湍流复原模型在真实数据上的泛化能力。

**[EHETM: High-Quality and Efficient Turbulence Mitigation with Events](high-quality_and_efficient_turbulence_mitigation_with_events.md)**

:   提出EHETM，首次利用事件相机的微秒时间分辨率突破传统多帧湍流缓解(TM)方法的精度-效率瓶颈，发现两个关键物理现象——湍流诱导事件的极性交替与清晰梯度相关、动态物体形成时空相干"事件管"——设计极性加权梯度和事件管约束两个互补模块，数据开销降低77.3%、系统延迟降低89.5%，尤其在动态物体场景显著超越SOTA。

**[NESTOR: A Nested MOE-based Neural Operator for Large-Scale PDE Pre-Training](nestor_a_nested_moe-based_neural_operator_for_large-scale_pde_pre-training.md)**

:   提出嵌套式 MoE 神经算子 NESTOR，通过 image-level MoE 捕获不同 PDE 类型的全局特征 + token-level Sub-MoE 捕获物理场内局部相关性，在 12 个 PDE 数据集上实现大规模预训练并有效迁移到下游任务。

**[PhysSkin: Real-Time and Generalizable Physics-Based Skin Simulation](physskin_real-time_and_generalizable_physics-based_animation_via_self-supervised.md)**

:   提出 PhysSkin，一个泛化的物理信息框架——通过神经蒙皮场自编码器从静态 3D 几何体直接学习连续蒙皮权重场，配合物理信息自监督学习策略（能量最小化+平滑性+正交性约束），实现跨形状、跨离散化的实时物理动画，无需任何标注数据或仿真轨迹。

**[QKD: Quantum-Gated Task-interaction Knowledge Distillation for Class-Incremental Learning](qkd_quantum_gated_incremental_learning.md)**

:   QKD 将量子门控引入类增量学习，通过参数化量子电路在高维 Hilbert 空间中建模样本-任务相关性，引导跨任务知识蒸馏和推理时适配器融合，在 5 个基准上达到 SOTA。
