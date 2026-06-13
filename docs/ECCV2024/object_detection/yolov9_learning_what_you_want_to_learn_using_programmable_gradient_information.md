---
title: >-
  [论文解读] YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information
description: >-
  [ECCV 2024][目标检测][YOLO] YOLOv9 提出可编程梯度信息 (PGI) 和广义高效层聚合网络 (GELAN) 来解决深度网络中的信息瓶颈问题，在 MS COCO 上以更少参数和计算量全面超越现有实时目标检测器，从零训练即可超过使用大数据集预训练的方法。 领域现状： 深度学习已在计算机视觉等领域取得远超传…
tags:
  - "ECCV 2024"
  - "目标检测"
  - "YOLO"
  - "可编程梯度信息"
  - "信息瓶颈"
  - "轻量级网络"
  - "实时目标检测"
---

# YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information

**会议**: ECCV 2024  
**arXiv**: [2402.13616](https://arxiv.org/abs/2402.13616)  
**代码**: [https://github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)  
**领域**: 目标检测  
**关键词**: YOLO, 可编程梯度信息, 信息瓶颈, 轻量级网络, 实时目标检测

## 一句话总结

YOLOv9 提出可编程梯度信息 (PGI) 和广义高效层聚合网络 (GELAN) 来解决深度网络中的信息瓶颈问题，在 MS COCO 上以更少参数和计算量全面超越现有实时目标检测器，从零训练即可超过使用大数据集预训练的方法。

## 研究背景与动机

**领域现状**: 深度学习已在计算机视觉等领域取得远超传统方法的性能，研究者主要聚焦于更强大的网络架构（CNN、Transformer、Mamba）和更合适的目标函数（损失函数、标签分配、辅助监督）。YOLO 系列是当前最广泛使用的实时目标检测方案。

**现有痛点**: 当输入数据经过逐层特征提取和空间变换时，大量信息会丢失（信息瓶颈现象）。这导致损失函数无法生成可靠的梯度来更新模型权重，进而使模型在数据和目标之间建立错误的关联。

**核心矛盾**: 现有解决信息丢失的方法各有缺陷：(1) 可逆架构（如 RevCol）虽能保留信息，但显著增加推理成本（额外连接增加 20% 推理时间，甚至翻倍）；(2) 掩码建模的重建损失可能与目标损失冲突；(3) 深度监督会产生误差累积，且在浅层模型和轻量级模型上效果不佳。

**本文目标** 如何在不增加推理成本的前提下，为深度网络提供完整可靠的梯度信息，使得从轻量级到大型的各种模型都能从辅助监督中受益？

**切入角度**: 从信息瓶颈原理和可逆函数理论出发，将可逆架构放在辅助分支而非主分支上，仅训练时使用以生成可靠梯度，推理时去掉以避免额外开销。

**核心 idea**: 通过辅助可逆分支在不同语义层级编程梯度信息传播，让主分支获得可靠梯度更新，同时设计 GELAN 以常规卷积实现超越深度可分离卷积的参数利用效率。

## 方法详解

### 整体框架

YOLOv9 由两个核心创新组成：**PGI (Programmable Gradient Information)** 和 **GELAN (Generalized Efficient Layer Aggregation Network)**。PGI 是一个辅助监督框架，包含主分支、辅助可逆分支和多级辅助信息三个组件。GELAN 是一个结合 CSPNet 和 ELAN 设计思想的新型网络架构。推理时仅使用主分支（GELAN），不引入额外计算。

### 关键设计

1. **辅助可逆分支 (Auxiliary Reversible Branch)**:

    - **功能**: 在训练阶段引入可逆架构作为辅助分支，生成可靠梯度来更新主分支参数。
    - **核心思路**: 根据信息瓶颈原理，数据 $X$ 经过变换后互信息递减：$I(X,X) \geq I(X,f_\theta(X)) \geq I(X,g_\phi(f_\theta(X)))$。可逆函数 $r$ 满足 $X = v_\zeta(r_\psi(X))$，可保持信息不丢失：$I(X,X) = I(X,r_\psi(X))$。将可逆分支放在辅助位置，推理时可直接移除。
    - **设计动机**: 主分支深层特征因信息瓶颈丢失了重要信息，辅助可逆分支提供完整信息的梯度来"修正"主分支的参数学习。具体实现采用 DHLC (DynamicDet) 连接方式构建 ICN (Information Complete Network)。
    - **优势**: 不增加推理成本；适用于浅层网络（传统可逆架构在浅层网络反而表现更差）。

2. **多级辅助信息 (Multi-level Auxiliary Information)**:

    - **功能**: 在辅助监督的特征金字塔层级之间插入集成网络，聚合来自不同预测头的梯度信息。
    - **核心思路**: 传统深度监督中，不同特征金字塔被引导学习不同尺度目标（如浅层学小目标），这会导致其他尺度信息丢失（信息断裂问题）。多级辅助信息确保每个金字塔层级都接收到所有目标对象的信息。
    - **设计动机**: 解决深度监督中误差累积和信息断裂的问题。可使用 FPN、PAN 等不同集成网络来规划所需的语义级别，使不同规模的网络架构都能从中受益。

3. **GELAN (Generalized Efficient Layer Aggregation Network)**:

    - **功能**: 结合 CSPNet 的跨阶段部分连接和 ELAN 的高效层聚合设计，构建支持任意计算块的通用网络架构。
    - **核心思路**: 将 ELAN 中原本仅使用卷积层堆叠的设计泛化为支持任意计算块（Res block、Dark block、CSP block 等），用户可根据推理设备自由选择合适的计算块。
    - **设计动机**: 基于梯度路径规划设计，确保梯度能高效传播。实验证明使用 CSP block 效果最佳（参数更少、计算量更小、AP 更高）。GELAN 对深度设置不敏感，ELAN 深度和 CSP 深度的变化呈线性关系，无需特殊调优即可获得稳定性能。

### 损失函数 / 训练策略

- 完全遵循 YOLOv7 AF 的训练设置，从零训练 500 epochs
- 前 3 个 epoch 使用线性学习率热身
- 最后 15 个 epoch 关闭 mosaic 数据增强
- PGI 的辅助损失部分完全沿用 YOLOv7 的辅助头设置
- 采用 lead-head guided (LHG) 标签分配策略用于 PGI 的辅助监督，效果更佳
- 使用 anchor-free 预测头

## 实验关键数据

### 主实验

在 MS COCO 2017 val 集上与现有 SOTA 实时目标检测器的全面对比（全部从零训练）：

| 模型 | 参数量 (M) | FLOPs (G) | AP$_{50:95}$ (%) | 与同级别SOTA对比 |
|---|---|---|---|---|
| YOLO MS-S (之前SOTA-轻量) | 8.1 | 31.2 | 46.2 | - |
| **YOLOv9-S (ours)** | **7.1** | **26.4** | **46.8** | +0.6, 参数-12%, FLOPs-15% |
| YOLO MS (之前SOTA-中型) | 22.2 | 80.2 | 51.0 | - |
| **YOLOv9-M (ours)** | **20.0** | **76.3** | **51.4** | +0.4, 参数-10%, FLOPs-5% |
| YOLOv7 AF (之前SOTA-通用) | 43.6 | 130.5 | 53.0 | - |
| **YOLOv9-C (ours)** | **25.3** | **102.1** | **53.0** | 相同AP, 参数-42%, FLOPs-22% |
| YOLOv8-X (之前SOTA-大型) | 68.2 | 257.8 | 53.9 | - |
| **YOLOv9-E (ours)** | **57.3** | **189.0** | **55.6** | +1.7, 参数-16%, FLOPs-27% |

与 ImageNet 预训练方法对比：YOLOv9 从零训练超越 RT-DETR (使用 ImageNet 预训练)，仅需 RT-DETR-X 66% 的参数即可达到相同精度。

### 消融实验

**GELAN 计算块消融 (Table 2)**:

| 计算块类型 | 参数量 | FLOPs | AP$_{50:95}$ | 说明 |
|---|---|---|---|---|
| Conv (原始 ELAN) | 6.2M | 23.5G | 44.8% | 基线 |
| Res block | 5.4M | 21.0G | 44.3% | 参数最少但精度略降 |
| Dark block | 5.7M | 21.8G | 44.5% | 参数和精度居中 |
| **CSP block** | **5.9M** | **22.4G** | **45.5%** | 参数减少但AP提升0.7% |

**PGI 对不同规模模型的影响 (Table 5)**:

| 模型 | 基线 AP | +深度监督 | +PGI | 说明 |
|---|---|---|---|---|
| GELAN-S (轻量) | 46.7% | 46.5% (-0.2) | 46.8% (+0.1) | DS对轻量模型有害，PGI有效 |
| GELAN-M (中型) | 51.1% | 51.2% (+0.1) | 51.4% (+0.3) | DS不稳定，PGI稳定提升 |
| GELAN-C (通用) | 52.5% | 52.5% (=) | 53.0% (+0.5) | DS无增益，PGI显著提升 |
| GELAN-E (大型) | 55.0% | 55.3% (+0.3) | 55.6% (+0.6) | DS仅在大模型有效，PGI更优 |

### 关键发现

1. **信息保留可视化**: 使用随机初始权重前馈可视化，PlainNet 在 50 层后丢失目标位置信息，ResNet 在 100 层后变模糊，而 GELAN 在 200 层仍可清晰辨识目标边界。
2. **PGI 梯度质量可视化**: 仅经过一个 epoch 的 bias warm-up，PGI 就能准确地将注意力聚焦在目标区域，而不使用 PGI 的 GELAN 在目标边界处存在发散现象。
3. **CSP block 是 GELAN 的最佳计算块**: 参数减少但精度反升 0.7%。
4. **深度监督对轻量模型有害**: DS 在 GELAN-S 上导致 0.2% AP 下降，而 PGI 在所有规模模型上均有正向增益。
5. **GELAN 深度不敏感**: 参数/计算量/精度随深度呈线性关系，无需精心调参。
6. **常规卷积超越深度可分离卷积**: GELAN 使用常规卷积即可超越基于 depthwise convolution 的 YOLO MS 在参数利用效率上的表现。

## 亮点与洞察

- **理论驱动的实用创新**: 从信息瓶颈原理和可逆函数的理论分析出发，推导出信息丢失是深度网络训练困难的根本原因，而非传统认为的梯度消失/饱和。
- **辅助分支零推理成本**: 可逆分支仅在训练时存在，推理时完全移除，实现了"鱼和熊掌兼得"。
- **打破深度监督的限制**: 传统深度监督仅适用于极深网络，PGI 使得轻量模型也能从辅助监督中获益。
- **从零训练胜过预训练**: YOLOv9 不依赖 ImageNet 预训练，仅从零训练即可超越 RT-DETR 等使用大数据集预训练的方法。
- **模块化设计哲学**: GELAN 允许自由替换计算块，适配不同推理设备需求。

## 局限与展望

- 辅助可逆分支的设计（ICN）增加了训练时的显存和计算开销，虽然推理无影响但训练成本较高。
- 论文仅在 MS COCO 目标检测上验证，未展示在其他任务（实例分割、关键点检测等）上的泛化性。
- GELAN 的 ELAN/CSP 深度虽不敏感，但缺乏自动搜索最优配置的机制（依赖手动设定）。
- 与 NAS (神经架构搜索) 方法的对比缺失，GELAN 是否在搜索空间中是最优的尚不明确。
- PGI 的辅助分支设计与具体的检测框架耦合较深，迁移到其他任务可能需要重新设计。

## 相关工作与启发

- **信息瓶颈理论**: Tishby et al. 提出的信息瓶颈原理为理解深度网络信息丢失提供了理论基础。
- **可逆架构 (RevCol, RevNet)**: 保证信息不丢失但推理成本高，本文将其巧妙地放在辅助分支解决了这一矛盾。
- **ELAN / CSPNet**: GELAN 直接建立在这两者的梯度路径规划思想之上。
- **DynamicDet / CBNet**: DHLC 连接方式被用于构建辅助可逆分支 ICN。
- **深度监督**: PGI 本质上是深度监督的广义版本，但通过多级辅助信息解决了误差累积问题。

## 评分

- **新颖性**: ⭐⭐⭐⭐ PGI 从信息瓶颈理论出发提出辅助可逆分支的思路新颖；GELAN 是成熟技术的巧妙组合
- **实验充分度**: ⭐⭐⭐⭐⭐ 消融实验极其详尽（计算块类型、深度、PGI 组件、不同模型规模），可视化有力
- **写作质量**: ⭐⭐⭐⭐ 理论分析清晰，从问题到方案的逻辑链完整
- **价值**: ⭐⭐⭐⭐⭐ YOLO 系列影响力巨大，PGI 和 GELAN 的设计理念对后续工作有深远影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Can OOD Object Detectors Learn from Foundation Models?](can_ood_object_detectors_learn_from_foundation_models.md)
- [\[CVPR 2025\] Boosting Domain Incremental Learning: Selecting the Optimal Parameters Is All You Need](../../CVPR2025/object_detection/boosting_domain_incremental_learning_selecting_the_optimal_parameters_is_all_you.md)
- [\[ECCV 2024\] Bridge Past and Future: Overcoming Information Asymmetry in Incremental Object Detection](bridge_past_and_future_overcoming_information_asymmetry_in_incremental_object_de.md)
- [\[ICML 2025\] Outlier Gradient Analysis: Efficiently Identifying Detrimental Training Samples for Deep Learning Models](../../ICML2025/object_detection/outlier_gradient_analysis_efficiently_identifying_detrimental_training_samples_f.md)
- [\[ECCV 2024\] AugDETR: Improving Multi-scale Learning for Detection Transformer](augdetr_improving_multi-scale_learning_for_detection_transformer.md)

</div>

<!-- RELATED:END -->
