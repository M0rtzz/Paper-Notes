---
title: >-
  [论文解读] UniFusion: A Unified Image Fusion Framework with Robust Representation and Source-Aware Preservation
description: >-
  [CVPR2026][优化][unified image fusion] 提出 UniFusion 统一图像融合框架，利用 DINOv3 自监督语义先验构建跨模态共享特征空间，通过重建对齐机制保留源图信息，并以双层优化策略解耦重建与融合目标，在红外-可见光、多曝光、多焦点、医学图像等多任务上均达到 SOTA。
tags:
  - CVPR2026
  - 优化
  - unified image fusion
  - DINOv3
  - bilevel optimization
  - reconstruction alignment
  - cross-task generalization
---

# UniFusion: A Unified Image Fusion Framework with Robust Representation and Source-Aware Preservation

**会议**: CVPR2026  
**arXiv**: [2603.14214](https://arxiv.org/abs/2603.14214)  
**代码**: [dusongcheng/UniFusion](https://github.com/dusongcheng/UniFusion)  
**领域**: optimization / image fusion  
**关键词**: unified image fusion, DINOv3, bilevel optimization, reconstruction alignment, cross-task generalization

## 一句话总结

提出 UniFusion 统一图像融合框架，利用 DINOv3 自监督语义先验构建跨模态共享特征空间，通过重建对齐机制保留源图信息，并以双层优化策略解耦重建与融合目标，在红外-可见光、多曝光、多焦点、医学图像等多任务上均达到 SOTA。

## 研究背景与动机

**图像融合的核心目标**：将多源图像的互补信息整合为单一、信息丰富且视觉一致的表示，服务于目标检测、医学诊断、自动驾驶等下游任务。

**任务特定方法的局限**：现有方法（CDDFuse、CoCoNet、LRRNet 等）大多针对特定融合场景设计（红外-可见光、多曝光、多焦点），采用定制化的 CNN/AE/GAN 架构，泛化能力有限，难以适应多样化融合需求。

**通用融合框架的现状**：近期的 Transformer 架构（SwinFusion）、扩散模型方法、以及 TC-MoA 等尝试用单一模型处理多任务，但仍受限于两个核心瓶颈。

**瓶颈一：缺乏模态一致的特征提取机制**——现有共享 backbone 无法在异质信号（红外热成像 vs. 可见光纹理）间建立原则性的、鲁棒的统一编码。

**瓶颈二：深层传播中源信息退化**——特征在深层网络中传播时，模态特有线索（如可见光纹理、红外辐射对比）逐渐丢失，导致融合质量次优。

**本文切入点**：能否利用大规模自监督预训练模型（DINOv3）的强语义先验，配合显式的重建约束和优化解耦策略，同时解决上述两个瓶颈？

## 方法详解

### 整体框架

UniFusion 由三大模块组成：(1) 基于冻结 DINOv3 的双分支语义特征提取 + 轻量 Adapter 域适配；(2) Cross-Attention 融合模块；(3) 重建对齐分支。训练采用双层优化：内层更新 Adapter + 重建分支参数 $\phi$（重建目标），外层更新融合网络参数 $\theta$（融合目标），两者交替迭代实现协同优化。

### 关键设计 1：DINOv3 语义先验适配（Semantic Prior Adaptation）

- **功能**：以冻结的 DINOv3 ViT 作为通用语义 backbone，从两个模态分别提取多层特征 $f^{(l_2)}, f^{(l_5)}, f^{(l_8)}, f^{(l_{11})}$，再通过轻量级层次化 Adapter 进行渐进式特征校准。
- **核心思路**：Adapter 作为"特征翻译器"，通过多阶段残差融合和上采样，将深层全局语义与浅层细粒度结构逐级整合，输出模态对齐的嵌入。
- **设计动机**：DINOv3 在大规模自然图像上预训练，具有强大的物体中心先验和长程上下文依赖，但其潜空间与特定模态（红外、医学影像）存在域偏差。Adapter 以极低参数开销弥补这一偏差，同时保持冻结 backbone 的泛化能力，避免灾难性遗忘。

### 关键设计 2：重建对齐机制（Reconstruction Alignment）

- **功能**：为每个模态分支增加一个轻量重建分支 $R_m$（数层 Transformer + 投影头），从 Adapter 输出的校准特征 $\hat{\mathbf{F}}_m$ 重建原始输入 $\bar{I}_m = R_m(\hat{\mathbf{F}}_m)$。
- **核心思路**：通过自重建约束，强制编码器在共享潜空间中保留足够的模态特有信息（纹理、辐射对比等），防止语义漂移和信息丢失。
- **设计动机**：传统方法仅约束融合输出与源图的相似性（像素级 L1/SSIM），容易偏向浅层纹理模仿而忽视深层语义对应。重建对齐从编码端入手，确保特征本身是"可逆的"——能还原回源图，从而在特征层面保证信息完备性。消融实验（Fig. 8）直观显示，去掉重建分支后，编码特征丢失了显著的模态特有语义表示。

### 关键设计 3：双层优化策略（Bilevel Optimization）

- **功能**：将训练形式化为双层优化：内层（lower-level）快速更新 Adapter + 重建参数 $\phi$ 以捕获模态特有语义，外层（upper-level）基于更新后的特征空间缓慢调整融合网络参数 $\theta$。
- **核心思路**：
  $$\phi^* = \arg\min_\phi \mathcal{L}_{\text{rec}}(\phi), \quad \theta^* = \arg\min_\theta \mathcal{L}_{\text{fuse}}(\theta; \phi^*)$$
  实际采用一阶交替方案：每次迭代先以较大学习率 $\eta_L$ 更新 $\phi$，再以较小学习率 $\eta_U$ 更新 $\theta$，并对 $\theta$ 施加 EMA 正则化以增强时序稳定性。
- **设计动机**：重建和融合是两个存在耦合的目标——如果联合端到端训练，重建信号可能干扰融合梯度，导致收敛不稳定。双层优化将两者解耦为不同时间尺度的子问题：先确保特征"记住"源图信息（内层），再在此基础上学习最优融合策略（外层），实现信息保留与融合质量的平衡。

### 关键设计 4：Cross-Attention 融合模块

- **功能**：4 个 Cross-Attention Block 对两个模态的适配特征进行动态交互，建模跨模态依赖并强调互补信息。
- **核心思路**：基于注意力机制，让每个模态特征作为 query 去关注另一模态的 key/value，自适应选择和强化有价值的互补区域。
- **设计动机**：相比简单的拼接或加权平均，cross-attention 能在空间和语义层面实现细粒度的信息交换，特别适合处理红外-可见光等信息高度互补的场景。

## 实验关键数据

### 表 1：多模态 & 多曝光融合定量对比

| 方法 | M3FD MI↑ | M3FD VIF↑ | M3FD $Q_{abf}$↑ | M3FD $Q_y$↑ | MEFB MI↑ | MEFB VIF↑ | MEFB CC↑ | MEFB PSNR↑ |
|------|----------|-----------|-----------------|-------------|----------|-----------|----------|------------|
| CDDFuse | 3.776 | 0.839 | 0.610 | 0.978 | 6.575 | 1.430 | 0.837 | 56.809 |
| SwinFusion | 2.945 | 0.618 | 0.480 | 0.936 | 5.318 | 1.459 | 0.894 | 59.009 |
| TC-MoA | 3.466 | 0.870 | 0.636 | 0.983 | 4.889 | 1.406 | 0.885 | 59.152 |
| **UniFusion** | **4.268** | **0.899** | **0.637** | 0.982 | **6.861** | **1.484** | **0.906** | **59.219** |

- UniFusion 在 M3FD 上 MI 指标达到 4.268，大幅领先 TC-MoA（3.466）约 23%
- MEFB 上四项指标全面最优，VIF 达 1.484（超越 SwinFusion 的 1.459）

### 表 2：消融实验（M3FD / MEFB / MFIF）

| 配置 | M3FD MI↑ | M3FD VIF↑ | MEFB MI↑ | MEFB VIF↑ | MFIF MI↑ | MFIF $Q_{abf}$↑ |
|------|----------|-----------|----------|-----------|----------|-----------------|
| w/o Adapter | 3.646 | 0.863 | 5.512 | 1.232 | 5.375 | 0.532 |
| w/o DINOv3 | 3.681 | 0.879 | 5.709 | 1.334 | 5.624 | 0.491 |
| w/o Reconstruction | 3.846 | 0.870 | 6.434 | 1.396 | 5.838 | 0.579 |
| w/o Bilevel Opt | 3.924 | 0.876 | 6.374 | 1.424 | 6.021 | 0.583 |
| **Full Model** | **4.268** | **0.899** | **6.861** | **1.484** | **6.253** | **0.685** |

- 每个组件均有显著贡献；去掉 Adapter 后 MFIF $Q_{abf}$ 从 0.685 降至 0.532（-22%）
- DINOv3 编码器的语义先验是基础，替换为普通 4 层 Transformer 后跨任务性能全面下降
- 重建对齐和双层优化各自独立贡献，两者协同效果最佳

## 亮点与洞察

1. **DINOv3 作为通用语义骨架的思路具有启发性**：冻结预训练 ViT + 轻量 Adapter 的范式类似 NLP 中的 LoRA/Adapter-tuning，首次在图像融合领域系统性验证了这一路线的有效性。
2. **重建对齐是一个优雅的信息保留机制**：不在融合输出端做约束，而在编码端通过自重建确保特征的信息完备性，思路新颖且 Fig. 8 的可视化令人信服。
3. **双层优化的形式化清晰**：将重建与融合解耦为不同时间尺度的优化子问题，理论上有据（bilevel optimization），实践中通过一阶交替近似高效实现。
4. **跨任务泛化能力强**：单一模型在 IVIF、MIF、MEF、MFF 四类任务上均达到或接近 SOTA，训练仅需 10K 迭代，具有实际应用价值。

## 局限与展望

1. **DINOv3 依赖**：冻结的 DINOv3 backbone 参数量大（ViT-Large/Giant 级别），推理开销较高，部署于边缘设备有挑战。可以探索蒸馏到更小的 backbone。
2. **双层优化的计算成本**：虽然采用了一阶近似，但两阶段交替更新仍增加了每迭代的计算量，且 $\eta_L / \eta_U$ 的比率需要仔细调参。
3. **缺乏非对齐场景的评估**：实验中未涉及源图存在几何失配（如手持多曝光、运动模糊）的场景，实际应用中这类情况很常见。
4. **重建分支的必要性讨论不足**：推理时重建分支是否可以去掉以加速？论文未明确说明推理阶段的架构简化策略。
5. **融合损失直接沿用 SwinFusion**：未对融合损失函数本身做创新，可能存在进一步优化空间。

## 相关工作与启发

- **TC-MoA** [Zhu et al.]：基于任务特定路由网络的通用融合方法，是本文最强 baseline；UniFusion 通过更强的语义先验和双层优化策略超越之。
- **SwinFusion** [Ma et al.]：跨域 Swin Transformer 框架，UniFusion 沿用其融合损失设计并在此基础上大幅提升。
- **U2Fusion** [Xu et al.]：先驱性的 all-in-one 融合方法，启发了后续统一框架研究。
- **DINOv2/v3**：自监督 ViT 预训练范式，本文验证了其在低级视觉任务中的迁移潜力。
- **Bilevel Optimization in Vision**：在 meta-learning、NAS、超参优化中有广泛应用，本文将其引入图像融合是有意义的尝试。
- **Adapter-tuning 的启发**：冻结大模型 + 轻量适配器的范式在 NLP 已成熟，本文在 CV 低级任务上的成功应用值得关注。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — DINOv3 + Adapter 作为通用融合 backbone 的思路新颖，重建对齐机制设计巧妙，双层优化的引入有理论支撑；但各组件（Adapter-tuning、bilevel opt）本身并非全新，贡献在于有效组合。
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖 IVIF/MIF/MEF/MFF 四大类任务共 6+ 个 benchmark，与 10 个 SOTA 方法对比，消融实验完整（4 个变体），定性可视化丰富（特征图、融合结果），并在附录中提供下游任务验证。
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，方法描述逻辑性强，公式推导完整；图表质量高，Fig. 8 的特征可视化很有说服力；但部分符号定义可以更早引入。
- **价值**: ⭐⭐⭐⭐ — 提供了一个实用的统一融合框架，单模型跨任务泛化具有工程价值；DINOv3 + Adapter 范式可推广到其他低级视觉任务；代码开源进一步增加可复现性和影响力。

<!-- RELATED:START -->

## 相关论文

- [AutoOpt: A Dataset and a Unified Framework for Automating Optimization Problem Solving](../../NeurIPS2025/optimization/autoopt_a_dataset_and_a_unified_framework_for_automating_optimization_problem_so.md)
- [SMoFi: Step-wise Momentum Fusion for Split Federated Learning on Heterogeneous Data](../../AAAI2026/optimization/smofi_step-wise_momentum_fusion_for_split_federated_learning_on_heterogeneous_da.md)
- [Efficient Curvature-Aware Hypergradient Approximation for Bilevel Optimization](../../ICML2025/optimization/efficient_curvature-aware_hypergradient_approximation_for_bilevel_optimization.md)
- [Random Feature Representation Boosting](../../ICML2025/optimization/random_feature_representation_boosting.md)
- [Tilted Sharpness-Aware Minimization](../../ICML2025/optimization/tilted_sharpness-aware_minimization.md)

<!-- RELATED:END -->
