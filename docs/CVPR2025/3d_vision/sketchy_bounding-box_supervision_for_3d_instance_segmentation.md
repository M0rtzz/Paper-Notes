---
title: >-
  [论文解读] Sketchy Bounding-Box Supervision for 3D Instance Segmentation
description: >-
  [CVPR 2025][3D视觉][弱监督3D实例分割] 提出 Sketchy-3DIS 框架，首次将不精确（sketchy）的3D包围盒标注引入弱监督3D实例分割，通过自适应 box-to-point 伪标签生成器和粗到精实例分割器的联合训练，在 ScanNetV2 和 S3DIS 上达到 SOTA，甚至超越部分全监督方法。
tags:
  - CVPR 2025
  - 3D视觉
  - 弱监督3D实例分割
  - 不精确包围盒
  - 伪标签
  - 粗到精分割
  - 点云理解
---

# Sketchy Bounding-Box Supervision for 3D Instance Segmentation

**会议**: CVPR 2025  
**arXiv**: [2505.16399](https://arxiv.org/abs/2505.16399)  
**代码**: https://github.com/dengq7/Sketchy-3DIS  
**领域**: 3D视觉  
**关键词**: 弱监督3D实例分割, 不精确包围盒, 伪标签, 粗到精分割, 点云理解

## 一句话总结

提出 Sketchy-3DIS 框架，首次将不精确（sketchy）的3D包围盒标注引入弱监督3D实例分割，通过自适应 box-to-point 伪标签生成器和粗到精实例分割器的联合训练，在 ScanNetV2 和 S3DIS 上达到 SOTA，甚至超越部分全监督方法。

## 研究背景与动机

**领域现状**：3D实例分割是点云场景理解的核心任务，当前主流方法（如 SPFormer、Mask3D）依赖逐点标注，但在 ScanNet 上标注一个场景需要约 22.3 分钟，而标注包围盒只需 1.93 分钟。因此近年来大量工作采用包围盒作为弱监督信号。

**现有痛点**：现有的包围盒弱监督方法（Box2Mask、GaPro、BSNet 等）都假设标注的包围盒是精确且紧凑的。但在实际标注中，获得完全精确的3D包围盒非常困难——标注者通常会引入缩放偏差、平移偏差和旋转偏差。实验表明 GaPro 在使用 scaled sketchy box 时性能大幅下降。

**核心矛盾**：实际场景中包围盒不可能完全精确，但现有方法对包围盒精度高度敏感。不精确包围盒会导致大量点被错误分配到邻近实例，产生噪声伪标签，进而严重影响分割质量。

**本文目标**：设计一个对包围盒噪声具有鲁棒性的弱监督3D实例分割框架，能够(1)从不精确包围盒中生成高质量伪标签，(2)基于伪标签训练出高精度分割器。

**切入角度**：作者观察到不精确包围盒的核心问题在于重叠区域的点分配。如果能学习点与包围盒之间的相似性，就可以自适应地将重叠区域中的点分配到正确的实例。

**核心 idea**：联合训练一个自适应伪标签生成器（将 sketchy box 转化为紧凑 box 并生成点级伪标签）和一个粗到精实例分割器（通过多级注意力逐步细化实例），两者互相促进，逐步提升分割质量。

## 方法详解

### 整体框架

输入为一个3D点云场景及其对应的一组 sketchy 包围盒标注。首先通过 3D U-Net 骨干网络提取点特征，然后进入两个并行分支：(1) 自适应 box-to-point 伪标签生成器，将粗糙的包围盒标注转化为精细的点级实例标签；(2) 粗到精实例分割器，利用 Transformer 查询机制预测实例。最终通过双边匹配将伪标签与预测实例配对，联合训练两个组件。推理时只需要骨干网络和分割器。

### 关键设计

1. **Sketchy Bounding Box 生成**:

    - 功能：模拟实际标注中不精确的包围盒
    - 核心思路：对真实包围盒施加三种扰动——缩放（$\alpha=5\%$）、平移（$\beta=5\%$）、旋转（$\gamma=5°$），通过组合这三种基础操作生成四种不同"sketchy程度"的包围盒（$S_1$到$S_4$）。扰动幅度的设置基于实际标注偏差的合理范围。
    - 设计动机：提供一个可控的实验框架来研究标注噪声对弱监督方法的影响

2. **自适应 Box-to-Point 伪标签生成器**:

    - 功能：将不精确的包围盒标注转化为高质量的点级实例伪标签
    - 核心思路：分三步处理不同类型的点。(a) 包围盒外的点直接标记为背景。(b) 对于仅位于单个包围盒内的点，通过特征空间余弦相似度与坐标空间距离的乘积 $s_{p,B} = \cos(f_p, f_B) \times e^{-|c_B c_p|}$ 来过滤背景点。(c) 对于位于多个包围盒重叠区域的点，先移除重叠部分取可靠点（只在一个 box 内的点），用这些可靠点的特征代替 box 特征，然后通过 MLP 学习点到各 box 的分配概率，用交叉熵损失 $L_{pl}$ 监督。
    - 设计动机：不精确包围盒最大的问题是重叠区域扩大，导致更多点被错误分配。通过学习相似性而非硬规则，可以自适应地处理不同程度的不精确标注。

3. **粗到精实例分割器（Multi-level Attention Block）**:

    - 功能：从全局到局部逐步细化实例预测
    - 核心思路：实例查询先与全场景点特征做全局交叉注意力得到粗实例，再分别与可靠区域特征（$F^{rel}$，由预测 box 与 mask box 的交集加权）和核心区域特征（$F^{un}$，由缩放后的核心 box 与 mask box 交集）做局部注意力。六层堆叠的 Multi-level Attention Block 逐步精细化。可靠区域由预测框和 mask 框的 IoU 加权：$F^{rel} = \sigma(F, M \odot e^{IoU(B_{pred}, B_{mask})})$。
    - 设计动机：弱监督中粗标签导致初始预测不精确，通过分层次的从全局到核心区域的注意力机制，让查询逐步聚焦到目标核心区域，提升边界精度。

### 损失函数 / 训练策略

总损失 $L = L_{pl} + L_{seg}$，其中伪标签损失 $L_{pl}$ 是可靠点分配的交叉熵损失，实例分割损失 $L_{seg}$ 包含分类损失（交叉熵）、mask 损失（BCE + Dice）和 box 损失（L1 + core-box MSE）。使用匈牙利匹配建立伪标签与预测实例的对应关系。训练采用 AdamW 优化器，学习率 0.0002，权重衰减 0.05，在单个 RTX 3090 上训练。联合训练策略让伪标签质量和分割器性能互相提升。

## 实验关键数据

### 主实验

| 数据集 | 方法 | 监督 | AP50 | AP25 |
|--------|------|------|------|------|
| ScanNetV2 Val | GaPro+SPFormer | 精确box S0 | 70.4 | 79.9 |
| ScanNetV2 Val | BSNet+SPFormer | 精确box S0 | 72.7 | 83.4 |
| ScanNetV2 Val | **Sketchy-3DIS** | **精确box S0** | **68.8** | **83.6** |
| ScanNetV2 Val | GaPro+SPFormer | Sketchy S1 | 53.5 | 72.2 |
| ScanNetV2 Val | **Sketchy-3DIS** | **Sketchy S1** | **65.8** | **83.1** |
| S3DIS Area 5 | GaPro+ISBNet | 精确box S0 | 61.2 | - |
| S3DIS Area 5 | BSNet+ISBNet | 精确box S0 | 64.3 | - |
| S3DIS Area 5 | **Sketchy-3DIS** | **精确box S0** | **69.1** | - |
| S3DIS Area 5 | ISBNet (全监督) | Mask | 65.8 | - |

### 消融实验

| 配置 | AP | AP50 | AP25 | 说明 |
|------|-----|------|------|------|
| 无伪标签生成器 (Partition only) | 15.9 | 32.2 | 58.5 | 仅空间分配，大量错误标签 |
| +Assign (重叠区域分配) | 41.8 | 64.8 | 72.3 | 重叠区域处理至关重要 |
| +Similarity (背景过滤) | 45.2 | 67.3 | 83.4 | 过滤非目标点进一步提升 |
| Full model | 46.0 | 68.8 | 83.6 | 完整模型 |
| Disjoint训练 (非联合) | 45.3 | 60.4 | 70.0 | 联合训练显著优于分开训练 |
| Joint训练 | 53.4 | 69.1 | 77.5 | 伪标签+分割器互相促进 |

### 关键发现

- **重叠区域点分配**是性能提升最关键的模块，从 32.2 提升到 64.8（AP50 提升 32.6%），因为 sketchy box 导致的重叠区域被极大扩大
- 在 S3DIS 上，Sketchy-3DIS 即使使用不精确标注（S0）也超过全监督 ISBNet +3.3 AP50（69.1 vs 65.8），说明精心设计的弱监督方法可以超越全监督
- 从 S1 到 S4 不同程度的 sketchy box，AP50 仅从 65.8 下降到 62.5（ScanNetV2），展现了良好的鲁棒性
- 联合训练（joint）vs 分开训练（disjoint）差距巨大：AP50 69.1 vs 60.4，验证了伪标签生成和分割器必须协同优化
- Multi-level Attention Block 中全部四种注意力（Scene/Coarse/Core/Self）都有贡献，其中 Coarse 和 Core 的组合最关键

## 亮点与洞察

- **首次研究 sketchy box 设定**：之前所有 box-supervised 方法都假设 box 精确，本文首次系统性研究了不精确标注的影响，更贴近实际场景。这个问题设定本身就有很大价值。
- **联合训练的互促进机制**：伪标签生成器和分割器共用骨干特征，通过联合优化实现良性循环——更好的特征产生更好的伪标签，更好的伪标签训练更好的分割器。
- **可靠点 vs 不可靠点的分离策略**：在重叠区域只用可靠点（非重叠部分）来学习分配规则，然后推广到不可靠点，避免了噪声标签的影响。这个思路可以迁移到其他弱监督任务中。

## 局限与展望

- 模型对于不同物体大小变化很大的场景可能效果不佳，因为 sketchy box 的扰动比例是全局统一的
- 只研究了模拟的 sketchy box，没有在真实标注者产生的不精确标注上验证
- 推理速度和计算开销没有报告，联合训练两个分支可能导致训练成本较高
- 可以考虑引入 SAM 等 2D 基础模型辅助处理重叠区域的歧义
- 扩展到 outdoor 场景（如自动驾驶）的大规模点云是一个有价值的方向

## 相关工作与启发

- **vs GaPro**: GaPro 用高斯过程生成可靠伪标签，但依赖精确 box，在 S1 sketchy box 下 AP50 从 70.4 暴跌到 53.5。Sketchy-3DIS 在 S1 下仍有 65.8
- **vs BSNet**: BSNet 通过 mean teacher 和场景合成增强数据，在精确 box 下更强（72.7 vs 68.8 AP50），但未验证不精确标注下的鲁棒性
- **vs Box2Mask**: Box2Mask 作为首个 box-supervised 方法，方法较简单，在 sketchy box 下 AP50 只有 52.4

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次研究 sketchy box 设定，问题定义有价值，但具体方法是已有技术的组合
- 实验充分度: ⭐⭐⭐⭐⭐ 两个数据集、多种 sketchy 程度、详细消融、可视化分析
- 写作质量: ⭐⭐⭐⭐ 动机清晰，框架图清楚，但部分细节描述可以更简洁
- 价值: ⭐⭐⭐⭐ 实用性强，不精确标注是真实需求，方法有效且代码开源

<!-- RELATED:START -->

## 相关论文

- [Any3DIS: Class-Agnostic 3D Instance Segmentation by 2D Mask Tracking](any3dis_class-agnostic_3d_instance_segmentation_by_2d_mask_tracking.md)
- [Relation3D: Enhancing Relation Modeling for Point Cloud Instance Segmentation](relation3d_enhancing_relation_modeling_for_point_cloud_instance_segmentation.md)
- [Retrieving Objects from 3D Scenes with Box-Guided Open-Vocabulary Instance Segmentation](../../AAAI2026/3d_vision/retrieving_objects_from_3d_scenes_with_box-guided_open-vocabulary_instance_segme.md)
- [CutS3D: Cutting Semantics in 3D for 2D Unsupervised Instance Segmentation](../../ICCV2025/3d_vision/cuts3d_cutting_semantics_in_3d_for_2d_unsupervised_instance_segmentation.md)
- [Part2Object: Hierarchical Unsupervised 3D Instance Segmentation](../../ECCV2024/3d_vision/part2object_hierarchical_unsupervised_3d_instance_segmentation.md)

<!-- RELATED:END -->
