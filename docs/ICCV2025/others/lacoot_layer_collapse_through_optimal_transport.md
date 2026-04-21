---
title: >-
  [论文解读] LaCoOT: Layer Collapse through Optimal Transport
description: >-
  [ICCV2025][optimal transport] 提出 LaCoOT，一种基于最优传输的正则化策略，通过最小化网络内部中间特征分布之间的 Max-Sliced Wasserstein 距离，使得训练后可以直接移除整个网络层，在保持性能的同时显著减少模型深度和推理时间。
tags:
  - ICCV2025
  - optimal transport
  - layer removal
  - depth reduction
  - Max-Sliced Wasserstein distance
  - 模型压缩
---

# LaCoOT: Layer Collapse through Optimal Transport

**会议**: ICCV2025  
**arXiv**: [2406.08933](https://arxiv.org/abs/2406.08933)  
**代码**: [VGCQ/LaCoOT](https://github.com/VGCQ/LaCoOT)  
**领域**: 模型压缩 / 网络深度缩减 / 最优传输  
**关键词**: optimal transport, layer removal, depth reduction, Max-Sliced Wasserstein distance, model compression

## 一句话总结

提出 LaCoOT，一种基于最优传输的正则化策略，通过最小化网络内部中间特征分布之间的 Max-Sliced Wasserstein 距离，使得训练后可以直接移除整个网络层，在保持性能的同时显著减少模型深度和推理时间。

## 研究背景与动机

### 基础模型的计算挑战

近年来基础模型（CLIP、Stable Diffusion、DiT 等）的兴起带来了巨大的计算开销。训练一个生成模型相当于驾车行驶 10km 的碳排放，生成 10k 样本则相当于 160km。虽然训练成本高昂，但模型开源后无数用户的推理成本更是成倍增长。

### 现有压缩方法的局限

主流的复杂度缩减方法包括：
- **非结构化剪枝**：在通用硬件上缺乏实际加速效果
- **结构化剪枝**（通道/滤波器剪枝）：在现代并行计算架构上边际效果有限，真正的瓶颈是**计算关键路径长度**（critical path length）
- **知识蒸馏到浅层学生模型**：目标架构未知，可能造成性能损失

### 网络深度缩减的困难

现有的深度缩减方法主要通过移除非线性激活函数来合并相邻线性层：
- **Layer Folding**：将 ReLU 替换为 PReLU 来评估是否可丢弃非线性
- **EGP**：基于熵度量优先剪枝非线性利用率低的层
- **NEPENTHE**：改进了 EGP 的熵估计器
- **EASIER**：通过验证集评估移除非线性的影响

然而这些方法存在根本性问题：
1. ResNet 架构中第二个卷积层有 padding 时，没有解析解可以合并两个相邻卷积层
2. 残差连接处移除激活函数后无法进行层融合
3. 依赖迭代搜索，计算开销大（EASIER 需 34 次训练达到一个目标）

## 方法详解

### 核心思路

LaCoOT 不依赖线性化激活和层合并，而是直接在训练中最小化网络各模块（block）输入输出特征分布的差异。训练后，差异最小的模块可以被完整移除——因为它近似恒等映射。

### 正则化策略

将 DNN 表示为模块的级联，每个模块有输入特征分布和输出特征分布。

**OT 正则项**：对所有模块的 Max-Sliced Wasserstein 距离取均值，作为训练时的正则化项。Max-Sliced Wasserstein 距离是 2-Wasserstein 距离的最大投影变体，具有封闭形式解，免于维度灾难。

**总体训练目标**：任务损失 + 正则化系数 * OT 正则项。正则化系数控制强度：越大则更多层可被移除但可能影响性能。

### 层移除流程

1. 用正则化训练网络，评估性能
2. 计算每个模块的 Max-Sliced Wasserstein 距离
3. 找出距离最小的模块，将其替换为 Identity
4. 重新评估，若性能下降未超阈值则继续移除下一个
5. 直到性能下降超阈值停止

### 理论分析

**软 1-Lipschitz 约束**：正则化等价于对模块 Jacobian 施加正交约束，实现逐块的软 1-Lipschitz 约束。文献表明 1-Lipschitz 约束不限制网络的分类表达能力，反而有助于泛化。

**驻点分析**：损失和正则项互为对抗——未受约束的 DNN 倾向于在中间层引入不必要的分布变化。正则化引导网络走"最短路径"，消除冗余的分布变化。

**三角不等式下界**：输入分布与真值分布的距离定义了正则化值的紧下界，保证网络保持最小必要的分布变换能力不至于欠拟合。

## 实验结果

### 实验设置

- **分类模型**：ResNet-18、MobileNetV2、Swin-T
- **分类数据集**：CIFAR-10、Tiny-ImageNet-200、PACS、VLCS、Flowers-102、DTD、Aircraft
- **生成模型**：DiT-XL/2（ImageNet 微调）
- **基线**：Layer Folding、EGP、NEPENTHE、EASIER

### 关键路径长度与性能

在 CIFAR-10 + ResNet-18 上：

| 方法 | Top-1 Acc. | MACs (M) | 推理时间 (ms) | 训练时间 |
|------|-----------|----------|-------------|---------|
| Original | 91.77% | 140.19 | 7.90 | 30 min |
| Layer Folding | 88.76% | 147.53 | 9.89 | 160 min |
| EGP | 90.64% | 140.19 | 7.62 | 376 min |
| NEPENTHE | 89.26% | 140.19 | 7.71 | 288 min |
| EASIER | 90.35% | 140.19 | 7.07 | 533 min |
| **LaCoOT** | **90.99%** | **64.69** | **4.78** | **40 min** |

关键发现：

1. **MACs 直接减半**：LaCoOT 将 MACs 从 140M 降至 65M（-54%），而基线方法因无法合并层，MACs 几乎不变
2. **推理速度提升 40%**：4.78ms vs 7.90ms
3. **训练效率最高**：40 min vs EASIER 的 533 min（13x 更快）
4. **性能损失最小**：90.99% vs 原始 91.77%，仅损失 0.78%

### 生成模型：DiT-XL/2

在 ImageNet 上微调 DiT-XL/2 仅 5k 步：

- 移除 2 个 DiT block 时，LaCoOT 的 FID-50k 为 56.2，而无正则化的为 118.6（LaCoOT 将 FID 降低一半）
- 生成图像质量明显保留更好——无正则化时内容完全崩坏

### 关键发现（跨架构）

- Swin-T on Tiny-ImageNet-200：LaCoOT 在相同关键路径长度下比基线高 10%
- MobileNetV2：在已高度优化的架构上，EASIER 表现略好，但需 20x 训练时间
- EGP 在 MobileNetv2 上完全失败：首轮就剪掉了分类头前的最后一层，切断信息流

### 消融实验

- **无正则化时**：Max-Sliced Wasserstein 距离不能作为可靠的模块重要性指标，甚至不如随机移除
- **正则化系数 = 5**：可移除近半数模块且几乎无性能损失
- LaCoOT 的指标优于 Block Influence（逐个尝试移除）和 Random

## 优势与局限

### 优势

- **直接移除整层**而非仅移除非线性，真正减少关键路径长度
- **模型无关**：适用于 ResNet、Swin、MobileNet、DiT 等架构
- **只需一次训练**：相比 EASIER 等迭代方法高效一个数量级
- **可扩展至基础模型**：仅需少量微调步即可应用于 DiT

### 局限

- 对已欠拟合的高效架构（如 MobileNetV2 on Tiny-ImageNet）效果有限
- 仅应用于输入输出维度相同的模块，跨维度模块需未来探索 Gromov-Wasserstein 距离
- 移除层后不做重训练；加入 healing 阶段可进一步恢复性能

## 个人思考

1. 用最优传输量化层的"冗余程度"非常自然——如果一层的输入输出分布几乎相同，那它就是恒等映射
2. 与结构化剪枝（减少宽度）互补：LaCoOT 减少深度，两者配合可实现更高效的压缩
3. 正则化的副作用可能值得研究——1-Lipschitz 约束是否影响模型在分布外数据上的鲁棒性
4. 对生成模型的初步结果（DiT）非常有前景，值得在更大规模上验证

<!-- RELATED:START -->

## 相关论文

- [Hierarchical Refinement: Optimal Transport to Infinity and Beyond](../../ICML2025/others/hierarchical_refinement_optimal_transport_to_infinity_and_beyond.md)
- [Lightspeed Geometric Dataset Distance via Sliced Optimal Transport](../../ICML2025/others/lightspeed_geometric_dataset_distance_via_sliced_optimal_transport.md)
- [Quantifying Lexical Semantic Shift via Unbalanced Optimal Transport](../../ACL2025/others/quantifying_lexical_semantic_shift_via_unbalanced_optimal_transport.md)
- [Theoretical Performance Guarantees for Partial Domain Adaptation via Partial Optimal Transport](../../ICML2025/others/theoretical_performance_guarantees_for_partial_domain_adaptation_via_partial_opt.md)
- [Variational Regularized Unbalanced Optimal Transport: Single Network, Least Action](../../NeurIPS2025/others/variational_regularized_unbalanced_optimal_transport_single_network_least_action.md)

<!-- RELATED:END -->
