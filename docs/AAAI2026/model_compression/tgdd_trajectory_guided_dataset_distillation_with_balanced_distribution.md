---
title: >-
  [论文解读] TGDD: Trajectory Guided Dataset Distillation with Balanced Distribution
description: >-
  [AAAI 2026][模型压缩][数据集蒸馏] 提出 TGDD，将静态分布匹配重新定义为沿训练轨迹的动态对齐过程，通过阶段式分布匹配（Stage-wise Distribution Matching）捕获演化语义 + 分布约束正则化（Stage-wise Distribution Constraint）减少类间重叠，在 10 个数据集上达到 SOTA，高分辨率基准上准确率提升 5.0%。
tags:
  - AAAI 2026
  - 模型压缩
  - 数据集蒸馏
  - 分布匹配
  - 专家轨迹
  - 分布约束
  - 合成数据
---

# TGDD: Trajectory Guided Dataset Distillation with Balanced Distribution

**会议**: AAAI 2026  
**arXiv**: [2512.02469](https://arxiv.org/abs/2512.02469)  
**代码**: [github.com/FlyFinley/TGDD](https://github.com/FlyFinley/TGDD)  
**领域**: 模型压缩  
**关键词**: 数据集蒸馏, 分布匹配, 专家轨迹, 分布约束, 合成数据

## 一句话总结

提出 TGDD，将静态分布匹配重新定义为沿训练轨迹的动态对齐过程，通过阶段式分布匹配（Stage-wise Distribution Matching）捕获演化语义 + 分布约束正则化（Stage-wise Distribution Constraint）减少类间重叠，在 10 个数据集上达到 SOTA，高分辨率基准上准确率提升 5.0%。

## 研究背景与动机

数据集蒸馏旨在将大规模数据集压缩为紧凑的合成数据集，同时尽可能保留原始数据集的训练效果。现有方法分为两大类：

**优化导向（OO-based）方法**（如 MTT、FTD）：通过梯度匹配或轨迹匹配在整个训练过程中对齐合成数据与原始数据的学习动态。性能较好，但需要网络和数据的双层优化，**计算成本极高**，难以扩展。

**分布匹配（DM-based）方法**（如 DM、IDM、M3D）：直接对齐合成数据和原始数据在嵌入空间中的特征分布。效率高，但存在两个关键缺陷：
   - **忽略表征演化**：大多使用随机初始化的网络作为特征提取器，只捕获训练早期的特征分布，无法反映模型训练过程中从低级到高级语义的演化
   - **类间分离度差**：MMD 仅约束分布均值，对类边界的区分力不足，导致合成数据的类间特征高度重叠

作者通过 t-SNE 可视化直观展示了这一问题：DM 方法生成的合成数据在不同训练阶段的模型下类分离度差异极大——只有高度优化的模型才能区分类别。这使得合成数据难以学习，下游性能受限。

**核心动机**：既然专家轨迹（训练过程中保存的模型快照）已经包含了训练各阶段的特征表示，何不直接利用它们来做分布匹配？这样既保留了 DM 方法的效率，又引入了 OO 方法中对训练动态的感知。

## 方法详解

### 整体框架

TGDD 分三步：
1. **预训练专家轨迹**：用原始数据训练 N 条专家轨迹，每条包含 M 个模型快照（这一步只需做一次，后续可复用）
2. **阶段式分布匹配**：从专家轨迹中随机采样模型快照作为特征提取器，在不同训练阶段对齐合成数据和原始数据的特征分布
3. **阶段式分布约束**：从同一轨迹的"专家区域"中采样另一个模型，对合成数据施加分类约束以强化类间分离

### 关键设计

1. **专家轨迹构建（Expert Trajectory Construction）**

   训练 N 个随机初始化的神经网络直到收敛（M 个 epoch），保存所有中间快照：
   $$\mathbf{P} = \{p_{i,j} \mid 0 \leq i \leq N, \ 0 \leq j \leq M\}$$

   与 MTT 需要 200 条轨迹不同，TGDD 仅需 **5 条轨迹**即可达到竞争性能（图 6(c) 实验验证）。关键优势是轨迹的预训练只使用原始数据，**无需在蒸馏过程中训练网络**，因此可以提前预训练并复用。

2. **阶段式分布匹配（Stage-wise Distribution Matching）**

   每次蒸馏迭代中，随机选择一条轨迹 $\mathbf{P}_i$，再随机选择一个训练阶段的模型快照 $\theta_{ext} = p_{i,j}$ 作为特征提取器。使用 MMD 对齐每个类别的合成数据和原始数据的特征分布：

   $$L_{MMD} = \sum_{c=1}^{C} \left\| \frac{1}{|B^T_c|} \sum_{i=1}^{|B^T_c|} \psi_{\theta_{ext}}(x_i) - \frac{1}{|B^S_c|} \sum_{i=1}^{|B^S_c|} \psi_{\theta_{ext}}(s_i) \right\|^2$$

   通过在不同训练阶段采样特征提取器，合成数据被迫在**所有阶段**都与原始数据对齐，从而丰富了语义多样性。

3. **阶段式分布约束（Stage-wise Distribution Constraint）**

   仅用 MMD 对齐均值不足以保证类间分离。给定特征提取器 $\theta_{ext} = p_{i,j}$，从"专家区域" $P_{er} = \{p_{i,j}, ..., p_{i,j+L-1}\}$ 中随机选择一个专家模型 $\theta_{exp}$，对合成数据施加分类损失：

   $$L_{SDC} = \frac{1}{B^S_c} \sum_{c=1}^{C} \sum_{i=1}^{|B^S_c|} l(\phi_{exp}(s_i), y_i)$$

   这个损失直接鼓励合成数据在专家模型眼中具有**高类内紧凑度**，从而改善类间分离。使用不同迭代的不同专家实现了**集成效果**而无需额外训练成本。

### 损失函数 / 训练策略

总损失为分布匹配和分布约束的加权组合：
$$L_{overall} = L_{MMD} + \alpha L_{SDC}$$

- $\alpha$ 在 IPC=1,10 时设为 2.5，IPC=50 时设为 0.5
- 专家区域距离 $L = 7$
- 学习率：ImageNet 子集 0.1，其他 0.01
- 5 条专家轨迹，60 epochs（低分辨率）/ 80 epochs（高分辨率）
- 使用差异增强（颜色变换、随机裁剪、Cutout 等）和多形态参数化

## 实验关键数据

### 主实验

**低/中分辨率数据集（ConvNet-3，CIFAR-10/100, TinyImageNet）：**

| 方法 | 类型 | CIFAR-10 IPC10 | CIFAR-10 IPC50 | CIFAR-100 IPC10 | CIFAR-100 IPC50 | TinyIm IPC10 | TinyIm IPC50 |
|------|------|------|------|------|------|------|------|
| DM | DM | 48.9 | 63.0 | 29.7 | 43.6 | 12.9 | 24.1 |
| MTT | OO | 65.3 | 71.6 | 40.1 | 47.7 | 23.2 | 28.0 |
| FTD | OO | 66.6 | 73.8 | 43.4 | 50.7 | 24.5 | — |
| M3D | DM | 63.5 | 69.9 | 42.4 | 50.9 | — | — |
| DANCE | DM | 70.8 | 76.1 | 49.8 | 52.8 | 26.4 | 28.9 |
| **TGDD** | **DM** | **71.9** | **76.5** | **51.3** | **54.6** | **29.3** | **30.9** |

**高分辨率数据集（ImageNet 子集，IPC=10）：**

| 方法 | ImageNette | ImageWoof | ImageFruit | ImageMeow | ImageSquawk | ImageYellow |
|------|-----------|-----------|-----------|-----------|------------|------------|
| MTT | 63.0 | 35.8 | 40.3 | 40.4 | 52.3 | 60.0 |
| FTD | 67.7 | 38.8 | 44.9 | 43.3 | — | — |
| DANCE | 80.2 | 57.8 | 52.8 | 60.4 | 77.2 | 78.8 |
| **TGDD** | **82.0** | **58.4** | **57.8** | **62.8** | **78.0** | **76.6** |

### 消融实验

| 增强 Aug | $L_{MMD}$ | $L_{SDC}$ | CIFAR-10 IPC10 | CIFAR-10 IPC50 | CIFAR-100 IPC10 | CIFAR-100 IPC50 |
|---------|-----------|-----------|--------|--------|---------|---------|
| ✗ | ✗ | ✗ | 55.2 | 65.3 | 33.7 | 44.5 |
| ✓ | ✗ | ✗ | 63.2 | 69.5 | 40.5 | 47.2 |
| ✓ | ✓ | ✗ | 65.8 | 75.2 | 47.0 | 53.0 |
| ✓ | ✓ | ✓ | **71.9** | **76.5** | **51.3** | **54.6** |

**跨架构泛化（CIFAR-10, IPC50, ConvNet-3 蒸馏 → 其他架构评估）：**

| 方法 | ConvNet-3 | ResNet-10 | DenseNet-121 |
|------|----------|----------|-------------|
| DM | 63.0 | 58.6 | 57.4 |
| DANCE | 76.1 | 68.0 | 64.8 |
| **TGDD** | **76.5** | **74.9** | **74.3** |

### 关键发现

- **阶段式分布匹配是性能提升的主要来源**：从消融表看，加入 $L_{MMD}$ 后 CIFAR-10 IPC50 从 69.5→75.2（+5.7%）
- **分布约束进一步提升类分离度**：加入 $L_{SDC}$ 后 CIFAR-100 IPC10 从 47.0→51.3（+4.3%）
- **跨架构泛化能力出色**：TGDD 在 ResNet-10 和 DenseNet-121 上的性能保持率远高于 DANCE（分别 74.9 vs 68.0, 74.3 vs 64.8）
- **高分辨率数据集优势显著**：ImageFruit IPC10 上比 DANCE 高 5.0%
- 性能对 $\alpha$ 中等敏感（2.8% 波动范围），对轨迹数量和长度鲁棒
- 仅 1 条轨迹即可达到接近 5 条的性能，远低于 MTT 的 200 条

## 亮点与洞察

- **巧妙融合两种范式的优势**：用 OO 方法的轨迹信息增强 DM 方法的特征提取，保持了 DM 的效率同时获得了 OO 的表达力
- **专家区域的设计精巧**：特征匹配和分布约束用同一轨迹上的不同快照，自然地实现了不同训练阶段的知识注入
- **极低的存储需求**：仅需 5 条轨迹（vs MTT 的 200 条），且可预训练复用
- **性能-效率帕累托最优**（图 1）：在相同 GPU 内存和蒸馏时间约束下达到最高精度

## 局限与展望

- 仅在 ConvNet 架构上蒸馏，未探索使用 ViT 或 ResNet 作为蒸馏架构
- 跨架构泛化虽优于基线但仍有下降空间，尤其是 ConvNet→DenseNet 的差距
- 高分辨率数据集整体性能仍然与全数据集训练差距较大（62.8% vs 66.7%）
- 仅支持分类任务，未探索检测、分割等下游任务的数据蒸馏
- 分布约束使用的是标准交叉熵损失，可以考虑更精细的度量（如对比学习损失）
- 持续学习实验中步长选择较小（5 和 10），更长期的遗忘效应有待研究

## 相关工作与启发

- **MTT / FTD**：轨迹匹配方法的代表，虽然性能好但计算开销巨大；TGDD 借鉴了轨迹的概念但仅用于特征提取而非梯度匹配
- **DM / M3D / DANCE**：DM 系列的进化路径，TGDD 是这条路线的自然延伸，解决了静态特征提取的根本问题
- **IDM** 使用动态模型队列来获取更丰富的表征，但引入了额外训练成本；TGDD 用预训练轨迹实现了相同效果且不增加成本
- 持续学习实验开拓了数据蒸馏的应用场景

## 评分

- 新颖性: ⭐⭐⭐⭐ — 轨迹引导分布匹配的思路自然但有效
- 实验充分度: ⭐⭐⭐⭐⭐ — 10 个数据集、跨架构、消融、超参分析、持续学习
- 写作质量: ⭐⭐⭐⭐ — 表述清晰、可视化丰富
- 价值: ⭐⭐⭐⭐ — DM 系列的重要改进，兼具性能和效率

<!-- RELATED:START -->

## 相关论文

- [DP-GenG: Differentially Private Dataset Distillation Guided by DP-Generated Data](dp-geng_differentially_private_dataset_distillation_guided_by_dp-generated_data.md)
- [Rethinking Long-tailed Dataset Distillation: A Uni-Level Framework with Unbiased Recovery and Relabeling](rethinking_long-tailed_dataset_distillation_a_uni-level_framework_with_unbiased_.md)
- [Dataset Distillation via the Wasserstein Metric](../../ICCV2025/model_compression/dataset_distillation_via_the_wasserstein_metric.md)
- [Dataset Distillation as Pushforward Optimal Quantization](../../ICLR2026/model_compression/dataset_distillation_as_pushforward_optimal_quantization.md)
- [Rectified Decoupled Dataset Distillation: A Closer Look for Fair and Comprehensive Evaluation](../../ICLR2026/model_compression/rectified_decoupled_dataset_distillation_a_closer_look_for_fair_and_comprehensiv.md)

<!-- RELATED:END -->
