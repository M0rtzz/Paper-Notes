---
title: >-
  [论文解读] Deep Nets with Subsampling Layers Unwittingly Discard Useful Activations at Test-Time
description: >-
  [ECCV 2024][图像分割][测试时增强] 发现深度网络中下采样层在默认前向传播中丢弃了大量有用激活，提出一个搜索+聚合框架在测试时利用这些被丢弃的激活图来提升分类和分割性能，与传统TTA方法正交互补。
tags:
  - ECCV 2024
  - 图像分割
  - 测试时增强
  - 下采样层
  - 激活图搜索
  - 注意力聚合
  - 语义分割
---

# Deep Nets with Subsampling Layers Unwittingly Discard Useful Activations at Test-Time

**会议**: ECCV 2024  
**arXiv**: [2410.01083](https://arxiv.org/abs/2410.01083)  
**代码**: https://github.com/ca-joe-yang/discard-in-subsampling  
**领域**: 图像分割 / 图像分类  
**关键词**: 测试时增强, 下采样层, 激活图搜索, 注意力聚合, 语义分割

## 一句话总结

发现深度网络中下采样层在默认前向传播中丢弃了大量有用激活，提出一个搜索+聚合框架在测试时利用这些被丢弃的激活图来提升分类和分割性能，与传统TTA方法正交互补。

## 研究背景与动机

**领域现状**：测试时增强（Test-Time Augmentation, TTA）是提升模型性能的常用手段，通过对输入图像应用多种增广（随机裁剪、翻转、旋转等）并聚合预测结果。然而，现有TTA方法全部在**图像空间**操作，计算开销巨大（如ImageNet上需144次前向传播）。

**核心观察与痛点**：几乎所有深度网络都包含下采样层（stride convolution、pooling等），这些层在降低空间维度时会丢弃大部分激活。例如stride=2的2D卷积会丢弃 $\frac{3}{4}$ 的空间激活。默认实现总是选择 $s=0$（偶数索引），而 $s=1$（奇数索引）的激活完全被抛弃。这些被丢弃的激活包含了关于输入图像的有用信息。

**核心矛盾**：(a) 如何判断哪些被丢弃的激活是有用的？(b) 如何高效聚合这些激活？暴力枚举所有可能的选择索引组合是指数级的。

**核心idea**：将选择不同下采样索引视为一种"特征空间的增广"，通过贪心搜索找到有用的激活图，用注意力机制聚合它们，实现与图像空间TTA正交的测试时性能提升。

## 方法详解

### 整体框架

给定一个包含 $L$ 个下采样层的预训练网络，每层的下采样因子为 $R^{(l)}$。定义选择向量 $\mathbf{s} = (s_1, s_2, ..., s_L)$，其中 $s_l \in \{0, ..., R^{(l)}-1\}$。默认前向传播使用 $\mathbf{s} = \mathbf{0}$。方法包含两个核心步骤：(1) 贪心搜索找到一组有用的选择索引 $\hat{\mathcal{S}}$；(2) 用注意力模块聚合对应的特征图进行预测。

### 关键设计

1. **注意力聚合模块（Attention Aggregation）**：对选定的 $B_{ours}$ 组特征 $\mathcal{F} = \{\mathbf{f}_\mathbf{s} | \mathbf{s} \in \hat{\mathcal{S}}\}$，使用多头自注意力学习特征间的相对重要性：

$$A_{learned}(\mathcal{F}) = \frac{1}{B_{ours}} \sum_{\mathbf{s} \in \hat{\mathcal{S}}} \left(\mathbf{f}_\mathbf{s} + \text{MLP}\left(\sum_{\mathbf{s}' \in \hat{\mathcal{S}}} W_{\mathbf{s}\mathbf{s}'}\mathbf{v}_{\mathbf{s}'}\right)\right)$$

其中注意力权重 $W_{\mathbf{s}\mathbf{s}'} = \frac{\exp(\mathbf{q}_\mathbf{s}^\top \mathbf{k}_{\mathbf{s}'})}{\sum_{\mathbf{s}''} \exp(\mathbf{q}_\mathbf{s}^\top \mathbf{k}_{\mathbf{s}''})}$。**设计动机**：注意力是集合运算符，训练一次可在任意测试时预算下使用，无需重训。同时提供了无学习版本——基于熵的加权：$w_\mathbf{s} = \frac{1}{Z}(1 - \frac{H(C_\phi(\mathbf{f}_\mathbf{s}))}{\log K})$，低熵（高置信度）的特征获得更高权重。

2. **空间对齐（Spatial Alignment）**：不同选择索引 $\mathbf{s}$ 产生的激活图在空间上存在偏移。通过计算相对位移进行对齐：

$$\Delta = \sum_{l=1}^{L} s_l \prod_{l'=1}^{l} R^{(l')}$$

将激活图上采样到输入分辨率后按 $\Delta$ 进行平移对齐，避免空间不匹配导致的性能下降。消融实验显示对齐带来0.36%的精度提升。

3. **贪心搜索算法（Greedy Search）**：由于搜索空间 $|\mathcal{S}|$ 随下采样层数指数增长，采用自顶向下的贪心搜索。使用优先队列，从默认状态 $\mathbf{0}$ 开始，每次展开当前最优节点的邻居，基于评分准则优先访问最有潜力的状态。**学习准则**使用注意力权重的倒数（高权重→优先展开），**无学习准则**使用预测熵（低熵→高置信→优先展开）。

### 训练策略

- 聚合模块在冻结的预训练模型上训练，使用ImageNet的20K训练子集
- 训练时固定 $B_{ours}=30$，测试时可用任意预算（注意力模块的集合运算特性）
- 使用AdamW优化器，cosine-annealing调度，lr=$1e^{-6}$
- 省略首层和末层的搜索空间以平衡性能和速度

## 实验关键数据

### 主实验（ImageNet图像分类，$B_{total}=150$）

| 方法组合 | ResNet18 | ResNet50 | MobileNetV2 | InceptionV3 |
|---------|----------|----------|-------------|-------------|
| GPS (无Ours) | 70.69 | 76.87 | 72.58 | 72.02 |
| GPS + Ours | **70.74** | **76.87** | **72.58** | **72.02** |
| ClassTTA (无Ours) | 66.40 | 73.56 | 67.81 | 70.34 |
| ClassTTA + Ours | **70.37** | **76.65** | **71.63** | **72.00** |
| AugTTA (无Ours) | 70.28 | 76.47 | 72.46 | 71.98 |
| AugTTA + Ours | **70.74** | **76.89** | **72.58** | **72.24** |

平均提升：GPS +0.32%, ClassTTA +2.01%, AugTTA +0.19%

### 语义分割（Cityscapes & ADE20K, mIoU）

| 架构 | 数据集 | Baseline | +Ours($B=4$) | +Ours($B=10$) |
|------|--------|----------|-------------|---------------|
| ResNet50-DeepLab | Cityscapes | 79.60 | 79.72 | 79.73 |
| MiT-SegFormer | Cityscapes | 76.54 | 77.01 | **77.05** |
| ResNet50-DeepLab | ADE20K | 42.72 | 42.77 | 42.81 |
| MiT-SegFormer | ADE20K | 37.41 | **37.68** | 37.67 |

### 消融实验

| 消融项 | 配置 | Top-1 Acc(%) | 说明 |
|--------|------|-------------|------|
| 聚合方法 | Average | 79.38 | 简单平均 |
| | Entropy加权 | 79.44 | 无学习方案 |
| | Ours w/o Align | 79.52 | 注意力但无对齐 |
| | **Ours (w/ Align)** | **79.88** | 完整方案 |
| 搜索准则 | Random | 79.80 | 随机展开 |
| | Δ (位移) | 79.72 | 按偏移量排序 |
| | Entropy | 79.86 | 无学习准则 |
| | **Learned (Ours)** | **79.88** | 注意力准则 |
| 搜索空间 | {1,...,L} | 79.64 | 全部层，延迟53.46 |
| | {2,...,L-1} | **79.88** | 去首末层，延迟21.21 |

### 关键发现

1. **被丢弃的激活确实有用**：在9种不同架构上，利用丢弃的激活图一致地提升了模型性能
2. **与TTA正交互补**：在已有TTA基础上叠加本方法可获得额外增益，因为本方法在特征空间而非图像空间操作
3. **性能提升约在 $B_{ours}≈15$ 时饱和**：不需要很大的测试时预算即可获益
4. **一次训练、任意预算**：利用注意力的集合运算性质，训练一次即可在不同预算下泛化

## 亮点与洞察

- 视角新颖：首次从下采样层"丢弃信息"的角度出发设计测试时方法，是TTA之外的全新路径
- 理论洞察深刻：将stride convolution分解为conv + subsampling，统一了所有含下采样操作的网络
- 空间对齐设计精巧，解决了不同选择索引之间的空间不匹配问题
- 方法具有很强的通用性，在CNN和ViT上均有效

## 局限与展望

- 测试时计算开销增加（需多次前向传播），不适用于实时应用场景
- 性能提升幅度相对有限（分类约0.3-2%，分割约0.2-0.5 mIoU），在已经很强的模型上收益递减
- 搜索过程的贪心策略可能错过全局最优的激活组合
- 未探索在训练时就利用多种选择索引来增强模型的可能性
- 对于没有显式下采样层的架构（如纯MLP）不适用

## 相关工作与启发

- GPS [Molchanov et al.] 和 ClassTTA/AugTTA [Shanmugam et al.] 是图像空间TTA的代表
- Anti-aliased CNN [Zhang 2019] 通过blur pool保留下采样丢失的信息，属于训练时方案
- 本方法启发：是否可以在训练时使用随机选择索引作为数据增强？
- 可考虑将本方法应用于视频理解、3D视觉等时空下采样更多的任务

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 全新视角，从网络架构内部挖掘测试时可利用的信息
- 实验充分度: ⭐⭐⭐⭐⭐ 9种架构、两个任务、多个数据集、详细消融
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，公式推导严谨，图示直观
- 价值: ⭐⭐⭐⭐ 与TTA正交的新方向，但绝对增益有限

<!-- RELATED:START -->

## 相关论文

- [IT³: Idempotent Test-Time Training](../../ICML2025/segmentation/it3_idempotent_test-time_training.md)
- [Hybrid-TTA: Continual Test-time Adaptation via Dynamic Domain Shift Detection](../../ICCV2025/segmentation/hybrid-tta_continual_test-time_adaptation_via_dynamic_domain_shift_detection.md)
- [Show and Tell: Visually Explainable Deep Neural Nets via Spatially-Aware Concept Bottleneck Models](../../CVPR2025/segmentation/show_and_tell_visually_explainable_deep_neural_nets_via_spatially-aware_concept_.md)
- [Correspondence as Video: Test-Time Adaption on SAM2 for Reference Segmentation in the Wild](../../ICCV2025/segmentation/correspondence_as_video_testtime_adaption_on_sam2_for_refere.md)
- [TopoTTA: Topology-Enhanced Test-Time Adaptation for Tubular Structure Segmentation](../../ICCV2025/segmentation/topotta_topology-enhanced_test-time_adaptation_for_tubular_structure_segmentatio.md)

<!-- RELATED:END -->
