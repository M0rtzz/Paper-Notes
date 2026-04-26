---
title: >-
  [论文解读] Frequency Dynamic Convolution for Dense Image Prediction
description: >-
  [CVPR 2025][图像分割][动态卷积] FDConv 从频率域角度重新设计动态卷积，通过傅里叶不相交权重（FDW）在不增加参数的前提下构建频率多样的卷积核，结合核空间调制（KSM）和频带调制（FBM）实现精细的频率自适应，仅增加 3.6M 参数即超越需要 65-90M 额外参数的现有动态卷积方法。
tags:
  - CVPR 2025
  - 图像分割
  - 动态卷积
  - 频率域
  - 密集预测
  - 分割
  - 目标检测
---

# Frequency Dynamic Convolution for Dense Image Prediction

**会议**: CVPR 2025  
**arXiv**: [2503.18783](https://arxiv.org/abs/2503.18783)  
**代码**: https://github.com/Linwei-Chen/FDConv  
**领域**: 分割  
**关键词**: 动态卷积, 频率域, 密集预测, 分割, 目标检测

## 一句话总结
FDConv 从频率域角度重新设计动态卷积，通过傅里叶不相交权重（FDW）在不增加参数的前提下构建频率多样的卷积核，结合核空间调制（KSM）和频带调制（FBM）实现精细的频率自适应，仅增加 3.6M 参数即超越需要 65-90M 额外参数的现有动态卷积方法。

## 研究背景与动机

**领域现状**：动态卷积（Dynamic Convolution）通过学习多组并行权重并用注意力机制进行线性组合，实现输入自适应的权重选择，在目标检测和分割等密集预测任务上展现了良好性能。代表方法包括 CondConv、DY-Conv、ODConv 等。

**现有痛点**：现有动态卷积方法的多组并行权重在频率响应上高度相似。作者的分析揭示 ODConv 的 4 组权重余弦相似度超过 0.88，t-SNE 可视化显示滤波器紧密聚集。这意味着虽然参数量增加了 4 倍，但实际上学到的频率信息高度冗余，模型的频率自适应能力有限。

**核心矛盾**：传统动态卷积在空间域学习多组权重，因为没有显式的频率约束，优化过程自然收敛到相似的频率响应——参数代价高但多样性低。低频信息有助于抑制噪声，高频信息捕捉边界和细节，缺乏频率多样性限制了模型自适应提取不同频率特征的能力。

**本文目标** (1) 如何在不增加参数的前提下构建频率响应多样的卷积权重？(2) 如何让卷积自适应地在空间维度和频率维度上调节频率响应？

**切入角度**：作者观察到在傅里叶域，不同频率分量对应不同的空间模式。如果将参数按频率索引分组，每组只包含不相交的频率分量，那么由这些分组经 iDFT 变换得到的权重必然具有不同的频率响应——这是由数学性质保证的。

**核心 idea**：在傅里叶域将固定参数预算按频率索引不相交分组，通过 iDFT 构建频率多样的卷积权重，并以核空间调制和频带调制增强频率自适应能力。

## 方法详解

### 整体框架
FDConv 由三个核心模块组成：(1) **Fourier Disjoint Weight (FDW)** 在傅里叶域将参数按频率分组，构建具有不同频率响应的多组权重；(2) **Kernel Spatial Modulation (KSM)** 在核空间级别动态调制每个滤波器元素的频率响应；(3) **Frequency Band Modulation (FBM)** 在频率域将权重分解为不同频带，实现空间变化的频率调制。FDConv 可以直接替换标准卷积层集成到 ResNet、ConvNeXt、Swin Transformer 等多种架构中。

### 关键设计

1. **Fourier Disjoint Weight (FDW, 傅里叶不相交权重)**:

    - 功能：用固定的参数预算 $k \times k \times C_{in} \times C_{out}$ 构建 $n > 10$ 组频率响应多样的权重，而传统方法需要 $n$ 倍参数且仅能生成少量（$n < 10$）相似权重
    - 核心思路：将参数重塑为 $\mathbf{P} \in \mathbb{R}^{kC_{in} \times kC_{out}}$，每个参数对应一个傅里叶索引 $(u,v)$。按索引的 $L_2$ 范数从低频到高频排序，均匀分为 $n$ 个不相交组。每组参数经 iDFT 变换到空间域，再裁剪为 $k \times k$ 的 patches 并重组为标准权重张量。由于每组只包含特定频段的傅里叶系数，变换后的权重必然具有不同的频率响应（余弦相似度为 0），然后通过注意力系数线性组合。
    - 设计动机：传统方法在空间域学习权重无法保证频率多样性。FDW 利用傅里叶变换的数学性质从根本上解决了这个问题：不相交的频率索引 → 不同的频率响应。且实际参数数目不变，只是对同一组参数做了不同的频率分片。

2. **Kernel Spatial Modulation (KSM, 核空间调制)**:

    - 功能：对每个权重中的每个滤波器元素进行独立的动态调制，精细调节频率响应
    - 核心思路：预测一个密集调制矩阵 $\alpha \in \mathbb{R}^{k \times k \times C_{in} \times C_{out}}$，对权重逐元素调制。包含两个分支：**局部通道分支**用轻量 1D 卷积捕捉局部通道信息，预测完整的密集调制矩阵；**全局通道分支**用全连接层捕捉全局信息，预测输入通道、输出通道、核空间三个维度的稀疏调制值。两分支融合得到最终调制矩阵。
    - 设计动机：FDW 的权重级混合（Eq.1）太粗糙，无法独立调整每个 $k \times k$ 滤波器的频率响应。KSM 提供了逐元素的细粒度控制。用 1D 卷积替代全连接层大幅降低了预测密集矩阵的参数开销。

3. **Frequency Band Modulation (FBM, 频带调制)**:

    - 功能：将卷积权重分解为不同频带，对每个频带在不同空间位置独立调制，实现空间变化的频率自适应
    - 核心思路：在频率域用二值 mask $\mathcal{M}_b$ 将权重的频率响应分解为 $B$ 个频带（默认 4 个，按 octave 划分：$\{0, 1/16, 1/8, 1/4, 1/2\}$）。利用卷积定理在频率域做卷积（点乘），避免空间域理想滤波器无限支撑的问题。对每个频带的输出用标准卷积+sigmoid 预测空间调制图 $\mathbf{A}_b \in \mathbb{R}^{h \times w}$，最终输出 $\mathbf{Y} = \sum_{b=0}^{B-1} \mathbf{A}_b \odot \mathbf{Y}_b$。由对偶性，分解核频带等价于分解特征频带。
    - 设计动机：FDW 和 KSM 都是空间不变的（权重在整个特征图上共享），但自然图像中不同空间区域需要不同的频率处理——背景需要抑制高频噪声，边界需要保留高频细节。FBM 让频率响应可以随空间位置动态变化，实现了真正的内容自适应频率调制。

### 损失函数 / 训练策略
FDConv 作为即插即用模块，遵循下游任务的标准训练策略。权重数目 $n$ 默认设为 64。训练沿用各原始框架的设置（如 Mask R-CNN 使用 1x schedule 训练 12 epochs）。

## 实验关键数据

### 主实验

| 模型/框架 | 方法 | 参数增量 | AP_box | AP_mask |
|-----------|------|---------|--------|---------|
| Faster R-CNN/R50 | Baseline | - | 37.2 | - |
| Faster R-CNN/R50 | +CondConv (8×) | +90.0M | 38.1 | - |
| Faster R-CNN/R50 | +ODConv (4×) | +65.1M | 39.2 | - |
| Faster R-CNN/R50 | **+FDConv** | **+3.6M** | **39.4** | - |
| Mask R-CNN/R50 | Baseline | - | 39.6 | 36.4 |
| Mask R-CNN/R50 | +KW (4×) | +76.5M | 42.4 | 38.9 |
| Mask R-CNN/R50 | **+FDConv** | **+3.6M** | **42.4** | 38.6 |
| UPerNet/R50 | Baseline | - | mIoU 40.7 | - |
| UPerNet/R50 | +ODConv (4×) | +65M | mIoU 43.3 | - |
| UPerNet/R50 | **+FDConv** | **+4M** | **mIoU 43.8** | - |

### 消融实验

| 架构/方法 | 额外参数 | AP_box | AP_mask |
|-----------|---------|--------|---------|
| ConvNeXt-T + KW | +4M | 44.8 | 40.6 |
| **ConvNeXt-T + FDConv** | **+3M** | **45.2** | **40.8** |
| Swin-T + FDConv | +3M | 44.5 | 40.5 |
| Mask2Former-R50 | - | mIoU 79.4 | - |
| **Mask2Former-R50 + FDConv** | - | **mIoU 80.4** | - |
| MaskDINO-Swin-L† + FDConv | - | mIoU 57.2 (+0.5) | - |

### 关键发现
- **参数效率极高**：FDConv 仅增加 3.6M 参数即达到甚至超越 CondConv (+90M)、ODConv (+65.1M)、KW (+76.5M) 的性能，参数效率提升约 20 倍
- **频率多样性是关键**：ODConv 的 4 组权重余弦相似度 >0.88，FDConv 为 0——这直接转化为更好的表征能力
- FBM 可视化显示高频调制值集中在物体边界，低频调制值集中在物体内部，符合直觉且确认了空间变化频率调制的有效性
- FDConv 具有良好的跨架构泛化性，可无缝集成到 CNN（ResNet, ConvNeXt）和 Transformer（Swin）架构中

## 亮点与洞察
- **"用相同参数制造多样性"**的核心思想极为优雅——不是增加参数，而是在傅里叶域对同一参数做不同频率切片，由数学性质保证多样性。这个思路可推广到任何需要多样化基函数的场景
- **卷积定理的巧妙应用**：在频率域做卷积避免了空间域理想滤波器的无限支撑问题，同时利用对偶性证明分解核频带等价于分解特征频带，提供了实现灵活性
- **FBM 的空间变化频率调制**弥合了动态卷积（空间不变权重）和可变形卷积（空间变化采样）之间的差距

## 局限与展望
- FBM 引入了额外的频率域计算（FFT 和 iFFT），增加了 1.8G FLOPs，在实时推理场景中可能不可忽视
- 频带数目 $B=4$ 和频率划分阈值是手动设定的，可考虑自适应学习
- 仅在视觉任务上验证，可探索在音频、信号处理等其他频率敏感领域的应用
- 与可变形卷积的组合值得探索——FDConv 解决了频率自适应但采样位置固定

## 相关工作与启发
- **vs ODConv**: ODConv 在空间域学习 4 组权重并用通道/滤波器/空间三维注意力调制，但权重频率响应高度相似。FDConv 从频率域切入，用更少参数实现了更好的多样性和性能
- **vs KW**: KW 通过将权重分解为可共享的小单元来降低参数，但本质上仍在空间域。FDConv 在频率域分组，多样性有数学保证。KW (4×) 用 76.5M 参数达到的性能，FDConv 用 3.6M 即可匹配
- **vs FADC**: FADC 根据特征频率特性调整膨胀率，而 FDConv 直接对卷积核做频带分解和空间调制，粒度更细

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从频率域重新理解动态卷积是全新视角，FDW 的傅里叶不相交分组保证多样性的设计极为精巧
- 实验充分度: ⭐⭐⭐⭐ 覆盖检测、实例分割、语义分割多个任务，多种架构验证，分析深入
- 写作质量: ⭐⭐⭐⭐ 频率分析可视化（Fig.1, Fig.5）直观有说服力，方法阐述清晰
- 价值: ⭐⭐⭐⭐⭐ 是动态卷积方向的重要突破，20× 参数效率提升具有极强的实用价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception](declip_decoupled_learning_for_open-vocabulary_dense_perception.md)
- [\[ICCV 2025\] Dynamic Dictionary Learning for Remote Sensing Image Segmentation](../../ICCV2025/segmentation/dynamic_dictionary_learning_for_remote_sensing_image_segmentation.md)
- [\[CVPR 2025\] OverLoCK: An Overview-first-Look-Closely-next ConvNet with Context-Mixing Dynamic Kernels](overlock_an_overview-first-look-closely-next_convnet_with_context-mixing_dynamic.md)
- [\[CVPR 2025\] HFP-SAM: Hierarchical Frequency Prompted SAM for Efficient Marine Animal Segmentation](hfp-sam_hierarchical_frequency_prompted_sam_for_efficient_marine_animal_segmenta.md)
- [\[CVPR 2025\] Exploring CLIP's Dense Knowledge for Weakly Supervised Semantic Segmentation](exploring_clips_dense_knowledge_for_weakly_supervised_semantic_segmentation.md)

<!-- RELATED:END -->
