---
title: >-
  [论文解读] UniConvNet: Expanding Effective Receptive Field while Maintaining Asymptotically Gaussian Distribution for ConvNets of Any Scale
description: >-
  [ICCV 2025][模型压缩][卷积神经网络] 提出UniConvNet，通过合理组合较小卷积核（7×7, 9×9, 11×11）的三层感受野聚合器（RFA），在扩大有效感受野（ERF）的同时保持其渐近高斯分布（AGD），从而在轻量级到大规模模型上全面超越现有CNN和ViT。 大核卷积网络（如SLaK、UniRepLKN…
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "卷积神经网络"
  - "有效感受野"
  - "渐近高斯分布"
  - "轻量化网络"
  - "大核卷积"
---

# UniConvNet: Expanding Effective Receptive Field while Maintaining Asymptotically Gaussian Distribution for ConvNets of Any Scale

**会议**: ICCV 2025  
**arXiv**: [2508.09000](https://arxiv.org/abs/2508.09000)  
**代码**: [https://github.com/ai-paperwithcode/UniConvNet](https://github.com/ai-paperwithcode/UniConvNet)  
**领域**: 模型压缩 / 高效网络设计  
**关键词**: 卷积神经网络, 有效感受野, 渐近高斯分布, 轻量化网络, 大核卷积

## 一句话总结

提出UniConvNet，通过合理组合较小卷积核（7×7, 9×9, 11×11）的三层感受野聚合器（RFA），在扩大有效感受野（ERF）的同时保持其渐近高斯分布（AGD），从而在轻量级到大规模模型上全面超越现有CNN和ViT。

## 研究背景与动机

大核卷积网络（如SLaK、UniRepLKNet）虽然能获得更大的有效感受野，但存在两个关键问题：

**高参数和计算成本**：极大的卷积核带来显著的参数和FLOPs开销

**破坏ERF的渐近高斯分布**：大核卷积会导致ERF的多尺度影响分布不再符合"距离输出像素越近、影响越大"的自然直觉

传统小核网络（如ResNet-101）通过堆叠3×3卷积虽然ERF较小，但其多尺度梯度影响天然符合AGD。本文的核心问题是：**能否通过合理组合较小的卷积核，既扩大ERF又保持AGD？**

## 方法详解

### 整体框架

UniConvNet采用四阶段金字塔结构（stem + 4 stages），每个stage由多个Three-layer RFA模块堆叠构成。整体架构基于InternImage设计，将其中的卷积替换为提出的RFA模块，并采用DCNV3残差连接（去除了softmax归一化）。

### 关键设计

1. **感受野聚合器（Receptive Field Aggregator, RFA）**：

    - 将输入沿通道维度分为N+1个head：$A_1, H_1, ..., H_N$
    - $A_1$首先送入Layer Operator 1，输出$A_2$的通道数从$\frac{C}{N+1}$增长到$\frac{2C}{N+1}$
    - 递归地将$A_n$送入后续LO，通道维度呈金字塔递增，降低参数量和FLOPs
    - 剩余的$H_n$在每层与$A_n$交互，通过1×1卷积做投影增强特征多样性
    - 设计动机：直接在浅层模块中为不同尺度的感受野分配判别性影响

2. **层操作符（Layer Operator, LO）**：

    - **放大器（Amplifier, Amp）**：对$a_{n,1}$进行深度可分离大核$K \times K$卷积+GELU激活后，与$a_{n,2}$做逐元素乘法。扩展感受野并放大显著像素的影响
    - **判别器（Discriminator, Dis）**：融合深度可分离$K \times K$和$k \times k$（k=3）卷积的特征，为大感受野引入小尺度新像素的判别性影响
    - 两者拼接后形成具有两层AGD的输出，通道数递增
    - 设计动机：从感受野的角度构建空间编码器，通过乘法放大显著特征并添加局部细节

3. **三层RFA配置（Three-layer RFA）**：

    - 对于224×224输入，使用N=3层，卷积核尺寸分别为$K=2n+5$，即7×7、9×9、11×11
    - 小核尺寸k=3，最终形成四层AGD的感受野
    - 最大核11×11确保stage 3的14×14特征图中边角像素最多有四分之一重叠
    - 通过堆叠多个RFA模块可持续扩展ERF同时维持AGD

### 损失函数 / 训练策略

- ImageNet-1K训练300 epochs，使用AdamW优化器
- 大模型（UniConvNet-L/XL）先在ImageNet-22K预训练90 epochs，再在ImageNet-1K微调20 epochs
- 下游任务（COCO检测、ADE20K分割）采用标准训练策略

## 实验关键数据

### 主实验 - ImageNet-1K分类（表格）

| 模型 | 参数量 | FLOPs | Top-1 Acc |
|------|--------|-------|-----------|
| UniRepLKNet-A | 4.4M | 0.6G | 77.0% |
| **UniConvNet-A** | **3.4M** | **0.589G** | **77.0%** |
| DCNV4 | 5.3M | 0.805G | 78.5% |
| **UniConvNet-P0** | **5.2M** | **0.832G** | **79.1%** |
| ConvNeXt-T | 29.0M | 5.0G | 82.1% |
| InternImage-T | 30.0M | 5.0G | 83.5% |
| **UniConvNet-T** | **30.3M** | **5.1G** | **84.2%** |
| InternImage-B | 97.0M | 16.0G | 84.9% |
| **UniConvNet-B** | **97.6M** | **15.9G** | **85.0%** |
| InternImage-XL† | 335M | 163G | 88.0% |
| **UniConvNet-XL†** | **226.7M** | **115.2G** | **88.4%** |

### 消融实验 - 核尺寸选择（表格）

| 模型 | 核尺寸 | 参数量 | FLOPs | Acc |
|------|--------|--------|-------|-----|
| UniConvNet-A | 5,7,9 | 3.5M | 0.564G | 76.6% |
| **UniConvNet-A** | **7,9,11** | **3.4M** | **0.589G** | **77.0%** |
| UniConvNet-A | 9,11,13 | 3.5M | 0.579G | 76.9% |
| UniConvNet-T | 5,7,9 | 30.0M | 5.0G | 84.1% |
| **UniConvNet-T** | **7,9,11** | **30.3M** | **5.1G** | **84.2%** |

### 关键发现

- UniConvNet-T以30M参数和5.1G FLOPs达到84.2% top-1精度，领先同规模模型至少0.6个百分点
- UniConvNet-XL突破CNN瓶颈，达到88.4% top-1精度，参数量和FLOPs显著优于同级别模型
- 下游任务同样优异：COCO检测55.7 APb（Cascade Mask R-CNN），ADE20K分割55.1 mIoU（UperNet）
- 7,9,11的核尺寸组合对224×224输入最优，过大或过小都会降低性能
- 轻量级到大规模模型均有一致提升，验证了方法的通用性

## 亮点与洞察

1. **理论洞察深刻**：首次从ERF的渐近高斯分布角度解释了小核/大核网络的优劣，提出"扩大ERF同时维持AGD"的新范式
2. **设计原则清晰**：通过金字塔通道递增和递归结构，用较小卷积核达到大核效果，参数和计算量显著降低
3. **通用性极强**：从3.4M到226.7M参数的模型变体均表现优异，覆盖移动端到服务器端全场景
4. **即插即用**：Three-layer RFA可作为plug-and-play模块替换任何ConvNet中的卷积

## 局限与展望

- 三层RFA的层数N=3是针对224×224图像的经验设定，对更高分辨率输入的最优配置未充分探索
- 基于InternImage骨架设计，不确定换用其他骨架是否同样有效
- 与最新的Mamba等状态空间模型缺乏比较
- ERF的AGD特性虽可视化展示但缺乏严格的数学证明

## 相关工作与启发

- 与UniRepLKNet等大核方法形成对比，展示了"合理组合小核"比"纯粹放大核"更有效
- 放大器+判别器的设计思路可启发其他需要多尺度特征融合的任务
- 金字塔通道递增策略可用于其他需要降低计算成本的模块设计

## 评分

- **新颖性**: ⭐⭐⭐⭐ 从ERF的AGD角度重新审视卷积网络设计，提出新的设计范式
- **实验充分度**: ⭐⭐⭐⭐⭐ 覆盖分类/检测/分割三大任务，从轻量级到大规模多个变体，消融详尽
- **写作质量**: ⭐⭐⭐⭐ 动机清晰，图示直观，但部分公式排版有小瑕疵
- **价值**: ⭐⭐⭐⭐ 提供了CNN设计的新视角和强baseline，对移动端和服务器端均有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Gradient Short-Circuit: Efficient Out-of-Distribution Detection via Feature Intervention](gradient_short-circuit_efficient_out-of-distribution_detection_via_feature_inter.md)
- [\[CVPR 2025\] MobileMamba: Lightweight Multi-Receptive Visual Mamba Network](../../CVPR2025/model_compression/mobilemamba_lightweight_multi-receptive_visual_mamba_network.md)
- [\[CVPR 2025\] DKDM: Data-Free Knowledge Distillation for Diffusion Models with Any Architecture](../../CVPR2025/model_compression/dkdm_data-free_knowledge_distillation_for_diffusion_models_with_any_architecture.md)
- [\[ICML 2025\] FGFP: A Fractional Gaussian Filter and Pruning for Deep Neural Networks Compression](../../ICML2025/model_compression/fgfp_a_fractional_gaussian_filter_and_pruning_for_deep_neural_networks_compressi.md)
- [\[ICCV 2025\] Beyond Low-Rank Tuning: Model Prior-Guided Rank Allocation for Effective Transfer in Low-Data and Large-Gap Regimes](beyond_low-rank_tuning_model_prior-guided_rank_allocation_for_effective_transfer.md)

</div>

<!-- RELATED:END -->
