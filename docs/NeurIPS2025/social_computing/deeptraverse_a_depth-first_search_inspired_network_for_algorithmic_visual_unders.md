---
title: >-
  [论文解读] DeepTraverse: A Depth-First Search Inspired Network for Algorithmic Visual Understanding
description: >-
  [NeurIPS 2025][vision backbone] 受深度优先搜索（DFS）算法启发，设计了 DeepTraverse 视觉骨干网络，通过参数共享的递归探索模块和自适应通道校准模块，在极少参数下实现高竞争力的图像分类性能。
tags:
  - NeurIPS 2025
  - vision backbone
  - DFS inspired
  - recursive exploration
  - channel recalibration
  - parameter efficiency
---

# DeepTraverse: A Depth-First Search Inspired Network for Algorithmic Visual Understanding

**会议**: NeurIPS 2025  
**arXiv**: [2506.10084](https://arxiv.org/abs/2506.10084)  
**代码**: 无  
**领域**: model_compression  
**关键词**: vision backbone, DFS inspired, recursive exploration, channel recalibration, parameter efficiency

## 一句话总结
受深度优先搜索（DFS）算法启发，设计了 DeepTraverse 视觉骨干网络，通过参数共享的递归探索模块和自适应通道校准模块，在极少参数下实现高竞争力的图像分类性能。

## 研究背景与动机

**领域现状**：视觉骨干网络（CNN/ViT）通常采用层层堆叠的均匀级联结构，隐式地逐层抽象特征。轻量级网络如 MobileNet、ShuffleNet、GhostNet 通过优化卷积算子降低 FLOPs。

**现有痛点**：传统网络的特征抽象路径缺乏显式的迭代精炼机制——增加深度会线性增加参数量，增加注意力会引入额外计算。缺少一种"有策略地探索特征空间"的范式。

**核心矛盾**：想要更深层的特征精炼就需更多参数，但资源受限场景不允许。参数效率和表征深度之间存在 trade-off。

**本文目标**：在极低参数预算下实现深层特征精炼，打破"深度=参数"的线性关系。

**切入角度**：从经典的 DFS 算法获得灵感——DFS 通过系统性地沿路径深入探索然后回溯评估来收集信息。将这种"深入-评估-调整"的范式嵌入网络结构。

**核心 idea**：用参数共享的递归模块模拟 DFS 的深入探索（多次迭代但参数不增加），用通道注意力模拟 DFS 的回溯评估（动态重新加权特征通道）。

## 方法详解

### 整体框架
DeepTraverse 是多阶段层级结构，基本单元为 DFSBlock。每个 DFSBlock 包含两个核心子模块：DFS-EB（探索模块）负责递归特征精炼，DFS-BB（回溯模块）负责全局上下文校准。多个 DFSBlock 以层级方式堆叠，配合残差连接。

### 关键设计

1. **DFS-Inspired Exploration Block (DFS-EB)**:

    - 功能：对输入特征进行 $R$ 轮递归精炼，每轮使用相同参数。
    - 核心思路：初始特征提取 $F_0 = \Phi_{\text{extract}}(X)$，其中 $\Phi_{\text{extract}}$ 包含 depthwise 卷积 + BN + ReLU + pointwise 卷积。然后递归 $R$ 步：$F_i = F_{i-1} + \Phi_{\text{recursive}}(F_{i-1})$，$\Phi_{\text{recursive}}$ 与 $\Phi_{\text{extract}}$ 结构相同但参数跨所有 $R$ 步共享。
    - 最终输出：$Y_{EB} = F_R = \Phi_{\text{extract}}(X) + \sum_{j=1}^{R} \Phi_{\text{recursive}}(F_{j-1})$
    - 设计动机：参数共享让网络可以进行等效 $R$ 倍深度的计算而不增加独立参数，类似 DFS 沿同一路径不断深入探索。使用 depthwise separable 卷积进一步压缩参数。

2. **DFS-Inspired Backtrack Block (DFS-BB)**:

    - 功能：对探索后的特征进行全局上下文感知的通道重校准。
    - 核心思路：全局平均池化得 $z = \text{AdaptiveAvgPool}(F)$，然后通过 bottleneck FC 网络计算通道注意力 $s = \sigma(W_2 \delta(W_1 z))$，最后 $F' = F \odot s$。
    - 设计动机：模拟 DFS 回溯时的评估步骤——探索完后根据全局上下文重新判断哪些特征通道更重要，抑制无关通道、增强判别性通道。本质上类似 SE-Net 的通道注意力但嵌入在 DFS 框架中有了算法动机。

3. **Integrated DFSBlock**:

    - 功能：将 DFS-EB 和 DFS-BB 串联，加上残差快捷连接构成完整单元。
    - 公式：$X_{\text{out}} = \delta(F_{\text{recalibrated}} + S(X_{\text{in}}))$，其中 $S$ 是维度对齐的投影快捷连接。
    - 设计动机：残差连接保证梯度流通，DFS-EB 提供多层次特征探索，DFS-BB 提供全局评估，三者协同完成一个完整的"探索-评估-整合"循环。

### 训练策略
- 从头训练 100 epochs，初始学习率 0.1
- 在 NVIDIA RTX 2080 Ti 上训练
- 使用 timm 和 thop 库评估参数量和 FLOPs

## 实验关键数据

### 主实验

| 方法 | 参数量 | FLOPs | CIFAR-100 | CIFAR-10 |
|------|--------|-------|-----------|----------|
| **DeepTraverse** | **0.26M** | **0.03G** | **73.84** | **93.25** |
| DenseNet | 0.60M | 0.20G | 73.02 | 92.75 |
| EfficientNet | 4.14M | 0.12G | 73.14 | 90.20 |
| StarNet | 2.70M | 0.28G | 72.27 | 92.13 |
| GhostNet | 2.76M | 0.04G | 72.46 | 91.79 |
| ResNet20 | 0.28M | 0.08G | 69.53 | 92.36 |

DeepTraverse 仅 0.26M 参数就超越了 DenseNet (0.6M)、EfficientNet (4.14M) 等。

| 方法 | 参数量 | FLOPs | ImageNet-1k Top-1 | Top-5 |
|------|--------|-------|--------------------|-------|
| **DeepTraverse** | **5.04M** | **0.84G** | **83.16** | **96.54** |
| DenseNet | 7.05M | 2.89G | 81.44 | 95.74 |
| GhostNet | 5.30M | 0.28G | 80.34 | 95.16 |
| EfficientNet | 7.28M | 0.42G | 81.18 | 95.24 |
| ResNet50 | 25.56M | 4.13G | 78.76 | 94.18 |

### 消融实验（Wide 版本对比）

| 配置 | 参数量 | FLOPs | CIFAR-100 |
|------|--------|-------|-----------|
| DeepTraverse (Wide) | 14.26M | 1.78G | 82.20 |
| WideResNet | 36.54M | 5.25G | 80.81 |

Wide 版本仅用 WideResNet 39% 的参数和 34% 的 FLOPs 就高出 1.39 点。

### 关键发现
- 参数效率极高：在 CIFAR-100 上 0.26M 参数超越 4M+ 参数的 EfficientNet
- 扩展性好：从小模型到 Wide 版本都保持性能优势
- ImageNet-64 上 0.59M 参数达到 71.50% Top-1，比 GhostNet (2.74M) 高 0.76 点
- 递归探索模块的参数共享是效率的关键来源

## 亮点与洞察
- **参数共享递归**是最核心的 trick：多轮迭代共享权重，等效于加深网络但不加参数，类似于 Universal Transformer 的思想在 CNN 中的应用。
- **算法启发设计**的叙事方式值得学习：虽然核心组件（depthwise conv + SE attention + 参数共享）都是已有技术，但 DFS 算法框架给出了清晰的组合动机和直觉。
- 通道注意力 + 递归精炼的组合范式可迁移到检测、分割等下游任务的轻量级 backbone 设计。

## 局限与展望
- 只在分类任务上验证，未涉及检测和分割——作者也承认受限于计算资源未在 ImageNet-21k 和下游任务上测试
- DFS 算法类比较为松散——递归卷积 + SE 注意力的组合并非 DFS 的严格实现，更像是借鉴了"深入探索 + 回溯评估"的高层直觉
- 训练成本未与其他轻量模型详细对比——参数少不代表训练快，递归 $R$ 步实际上增加了前向传播时间
- ImageNet-1k 只在 100 类子集上评估（Table 3 的结果似乎是子集而非完整 1000 类）

## 相关工作与启发
- **vs MobileNet/ShuffleNet**: 这些方法优化卷积算子结构降低单次运算成本，DeepTraverse 则通过参数共享递归在不增参的同时加深计算
- **vs SE-Net**: DFS-BB 本质上就是 SE 模块，区别在于与递归探索的协同使用有了更强的设计动机
- **vs Universal Transformer**: 思路类似但应用于 CNN——权重共享迭代精炼是一个跨模态的通用策略

## 评分
- 新颖性: ⭐⭐⭐ DFS 类比有新意但核心组件是已有技术的组合
- 实验充分度: ⭐⭐⭐ 多数据集验证但缺乏下游任务和大规模 ImageNet 完整评估
- 写作质量: ⭐⭐⭐⭐ 叙事结构清晰，DFS 类比讲得引人入胜
- 价值: ⭐⭐⭐ 参数效率确实出色，但缺少真实部署场景验证和延迟/吞吐量对比
