---
title: >-
  [论文解读] EA-ViT: Efficient Adaptation for Elastic Vision Transformer
description: >-
  [ICCV 2025][模型压缩][Transformer] 提出首个在适配（adaptation）阶段引入弹性结构的ViT框架，通过多维弹性架构+课程学习+轻量路由器，一次适配即生成覆盖10^26种配置的子模型，在多个下游任务上持续优于现有弹性方法。
tags:
  - ICCV 2025
  - 模型压缩
  - Transformer
  - 弹性架构
  - 子模型选择
  - 课程学习
  - Pareto优化
---

# EA-ViT: Efficient Adaptation for Elastic Vision Transformer

**会议**: ICCV 2025  
**arXiv**: [2507.19360](https://arxiv.org/abs/2507.19360)  
**代码**: [https://github.com/zcxcf/EA-ViT](https://github.com/zcxcf/EA-ViT)  
**领域**: 模型压缩 / 弹性网络  
**关键词**: Vision Transformer, 弹性架构, 子模型选择, 课程学习, Pareto优化

## 一句话总结

提出首个在适配（adaptation）阶段引入弹性结构的ViT框架，通过多维弹性架构+课程学习+轻量路由器，一次适配即生成覆盖10^26种配置的子模型，在多个下游任务上持续优于现有弹性方法。

## 研究背景与动机

Vision Transformer (ViT) 在预训练后需要适配到不同下游任务，但不同部署平台（手机、笔记本、GPU集群）对模型大小有不同要求。现有方案存在两个核心痛点：

**传统剪枝方法**需要为每个目标设备单独训练和微调模型，计算成本高且版本管理复杂

**已有弹性方法**（DynaBERT、MatFormer、HydraViT、Flextron）要么仅支持有限的弹性维度，要么在预训练阶段引入弹性结构，迁移到下游任务仍需额外微调

EA-ViT 的核心动机是：能否在"适配阶段"（而非预训练阶段）将一个预训练ViT转变为弹性结构，一次适配即可生成适配多种资源约束的子模型？

## 方法详解

### 整体框架

EA-ViT 分为两个阶段：
- **Stage 1**: 构建多维弹性架构 + 课程式弹性适配
- **Stage 2**: Pareto最优子模型搜索初始化路由器 + 路由器与骨干联合训练

### 关键设计

1. **多维弹性架构 (Multi-Dimensional Elastic Architecture)**

   在四个维度上引入弹性配置：
   - MLP扩展比 $R^{(l)}$：控制每层MLP隐藏维度
   - 注意力头数 $H^{(l)}$：控制每层活跃的注意力头
   - 嵌入维度 $E$：全层共享的统一嵌入维度
   - 网络深度 $D^{(l)}$：二值指示器，通过identity mapping跳过特定层

   整体配置空间定义为：
   $$\theta = (\{R^{(l)}\}_{l=1}^{L}, \{H^{(l)}\}_{l=1}^{L}, E, \{D_{\text{MLP}}^{(l)}, D_{\text{MHA}}^{(l)}\}_{l=1}^{L})$$

   以ViT-Base为例，搜索空间可达 $10^{26}$ 种子模型，远超前述方法（最多 $10^{14}$）。采用嵌套式（nested）弹性结构，按重要性排序嵌入维度和注意力头，使关键组件在更多子模型间共享参数。

2. **课程式弹性适配 (Curriculum Elastic Adaptation, CEA)**

   同时训练大量子模型会因梯度冲突导致参数干扰。CEA采用课程学习策略，从简单到复杂逐步扩展弹性范围：
   - 初始阶段仅训练最大子模型（$R_{\min}=R_{\max}$, $H_{\min}=H_{\max}$等）
   - 训练过程中按预设步骤逐步降低采样下界，引入更小的子模型
   - 各维度参数从均匀分布中采样：$R \sim \mathcal{U}(R_{\min}^{(t)}, R_{\max})$

   这确保了预训练知识的保留和大模型性能不受干扰。

3. **Pareto最优子模型搜索 + 约束感知路由器**

   **Pareto搜索**：使用定制的NSGA-II进化算法在弹性空间中搜索Pareto前沿上的候选子模型。引入分区选择策略和迭代拥挤距离机制，提高种群多样性和复杂度空间覆盖。

   **路由器**：轻量两层MLP，输入归一化的MACs约束 $M_t \in [0,1]$，输出对应的子模型配置 $\theta$。使用Gumbel-Sigmoid技巧实现离散决策的可微分优化：
   $$y_{\text{soft}} = \text{Sigmoid}\left(\frac{\text{logits} + G_1 - G_2}{\tau}\right)$$
   通过straight-through estimator保持梯度流。

### 损失函数 / 训练策略

Stage 2的统一损失函数：

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_1 \left(\frac{\text{MACs}(\theta)}{M_0} - M_t\right)^2 + \lambda_2 \|\theta - \theta^*(M_t)\|_2^2$$

- 第一项：交叉熵分类损失
- 第二项：计算量约束惩罚（MACs偏差）
- 第三项：Pareto参考正则化，$\lambda_2$ 随训练逐步退火至零，鼓励路由器自主探索

MACs的可微分计算：
$$\text{MACs}(\theta) = \sum_{l=1}^{L}[D_{\text{MLP}}^{(l)} \cdot 2NE^2R^{(l)} + D_{\text{MHA}}^{(l)} \cdot Nd_{\text{head}}H^{(l)}(4E+2N)]$$

## 实验关键数据

### 主实验

在9个图像分类基准上与DynaBERT、MatFormer、HydraViT、Flextron比较（8 GMACs约束，ViT-Base骨干）：

| 方法 | Cifar10 | Cifar100 | SVHN | Flowers | Food101 | Aircraft | Cars | DTD | Pets |
|------|---------|----------|------|---------|---------|----------|------|-----|------|
| DynaBERT | 94.24 | 78.30 | 96.58 | 24.53 | 82.24 | 71.19 | 80.19 | 36.88 | 57.05 |
| MatFormer | 96.17 | 86.18 | 97.45 | 69.00 | 86.31 | 76.13 | 88.10 | 49.25 | 84.53 |
| HydraViT | 96.11 | 85.39 | 96.98 | 29.27 | 85.49 | 75.95 | 85.07 | 43.53 | 75.32 |
| Flextron | 97.11 | 85.95 | 97.61 | 73.80 | 87.79 | 79.36 | 88.62 | 61.21 | 86.35 |
| **EA-ViT** | **97.98** | **88.20** | **97.63** | **85.39** | **90.05** | **83.37** | **90.09** | **67.56** | **90.57** |

在Flowers-102上领先最佳基线方法26.39%，说明在低MACs条件下EA-ViT优势尤为显著。

### 消融实验

**多维弹性贡献**（Cifar100上验证）：

| MLP | MHA | Embed | Depth | MACs_min↓ | Acc_max↑ | AUC↑ |
|-----|-----|-------|-------|-----------|----------|------|
| ✓ | - | - | - | 7.80 | 93.01 | 0.505 |
| ✓ | ✓ | - | - | 6.04 | 93.12 | 0.646 |
| ✓ | ✓ | ✓ | - | 5.11 | 93.31 | 0.651 |
| ✓ | ✓ | ✓ | ✓ | **4.60** | **93.32** | **0.663** |

逐步增加弹性维度可持续提高MACs覆盖范围、最大精度和AUC。

### 关键发现

- **路由器有效性**：路由器自动选择的子模型始终优于手动选择的固定架构子模型
- **CEA的必要性**：去除CEA后虽然小模型略有改善，但大模型性能明显下降；Stage 2后，有CEA的模型在全MACs范围内持续优于无CEA版本
- **Pareto初始化**：用Pareto前沿配置初始化路由器，使训练更有效地利用有限资源
- **t-SNE分析**：不同数据集展现出不同的子模型结构偏好（如DTD和SVHN偏好与通用数据集不同），验证了路由器的适应性

## 亮点与洞察

1. **首个适配阶段弹性框架**：将弹性引入适配而非预训练阶段，大幅降低训练和存储成本
2. **搜索空间巨大**：$10^{26}$ 种子模型配置，比最灵活的现有方法多12个数量级
3. **实用性强**：同一框架适用于分类、分割、医学影像、遥感等多种任务
4. **数据集敏感度分析**：简单数据集（SVHN）对模型缩小不敏感，复杂数据集（DTD、Flowers）需要更大模型表达能力

## 局限与展望

- 目前以MACs作为主要约束，可扩展到延迟、参数量等其他指标
- 弹性适配仍需一定训练成本，能否进一步减少Stage 1的训练开销值得探索
- 路由器预测的是静态配置（基于设备约束），未考虑输入样本的动态适配

## 相关工作与启发

- 与OFA (Once-for-All)类似的理念但针对ViT且在适配阶段操作
- Gumbel-Sigmoid实现离散决策的可微分是值得借鉴的通用技巧
- Pareto搜索+路由器联合优化的两阶段策略可推广到其他配置搜索问题

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个在适配阶段引入弹性的ViT框架，四维弹性设计全面
- **实验充分度**: ⭐⭐⭐⭐⭐ 13个数据集覆盖分类/分割/医学/遥感，消融全面
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，公式推导完整
- **实用价值**: ⭐⭐⭐⭐⭐ 直接面向多设备部署场景，代码已开源

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Efficient Adaptation of Pre-Trained Vision Transformer Underpinned by Approximation Theory](efficient_adaptation_of_pre-trained_vision_transformer_underpinned_by_approximat.md)
- [\[ICCV 2025\] ViT-Linearizer: Distilling Quadratic Knowledge into Linear-Time Vision Models](vit-linearizer_distilling_quadratic_knowledge_into_linear-time_vision_models.md)
- [\[CVPR 2025\] BHViT: Binarized Hybrid Vision Transformer](../../CVPR2025/model_compression/bhvit_binarized_hybrid_vision_transformer.md)
- [\[ICCV 2025\] Context Guided Transformer Entropy Modeling for Video Compression](context_guided_transformer_entropy_modeling_for_video_compression.md)
- [\[ICCV 2025\] MixA-Q: Revisiting Activation Sparsity for Vision Transformers from a Mixed-Precision Quantization Perspective](mixa-q_revisiting_activation_sparsity_for_vision_transformers_from_a_mixed-preci.md)

<!-- RELATED:END -->
