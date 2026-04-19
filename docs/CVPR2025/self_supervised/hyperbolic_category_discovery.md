---
title: >-
  [论文解读] Hyperbolic Category Discovery
description: >-
  [CVPR 2025][自监督学习][广义类别发现] 提出HypCD框架，将广义类别发现（GCD）中的表示学习从欧氏/球面空间迁移到双曲空间（Poincaré球模型），利用双曲空间指数级体积增长天然适合编码层次结构的特性，通过距离-角度混合相似度学习和双曲分类器，在CUB上将SelEx从69.1%提升到71.8%，在ImageNet-100上从87.1%提升到88.3%。
tags:
  - CVPR 2025
  - 自监督学习
  - 广义类别发现
  - 双曲空间
  - Poincaré球
  - 层次结构
  - 混合相似度
---

# Hyperbolic Category Discovery

**会议**: CVPR 2025  
**arXiv**: [2504.06120](https://arxiv.org/abs/2504.06120)  
**代码**: https://visual-ai.github.io/hypcd/  
**领域**: 自监督学习  
**关键词**: 广义类别发现、双曲空间、Poincaré球、层次结构、混合相似度

## 一句话总结
提出HypCD框架，将广义类别发现（GCD）中的表示学习从欧氏/球面空间迁移到双曲空间（Poincaré球模型），利用双曲空间指数级体积增长天然适合编码层次结构的特性，通过距离-角度混合相似度学习和双曲分类器，在CUB上将SelEx从69.1%提升到71.8%，在ImageNet-100上从87.1%提升到88.3%。

## 研究背景与动机

**领域现状**：广义类别发现（GCD）需要在仅有部分类别标注的情况下，同时识别已知类和发现未知新类。现有方法在欧氏或球面空间中学习表示并用半监督k-means做标签分配。

**现有痛点**：欧氏空间体积多项式增长、球面空间体积恒定，两者都不适合编码现实中普遍存在的层次结构（如"动物→鸟类→麻雀"的层级关系）。细粒度类别间的层次关系（部件组合、尺度变化）无法被这些空间高效编码。

**核心矛盾**：GCD的本质挑战在于区分细粒度类别，而细粒度类别天然具有层次结构（共享部件但细节不同）。欧氏空间无法自然表达这种树状嵌套关系。

**本文目标** 利用更适合层次数据的几何空间来增强GCD中细粒度类别的区分能力。

**切入角度**：双曲空间体积随距离指数增长，天然适合嵌入树状层次结构——树的叶子节点在双曲空间中自然获得更大的"活动空间"来保持区分度。

**核心 idea**：将GCD的表示学习搬到Poincaré球双曲空间，用距离+角度混合相似度替代单一距离度量，利用指数级体积增长更好地编码细粒度类别的层次结构。

## 方法详解

### 整体框架
在任何GCD baseline之上添加双曲空间模块：（1）将DINO/ViT提取的欧氏特征通过指数映射投射到Poincaré球上；（2）在双曲空间中用距离-角度混合相似度进行对比学习；（3）对参数化方法用双曲分类器替代欧氏MLP；（4）推理时用标准欧氏k-means（双曲训练的层次知识已迁移到特征中）。

### 关键设计

1. **双曲空间映射**:

    - 功能：将欧氏特征映射到Poincaré球模型中
    - 核心思路：用指数映射$\exp_o^c(z)$将欧氏嵌入投射到曲率为$-c$的Poincaré球上。关键加入特征裁剪操作防止梯度消失——当嵌入接近球的边界时梯度趋于无穷，裁剪值$r$控制最大范数
    - 设计动机：Poincaré球模型的保角性使角度信息在映射后得以保留，同时双曲距离在球心附近像欧氏距离、在边界附近快速增长

2. **距离-角度混合相似度**:

    - 功能：在双曲空间中同时利用距离和角度信息进行对比学习
    - 核心思路：定义两种相似度——距离相似度$\mathcal{S}_d = -\mathcal{D}_\mathbb{H}$（双曲距离的负数）和角度相似度（余弦相似度，因保角性在双曲空间中有效）。总损失$\mathcal{L}_{hrep} = \alpha_d \mathcal{L}_{dis} + (1-\alpha_d) \mathcal{L}_{ang}$，$\alpha_d \approx 0.5$最优
    - 设计动机：单独用距离或角度都不够——距离捕获层次深度（离原点远的点更在层次底部），角度捕获同层相似性（同一父节点下的兄弟类在角度上相近）。两者互补

3. **双曲分类器（HypFFN）**:

    - 功能：替代欧氏MLP进行双曲空间中的分类
    - 核心思路：使用Möbius加法和Möbius标量乘法替代标准线性层的加法和乘法：$v_i = w \otimes_c z_i^\mathbb{H}$。加入安全投影防止数值溢出
    - 设计动机：欧氏空间的线性层假设平坦几何，在弯曲的双曲空间中无效

### 损失函数 / 训练策略
标准对比学习损失在双曲空间中的版本，标注数据用监督对比、无标注数据用自监督对比。推理时可以直接在欧氏空间做k-means——实验发现双曲训练的层次知识会迁移到欧氏表示中。

## 实验关键数据

### 主实验（DINO backbone）

| 基线方法 | 数据集 | 原始 All Acc | +HypCD All Acc | 提升 |
|---------|--------|-------------|---------------|------|
| SPTNet | CUB | 65.8% | 68.2% | +2.4% |
| SelEx | CUB | 69.1% | 71.8% | +2.7% |
| SPTNet | Stanford Cars | 59.0% | 62.1% | +3.1% |
| SelEx | Stanford Cars | 60.8% | 63.4% | +2.6% |
| SPTNet | CIFAR-100 | 81.3% | 82.9% | +1.6% |
| SelEx | ImageNet-100 | 87.1% | 88.3% | +1.2% |

在所有基线和数据集上一致提升，细粒度数据集（CUB、Cars）上提升更大。

### 消融实验

| 配置 | 说明 |
|------|------|
| 仅距离相似度 | 不如混合 |
| 仅角度相似度 | 不如混合 |
| $\alpha_d = 0.5$ | 最优平衡 |
| 无特征裁剪 | 梯度爆炸/消失 |
| 欧氏k-means（推理） | 性能保持，双曲知识已迁移 |

### 关键发现
- 双曲空间在细粒度数据集上优势更大（CUB +2.7%, Cars +3.1%），因为细粒度类别的层次结构更显著
- 双曲训练产生的表示即使投回欧氏空间做k-means也有效，说明层次结构信息已编码到特征中
- 混合相似度优于单一度量，$\alpha_d = 0.5$说明距离和角度信息同等重要
- 框架作为即插即用模块可以提升多种GCD基线（SPTNet、SelEx），具有通用性

## 亮点与洞察
- **几何空间选择对GCD很重要**：以往GCD研究关注损失函数和训练策略，本文首次系统研究了表示空间的几何性质对性能的影响
- **"训练时双曲、推理时欧氏"的实用性**：避免了双曲空间在推理时的计算复杂度，同时保留了层次编码的好处
- **距离-角度混合的物理直觉**：距离表征"在层次树中的深度"，角度表征"在同层中的位置"，这个分解非常直观

## 局限与展望
- Poincaré球的曲率$c$是人工设定的，自适应曲率学习可能更好
- 推理时用欧氏k-means是一种妥协——双曲k-means可能进一步提升
- 仅在视觉GCD上验证，文本/多模态GCD未涉及
- 层次结构的评估缺少显式的层次度量（如对层次一致性的量化分析）

## 相关工作与启发
- **vs SimGCD/SPTNet**: 这些方法在欧氏/球面空间工作；HypCD通过改变几何空间获得了一致但显著的提升（+2-3%）
- **vs 双曲表示学习（HYPE等）**: HYPE用于图嵌入；HypCD首次将双曲空间引入GCD任务，设计了GCD特有的混合相似度
- **vs 度量学习**: 传统度量学习用欧氏距离；双曲距离在细粒度区分上有理论优势（体积指数增长→更多空间区分相似类别）

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将双曲空间引入GCD，距离-角度混合相似度设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 多基线、多数据集、消融完整
- 写作质量: ⭐⭐⭐⭐ 双曲空间的直觉解释清晰
- 价值: ⭐⭐⭐⭐ 即插即用的GCD增强模块，对细粒度类别发现有显著贡献

<!-- RELATED:START -->

## 相关论文

- [SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery](../../NeurIPS2025/self_supervised/seal_semantic-aware_hierarchical_learning_for_generalized_category_discovery.md)
- [PromptCCD: Learning Gaussian Mixture Prompt Pool for Continual Category Discovery](../../ECCV2024/self_supervised/promptccd_learning_gaussian_mixture_prompt_pool_for_continual_category_discovery.md)
- [Foundation Models for Scientific Discovery: From Paradigm Enhancement to Paradigm Transition](../../NeurIPS2025/self_supervised/foundation_models_for_scientific_discovery_from_paradigm_enhancement_to_paradigm.md)
- [Towards LLM-Empowered Knowledge Tracing via LLM-Student Hierarchical Behavior Alignment in Hyperbolic Space](../../AAAI2026/self_supervised/towards_llm-empowered_knowledge_tracing_via_llm-student_hierarchical_behavior_al.md)
- [SMILE: Infusing Spatial and Motion Semantics in Masked Video Learning](smile_infusing_spatial_and_motion_semantics_in_masked_video_learning.md)

<!-- RELATED:END -->
