---
title: >-
  [论文解读] DINO-QPM: Adapting Visual Foundation Models for Globally Interpretable Image Classification
description: >-
  [CVPR 2026][可解释分类] 提出 DINO-QPM，一种轻量级可解释性适配器，将冻结的 DINOv2 骨干网络的复杂高维特征转换为对比性的、类无关的可解释表示，通过二次规划进行稀疏特征选择和类级特征分配，在 CUB-2011 和 Stanford Cars 上同时超越了 DINOv2 线性探测的准确率和所有可比方法的可解释性。
tags:
  - CVPR 2026
  - 可解释分类
  - DINOv2
  - 二次规划
  - 视觉基础模型
  - 特征稀疏化
---

# DINO-QPM: Adapting Visual Foundation Models for Globally Interpretable Image Classification

**会议**: CVPR 2026  
**arXiv**: [2604.07166](https://arxiv.org/abs/2604.07166)  
**代码**: [https://github.com/RobertZimm/DINO-QPM](https://github.com/RobertZimm/DINO-QPM)  
**领域**: Model Interpretability / 模型可解释性  
**关键词**: 可解释分类, DINOv2, 二次规划, 视觉基础模型, 特征稀疏化

## 一句话总结

提出 DINO-QPM，一种轻量级可解释性适配器，将冻结的 DINOv2 骨干网络的复杂高维特征转换为对比性的、类无关的可解释表示，通过二次规划进行稀疏特征选择和类级特征分配，在 CUB-2011 和 Stanford Cars 上同时超越了 DINOv2 线性探测的准确率和所有可比方法的可解释性。

## 研究背景与动机

视觉基础模型（如 DINOv2）作为特征提取器表现卓越，但其复杂、高维的表示为可解释性带来了巨大挑战。现有方法面临以下问题：

**后验解释方法不可靠**：注意力图、Grad-CAM 等方法是外部近似，并非模型决策过程的真实反映；注意力图与下游任务无关，经常忽略分类所需的关键信息

**端到端可解释模型资源消耗大**：如原型网络等方法需要对整个骨干进行微调，计算开销极高

**冻结骨干上的方法精度不足**：后验概念瓶颈模型（Post-hoc CBMs）依赖文本概念监督，无法提供直接的空间定位；且精度通常不及全模型微调方法

**原型模型的可解释性有欺骗性**：其相似度计算不一定与人类认知一致

核心思路：能否在完全冻结 DINOv2 骨干的前提下，构建一个轻量适配器，将其强大但纠缠的特征转换为稀疏、可空间定位的、全局可解释的类表示？

## 方法详解

### 整体框架

DINO-QPM 的流程：
1. 冻结的 DINOv2 提取 patch 嵌入 $\boldsymbol{F}^{\text{froz}} \in \mathbb{R}^{N_p \times D}$
2. MLP 将 patch 嵌入投射到问题特定的特征空间 $\boldsymbol{F} \in \mathbb{R}^{N_p \times N_f}$
3. 平均池化得到特征向量 $\boldsymbol{f} = \text{AvgPool}(\boldsymbol{F}) \in \mathbb{R}^{N_f}$
4. BLDD（二值低维决策层）执行稀疏特征分配进行分类

关键设计选择：**丢弃 CLS token，仅使用 patch 嵌入**。这使得每个特征都有对应的空间特征图，实现高保真的空间定位。

### 关键设计

1. **MLP 特征变换**：

    - 功能：将 DINOv2 的 D 维 patch 表示映射到 $N_f$ 个问题特定的特征
    - 核心思路：BLDD 层本身是二值稀疏矩阵，没有表示变换能力，因此需要 MLP 在上游完成特征变换
    - 消融发现：MLP 层数对稠密模型影响不大，但对稀疏 QPM 至关重要——QPM 在多层 MLP 下可超越稠密模型近 10%

2. **二次规划特征选择（QP）**：

    - 功能：从 $N_f$ 个特征中选出 $N_f^* = 50$ 个特征，并为每个类分配 $N_f^c = 5$ 个特征
    - 核心思路：优化三个目标——最大化类与已分配特征激活的相关性、最小化选定特征之间的相似性、最大化偏置以优先局部特征
    - 设计动机：通过数学规划而非学习来进行特征选择，确保特征集的多样性、对比性和紧凑性

3. **平均池化实现空间定位**：

    - 功能：用 AvgPool 代替标准的 CLS token 聚合
    - 核心思路：特征向量的每个维度是对应特征图的空间平均，因此特征图可直接上采样到原始图像分辨率作为显著性图
    - 实验验证：在有 register token 的 DINOv2 上，仅用 patch 嵌入即可达到 88.3% 准确率（超越 CLS 的 87.6%）

4. **特征图稀疏损失 $\mathcal{L}_{\text{L1-FM}}$**：

    - 功能：对特征图施加 L1 正则化
    - 核心思路：迫使特征激活集中在与分类相关的物体区域，抑制背景噪声和空间散射
    - 惊人发现：该损失不仅提升 Plausibility，还显著提升分类准确率（准确率与 Plausibility 高度相关）

### 损失函数 / 训练策略

三阶段训练流程：
1. **稠密训练**：使用交叉熵 $\mathcal{L}_{\text{CE}}$ + 特征多样性损失 $\mathcal{L}_{\text{div}}$ + L1 稀疏损失训练
2. **二次规划**：解 QP 确定特征选择向量 $\boldsymbol{s}$ 和稀疏权重 $\boldsymbol{W}^{\text{sparse}}$
3. **稀疏微调**：固定 $\boldsymbol{W}^{\text{sparse}}$，仅在选定特征上微调

总损失：$\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_{\text{div}} \mathcal{L}_{\text{div}} + \lambda_{\text{L1-FV}} \mathcal{L}_{\text{L1-FV}} + \lambda_{\text{L1-FM}} \mathcal{L}_{\text{L1-FM}}$

## 实验关键数据

### 主实验

| 方法 | CUB Acc↑ | CARS Acc↑ | CUB Plausib.↑ | Contrast.↑ |
|------|---------|----------|--------------|-----------|
| DINOv2 CLS 线性探测 | 87.9 | 91.7 | 42.6 | 59.2 |
| Dense $\boldsymbol{F}^{\text{froz}}$ | 78.1 | 92.9 | 32.7 | 84.5 |
| ResNet50 QPM | 82.9 | 92.1 | 82.9 | 93.6 |
| DINO-SLDD | 84.6 | 92.9 | 78.0 | 93.0 |
| DINO-QSENN | 85.4 | 93.3 | 86.0 | 94.4 |
| **DINO-QPM (本文)** | **88.3** | **94.0** | **95.0** | **100.0** |

DINO-QPM 在准确率上超越不可解释的 DINOv2 线性探测（88.3 vs 87.9），同时 Plausibility 从 42.6 跃升至 95.0。

### 消融实验

| 配置 | CUB Acc(%) | 说明 |
|------|-----------|------|
| CLS + 无 register | 87.3 | CLS token 无空间定位 |
| CLS + register | 87.6 | register 帮助 CLS |
| Patch + 无 register | 83.3 | 无 register 时 patch 表示差 |
| **Patch + register** | **88.3** | register tokens 至关重要 |

| 骨干大小 | CUB Acc(%) | Patch Contextualization↑ |
|---------|-----------|------------------------|
| DINO ViT-B/16 | 37.1 | 8.9 |
| DINOv2 ViT-S/14 Reg | 83.4 | 42.9 |
| DINOv2 ViT-B/14 Reg | 88.3 | 43.9 |
| DINOv2 ViT-L/14 Reg | 86.5 | 2.2 |

### 关键发现

1. **Register token 是关键**：没有 register 时，patch 嵌入的空间信息质量差，准确率下降约 5%。register 使 patch 不再承担全局上下文存储的"异常 token"角色
2. **Plausibility 与准确率高度正相关**：L1-FM 损失同时提升两者，说明迫使模型关注物体区域对分类本身有益
3. **紧凑性 vs 准确率的 trade-off 很小**：将每类特征从 5 减少到 4（Compact 版本），准确率几乎不变
4. **有趣的鸟类分类案例**：模型区分 Brewer's Blackbird 和 Rusty Blackbird 时自动定位到喙部，与鸟类学专家的鉴定依据完全一致

## 亮点与洞察

1. **不可解释的模型不一定更准**：DINO-QPM 用 50 个特征、每类仅 5 个特征的极端稀疏约束，反而超越了使用 768 个特征的线性探测
2. **反直觉的架构选择**：丢弃 CLS token（分类文献的标准选择）反而更好，因为直接从局部证据构建的全局表示比内部预聚合的不透明表示更可解释、也更有效
3. **训练效率极高**：由于骨干完全冻结，可预计算 patch 嵌入，每 epoch 训练仅 6 秒
4. **Plausibility 指标设计合理**：引入膨胀掩码处理 patch 边界效应，避免对精确轮廓上的激活不公平惩罚

## 局限与展望

1. 仅在细粒度分类（CUB-2011、Stanford Cars）上验证，通用图像分类场景待测试
2. ViT-L 骨干效果反而变差，可能需要针对不同骨干大小调整适配器设计
3. 二次规划特征选择是一次性的，不随训练动态调整，可能限制最优特征组合的发现
4. 目前每类固定分配 5 个特征，不同复杂度的类可能需要不同数量的特征

## 相关工作与启发

- **QPM / ChiQPM**：端到端训练的二次规划可解释模型，本文将其迁移到冻结骨干上
- **Post-hoc CBM**：通过文本概念进行可解释分类，但依赖外部语言模型且缺乏空间定位
- **ProtoViT / Zhu et al.**：基于原型的可解释方法，但需要微调骨干进行原型聚类
- 启发：冻结骨干 + 轻量适配器 = 高效可解释方案，这一范式值得在更多视觉任务上探索

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将 QPM 迁移到冻结视觉基础模型上是自然但有效的创新
- 实验充分度: ⭐⭐⭐⭐⭐ — 消融充分，指标设计严谨，跨骨干验证完整
- 写作质量: ⭐⭐⭐⭐⭐ — 数学推导清晰，可视化出色（鸟类特征定位案例尤为精彩）
- 价值: ⭐⭐⭐⭐ — 为冻结基础模型的可解释分类提供了强有力的工具
