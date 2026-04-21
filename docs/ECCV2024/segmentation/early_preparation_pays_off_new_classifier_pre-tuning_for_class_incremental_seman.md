---
title: >-
  [论文解读] Early Preparation Pays Off: New Classifier Pre-tuning for Class Incremental Semantic Segmentation
description: >-
  [ECCV 2024][图像分割][类增量语义分割] 提出NeST（New claSsifier pre-Tuning）方法，在正式训练前通过学习从所有旧分类器到新分类器的线性变换来初始化新分类器权重，并设计基于跨任务类别相似性的变换矩阵初始化策略，在Pascal VOC和ADE20K上显著提升多种CISS方法的性能。
tags:
  - ECCV 2024
  - 图像分割
  - 类增量语义分割
  - 新分类器预调优
  - 灾难性遗忘
  - 稳定性-可塑性权衡
  - 知识迁移
---

# Early Preparation Pays Off: New Classifier Pre-tuning for Class Incremental Semantic Segmentation

**会议**: ECCV 2024  
**arXiv**: [2407.14142](https://arxiv.org/abs/2407.14142)  
**代码**: https://github.com/zhengyuan-xie/ECCV24_NeST (有)  
**领域**: 图像分割  
**关键词**: 类增量语义分割, 新分类器预调优, 灾难性遗忘, 稳定性-可塑性权衡, 知识迁移

## 一句话总结
提出NeST（New claSsifier pre-Tuning）方法，在正式训练前通过学习从所有旧分类器到新分类器的线性变换来初始化新分类器权重，并设计基于跨任务类别相似性的变换矩阵初始化策略，在Pascal VOC和ADE20K上显著提升多种CISS方法的性能。

## 研究背景与动机

**领域现状**：类增量语义分割（CISS）要求模型在连续学习新类别的同时保留旧类别知识，面临灾难性遗忘和背景偏移两大挑战。现有方法主要通过知识蒸馏（MiB、PLOP）、伪标签、特征正则化等方式缓解遗忘。

**现有痛点**：

**新分类器初始化不当**：随机初始化导致新分类器与backbone特征不对齐，训练初期梯度剧烈变化破坏旧知识

**背景分类器初始化的局限**：MiB用背景分类器初始化新分类器，但会导致真实背景像素误分类

**辅助分类器训练的偏差**：SSUL/DKD预训练辅助分类器，但缺少真实未来数据导致偏差

**忽视新类别间差异**：现有方法对所有新类别使用相同初始化策略，忽视不同新类别与旧类别的关联差异

**AWT的限制**：AWT通过梯度归因从背景分类器转移权重，但忽略其他旧分类器，且内存开销巨大（>24GB）

**核心矛盾**：如何初始化新分类器使其既能快速适应新类别数据（可塑性），又不对backbone造成剧烈更新从而破坏旧知识（稳定性）？

**切入角度**：不直接调参新分类器，而是学习一个从所有旧分类器到新分类器的线性变换——利用重要性矩阵和投影矩阵实现的通道级加权+旧分类器线性组合。

**核心idea**：在正式训练之前插入一个预调优阶段（pre-tuning），冻结backbone仅训练变换矩阵，使生成的新分类器自然对齐backbone特征空间，而基于跨任务类别相似性的矩阵初始化确保了稳定性-可塑性的权衡。

## 方法详解

### 整体框架
NeST在每个增量步骤的正式训练前增加预调优阶段：
1. 基于跨任务类别相似性初始化重要性矩阵和投影矩阵
2. 冻结backbone和旧分类器，仅训练变换矩阵
3. 用训练好的矩阵从旧分类器生成新分类器权重
4. 用生成的权重初始化新分类器，移除额外参数，进入正式训练

### 关键设计

1. **新分类器生成（New Classifier Generation）**：
   对每个新类别 $c \in \mathcal{C}_t$，分配两个可学习矩阵：
    - **重要性矩阵** $\mathbf{M}_c \in \mathbb{R}^{d \times n_{old}}$：学习旧分类器每个通道对新类别的重要性权重
    - **投影矩阵** $\mathbf{P}_c \in \mathbb{R}^{n_{old} \times 1}$：学习旧分类器的线性组合系数
   
   新分类器权重生成公式：
    $\mathbf{w}_c = (\mathbf{M}_c \odot \mathbf{W}_{old}) \mathbf{P}_c$
   其中 $\mathbf{W}_{old} \in \mathbb{R}^{d \times n_{old}}$ 是所有旧分类器权重。$\odot$ 为Hadamard积实现通道级加权，$\mathbf{P}_c$ 完成旧分类器的加权组合。
   
   对背景类也学习变换以适应新任务：$\hat{\mathbf{w}}_0 = (\mathbf{M}_0 \odot \mathbf{w}_0) \mathbf{P}_0$

2. **预调优过程（Pre-tuning）**：

    - 冻结backbone和旧分类器权重，仅优化 $\{\mathbf{M}_c, \mathbf{P}_c\}$
    - 每次前向传播先由矩阵生成新分类器权重，拼接到旧分类器后进行预测
    - 使用unbiased cross-entropy损失避免过拟合：
    $\mathcal{L}_{unce} = -\frac{1}{|\mathcal{I}|} \sum_{i \in \mathcal{I}} \log \tilde{q}_x^t(i, y_i)$
    - 预调优完成后用最终矩阵生成新分类器权重作为初始化，移除额外参数

3. **跨任务类别相似性初始化（Cross-Task Class Similarity）**：

    - 将当前步训练图像送入旧模型，对每个新类别像素embedding $\mathbf{p}_u$，分解分类过程：
    $\mathbf{H}_u = \mathbf{W}_{old} \odot \mathbf{p}_u', \quad \mathbf{s}_u = \text{softmax}(\text{sum}(\mathbf{H}_u))^\top$
    - 正值位置标记为贡献通道：$\mathbf{H}_u^{mask}(i,j) = \mathbb{1}[\mathbf{H}_u(i,j) > 0]$
    - 对新类别所有像素取平均得到重要性矩阵初值：
    $\mathbf{M}_{c_{new}} = \frac{1}{N} \sum_{\mathbf{p}_u} \mathbf{H}_u^{mask} \odot \mathbf{s}_u'$
    - 投影矩阵由重要性矩阵沿通道维求和+softmax得到：
    $\mathbf{P}_{c_{new}} = (\text{softmax}(\text{sum}(\mathbf{M}_{c_{new}})))^\top$
    - 核心思想：如果某旧类别与新类别更相似（旧模型对新类别像素预测该旧类偏高），则在变换中赋予更大初始权重

### 损失函数 / 训练策略
- 预调优阶段：unbiased cross-entropy，VOC预调优5 epochs（lr=0.001），ADE20K预调优15 epochs
- 正式训练：沿用基线方法的损失函数（如MiB的知识蒸馏+unbiased CE）
- SGD优化器，初始步lr=0.02，增量步lr=0.001，poly衰减
- 预调优额外开销极小：仅增加约5.4K参数和2.7%训练时间

## 实验关键数据

### 主实验

**Pascal VOC 2012（Overlapped设定，mIoU %）**：

| 方法 | Backbone | 10-1 (all) | 15-1 (all) | 15-5 (all) | 19-1 (all) |
|------|----------|-----------|-----------|-----------|-----------|
| MiB | Res101 | 10.2 | 38.2 | 70.2 | 69.6 |
| MiB+NeST | Res101 | **37.4** (+27.2) | **51.8** (+13.6) | 70.7 | 69.7 |
| PLOP | Res101 | 32.2 | 56.2 | 70.8 | 74.0 |
| PLOP+NeST | Res101 | **36.9** (+4.7) | **63.1** (+6.9) | **72.4** | **75.7** |
| RCIL | Res101 | 33.1 | 58.9 | 72.5 | 74.5 |
| RCIL+NeST | Res101 | **36.8** (+3.7) | **61.4** (+2.5) | **72.8** | **74.9** |
| MiB | Swin-B | 15.0 | 36.9 | 77.3 | 78.3 |
| MiB+NeST | Swin-B | **51.2** (+36.2) | **71.4** (+34.5) | **77.9** | **78.8** |

**ADE20K（mIoU %）**：

| 方法 | 100-50 (all) | 100-10 (all) | 100-5 (all) | 50-50 (all) |
|------|-------------|-------------|-------------|-------------|
| MiB+NeST | **35.1** | **33.7** (+4.4) | **32.7** (+6.8) | **33.2** |
| PLOP+NeST | **36.3** (+3.0) | **34.7** (+2.7) | **32.0** (+2.8) | **34.8** |

### 消融实验

**分类器初始化策略比较（VOC 15-1）**：

| 初始化方法 | 0-15 mIoU | 16-20 mIoU | all mIoU |
|-----------|----------|----------|---------|
| 随机初始化 | 43.5 | 4.2 | 34.1 |
| 背景分类器 (MiB) | 45.2 | 15.7 | 38.2 |
| Two-Stage直接调优 | 46.0 | 15.3 | 38.7 |
| **NeST (Ours)** | **61.7** | **20.4** | **51.8** |

**组件级消融（VOC 15-1）**：

| 投影矩阵 | 重要性矩阵 | 0-15 | 16-20 | all |
|---------|----------|------|-------|-----|
| ✗ | ✗ | 45.2 | 15.7 | 38.2 |
| ✓ | ✗ | 53.3 | 15.4 | 44.3 |
| ✗ | ✓ | 59.7 | 19.2 | 50.1 |
| ✓ | ✓ | **61.7** | **20.4** | **51.8** |

**变换矩阵初始化策略**：

| 初始化 | 0-15 | 16-20 | all |
|--------|------|-------|-----|
| 随机矩阵初始化 | 53.8 | 7.5 | 42.8 |
| 跨任务相似性初始化 | **61.7** | **20.4** | **51.8** |

### 关键发现
- **预调优显著缩小stability gap**：特征余弦相似性在整个恢复过程中高于baseline，损失从训练一开始就更低
- **重要性矩阵贡献大于投影矩阵**：单独使用重要性矩阵（50.1）远优于单独使用投影矩阵（44.3），说明通道级区分比旧分类器组合更关键
- **矩阵初始化策略至关重要**：随机初始化矩阵虽提升旧类（53.8），但新类崩溃（7.5）；跨任务相似性初始化实现了稳定性-可塑性的平衡
- **极低计算开销**：仅增加约5.4K参数，训练时间增加2.7%（3.7分钟/2.3小时），内存仅6.47GB vs AWT的>24GB
- **跨backbone通用**：在ResNet-101和Swin-B上均有效，在Swin-B上提升更加显著

## 亮点与洞察
- **"学变换而非学权重"的思路优雅**：不直接调新分类器参数，而是学习旧→新的映射，天然继承旧知识
- **双矩阵设计（重要性+投影）**巧妙分解了"哪些通道重要"和"如何组合旧分类器"两个问题
- **预调优阶段冻结backbone**确保了不在初始化阶段就破坏旧知识，同时让新分类器适应数据
- **通用即插即用**：可直接应用于 MiB/PLOP/RCIL 等现有方法，无需修改原方法的训练流程

## 局限与展望
- 引入了额外预调优阶段的计算开销（虽然很小但非零）
- 当旧类别数量很多时，重要性矩阵的规模会线性增长
- 仅在语义分割任务上验证，可扩展到实例分割、全景分割等增量学习场景
- 矩阵初始化依赖旧模型的预测质量，如果旧模型本身退化严重，相似性估计可能不准确

## 相关工作与启发
- MiB：首个系统解决背景偏移问题的CISS方法 → NeST在此基础上改进分类器初始化
- AWT：梯度归因选择相关权重 → NeST用学习替代归因，更灵活且内存友好
- RCIL：重参数化用于持续分割 → NeST作为正交组件可叠加使用
- EWF：权重融合策略 → 与NeST的预调优思路互补
- 启发：预调优思想可推广到其他增量学习场景（分类、检测），核心是"先让新组件适配旧特征空间"

## 评分
- 新颖性: ⭐⭐⭐⭐ 从"学变换"而非"学权重"的角度解决分类器初始化，跨任务相似性初始化设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 两个数据集、多种设定（10-1/15-1/15-5/19-1等）、适配三种baseline、ResNet和Swin双backbone、完整消融
- 写作质量: ⭐⭐⭐⭐ 逻辑链清晰，从问题观察到方法设计到实验验证环环相扣
- 价值: ⭐⭐⭐⭐⭐ 即插即用的通用改进策略，开销极小效果显著，对CISS社区有直接实用价值

<!-- RELATED:START -->

## 相关论文

- [Cs2K: Class-Specific and Class-Shared Knowledge Guidance for Incremental Semantic Segmentation](cs2k_class-specific_and_class-shared_knowledge_guidance_for_incremental_semantic.md)
- [Learning from the Web: Language Drives Weakly-Supervised Incremental Learning for Semantic Segmentation](learning_from_the_web_language_drives_weakly-supervised_incremental_learning_for.md)
- [CPM: Class-Conditional Prompting Machine for Audio-Visual Segmentation](cpm_class-conditional_prompting_machine_for_audio-visual_segmentation.md)
- [DreamLIP: Language-Image Pre-training with Long Captions](dreamlip_language-image_pre-training_with_long_captions.md)
- [Efficient and Versatile Robust Fine-Tuning of Zero-shot Models](efficient_and_versatile_robust_fine-tuning_of_zero-shot_models.md)

<!-- RELATED:END -->
