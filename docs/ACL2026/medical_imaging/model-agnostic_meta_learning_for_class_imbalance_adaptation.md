---
title: >-
  [论文解读] Model-Agnostic Meta Learning for Class Imbalance Adaptation
description: >-
  [ACL 2026][医学图像][类不平衡] 本文提出 HAMR（Hardness-Aware Meta-Resample），一个统一的元学习框架，通过双层优化动态估计实例级权重优先处理真正困难的样本，配合邻域感知重采样机制将训练焦点放在困难样本及其语义邻居上，在 6 个不平衡 NLP 数据集上持续超越强基线。
tags:
  - ACL 2026
  - 医学图像
  - 类不平衡
  - 元学习
  - 自适应权重
  - 难度感知重采样
  - 双层优化
---

# Model-Agnostic Meta Learning for Class Imbalance Adaptation

**会议**: ACL 2026  
**arXiv**: [2604.18759](https://arxiv.org/abs/2604.18759)  
**代码**: [GitHub](https://github.com/trust-nlp/ImbalanceLearning)  
**领域**: NLP通用 / 类不平衡学习  
**关键词**: 类不平衡, 元学习, 自适应权重, 难度感知重采样, 双层优化

## 一句话总结

本文提出 HAMR（Hardness-Aware Meta-Resample），一个统一的元学习框架，通过双层优化动态估计实例级权重优先处理真正困难的样本，配合邻域感知重采样机制将训练焦点放在困难样本及其语义邻居上，在 6 个不平衡 NLP 数据集上持续超越强基线。

## 研究背景与动机

**领域现状**：类不平衡在文本分类、命名实体识别等 NLP 任务中普遍存在。现有方法主要分为两类：损失重加权（如 Focal Loss、Dice Loss）和数据重采样（过采样/合成生成）。

**现有痛点**：(1) 这些方法通常依赖预定义的静态启发式——对同一类别内的所有样本施加相同的调整比率；(2) 样本难度并不等同于类别归属——并非所有少数类实例都天然困难，也并非所有多数类样本都平凡；(3) 静态方案可能错误地降低有信息量的多数类样本权重，同时过度强调简单的少数类实例。

**核心矛盾**：需要一种能动态识别和优先处理真正困难样本的方法——不论其类别归属如何，根据模型不断演化的数据理解来调整学习策略。

**本文目标**：设计一个统一框架，同时解决类不平衡和实例级难度两个问题，动态引导模型学习焦点。

**切入角度**：将"模型应该关注什么"（自适应权重）和"模型应该看到什么"（重采样）解耦为两个互补模块，通过元学习框架统一优化。

**核心 idea**：用双层元优化动态学习实例重要性权重（内层用 pre-meta 权重做中间更新，外层在平衡元验证集上更新权重网络，得到 post-meta 权重做实际更新），配合基于 FAISS 的邻域增强重采样，将训练分布偏移到困难语义区域。

## 方法详解

### 整体框架

HAMR 包含两个核心模块：(1) 自适应权重估计——通过轻量权重网络 $f_\theta$ 将归一化的样本损失映射为重要性权重，通过双层元优化动态调整；(2) 难度感知区域重采样——基于 EMA 平滑的难度分数和 KNN 邻域增强，动态调整训练分布。两个模块在统一的训练循环中协同工作。

### 关键设计

1. **双层元优化自适应权重 (Adaptive Weight Estimation)**:

    - 功能：根据模型当前学习状态动态调整每个训练样本的重要性
    - 核心思路：内层用当前权重网络的 pre-meta 权重 $w_i^{\text{pre}}$ 做中间梯度更新得到 $\phi'$，外层在平衡元验证集 $\mathcal{D}_{\text{meta}}$ 上评估 $f_{\phi'}$ 并更新权重网络参数 $\theta$，然后用更新后的权重网络重新计算 post-meta 权重 $w_i^{\text{post}}$ 做实际模型更新。对token级任务用句子级最大token损失，对分类任务用样本交叉熵
    - 设计动机：pre-meta 权重反映的是"更新前模型认为什么重要"，post-meta 权重反映的是"在平衡验证集指导下什么才真正重要"——这种先试后改的策略比静态启发式更能适应训练动态

2. **邻域增强难度感知重采样 (Hardness-Aware Region Resampling)**:

    - 功能：动态调整训练分布，将模型更多地暴露给困难语义区域
    - 核心思路：用 EMA 平滑 post-meta 权重得到全局难度分数 $h_i \leftarrow \gamma \cdot h_i + (1-\gamma) \cdot w_i^{\text{post}}$。选择 top 20% 困难样本，用 FAISS 加速的 KNN 找到每个困难样本的 k 个语义邻居，计算邻域增强分数 $b_i$。最终采样概率 $p_i \propto (h_i + \varepsilon)^\tau \cdot (1 + \lambda b_i)$，温度 $\tau < 1$ 鼓励平衡探索
    - 设计动机：仅关注孤立的困难样本不够——困难样本的语义邻居往往也具有相似的挑战性。邻域增强将难度从单个样本扩散到整个语义区域

3. **统一训练循环**:

    - 功能：将权重估计和重采样无缝集成为端到端训练过程
    - 核心思路：每 F 个 epoch 更新一次邻域增强（避免每步都做 KNN 的开销），每个 mini-batch 经过采样→pre-meta权重→内层更新→元步→post-meta权重→外层更新→EMA更新的完整流程
    - 设计动机：两个模块相互增强——权重估计提供实例级重要性信号，重采样确保模型能看到足够多的困难区域样本

### 损失函数 / 训练策略

主损失为加权交叉熵/token级损失。元验证集通过取验证集全量+按中位数类别数从训练集补采样构建平衡集。权重经 batch-wise z-score 归一化后由权重网络处理，输出裁剪到固定范围确保数值稳定。

## 实验关键数据

### 主实验

| 数据集 | 任务 | HAMR Macro-F1 | 最佳基线 Macro-F1 | 提升 |
|--------|------|-------------|----------------|------|
| BioNLP | NER | 72.7 | 70.6 (Dice) | +2.1 |
| TweetNER | NER | 60.2 | 59.0 (Dice/LNR) | +1.2 |
| MIT-Restaurant | NER | 81.1 | 80.4 (Dice) | +0.7 |
| Hurricane-Irma17 | CLS | 73.4 | 72.7 (ICF) | +0.7 |
| Cyclone-Idai19 | CLS | 65.7 | 63.8 (ICF) | +1.9 |
| SST-5 | CLS | 57.0 | 56.3 (ICF) | +0.7 |

### 消融实验

| 配置 | BioNLP F1 | Irma17 F1 | 说明 |
|------|----------|----------|------|
| HAMR (完整) | 72.7 | 73.4 | 完整模型 |
| w/o 重采样 | 71.4 | 72.1 | 去掉邻域重采样 |
| w/o 元权重 | 70.9 | 71.8 | 去掉自适应权重 |
| w/o 邻域增强 | 71.8 | 72.5 | 去掉KNN邻域boost |

### 关键发现

- HAMR 在所有 6 个数据集上均取得最佳 Macro-F1，且在高不平衡比数据集（Cyclone-Idai19 IR=98.4）上优势最大（+1.9pp）
- 两个模块协同贡献——单独去除任一模块都导致性能下降，但元权重的贡献略大于重采样
- 邻域增强提供了一致的边际提升（+0.6-0.9pp），证明了从单点难度到区域难度扩散的价值

## 亮点与洞察

- 双层元优化的"先试后改"策略很优雅——pre-meta 权重像是"草稿"，元验证集的反馈让权重网络学会什么样的权重分配真正有利于泛化。这个思路可迁移到任何需要动态调整训练优先级的场景
- 邻域增强重采样的思路有独到之处——将困难样本视为"种子"，通过语义邻域扩散难度信号，比只关注孤立困难样本更鲁棒
- 统一框架的设计解耦了"关注什么"和"看到什么"——权重决定how to learn，重采样决定what to learn from

## 局限与展望

- 依赖 FAISS 做 KNN，对非常大的数据集可能有计算瓶颈
- 元验证集的构建依赖于合理的类别分布假设
- 仅在 BERT 为基础的编码器上验证，LLM 时代的适用性未知
- 未探索与合成数据增强方法的结合

## 相关工作与启发

- **vs Focal Loss/Dice Loss**: 静态启发式不区分实例难度，HAMR 动态学习实例权重
- **vs Meta-Weight-Net**: 类似元学习框架但缺乏邻域重采样，HAMR 增加了区域级训练分布调整
- **vs SMOTE**: 合成新样本而非动态调整现有样本权重，HAMR 更轻量且无生成噪声

## 评分

- 新颖性: ⭐⭐⭐⭐ 双层元优化+邻域重采样的组合新颖，但各组件有先例
- 实验充分度: ⭐⭐⭐⭐ 6数据集2任务+详细消融，但缺少与更多最新方法的对比
- 写作质量: ⭐⭐⭐⭐ 方法清晰，算法伪代码完整
- 价值: ⭐⭐⭐⭐ 对NLP类不平衡问题提供了通用且有效的解决方案

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Meta-Learning an In-Context Transformer Model of Human Higher Visual Cortex](../../NeurIPS2025/medical_imaging/meta-learning_an_in-context_transformer_model_of_human_higher_visual_cortex.md)
- [\[CVPR 2025\] Unsupervised Foundation Model-Agnostic Slide-Level Representation Learning](../../CVPR2025/medical_imaging/unsupervised_foundation_model-agnostic_slide-level_representation_learning.md)
- [\[CVPR 2026\] Meta-learning In-Context Enables Training-Free Cross Subject Brain Decoding](../../CVPR2026/medical_imaging/meta-learning_in-context_enables_training-free_cross_subject_brain_decoding.md)
- [\[ICCV 2025\] Toward Long-Tailed Online Anomaly Detection through Class-Agnostic Concepts](../../ICCV2025/medical_imaging/toward_long-tailed_online_anomaly_detection_through_class-agnostic_concepts.md)
- [\[CVPR 2026\] SCDL: Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](../../CVPR2026/medical_imaging/semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)

<!-- RELATED:END -->
