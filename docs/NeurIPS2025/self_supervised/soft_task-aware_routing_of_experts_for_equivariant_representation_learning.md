---
title: >-
  [论文解读] Soft Task-Aware Routing of Experts for Equivariant Representation Learning
description: >-
  [NeurIPS 2025][自监督学习][equivariant representation] 提出 STAR（Soft Task-Aware Routing），通过 MoE 路由机制协调不变性和等变性表示学习任务间的共享与专属信息，减少冗余特征学习，提升下游任务迁移性能。
tags:
  - "NeurIPS 2025"
  - "自监督学习"
  - "equivariant representation"
  - "mixture of experts"
  - "redundant feature learning"
  - "routing"
---

# Soft Task-Aware Routing of Experts for Equivariant Representation Learning

**会议**: NeurIPS 2025  
**arXiv**: [2510.27222](https://arxiv.org/abs/2510.27222)  
**代码**: [https://github.com/YonseiML/star](https://github.com/YonseiML/star)  
**领域**: Self-Supervised Learning  
**关键词**: equivariant representation, mixture of experts, self-supervised learning, redundant feature learning, routing

## 一句话总结

提出 STAR（Soft Task-Aware Routing），通过 MoE 路由机制协调不变性和等变性表示学习任务间的共享与专属信息，减少冗余特征学习，提升下游任务迁移性能。

## 研究背景与动机

自监督学习（SSL）通过在未标注数据上学习表示已成为主流范式。其中**不变性表示学习**（如 SimCLR）让不同增强视图映射到相同表示以保留语义，**等变性表示学习**（如 EquiMod）则让表示捕捉增强变换引起的结构化变化。近期研究发现，联合学习两种表示通常对下游任务有益。

**现有痛点**：

**冗余特征学习**：当前方法（如 EquiMod）使用两个独立的投影头分别处理不变性和等变性目标，隐含假设两个任务独立。但实际上它们是内在相关的——理解语义类别有助于推断光照方向，理解光照也有助于判断语义（"陨石坑错觉"例子：同一月球表面因光照不同看起来像陨石坑或穹顶）

**信息浪费**：独立投影头导致两个分支重复捕获共享信息，造成模型容量的低效利用

**梯度质量差**：冗余特征学习使投影头收敛慢，导致传递到骨干网络的梯度信号质量不佳

**核心 idea**：将投影头视为"专家"，引入 MoE 路由机制，让模型自适应地分配专家到不变性/等变性/共享任务，从而减少冗余、提升专家专业化和梯度质量。

## 方法详解

### 整体框架

给定图像 $x$，生成两个增强视图 $v = T(x; a)$ 和 $v' = T(x; a')$，通过共享编码器 $f$ 提取特征，然后经过 STAR 投影模块生成不变性嵌入 $z^{\text{inv}}$ 和等变性嵌入 $z^{\text{eq}}$。不变性分支用 InfoNCE 损失让同一图像不同视图的嵌入相似，等变性分支用等变预测器 $\phi_T$ 预测增强引起的嵌入偏移。

### 关键设计

1. **Single Shared Projection（STAR-SS）**：

    - **功能**：最简单的共享信息建模
    - **核心思路**：定义三个专家——不变专家 $E^{\text{inv}}$、等变专家 $E^{\text{eq}}$、共享专家 $E^{\text{sh}}$。嵌入计算为：$z_i^{\text{inv}} = E^{\text{inv}}(f(v_i)) + E^{\text{sh}}(f(v_i))$，$z_i^{\text{eq}} = E^{\text{eq}}(f(v_i)) + E^{\text{sh}}(f(v_i))$
    - **设计动机**：共享专家因同时参与两个目标的优化而自然学到两个任务都需要的信息。但权重固定（等权加法），不够灵活

2. **MMoE Projection（STAR-MMoE）**：

    - **功能**：自适应地将专家路由到不同任务
    - **核心思路**：$N$ 个共享专家 $\{E_k\}_{k=1}^N$ + 两个任务特定路由器 $R^{\text{inv}}$ 和 $R^{\text{eq}}$。路由器计算 softmax 权重：$s_{i,k}^{\text{inv}} = \text{softmax}_k(R^{\text{inv}}(f(v_i)))$。嵌入为专家输出的加权和：$z_i^{\text{inv}} = \sum_k s_{i,k}^{\text{inv}} E_k(f(v_i))$
    - **设计动机**：不同图像、不同任务对共享/专属信息的需求不同，软路由允许动态分配。STAR-SS 是 STAR-MMoE 的退化情况（所有权重相等）
    - **关键约束**：必须用软路由（soft routing），因为稀疏路由（top-k）会导致 batch normalization 不稳定

3. **等变学习设计**：

    - **功能**：建模增强在嵌入空间中引起的偏移
    - **核心思路**：$\hat{z}_i^{\text{eq}} = z_i^o + \phi_T(z_i^o, \psi(a_i))$，其中 $z_i^o$ 是原始图像的等变嵌入，$\psi$ 将增强参数投影到嵌入空间，$\phi_T$ 是 3 层 MLP 等变预测器
    - **残差连接**的设计保证语义内容被保留，同时有效建模变换偏移
    - **等变损失**采用 InfoNCE 形式，预测嵌入 $\hat{z}^{\text{eq}}$ 与目标嵌入 $z^{\text{eq}}$ 构成正样本对

### 损失函数 / 训练策略

- 总损失：$\mathcal{L} = \mathcal{L}^{\text{inv}} + \lambda \mathcal{L}^{\text{eq}}$，$\lambda = 1$，温度 $\tau = 0.2$
- 不变性损失用 SimCLR 的 InfoNCE 损失
- 预训练完成后**丢弃投影头**，仅迁移编码器到下游任务，完全消除 MoE 的可迁移性限制
- STL10 用 16 个专家训练 200 epochs，ImageNet100 用 8 个专家训练 500 epochs，batch size 256

## 实验关键数据

### 主实验

**跨域分类（ImageNet100 预训练 ResNet-50 → 11 个下游数据集）**：

| 方法 | CIFAR10 | CIFAR100 | Food | Flowers | Cars | 均值 | 平均排名 |
|------|---------|----------|------|---------|------|------|---------|
| SimCLR | 87.88 | 67.92 | 63.60 | 88.37 | 47.09 | 68.42 | 5.00 |
| EquiMod | 88.99 | 70.22 | 64.43 | 90.33 | 48.94 | 69.72 | 3.18 |
| **STAR-MMoE** | **90.09** | **72.31** | **67.05** | **91.45** | **51.54** | **71.23** | **1.18** |

STAR-MMoE 在 10/11 个数据集上排名第一，均值提升 1.51%（vs EquiMod）。

**STL10 预训练 ResNet-18 → 下游任务**：

| 方法 | 均值 | 平均排名 |
|------|------|---------|
| SimCLR | 45.55 | 5.73 |
| EquiMod | 49.54 | 3.82 |
| **STAR-MMoE** | **53.07** | **1.36** |

STAR-MMoE 均值提升 3.53%（vs EquiMod）。

**目标检测（VOC07+12, Faster R-CNN, 冻结 ResNet-50-C4 backbone）**：

| 方法 | AP | AP50 | AP75 |
|------|-----|------|------|
| SimCLR | 47.96 | 76.35 | 51.62 |
| EquiMod | 48.52 | 76.55 | 52.82 |
| **STAR-MMoE** | **48.85** | **76.81** | **53.01** |

### 消融实验

| 专家数量 | 平均典型相关 | 均值准确率 | 说明 |
|---------|------------|----------|------|
| 2 (≈EquiMod) | ~0.55 | ~48.5% | 冗余特征学习严重 |
| 4 | ~0.42 | ~50.5% | 冗余减少 |
| 8 | ~0.35 | ~52.0% | 进一步改善 |
| 16 | ~0.30 | ~53.0% | 冗余最小，性能最佳 |

**等变性评估**：

| 方法 | R-equiv. ↑ | P-equiv. ↓ |
|------|------------|------------|
| SimCLR | 0.74 | 0.72 |
| EquiMod | 0.91 | 0.38 |
| **STAR-MMoE** | **0.98** | **0.27** |

STAR 在等变性度量上全面超越所有方法。

### 关键发现

1. **专家自然分工**：8 个专家中，Expert 1 被两个路由器均衡使用（共享专家），Experts 2-6 主要服务不变性目标，Experts 7-8 主要服务等变性目标
2. **冗余与泛化负相关**：典型相关越低（冗余越少），下游准确率越高，呈现明确的正相关
3. **收敛速度更快**：MMoE 投影中的专家比 EquiMod 的投影头收敛更快，提供更高质量的梯度信号
4. **等变专家 vs 不变专家的 kNN 检索**确认了各专家确实在捕获不同类型的信息

## 亮点与洞察

1. **冗余特征学习的诊断与解决**：首次明确指出不变性和等变性学习中投影头的冗余问题，并用典型相关分析量化验证
2. **MoE 用于 SSL 预训练的新视角**：将 MoE 限制在投影头中，训练后丢弃，巧妙解决了 MoE 模型的迁移性问题
3. **理论动机清晰**："陨石坑错觉"的直觉例子很好地解释了不变性/等变性任务的内在相关性
4. **一致性**：在分类、检测、少样本学习三类下游任务上都有稳定提升

## 局限与展望

- 只能用软路由，无法利用稀疏路由（top-k）来提升效率和扩展性，因为批归一化需要所有专家在每个 batch 都接收输入
- 专家数量的选择需要超参数搜索
- 目前仅在 SimCLR 框架下验证，与其他 SSL 框架（BYOL、DINO 等）的兼容性未探索
- 仅在 ResNet-18/50 上实验，更大模型和 ViT 的效果未知

## 相关工作与启发

- **EquiMod (2023)**：STAR 的直接基线，用独立投影头分别处理不变/等变目标
- **MMoE (2018)**：STAR 借鉴了多门 MoE 的多任务学习思路，但创新性地应用于 SSL 预训练的投影头中
- **V-MoE (2022) & Neural Experts (2024)**：MoE 在视觉中的应用，但 STAR 的关键区别在于训练后丢弃 MoE 结构
- **启发**：将多任务学习的工具（如 MoE）引入 SSL 的投影头设计是一个有潜力的方向

## 评分

- 新颖性: ⭐⭐⭐⭐ 冗余特征学习的视角新颖，MoE 用于 SSL 投影头的设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 分类/检测/少样本 + 专家分析 + 冗余分析 + 等变性评估，非常全面
- 写作质量: ⭐⭐⭐⭐ 动机清晰，分析部分有深度
- 价值: ⭐⭐⭐⭐ 对等变表示学习有实际推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking](starformer_semi-supervised_task-informed_representation_learning_via_dynamic_att.md)
- [\[ICML 2025\] Global Context-aware Representation Learning for Spatially Resolved Transcriptomics](../../ICML2025/self_supervised/global_context-aware_representation_learning_for_spatially_resolved_transcriptom.md)
- [\[NeurIPS 2025\] SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery](seal_semantic-aware_hierarchical_learning_for_generalized_category_discovery.md)
- [\[ICML 2025\] MTL-UE: Learning to Learn Nothing for Multi-Task Learning](../../ICML2025/self_supervised/mtl-ue_learning_to_learn_nothing_for_multi-task_learning.md)
- [\[ICLR 2026\] Soft Equivariance Regularization for Invariant Self-Supervised Learning](../../ICLR2026/self_supervised/soft_equivariance_regularization_for_invariant_self-supervised_learning.md)

</div>

<!-- RELATED:END -->
