---
title: >-
  [论文解读] Principled Data Augmentation for Learning to Solve Quadratic Programming Problems
description: >-
  [NeurIPS 2025][图学习][数据增强] 提出基于KKT系统仿射变换的原则性数据增强框架，为线性规划(LP)和二次规划(QP)的MPNN学习优化(L2O)任务生成保最优性的增强实例，并结合对比学习预训练，在数据稀缺和OOD泛化场景下大幅提升性能。
tags:
  - NeurIPS 2025
  - 图学习
  - 数据增强
  - 二次规划
  - 图神经网络
  - 对比学习
  - 学习优化
  - KKT条件
---

# Principled Data Augmentation for Learning to Solve Quadratic Programming Problems

**会议**: NeurIPS 2025  
**arXiv**: [2506.01728](https://arxiv.org/abs/2506.01728)  
**作者**: Chendi Qian, Christopher Morris (RWTH Aachen University)  
**代码**: [GitHub](https://github.com/chendiqian/Data-Augmentation-for-Learning-to-Optimize)  
**领域**: graph_learning  
**关键词**: 数据增强, 二次规划, 图神经网络, 对比学习, 学习优化, KKT条件  

## 一句话总结

提出基于KKT系统仿射变换的原则性数据增强框架，为线性规划(LP)和二次规划(QP)的MPNN学习优化(L2O)任务生成保最优性的增强实例，并结合对比学习预训练，在数据稀缺和OOD泛化场景下大幅提升性能。

## 研究背景与动机

### 问题背景
线性规划和二次规划是机器学习、运筹学和科学计算中的基础优化问题。近年来，基于消息传递图神经网络(MPNN)的学习优化(L2O)方法受到广泛关注，它们将LP/QP实例编码为约束-变量二部图，通过数据驱动方式学习求解或辅助求解器（如替代强分支评分计算）。

### 已有工作的不足
- **数据稀缺问题**：L2O方法需要大量标注数据（最优解），而获取标注成本高昂，尤其对复杂QP问题
- **通用图增强方法不适用**：GraphCL等方法的节点删除、边扰动、特征掩码等操作会破坏LP/QP的约束结构和最优性条件，导致性能下降
- **缺乏针对优化问题的增强策略**：现有L2O工作几乎未探索如何为LP/QP设计保结构、保最优性的数据增强方法
- **对比预训练空白**：虽然图对比学习在一般图任务上成功，但在LP/QP的L2O场景中缺少有效的无监督预训练策略

### 核心动机
设计一套理论上有保证的数据增强方法，能在不重新求解的情况下生成结构多样且保最优性的新LP/QP实例，同时支持监督学习和对比预训练两种训练范式。

## 方法详解

### 核心思想：KKT系统的仿射变换
给定线性约束二次规划(LCQP)实例 $I = (\mathbf{Q}, \mathbf{A}, \mathbf{b}, \mathbf{c})$，其最优原始-对偶解满足KKT条件，可紧凑写为：

$$\begin{bmatrix} \mathbf{Q} & \mathbf{A}^\top \\ \mathbf{A} & \mathbf{0} \end{bmatrix} \begin{bmatrix} \mathbf{x}^* \\ \boldsymbol{\lambda}^* \end{bmatrix} = \begin{bmatrix} -\mathbf{c} \\ \mathbf{b} - \mathbf{s}^* \end{bmatrix}$$

核心观察：对KKT系统施加仿射变换 $\mathbf{M}$（乘法项）和 $\mathbf{B}, \boldsymbol{\beta}$（偏置项），可以生成新的有效KKT系统，从而构造新的QP实例并直接通过矩阵运算恢复其最优解，无需调用求解器。

### 变换有效性条件
为使变换后的系统仍为有效KKT形式，需满足：
1. $\mathbf{T}_{11} = \mathbf{M}_1 \mathbf{Q} \mathbf{M}_1^\top$ 正定（新目标函数凸性）
2. $\mathbf{T}_{22} = \mathbf{0}$（约束结构一致性）
3. $\mathbf{T}_{12} = \mathbf{T}_{21}^\top$（对称性）
4. $\mathbf{M}_{22}$ 满足Proposition 2.1的条件（原始解可恢复）

### 六种具体变换

| 变换方法 | 操作 | 高效可恢复 | 解无关 |
|---------|------|:---------:|:-----:|
| 移除空闲变量 | 删除最优解为零的变量 | ✓ | ✗ |
| 移除非活跃约束 | 删除严格非活跃的不等式约束 | ✓ | ✗ |
| 缩放变量系数 | 对角矩阵 $\mathbf{M}_1$ 缩放目标和约束中的变量系数 | ✓ | ✓ |
| 缩放约束 | 对角矩阵 $\mathbf{M}_2$ 缩放约束系数和右端项 | ✓ | ✓ |
| 添加变量 | 通过 $\mathbf{M}_1 = [\mathbf{I}; \mathbf{q}^\top]$ 引入新变量 | ✓ | ✓ |
| 添加约束 | 向问题中添加新的不等式约束 | ✓ | ✓ |

其中"高效可恢复"指新解可在 $O(n)$ 时间内从原解计算得到；"解无关"指变换参数不依赖原始最优解，可用于无监督设定。

### 训练范式

**监督学习增强**：在训练过程中动态应用变换生成增强实例，用监督损失训练MPNN预测目标值。

**对比预训练**：使用NT-Xent损失进行自监督预训练。对每个实例应用解无关变换生成两个增强视图作为正对，其他实例作为负样本。预训练只训练MPNN骨干网络，随后接MLP头进行监督微调。

## 实验关键数据

### 实验1：监督学习下的数据增强效果

在合成LP/QP数据集（100变量×100约束，10000实例）上评估不同增强方法。评价指标为平均相对目标误差(%)。

| 增强方法 | LP 10% | LP 20% | LP 100% | QP 10% | QP 20% | QP 100% |
|---------|--------|--------|---------|--------|--------|---------|
| 无增强 | 8.539 | 6.149 | 2.784 | 5.304 | 3.567 | 1.240 |
| Drop node | 8.618 | 7.131 | 4.654 | 6.355 | 4.867 | 2.594 |
| Mask node | 8.979 | 7.591 | 3.216 | 5.865 | 4.005 | 1.347 |
| Flip edge | 7.794 | 6.503 | 4.358 | 5.485 | 4.361 | 2.561 |
| **Drop vars (本文)** | 4.996 | 3.484 | 1.484 | 4.232 | 2.374 | 0.835 |
| **Drop cons (本文)** | 5.563 | 3.691 | 1.656 | **3.407** | **2.104** | 0.821 |
| **Scale vars (本文)** | **4.682** | **3.208** | **1.241** | 3.790 | 2.468 | **0.649** |
| **Combo (本文)** | **3.465** | **2.300** | **1.051** | **2.434** | **1.532** | **0.542** |

通用图增强方法(Drop/Mask/Flip)反而损害性能；本文方法组合使用(Combo)在LP 20%设定下相对改进达62.6%。

### 实验2：对比预训练+微调效果

使用全部无标注训练集进行对比预训练，然后在不同比例标注数据上微调。

| 预训练方法 | LP 10% | LP 20% | LP 100% | QP 10% | QP 20% | QP 100% |
|-----------|--------|--------|---------|--------|--------|---------|
| 无预训练 | 8.539 | 6.149 | 2.784 | 5.304 | 3.567 | 1.240 |
| GraphCL | 9.694 | 6.033 | 2.613 | 6.024 | 3.865 | 1.253 |
| GCC | 8.248 | 5.761 | 2.686 | 11.344 | 6.356 | 1.472 |
| IGSD | 18.097 | 7.631 | 3.082 | 12.799 | 6.306 | 1.302 |
| MVGRL | 20.392 | 9.072 | 2.852 | 8.440 | 4.932 | 1.343 |
| **本文** | **3.472** | **2.794** | **1.588** | **3.791** | **2.427** | **0.926** |

本文预训练在LP 10%设定下将误差从8.539降至3.472（59.4%改进），QP 10%设定下从5.304降至3.791（28.5%改进）。现有图对比学习方法(IGSD, MVGRL等)大多导致性能劣化。

### 实验3：OOD泛化能力

预训练模型在未见过的优化问题族上微调，验证迁移学习能力。

**LP OOD泛化（Set Cover/MIS/CA/CFL）**：
| 问题族 | 预训练 | 10% | 20% | 100% |
|-------|:------:|-----|-----|------|
| Set Cover | 否 | 5.297 | 2.531 | 0.752 |
| Set Cover | **是** | **1.965** | **1.349** | **0.662** |
| CA | 否 | 2.381 | 1.803 | 0.906 |
| CA | **是** | **2.303** | **1.713** | **0.753** |

**QP OOD泛化（SVM/Portfolio/LASSO）**：
| 问题族 | 预训练 | 10% | 20% | 100% |
|-------|:------:|-----|-----|------|
| SVM | 否 | 0.191 | 0.109 | 0.027 |
| SVM | **是** | **0.141** | **0.077** | 0.024 |
| LASSO | 否 | 5.405 | 4.083 | 2.159 |
| LASSO | **是** | **5.169** | **3.726** | **1.178** |

预训练模型在几乎所有OOD任务上优于从头训练，展现出强迁移能力。

## 亮点

- **理论优雅**：基于KKT系统的仿射变换框架具有统一的数学基础，所有变换都有严格的理论保证（保最优性、保凸性），区别于启发式图增强方法
- **实用性强**：六种变换操作均可在 $O(n)$ 时间内完成解恢复，无需重新调用QP求解器；解无关变换可直接用于无监督预训练
- **大幅改进**：组合增强在数据稀缺场景下将误差降低超60%；对比预训练在LP/QP上均显著优于GraphCL等通用方法
- **OOD泛化**：预训练模型成功迁移到Set Cover、SVM、Portfolio等不同问题族，证明学到的是优化问题的通用结构表示
- **反面证据有力**：实验清楚展示通用图增强方法(节点删除、边扰动等)在LP/QP任务上的负面效果，强调了领域特定增强的必要性

## 局限与展望

- **仅限LCQP**：框架要求 $\mathbf{Q}$ 正定，不适用于非凸QP、半定规划(SDP)或混合整数规划(MILP)的整数部分
- **变换多样性有限**：实际可用的解无关变换仅4种（缩放变量/约束、添加变量/约束），移除操作需要解信息
- **图回归任务单一**：仅验证了目标值预测任务，未在更实际的L2O应用场景（如强分支评分、切割平面选择）上测试
- **问题规模较小**：实验固定为100变量×100约束，未充分验证在大规模实例（数千变量）上的扩展性
- **预训练策略简单**：仅使用NT-Xent对比损失，未探索更先进的SSL方法（如BYOL、VICReg等）
- **变换组合策略**：Combo采用随机组合，缺乏自适应选择机制来根据问题实例特性选择最优变换子集

## 与相关工作的对比

- **GraphCL (You et al., 2020)**：通用图增强（节点删除/边扰动/特征掩码），在LP/QP上反而损害性能，因为这些操作破坏了优化问题的约束结构
- **GCC (Qiu et al., 2020)**：基于随机游走的子图采样，在QP上导致严重性能下降（11.344 vs 5.304）
- **Li et al. (2024c)**：唯一针对LP的对比方法，采用CLIP风格公式，本文框架更通用（涵盖LP和QP）且变换有理论保证
- **Duan et al. (2022)**：为SAT问题设计满足性保持增强，类似思路但本文针对连续优化问题
- **Gasse et al. (2019)**：开创性的MPNN用于MILP强分支，但未考虑数据增强
- **Chen et al. (2024a)**：将LP的二部图表示扩展到QP（变量间加边），本文采用该图表示作为MPNN输入

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次系统性地为LP/QP的L2O设计基于KKT理论的数据增强框架
- 实验充分度: ⭐⭐⭐⭐ — 覆盖监督/预训练/OOD三类场景，基线对比全面，但问题规模偏小
- 写作质量: ⭐⭐⭐⭐⭐ — 理论推导清晰，变换分类系统化，实验设计合理
- 价值: ⭐⭐⭐⭐ — 填补LP/QP数据增强空白，方法简洁实用，但应用场景需进一步拓展

<!-- RELATED:START -->

## 相关论文

- [Toward Data-centric Directed Graph Learning: An Entropy-driven Approach](../../ICML2025/graph_learning/toward_data-centric_directed_graph_learning_an_entropy-driven_approach.md)
- [Inductive Transfer Learning for Graph-Based Recommenders](inductive_transfer_learning_for_graph-based_recommenders.md)
- [Learning Repetition-Invariant Representations for Polymer Informatics](learning_repetition-invariant_representations_for_polymer_informatics.md)
- [MoEMeta: Mixture-of-Experts Meta Learning for Few-Shot Relational Learning](moemeta_mixture-of-experts_meta_learning_for_few-shot_relational_learning.md)
- [Sketch-Augmented Features Improve Learning Long-Range Dependencies in Graph Neural Networks](sketch-augmented_features_improve_learning_long-range_dependencies_in_graph_neur.md)

<!-- RELATED:END -->
