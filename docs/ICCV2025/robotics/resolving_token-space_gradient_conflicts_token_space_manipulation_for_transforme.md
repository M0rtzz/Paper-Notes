---
title: >-
  [论文解读] Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning
description: >-
  [ICCV 2025][机器人][多任务学习] 提出 DTME-MTL 框架，通过在 token 空间中识别和分类梯度冲突（值域空间冲突 vs 零空间冲突），分别采用 Token Modulation（仿射变换）和 Token Expansion（添加任务特定token）来缓解 Transformer 多任务学习中的负迁移问题，以极低参数开销实现一致性能提升。
tags:
  - ICCV 2025
  - 机器人
  - 多任务学习
  - 梯度冲突
  - Token空间
  - Transformer
  - 动态网络扩展
---

# Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning

**会议**: ICCV 2025  
**arXiv**: [2507.07485](https://arxiv.org/abs/2507.07485)  
**代码**: [GitHub](https://github.com/wooseong97/DTME-MTL)  
**领域**: 机器人  
**关键词**: 多任务学习, 梯度冲突, Token空间, Transformer, 动态网络扩展

## 一句话总结

提出 DTME-MTL 框架，通过在 token 空间中识别和分类梯度冲突（值域空间冲突 vs 零空间冲突），分别采用 Token Modulation（仿射变换）和 Token Expansion（添加任务特定token）来缓解 Transformer 多任务学习中的负迁移问题，以极低参数开销实现一致性能提升。

## 研究背景与动机

多任务学习（MTL）通过在共享网络中同时学习多个任务来提高泛化性和效率，但不同任务的目标差异可能导致**负迁移**（negative transfer），即一个任务的学习会降低另一个任务的性能。基于预训练 Transformer 的 MTL 架构（如 Task Prompter、MoE）虽然具有强大的泛化能力，但其**固定的网络容量**和**刚性结构**限制了自适应能力。

现有方法的局限性：

**多任务优化方法**（如 PCGrad、Nash-MTL）通过调整任务损失权重或修改梯度来缓解负迁移，但受限于固定网络设计，无法扩展模型容量。

**动态网络架构**（如 Recon）直接将共享参数转化为任务特定参数，但这种方式在 Transformer 中会导致参数效率低下、计算开销过大、过拟合风险增加。
3. 简单增大 Transformer backbone 的方法无法利用预训练网络的优势，需要从头训练大网络，计算成本极高。

**核心洞察**：与其在参数空间直接操作（容易过拟合），不如在 token 空间进行操作——将 token 视为可学习参数，通过 SVD 分析 token 空间结构，识别梯度冲突类型并自适应解决。这种方式更加高效，且能够避免参数级别操作带来的过拟合问题。

## 方法详解

### 整体框架

DTME-MTL（Dynamic Token Modulation and Expansion）是一个即插即用的框架，可应用于任何基于 Transformer 的 MTL 架构。核心流程为：

1. 对每一层 Transformer 的共享 token 计算非中心协方差矩阵
2. 通过 SVD 分解定义 token 空间的值域空间和零空间
3. 将任务梯度投影到这两个子空间，检测不同类型的梯度冲突
4. 根据冲突类型，自适应地应用 Token Modulation 或 Token Expansion

### 关键设计

1. **基于 SVD 的 Token 空间定义**:

    - 功能：构建共享 token 的向量空间，为梯度冲突分类提供数学基础
    - 核心思路：对第 $d$ 层的共享 token $\mathcal{T}_s^{l,d}$，计算非中心协方差矩阵 $\widetilde{\mathcal{T}}_s^d = \frac{1}{n}\sum_{l=1}^{n}(\mathcal{T}_s^{l,d})(\mathcal{T}_s^{l,d})^T$，然后进行 SVD 分解 $\widetilde{\mathcal{T}}_s^d = \mathcal{U}\Lambda\mathcal{V}^T$，根据方差比例阈值 $r$ 将特征值分为值域空间 $\Lambda_\mathcal{R}$ 和零空间 $\Lambda_\mathcal{N}$
    - 设计动机：实际中 token 协方差矩阵的特征值不会精确为零，因此采用 SVD 中的方差比例准则 $r = \frac{\sum_{\lambda \in \Lambda_\mathcal{N}} \lambda}{\sum_{\lambda \in \Lambda_\mathcal{R}} \lambda}$ 自动划分两个空间边界

2. **梯度冲突分类与投影**:

    - 功能：将任务梯度分解到值域空间和零空间，分别检测两种类型的冲突
    - 核心思路：对任务 $\tau_i$ 的梯度 $g_i = \nabla_{\mathcal{T}_{s,k}} \mathcal{L}_i$，通过投影矩阵分解为值域分量 $g_{\mathcal{R},i} = (\mathcal{U}_\mathcal{R}\mathcal{U}_\mathcal{R}^T)g_i$ 和零空间分量 $g_{\mathcal{N},i} = (\mathcal{U}_\mathcal{N}\mathcal{U}_\mathcal{N}^T)g_i$；当 $g_{\mathcal{R},i} \cdot g_{\mathcal{R},j} \leq 0$ 时为值域空间冲突，当 $g_{\mathcal{N},i} \cdot g_{\mathcal{N},j} \leq 0$ 时为零空间冲突
    - 设计动机：在迁移学习场景中，预训练权重使模型限制在损失景观的同一盆地中，值域空间的冲突说明网络已有相关解释能力但需要旋转/缩放，零空间冲突说明需要新特征来增强模型容量

3. **Token Modulation（值域空间冲突解决）**:

    - 功能：当值域空间存在梯度冲突时，为冲突的任务对添加任务特定的仿射变换调制器
    - 核心思路：调制器 $\mathcal{M}$ 对共享 token 进行仿射变换 $W \odot \mathcal{T}_{s,i} + b$，其中 $W, b \in \mathbb{R}^p$。理论保证（Proposition 1）：当输入 token 跨越值域空间时，优化调制器可减少值域空间的梯度冲突并降低多任务损失
    - 设计动机：值域空间的冲突意味着现有特征已经包含相关信息，只需通过轻量级的通道维度变换即可为不同任务提供差异化的特征表示

4. **Token Expansion（零空间冲突解决）**:

    - 功能：当零空间存在梯度冲突时，引入新的任务特定 token 来扩展特征空间
    - 核心思路：将任务特定 token $\{\mathcal{T}_i\}_{i=1}^\mathcal{K}$ 与共享 token 拼接后送入 Transformer block，扩展注意力计算范围。理论保证（Proposition 2）：当输入 token 跨越零空间时，token 扩展可缓解零空间梯度冲突导致的多任务损失增加
    - 设计动机：零空间冲突表明网络缺乏必要的特征维度来区分不同任务的需求，需要通过增加 token 来提供额外的信息通道

### 损失函数 / 训练策略

- 使用标准的多任务加权损失：$\Theta^* = \arg\min_\Theta \sum_{i=1}^\mathcal{K} w_i \mathcal{L}_i(\Theta_s, \Theta_i)$
- 网络扩展的最佳时机：实验表明在训练早期进行扩展效果最优，这与利用预训练backbone的设计一致
- SVD 和梯度冲突计算作为一次性预处理步骤，计算成本相对较低（ViT-L 约 1 小时）

## 实验关键数据

### 主实验

**NYUD-v2（4任务）与 PASCAL-Context（5任务）消融实验：**

| 方法 | NYUD Semseg mIoU↑ | NYUD Depth RMSE↓ | PASCAL Semseg mIoU↑ | PASCAL Normal mErr↓ | 参数增量 |
|------|-------------------|------------------|--------------------|--------------------|---------|
| Baseline (ST) | 39.35 | 0.6611 | 67.96 | 15.65 | - |
| Baseline (MT) | 34.13 | 0.6732 | 54.47 | 16.22 | - |
| TM | 37.85 | 0.6490 | 64.28 | 15.40 | 0.24% |
| TE | 37.25 | 0.6553 | 60.51 | 15.55 | 0.30% |
| TM+TE | 38.27 | 0.6370 | 66.18 | 15.26 | 0.30% |

**Taskonomy（11任务）与 SOTA 优化方法对比：**

| 方法 | 多任务性能 △m↑ | 说明 |
|------|-------------|------|
| GD | -7.83% | 梯度下降基线 |
| PCGrad | -8.29% | 梯度投影方法 |
| Nash-MTL | -5.01% | 纳什博弈方法 |
| FAMO | -7.87% | 损失平衡方法 |
| DTME-MTL | **+4.67%** | 仅增加0.118%参数 |

### 消融实验

| 配置 | NYUD △m↑ | PASCAL △m↑ | 说明 |
|------|---------|-----------|------|
| TM+TE (最高冲突层) | 0.044 | -1.289 | 基于冲突检测选层（最优） |
| TM+TE (随机选层) | 低于最优 | 低于最优 | 随机选择层进行扩展 |
| TM+TE (最低冲突层) | 最差 | 最差 | 反向选层验证策略有效性 |

**应用于 SOTA MTL 方法（NYUD-v2）：**

| 基线方法 | Semseg mIoU↑ | + DTME-MTL | Depth RMSE↓ | + DTME-MTL |
|---------|-------------|-----------|------------|-----------|
| InvPT | 53.56 | **54.38** | 0.5183 | **0.5020** |
| Taskprompter | 55.30 | **56.36** | 0.5152 | **0.5122** |

### 关键发现

1. 仅增加 0.2-0.3% 参数即可将多任务性能从负迁移状态提升至接近单任务基线
2. 基于冲突检测的层选择策略显著优于随机选择，验证了token空间梯度冲突分析的有效性
3. Recon等直接在参数空间操作的方法在 Transformer 上严重过拟合，而 DTME-MTL 通过token空间操作避免了这一问题
4. 训练早期进行网络扩展效果最佳，与预训练模型的特性一致

## 亮点与洞察

- **理论贡献突出**：将梯度冲突分解为值域空间和零空间两种类型，并分别给出了数学保证（Proposition 1&2），为操作提供了理论依据
- **即插即用设计**：可无缝集成到现有 SOTA 多任务 Transformer 架构中（InvPT、Taskprompter），无需修改基础架构
- **效率极高**：参数增量不到 0.3%，推理时间增加约 13.4%，SVD 预处理约 1 小时
- **深层洞察**：揭示了参数空间的梯度冲突并非总是负迁移的可靠指标，token 空间的分析提供了更有效的视角

## 局限与展望

1. 方差比例阈值 $r$ 需要手动设定，可能影响值域/零空间的划分质量
2. SVD 计算需要对完整训练集进行前向传播，对于超大规模数据集可能存在效率问题
3. 目前仅聚焦于视觉密集预测任务，是否能泛化到 NLP 多任务场景有待验证
4. Token Expansion 增加的注意力计算量与任务数成正比，在任务数很多时可能带来显著开销

## 相关工作与启发

- 与 Recon 的对比揭示了一个重要发现：在 Transformer 中直接将共享参数转为任务特定参数会导致过拟合，这为后续动态网络架构设计提供了重要警示
- Token Modulation 的仿射变换思路与 LoRA 等参数高效微调方法有内在联系，可探索其在其他场景中的应用
- 将 SVD 分析从传统的特征降维工具提升为梯度冲突检测手段，具有方法论上的创新意义

## 评分

- 新颖性: ⭐⭐⭐⭐ 将梯度冲突分析从参数空间转移到token空间是全新视角，但仿射变换和token拼接本身是成熟技术
- 实验充分度: ⭐⭐⭐⭐⭐ 三个benchmark，11个任务的Taskonomy，大量消融和对比实验
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，但数学符号较多，可读性有提升空间
- 价值: ⭐⭐⭐⭐ 即插即用的设计使其具有很高的实用价值，且理论分析为多任务学习提供了新的理解角度

<!-- RELATED:START -->

## 相关论文

- [Beyond Losses Reweighting: Empowering Multi-Task Learning via the Generalization Perspective](beyond_losses_reweighting_empowering_multi-task_learning_via_the_generalization_.md)
- [Domain Expansion: A Latent Space Construction Framework for Multi-Task Learning](../../ICLR2026/robotics/domain_expansion_a_latent_space_construction_framework_for_multi-task_learning.md)
- [Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos](moto_latent_motion_token_as_the_bridging_language_for_learning_robot_manipulatio.md)
- [Rep-MTL: Unleashing the Power of Representation-Level Task Saliency for Multi-Task Learning](rep-mtl_unleashing_the_power_of_representation-level_task_saliency_for_multi-tas.md)
- [TransiT: Transient Transformer for Non-line-of-sight Videography](transit_transient_transformer_for_non-line-of-sight_videography.md)

<!-- RELATED:END -->
