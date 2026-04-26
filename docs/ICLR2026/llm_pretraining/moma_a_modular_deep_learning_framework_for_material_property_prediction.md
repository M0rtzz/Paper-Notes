---
title: >-
  [论文解读] MoMa: A Simple Modular Deep Learning Framework for Material Property Prediction
description: >-
  [ICLR 2026][材料属性预测] 提出 MoMa 模块化材料属性预测框架，先在多任务上训练专用模块并集中存储为 MoMa Hub，再通过表示驱动的无训练自适应模块组合算法（AMC）为下游任务定制模型，在 17 个数据集上平均超越最强基线 14%。
tags:
  - ICLR 2026
  - 材料属性预测
  - 模块化学习
  - 自适应组合
  - 迁移学习
  - 图神经网络
---

# MoMa: A Simple Modular Deep Learning Framework for Material Property Prediction

**会议**: ICLR 2026  
**arXiv**: [2502.15483](https://arxiv.org/abs/2502.15483)  
**代码**: [https://github.com/GenSI-THUAIR/MoMa](https://github.com/GenSI-THUAIR/MoMa)  
**领域**: 材料科学 / 模块化深度学习  
**关键词**: 材料属性预测, 模块化学习, 自适应组合, 迁移学习, 图神经网络

## 一句话总结
提出 MoMa 模块化材料属性预测框架，先在多任务上训练专用模块并集中存储为 MoMa Hub，再通过表示驱动的无训练自适应模块组合算法（AMC）为下游任务定制模型，在 17 个数据集上平均超越最强基线 14%。

## 研究背景与动机

**领域现状**：深度学习材料属性预测主要有两条路线：(1) 专用模型从头训练（如 CGCNN）；(2) 预训练-微调范式，尤其是大规模力场模型（如 JMP）在势能面数据上预训练后微调到下游任务。后者在多个任务上取得显著成功。

**现有痛点**：(1) **多样性**：材料任务涉及多种体系（晶体、有机分子）和属性（热稳定性、电子行为、力学性质），力场模型仅在势能面相关属性上预训练，泛化受限；(2) **差异性**：不同材料属性由不同物理定律支配，多任务联合训练导致知识冲突，单一模型难以兼顾。

**核心矛盾**：预训练范式追求"一个模型解决所有问题"，但材料任务的内在差异性使得联合训练反而引入负迁移。

**本文目标** (1) 如何利用多样化材料数据的知识而避免任务间冲突？(2) 如何在下游数据稀缺时高效适配？

**切入角度**：模块化学习——将每个任务封装为独立模块训练避免干扰，下游时自适应选择和组合最有协同效应的模块。

**核心 idea**：训练多样化任务专用模块存入 MoMa Hub，通过 kNN 表示空间估计 + 凸优化权重求解实现无训练自适应模块组合，再微调适配下游任务。

## 方法详解

### 整体框架
阶段一（Module Training & Centralization）：在 18 个高资源材料属性预测数据集上分别训练模块（Full 或 Adapter），存入 MoMa Hub。阶段二（AMC & Fine-tuning）：对新任务，AMC 算法三步走——(1) kNN 预测估计各模块亲和度 → (2) 凸优化求最优组合权重 → (3) 权重空间模块合并 → 微调。

### 关键设计

1. **MoMa Hub 模块训练与存储**:

    - 做什么：为每个高资源材料任务训练独立模块并集中管理
    - 核心思路：以预训练 JMP 为骨干，提供两种模块形式——Full Module（全模型微调作为模块）和 Adapter Module（冻结骨干仅训练插入的 adapter 层，参数高效）。模块集合 $\mathcal{H} = \{g_1, g_2, \ldots, g_N\}$，当前包含 18 个来自 Matminer 的任务模块
    - 设计动机：独立训练避免任务间知识冲突；Hub 设计保护数据隐私（只共享模型参数不共享数据），支持社区贡献

2. **Adaptive Module Composition (AMC)**:

    - 做什么：为下游任务自适应选择和加权组合最优模块
    - 核心思路：三步流程——① 对每个模块 $g_j$，编码下游数据得到表示 $\mathcal{X}^j$，用 leave-one-out kNN 标签传播估计预测 $\hat{y}_i^j = \sum_{k \in \mathcal{N}_i} \frac{f_d(\mathbf{x}_i^j, \mathbf{x}_k^j)}{Z_i^j} y_k$；② 最小化集成代理误差 $E_\mathcal{D}(\mathbf{w}) = \frac{1}{M}\|\sum_j w_j \hat{\mathbf{y}}^j - \mathbf{y}\|_2^2$，约束 $\sum w_j = 1, w_j \geq 0$，凸优化求解；③ 权重空间合并 $g_\mathcal{D} = \sum_j w_j^* g_j$
    - 设计动机：搜索式方法在高差异性模块上误差信号嘈杂，路由网络在数据稀缺时过拟合。AMC 使用表示质量代替预测误差做监督，无需额外训练参数，30 秒内收敛

3. **权重空间模块合并与微调**:

    - 做什么：将加权模块合并为单一初始化模型，再在下游数据上微调
    - 核心思路：利用线性模式连通性（linear mode connectivity）：所有模块来自同一预训练初始化，参数空间兼容。合并后加任务特定头，在 $\mathcal{D}$ 上微调至收敛
    - 设计动机：直接加权平均简单高效，避免推理时加载多个模块的计算开销

### 损失函数 / 训练策略
模块训练使用标准 MAE 损失。AMC 阶段无训练（凸优化求解权重），代理误差与最终微调 MAE 的 Pearson 相关 > 0.6（经验验证）。微调使用下游任务标准设置。实验在 17 个低数据材料属性预测任务上评估，每个任务 5 个数据分割 × 5 个随机种子。

## 实验关键数据

### 主实验（17 个材料属性预测任务）

| 方法 | 平均排名 | Best/17 | 代表性任务 (3D Band Gap MAE) |
|--------|------|------|----------|
| MoMa (Full) | 1.35 | 14 | 0.200 |
| MoMa (Adapter) | 2.59 | 2 | 0.245 |
| JMP-FT | 3.12 | 1 | 0.249 |
| JMP-MT | 4.53 | 0 | 0.423 |
| UMA | 4.53 | 0 | 0.268 |
| MoE-(18) | 4.71 | 0 | 0.361 |
| CGCNN | 6.88 | 0 | 0.492 |

### 消融实验（AMC 组合策略对比）

| 配置 | vs AMC (MAE 增幅) | 说明 |
|------|---------|------|
| AMC (完整) | — | 最优 |
| Select Average | +11.0% | 保留 AMC 选的模块但均匀平均 |
| All Average (Model Soup) | +18.0% | 平均所有模块 |
| Random Selection | +20.2% | 随机选等数量模块 |

### 关键发现
- MoMa (Full) 在 16/17 任务上位列前二，平均提升 14% vs JMP-FT，24.8% vs JMP-MT
- 少样本场景优势更大：10-shot 下 MoMa 归一化 MAE 0.550 vs JMP-FT 0.700（-21%）
- Hub 扩展性：从 5 模块到 30 模块，归一化 MAE 从 0.204 单调下降到 0.176
- AMC 权重提供材料属性关系洞察：如介电常数预测高权重分配给带隙模块（物理直觉一致）
- Orb-v2 骨干上同样有效（13/17 任务提升，平均 +6.1%），验证骨干无关性

## 亮点与洞察
- 模块化范式巧妙解决了材料科学中"多样性 vs 差异性"的矛盾：独立训练避免冲突，自适应组合利用协同，两全其美
- AMC 的表示驱动 + 无训练设计特别适合数据稀缺的材料科学场景，代理误差的理论和经验验证令人信服

## 局限与展望
- MoMa Hub 目前仅包含 18 个晶体任务 + 12 个 QM9 分子任务，覆盖面仍有限
- 权重空间合并依赖线性模式连通性假设，当模块极度分化时可能失效
- AMC 的 kNN 估计在极低数据（<10 样本）时可能不稳定

## 相关工作与启发
- **vs JMP**: JMP 用多任务联合预训练（MT），MoMa 证明独立训练 + 自适应组合优于 MT（16/17 任务），暗示多任务损失在异质材料任务上引入严重负迁移
- **vs Model Soup**: 简单平均所有模块表现差（+18% MAE），AMC 的自适应权重优化至关重要
- **vs MoE-(18)**: 路由网络方法在 17 个任务上排名 4.71，AMC 的表示驱动策略更稳定

## 评分
- 新颖性: ⭐⭐⭐⭐ 模块化学习引入材料科学领域是新范式，AMC 算法设计合理
- 实验充分度: ⭐⭐⭐⭐⭐ 17 任务 × 5 分割 × 5 种子，消融/少样本/扩展性/多骨干全面覆盖
- 写作质量: ⭐⭐⭐⭐ 框架图清晰，实验叙述完整，动机铺垫充分
- 价值: ⭐⭐⭐⭐ 开源平台设计有望推动材料科学模块化知识共享

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Intrinsic Training Dynamics of Deep Neural Networks](intrinsic_training_dynamics_of_deep_neural_networks.md)
- [\[CVPR 2025\] A Unified Framework for Heterogeneous Semi-supervised Learning](../../CVPR2025/llm_pretraining/a_unified_framework_for_heterogeneous_semi-supervised_learning.md)
- [\[ICML 2025\] Algebra Unveils Deep Learning -- An Invitation to Neuroalgebraic Geometry](../../ICML2025/llm_pretraining/algebra_unveils_deep_learning_--_an_invitation_to_neuroalgebraic_geometry.md)
- [\[ICLR 2026\] Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning](pre-training_llm_without_learning_rate_decay_enhances_supervised_fine-tuning.md)
- [\[ICML 2025\] Position: The Future of Bayesian Prediction Is Prior-Fitted](../../ICML2025/llm_pretraining/position_the_future_of_bayesian_prediction_is_prior-fitted.md)

<!-- RELATED:END -->
