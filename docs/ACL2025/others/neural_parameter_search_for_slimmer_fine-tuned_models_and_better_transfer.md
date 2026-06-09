---
title: >-
  [论文解读] Neural Parameter Search for Slimmer Fine-Tuned Models and Better Transfer
description: >-
  [ACL2025][Task Vector] 提出Neural Parameter Search (NPS)，通过在task vector的低秩子空间中搜索最优权重系数来提升微调模型的剪枝效率，在知识迁移（+1.5%）、模型融合（+2.1%）和压缩（40%效率提升）三个场景下均取得显著改进。
tags:
  - "ACL2025"
  - "Task Vector"
  - "剪枝"
  - "Evolutionary Search"
  - "Knowledge Transfer"
  - "Model Merging"
---

# Neural Parameter Search for Slimmer Fine-Tuned Models and Better Transfer

**会议**: ACL2025  
**arXiv**: [2505.18713](https://arxiv.org/abs/2505.18713)  
**代码**: [NPS-Pruning](https://github.com/duguodong7/NPS-Pruning)  
**领域**: 其他  
**关键词**: Task Vector, Model Pruning, Evolutionary Search, Knowledge Transfer, Model Merging  

## 一句话总结

提出Neural Parameter Search (NPS)，通过在task vector的低秩子空间中搜索最优权重系数来提升微调模型的剪枝效率，在知识迁移（+1.5%）、模型融合（+2.1%）和压缩（40%效率提升）三个场景下均取得显著改进。

## 研究背景与动机

- **微调模型的冗余性**：微调后的模型参数修改存在大量冗余，不同task vector子空间对性能贡献差异很大
- **剪枝对知识管理的三重价值**：
  1. 减少微调模型与预训练模型在知识迁移时的冲突，增强抗灾难性遗忘能力
  2. 减少多个微调模型融合时的参数干扰，提升多任务泛化
  3. 降低存储成本，同时保持多任务性能
- **现有剪枝方法不足**：
    - TIES根据幅度直接剪枝，未考虑子空间贡献的不均匀性
    - DARE随机选择和缩放参数，缺乏精细控制
    - Model Tailor基于显著性和敏感性的掩码生成，计算开销较大
- **核心观察**：task vector中不同幅度区间的参数子空间对模型性能的贡献显著不同（Figure 2），需要更精细的重加权策略

## 方法详解

### 整体框架

NPS的核心思路：
1. 计算task vector τ = θ_ft - θ_pre（微调参数与预训练参数的差）
2. 按参数幅度将τ分为M个子空间
3. 用进化算法搜索每个子空间的最优权重
4. 基于重加权后的幅度进行剪枝
5. 将剪枝后的task vector应用于三个场景

### Task Vector子空间分解

将task vector τ按参数幅度排序后分为M个独立子空间：

$$\tau = \sum_{m=1}^{M} q_m$$

然后为每个子空间分配可学习权重：

$$\tau = \sum_{m=1}^{M} w_m * q_m$$

初始时所有权重设为1，后续通过优化搜索最优权重组合。

### 进化搜索（CMA-ES）

使用Covariance Matrix Adaptive Evolution Strategies (CMA-ES)进行优化：
- 无需梯度计算，轻量高效
- 通过协方差矩阵动态调整搜索分布
- 基于校准数据集上的验证accuracy更新权重
- 收敛后得到最优权重{w₁,...,wM}

### 幅度剪枝

搜索完成后，按调整后参数的幅度进行剪枝：

$$m_d = \begin{cases} 1, & \text{if } \tau_d \geq \text{sorted}(\tau)[r \times d] \\ 0, & \text{otherwise} \end{cases}$$

最终剪枝模型：$\hat{\theta}_{ft} = \theta_{pre} + m \odot \tau$

其中r为稀疏率（保留参数比例）。

### 三个应用场景

**1. 知识迁移**：缓解灾难性遗忘

$$\hat{\theta}_{ft} = \theta_{pre} + \lambda \cdot m \odot \tau$$

通过剪枝减少微调模型与预训练模型的干扰，λ控制迁移强度。

**2. 知识融合**：多任务模型合并

$$\theta_m = \theta_{pre} + \sum_{i=1}^{n}(\lambda_i \cdot m_i \odot \tau_i) / \sum_{i=1}^{n}\lambda_i$$

各任务的剪枝task vector加权平均，λᵢ用进化策略优化。

**3. 知识压缩**：高效存储

$$\hat{\theta}_{ft_1}, ..., \hat{\theta}_{ft_n} = \theta_{pre} + [m_1 \odot \tau_1, ..., m_n \odot \tau_n]$$

只需存储预训练模型+稀疏task vector+二值掩码，大幅降低存储成本。

## 实验

### 知识迁移：LLaVA-1.5 (Vicuna-7B), 10%稀疏率

| 方法 | Avg | H-score | 修改参数量 |
|------|-----|---------|-----------|
| Zero-shot | 42.33 | 29.05 | - |
| Fine-tune | 56.42 | 63.40 | 2.7B |
| DARE | 60.12 | 36.64 | 273M |
| Grafting | 61.56 | 60.03 | 273M |
| Model Tailor | 61.87 | 66.94 | 273M |
| **NPS** | **62.38** | **67.54** | **273M** |

NPS在仅修改273M参数（10%稀疏率）的情况下，Avg和H-score均超越所有基线，有效缓解灾难性遗忘。

### 知识融合：多场景模型合并

| 设置 | Task Arithmetic | TIES | Consensus TIES | **NPS** |
|------|----------------|------|----------------|---------|
| T5-Base (7任务NLP) | 73.0 | 73.6 | 73.4 | **75.7(+2.1)** |
| T5-Large (7任务NLP) | 80.2 | 80.3 | 80.5 | **82.1(+1.6)** |
| (IA)³ (11 PEFT任务) | 63.9 | 66.8 | 66.6 | **68.2(+1.4)** |
| LLaMa2 (3 LLM任务) | 30.4 | 34.2 | 34.4 | **35.3(+0.9)** |
| ViT-B/32 (8视觉任务) | 70.1 | 73.6 | 73.3 | **76.5(+3.0)** |
| ViT-L/14 (8视觉任务) | 84.5 | 86.0 | 86.2 | **87.6(+1.4)** |
| T5-Base (5情感域) | 33.6 | 34.5 | 34.4 | **35.7(+1.3)** |
| RoBERTa (5情感域) | 38.3 | 39.7 | 39.8 | **40.9(+0.9)** |

NPS在所有8个实验设置中均取得最优结果，跨越NLP、视觉、多模态和情感分析等不同模态与任务。

### 知识压缩

- 在ViT-B/32的8个视觉任务上测试不同任务组合数量的压缩效果
- NPS在所有任务数量级别（2-8任务）上保持近原始精度
- 与基线方法相比，压缩效率提升约40%
- 稀疏率低至0.04时NPS仍保持准确率，而TIES和DARE在0.2以下急剧下降

### 剪枝效率对比

- 当稀疏率>0.2时，大多数方法保持与微调模型相当的性能
- 稀疏率<0.2时准确率急剧下降
- NPS在稀疏率低至0.04时仍能保持原始模型准确率
- 对低稀疏率的容忍能力远超TIES、DARE等基线

## 亮点与洞察

1. **核心观察的价值**：不同子空间贡献不均匀（Figure 2）是整篇工作的基石，观察简洁但启发深刻
2. **无梯度搜索**：使用CMA-ES进化算法搜索权重，避免了梯度计算的开销，使方法适用于超大模型
3. **一法三用**：同一个NPS方法自然地应用于知识迁移、融合和压缩三个场景，体现了方法的通用性
4. **低稀疏率鲁棒性**：在仅保留4%参数时仍能保持精度，说明重加权让剪枝决策变得更精确
5. **跨模态验证**：NLP(T5, LLaMa2)、视觉(ViT, CLIP)、多模态(LLaVA)全面覆盖
6. **与PEFT兼容**：(IA)³实验证明NPS也适用于参数高效微调的adapter合并

## 局限性

1. **搜索开销**：CMA-ES虽无需梯度，但需要多次模型评估，子空间数M和种群大小影响效率
2. **校准数据依赖**：优化结果依赖于校准数据的质量和代表性
3. **超参数敏感性**：子空间数量M的选择对最终效果有影响，论文未详细分析
4. **幅度分解的假设**：按参数幅度排序分组可能不是最优的子空间划分方式
5. **评估不够全面**：LLM实验仅融合3个任务，规模较小
6. **与剪枝后微调的对比缺失**：未与先剪枝后继续训练的方法做比较

## 相关工作

- **Task Vector**：Ilharco等人提出的任务算术，通过参数差异实现知识编辑
- **模型合并**：Task Arithmetic、TIES-Merging、Fisher Merging、RegMean等
- **模型剪枝**：SparseGPT、Wanda等传统剪枝与DARE、Model Tailor等针对微调模型的剪枝
- **知识迁移**：WiSE-FT、Model Tailor等利用预训练模型参数改善迁移
- **TALL-masks**：通过掩码定位task vector中的关键任务信息

## 评分 ⭐⭐⭐⭐

- **创新性**：⭐⭐⭐⭐ 子空间重加权+进化搜索的组合简洁有效
- **实验充分性**：⭐⭐⭐⭐⭐ 三个场景×多种模态×多个模型的全面验证
- **实用价值**：⭐⭐⭐⭐ 对模型部署和多任务合并有直接应用价值
- **写作质量**：⭐⭐⭐⭐ 方法直观清晰，实验组织系统

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Loss Functions for Predictor-based Neural Architecture Search](../../ICCV2025/others/loss_functions_for_predictor-based_neural_architecture_search.md)
- [\[ACL 2025\] Better Embeddings with Coupled Adam](better_embeddings_with_coupled_adam.md)
- [\[ACL 2025\] Predicting Through Generation: Why Generation Is Better for Prediction](predicting_through_generation_why_generation_is_better_for_prediction.md)
- [\[ICML 2025\] GLGENN: A Novel Parameter-Light Equivariant Neural Networks Architecture Based on Clifford Geometric Algebras](../../ICML2025/others/glgenn_a_novel_parameter-light_equivariant_neural_networks_architecture_based_on.md)
- [\[NeurIPS 2025\] A High-Dimensional Statistical Method for Optimizing Transfer Quantities in Multi-Source Transfer Learning](../../NeurIPS2025/others/a_highdimensional_statistical_method_for_optimizing_transfer.md)

</div>

<!-- RELATED:END -->
