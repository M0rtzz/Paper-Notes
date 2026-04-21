---
title: >-
  [论文解读] DeepAFL: Deep Analytic Federated Learning
description: >-
  [ICLR 2026][优化][联邦学习] 提出 DeepAFL，通过设计无梯度的解析残差块并引入逐层联邦训练协议，首次实现了具有表征学习能力的深度解析联邦学习模型，既保持了对数据异质性的理想不变性，又突破了现有解析方法仅限于单层线性模型的局限，在三个基准数据集上超越 SOTA 5.68%-8.42%。
tags:
  - ICLR 2026
  - 优化
  - 联邦学习
  - 解析学习
  - 无梯度训练
  - 残差块
  - 数据异质性
---

# DeepAFL: Deep Analytic Federated Learning

**会议**: ICLR 2026  
**arXiv**: [2603.00579](https://arxiv.org/abs/2603.00579)  
**代码**: 无  
**领域**: 优化 / 联邦学习  
**关键词**: 联邦学习, 解析学习, 无梯度训练, 残差块, 数据异质性

## 一句话总结

提出 DeepAFL，通过设计无梯度的解析残差块并引入逐层联邦训练协议，首次实现了具有表征学习能力的深度解析联邦学习模型，既保持了对数据异质性的理想不变性，又突破了现有解析方法仅限于单层线性模型的局限，在三个基准数据集上超越 SOTA 5.68%-8.42%。

## 研究背景与动机

**联邦学习（FL）** 是打破数据孤岛的主流分布式学习范式。然而，传统的基于梯度的 FL 方法（如 FedAvg、FedProx、SCAFFOLD 等）面临四大核心问题：（1）**数据异质性**——不同客户端的数据分布差异导致模型聚合后性能下降（尤其在非 IID 场景）；（2）**收敛性**——异质数据导致客户端模型发散，聚合后可能偏离全局最优；（3）**可扩展性**——大量客户端参与时通信和计算开销成倍增长；（4）**通信开销**——多轮梯度交换需要大量带宽。

近年来，**解析学习（Analytic Learning）** 为上述问题提供了一条新思路。其核心想法是：通过封闭形式（closed-form）解替代迭代梯度更新，从根本上消除梯度训练的不稳定性。已有一些工作将解析学习引入联邦设定（如 FedAnalytic），在数据异质性不变性上表现优异——因为封闭形式解不依赖学习率、不需要多轮迭代，因此不受非 IID 数据分布的影响。

但现有解析 FL 方法存在一个**根本性瓶颈**：它们仅限于在冻结的预训练 backbone 上训练**单层线性模型**（如岭回归/最小二乘分类器）。由于没有表征学习能力，模型只能依赖预训练特征的质量，在需要特征适应的任务上表现次优。

本文的核心矛盾是：**如何在保持数据异质性不变性的前提下，赋予解析模型深层表征学习能力？** 核心 idea 是借鉴 ResNet 的成功经验，设计无梯度的解析残差块——每一层都有封闭形式解，通过逐层堆叠实现深度表征学习。

## 方法详解

### 整体框架

DeepAFL 的整体 pipeline：（1）所有客户端共享一个预训练 backbone 提取基础特征；（2）在 backbone 之上逐层堆叠解析残差块；（3）每层的训练通过客户端本地计算 + 服务器聚合一轮完成（无需多轮通信）；（4）逐层训练完毕后得到完整的深度模型。输入是各客户端的本地数据，输出是全局可用的深层分类/回归模型。

### 关键设计

1. **无梯度解析残差块（Gradient-Free Residual Block）**: 受 ResNet 启发，每个残差块的形式为 $\mathbf{h}^{(l+1)} = \mathbf{h}^{(l)} + f^{(l)}(\mathbf{h}^{(l)})$，其中 $f^{(l)}$ 是带有非线性激活的线性变换。关键创新在于：$f^{(l)}$ 的参数通过**最小二乘法**（而非梯度下降）求解封闭形式解。

   具体来说，给定第 $l$ 层的输入特征矩阵 $\mathbf{H}^{(l)}$ 和目标 $\mathbf{Y}$，残差映射 $f^{(l)}$ 的参数 $\mathbf{W}^{(l)}$ 通过求解以下优化问题获得：

   $$\mathbf{W}^{(l)} = \arg\min_{\mathbf{W}} \|\phi(\mathbf{H}^{(l)}) \mathbf{W} - (\mathbf{Y} - \mathbf{H}^{(l)})\|_F^2 + \lambda \|\mathbf{W}\|_F^2$$

   其中 $\phi(\cdot)$ 是非线性特征映射（如随机特征或核近似），$\lambda$ 是正则化系数。这个问题有封闭形式解：$\mathbf{W}^{(l)} = (\phi(\mathbf{H}^{(l)})^\top \phi(\mathbf{H}^{(l)}) + \lambda \mathbf{I})^{-1} \phi(\mathbf{H}^{(l)})^\top (\mathbf{Y} - \mathbf{H}^{(l)})$。

   残差连接确保了信息流的稳定性——即使某一层的映射不理想，跳接保证了输入信息的传递。多层堆叠使模型具备了渐进式的特征精炼能力。

2. **逐层联邦训练协议（Layer-Wise FL Protocol）**: 传统 FL 训练整个模型，需要多轮通信。DeepAFL 采用逐层协议：对于第 $l$ 层，每个客户端 $k$ 在本地计算协方差矩阵 $\mathbf{A}_k^{(l)} = \phi(\mathbf{H}_k^{(l)})^\top \phi(\mathbf{H}_k^{(l)})$ 和交叉协方差矩阵 $\mathbf{B}_k^{(l)} = \phi(\mathbf{H}_k^{(l)})^\top (\mathbf{Y}_k - \mathbf{H}_k^{(l)})$，然后将这些矩阵发送给服务器。

   服务器端只需简单地**求和聚合**：$\mathbf{A}^{(l)} = \sum_k \mathbf{A}_k^{(l)}$，$\mathbf{B}^{(l)} = \sum_k \mathbf{B}_k^{(l)}$，然后计算全局解：$\mathbf{W}^{(l)} = (\mathbf{A}^{(l)} + \lambda \mathbf{I})^{-1} \mathbf{B}^{(l)}$。

   这个协议有三个关键优势：
   - **数据异质性不变**：由于矩阵求和操作的结合律，无论数据如何分布在各客户端，聚合结果与集中式训练完全一致
   - **单轮通信**：每层只需一轮通信（上传矩阵 → 服务器计算 → 下发参数），无需迭代
   - **隐私友好**：传输的是聚合统计量而非原始数据或梯度

3. **特征映射策略**: 为了在保持封闭形式解的同时引入非线性，DeepAFL 使用随机特征（Random Features）来近似核映射。这是一种经典技术：通过随机投影 + 非线性激活来隐式计算高维核特征，计算复杂度可控。每一层可以使用不同的随机特征映射，增加表征多样性。

### 损失函数 / 训练策略

每一层的训练目标是正则化最小二乘回归，目标函数为：

$$\mathcal{L}^{(l)} = \|\phi(\mathbf{H}^{(l)}) \mathbf{W}^{(l)} - (\mathbf{Y} - \mathbf{H}^{(l)})\|_F^2 + \lambda \|\mathbf{W}^{(l)}\|_F^2$$

由于是凸问题，有唯一全局最优解。训练流程是纯前向的——逐层从底到顶，每层一次求解，不需要反向传播。总训练轮数等于模型层数（而非传统 FL 中的数百甚至数千轮通信）。

## 实验关键数据

### 主实验

在三个基准数据集上的比较（非 IID 联邦设置）：

| 方法 | 数据集 1 | 数据集 2 | 数据集 3 | 训练方式 |
|------|---------|---------|---------|---------|
| FedAvg | 基线 | 基线 | 基线 | 多轮梯度 |
| FedProx | ~FedAvg | ~FedAvg | ~FedAvg | 多轮梯度+正则化 |
| SCAFFOLD | 优于 FedAvg | 优于 FedAvg | 优于 FedAvg | 方差减少 |
| FedAnalytic (单层) | 受限于线性模型 | 受限于线性模型 | 受限于线性模型 | 单层解析 |
| **DeepAFL** | **SOTA (+5.68%~8.42%)** | **SOTA** | **SOTA** | 深层解析 |

DeepAFL 相比之前的 SOTA 方法在三个基准数据集上提升 5.68%-8.42%。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 1 层 vs 多层 | 多层显著更好 | 证明深度表征学习的必要性 |
| 有残差连接 vs 无残差连接 | 有残差更稳定 | 残差确保信息流 |
| 不同层数 | 回报递减 | 3-5 层后提升放缓 |
| IID vs 非 IID | 性能差距极小 | 证明数据异质性不变性 |
| 不同客户端数量 | 稳定 | 可扩展性好 |

### 关键发现

- **深度 + 解析 = 双赢**: DeepAFL 首次证明解析学习可以"变深"，且深度确实带来了显著的性能提升（超越单层解析方法和多轮梯度方法）
- **数据异质性不变性得到理论和实验双重验证**: 无论数据如何非 IID 划分，DeepAFL 的结果与集中式训练一致，这是梯度式 FL 无法实现的
- **通信效率极高**: 每层只需一轮通信，总通信轮数等于层数（通常 3-5 轮），远少于梯度式方法的数百轮
- **无超参数调优负担**: 没有学习率、动量等超参数需要调，正则化系数 $\lambda$ 是唯一需要设的超参

## 亮点与洞察

- **打破了"解析学习 = 浅层模型"的认知**: 通过解析残差块的设计，证明了无梯度方法也能构建深层网络，这是方法论上的突破
- **ResNet 思想的优雅迁移**: 将深度学习中最成功的架构设计（残差连接）迁移到解析学习中，体现了跨范式的方法论融合
- **联邦学习的范式替代**: 对于"数据异质性"这一 FL 的核心难题，DeepAFL 从根本上消除了它的影响（而不是用各种技巧去缓解），这是一种质变而非量变的改进
- **极简的算法设计**: 整个方法只涉及矩阵乘法、求逆和求和，实现简单、理论清晰
- **理论保证完备**: 异质性不变性有严格的数学证明，不仅仅是经验观察

## 局限与展望

- **依赖预训练 backbone 的质量**: 虽然 DeepAFL 增加了表征学习能力，但仍然在冻结的预训练特征之上操作。如果 backbone 的特征质量差，深层解析块也难以弥补
- **矩阵求逆的计算瓶颈**: 每一层需要对 $d \times d$ 的矩阵求逆（$d$ 为特征维度），当特征维度很高时（如使用 ViT-Large 的 1024 维特征），计算开销不可忽视
- **随机特征的局限性**: 使用随机特征近似核映射虽然高效，但与真实的深度网络学到的分层特征相比，表征能力仍有差距
- **任务类型受限**: 目前仅在分类任务上验证。对于生成任务（如联邦 LLM 训练）是否适用尚不清楚
- **传输矩阵的隐私风险**: 虽然传输的是聚合统计量而非原始数据，但协方差矩阵可能泄露客户端数据的统计特征，需要进一步的差分隐私分析
- **可能的改进方向**: 与差分隐私的结合；端到端的解析特征学习（不冻结 backbone）；更高效的矩阵运算方法（如 Woodbury 恒等式）

## 相关工作与启发

- **FedAvg**（McMahan et al., 2017）: 联邦学习的基础算法，通过多轮平均聚合客户端模型。DeepAFL 用单轮精确求和替代了多轮近似平均
- **解析联邦学习**（如 FedCR, ACIL-FL）: DeepAFL 的直接前身，但被限制在单层线性模型。DeepAFL 的残差块设计突破了这一根本限制
- **极端学习机（ELM）**: 随机特征 + 最小二乘求解的经典方法，可以视为 DeepAFL 单层的特例
- **深度展开（Deep Unfolding）**: 在优化算法中逐层展开迭代步骤的思想，与 DeepAFL 的逐层求解有概念上的相似性
- **启发**: 解析学习作为梯度学习的替代范式，在联邦学习这种对收敛稳定性要求极高的场景中展现出了独特优势。未来可以探索解析学习在其他分布式/去中心化场景中的应用

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Convex Dominance in Deep Learning I: A Scaling Law of Loss and Learning Rate](convex_dominance_in_deep_learning_i_a_scaling_law_of_loss_and_learning_rate.md)
- [Deep FlexQP: Accelerated Nonlinear Programming via Deep Unfolding](deep_flexqp_accelerated_nonlinear_programming_via_deep_unfolding.md)
- [FedPM: Federated Learning Using Second-order Optimization with Preconditioned Mixing of Local Parameters](../../AAAI2026/optimization/fedpm_federated_learning_using_second-order_optimization_with_preconditioned_mix.md)
- [FedDAG: Clustered Federated Learning via Global Data and Gradient Integration for Heterogeneous Environments](feddag_clustered_federated_learning_via_global_data_and_gradient_integration_for.md)
- [Data Heterogeneity and Forgotten Labels in Split Federated Learning](../../AAAI2026/optimization/data_heterogeneity_and_forgotten_labels_in_split_federated_learning.md)

<!-- RELATED:END -->
