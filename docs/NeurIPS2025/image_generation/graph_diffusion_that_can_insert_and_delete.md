---
title: >-
  [论文解读] Graph Diffusion that can Insert and Delete
description: >-
  [NeurIPS 2025][图像生成][图扩散] 提出 GrIDDD 模型，首次将离散去噪扩散模型（DDPM）扩展为支持在生成过程中**动态插入和删除图节点**，使分子图的大小可在扩散过程中自适应变化，在性质靶向和分子优化任务上达到或超过现有方法。
tags:
  - NeurIPS 2025
  - 图像生成
  - 图扩散
  - 节点插入删除
  - 分子生成
  - 性质靶向
  - 分子优化
---

# Graph Diffusion that can Insert and Delete

**会议**: NeurIPS 2025  
**arXiv**: [2506.15725](https://arxiv.org/abs/2506.15725)  
**代码**: 无  
**领域**: 图生成 / 分子生成  
**关键词**: 图扩散, 节点插入删除, 分子生成, 性质靶向, 分子优化

## 一句话总结

提出 GrIDDD 模型，首次将离散去噪扩散模型（DDPM）扩展为支持在生成过程中**动态插入和删除图节点**，使分子图的大小可在扩散过程中自适应变化，在性质靶向和分子优化任务上达到或超过现有方法。

## 研究背景与动机

### 分子生成的核心挑战

基于图的分子生成需要处理离散和组合性连接结构，化学约束使大部分原子-键组合无效。当前主流方法包括：

| 方法类型 | 代表工作 | 优势 | 劣势 |
|:---|:---|:---|:---|
| 自回归 | You et al., JT-VAE | 自然处理变长 | 顺序依赖，采样慢 |
| 一次生成 | GraphVAE | 并行采样 | 难以学习高阶交互 |
| 图扩散 | DiGress, FreeGress | 置换不变，迭代精化 | **图大小固定** |

### 固定大小的局限

现有图扩散模型的根本限制是：**样本大小在整个生成过程中保持不变**。这会带来两个严重问题：

**性质靶向（Property Targeting）**：目标性质与分子大小相关时（如分子量），固定大小的模型无法响应

**分子优化（Property Optimization）**：无法通过增减原子来优化目标性质

现有的解决方案：
- DiGress、E(3) Equivariant Diffusion：从训练集经验分布采样图大小
- FreeGress：用辅助分类器预测图大小
- Lift：在反向扩散中采样额外节点，但未纳入训练

这些都是权宜之计，无法实现在扩散过程中的动态大小调整。

## 方法详解

### 整体框架

GrIDDD 的核心创新在于重新设计了前向和反向扩散过程，使其支持**单调**的节点插入和删除：

- **前向过程**：从干净图 $G_0$（$n_0$ 个节点）逐步加噪至 $G_T$（$n_T$ 个节点）
- **反向过程**：从 $G_T$ 逐步去噪并调整大小至 $G_0$
- 当 $n_T > n_0$（需要删除），或 $n_T < n_0$（需要插入）

### 关键设计

#### 节点删除机制

引入两个特殊原子类型：DEL 和 DEL*

- **DEL**：表示已完全删除的节点（吸收态）
- **DEL***：中间态，节点刚被标记为删除，下一步将变为 DEL

为什么需要两种状态？
- 如果只有 DEL，节点一旦进入吸收态就无法在反向过程中被恢复为正常原子类型
- DEL* 保证在反向过程中，重新插入的节点可以在下一步切换回正常原子类型

#### 混合扩散过程

对于需要删除 $|\Delta_T|$ 个节点的情况，采用混合前向过程：
- $n_0 - |\Delta_T|$ 个节点使用标准扩散转移矩阵 $Q_t$
- $|\Delta_T|$ 个节点使用特殊转移矩阵 $Q_t^*$

特殊转移矩阵的形式：

$$Q_t^* = \zeta(t)(\alpha_t A^* + (1-\alpha_t) B^*) + (1-\zeta(t)) C^*$$

其中：
- $A^*, B^*$ 是标准矩阵的扩展版本（加入 DEL/DEL* 类型）
- $C^*$ 描述从正常类型到 DEL 的转移
- $\zeta(t)$ 是 logistic 分布 CDF，控制删除时机

#### 插入/删除时机调度

使用 logistic 分布的概率密度控制何时插入/删除：

$$\zeta'(t) = -\frac{e^{\frac{t-D}{w}}}{w(e^{\frac{t-D}{w}} + 1)^2}$$

参数 $D$ 控制高峰时间步，$w$ 控制曲线陡峭程度。这保证了插入/删除在扩散过程中平滑分布。

#### 目标大小采样

训练时，从离散分布 $h_{n_0}(n)$ 采样目标大小 $n_T$：

$$h_{n_0}(n) = p_{max} + \frac{p_{min} - p_{max}}{n_{max}} (n - n_0)$$

最高概率出现在 $n = n_0$（大小不变），概率随大小差异线性下降。

### 损失函数 / 训练策略

GrIDDD 使用 classifier-free guidance 进行条件生成：
- 训练时将条件信息直接注入去噪网络
- 使用交叉熵损失训练去噪器预测 $G_0$
- 对节点类型和边类型分别优化
- 反向过程中需要处理三种情况（插入、删除、大小不变）的后验概率

## 实验关键数据

### 主实验：性质靶向（Property Targeting）

在 QM9 和 ZINC-250k 数据集上的结果：

| 方法 | QM9 MAE ↓ | QM9 Validity ↑ | ZINC MAE ↓ | ZINC Validity ↑ |
|:---|:---|:---|:---|:---|
| DiGress | 较高 | ~95% | 较高 | ~85% |
| FreeGress | 中等 | ~97% | 中等 | ~90% |
| GCPN | 中等 | ~90% | 中等 | ~80% |
| JT-VAE | 中等 | ~100% | 中等 | ~100% |
| **GrIDDD** | **最低或持平** | **~96-98%** | **最低或持平** | **~88-92%** |

GrIDDD 在性质靶向误差（MAE）上持平或优于现有最佳方法，同时保持高化学有效性。关键在于 GrIDDD 解决的是一个**更难的问题**（同时调整大小和结构），却仍取得了竞争力结果。

### 分子优化实验

| 方法 | 平均改进 ↑ | 优化成功率 ↑ | 类型 |
|:---|:---|:---|:---|
| VJTNN | 中等 | 中等 | 翻译模型 |
| Seq2Seq | 低 | 低 | 翻译模型 |
| HierG2G | 中-高 | 中等 | 翻译模型 |
| GCPN | 中等 | 中等 | RL 优化 |
| JT-VAE | 中等 | 中等 | VAE 优化 |
| **GrIDDD** | **最高** | **最高** | 扩散模型 |

GrIDDD 在分子优化任务上显著优于专门的优化模型。这正是因为它可以通过增删原子来优化目标性质，而非仅修改现有原子的类型。

### 消融实验

#### 大小自适应的效果

| 配置 | QM9 MAE | 分子大小方差 |
|:---|:---|:---|
| 固定大小（经验分布采样） | 基准 | 低（受限于预采样） |
| 固定大小（分类器预测） | 稍好 | 中 |
| **GrIDDD（动态调整）** | **最好** | **高（自适应）** |

动态大小调整使模型能够生成更多样化大小的分子，特别是在目标性质与分子大小强相关时表现突出。

### 关键发现

1. **大小自适应是条件生成的关键**：当目标性质（如分子量、极化率）与分子大小相关时，GrIDDD 的优势最为明显。

2. **更难的训练 ≠ 更差的结果**：尽管 GrIDDD 需要同时学习去噪+大小调整，但在性质靶向上并未因任务复杂而降低性能。

3. **分子优化中的结构性优势**：传统方法无法通过改变原子数量来优化性质，GrIDDD 打破了这一限制。

4. **DEL/DEL* 双状态设计的必要性**：单删除态会导致节点无法在反向过程中恢复，双状态设计是关键技术贡献。

## 亮点与洞察

- **理论贡献**：首次在图扩散框架中形式化了节点级的插入删除操作，推导出新的前向/反向转移矩阵
- **实用价值**：打开了尺寸自适应分子生成的大门，对药物设计有直接意义
- **简洁设计**：通过增加两个特殊原子类型（DEL, DEL*）和混合扩散实现了看似复杂的功能
- **可扩展性**：该方法可应用于任何基于离散扩散的图生成框架

## 局限与展望

- 插入/删除是单调的（反向过程中只能持续插入或持续删除），不支持在同一生成过程中交替插入和删除
- logistic 分布调度的超参数（$D$, $w$）需要调优
- 计算开销：由于需要处理额外的删除类型，转移矩阵维度增加
- 仅在 QM9 和 ZINC-250k 上验证，更大规模的数据集和更复杂的分子结构（如蛋白质）未测试
- 阈值 $p^*$ 的确定可能依赖任务

## 相关工作与启发

- **DiGress (Vignac et al., 2023)**：分类扩散 + Graph Transformer，本文的主要基准
- **FreeGress (Ngo et al., 2024)**：Classifier-free 条件生成，本文的训练范式借鉴
- **Johnson et al. (2021)**：首次在文本扩散中提出插入/删除概念，但方法较复杂
- **Ketata et al. (2025)**：Lift 方法在反向过程中采样额外节点，但未纳入训练
- **GCPN (You et al., 2018)**：基于 RL 的分子生成，性质靶向的经典方法
- 本文的创新可推广至蛋白质设计、材料发现等需要结构自适应的领域

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次实现图扩散中的大小自适应生成
- **技术深度**: ⭐⭐⭐⭐ — 转移矩阵设计和后验推导有深度
- **实用性**: ⭐⭐⭐⭐⭐ — 对分子设计有直接应用价值
- **清晰度**: ⭐⭐⭐⭐ — 方法描述清晰，图例直观
- **综合评分**: 8.5/10

<!-- RELATED:START -->

## 相关论文

- [Can Knowledge-Graph-based Retrieval Augmented Generation Really Retrieve What You Need?](can_knowledge-graph-based_retrieval_augmented_generation_really_retrieve_what_yo.md)
- [Toward a Unified Geometry Understanding: Riemannian Diffusion Framework for Graph Generation and Prediction](toward_a_unified_geometry_understanding_riemannian_diffusion_framework_for_graph.md)
- [AugGen: Synthetic Augmentation using Diffusion Models Can Improve Recognition](auggen_synthetic_augmentation_using_diffusion_models_can_imp.md)
- [Scaling Can Lead to Compositional Generalization](scaling_can_lead_to_compositional_generalization.md)
- [Graph Distance as Surprise: Free Energy Minimization in Knowledge Graph Reasoning](graph_distance_as_surprise_free_energy_minimization_in_knowledge_graph_reasoning.md)

<!-- RELATED:END -->
