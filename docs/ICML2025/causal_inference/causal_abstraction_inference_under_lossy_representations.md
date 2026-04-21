---
title: >-
  [论文解读] Causal Abstraction Inference under Lossy Representations
description: >-
  [ICML 2025][因果抽象] 提出 **投影抽象（Projected Abstraction）** 框架，突破现有因果抽象理论对"抽象不变性条件（AIC）"的依赖，使得在有损/降维表示下仍能进行数学一致的因果推断，并给出图模型层面的可识别性判据。
tags:
  - ICML 2025
  - 因果抽象
  - 有损表示
  - 结构因果模型
  - 因果推断
  - 表示学习
---

# Causal Abstraction Inference under Lossy Representations

**会议**: ICML 2025  
**arXiv**: [2509.21607](https://arxiv.org/abs/2509.21607)  
**代码**: [CausalAILab/ProjectedCausalAbstractions](https://github.com/CausalAILab/ProjectedCausalAbstractions)  
**领域**: 因果推断  
**关键词**: 因果抽象, 有损表示, 结构因果模型, 因果推断, 表示学习

## 一句话总结

提出 **投影抽象（Projected Abstraction）** 框架，突破现有因果抽象理论对"抽象不变性条件（AIC）"的依赖，使得在有损/降维表示下仍能进行数学一致的因果推断，并给出图模型层面的可识别性判据。

## 研究背景与动机

因果推断与抽象推理是人类智能的两大核心能力。因果推断通常在结构因果模型（SCM）语义下研究，SCM 诱导出"因果阶梯"（Pearl Causal Hierarchy, PCH）的三层分布：观测层 $\mathcal{L}_1$、干预层 $\mathcal{L}_2$、反事实层 $\mathcal{L}_3$。

因果抽象（Causal Abstraction）旨在建立复杂低层因果模型 $\mathcal{M}_L$ 与简单高层模型 $\mathcal{M}_H$ 之间的对应关系。已有工作（Rubenstein et al., 2017; Beckers & Halpern, 2019; Xia & Bareinboim, 2024 等）取得了显著进展，但**几乎所有定义都依赖于抽象不变性条件（Abstract Invariance Condition, AIC）**：

> AIC 要求：若两个低层值映射到同一高层值，则它们对下游变量的影响必须一致。

这一限制在实践中非常苛刻——表示学习和降维本质上就是有损变换，AIC 几乎不可能满足。例如，将 HDL 和 LDL 胆固醇加总为"总胆固醇"时，二者对心脏病的影响方向相反，AIC 立即被违反。类似地，将高维图像压缩为低维表示时，AIC 往往无法验证也不太可能成立。

本文的核心动机就是：**即使 AIC 不成立，能否仍然定义数学一致的因果抽象，并从有限数据做因果推断？**

## 方法详解

### 整体框架

本文提出的技术路线分三步：

1. **定义投影抽象（Projected Abstraction）**：将 AIC 违反时丢失的信息编码到外生变量空间，使高层模型在数学上仍然一致（Section 2）
2. **构造投影 C-DAG（Partially Projected C-DAG）**：提供新的图模型表示，捕捉因 AIC 违反而产生的额外依赖关系（Section 3）
3. **建立抽象可识别性定理**：将跨抽象层的因果推断问题归约为经典可识别性问题（Section 3）

### 关键设计

#### 1. 部分 SCM 投影（Partial SCM Projection）

传统 SCM 投影只允许整个变量被包含或排除。本文引入**部分投影**的概念：对每个变量 $W$，通过满射函数 $\delta$ 将其拆分为：

- **观测部分** $W^o$：保留在高层模型中的信息
- **未观测部分** $W^u$：因有损映射丢失的信息，被编码为新的外生变量

关键等式（Prop. 1）保证在部分投影后，任何干预下的输出保持一致：

$$\mathbf{w}^o_{\mathbf{x}} = \mathcal{M}'_{[\mathbf{x}^o]}(\mathbf{u}, \mathbf{x}^u, \mathbf{z}^u)$$

其中 $\delta(\mathbf{w}^o_{\mathbf{x}}, \mathbf{w}^u_{\mathbf{x}}) = \mathbf{W}_{\mathbf{x}}(\mathbf{u})$。

#### 2. 软干预作为高层硬干预的低层对应

AIC 违反导致高层硬干预 $X_H \leftarrow x_H$ 在低层是模糊的（可能对应多个不同效果的 $x_L$）。本文将其解释为**软干预** $\sigma_{X_L}$：

$$P(\sigma_{\mathbf{C}_i} = \mathbf{c}_i) = P(\mathbf{c}_i \mid \tau(\mathbf{c}_i) = v_{H,i}, \mathbf{pa}_{V_{H,i}}, \mathbf{u}^c_{V_{H,i}})$$

直觉上，高层干预等价于在对应的低层干预值上按先验概率做加权，权重取决于父节点值。

**保险公司示例**：将保险计划 $x_1$（廉价高效）、$x_2$（廉价低效）抽象为"廉价计划" $x_C$ 时，干预 $X_H \leftarrow x_C$ 在低层表现为：

$$\sigma_{X_L} = \begin{cases} x_1 & \text{概率 } P(x_1 \mid X_L \in \{x_1,x_2\}, z) \\ x_2 & \text{概率 } P(x_2 \mid X_L \in \{x_1,x_2\}, z) \end{cases}$$

#### 3. 算法构造高层模型（Algorithm 1）

给定低层模型 $\mathcal{M}_L$ 和构造性抽象函数 $\tau$，算法系统性地构建投影抽象 $\mathcal{M}_H$：

- 对每个低层变量 $W$，拆分为 $(W^o, W^u)$
- 将 $W^u$ 纳入外生变量集 $\mathbf{U}_H$
- 根据软干预公式设定 $W^u$ 的分布
- 高层函数 $f^H_i$ 通过 $\delta$ 重建低层输入，再用 $\tau$ 映射输出

**Theorem 1** 证明该算法构造的 $\mathcal{M}_H$ 对所有 $\mathcal{L}_3$ 查询都与 $\mathcal{M}_L$ 保持 $Q$-$\tau$ 一致性。

#### 4. 投影 C-DAG（Partially Projected C-DAG）

当 AIC 不成立时，传统 C-DAG 的约束可能不正确（Prop. 2）。本文定义了投影 C-DAG $\mathcal{G}^\dagger_\mathbb{C}$，通过三条规则向图中添加因 AIC 违反而产生的新边：

| 规则 | 原始结构 | 新增边 | 直觉 |
|------|----------|--------|------|
| (1) | $Z \to X \to Y$，$X$ 违反 AIC | 添加 $Z \to Y$ | $X^u$ 依赖 $Z$，传递因果效应 |
| (2) | $Z \leftrightarrow X \to Y$，$X$ 违反 AIC | 添加 $Z \leftrightarrow Y$ 和 $X \leftrightarrow Y$ | 混淆传播 |
| (3) | $Z \leftarrow X \to Y$，$X$ 违反 AIC | 添加 $Z \leftrightarrow Y$ | 共因的未观测部分引入混淆 |

**Theorem 2** 证明投影 C-DAG 完整描述了高层变量上的所有约束——既充分又必要。

#### 5. 抽象可识别性（Abstract Identification）

**Theorem 3（对偶抽象可识别性）** 是本文最具实用价值的结果：

> 低层查询 $Q$ 在投影 C-DAG $\mathcal{G}^\dagger_\mathbb{C}$ 和数据 $\mathbb{Z}$ 下 $\tau$-可识别，当且仅当高层查询 $\tau(Q)$ 在 $\mathcal{G}^\dagger_\mathbb{C}$ 和 $\tau(\mathbb{Z})$ 下经典可识别。

这意味着研究者可以直接复用已有的因果识别算法（后门准则、前门准则等），只需在投影 C-DAG 上操作即可。

### 损失函数 / 训练策略

实验中使用 GAN-NCM（Neural Causal Models 的 GAN 实现）进行训练：

- **生成器**：按因果图结构建模，每个变量对应一个子网络
- **判别器**：区分真实数据和生成数据
- **投影采样（Projected Sampling）**：利用软干预定义直接建模和采样，即使在极端降维下也能重建低层数据

## 实验关键数据

### 主实验

**实验 1：MNIST 估计任务（验证投影 C-DAG 的必要性）**

设定：$Z$ 为数字（0-9），$X$ 为彩色 MNIST 图像，$Y$ 为颜色预测标签。$\tau(X)$ 将图像映射为二值变量（明/暗）。目标查询 $P(y_x \mid z)$。

| 方法 | 使用图模型 | 收敛行为 | 大样本 MAE |
|------|-----------|----------|-----------|
| 无抽象 NCM | 原始图 | 收敛但慢 | ~0.04 |
| C-DAG NCM | C-DAG（不考虑 AIC 违反） | **不收敛** | ~0.08（偏差） |
| 投影 C-DAG NCM | 投影 C-DAG | **收敛且快** | **~0.02** |

**实验 2：因果彩色 MNIST（投影采样质量）**

设定：数字 $D$ 和颜色 $C$ 共同决定图像 $I$，但 $D$ 和 $C$ 存在混淆（如 0 倾向红色，5 倾向青色）。

| 方法 | 表示维度 | $\mathcal{L}_1$: $P(I \mid D=0)$ | $\mathcal{L}_2$: $P(I_{D=0})$ | $\mathcal{L}_3$: $P(I_{D=0} \mid D=5)$ |
|------|---------|------|------|------|
| Non-causal | 原始 | ✓ 正确（红色0） | ✗ 失败 | ✗ 失败 |
| RNCM (dim=16) | 16维 | ✓ | ✓ 多色0 | ✓ 青色0 |
| RNCM (dim=2) | 2维 | ✗ 模糊 | ✗ 模糊 | ✗ 模糊 |
| **投影采样（dim=2）** | **2维** | **✓** | **✓** | **✓** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无抽象（原始空间） | MAE 较高，收敛慢 | 高维空间导致学习困难 |
| C-DAG 约束 | MAE 有偏差，不收敛 | 图约束错误，AIC 违反未被考虑 |
| 投影 C-DAG 约束 | MAE 最低，快速收敛 | 正确的图约束 |
| RNCM 高维表示 (16d) | 图像质量好 | 受 AIC 限制无法进一步压缩 |
| 投影采样 低维表示 (2d) | 图像质量同样好 | 突破 AIC 限制，极端压缩仍有效 |

### 关键发现

1. **C-DAG 在 AIC 违反下是不充分的**：使用错误图约束会导致系统性偏差，增加数据量也无法修正
2. **投影 C-DAG 既充分又必要**：正确捕捉了因 AIC 违反而产生的额外依赖
3. **投影采样突破了表示维度的瓶颈**：即使将高维图像压缩到二值表示，仍能正确生成三个层次的因果查询
4. **抽象带来效率提升**：在高维空间中，使用抽象的方法比不使用抽象的方法收敛更快、误差更低

## 亮点与洞察

1. **核心洞察极为优雅**：将 AIC 违反产生的信息损失编码为外生变量，而非试图强制满足 AIC。这是一种"承认并量化不确定性"的思路
2. **对偶可识别性定理（Thm 3）打通了理论与实践的桥梁**：将全新的抽象可识别性问题直接归约为经典问题，使得数十年的因果推断算法都可以复用
3. **投影 C-DAG 的构造规则简洁直观**：仅三条规则就完整描述了 AIC 违反对图结构的影响
4. **同时覆盖观测层、干预层、反事实层**：Theorem 1 保证了 $\mathcal{L}_3$ 级别的一致性，这是最强的保证
5. **投影采样的实用价值**：为表示学习 + 因果推断的结合提供了实际可行的方案

## 局限与展望

1. **假设低层因果图已知**：投影 C-DAG 的构造需要知道低层的因果图结构和 AIC 违反变量集 $\mathbf{V}^\dagger_H$，在实际中这些信息可能不完全可用
2. **离散变量假设**：理论框架限定在有限离散域的内生变量上，连续变量的扩展尚未讨论
3. **递归 SCM 假设**：要求因果图无环，排除了反馈系统
4. **实验规模有限**：仅在 MNIST 级别的简单图像上验证，尚未在大规模真实数据上测试
5. **软干预分布的选择**：Eq. 11 给出的是一个"自然"的选择，但存在多种可能的替代方案，最优选择可能依赖于具体场景

## 相关工作与启发

- **Xia & Bareinboim (2024)**：本文的直接前身，建立了构造性抽象函数和 PCH 层面的抽象一致性理论，但仍需要 AIC
- **Beckers & Halpern (2019)**：τ-abstraction 的经典定义，本文的投影抽象是对其部分投影的推广
- **Anand et al. (2023)**：提出 C-DAG 用于集群因果推断，本文展示了 C-DAG 在 AIC 违反下的不足并给出修正
- **NCM (Xia et al., 2021, 2023)**：神经因果模型，本文实验的基础工具
- **因果表示学习**（Ahuja et al., 2023; Brehmer et al., 2022）：与本文互补，本文提供了表示学习中因果一致性的理论保证

## 评分

| 维度 | 评分 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | ⭐⭐⭐⭐⭐ | 首次系统解决 AIC 违反下的因果抽象问题 |
| 理论深度 | ⭐⭐⭐⭐⭐ | 三个主定理环环相扣，理论体系完整 |
| 实验充分性 | ⭐⭐⭐ | 实验设计精巧但规模偏小 |
| 写作质量 | ⭐⭐⭐⭐ | 数学符号密集但示例清晰，有系统性 |
| 实用价值 | ⭐⭐⭐⭐ | 对偶定理直接可用，投影采样有实践价值 |
| **综合** | **⭐⭐⭐⭐☆** | 因果抽象理论的重要推进，实验可进一步加强 |

<!-- RELATED:START -->

## 相关论文

- [Practical do-Shapley Explanations with Estimand-Agnostic Causal Inference](../../NeurIPS2025/causal_inference/practical_do-shapley_explanations_with_estimand-agnostic_causal_inference.md)
- [Latent Variable Causal Discovery under Selection Bias](latent_variable_causal_discovery_under_selection_bias.md)
- [Cyclic Counterfactuals under Shift–Scale Interventions](../../NeurIPS2025/causal_inference/cyclic_counterfactuals_under_shift-scale_interventions.md)
- [GST-UNet: A Neural Framework for Spatiotemporal Causal Inference with Time-Varying Confounding](../../NeurIPS2025/causal_inference/gst-unet_a_neural_framework_for_spatiotemporal_causal_inference_with_time-varyin.md)
- [Learning Robust Intervention Representations with Delta Embeddings](../../ICLR2026/causal_inference/learning_robust_intervention_representations_with_delta_embeddings.md)

<!-- RELATED:END -->
