---
title: >-
  [论文解读] A Recipe for Causal Graph Regression: Confounding Effects Revisited
description: >-
  [ICML 2025][图学习][因果图学习] 首次系统性地将因果图学习从分类扩展到回归任务，通过增强型图信息瓶颈（Enhanced GIB）承认混淆子图的预测能力，并用对比学习替代依赖离散标签的因果干预方法，在图级 OOD 回归基准上显著超越现有方法。
tags:
  - ICML 2025
  - 图学习
  - 因果图学习
  - 图回归
  - 混淆效应
  - 图信息瓶颈
  - 对比学习
---

# A Recipe for Causal Graph Regression: Confounding Effects Revisited

**会议**: ICML 2025  
**arXiv**: [2507.00440](https://arxiv.org/abs/2507.00440)  
**代码**: [causal-graph/CGR](https://github.com/causal-graph/CGR)  
**领域**: 图学习  
**关键词**: 因果图学习, 图回归, 混淆效应, 图信息瓶颈, 对比学习

## 一句话总结

首次系统性地将因果图学习从分类扩展到回归任务，通过增强型图信息瓶颈（Enhanced GIB）承认混淆子图的预测能力，并用对比学习替代依赖离散标签的因果干预方法，在图级 OOD 回归基准上显著超越现有方法。

## 研究背景与动机

因果图学习（Causal Graph Learning, CGL）通过识别因果子图来提升 GNN 在分布外（OOD）场景下的泛化能力，已在分类任务上取得成功。然而，**图级回归任务**——一个更具挑战性的设置——几乎被完全忽视。现有 CGL 方法无法直接迁移到回归的核心原因有两点：

**离散标签不可用**：从有限支持到无限支持的转变使得图无法按类别分组，CAL 的后门调整和 DisC 的反事实推理都依赖离散标签信息。

**混淆子图假设过强**：现有方法（如 CAL、DisC）假设混淆子图 $S$ **完全不含预测信息**，但在实际场景中（如分子属性预测中，分子量与毒性虽非因果关系但有强相关），这一假设明显不合理。

作者在实验中验证了"直接搬用"的失败：将分类 CGL 方法简单适配到回归后，其 OOD 性能甚至不如最基础的 ERM（经验风险最小化）。

## 方法详解

### 整体框架

框架遵循结构因果模型（SCM）$G \to C, G \to S, C \to Y, S \to C$，整体流程分四步：

1. **图编码**：GNN encoder 对输入图 $G = (\mathbf{A}, \mathbf{X})$ 计算图嵌入 $H_g$
2. **子图分离**：注意力模块生成软掩码 $\mathbf{M}_{\text{edge}}, \mathbf{M}_{\text{node}}$，提取因果子图 $C$ 和混淆子图 $S = G - C$
3. **双路编码**：两个共享参数的 GNN 模块 $\mathcal{G}_c, \mathcal{G}_s$ 分别处理 $C$ 和 $S$，经 readout 层输出回归预测
4. **对比干预**：随机组合 $H_{c,i} + H_{s,j}$ 生成反事实混合表示 $H_{\text{mix}}$，用对比损失执行因果干预

总损失为：$L = L_{\text{GIB}} + \lambda L_{\text{CI}}$

### 关键设计

#### 设计一：增强型图信息瓶颈（Enhanced GIB）

标准 GIB 目标为 $-I(C;Y) + \alpha I(C;G)$，仅压缩因果子图 $C$ 的冗余信息。本文认为这忽略了混淆子图 $S$ 的预测能力，导致模型将所有 $Y$ 相关信息都压入 $C$，分离不彻底。

**增强目标**新增混淆子图互信息项：

$$L_{\text{GIB}} = -I(C;Y) + \alpha I(C;G) - \beta I(S;Y)$$

其中 $\beta I(S;Y)$ 显式鼓励 $S$ 也保留一定的预测信息，避免 $C$ 过载。注意故意不加 $I(S;G)$ 项，因为 $S$ 主要引入捷径相关，过度施加结构正则化会破坏分离效果。

**变分近似上界**：

- $I(C;G)$：假设 $p(C|G) = \mathcal{N}(\mu_\phi(G), I)$, $q(C) = \mathcal{N}(0, I)$，KL 散度简化为 $\frac{1}{2}\|\mu_\phi(G)\|^2$
- $I(C;Y)$：建模 $p(Y|H_c) = \mathcal{N}(Y; \mu_{(c)}, \sigma^2_{(c)})$，假设常数方差 $\sigma^2 = 1$ 后退化为最小二乘损失
- $I(S;Y)$：同理退化为混淆子图的最小二乘损失

实际计算的损失为：

$$L_{\text{GIB}} = \underbrace{\frac{1}{N}\sum_i(Y_i - \mu_{(c),i})^2}_{L_c: \text{因果回归损失}} + \alpha \cdot \underbrace{\frac{1}{2}\mathbb{E}[\|\mu_\phi(G)\|^2]}_{\text{压缩正则}} + \beta \cdot \underbrace{\frac{1}{N}\sum_i(Y_i - \mu_{(s),i})^2}_{L_s: \text{混淆回归损失}}$$

#### 设计二：基于对比学习的因果干预

现有因果干预（后门调整）依赖按类别分层的 $P(Y|C,S)$，在连续 $Y$ 下不可行。本文以**实例判别**（instance discrimination）替代**类别分离**（class separation），用对比学习实现无标签依赖的因果干预。

**反事实样本构造**：随机配对第 $i$ 个因果子图和第 $j$ 个混淆子图的表示：

$$H_{\text{mix},ij} = H_{c,i} + H_{s,j}$$

**对比损失（InfoNCE 风格）**：原始图 $H_{g,i}$ 与其混合图 $H_{\text{mix},ij}$ 为正样本对，与其他图 $H_{g,k}$ 为负样本对：

$$L_{\text{CI}} = -\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp(\text{sim}(H_{g,i}, H_{\text{mix},ij}))}{\sum_{k \neq i}\exp(\text{sim}(H_{g,i}, H_{g,k}))}$$

**核心直觉**：若因果子图 $C_i$ 被正确提取，那么无论搭配哪个混淆子图 $S_j$，混合表示都应与原图表示对齐——因为预测由因果部分决定。这迫使模型学到**对混淆变化不变**的因果表示。

#### 设计三：掩码生成机制

因果/混淆子图通过可学习的软掩码分离：

$$C = (\mathbf{M}_{\text{edge}} \odot \mathbf{A},\ \mathbf{M}_{\text{node}} \cdot \mathbf{X})$$

$\mathbf{M}_{\text{edge}} \in [0,1]^{n \times n}$ 和 $\mathbf{M}_{\text{node}}$（对角元素 $\in [0,1]$）由以 $G$ 表示为条件的 MLP 生成，端到端优化。

### 损失函数 / 训练策略

最终训练目标为三部分加权组合：

$$L = L_c + \alpha \cdot L_{\text{reg}} + \beta \cdot L_s + \lambda \cdot L_{\text{CI}}$$

- $L_c$: 因果子图 MSE 回归损失（主任务）
- $L_{\text{reg}}$: 因果嵌入 $\ell_2$ 正则（信息压缩）
- $L_s$: 混淆子图 MSE 回归损失（承认混淆的预测力）
- $L_{\text{CI}}$: InfoNCE 对比损失（因果干预）
- 超参数 $\alpha, \beta, \lambda$ 控制各项权重

## 实验关键数据

### 主实验

在 GOOD-ZINC 数据集上的 OOD MAE（越低越好）：

| 方法 | Scaffold-OOD | Size-OOD | 类型 |
|------|-------------|----------|------|
| ERM | 0.1660 | 0.1248 | 基线 |
| IRM | 0.2313 | 0.1245 | 不变学习 |
| VREx | 0.1561 | 0.1271 | 不变学习 |
| DANN | 0.1734 | 0.1289 | 域适应 |
| Coral | 0.1734 | 0.1260 | 域适应 |
| CIGA | 0.2986 | 0.2415 | 因果分类方法 |
| DIR | 0.3650 | 0.2619 | 因果分类方法 |
| **CGR (本文)** | **最优** | **最优** | 因果回归 |

**发现**：专为分类设计的因果方法（CIGA、DIR）直接用于回归时，OOD 性能甚至**远差于 ERM 基线**，验证了本文动机。

### 消融实验

| 配置 | 关键效果 | 说明 |
|------|---------|------|
| 去掉 $\beta I(S;Y)$ | OOD 性能显著下降 | 忽略混淆预测力导致分离不充分 |
| 去掉 $L_{\text{CI}}$ | OOD 性能下降 | 缺乏因果干预，泛化能力减弱 |
| 用标签监督替代对比损失 | 性能不如对比版本 | 连续标签下标签监督效果有限 |
| 去掉 $I(C;G)$ 正则 | 因果子图信息冗余 | 压缩正则对精确分离必要 |
| 完整模型 | 最优 OOD | 三个组件互补 |

### 关键发现

1. **分类因果方法不适用回归**：CIGA 和 DIR 在回归上的 OOD 表现远逊于简单 ERM，证明回归需要专门设计
2. **混淆子图确实有预测力**：加入 $\beta I(S;Y)$ 后性能提升，说明"混淆=纯噪声"假设在回归中不成立
3. **对比学习优于标签监督**：在连续标签场景下，InfoNCE 风格的无监督对比损失比有监督方式更适合因果干预

## 亮点与洞察

- **关键洞察**：将"混淆子图无预测力"这一分类时代的强假设翻转为"混淆有预测力但非因果"，更贴合回归实际
- **技术桥梁**：用对比学习的实例判别原则替代分类中按标签分层的后门调整，是因果推理从分类到回归的自然推广
- **理论优雅**：通过变分近似将增强 GIB 目标简化为两个 MSE 损失 + 一个 $\ell_2$ 正则，实现简洁且有理论支撑
- **通用性**：框架与具体 GNN backbone 无关，可嵌入任意图神经网络架构

## 局限与展望

1. **超参数较多**：$\alpha, \beta, \lambda$ 三个权重需要调节，sensitivity 分析的成本较高
2. **常数方差假设**：$p(Y|H_c)$ 假设 $\sigma^2 = 1$ 是为简化，异方差建模可能进一步提升
3. **混合方式简单**：$H_{\text{mix}} = H_c + H_s$ 的加法组合可能不是最优，更复杂的融合策略值得探索
4. **因果子图的先验**：当前使用数据驱动的软掩码，引入领域知识（如分子官能团）可能改善分离质量
5. **仅评估图级回归**：节点级或边级回归任务未涉及

## 相关工作与启发

- **CAL/CAL+** (Sui et al., 2022/2024)：后门调整框架，本文继承其子图分离架构但改进损失设计
- **DisC** (Fan et al., 2022)：反事实推理，依赖离散标签，本文用对比学习替代
- **GSAT** (Miao et al., 2022)：随机注意力采样的可解释 GNN，GIB 理论基础共通
- **CIGA** (Chen et al., 2022)：不变子图学习，本文实验表明其不适用于回归
- **GIB** (Wu et al., 2020)：图信息瓶颈原理，本文在此基础上引入混淆互信息项
- 启发：因果学习从分类到回归的迁移不仅仅是损失函数替换，需要重新审视对混淆效应的建模假设

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次系统处理因果图回归，两个核心改进（Enhanced GIB + 对比干预）定位精准
- 理论性: ⭐⭐⭐⭐ — 变分近似推导完整，动机与方法逻辑自洽
- 实验充分度: ⭐⭐⭐⭐ — 多个 OOD 基准验证，消融充分，基线覆盖全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，符号一致，图表信息量大
- 价值: ⭐⭐⭐⭐ — 框架通用，代码开源，适用于分子属性预测等实际场景

<!-- RELATED:START -->

## 相关论文

- [LLM Enhancers for GNNs: An Analysis from the Perspective of Causal Mechanism Identification](llm_enhancers_for_gnns_an_analysis_from_the_perspective_of_causal_mechanism_iden.md)
- [NeuroCircuitry-Inspired Hierarchical Graph Causal Attention Networks for Explainable Depression Identification](../../ICLR2026/graph_learning/neurocircuitry-inspired_hierarchical_graph_causal_attention_networks_for_explain.md)
- [Graph Attention is Not Always Beneficial: A Theoretical Analysis of Graph Attention Mechanisms via Contextual Stochastic Block Models](graph_attention_is_not_always_beneficial_a_theoretical_analysis_of_graph_attenti.md)
- [On Measuring Long-Range Interactions in Graph Neural Networks](on_measuring_long-range_interactions_in_graph_neural_networks.md)
- [Toward Data-centric Directed Graph Learning: An Entropy-driven Approach](toward_data-centric_directed_graph_learning_an_entropy-driven_approach.md)

<!-- RELATED:END -->
