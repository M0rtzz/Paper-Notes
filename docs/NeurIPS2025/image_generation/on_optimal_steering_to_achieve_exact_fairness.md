---
title: >-
  [论文解读] On Optimal Steering to Achieve Exact Fairness
description: >-
  [NeurIPS 2025][图像生成][fairness] 本文定义了"理想分布"——使任意代价敏感风险下的 Bayes 最优分类器都满足精确公平性的数据分布，并提出通过 KL 散度最小化寻找最近理想分布的优化框架，为公平预处理和 LLM 表示引导提供了可证明的公平性保证。
tags:
  - NeurIPS 2025
  - 图像生成
  - fairness
  - optimal-transport
  - distribution-steering
  - LLM-representation
  - Bayes-optimal
---

# On Optimal Steering to Achieve Exact Fairness

**会议**: NeurIPS 2025  
**arXiv**: [2509.15759](https://arxiv.org/abs/2509.15759)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: fairness, optimal-transport, distribution-steering, LLM-representation, Bayes-optimal

## 一句话总结

本文定义了"理想分布"——使任意代价敏感风险下的 Bayes 最优分类器都满足精确公平性的数据分布，并提出通过 KL 散度最小化寻找最近理想分布的优化框架，为公平预处理和 LLM 表示引导提供了可证明的公平性保证。

## 背景与动机

1. **"偏见输入、偏见输出"问题**：ML 系统中不公平结果的根源在于训练数据本身的偏见。在有偏数据上施加公平约束训练并不能保证在无偏测试集上的公平性，甚至基于有偏验证数据的后处理也不可靠。
2. **公平预处理的局限**：Kamiran & Calders (2007) 的经典重权方法仅调整类-群组先验概率，忽略了特征分布，因此无法在与精度最大化结合时提供公平性保证。
3. **已有方法缺乏可证明保证**：Calmon et al. (2017) 的预处理框架在群组公平与个体公平不相容时可能不可行；Dutta et al. (2020) 使用 Chernoff Information 差距作为代理度量，无法直接转化为 DP/EO 等标准公平指标。
4. **理想分布的概念缺失**：此前工作聚焦于"公平分类器"（给定分布上约束最优模型），而非从数据源头定义"对任意下游任务天然公平"的分布，缺少这一层面的形式化理论。
5. **LLM 表示引导的兴起**：近年来 LLM 内部表示的可引导性（steering）成为研究热点，但理论基础薄弱，亟需可证明的公平性保证来指导表示变换。
6. **参数化条件的可操作性**：当群组-类条件分布属于常见参数族（如高斯、对数正态）时，理想分布的条件可简化为参数约束，使优化可行。

## 方法详解

### 理想分布的定义（Definition 3.1）

对于假设类 H 中的群组感知分类器 h: X×A → Y，分布 D 是**理想的**，当且仅当对所有代价矩阵 C ∈ R^{|Y|×|Y|}，D 上 C-代价敏感风险的 Bayes 最优分类器 h*_C 满足精确公平（如 ΔDP = 0 或 ΔEO = 0）。

关键特性：理想分布不存在公平性-精度权衡，因为最优分类器本身就是精确公平的。

### 参数化条件（Proposition 3.2 & 3.3）

当 X|Y=i,A=a ~ N(μ_{ia}, Σ_{ia}) 为多元高斯时，理想分布的充分条件为：

1. **标准化均值差一致**：Σ_{ia}^{-1/2}(μ_{ia} − μ_{ja}) = Σ_{ia'}^{-1/2}(μ_{ia'} − μ_{ja'})
2. **协方差比一致**：Σ_{ia}^{1/2} Σ_{ja}^{-1} Σ_{ia}^{1/2} = Σ_{ia'}^{1/2} Σ_{ja'}^{-1} Σ_{ia'}^{1/2}
3. **类别先验比一致**：q_{ia}/q_{ja} = q_{ia'}/q_{ja'}

对于一元高斯的二分类二群组情况（Proposition 3.3），上述条件是充要的，且同时保证 DP、EO 和 Equalized Odds。

### 最近理想分布的优化

目标：min_{D̃: ideal} D_KL(D̃ || D)

- 一般情况下非凸，但在特定干预策略下可高效求解。

### 平权行动干预（Theorem 4.1 & Corollary 4.2）

**仅改变弱势群组**的分布参数：
- 多元高斯情况下目标函数为凸（Theorem 4.1），可高效求解。
- 一元高斯的闭式解（Corollary 4.2）：σ̃_{i0} = γ* σ_{i1}，μ̃_{10} 为加权平均，μ̃_{00} = μ̃_{10} + γ*(μ_{01} − μ_{11})。

### 全子群干预（Proposition 4.3）

改变所有四个子群的分布参数：
- 优化问题非凸，但可通过对缩放因子 γ 的线搜索高效近似求解。
- 所有新参数都可表示为 γ* 和原始参数的函数。

### 精度-公平性的理论保证（Proposition 4.4）

- |err(h̃, D) − err(h̃, D̃)| ≤ √(2 D_KL(D̃, D))
- Δ_EO(h̃, D) ≤ √(8 D_KL(D̃ || D))

当理想分布足够接近原始分布时，精度损失和残余不公平性均可控。

### LLM 表示引导

1. **多类公平分类（Bios 数据集）**：使用 Llama-2 7B 嵌入，估计各子群（职业×性别）的前两阶矩，计算理想分布的目标矩，通过仿射变换引导表示。
2. **情感引导（GCS 框架）**：在 Llama-3 8B 上应用高斯概念引导（Gaussian Concept Steer），对弱势群组（恐怖类影评）的引导向量施加 EF 干预，以缩小群组间的"快乐度"差异。

## 实验结果

### 表1：一元高斯干预效果对比（合成数据）

| 干预方法 | Bayes Error (↓) | ΔDP (↓) | ΔEO (↓) | KL 距离 (↓) |
|---|---|---|---|---|
| 原始分布 | 0.15 | 0.25 | 0.18 | 0 |
| EF Affirmative（仅改弱势群组） | 0.12 | 0 | 0 | 小 |
| EF-All Subgroups（改全部） | 0.11 | 0 | 0 | 最小 |
| Mean Matching（均值匹配） | 0.14 | ~0.05 | ~0.03 | 中 |

**关键观察**：EF-All Subgroups 在保持接近原分布的同时实现完美公平和最低 Bayes Error；单纯均值匹配无法保证精确公平。

### 表2：Bios 数据集多类分类 TPR-gap（Llama-2 7B 嵌入）

| 方法 | 平均准确率 | 平均 RMS TPR-gap (↓) |
|---|---|---|
| 原始（无干预） | 0.79 | 高 |
| LEACE (Belrose et al.) | 0.77 | 中 |
| MiMiC (Singh et al.) | 0.78 | 中低 |
| EF Affirmative（本文） | 0.78 | 最低 |

**关键观察**：本文方法在所有职业类别上一致性地降低了 TPR-gap，准确率保持在 0.77–0.79 范围内，在多数类别上优于或持平 MiMiC 和 LEACE。

### 情感引导实验

在 Llama-3 8B 上引导电影评论生成"快乐"情感：
- Comedy 组基线引导效果好，Horror 组较差。
- 对 Horror 组应用 EF 干预后，适度的 α 值（如 0.03–0.05）显著提升其 Δ-Joyful 分数，缩小与 Comedy 组的差距。
- 过大 α 会扭曲引导向量，效果反降。

## 亮点

- **理想分布的形式化定义**：首次提出对任意代价敏感风险的 Bayes 最优分类器都精确公平的数据分布概念，巧妙避开了不可能定理的限制。
- **从数据源头解决公平问题**：不同于约束模型的"公平分类器"思路，从分布层面保证公平性，使得后续任意合理模型都继承公平性。
- **高效算法与闭式解**：平权行动干预在多元高斯下为凸优化，一元高斯下有闭式解，实用性强。
- **理论保证可迁移到原始分布**：Proposition 4.4 给出了理想分布上训练的分类器在原始分布上的精度和公平性偏差上界，弥合了理论与实践。
- **LLM 表示引导的理论增强**：将理想分布理论应用于 LLM 内部表示的仿射引导，为当前缺乏理论基础的 representation steering 提供了可证明保证。

## 局限性

- **参数化假设**：理论结果依赖于群组-类条件分布属于高斯等参数族的假设，真实数据可能不满足。
- **Bayes 最优假设的差距**：实际模型并非 Bayes 最优分类器，理论保证与实践之间存在差距。
- **非凸优化的全局性**：全子群干预（Proposition 4.3）的线搜索仅能找到近似全局最优，缺乏全局最优性保证。
- **多类多群组的可扩展性**：虽然框架支持多类多群组，但优化问题的参数数量和约束量随类-群组对的增加急剧增长。
- **有限样本设置**：理论基于总体分布层面，有限样本下参数估计误差如何传播到公平性保证尚未深入分析。

## 相关工作对比

| 对比维度 | 本文 | Dutta et al. (2020) |
|---|---|---|
| **目标** | 找最近理想分布使 Bayes 最优分类器精确公平 | 找最近分布使 Chernoff Information 差距为零 |
| **公平度量** | 直接保证 DP/EO/Equalized Odds | 通过信息论代理度量间接关联公平指标 |
| **可解性** | 平权行动干预为凸优化，有闭式解 | 优化问题不确定是否高效可解 |
| **适用性** | 参数化条件直接对应标准公平指标 | Chernoff Information 无法直接转化为 DP/EO |

| 对比维度 | 本文 | Singh et al. (2024) MiMiC |
|---|---|---|
| **方法** | 基于理想分布理论的仿射引导，有公平性证明 | 最小二乘矩匹配的仿射引导，经验有效 |
| **理论保证** | Bayes 最优分类器在理想分布上精确公平 | 无显式公平性理论保证 |
| **干预策略** | 仅改变弱势群组或所有群组，由优化问题决定 | 匹配群组间的前两阶矩 |
| **实验效果** | 在 Bios 数据集上 TPR-gap 一致更低或持平 | TPR-gap 降低但不如本文一致 |

| 对比维度 | 本文 | Kamiran & Calders (2007) |
|---|---|---|
| **干预层面** | 同时调整先验概率和特征分布参数 | 仅重权先验概率 q_{ia} |
| **公平保证** | 对任意代价敏感风险精确公平 | 无公平性理论保证 |
| **关系** | 本文条件蕴含 K&C 的重权（Remark 3.4），可视为两阶段方法的第二阶段 | 仅是本文方法的一个特例（先验部分） |

## 评分

| 维度 | 评分 | 说明 |
|---|---|---|
| 新颖性 | ⭐⭐⭐⭐⭐ | 首次形式化"理想分布"概念并给出参数化的充要条件，理论框架新颖 |
| 理论深度 | ⭐⭐⭐⭐ | 参数化条件推导严谨，凸性证明完整，KL 距离到公平性的传递上界有价值 |
| 实验充分性 | ⭐⭐⭐ | 合成数据分析透彻，但真实数据实验偏少（仅 Bios 和情感引导），缺少大规模基准对比 |
| 实用价值 | ⭐⭐⭐⭐ | 闭式解和凸优化使方法可直接应用于表示引导，但参数化假设限制了通用性 |

<!-- RELATED:START -->

## 相关论文

- [Counterfactual Identifiability via Dynamic Optimal Transport](counterfactual_identifiability_via_dynamic_optimal_transport.md)
- [LinEAS: End-to-end Learning of Activation Steering with a Distributional Loss](lineas_end-to-end_learning_of_activation_steering_with_a_distributional_loss.md)
- [On the Relation between Rectified Flows and Optimal Transport](on_the_relation_between_rectified_flows_and_optimal_transport.md)
- [Pairwise Optimal Transports for Training All-to-All Flow-Based Condition Transfer Model](pairwise_optimal_transports_for_training_all-to-all_flow-based_condition_transfe.md)
- [CountSteer: Steering Attention for Object Counting in Diffusion Models](../../AAAI2026/image_generation/countsteer_steering_attention_for_object_counting_in_diffusion_models.md)

<!-- RELATED:END -->
