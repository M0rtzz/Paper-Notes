---
title: >-
  [论文解读] FedFACT: A Provable Framework for Controllable Group-Fairness Calibration in Federated Learning
description: >-
  [NeurIPS 2025][3D视觉][联邦学习] 提出FedFACT框架，通过刻画联邦学习下的**贝叶斯最优公平分类器**结构，将公平联邦学习分别在训练中（in-processing）化归为**个性化代价敏感学习**、在训练后（post-processing）化归为**双层优化**…
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "联邦学习"
  - "群体公平性"
  - "贝叶斯最优分类器"
  - "代价敏感学习"
  - "后处理校准"
---

# FedFACT: A Provable Framework for Controllable Group-Fairness Calibration in Federated Learning

**会议**: NeurIPS 2025  
**arXiv**: [2506.03777](https://arxiv.org/abs/2506.03777)  
**代码**: 无  
**领域**: AI安全  
**关键词**: 联邦学习, 群体公平性, 贝叶斯最优分类器, 代价敏感学习, 后处理校准

## 一句话总结

提出FedFACT框架，通过刻画联邦学习下的**贝叶斯最优公平分类器**结构，将公平联邦学习分别在训练中（in-processing）化归为**个性化代价敏感学习**、在训练后（post-processing）化归为**双层优化**，首次实现多类别场景下全局公平性与局部公平性的可控协调，并提供收敛及泛化保证。

## 研究背景与动机

**领域现状**: 联邦学习日益应用于医疗、金融等高风险决策领域，需要保证对敏感群体（如性别、种族）的组间公平性。FL中存在两种公平性概念：**全局公平性**（跨所有客户端的整体模型差异）和**局部公平性**（每个客户端内部的差异）。

**现有痛点**: 
   - **全局与局部公平性固有矛盾**: 客户端数据分布的统计异质性导致两种公平性目标方向不同
   - **精度-公平性权衡不可控**: 公平性指标的**不可分解性**（non-decomposable）和**不可微性**（non-differentiable）使得优化极为困难
   - 现有方法多依赖替代公平性损失（surrogate fairness loss），不可避免的替代差距导致次优性能

**核心矛盾**: 现有方法要么只关注全局公平（牺牲局部），要么只关注局部公平（牺牲全局），且大多局限于二分类问题。在多类别×多敏感属性×全局/局部双公平约束下，如何找到精度退化最小的最优分类器？

**本文目标**: 设计一个理论上最优、可控的联邦群体公平性校准框架，同时保证全局和局部公平性，适用于多类别分类。

**切入角度**: 从**贝叶斯最优**的角度出发——先刻画联邦环境下公平约束下的最优分类器结构，再基于此设计实际算法。

**核心 idea**: 联邦贝叶斯最优公平分类器可以通过逐客户端的代价敏感分类或基于plug-in的双层优化来精确逼近，其中代价矩阵 $\mathbf{M}^{\lambda,\mu}(a,k)$ 由对偶参数控制全局/局部公平程度。

## 方法详解

### 整体框架

FedFACT包含两个互补的方法：
- **In-processing**: 训练阶段介入，通过修改训练损失来implicitly满足公平约束
- **Post-processing**: 训练后阶段，直接校准预训练模型的输出概率

两者都基于对联邦贝叶斯最优公平分类器的理论刻画。

### 关键设计

#### 1. 联邦贝叶斯最优公平分类器的刻画（Proposition 1 & 2）

- **功能**: 在全局+局部公平约束下，找到最小化分类风险的最优分类器
- **核心结构** (Proposition 2):
$$h_k^*(x) = e_y, \quad y \in \arg\max_{j \in [m]} \left(\sum_{a \in \mathcal{A}} \mathbb{P}(A=a|x,k) [\mathbf{M}^{\lambda,\mu}(a,k)]^\top \eta(x,a,k)\right)_j$$
  其中 $\mathbf{M}^{\lambda,\mu}(a,k) = \mathbf{I} - \frac{1}{p_{a,k}}[\text{全局对偶项} - \text{局部对偶项}]$
- **设计动机**: Proposition 1证明最优分类器可以分解为各客户端的确定性分类器的线性组合——这为联邦优化提供了基础

#### 2. In-processing: 个性化代价敏感学习（Algorithm 1）

- **功能**: 将公平约束下的联邦优化转化为每个客户端的代价敏感分类问题
- **核心公式** (Proposition 3):
$$\ell_k(y, \mathbf{s}(x), a) = -\sum_{i=1}^{m} \overline{\mathbf{M}}^{\lambda,\mu}_{y,i}(a,k) \log \frac{\exp([\mathbf{s}(x)]_i)}{\sum_j \exp([\mathbf{s}(x)]_j)}$$
  其中 $\overline{\mathbf{M}} = \mathbf{M}^{\lambda,\mu} + \kappa \mathbf{1}_{m \times m}$（保证正定性）
- **训练流程**:
  1. 每轮：客户端集成统一模型 $\theta$ 和个性化模型 $\phi_k$
  2. 自适应权重 $w_k$ 平衡两个模型贡献
  3. 更新局部对偶参数 $\mu_k$（控制局部公平）
  4. 服务器聚合 $\theta$ 和全局对偶参数 $\lambda$（控制全局公平）
- **设计动机**: 代价矩阵 $\mathbf{M}$ 编码了公平约束，通过对偶参数动态调节，实现精度-公平性的可控权衡

#### 3. Post-processing: 基于Plug-in的双层优化（Algorithm 2）

- **功能**: 校准预训练联邦模型的分类概率以满足公平性
- **核心公式** (Theorem 5):
$$\min_{\lambda \in \Lambda} \left\{\hat{F}(\lambda) = \sum_k \hat{p}_k \hat{F}_k(\lambda)\right\}, \quad \hat{F}_k(\lambda) := \min_{\mu_k \in \mathcal{M}_k} \hat{H}_k(\lambda, \mu_k)$$
- **关键设计**: 用soft-max替代hard-max保证平滑性，$\mu_k$在本地更新（局部公平），$\lambda$在服务器聚合（全局公平）
- **设计动机**: 后处理不改变模型表示，开销低，适合资源受限场景

### 损失函数/训练策略

- **In-processing**: 个性化代价敏感交叉熵损失 + 集成权重的在线学习更新
- **Post-processing**: 双层优化中的梯度下降+投影
- **公平约束**: 支持DP (Demographic Parity)、EOP (Equalized Opportunity)、EO (Equalized Odds)
- **收敛率**: $\mathcal{O}(1/\sqrt{T})$（Theorem 4）

## 实验关键数据

### 主实验（4个数据集，$\gamma=0.5$ 分区）

| 方法 | Compas Acc | Compas $\mathscr{D}^g$ | Compas $\mathscr{D}^l$ | Adult Acc | Adult $\mathscr{D}^g$ | Adult $\mathscr{D}^l$ |
|------|-----------|-----------|-----------|-----------|-----------|-----------|
| FedAvg | 69.73 | 0.2766 | 0.3590 | 84.52 | 0.1765 | 0.2310 |
| FairFed | 59.39 | 0.1008 | 0.1022 | 80.73 | 0.0983 | 0.1434 |
| FedFACT$^g$(Post) | **67.27** | **0.0128** | 0.0660 | **82.83** | 0.0173 | 0.0276 |
| FedFACT$^l$(Post) | 67.49 | 0.0315 | **0.0552** | 82.79 | 0.0154 | **0.0267** |
| FedFACT$^{g\&l}$(Post) | 67.33 | 0.0139 | 0.0641 | 82.74 | **0.0134** | 0.0274 |

| 方法 | CelebA Acc | CelebA $\mathscr{D}^g$ | CelebA $\mathscr{D}^l$ | ENEM Acc | ENEM $\mathscr{D}^g$ | ENEM $\mathscr{D}^l$ |
|------|-----------|-----------|-----------|-----------|-----------|-----------|
| FedAvg | 89.14 | 0.1435 | 0.1308 | 67.56 | 0.2620 | 0.2462 |
| FedFACT$^g$(Post) | 87.25 | **0.0089** | 0.0253 | 66.15 | 0.0175 | **0.0181** |
| FedFACT$^l$(Post) | **87.36** | 0.0127 | **0.0163** | **66.54** | 0.0197 | 0.0240 |
| FedFACT$^{g\&l}$(Post) | 87.06 | 0.0093 | 0.0172 | 66.52 | **0.0162** | 0.0184 |

### 消融实验

| 变量 | 发现 |
|------|------|
| In-processing vs Post-processing | Post-processing精度更高但公平性略逊；In-processing公平性控制更细 |
| 全局$\xi^g$从0.00到0.10 | 公平性差异从0.01连续升至0.10，精度同步提升——可控trade-off |
| 局部$\xi^k$从0.00到0.10 | 类似可控行为 |
| Hetero(异质)分区 | FedFACT在更强异质性下依然显著优于基线 |

### 关键发现

1. **Post-processing模式精度最高**: FedFACT(Post)在几乎所有数据集上精度接近FedAvg但公平性提升10-20倍
2. **全局+局部联合约束有效**: FedFACT$^{g\&l}$同时降低全局和局部差异
3. **可控性**: 通过调节 $\xi^g$ 和 $\xi^k$ 可以连续地在精度-公平性之间权衡
4. **在异质性数据上鲁棒**: Hetero分区下FedFACT改善更显著

## 亮点与洞察

1. **理论上首次刻画多类别联邦贝叶斯最优公平分类器**: 从最优性出发设计算法而非启发式方法
2. **将公平FL化归为经典问题**: in-processing→代价敏感学习，post-processing→双层优化——巧妙且实用
3. **代价矩阵 $\mathbf{M}^{\lambda,\mu}$ 的统一视角**: 编码了公平约束、敏感属性、客户端分布的所有信息
4. **严格的收敛和泛化保证**: 不仅仅是实验有效，理论上保证逼近最优
5. **in/post-processing互补**: 前者改变表示但开销大，后者轻量但只调输出

## 局限与展望

1. **实验规模有限**: 4个数据集均为表格数据或简单图像，未在大规模视觉/NLP任务上验证
2. **通信开销**: 需额外传输对偶参数 $\lambda$ 和混淆矩阵
3. **假设Bayes score已知或可估计**: post-processing依赖 $\eta(x,a,k)$ 的准确估计
4. **in-processing和post-processing不兼容**: 理论上不能串联使用
5. **敏感属性需已知**: 实际场景中敏感属性可能缺失或不精确

## 相关工作与启发

- **与LogoFair的关系**: LogoFair也推导贝叶斯最优分类器，但仅限二分类+后处理；FedFACT扩展到多类别+in/post双模式
- **与FairFed/FedFB的关系**: 它们使用启发式重加权，FedFACT基于最优性理论
- **启发点**: 代价敏感学习在公平性中的应用值得深挖——不同样本的分类代价应该不同

## 评分

⭐⭐⭐⭐ (4/5)
- 理论贡献扎实，首次解决多类别全局+局部公平的可控问题
- 实验规模偏小，对真实复杂场景的适用性有待验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Copresheaf Topological Neural Networks: A Generalized Deep Learning Framework](copresheaf_topological_neural_networks_a_generalized_deep_learning_framework.md)
- [\[NeurIPS 2025\] Fair Representation Learning with Controllable High Confidence Guarantees via Adversarial Inference](fair_representation_learning_with_controllable_high_confidence_guarantees_via_ad.md)
- [\[ICCV 2025\] Faster and Better 3D Splatting via Group Training](../../ICCV2025/3d_vision/faster_and_better_3d_splatting_via_group_training.md)
- [\[CVPR 2025\] A Lightweight UDF Learning Framework for 3D Reconstruction Based on Local Shape Functions](../../CVPR2025/3d_vision/a_lightweight_udf_learning_framework_for_3d_reconstruction_based_on_local_shape_.md)
- [\[ICCV 2025\] Boost 3D Reconstruction using Diffusion-based Monocular Camera Calibration](../../ICCV2025/3d_vision/boost_3d_reconstruction_using_diffusion-based_monocular_camera_calibration.md)

</div>

<!-- RELATED:END -->
