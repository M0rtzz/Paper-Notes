---
title: >-
  [论文解读] AutoAL: Automated Active Learning with Differentiable Query Strategy Search
description: >-
  [ICML 2025][主动学习] 提出首个可微的主动学习策略搜索框架 AutoAL，通过 SearchNet 和 FitNet 两个网络在双层优化框架下协同训练，自动从多个候选 AL 策略中为给定任务选出最优策略，在自然图像和医学图像数据集上一致超越所有候选策略及其他 SOTA 方法。
tags:
  - ICML 2025
  - 主动学习
  - 可微查询策略搜索
  - 双层优化
  - 策略选择
  - 数据高效学习
---

# AutoAL: Automated Active Learning with Differentiable Query Strategy Search

**会议**: ICML 2025  
**arXiv**: [2410.13853](https://arxiv.org/abs/2410.13853)  
**代码**: [有](https://github.com/haizailache999/AutoAL)  
**领域**: 主动学习  
**关键词**: 主动学习, 可微查询策略搜索, 双层优化, 策略选择, 数据高效学习

## 一句话总结

提出首个可微的主动学习策略搜索框架 AutoAL，通过 SearchNet 和 FitNet 两个网络在双层优化框架下协同训练，自动从多个候选 AL 策略中为给定任务选出最优策略，在自然图像和医学图像数据集上一致超越所有候选策略及其他 SOTA 方法。

## 研究背景与动机

### 领域现状
主动学习（Active Learning）通过迭代地从未标注数据池中选择最有信息量的样本进行标注，可以大幅降低深度学习的标注成本。现有的 AL 策略主要分为两大类：
- **基于不确定性的方法**（如 Maximum Entropy、BALD）：选择模型最不确定的样本
- **基于多样性/代表性的方法**（如 CoreSet、KMeans）：选择能代表整个数据分布的样本子集

### 核心痛点
不同 AL 策略在不同的数据场景下表现差异极大。例如多样性方法在类别多、batch 大时表现好，不确定性方法在相反场景下更优。**没有一种策略能在所有任务上都表现最好**，这使得实际应用中的策略选择成为一个棘手问题。

### 已有尝试与不足
- **SelectAL**（NeurIPS 2024）：通过估计问题的相对预算大小来动态选择 AL 策略，但依赖于小子集上泛化误差减小的近似，难以捕捉实际任务的复杂性
- **ALBL**：将候选策略视为多臂赌博机问题，但无法利用梯度信息
- **Zhang et al.（NeurIPS 2024）**：从数百候选中选最优 batch，但计算开销大且缺乏可微性

### 核心矛盾
现有自适应选择方法都依赖于**不可微的离散选择**（黑箱搜索），计算效率低、数据利用不充分。

### 本文方案
提出 AutoAL——首个可微的 AL 策略搜索框架，将策略选择空间从离散松弛到连续，利用梯度下降高效优化，实现数据驱动的自动策略选择。

## 方法详解

### 整体框架

AutoAL 由两个神经网络组成，在双层优化框架下交替训练：
1. **FitNet** $\Omega_F$：在标注数据上训练，学习数据分布，输出每个样本的分类损失
2. **SearchNet** $\Omega_S$：基于 FitNet 的损失信号，学习为每个候选 AL 策略分配权重，输出综合信息量得分

核心巧妙设计：将标注集随机等分为训练集和验证集，SearchNet 将训练集当作"已标注池"、验证集当作"未标注池"，模拟真实 AL 的选择过程，无需访问实际未标注数据。

### 关键设计

#### 1. **双层优化公式化**

将 FitNet 和 SearchNet 的优化建模为双层优化问题：

$$\Omega_S^* = \arg\max_{\Omega_S} \sum_{j=1}^{M/2} \mathcal{L}_S((x_j, y_j), \Omega_S, \Omega_F^*)$$

$$\text{s.t.} \quad \Omega_F^* = \arg\min_{\Omega_F} \sum_{j=1}^{M/2} \mathcal{L}_F((x_j, y_j), \Omega_F)$$

- **下层优化**：FitNet 最小化分类损失，拟合数据分布
- **上层优化**：SearchNet 最大化信息损失，选择高损失（高信息量）的样本
- 设计动机：FitNet 提供对数据分布的理解，SearchNet 利用这一信息找到最需要标注的样本，两者形成互补的协作关系

#### 2. **概率查询策略与高斯混合模型**

对于每个样本 $x_j$，从 $K$ 个候选 AL 策略各获取一个选择得分 $\mathcal{S}_\kappa(x_j)$。使用高斯混合模型（GMM）建模所有策略得分的联合分布：

$$p(\mathcal{S}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathcal{S} | \mu_k, \Sigma_k)$$

然后从 GMM 采样确定选择阈值，结合 SearchNet 预测的策略权重 $W_{\kappa,j}$ 计算每个策略-样本对的综合得分：

$$\hat{\mathcal{S}}_\kappa(x_j, \Omega_S) = (\mathcal{S}_\kappa(x_j) - \vartheta_t(S_{\text{sample}})) \cdot W_{\kappa,j}$$

设计动机：GMM 自然地融合了多个策略的得分分布，阈值机制确保只选择真正突出的样本，权重 $W_{\kappa,j}$ 实现了**样本级别**的策略自适应选择。

#### 3. **可微松弛与连续化搜索空间**

为了使离散的策略选择可微，使用 Sigmoid 函数将离散的 $\{0,1\}$ 选择松弛到连续空间：

$$\bar{\mathcal{S}}(x_j) = \sum_{\kappa \in K} \frac{\lambda}{1 + \exp(-\Theta^{(j)}_{\mathcal{S}'_\kappa})} \hat{\mathcal{S}}_\kappa(x_j, \Omega_S)$$

其中 $\Theta^{(j)}$ 是每个样本的策略混合权重参数向量。这种松弛使得整个框架可以通过反向传播端到端优化。

设计动机：灵感来自 DARTS（可微架构搜索），将离散选择问题转化为连续优化问题，用梯度下降代替黑箱搜索，大幅提升优化效率。

### 损失函数 / 训练策略

**FitNet 损失**（公式 7）：
$$\mathcal{L}_F = \frac{1}{B} \sum_{j'} \bar{\mathcal{S}}_{\text{detach}}(x'_j) \cdot \mathcal{L}(x'_j, y'_j) + \bar{\lambda} \mathcal{L}_{re}(t, B)$$

- 使用 detach 的搜索得分加权交叉熵损失，使 FitNet 更关注被选中样本的损失最小化
- $\bar{\mathcal{S}}_{\text{detach}}$ 的梯度不回传到 SearchNet，保证双层优化的正确性

**SearchNet 损失**（公式 8）：
$$\mathcal{L}_S = -\frac{1}{B} \sum_j \bar{\mathcal{S}}(x_j) \cdot \mathcal{L}_{\text{detach}}(x_j, y_j) - \bar{\lambda} \mathcal{L}_{re}(t, B)$$

- 取负号实现梯度上升，目标是选择高损失样本
- FitNet 的损失被 detach，避免更新 FitNet 本身

**正则化损失**（公式 9）：
$$\mathcal{L}_{re}(t, B) = \frac{1}{1 + \exp(0.5 \cdot |\alpha - t \cdot B|)} - 0.5$$

- 控制被选择样本的数量，防止选择过多或过少样本
- $\alpha$ 是被选样本数，$t$ 是查询 batch size 与总数据池大小的比值

**训练流程**：
- 每个 AL 迭代包含 $\mathcal{C}$ 个 cycle
- FitNet 先用验证集训练 200 epochs
- 然后 FitNet、SearchNet 和损失预测模块交替更新共 400 epochs
- 骨干网络使用 ResNet-18，FitNet 用 Adam（lr=0.005），SearchNet 用 SGD（lr=0.005）

## 实验关键数据

### 主实验

实验在 7 个数据集上进行（4 个自然图像 + 3 个医学图像），候选策略池包含 7 种 AL 方法。

| 数据集 | 类别数 | 训练规模 | 不平衡比 | AutoAL表现 |
|--------|--------|----------|----------|------------|
| CIFAR-10 | 10 | 50K | 1.0 | 一致优于所有 baseline |
| CIFAR-100 | 100 | 50K | 1.0 | 在困难数据集上优势更明显 |
| SVHN | 10 | 73K | 3.0 | 全轮次领先 |
| TinyImageNet | 200 | 100K | 1.0 | 在大规模多类别上依然稳健 |
| OrganCMNIST | 11 | 13K | 5.0 | 小数据池场景也表现最佳 |
| PathMNIST | 9 | 90K | 1.6 | 持续优于候选策略 |
| TissueMNIST | 8 | 165K | 9.1 | 高不平衡比下仍然稳健 |

AutoAL 在所有 7 个数据集上一致超越 14 个 baseline（包括 Entropy、Margin、BALD、BADGE、LPL、VAAL、CoreSet、DDU、ALBL 等），且标准差更小。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| ResNet Backbone only | 性能下降明显 | 缺少损失预测模块，SearchNet 优化不充分 |
| Loss Prediction only | 性能下降最大 | 仅最小化损失而非选择最优策略，偏离目标 |
| ResNet + Loss Prediction (完整) | 最佳性能 | 两个组件互补，缺一不可 |
| 1个候选策略 | 比单独使用该策略好，但不如多候选 | SearchNet 仍通过损失预测增益 |
| 3个候选策略 | 接近最佳性能（CIFAR-100、OrganCMNIST） | 性价比较高 |
| 5个候选策略 | 在 SVHN 上进一步提升 | 上限因数据集而异 |
| 7个候选策略（完整） | 最佳或接近最佳 | 更多候选→更低标准差 |

### 关键发现

1. **策略动态切换现象**：通过可视化各轮次的策略得分（图5），发现 AutoAL 在早期轮次优先选择多样性策略（KMeans），后期转向不确定性策略（Least Confidence、MeanSTD）。这符合直觉——早期需要广泛探索数据分布，后期需要精细化决策边界
2. **计算开销可控**：AutoAL 的搜索部分（SearchNet + FitNet 更新）仅占总时间的约 3%，主要开销来自候选策略的得分计算。使用 3 个候选时总开销仅为 EntropySampling 的 1.3 倍
3. **鲁棒性**：AutoAL 的准确率曲线更平滑，无明显"掉点"（harmful data selection 导致的性能回退），标准差更小
4. **不同策略在不同数据集上的表现差异大**：Margin 在 SVHN 上差、KMeans 在 OrganCMNIST 上差、VAAL 在 CIFAR-10 上差，凸显了自动策略选择的必要性

## 亮点与洞察

1. **DARTS 思想的巧妙迁移**：将可微架构搜索的思路应用到 AL 策略搜索，将离散的策略选择松弛为连续优化，理念清晰、实现优雅
2. **训练-验证集模拟 AL 过程**：用标注数据的训练/验证划分模拟标注池/未标注池，无需访问真正的未标注数据就能训练 SearchNet，设计巧妙
3. **样本级别的策略选择**：不是为整个数据集选一种策略，而是为每个样本独立选择最优策略组合，粒度更细
4. **从探索到利用的自动过渡**：实验证实 AutoAL 会自动从多样性策略过渡到不确定性策略，与直觉一致且无需人工干预
5. **框架可扩展**：任何新的 AL 策略都可以作为候选加入池中，框架具有良好的开放性

## 局限与展望

1. **候选策略池的上限效应**：更多候选并非总是更好（CIFAR-100 和 OrganCMNIST 上 3 个就够了），如何自动确定最优候选集尚未解决
2. **所有候选得分需预计算**：AutoAL 的搜索开销低，但所有候选策略的得分计算开销随候选数线性增长，当候选池很大时可能成为瓶颈
3. **仅验证分类任务**：论文仅在图像分类上实验，未涉及检测、分割、NLP 等场景，泛化性有待验证
4. **未与学习型 AL 方法深度对比**：如 Meta-Query Net 等基于元学习的 AL 方法也有自适应能力，缺少与其的公平对比
5. **FitNet 和 SearchNet 的架构固定**：均使用 ResNet-18，未探索更轻量或更强大的网络对性能的影响
6. **超参敏感性未充分讨论**：如 cycle 数 $\mathcal{C}$、正则化系数 $\bar{\lambda}$、GMM 组件数等的影响

## 相关工作与启发

- **DARTS**（Liu et al., 2018）：可微架构搜索的先驱，AutoAL 的松弛策略直接受其启发
- **Learning Loss for AL**（Yoo & Kweon, 2019）：损失预测模块被集成到 AutoAL 的 SearchNet 中
- **SelectAL**（Hacohen & Weinshall, NeurIPS 2024）：基于预算估计的策略选择，是 AutoAL 的主要对比方法
- **ALBL**（Hsu & Lin, 2015）：多臂赌博机框架的自适应策略选择，但不可微
- 启发：可微优化 + 双层优化的组合可能也适用于其他"策略选择"问题，如数据增强策略搜索、课程学习策略搜索等

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个可微 AL 策略搜索框架，DARTS→AL 的迁移有价值，但核心思想不算革命性
- 实验充分度: ⭐⭐⭐⭐ 7 个数据集（含医学）、14 个 baseline、消融完整、策略可视化有深度，但缺少非分类任务验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰、动机阐述充分、公式推导完整，但部分符号使用稍显冗余
- 价值: ⭐⭐⭐⭐ 解决了 AL 策略选择的实际痛点，框架可扩展，对 AL 领域有推动作用

<!-- RELATED:START -->

## 相关论文

- [Exact Learning of Arithmetic with Differentiable Agents](../../NeurIPS2025/others/exact_learning_of_arithmetic_with_differentiable_agents.md)
- [Instance-wise Supervision-level Optimization in Active Learning](../../CVPR2025/others/instance-wise_supervision-level_optimization_in_active_learning.md)
- [Reliable Active Learning from Unreliable Labels via Neural Collapse Geometry](../../NeurIPS2025/others/reliable_active_learning_from_unreliable_labels_via_neural_collapse_geometry.md)
- [Active Learning for Decision Trees with Provable Guarantees](../../ICLR2026/others/active_learning_for_decision_trees_with_provable_guarantees.md)
- [Auto-GAS: Automated Proxy Discovery for Training-Free Generative Architecture Search](../../ECCV2024/others/auto-gas_automated_proxy_discovery_for_training-free_generative_architecture_sea.md)

<!-- RELATED:END -->
