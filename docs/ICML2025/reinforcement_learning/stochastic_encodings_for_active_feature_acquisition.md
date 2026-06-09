---
title: >-
  [论文解读] Stochastic Encodings for Active Feature Acquisition
description: >-
  [ICML 2025][强化学习][Active Feature Acquisition] 本文提出 SEFA (Stochastic Encodings for Feature Acquisition)，一种基于随机潜变量模型的主动特征获取方法…
tags:
  - "ICML 2025"
  - "强化学习"
  - "Active Feature Acquisition"
  - "Dynamic Feature Selection"
  - "Stochastic Encoder"
  - "Latent Space"
  - "Information Bottleneck"
---

# Stochastic Encodings for Active Feature Acquisition

**会议**: ICML 2025  
**arXiv**: [2508.01957](https://arxiv.org/abs/2508.01957)  
**代码**: [a-norcliffe/SEFA](https://github.com/a-norcliffe/SEFA)  
**领域**: 主动特征获取 / 序贯决策  
**关键词**: Active Feature Acquisition, Dynamic Feature Selection, Stochastic Encoder, Latent Space, Information Bottleneck  

## 一句话总结

本文提出 SEFA (Stochastic Encodings for Feature Acquisition)，一种基于随机潜变量模型的主动特征获取方法，通过在正则化潜空间中跨多种未观测特征实现进行推理来替代 RL 和贪心 CMI 最大化，在合成和真实数据集（含癌症分类）上一致超越所有基线。

## 研究背景与动机

**主动特征获取 (AFA)** 是一个实例级的序贯决策问题：在测试时，模型基于当前已观测特征动态选择下一步获取哪个特征。典型场景：医生根据已知信息选择做哪项检查。

两类主流方法各有根本缺陷：

### 1. 强化学习方法
- 自然适配序贯决策，但受训练困难所累：稀疏奖励、探索-利用权衡、致命三角

### 2. 条件互信息 (CMI) 最大化
- 贪心选择 $i^* = \arg\max_i I(X_i; Y | \mathbf{x}_O)$
- **缺陷一：短视决策**。CMI 边际化未观测特征，$p(x_i, y|\mathbf{x}_O) = \int p(x_j, x_i, y|\mathbf{x}_O) dx_j$，无法考虑未来获取的影响。论文用**指示器问题**严格证明：

> 考虑 $d+1$ 个特征，$x_{d+1}$ 是指示器（决定 $d$ 个二元特征中哪个给出标签）。最优策略只需 2 步，但贪心 CMI 期望需要 $3 - 1/d$ 步。

- **缺陷二：CMI 不等于最佳 0-1 损失目标**。最小化熵可以通过使不太可能的类更不可能来实现，而非区分最可能的类：
    - $H([0.5, 0.5, 0.0]) = 0.693$（低熵但无法确定最可能类）
    - $H([0.7, 0.15, 0.15]) = 0.819$（高熵但明确识别最可能类）

## 方法详解

### 整体框架

SEFA 使用编码器-预测器架构，中间有随机潜变量 $Z$：

$$p_{\theta,\phi}(y|\mathbf{x}_S) = \mathbb{E}_{p_\theta(\mathbf{z}|\mathbf{x}_S)} p_\phi(y|\mathbf{z})$$

### 关键设计 1：特征独立编码

每个特征 $i$ 独立编码到 $l$ 个潜分量：

$$p_\theta(\mathbf{z}|\mathbf{x}_S) = \prod_{i=1}^d p_{\theta_i}(\mathbf{z}_{\mathcal{G}_i} | x_{S,i}, m_{S,i})$$

其中 $\mathcal{G}_i$ 索引特征 $i$ 负责的潜分量（如 $l=3$，则 $\mathcal{G}_1=\{1,2,3\}$, $\mathcal{G}_2=\{4,5,6\}$）。每个编码器输出正态分布的均值和方差。

### 关键设计 2：获取目标函数

$$R(\mathbf{x}_O, i) = \sum_{c \in [C]} p_{\theta,\phi}(Y=c|\mathbf{x}_O) \cdot \mathbb{E}_{p_\theta(\mathbf{z}|\mathbf{x}_O)} r(c, \mathbf{z}, i)$$

评分函数基于潜空间梯度：

$$r(c, \mathbf{z}, i) = \frac{\|\mathbf{g}_{\mathcal{G}_i}\|_2}{\sum_j \|\mathbf{g}_{\mathcal{G}_j}\|_2}, \quad \mathbf{g} = \nabla_\mathbf{z} p_\phi(Y=c|\mathbf{z})$$

三个核心组成：
1. **评分函数**：用潜空间梯度衡量每个特征对预测类别 $c$ 的重要性（类似可解释性中的梯度归因，但在潜空间进行）
2. **随机编码**：对 $p_\theta(\mathbf{z}|\mathbf{x}_O)$ 取期望，考虑**多种可能的未观测特征实现**——类比蒙特卡洛树搜索
3. **概率加权**：按当前预测概率 $p_{\theta,\phi}(Y=c|\mathbf{x}_O)$ 加权，聚焦更可能的类别

### 关键设计 3：信息瓶颈正则化

### 损失函数

$$L = \mathbb{E}_{p_\text{Data}(\mathbf{x}_S, y)} \mathbb{E}_{p_\text{Subsample}(S')} \left[ -\log p_{\theta,\phi}(y|\mathbf{x}_{S \cap S'}) + \beta D_\text{KL}(p_\theta(Z|\mathbf{x}_{S \cap S'}) \| p(Z)) \right]$$

- **预测损失**：负对数似然，用多次潜采样（非单次）估计
- **信息瓶颈**：约束潜变量只保留标签相关信息，去除特征级噪声
- **随机子采样**：训练时随机移除特征，使模型适应任意特征子集
- 先验 $p(Z) = \mathcal{N}(0, 1)$，KL 散度有闭式解

### 潜空间 vs 特征空间的优势

1. 潜空间梯度更有意义——所有分量连续且尺度相近
2. 信息瓶颈去除特征噪声——基于纯标签信息做决策
3. 无需训练生成模型——避免连续/离散变量建模、多模态密度等复杂性

## 实验关键数据

### 合成数据集（获取所有相关特征的平均步数，越低越好）

| 模型 | Syn 1 | Syn 2 | Syn 3 |
|------|-------|-------|-------|
| ACFlow | 7.730 | 7.527 | 9.194 |
| DIME | 4.079 | 4.581 | 5.667 |
| GDFS | 4.568 | 4.484 | 5.587 |
| Opportunistic RL | 4.201 | 4.846 | 5.850 |
| Random | 9.484 | 9.499 | 9.987 |
| **SEFA** | **4.017** | **4.099** | **5.084** |

SEFA 接近最优（Syn 1/2 最优为 3，Syn 3 最优为 5）。

### 真实数据集（获取过程中的平均评估指标，越高越好）

| 模型 | Bank Mktg | Calif. Housing | MiniBooNE | MNIST | Fashion | METABRIC | TCGA |
|------|-----------|---------------|-----------|-------|---------|----------|------|
| DIME | 0.907 | 0.661 | 0.951 | 0.731 | 0.703 | 0.670 | 0.805 |
| GDFS | 0.907 | 0.653 | 0.949 | 0.732 | 0.692 | 0.671 | 0.797 |
| Opp. RL | 0.910 | 0.657 | 0.953 | 0.740 | 0.708 | 0.706 | 0.838 |
| **SEFA** | **0.919** | **0.676** | **0.957** | **0.761** | **0.721** | **0.709** | **0.843** |

**SEFA 在全部 8 个真实数据集上均排名第一。**

### 消融实验（Syn 数据集，获取相关特征步数）

| 变体 | Syn 1 | Syn 2 | Syn 3 |
|------|-------|-------|-------|
| $\beta=0$（无信息瓶颈） | 4.520 | 4.578 | 5.716 |
| 1 个获取采样 | 4.683 | 4.862 | 5.700 |
| 1 个训练采样 | 4.421 | 4.713 | 5.188 |
| 确定性编码器 | 4.593 | 4.773 | 5.744 |
| **特征空间计算** | **5.111** | **5.461** | **5.977** |
| 无归一化 | 4.036 | 4.104 | 5.101 |
| **SEFA (full)** | **4.017** | **4.099** | **5.084** |

**最关键组件**：潜空间计算 > 多获取采样 > 随机编码 > 信息瓶颈。

### 癌症分类可解释性

在 TCGA 肿瘤定位任务中，SEFA 的获取顺序与医学文献一致：
- 几乎总是首先选择 **ST6GAL1**——已知在多种癌症中上调
- 乳腺癌：第 2 步选 DEF6（与乳腺癌转移相关）
- 肺/肝癌：第 2 步选 DNASE1L3（肝肺癌潜在生物标志物）
- 前列腺癌：第 2 步选 SERPINB1（与前列腺癌相关）

## 亮点与洞察

1. **理论 + 实践双重论证 CMI 的缺陷**：指示器问题的最优性证明（命题 4.1/4.2）+ 熵 vs 类识别的反例
2. **潜空间推理 ≈ 蒙特卡洛树搜索**：通过在随机潜空间中采样多种可能实现，实现了对未来获取效果的非贪心推理
3. **避免 RL 的训练**：获取目标是手工设计的而非学习的——用监督学习训练模型，用信息论目标做获取
4. **特征独立编码的巧妙权衡**：虽然特征间依赖无法通过编码器建模，但预测器网络可以隐式学习条件依赖——测量特征 1 后梯度改变，从而影响特征 2 的评分
5. **多训练采样的理论需求**：单采样训练使编码器或预测器倾向一致输出，减少潜空间多样性

## 局限性

1. **仅适用于分类**：需要类概率分离；回归任务需要新设计（论文建议双头方案）
2. **推理时内存开销大**：需要多次潜采样（200 次），但 CMI 生成模型方法也有类似需求
3. **缺乏与真实医学数据标准评估**：TCGA 分析更多是可解释性展示
4. **独立编码器的局限**：无法建模特征间的编码器级条件依赖

## 相关工作

- **RL 方法**：Opportunistic Learning, GSMRL, REFUEL
- **CMI 方法**：GDFS, DIME, ACFlow, EDDI
- **其他**：灵敏度方法 (Kachuee et al.), 模仿学习 (Valancius et al.), 决策树 (Xu et al.)
- **信息瓶颈**：Tishby et al. 1999, Alemi et al. 2017

## 评分

⭐⭐⭐⭐⭐ (5/5)

方法设计优美而深思熟虑——每个组件都有清晰的动机和理论支持。在全部数据集上的一致优势令人信服。CMI 缺陷的理论分析和癌症分类的可解释性验证是亮点。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] ALINE: Joint Amortization for Bayesian Inference and Active Data Acquisition](../../NeurIPS2025/reinforcement_learning/aline_joint_amortization_for_bayesian_inference_and_active_data_acquisition.md)
- [\[NeurIPS 2025\] Exploration via Feature Perturbation in Contextual Bandits](../../NeurIPS2025/reinforcement_learning/exploration_via_feature_perturbation_in_contextual_bandits.md)
- [\[ICML 2025\] LineFlow: A Framework to Learn Active Control of Production Lines](lineflow_a_framework_to_learn_active_control_of_production_lines.md)
- [\[NeurIPS 2025\] Open-World Drone Active Tracking with Goal-Centered Rewards](../../NeurIPS2025/reinforcement_learning/open-world_drone_active_tracking_with_goal-centered_rewards.md)
- [\[NeurIPS 2025\] Emergent World Beliefs: Exploring Transformers in Stochastic Games](../../NeurIPS2025/reinforcement_learning/emergent_world_beliefs_exploring_transformers_in_stochastic_games.md)

</div>

<!-- RELATED:END -->
