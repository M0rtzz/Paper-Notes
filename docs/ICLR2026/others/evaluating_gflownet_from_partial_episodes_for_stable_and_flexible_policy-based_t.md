---
title: >-
  [论文解读] Evaluating GFlowNet from Partial Episodes for Stable and Flexible Policy-Based Training
description: >-
  [ICLR 2026][GFlowNet] 建立GFlowNet中状态流函数与策略评价函数之间的理论联系，提出子轨迹评价平衡（Sub-EB）目标用于可靠学习评价函数，增强策略基GFlowNet训练的稳定性和灵活性。
tags:
  - ICLR 2026
  - GFlowNet
  - 策略梯度
  - 评价函数
  - 流平衡
  - 组合优化
---

# Evaluating GFlowNet from Partial Episodes for Stable and Flexible Policy-Based Training

**会议**: ICLR 2026  
**arXiv**: [2603.01047](https://arxiv.org/abs/2603.01047)  
**代码**: [github.com/niupuhua1234/Sub-EB](https://github.com/niupuhua1234/Sub-EB)  
**领域**: 其他  
**关键词**: GFlowNet, 策略梯度, 评价函数, 流平衡, 组合优化

## 一句话总结

建立GFlowNet中状态流函数与策略评价函数之间的理论联系，提出子轨迹评价平衡（Sub-EB）目标用于可靠学习评价函数，增强策略基GFlowNet训练的稳定性和灵活性。

## 研究背景与动机

### GFlowNet简介

生成流网络（GFlowNet）是一种在组合空间 $\mathcal{X}$ 上采样的生成模型，目标是以正比于奖励函数 $R(x)$ 的概率采样 $x \in \mathcal{X}$。生成过程被分解为有向无环图（DAG）上的增量轨迹，正向策略 $\pi_F$ 逐步构建对象。

### 两大训练范式

**值基方法（Value-based）**：引入流函数 $F(s)$，通过流平衡条件（如Sub-TB）来隐式最小化分布差异。优点是支持off-policy采样，但流平衡不直接反映奖励的真实信息

**策略基方法（Policy-based）**：借鉴RL的Actor-Critic框架，引入评价函数 $V(s)$ 来估计策略在状态 $s$ 处的KL散度，然后用策略梯度更新 $\pi_F$。优点是on-policy训练更高效

### 核心挑战

策略基方法的关键瓶颈在于**如何可靠地学习评价函数 $V(s)$**。现有方法（$\lambda$-TD目标）基于边级别的不匹配度，只考虑状态 $s$ 之后的事件和边级错配，学习信号不够充分，且要求后向策略 $\pi_B$ 固定。本文的关键洞察是：流函数 $F(s)$ 和评价函数 $V(s)$ 之间存在深层联系——对固定的 $\pi_F$，流平衡条件的解恰好等于真实评价函数。

## 方法详解

### 整体框架

本文的核心贡献是**Sub-EB（Subtrajectory Evaluation Balance）**条件和目标，将值基方法中成功的流平衡思想引入策略基框架中的评价函数学习。整体训练流程仍为Actor-Critic：
1. **Critic（评价函数学习）**：用Sub-EB目标优化 $V(\cdot; \phi)$
2. **Actor（策略更新）**：基于 $V$ 和策略梯度更新 $\pi_F(\cdot; \theta)$

### 关键设计

#### 1. 流函数与评价函数的理论联系

**定理3.1（核心定理）**：对于任意评价函数 $V$，以下等价：
$$V(s_h) = \log F^*(s_h) - D_{\text{KL}}(P_F(\tau_{h:}|s_h) \| P_B(\tau_{h:}|s_h))$$
当且仅当 $V$ 满足Sub-EB条件。

直觉解释：$V(s)$ 等于最优流的对数减去从 $s$ 出发的KL散度。即评价函数同时编码了"状态的重要性"和"当前策略的偏差程度"。

**定理3.2（与Sub-TB的关系）**：对于值基方法，流函数 $F$ 和最优策略 $\pi_F^*$ 一起满足Sub-TB条件，当且仅当 $\log F(s_h) = \log F^*(s_h) - D_{\text{KL}}(P_{F^*}(\tau_{h:}|s_h) \| P_B(\tau_{h:}|s_h))$ 且 $\pi_F = \pi_F^*$。

#### 2. Sub-EB条件

对所有 $i < j \in [H+1]$：

$$\mathbb{E}_{P_F(\tau_{i:j})} \left[ \log(P_F(\tau_{i:j}|s_i) \exp V(s_i)) \right] = \mathbb{E}_{P_F(\tau_{i:j})} \left[ \log(P_B(\tau_{i:j}|s_j) \exp V(s_j)) \right]$$

直觉：$V(s_i) - V(s_j)$ 的差值应等于从 $s_i$ 到 $s_j$ 之间子轨迹的真实散度。

#### 3. Sub-EB训练目标

$$\mathcal{L}_V(\phi) = \mathbb{E}_{P_F(\tau)} \left[ \sum_{\tau_{i:j}} w_{j-i} \left( \log \frac{P_F(\tau_{i:j}|s_i) \exp V(s_i; \phi)}{P_B(\tau_{i:j}|s_j) \exp V(s_j; \phi)} \right)^2 \right]$$

与Sub-TB目标形式高度相似，但关键区别：
- Sub-EB用于固定 $\pi_F$ 下学习 $V$（仅更新 $\phi$）
- Sub-TB用于联合优化 $(\pi_F, \log F)$（更新 $\theta$）
- Sub-EB的期望在 $P_F$ 下取（on-policy），Sub-TB在 $P_\mathcal{D}$ 下取（可off-policy）

#### 4. 参数化后向策略支持

$\lambda$-TD目标要求 $\pi_B$ 固定。而Sub-EB和Sub-TB目标天然支持**参数化的 $\pi_B$**——$\pi_B$ 可以与 $V$ 一起通过Sub-EB梯度联合更新，无需引入额外的后向阶段或辅助目标。

#### 5. 离线策略基训练

通过引入后向评价函数 $W$（评估 $\pi_B$ 的质量），可以实现**离线采样 + 策略基训练**：
- 用数据收集策略 $\pi_\mathcal{D}$ 采样终止状态 $x$
- 从 $x$ 用 $\pi_B$ 回溯生成轨迹
- 用后向Sub-EB目标学习 $W$
- 用 $W$ 的策略梯度更新 $\pi_B$

### 损失函数 / 训练策略

**在线训练（Algorithm 1）**：
1. 采样批次 $\mathcal{D} \sim P_F(\tau)$
2. 基于 $\mathcal{D}$ 用 $\nabla_\phi \hat{\mathcal{L}}_V(\phi)$ 更新评价函数 $V$
3. 基于 $\mathcal{D}$ 和 $V$ 用策略梯度更新 $\pi_F$

**离线训练（Algorithm 2）**：用 $\pi_\mathcal{D}$ 采样 → $\pi_B$ 回溯 → 更新 $W$ → 更新 $\pi_B$

权重系数 $w_{j-i} = \lambda^{j-i} / \sum \lambda^{j-i}$，$\lambda$ 控制短/长子轨迹的偏好。

## 实验关键数据

### 主实验

**Hypergrid实验（精确计算 $D_{\text{TV}}$）**

| 方法 | 256×256 收敛性 | 128×128×128 稳定性 | 64×64×64 最终性能 |
|------|:-:|:-:|:-:|
| CV（经验梯度） | 较差 | 较差 | 一般 |
| RL（$\lambda$-TD） | 中等（不稳定） | 不稳定 | 好 |
| **Sub-EB** | **好（稳定且快速）** | **稳定且快速** | **好** |
| Sub-TB（值基） | 中等 | 一般 | 一般 |

Sub-EB显著提升策略基方法的稳定性和收敛速度，特别在高维/大规模网格中优势明显。

**贝叶斯网络结构学习（真实任务，10/15个节点）**

| 方法 | 10节点 平均奖励 | 10节点 多样性 | 10节点 FCS |
|------|:-:|:-:|:-:|
| Sub-TB | 中等 | 中等 | 一般 |
| Q-Much | 中等 | 中等 | 一般 |
| RL | 高 | 好 | 好 |
| **Sub-EB** | **最高** | **好** | **好** |
| **Sub-EB-B** | **最高（+离线增强）** | 略降 | - |

### 消融实验

**参数化 $\pi_B$ 消融（256×256, 128×128, 64×64×64）**

| 方法 | 性能表现 |
|------|---------|
| Sub-EB-P（参数化 $\pi_B$） | 所有方法中最好，训练最稳定 |
| RL-P（两阶段） | 次优，额外后向阶段增加复杂度 |
| RL-MLE（最大似然） | 不如Sub-EB-P |
| Sub-TB-P（参数化 $\pi_B$） | 与Sub-TB类似 |

证实Sub-EB天然支持参数化后向策略的优势。

### 关键发现

1. **Sub-EB vs $\lambda$-TD**：利用子轨迹级别的平衡信息（而非仅边级别）显著提升评价函数学习的可靠性
2. **策略基 vs 值基**：策略基方法（RL、Sub-EB）在收敛速度和分布建模质量上通常优于值基方法（Sub-TB、Q-Much）
3. **离线增强的有效性**：Sub-EB-B（集成local search）在BN结构学习中达到最高奖励，验证了Sub-EB对离线技术的兼容性
4. **分子图设计**（$|\mathcal{X}| \approx 10^{16}$）：Sub-EB在大规模任务中表现稳健，实现更高平均奖励和更快收敛

## 亮点与洞察

1. **理论优雅**：揭示了流函数和评价函数之间"差一个KL散度"的关系，统一了值基和策略基视角
2. **即插即用**：Sub-EB目标可直接替换 $\lambda$-TD目标，几乎不改变训练流程
3. **三重灵活性**：支持（1）参数化后向策略，（2）离线数据采集，（3）灵活的子轨迹权重方案
4. **与值基方法的深度对应**：Sub-EB之于策略基方法，正如Sub-TB之于值基方法——形式对称，语义互补

## 局限与展望

1. 权重系数 $w_{j-i}$ 的选择策略可进一步优化（目前用固定的 $\lambda$-衰减）
2. 理论分析基于分级DAG假设（虽然一般DAG可转化），实际实现需加入dummy状态
3. 尚未集成到更高级的策略基方法（如TRPO的完整实现）
4. Hypergrid实验依赖精确计算，更大规模下需要FCS等近似度量
5. 对策略梯度的方差-偏差权衡尚未有理论最优的 $\gamma$ 选择指导

## 相关工作与启发

- **Sub-TB**（Madan et al., 2023）：值基方法的子轨迹平衡目标，Sub-EB的形式灵感来源
- **GFlowNet策略梯度**（Niu et al., 2024）：提出Actor-Critic框架和 $\lambda$-TD目标，Sub-EB直接改进其critic学习
- **Q-Much / RFI**：值基方法的RL变体，本文在BN实验中作为对比
- 启发：能否将Sub-EB的"平衡条件→评价函数"思路推广到一般RL中的value function学习？

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ★★★★☆ |
| 技术深度 | ★★★★★ |
| 实验充分性 | ★★★★☆ |
| 写作质量 | ★★★★☆ |
| 实用价值 | ★★★★☆ |
